"""Feature extraction for post-call scoring.

Pulls 9 observable signals from a finished call (transcript turns +
extracted atoms + client context) that the weighted-sum scorer turns
into a 0-100 quality score:

  next_step_booked     — binary: any commitment with a future-time anchor?
  objections_resolved  — ratio (resolved / raised); Phase 5 approximation
                         counts a "resolved" objection as one followed by
                         a commitment in the transcript order.
  talk_ratio_balance   — closer to call-type target ratio = higher.
                         Consulting: 40/60 you/them. Cold: 30/70.
                         Closing: 50/50.
  commitments_made     — count of `commitment` atoms, normalized.
  win_signals          — count of `win_signal` atoms, normalized.
  loss_signals         — count of `loss_signal` atoms, normalized (negative weight).
  confusion_events     — count of `confusion` atoms, normalized.
  completion_of_agenda — Phase 5 v1: returns 0.5 placeholder. Phase 6
                         distillation will populate the agenda model.
  primary_win_progress — LLM-judged 0-1 against the client's primary_win
                         field. Optional — returns 0.5 if no judge runs.

Each feature is normalized to [-1, 1] before multiplying by its weight,
so the weight values are interpretable directly (a +20 weight on
commitments_made means "going from 0 commitments to 'a lot' bumps the
score by 20 points").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from consultant_brain.schemas import Atom, AtomType, CallType, Speaker


# ────────────────────────────────────────────────────────────────────────────
# Feature inputs
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TranscriptTurnLite:
    """Slimmer than CompletedTurn for the scoring path so this module
    doesn't have to care about timestamps / item IDs.
    """

    speaker: Speaker
    text: str


@dataclass(frozen=True, slots=True)
class ScoreFeatures:
    """The 9 features the scorer consumes. Each is in [-1, 1] except
    where noted — normalization happens here so weight values stay
    interpretable.
    """

    next_step_booked: float          # 0 or 1
    objections_resolved: float       # 0..1 ratio
    talk_ratio_balance: float        # 0..1, 1 = exactly on target ratio
    commitments_made: float          # 0..1, saturates at 5 atoms
    win_signals: float               # 0..1, saturates at 4 atoms
    loss_signals: float              # 0..1, saturates at 3 atoms
    confusion_events: float          # 0..1, saturates at 3 atoms
    completion_of_agenda: float      # 0..1, placeholder 0.5 for v1
    primary_win_progress: float      # 0..1, LLM-judged; 0.5 if no judge

    def as_vector(self) -> list[float]:
        """Order matches `FEATURE_NAMES` — used by both the dot-product
        scorer and the retrain regression."""
        return [
            self.next_step_booked,
            self.objections_resolved,
            self.talk_ratio_balance,
            self.commitments_made,
            self.win_signals,
            self.loss_signals,
            self.confusion_events,
            self.completion_of_agenda,
            self.primary_win_progress,
        ]


FEATURE_NAMES: tuple[str, ...] = (
    "next_step_booked",
    "objections_resolved",
    "talk_ratio_balance",
    "commitments_made",
    "win_signals",
    "loss_signals",
    "confusion_events",
    "completion_of_agenda",
    "primary_win_progress",
)


# Per-call-type target talk ratio (consultant fraction of total chars).
# Consulting calls: the client should be doing most of the talking during
# discovery. Cold calls: even more — we're asking, they're answering. Closing:
# more balanced, because the consultant frames + asks for the commitment.
TALK_RATIO_TARGETS: dict[CallType, float] = {
    CallType.consulting_call: 0.40,
    CallType.ai_training: 0.55,
    CallType.cold_call: 0.30,
    CallType.closing_call: 0.50,
    CallType.follow_up: 0.45,
    CallType.implementation: 0.55,
}


# Saturation caps — counts above these cap out at 1.0. Tuned to typical
# call density; revisit after the first 20-call dataset.
SATURATION_CAPS: dict[str, int] = {
    "commitments_made": 5,
    "win_signals": 4,
    "loss_signals": 3,
    "confusion_events": 3,
}


# ────────────────────────────────────────────────────────────────────────────
# Primary-win judge (optional)
# ────────────────────────────────────────────────────────────────────────────


class PrimaryWinJudge(Protocol):
    """Pluggable — production uses a Claude-backed implementation; tests
    pass a deterministic stub. `judge` returns a 0-1 score: how much did
    this call advance the stated primary win?"""

    def judge(self, *, primary_win: str, transcript: str, call_type: CallType) -> float:
        ...


class NeutralWinJudge:
    """Default when no judge is provided — returns 0.5 (neither helped nor
    hurt). Keeps the scorer deterministic + free when running offline."""

    def judge(self, *, primary_win: str, transcript: str, call_type: CallType) -> float:
        _ = (primary_win, transcript, call_type)
        return 0.5


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────


def compute_features(
    *,
    atoms: list[Atom],
    turns: list[TranscriptTurnLite],
    call_type: CallType,
    primary_win: str | None = None,
    judge: PrimaryWinJudge | None = None,
) -> ScoreFeatures:
    """Build the 9-feature vector for one finished call."""
    counts = _atom_counts(atoms)
    commitments = counts[AtomType.commitment]
    objections = counts[AtomType.objection]
    win_signals = counts[AtomType.win_signal]
    loss_signals = counts[AtomType.loss_signal]
    confusion = counts[AtomType.confusion]

    return ScoreFeatures(
        next_step_booked=_next_step_booked(atoms),
        objections_resolved=_objections_resolved_ratio(atoms),
        talk_ratio_balance=_talk_ratio_score(turns=turns, call_type=call_type),
        commitments_made=_saturate(commitments, SATURATION_CAPS["commitments_made"]),
        win_signals=_saturate(win_signals, SATURATION_CAPS["win_signals"]),
        loss_signals=_saturate(loss_signals, SATURATION_CAPS["loss_signals"]),
        confusion_events=_saturate(confusion, SATURATION_CAPS["confusion_events"]),
        completion_of_agenda=0.5,  # Phase 6 populates a real agenda model
        primary_win_progress=_primary_win_progress(
            primary_win=primary_win,
            turns=turns,
            call_type=call_type,
            judge=judge,
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
# Individual feature implementations
# ────────────────────────────────────────────────────────────────────────────


def _atom_counts(atoms: list[Atom]) -> dict[AtomType, int]:
    counts: dict[AtomType, int] = {t: 0 for t in AtomType}
    for atom in atoms:
        counts[atom.type] += 1
    return counts


# Phrases that strongly indicate a future commitment with a time anchor.
# Tuned against Casper's tone — short, plain, no jargon. Tested against
# the sample fixture which has "by Friday" / "next Wednesday".
_TIME_ANCHOR_RE = re.compile(
    r"\b(?:"
    r"by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|"
    r"end of (?:day|week|month)|eod|eow|next week|"
    r"\d{1,2}(?::\d{2})?(?:am|pm)|"
    r"the (?:\d{1,2}(?:st|nd|rd|th)?))"
    r"|next (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)"
    r"|tomorrow at \d{1,2}"
    r"|\d{1,2}am|\d{1,2}pm"
    r"|on (?:monday|tuesday|wednesday|thursday|friday)"
    r")\b",
    re.IGNORECASE,
)


def _next_step_booked(atoms: list[Atom]) -> float:
    """Binary: is there a `commitment` atom whose body contains a future-
    time anchor? Catches "by Friday", "next Wednesday", "tomorrow at 9am".
    """
    for atom in atoms:
        if atom.type is AtomType.commitment and _TIME_ANCHOR_RE.search(atom.body):
            return 1.0
    return 0.0


def _objections_resolved_ratio(atoms: list[Atom]) -> float:
    """Phase 5 v1 approximation: an objection counts as "resolved" if the
    next atom (by created_at order) of any type is a `commitment`.
    Returns 1.0 if there were no objections (vacuously resolved)."""
    ordered = sorted(atoms, key=lambda a: a.created_at)
    raised = 0
    resolved = 0
    for i, atom in enumerate(ordered):
        if atom.type is not AtomType.objection:
            continue
        raised += 1
        # Look at the next 3 atoms — gives the consultant some room to
        # respond without requiring an immediate next-turn commitment.
        for next_atom in ordered[i + 1 : i + 4]:
            if next_atom.type is AtomType.commitment:
                resolved += 1
                break
    if raised == 0:
        return 1.0
    return resolved / raised


def _talk_ratio_score(*, turns: list[TranscriptTurnLite], call_type: CallType) -> float:
    """1.0 when the consultant's share of total characters is exactly on the
    call-type target ratio, decaying linearly to 0 at ±30 percentage points.
    """
    if not turns:
        return 0.5
    you_chars = sum(len(t.text) for t in turns if t.speaker is Speaker.you)
    total = sum(len(t.text) for t in turns) or 1
    actual = you_chars / total
    target = TALK_RATIO_TARGETS.get(call_type, 0.45)
    distance = abs(actual - target)
    # 0.30 chosen so a 10-point miss still rates ≥0.67; a 30-point miss = 0.
    return max(0.0, 1.0 - distance / 0.30)


def _saturate(count: int, cap: int) -> float:
    """Clamp + normalize a count to [0, 1]."""
    if cap <= 0:
        return 0.0
    return min(1.0, max(0, count) / cap)


def _primary_win_progress(
    *,
    primary_win: str | None,
    turns: list[TranscriptTurnLite],
    call_type: CallType,
    judge: PrimaryWinJudge | None,
) -> float:
    """Defer to the LLM judge when one is configured AND we have something to
    judge against. Otherwise return 0.5 (neutral)."""
    if not primary_win or not primary_win.strip():
        return 0.5
    if not turns:
        return 0.5
    actual_judge = judge or NeutralWinJudge()
    transcript = "\n\n".join(
        f"{'You' if t.speaker is Speaker.you else 'Them'}: {t.text.strip()}" for t in turns
    )
    raw = actual_judge.judge(primary_win=primary_win, transcript=transcript, call_type=call_type)
    return max(0.0, min(1.0, raw))
