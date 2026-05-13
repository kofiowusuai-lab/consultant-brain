"""Phase 13 — transcript-driven stage suggestor.

Given the current rolling transcript window + the current stage
index, return a `SuggestionResult` indicating whether the
conversation has cued a transition to a nearby stage and, if so,
which one.

Design choices (intentionally simple, easy to swap later):
  - Substring matching, not embeddings or an LLM call. Deterministic,
    cheap, runs inside the FastAPI request thread.
  - The suggestor only proposes the current stage's +1 or +2
    neighbors. Jumping further requires manual ←/→. Most legitimate
    transitions in a working session go to the very next stage; a
    "+2" suggestion catches the rare moment the user wants to skip
    a stage that's already covered.
  - Two high-signal "transition phrases" ("let's move on", "next
    section", etc.) count as a strong cue. Anything weaker requires
    >=2 cue-phrase hits in the recent window.
  - When two candidate stages tie on signal, the closer one wins so
    we never overshoot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from consultant_brain.facilitator.outline import (
    REECE_OUTLINE,
    FacilitatorStage,
)


# High-signal phrases that count as a single hit by themselves.
# These are the verbal cues a facilitator uses when consciously
# moving between sections, regardless of which specific stage cue
# they then name.
TRANSITION_PHRASES: tuple[str, ...] = (
    "let's move on",
    "let us move on",
    "moving on",
    "next section",
    "next stage",
    "next part",
    "switching gears",
    "let's pivot",
    "let's transition",
)

# Maximum stages to look ahead. +2 catches "skip one" intent without
# letting the suggestor recommend the close stage during an
# objective. Keep this small — wide jumps belong on the keyboard.
MAX_LOOKAHEAD = 2


@dataclass(frozen=True, slots=True)
class SuggestionResult:
    """Outcome of one suggest_next_stage call.

    `suggested_index` is None when the suggestor abstains (empty
    window, end of outline, not enough signal). `confidence` is
    bounded to [0.0, 1.0]; the UI clamps further when rendering.
    `reason` is a short human-readable hint surfaced as a tooltip
    on the toast.
    """

    suggested_index: Optional[int]
    confidence: float
    reason: str

    @property
    def abstains(self) -> bool:
        return self.suggested_index is None


def suggest_next_stage(
    *,
    transcript_window: str,
    current_stage_index: int,
    outline: tuple[FacilitatorStage, ...] = REECE_OUTLINE,
    recent_chars: int = 1200,
) -> SuggestionResult:
    """Walk the last ~`recent_chars` of the transcript window and
    decide if a transition is cued.

    `current_stage_index` is 0-based. Bounds-checked: when the user
    is already on the last stage, we always abstain.

    `recent_chars` defaults to ~1200 (roughly the last 6-10 turns of
    a typical working-session pace). Trim is suffix-based so the
    most recent speech wins ties.
    """
    if not outline:
        return SuggestionResult(None, 0.0, "no outline configured")
    if not transcript_window.strip():
        return SuggestionResult(None, 0.0, "empty transcript window")
    if current_stage_index < 0 or current_stage_index >= len(outline) - 1:
        return SuggestionResult(None, 0.0, "already at end of outline")

    window = transcript_window[-recent_chars:].lower()
    transition_hit = _first_match(window, TRANSITION_PHRASES)

    candidates = []
    last_candidate = min(current_stage_index + MAX_LOOKAHEAD, len(outline) - 1)
    for offset in range(1, last_candidate - current_stage_index + 1):
        candidate_index = current_stage_index + offset
        stage = outline[candidate_index]
        hits = list(_iter_matches(window, stage.cue_phrases))
        candidates.append((candidate_index, stage, hits))

    # Pick the closest candidate with enough signal. A high-signal
    # transition phrase OR >=2 cue hits qualifies. Closer wins
    # so the suggestor never skips a stage that's also a match.
    chosen = None
    for candidate_index, stage, hits in candidates:
        qualifies = bool(hits) and (transition_hit is not None or len(hits) >= 2)
        if qualifies:
            chosen = (candidate_index, stage, hits)
            break

    if chosen is None:
        return SuggestionResult(
            None,
            0.0,
            "no cue phrases matched in recent window"
            if not transition_hit
            else "transition phrase detected but no stage cued",
        )

    candidate_index, stage, hits = chosen
    confidence = _confidence(
        hit_count=len(hits),
        phrase_count=len(stage.cue_phrases),
        had_transition=transition_hit is not None,
    )
    reason = _build_reason(stage=stage, hits=hits, transition_hit=transition_hit)
    return SuggestionResult(candidate_index, confidence, reason)


def _confidence(*, hit_count: int, phrase_count: int, had_transition: bool) -> float:
    """Bounded confidence score.

    Base = hit_count / phrase_count, ramped slightly so a single
    high-signal transition phrase + 1 cue lands around 0.6 rather
    than 0.2. Capped at 0.95 — we never claim certainty because the
    user always taps to confirm.
    """
    if phrase_count == 0:
        return 0.0
    base = min(1.0, hit_count / phrase_count)
    if had_transition:
        base = min(1.0, base + 0.3)
    return round(min(0.95, max(0.05, base)), 3)


def _build_reason(
    *,
    stage: FacilitatorStage,
    hits: list[str],
    transition_hit: Optional[str],
) -> str:
    parts: list[str] = []
    if transition_hit:
        parts.append(f"transition phrase '{transition_hit}'")
    if hits:
        joined = ", ".join(f"'{h}'" for h in hits[:3])
        parts.append(f"cue {joined}")
    parts.append(f"→ stage {stage.id} {stage.title}")
    return "; ".join(parts)


def _first_match(window: str, needles: Iterable[str]) -> Optional[str]:
    for needle in needles:
        if needle in window:
            return needle
    return None


def _iter_matches(window: str, needles: Iterable[str]) -> Iterable[str]:
    seen: set[str] = set()
    for needle in needles:
        if needle in window and needle not in seen:
            seen.add(needle)
            yield needle
