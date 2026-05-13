"""Suggestion event log — the substrate for the acceptance-rate metric.

Events land in `<vault>/00_System/suggestion_log.jsonl`. `kind` is one of:

  emit          — every atom returned by GET /suggestions, one per atom.
                  Includes (call_id, atom_id, layer, score, emitted_at).
  referenced    — POST /suggestion_referenced from the Swift app when the
                  consultant references the suggestion within ~60s.
                  Includes (call_id, atom_id, referenced_at). Acts as the
                  legacy "did anything happen" signal.
  helpful       — Phase 11: user gave a thumbs-up on the atom in the
                  overlay. Strongest positive signal.
  useless       — Phase 11: user gave a thumbs-down. Strongest negative.
  dismissed     — Phase 11: user swiped/closed the atom row without
                  acting. Weak negative — they saw it, decided no.
  used_in_call  — Phase 11: post-call (or in-call automated detector)
                  marks the atom as actually invoked in the conversation.
                  Stronger than `referenced` because it survives the
                  60s acceptance window.

Append-only, one JSON per line. fsync per write for durability — losing
acceptance events silently distorts the metric.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


SUGGESTION_LOG_FILENAME = "suggestion_log.jsonl"

# Phase 11+12: feedback events the user explicitly produces. Kept as a
# tuple so /suggestion_feedback can validate input against the canonical
# list.
#
# Phase 11 introduced helpful / useless / dismissed / used_in_call.
# Phase 12 added the exposure funnel: shown / hidden / expanded /
# copied / followup_created. With these the brain can compute a
# complete funnel ratio (shown → expanded → copied → followup) instead
# of just the post-tap signals.
FEEDBACK_KINDS: tuple[str, ...] = (
    "helpful",
    "useless",
    "dismissed",
    "used_in_call",
    # Phase 12 exposure events
    "shown",
    "hidden",
    "expanded",
    "copied",
    "followup_created",
)


@dataclass(frozen=True, slots=True)
class SuggestionEvent:
    """One row in the suggestion log. `kind` discriminates the event class:
    `emit` / `referenced` / one of FEEDBACK_KINDS."""

    kind: str
    call_id: str
    atom_id: str
    layer: str | None  # populated on emit; None on every other event
    score: float | None  # populated on emit; None on every other event
    timestamp: str  # ISO-8601 UTC with trailing Z


def log_path(vault_root: Path) -> Path:
    return vault_root.expanduser().resolve() / "00_System" / SUGGESTION_LOG_FILENAME


def log_emit(
    *,
    vault_root: Path,
    call_id: str,
    atom_id: str,
    layer: str,
    score: float,
    now: datetime | None = None,
) -> None:
    """Append one emit event."""
    _append(
        vault_root=vault_root,
        event=SuggestionEvent(
            kind="emit",
            call_id=call_id,
            atom_id=atom_id,
            layer=layer,
            score=score,
            timestamp=_iso(now or datetime.now(timezone.utc)),
        ),
    )


def log_referenced(
    *,
    vault_root: Path,
    call_id: str,
    atom_id: str,
    now: datetime | None = None,
) -> None:
    """Append one referenced event (user acted on the suggestion)."""
    _append(
        vault_root=vault_root,
        event=SuggestionEvent(
            kind="referenced",
            call_id=call_id,
            atom_id=atom_id,
            layer=None,
            score=None,
            timestamp=_iso(now or datetime.now(timezone.utc)),
        ),
    )


def log_feedback(
    *,
    vault_root: Path,
    call_id: str,
    atom_id: str,
    kind: str,
    now: datetime | None = None,
) -> None:
    """Phase 11: append one explicit user feedback event.

    `kind` must be in `FEEDBACK_KINDS`. Raises `ValueError` otherwise so
    the HTTP layer can return a clean 400 to the Swift caller instead of
    silently corrupting the metric with arbitrary kinds.
    """
    if kind not in FEEDBACK_KINDS:
        raise ValueError(
            f"Unknown feedback kind {kind!r}. Expected one of {FEEDBACK_KINDS}."
        )
    _append(
        vault_root=vault_root,
        event=SuggestionEvent(
            kind=kind,
            call_id=call_id,
            atom_id=atom_id,
            layer=None,
            score=None,
            timestamp=_iso(now or datetime.now(timezone.utc)),
        ),
    )


def load_events(vault_root: Path) -> list[SuggestionEvent]:
    """Read every event. Malformed rows skipped to keep the metric job
    resilient against a single bad write."""
    path = log_path(vault_root)
    if not path.exists():
        return []
    out: list[SuggestionEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                out.append(
                    SuggestionEvent(
                        kind=raw["kind"],
                        call_id=raw["call_id"],
                        atom_id=raw["atom_id"],
                        layer=raw.get("layer"),
                        score=raw.get("score"),
                        timestamp=raw["timestamp"],
                    )
                )
            except (KeyError, json.JSONDecodeError, TypeError):
                continue
    return out


# ────────────────────────────────────────────────────────────────────────────


def _append(*, vault_root: Path, event: SuggestionEvent) -> None:
    path = log_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(event), sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
