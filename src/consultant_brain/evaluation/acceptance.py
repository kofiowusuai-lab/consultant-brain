"""Acceptance-rate metric: did the user act on a brain suggestion?

Walks `00_System/suggestion_log.jsonl`. For each `emit` event, looks for
a matching `referenced` event (same call_id + atom_id) within the
acceptance window (default 90s — looser than the master prompt's 60s
because real users sometimes reference an atom one turn later).

Acceptance rate = referenced_emits / total_emits.

Breakdown surfaces by layer (hot/warm/cold) and by atom type — useful
for spotting which suggestion paths actually pay off.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from consultant_brain.evaluation.suggestion_log import (
    SuggestionEvent,
    load_events,
)


DEFAULT_ACCEPTANCE_WINDOW_SECONDS = 90


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """Acceptance rate overall + per layer."""

    total_emits: int
    total_referenced: int
    acceptance_rate: float
    by_layer: dict[str, float] = field(default_factory=dict)  # layer → rate
    by_layer_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    # ↑ layer → (referenced_count, emit_count)


def compute_acceptance(
    *,
    vault_root: Path,
    window_seconds: int = DEFAULT_ACCEPTANCE_WINDOW_SECONDS,
) -> AcceptanceReport:
    """Compute acceptance rate over the entire suggestion log."""
    events = load_events(vault_root)
    if not events:
        return AcceptanceReport(total_emits=0, total_referenced=0, acceptance_rate=0.0)

    # Bucket events by (call_id, atom_id) so we can pair emits with the
    # earliest matching referenced event.
    by_pair: dict[tuple[str, str], list[SuggestionEvent]] = defaultdict(list)
    for event in events:
        by_pair[(event.call_id, event.atom_id)].append(event)

    total_emits = 0
    accepted_emits = 0
    layer_emit_counts: dict[str, int] = defaultdict(int)
    layer_accept_counts: dict[str, int] = defaultdict(int)

    for events_for_pair in by_pair.values():
        # Sort chronologically.
        sorted_events = sorted(events_for_pair, key=lambda e: e.timestamp)
        # Iterate emits; for each, look for a referenced event within
        # window_seconds AFTER it.
        for i, event in enumerate(sorted_events):
            if event.kind != "emit":
                continue
            total_emits += 1
            layer = event.layer or "unknown"
            layer_emit_counts[layer] += 1
            emit_time = _parse(event.timestamp)
            if emit_time is None:
                continue
            cutoff = emit_time + timedelta(seconds=window_seconds)
            for later in sorted_events[i + 1 :]:
                if later.kind != "referenced":
                    continue
                later_time = _parse(later.timestamp)
                if later_time is None:
                    continue
                if later_time > cutoff:
                    break  # past the window — list is sorted, can stop
                accepted_emits += 1
                layer_accept_counts[layer] += 1
                break  # one referenced event satisfies the emit

    rate = (accepted_emits / total_emits) if total_emits else 0.0
    by_layer = {}
    by_layer_counts = {}
    for layer, emits in layer_emit_counts.items():
        accepts = layer_accept_counts.get(layer, 0)
        by_layer[layer] = accepts / emits if emits else 0.0
        by_layer_counts[layer] = (accepts, emits)
    return AcceptanceReport(
        total_emits=total_emits,
        total_referenced=accepted_emits,
        acceptance_rate=rate,
        by_layer=by_layer,
        by_layer_counts=by_layer_counts,
    )


def _parse(iso: str) -> datetime | None:
    try:
        # Tolerate the "Z" suffix.
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
