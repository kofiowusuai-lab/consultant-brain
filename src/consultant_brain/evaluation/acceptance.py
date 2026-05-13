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
    """Acceptance rate overall + per layer.

    Phase 11 adds three explicit-feedback signals (helpful/useless/dismissed)
    plus a stronger acceptance signal (`used_in_call`) than the legacy
    60-90s `referenced` proxy. Rates are computed against `total_emits`
    so all four are directly comparable to `acceptance_rate`.

    Phase 12 adds the exposure funnel: shown → expanded → copied →
    followup_created. Each rate's denominator is the prior stage so
    "expand_rate" reads as "of atoms the user saw, how many did they
    open?" rather than diluting against emits the user never scrolled
    to.  `hidden` is tracked but not surfaced as a rate — it's noise
    in the funnel sense, useful only as a sanity check that shown ≈
    hidden over time.
    """

    total_emits: int
    total_referenced: int
    acceptance_rate: float
    # Phase 11 explicit-feedback counts. Each counts every event of the
    # kind whose (call_id, atom_id) pair is also seen in an emit. Events
    # without a matching emit (rare — would mean the brain restarted
    # mid-call) are ignored so the denominator stays meaningful.
    helpful_count: int = 0
    useless_count: int = 0
    dismissed_count: int = 0
    used_count: int = 0
    helpful_rate: float = 0.0
    useless_rate: float = 0.0
    dismissed_rate: float = 0.0
    used_rate: float = 0.0
    # Phase 12 exposure-funnel counts. `shown_count` is bounded by
    # `total_emits`; the rest are denominated against the prior stage
    # so the rates compose into a funnel ratio at a glance.
    shown_count: int = 0
    hidden_count: int = 0
    expanded_count: int = 0
    copied_count: int = 0
    followup_count: int = 0
    shown_rate: float = 0.0  # shown / total_emits
    expand_rate: float = 0.0  # expanded / shown
    copy_rate: float = 0.0  # copied / shown
    followup_rate: float = 0.0  # followup_created / shown
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

    # Phase 11+12: per-kind pair sets. A pair contributes at most one
    # event of each kind so a user double-tapping thumbs-up (or a row
    # rerendering many times) doesn't inflate the rate.
    helpful_pairs: set[tuple[str, str]] = set()
    useless_pairs: set[tuple[str, str]] = set()
    dismissed_pairs: set[tuple[str, str]] = set()
    used_pairs: set[tuple[str, str]] = set()
    shown_pairs: set[tuple[str, str]] = set()
    hidden_pairs: set[tuple[str, str]] = set()
    expanded_pairs: set[tuple[str, str]] = set()
    copied_pairs: set[tuple[str, str]] = set()
    followup_pairs: set[tuple[str, str]] = set()

    for pair_key, events_for_pair in by_pair.items():
        # Sort chronologically.
        sorted_events = sorted(events_for_pair, key=lambda e: e.timestamp)
        has_emit = any(e.kind == "emit" for e in sorted_events)
        # Track explicit feedback for this pair so the per-kind metric
        # is bounded by total_emits.
        if has_emit:
            for e in sorted_events:
                if e.kind == "helpful":
                    helpful_pairs.add(pair_key)
                elif e.kind == "useless":
                    useless_pairs.add(pair_key)
                elif e.kind == "dismissed":
                    dismissed_pairs.add(pair_key)
                elif e.kind == "used_in_call":
                    used_pairs.add(pair_key)
                elif e.kind == "shown":
                    shown_pairs.add(pair_key)
                elif e.kind == "hidden":
                    hidden_pairs.add(pair_key)
                elif e.kind == "expanded":
                    expanded_pairs.add(pair_key)
                elif e.kind == "copied":
                    copied_pairs.add(pair_key)
                elif e.kind == "followup_created":
                    followup_pairs.add(pair_key)
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

    helpful_count = len(helpful_pairs)
    useless_count = len(useless_pairs)
    dismissed_count = len(dismissed_pairs)
    used_count = len(used_pairs)
    shown_count = len(shown_pairs)
    hidden_count = len(hidden_pairs)
    expanded_count = len(expanded_pairs)
    copied_count = len(copied_pairs)
    followup_count = len(followup_pairs)
    denom = float(total_emits) if total_emits else 0.0
    # Phase 12 funnel denominators: `shown` rates against emits (how
    # often did the user actually see what we surfaced), then each
    # downstream stage rates against shown so the funnel composes.
    shown_denom = float(shown_count) if shown_count else 0.0

    return AcceptanceReport(
        total_emits=total_emits,
        total_referenced=accepted_emits,
        acceptance_rate=rate,
        helpful_count=helpful_count,
        useless_count=useless_count,
        dismissed_count=dismissed_count,
        used_count=used_count,
        helpful_rate=(helpful_count / denom) if denom else 0.0,
        useless_rate=(useless_count / denom) if denom else 0.0,
        dismissed_rate=(dismissed_count / denom) if denom else 0.0,
        used_rate=(used_count / denom) if denom else 0.0,
        shown_count=shown_count,
        hidden_count=hidden_count,
        expanded_count=expanded_count,
        copied_count=copied_count,
        followup_count=followup_count,
        shown_rate=(shown_count / denom) if denom else 0.0,
        expand_rate=(expanded_count / shown_denom) if shown_denom else 0.0,
        copy_rate=(copied_count / shown_denom) if shown_denom else 0.0,
        followup_rate=(followup_count / shown_denom) if shown_denom else 0.0,
        by_layer=by_layer,
        by_layer_counts=by_layer_counts,
    )


def _parse(iso: str) -> datetime | None:
    try:
        # Tolerate the "Z" suffix.
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
