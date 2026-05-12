"""Unified metrics orchestrator — runs all four Phase 7 evaluations and
returns one report. Powers `consultant-brain metrics`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from consultant_brain.evaluation.acceptance import (
    AcceptanceReport,
    compute_acceptance,
)
from consultant_brain.evaluation.pattern_stability import (
    PatternStabilityReport,
    compute_pattern_stability,
)
from consultant_brain.evaluation.precision import (
    PrecisionReport,
    compute_precision,
)
from consultant_brain.evaluation.score_mae import (
    ScoreMAEReport,
    compute_score_mae,
)


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """The all-up Phase 7 report. Each sub-report is None when its source
    of truth is missing (no labeled corpus / no acceptance events / no
    overrides / no snapshots yet) — the dashboard shows "—" in those
    cells rather than fake zeros.
    """

    generated_at: str
    precision: PrecisionReport | None
    acceptance: AcceptanceReport | None
    score_mae: ScoreMAEReport | None
    pattern_stability: PatternStabilityReport | None

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "precision": _precision_to_dict(self.precision),
            "acceptance": _acceptance_to_dict(self.acceptance),
            "score_mae": _score_mae_to_dict(self.score_mae),
            "pattern_stability": _stability_to_dict(self.pattern_stability),
        }


def compute_metrics(*, vault_root: Path, corpus_path: Path | None = None) -> MetricsReport:
    """Run every available metric. Missing inputs return None — the CLI
    handles the rendering of '—' cells."""
    precision = compute_precision(vault_root=vault_root, corpus_path=corpus_path) if corpus_path else None
    acceptance = compute_acceptance(vault_root=vault_root)
    score = compute_score_mae(vault_root=vault_root)
    stability = compute_pattern_stability(vault_root=vault_root)
    return MetricsReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        precision=precision,
        acceptance=acceptance if acceptance.total_emits > 0 else None,
        score_mae=score if not score.insufficient_data else None,
        pattern_stability=stability if not stability.insufficient_data else None,
    )


# ────────────────────────────────────────────────────────────────────────────
# JSON serialization
# ────────────────────────────────────────────────────────────────────────────


def _precision_to_dict(p: PrecisionReport | None) -> dict | None:
    if p is None:
        return None
    return {
        "corpus_size": p.corpus_size,
        "mean_p_at_1": p.mean_p_at_1,
        "mean_p_at_3": p.mean_p_at_3,
        "mean_p_at_5": p.mean_p_at_5,
        "perfect_top1_count": p.perfect_top1_count,
        "queries": [
            {
                "name": q.name,
                "p_at_1": q.p_at_1,
                "p_at_3": q.p_at_3,
                "p_at_5": q.p_at_5,
                "top_rank_of_relevant": q.top_rank_of_relevant,
                "returned": list(q.returned_ids[:5]),
                "relevant": list(q.relevant_ids),
            }
            for q in p.queries
        ],
    }


def _acceptance_to_dict(a: AcceptanceReport | None) -> dict | None:
    if a is None:
        return None
    return {
        "total_emits": a.total_emits,
        "total_referenced": a.total_referenced,
        "acceptance_rate": a.acceptance_rate,
        "by_layer": a.by_layer,
        "by_layer_counts": {k: list(v) for k, v in a.by_layer_counts.items()},
    }


def _score_mae_to_dict(s: ScoreMAEReport | None) -> dict | None:
    if s is None:
        return None
    return {
        "sample_count": s.sample_count,
        "mean_absolute_error": s.mean_absolute_error,
        "within_acceptable_delta": s.within_acceptable_delta,
        "by_call_type": {
            ct: {"sample_count": n, "mae": mae, "within": within}
            for ct, (n, mae, within) in s.by_call_type.items()
        },
    }


def _stability_to_dict(s: PatternStabilityReport | None) -> dict | None:
    if s is None:
        return None
    return {
        "snapshots_total": s.snapshots_total,
        "latest_snapshot_at": s.latest_snapshot_at,
        "older_snapshot_at": s.older_snapshot_at,
        "pair_gap_days": s.pair_gap_days,
        "persistence_rate": s.persistence_rate,
        "mean_observation_count_delta": s.mean_observation_count_delta,
    }


def render_report_text(report: MetricsReport) -> str:
    """Human-readable summary the CLI prints by default."""
    lines: list[str] = []
    lines.append(f"Consultant Brain Metrics · generated {report.generated_at}")
    lines.append("─" * 64)

    if report.precision is None:
        lines.append("Retrieval precision      — (no labeled corpus provided — pass --corpus)")
    else:
        p = report.precision
        lines.append(
            f"Retrieval precision      P@1={p.mean_p_at_1:.2f}  P@3={p.mean_p_at_3:.2f}  P@5={p.mean_p_at_5:.2f}  "
            f"({p.perfect_top1_count}/{p.corpus_size} perfect top-1)"
        )

    if report.acceptance is None:
        lines.append("Suggestion acceptance    — (no /suggestions emits logged yet)")
    else:
        a = report.acceptance
        lines.append(
            f"Suggestion acceptance    {a.acceptance_rate*100:5.1f}%  "
            f"({a.total_referenced}/{a.total_emits} emits referenced within 90s)"
        )
        for layer, rate in sorted(a.by_layer.items()):
            accepts, emits = a.by_layer_counts.get(layer, (0, 0))
            lines.append(f"  · {layer:<5} {rate*100:5.1f}%  ({accepts}/{emits})")

    if report.score_mae is None:
        lines.append("Score prediction MAE     — (no score corrections logged yet)")
    else:
        s = report.score_mae
        lines.append(
            f"Score prediction MAE     {s.mean_absolute_error:5.2f} points  "
            f"({s.within_acceptable_delta*100:.0f}% within ±10 across {s.sample_count} samples)"
        )
        for ct, (n, mae, within) in sorted(s.by_call_type.items()):
            lines.append(f"  · {ct:<18} {mae:5.2f} MAE  ({within*100:.0f}% within, n={n})")

    if report.pattern_stability is None:
        lines.append("Pattern stability        — (need ≥2 snapshots; run `distill` more)")
    else:
        ps = report.pattern_stability
        lines.append(
            f"Pattern stability        {ps.persistence_rate*100:5.1f}% persist over {ps.pair_gap_days:.1f}d  "
            f"(observations Δ {ps.mean_observation_count_delta:+.1f}/pattern)"
        )

    return "\n".join(lines)


def render_report_json(report: MetricsReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
