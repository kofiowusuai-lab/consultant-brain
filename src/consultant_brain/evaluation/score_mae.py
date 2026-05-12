"""Score-prediction MAE — mean absolute error between predicted and
user-overridden scores.

Master prompt's bar: "Auto-score within ±10 of user override on 80% of
calls after 30 calls of training data." We report:
  - MAE overall
  - % of corrections within ±10
  - Per-call-type breakdown (which types is the model worst at?)

Reads `00_System/score_corrections.jsonl` directly via the existing
`load_corrections()` helper — no new storage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from consultant_brain.scoring.corrections import load_corrections


# Master prompt's target. We surface "% within ±10" so a glance at the
# metric tells you whether the calibration is hitting the bar.
ACCEPTABLE_DELTA = 10.0


@dataclass(frozen=True, slots=True)
class ScoreMAEReport:
    sample_count: int
    mean_absolute_error: float
    within_acceptable_delta: float  # 0..1 fraction within ±ACCEPTABLE_DELTA
    by_call_type: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    # call_type → (sample_count, mae, % within ±10)
    insufficient_data: bool = False


def compute_score_mae(*, vault_root: Path) -> ScoreMAEReport:
    rows = load_corrections(vault_root)
    if not rows:
        return ScoreMAEReport(
            sample_count=0,
            mean_absolute_error=0.0,
            within_acceptable_delta=0.0,
            by_call_type={},
            insufficient_data=True,
        )

    abs_errors = [abs(r.user_score - r.predicted_score) for r in rows]
    mae = sum(abs_errors) / len(abs_errors)
    within = sum(1 for e in abs_errors if e <= ACCEPTABLE_DELTA) / len(abs_errors)

    by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_type[r.call_type].append(abs(r.user_score - r.predicted_score))

    breakdown: dict[str, tuple[int, float, float]] = {}
    for ct, errors in by_type.items():
        n = len(errors)
        ct_mae = sum(errors) / n
        ct_within = sum(1 for e in errors if e <= ACCEPTABLE_DELTA) / n
        breakdown[ct] = (n, ct_mae, ct_within)

    return ScoreMAEReport(
        sample_count=len(rows),
        mean_absolute_error=mae,
        within_acceptable_delta=within,
        by_call_type=breakdown,
        insufficient_data=False,
    )
