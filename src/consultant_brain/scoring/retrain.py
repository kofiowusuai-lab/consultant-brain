"""Linear-regression retrain over the user-correction log.

Per the master prompt: once 20+ corrections exist, fit a per-call-type
linear model on the (features → user_score) pairs and write the
resulting weights to `<vault>/00_System/scoring_weights.yaml`. Backup
the old weights before overwrite.

We fit the bias term too — it's just an extra "always 1" feature column.
Per-call-type fitting means a Cold-Call regression doesn't leak into the
Closing-Call profile. We need MIN_CORRECTIONS_PER_TYPE samples to fit a
type; types with fewer corrections keep their default weights.

Why linear regression and not something fancier:
  - The dataset is small (tens to low hundreds of points).
  - Linearity is a feature, not a bug — interpretable + the master prompt
    wrote the scorer as a linear combination.
  - numpy.linalg.lstsq is enough; no sklearn dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from consultant_brain.schemas import CallType
from consultant_brain.scoring.corrections import ScoreCorrection, load_corrections
from consultant_brain.scoring.features import FEATURE_NAMES
from consultant_brain.scoring.weights import (
    DEFAULT_BIAS,
    DEFAULT_WEIGHTS,
    ScoringWeightsTable,
)


# Minimum number of corrections in a given call_type before we'll re-fit
# its weights. Below this we keep defaults — small-sample regressions on
# this little data make wildly overfit weights.
MIN_CORRECTIONS_PER_TYPE = 5

# Total corrections across all types before we'll run retrain at all.
# Matches the master prompt's "Once 20 corrections exist".
MIN_TOTAL_CORRECTIONS = 20


@dataclass(frozen=True, slots=True)
class RetrainReport:
    """Outcome of one retrain pass — printed by the CLI."""

    total_corrections: int
    refit_call_types: list[str]
    skipped_call_types: list[str]
    backup_path: Path | None
    weights_path: Path
    insufficient_data: bool = False

    def summary_line(self) -> str:
        if self.insufficient_data:
            return (
                f"Retrain skipped: {self.total_corrections} corrections "
                f"(need ≥{MIN_TOTAL_CORRECTIONS}). No weights changed."
            )
        backup = f" · backup: {self.backup_path}" if self.backup_path else " · no prior weights to back up"
        refit = f"refit {', '.join(self.refit_call_types) or '(none)'}"
        skipped = f"; kept defaults for {', '.join(self.skipped_call_types)}" if self.skipped_call_types else ""
        return f"Retrain: {self.total_corrections} corrections · {refit}{skipped}{backup}"


def retrain_weights(*, vault_root: Path) -> RetrainReport:
    """Walk the correction log, fit per-call-type weights, write the new
    table. Returns a report the CLI can summarize."""
    corrections = load_corrections(vault_root)
    weights_path = vault_root.expanduser().resolve() / "00_System" / "scoring_weights.yaml"

    if len(corrections) < MIN_TOTAL_CORRECTIONS:
        return RetrainReport(
            total_corrections=len(corrections),
            refit_call_types=[],
            skipped_call_types=[],
            backup_path=None,
            weights_path=weights_path,
            insufficient_data=True,
        )

    table = ScoringWeightsTable.load(weights_path)

    refit: list[str] = []
    skipped: list[str] = []
    new_bias = table.bias  # default; per-type bias not supported in Phase 5

    for ct in CallType:
        type_rows = [c for c in corrections if c.call_type == ct.value]
        if len(type_rows) < MIN_CORRECTIONS_PER_TYPE:
            skipped.append(ct.value)
            continue
        new_weights, _residual = _fit_one_call_type(type_rows)
        # Update in place — preserves features added to FEATURE_NAMES later.
        table.by_call_type[ct] = {
            name: float(new_weights[i]) for i, name in enumerate(FEATURE_NAMES)
        }
        refit.append(ct.value)

    # Fit a global bias by averaging the regression intercepts of the
    # types we refit. Falls back to the original bias when nothing refit.
    if refit:
        per_type_biases: list[float] = []
        for ct in CallType:
            if ct.value not in refit:
                continue
            type_rows = [c for c in corrections if c.call_type == ct.value]
            _w, intercept = _fit_one_call_type(type_rows)
            per_type_biases.append(intercept)
        if per_type_biases:
            new_bias = float(np.mean(per_type_biases))
    table.bias = new_bias

    backup = table.save(weights_path)
    return RetrainReport(
        total_corrections=len(corrections),
        refit_call_types=refit,
        skipped_call_types=skipped,
        backup_path=backup,
        weights_path=weights_path,
    )


# ────────────────────────────────────────────────────────────────────────────
# Internal: per-call-type least-squares fit
# ────────────────────────────────────────────────────────────────────────────


def _fit_one_call_type(rows: list[ScoreCorrection]) -> tuple[np.ndarray, float]:
    """Linear regression: user_score = w·features + bias.

    Returns (feature_weights, bias_intercept). Uses np.linalg.lstsq for
    numerical stability on rank-deficient designs (early in life, several
    features will be zero across the whole sample → rank-deficient X).
    """
    n_features = len(FEATURE_NAMES)
    X = np.zeros((len(rows), n_features + 1), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows):
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = row.features.get(name, 0.0)
        X[i, -1] = 1.0  # bias column
        y[i] = row.user_score

    # rcond=None silences numpy's deprecation warning + uses the recommended
    # value (a small fraction of the largest singular value).
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[:-1], float(coef[-1])
