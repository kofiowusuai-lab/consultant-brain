"""Post-call score computation.

Takes the 9-feature vector + the per-call-type weights and produces a
0-100 score plus a per-feature contribution breakdown so the UI (and
`consultant-brain score --explain`) can show WHY the call rated where
it did.

Contract:
  score = clamp(bias + sum(features[i] * weights[i]), 0, 100)

Each contribution = features[i] * weights[i] — directly readable: "the
'commitments_made' feature contributed +14.4 points to this score".

The score returned here is "predicted" — the user override path lives in
the service module + correction log. Predicted score is what gets shown
on the dashboard the moment the call ends.
"""

from __future__ import annotations

from dataclasses import dataclass

from consultant_brain.schemas import CallType
from consultant_brain.scoring.features import FEATURE_NAMES, ScoreFeatures
from consultant_brain.scoring.weights import ScoringWeights


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One row in the per-feature breakdown."""

    name: str
    raw_value: float       # feature value in [-1, 1]
    weight: float          # weight applied (signed)
    contribution: float    # raw_value * weight — the points this feature added


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """The output the scorer produces. `score` is the user-facing 0-100;
    `raw_score` is the unclamped value (lets debugging see if a weight is
    overshooting the clip boundary).
    """

    score: float
    raw_score: float
    bias: float
    call_type: CallType
    contributions: list[FeatureContribution]

    def summary_line(self) -> str:
        """One-line CLI summary: '72/100 · consultingCall · top: commitments_made +14.4'"""
        if self.contributions:
            top = max(self.contributions, key=lambda c: c.contribution)
            top_part = f" · top: {top.name} {top.contribution:+.1f}"
        else:
            top_part = ""
        return f"{round(self.score)}/100 · {self.call_type.value}{top_part}"


def compute_score(
    *,
    features: ScoreFeatures,
    weights: ScoringWeights,
    clamp: bool = True,
) -> ScoreResult:
    """Dot-product the features against the weights, add the bias,
    optionally clamp to [0, 100]."""
    feature_vec = features.as_vector()
    weight_vec = weights.vector()
    contributions: list[FeatureContribution] = []
    accumulated = weights.bias
    for name, raw, w in zip(FEATURE_NAMES, feature_vec, weight_vec):
        contrib = raw * w
        accumulated += contrib
        contributions.append(
            FeatureContribution(name=name, raw_value=raw, weight=w, contribution=contrib)
        )
    raw_score = accumulated
    final = max(0.0, min(100.0, raw_score)) if clamp else raw_score
    return ScoreResult(
        score=final,
        raw_score=raw_score,
        bias=weights.bias,
        call_type=weights.call_type,
        contributions=contributions,
    )
