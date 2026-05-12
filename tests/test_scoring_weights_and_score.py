"""Weights + score-computation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from consultant_brain.schemas import CallType
from consultant_brain.scoring.features import FEATURE_NAMES, ScoreFeatures
from consultant_brain.scoring.score import FeatureContribution, ScoreResult, compute_score
from consultant_brain.scoring.weights import (
    DEFAULT_BIAS,
    DEFAULT_WEIGHTS,
    ScoringWeightsTable,
)


# ────────────────────────────────────────────────────────────────────────────
# Weights table
# ────────────────────────────────────────────────────────────────────────────


def test_default_weights_cover_every_call_type_and_feature() -> None:
    for ct in CallType:
        weights = DEFAULT_WEIGHTS[ct]
        for name in FEATURE_NAMES:
            assert name in weights, f"{ct.value} missing weight for {name}"


def test_default_weights_capture_call_type_intuition() -> None:
    """Cold calls weight `next_step_booked` heaviest; closing calls weight
    `commitments_made` heaviest. If these intuitions invert, retrain regressed."""
    cold = DEFAULT_WEIGHTS[CallType.cold_call]
    closing = DEFAULT_WEIGHTS[CallType.closing_call]
    training = DEFAULT_WEIGHTS[CallType.ai_training]

    assert cold["next_step_booked"] == max(cold[name] for name in FEATURE_NAMES)
    assert closing["commitments_made"] == max(closing[name] for name in FEATURE_NAMES)
    # Training is the one type where confusion is POSITIVE — questions
    # during training are good.
    assert training["confusion_events"] > 0
    # And confusion is NEGATIVE everywhere else.
    for other in (CallType.consulting_call, CallType.cold_call, CallType.closing_call,
                  CallType.follow_up, CallType.implementation):
        assert DEFAULT_WEIGHTS[other]["confusion_events"] < 0, f"confusion should hurt {other.value}"


def test_for_call_type_returns_resolved_weights() -> None:
    table = ScoringWeightsTable()
    weights = table.for_call_type(CallType.consulting_call)
    assert weights.call_type is CallType.consulting_call
    assert weights.bias == DEFAULT_BIAS
    assert weights.weights["next_step_booked"] == DEFAULT_WEIGHTS[CallType.consulting_call]["next_step_booked"]


def test_for_call_type_fills_missing_keys_from_defaults() -> None:
    """A partial YAML with only some features set shouldn't crash the scorer."""
    table = ScoringWeightsTable(
        by_call_type={CallType.consulting_call: {"next_step_booked": 99.0}}
    )
    weights = table.for_call_type(CallType.consulting_call)
    # The one explicit value flows through:
    assert weights.weights["next_step_booked"] == 99.0
    # Everything else falls back to default:
    assert weights.weights["commitments_made"] == DEFAULT_WEIGHTS[CallType.consulting_call]["commitments_made"]


# ────────────────────────────────────────────────────────────────────────────
# YAML persistence
# ────────────────────────────────────────────────────────────────────────────


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    table = ScoringWeightsTable.load(tmp_path / "no-file.yaml")
    assert table.bias == DEFAULT_BIAS
    for ct in CallType:
        assert table.by_call_type[ct] == DEFAULT_WEIGHTS[ct]


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    custom = ScoringWeightsTable(
        bias=60.0,
        by_call_type={ct: dict(DEFAULT_WEIGHTS[ct]) for ct in CallType},
    )
    custom.by_call_type[CallType.cold_call]["next_step_booked"] = 27.5
    path = tmp_path / "weights.yaml"
    custom.save(path)

    reloaded = ScoringWeightsTable.load(path)
    assert reloaded.bias == 60.0
    assert reloaded.by_call_type[CallType.cold_call]["next_step_booked"] == 27.5


def test_save_creates_backup_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "weights.yaml"
    table = ScoringWeightsTable()
    # First save — no backup created (no existing file).
    assert table.save(path) is None
    # Second save — backup of the first file.
    backup_path = table.save(path)
    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path.suffix.startswith(".bak.") is False  # nested suffixes confuse Path.suffix
    assert ".bak." in backup_path.name


def test_save_can_skip_backup_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "weights.yaml"
    table = ScoringWeightsTable()
    table.save(path)
    assert table.save(path, backup=False) is None


def test_load_malformed_yaml_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "weights.yaml"
    path.write_text("not: valid: yaml::")
    table = ScoringWeightsTable.load(path)
    # Falls back to defaults rather than crashing.
    assert table.bias == DEFAULT_BIAS


# ────────────────────────────────────────────────────────────────────────────
# Score computation
# ────────────────────────────────────────────────────────────────────────────


def _features(**overrides) -> ScoreFeatures:
    base = dict(
        next_step_booked=0.0,
        objections_resolved=0.0,
        talk_ratio_balance=0.0,
        commitments_made=0.0,
        win_signals=0.0,
        loss_signals=0.0,
        confusion_events=0.0,
        completion_of_agenda=0.0,
        primary_win_progress=0.0,
    )
    base.update(overrides)
    return ScoreFeatures(**base)


def test_score_at_all_zeros_equals_bias() -> None:
    table = ScoringWeightsTable()
    weights = table.for_call_type(CallType.consulting_call)
    result = compute_score(features=_features(), weights=weights)
    assert result.score == DEFAULT_BIAS
    assert result.raw_score == DEFAULT_BIAS


def test_score_with_saturated_positives_is_high() -> None:
    table = ScoringWeightsTable()
    weights = table.for_call_type(CallType.consulting_call)
    f = ScoreFeatures(
        next_step_booked=1.0,
        objections_resolved=1.0,
        talk_ratio_balance=1.0,
        commitments_made=1.0,
        win_signals=1.0,
        loss_signals=0.0,
        confusion_events=0.0,
        completion_of_agenda=1.0,
        primary_win_progress=1.0,
    )
    result = compute_score(features=f, weights=weights)
    # Positive-only weights sum to ~60; bias 50 → unclamped 110 → clamped 100.
    assert result.score == 100.0
    assert result.raw_score > 100


def test_score_with_saturated_negatives_drags_down() -> None:
    table = ScoringWeightsTable()
    weights = table.for_call_type(CallType.consulting_call)
    f = ScoreFeatures(
        next_step_booked=0.0,
        objections_resolved=0.0,
        talk_ratio_balance=0.0,
        commitments_made=0.0,
        win_signals=0.0,
        loss_signals=1.0,   # saturates a negative-weighted feature
        confusion_events=1.0,
        completion_of_agenda=0.0,
        primary_win_progress=0.0,
    )
    result = compute_score(features=f, weights=weights)
    # bias 50 + (-10 loss) + (-5 confusion) = 35
    assert result.score == pytest.approx(35.0, abs=0.01)


def test_score_breakdown_lists_every_feature() -> None:
    table = ScoringWeightsTable()
    weights = table.for_call_type(CallType.consulting_call)
    result = compute_score(features=_features(commitments_made=0.5), weights=weights)
    names = [c.name for c in result.contributions]
    assert names == list(FEATURE_NAMES)
    commits = next(c for c in result.contributions if c.name == "commitments_made")
    assert commits.raw_value == 0.5
    assert commits.contribution == pytest.approx(0.5 * weights.weights["commitments_made"])


def test_score_summary_line_includes_top_contributor() -> None:
    table = ScoringWeightsTable()
    weights = table.for_call_type(CallType.consulting_call)
    f = _features(commitments_made=1.0, next_step_booked=1.0, win_signals=0.5)
    result = compute_score(features=f, weights=weights)
    # commitments_made saturated × 8 weight = 8 contribution; next_step_booked
    # × 12 = 12 contribution → next_step_booked should be the top contributor.
    summary = result.summary_line()
    assert "consultingCall" in summary
    assert "next_step_booked" in summary
    assert "+12" in summary


def test_clamp_disabled_returns_unclamped_for_debugging() -> None:
    table = ScoringWeightsTable(bias=200.0)
    weights = table.for_call_type(CallType.consulting_call)
    result = compute_score(features=_features(), weights=weights, clamp=False)
    assert result.score == 200.0
