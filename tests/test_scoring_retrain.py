"""Retrain tests — sample-size gating, per-call-type fitting, weight
overwrite with backup.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from consultant_brain.scoring.corrections import (
    ScoreCorrection,
    append_correction,
    corrections_path,
)
from consultant_brain.scoring.features import FEATURE_NAMES
from consultant_brain.scoring.retrain import (
    MIN_CORRECTIONS_PER_TYPE,
    MIN_TOTAL_CORRECTIONS,
    retrain_weights,
)
from consultant_brain.scoring.weights import (
    DEFAULT_BIAS,
    DEFAULT_WEIGHTS,
    ScoringWeightsTable,
)
from consultant_brain.schemas import CallType
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


def _vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


def _correction(
    *,
    call_id: str,
    call_type: str,
    user_score: float,
    feature_values: dict[str, float] | None = None,
    predicted: float = 60.0,
) -> ScoreCorrection:
    features = {name: 0.0 for name in FEATURE_NAMES}
    if feature_values:
        features.update(feature_values)
    return ScoreCorrection(
        call_id=call_id,
        call_type=call_type,
        predicted_score=predicted,
        user_score=user_score,
        features=features,
        bias=DEFAULT_BIAS,
        created_at="2026-05-12T17:00:00Z",
    )


# ────────────────────────────────────────────────────────────────────────────
# Sample-size gating
# ────────────────────────────────────────────────────────────────────────────


def test_retrain_skipped_when_too_few_corrections(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    for i in range(MIN_TOTAL_CORRECTIONS - 1):
        append_correction(
            vault_root=vault_root,
            correction=_correction(call_id=f"c{i}", call_type="consultingCall", user_score=70),
        )
    report = retrain_weights(vault_root=vault_root)
    assert report.insufficient_data is True
    assert report.refit_call_types == []
    # No weights file written.
    assert not (vault_root / "00_System" / "scoring_weights.yaml").exists()


def test_retrain_only_refits_types_above_threshold(tmp_path: Path) -> None:
    """20 total → eligible. But if only consultingCall has ≥5 entries and
    other types have <5, only consultingCall gets refit."""
    vault_root = _vault(tmp_path)
    # 16 consulting corrections, 2 cold corrections, 2 follow_up = 20 total
    for i in range(16):
        append_correction(
            vault_root=vault_root,
            correction=_correction(
                call_id=f"cons{i}",
                call_type="consultingCall",
                user_score=70 + i % 5,
                feature_values={"commitments_made": (i % 5) / 5},
            ),
        )
    for i in range(2):
        append_correction(
            vault_root=vault_root,
            correction=_correction(call_id=f"cold{i}", call_type="coldCall", user_score=50),
        )
    for i in range(2):
        append_correction(
            vault_root=vault_root,
            correction=_correction(call_id=f"fu{i}", call_type="followUp", user_score=80),
        )

    report = retrain_weights(vault_root=vault_root)
    assert report.insufficient_data is False
    assert "consultingCall" in report.refit_call_types
    assert "coldCall" in report.skipped_call_types
    assert "followUp" in report.skipped_call_types


# ────────────────────────────────────────────────────────────────────────────
# Fit quality
# ────────────────────────────────────────────────────────────────────────────


def test_retrain_recovers_planted_weights(tmp_path: Path) -> None:
    """Generate synthetic corrections from a known ground-truth linear
    model; the retrain output should approximately recover the weights."""
    vault_root = _vault(tmp_path)
    rng = random.Random(42)

    # Ground truth: y = 30 (bias) + 25*next_step_booked + 15*commitments_made
    # All other features have weight 0.
    for i in range(40):
        next_step = float(rng.choice([0.0, 1.0]))
        commits = round(rng.random(), 2)
        y = 30 + 25 * next_step + 15 * commits + rng.gauss(0, 1.0)  # tiny noise
        y = max(0.0, min(100.0, y))
        append_correction(
            vault_root=vault_root,
            correction=_correction(
                call_id=f"syn{i}",
                call_type="consultingCall",
                user_score=y,
                feature_values={
                    "next_step_booked": next_step,
                    "commitments_made": commits,
                },
            ),
        )

    report = retrain_weights(vault_root=vault_root)
    assert "consultingCall" in report.refit_call_types

    table = ScoringWeightsTable.load(report.weights_path)
    weights = table.by_call_type[CallType.consulting_call]
    # Recovered weights should be within ~3 points of the planted values.
    assert abs(weights["next_step_booked"] - 25.0) < 3.0
    assert abs(weights["commitments_made"] - 15.0) < 3.0
    # Untouched features should be near 0 (the model wasn't given signal there).
    for name in FEATURE_NAMES:
        if name in ("next_step_booked", "commitments_made"):
            continue
        assert abs(weights[name]) < 5.0, f"{name} weight drifted: {weights[name]}"


# ────────────────────────────────────────────────────────────────────────────
# Backup behavior
# ────────────────────────────────────────────────────────────────────────────


def test_retrain_creates_backup_when_prior_weights_exist(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    # Pre-populate with a hand-edited weights file.
    prior = ScoringWeightsTable()
    prior_path = vault_root / "00_System" / "scoring_weights.yaml"
    prior.save(prior_path, backup=False)

    # Add enough synthetic corrections to trigger retrain.
    rng = random.Random(7)
    for i in range(MIN_TOTAL_CORRECTIONS):
        append_correction(
            vault_root=vault_root,
            correction=_correction(
                call_id=f"c{i}",
                call_type="consultingCall",
                user_score=50 + rng.random() * 30,
                feature_values={"commitments_made": rng.random()},
            ),
        )
    report = retrain_weights(vault_root=vault_root)
    assert report.backup_path is not None
    assert report.backup_path.exists()
    assert ".bak." in report.backup_path.name


def test_retrain_skips_backup_when_no_prior_weights(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    rng = random.Random(11)
    for i in range(MIN_TOTAL_CORRECTIONS):
        append_correction(
            vault_root=vault_root,
            correction=_correction(
                call_id=f"c{i}", call_type="consultingCall", user_score=70 + rng.random() * 5
            ),
        )
    report = retrain_weights(vault_root=vault_root)
    # No prior file → no backup file path.
    assert report.backup_path is None
    # But the new file IS written.
    assert report.weights_path.exists()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def test_retrain_cli_reports_insufficient_data(tmp_path: Path) -> None:
    from typer.testing import CliRunner
    from consultant_brain.cli import app

    runner = CliRunner()
    vault_root = _vault(tmp_path)
    result = runner.invoke(app, ["retrain", "--vault", str(vault_root)])
    assert result.exit_code == 0
    assert "Retrain skipped" in result.output
