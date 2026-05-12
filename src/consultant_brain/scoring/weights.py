"""Scoring weights — per-call-type profiles + YAML persistence.

Each profile is a 9-vector of floats, one per feature in FEATURE_NAMES.
Weights are interpretable directly: a +20 commitments_made weight means
"going from 0 commitments (feature=0) to saturated (feature=1) bumps the
score by 20 points before clipping".

Defaults bake in the master prompt's call-type intuition:
  Closing  — commitments + next_step heavy.
  Cold     — next_step + win_signals dominate.
  Training — confusion is POSITIVE (questions are good when learning).
             other features matter less; this is a low-stakes scoring
             surface vs sales calls.
  Follow-up + Implementation — emphasize commitments and resolved
             objections (this is execution time).

The defaults are intentionally conservative: every feature contributes
to a 60-65 baseline score with maybe 20 points of headroom for an
excellent call. The retrain step learns sharper weights once the
correction log has 20+ entries.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from consultant_brain.schemas import CallType
from consultant_brain.scoring.features import FEATURE_NAMES, ScoreFeatures


# Bias term (intercept) — a baseline score before any features fire.
# Tuned so a "median call" lands around 55-65 before per-feature bumps.
DEFAULT_BIAS = 50.0


# Default per-call-type weight profiles. Sum of (positives) ≈ 50 points so
# a saturated-positive call lands near 100; saturated-negative lands near 0.
DEFAULT_WEIGHTS: dict[CallType, dict[str, float]] = {
    CallType.consulting_call: {
        "next_step_booked": 12.0,
        "objections_resolved": 10.0,
        "talk_ratio_balance": 7.0,
        "commitments_made": 8.0,
        "win_signals": 6.0,
        "loss_signals": -10.0,
        "confusion_events": -5.0,
        "completion_of_agenda": 5.0,
        "primary_win_progress": 12.0,
    },
    CallType.ai_training: {
        "next_step_booked": 4.0,
        "objections_resolved": 4.0,
        "talk_ratio_balance": 8.0,
        "commitments_made": 5.0,
        "win_signals": 6.0,
        "loss_signals": -6.0,
        "confusion_events": 8.0,  # Questions during training = good
        "completion_of_agenda": 12.0,
        "primary_win_progress": 12.0,
    },
    CallType.cold_call: {
        "next_step_booked": 22.0,  # The single most important signal
        "objections_resolved": 6.0,
        "talk_ratio_balance": 5.0,
        "commitments_made": 4.0,
        "win_signals": 6.0,
        "loss_signals": -8.0,
        "confusion_events": -3.0,
        "completion_of_agenda": 3.0,
        "primary_win_progress": 8.0,
    },
    CallType.closing_call: {
        "next_step_booked": 8.0,
        "objections_resolved": 12.0,
        "talk_ratio_balance": 5.0,
        "commitments_made": 18.0,  # Closing IS commitment-making
        "win_signals": 8.0,
        "loss_signals": -10.0,
        "confusion_events": -8.0,  # Confusion at closing = trouble
        "completion_of_agenda": 4.0,
        "primary_win_progress": 10.0,
    },
    CallType.follow_up: {
        "next_step_booked": 10.0,
        "objections_resolved": 10.0,
        "talk_ratio_balance": 5.0,
        "commitments_made": 12.0,
        "win_signals": 5.0,
        "loss_signals": -8.0,
        "confusion_events": -4.0,
        "completion_of_agenda": 6.0,
        "primary_win_progress": 8.0,
    },
    CallType.implementation: {
        "next_step_booked": 8.0,
        "objections_resolved": 8.0,
        "talk_ratio_balance": 5.0,
        "commitments_made": 14.0,
        "win_signals": 4.0,
        "loss_signals": -8.0,
        "confusion_events": -6.0,
        "completion_of_agenda": 10.0,
        "primary_win_progress": 10.0,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """Resolved weights for ONE call_type — what the scorer actually uses."""

    call_type: CallType
    bias: float
    weights: dict[str, float]  # keys = FEATURE_NAMES

    def vector(self) -> list[float]:
        """Order matches FEATURE_NAMES so we can `np.dot(features, weights)`."""
        return [self.weights[name] for name in FEATURE_NAMES]


@dataclass
class ScoringWeightsTable:
    """All-call-types weight table, plus the on-disk path it loads from /
    saves to. Behaves like a small repository — load() / save() handle the
    YAML round-trip + .bak backup.
    """

    bias: float = field(default=DEFAULT_BIAS)
    by_call_type: dict[CallType, dict[str, float]] = field(
        default_factory=lambda: {ct: dict(DEFAULT_WEIGHTS[ct]) for ct in CallType}
    )
    last_loaded_path: Path | None = None

    # ---- Convenience getters ----

    def for_call_type(self, call_type: CallType) -> ScoringWeights:
        weights = self.by_call_type.get(call_type) or DEFAULT_WEIGHTS[call_type]
        # Fill any missing keys from defaults so a partial YAML doesn't crash.
        filled = {name: weights.get(name, DEFAULT_WEIGHTS[call_type][name]) for name in FEATURE_NAMES}
        return ScoringWeights(call_type=call_type, bias=self.bias, weights=filled)

    # ---- Persistence ----

    @classmethod
    def load(cls, path: Path) -> "ScoringWeightsTable":
        """Read a weights YAML. Missing file → defaults. Malformed file →
        falls back to defaults + writes nothing (callers can decide whether
        to overwrite).
        """
        if not path.exists():
            return cls(last_loaded_path=path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return cls(last_loaded_path=path)
        bias = float(data.get("bias", DEFAULT_BIAS))
        by_type_raw = data.get("by_call_type", {}) or {}
        by_type: dict[CallType, dict[str, float]] = {}
        for ct in CallType:
            raw = by_type_raw.get(ct.value, {}) or {}
            # Same default-fill discipline as `for_call_type` so partial
            # YAML never raises a KeyError downstream.
            filled = {
                name: float(raw.get(name, DEFAULT_WEIGHTS[ct][name]))
                for name in FEATURE_NAMES
            }
            by_type[ct] = filled
        return cls(bias=bias, by_call_type=by_type, last_loaded_path=path)

    def save(self, path: Path, *, backup: bool = True) -> Path | None:
        """Atomic write with optional `.bak` of the existing file. Returns
        the backup path on disk (None when no backup was made)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_path: Path | None = None
        if backup and path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = path.with_suffix(path.suffix + f".bak.{stamp}")
            shutil.copy2(path, backup_path)

        out: dict[str, Any] = {
            "_doc": "Scoring weights for the consultant-brain post-call scorer. Bias + 9 feature weights per call type. Edit by hand if you know what you're doing; `consultant-brain retrain` regenerates from score_corrections.jsonl.",
            "bias": self.bias,
            "by_call_type": {
                ct.value: dict(self.by_call_type.get(ct, DEFAULT_WEIGHTS[ct])) for ct in CallType
            },
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        # write-tmp + rename for atomicity (same pattern as vault.py)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        self.last_loaded_path = path
        return backup_path
