"""User-override correction log.

When the user disagrees with the system's predicted score, they POST to
/score_override with `{call_id, user_score}`. We append one line to
`<vault>/00_System/score_corrections.jsonl` capturing:

  - the call_id + call_type
  - the predicted score + raw features (so retrain can re-fit weights)
  - the user-supplied corrected score
  - timestamp

After 20+ corrections accumulate, `consultant-brain retrain` walks this
file and fits a linear regression per call_type → new weights.

The file is append-only JSONL — safe for concurrent writes (one line per
write, the OS guarantees atomicity for small append() calls). Reads load
the whole file; for a single-user dev tool with <1000 entries this is
fine.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from consultant_brain.scoring.features import FEATURE_NAMES


CORRECTIONS_FILENAME = "score_corrections.jsonl"


@dataclass(frozen=True, slots=True)
class ScoreCorrection:
    """One row in the corrections log."""

    call_id: str
    call_type: str  # raw string so unknown enum values don't break replay
    predicted_score: float
    user_score: float
    features: dict[str, float]
    bias: float
    created_at: str  # ISO-8601 UTC with trailing Z

    @classmethod
    def make(
        cls,
        *,
        call_id: str,
        call_type: str,
        predicted_score: float,
        user_score: float,
        features: dict[str, float],
        bias: float,
        now: datetime | None = None,
    ) -> "ScoreCorrection":
        timestamp = now or datetime.now(timezone.utc)
        return cls(
            call_id=call_id,
            call_type=call_type,
            predicted_score=predicted_score,
            user_score=user_score,
            features=dict(features),
            bias=bias,
            created_at=timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


def corrections_path(vault_root: Path) -> Path:
    return vault_root.expanduser().resolve() / "00_System" / CORRECTIONS_FILENAME


def append_correction(*, vault_root: Path, correction: ScoreCorrection) -> Path:
    """Append one correction row. Returns the path."""
    path = corrections_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(correction), sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        # fsync makes the durability story honest: a kernel panic 1ms after
        # we tell the user "saved" shouldn't lose the correction.
        os.fsync(f.fileno())
    return path


def load_corrections(vault_root: Path) -> list[ScoreCorrection]:
    """Read every correction row. Tolerates malformed lines by skipping them
    — a single bad row shouldn't stop the retrain job from running on the
    100 good rows around it.
    """
    path = corrections_path(vault_root)
    if not path.exists():
        return []
    out: list[ScoreCorrection] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                features = raw.get("features", {})
                # Pad with 0 for any FEATURE_NAMES not present (lets us add
                # features over time without breaking old corrections).
                padded = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
                out.append(
                    ScoreCorrection(
                        call_id=raw["call_id"],
                        call_type=raw["call_type"],
                        predicted_score=float(raw["predicted_score"]),
                        user_score=float(raw["user_score"]),
                        features=padded,
                        bias=float(raw.get("bias", 50.0)),
                        created_at=raw.get("created_at", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out
