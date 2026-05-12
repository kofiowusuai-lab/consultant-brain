"""Pattern stability metric.

The premise (from the master prompt): "Patterns mined in month 2
demonstrably outperform patterns from month 1 on the same call types".
We approximate this by snapshotting the pattern state after every
`distill` run and computing two metrics across the latest pair of
snapshots taken ≥COMPARISON_GAP_DAYS apart:

  persistence_rate    — % of patterns in the OLDER snapshot still active in the NEWER one
  mean_obs_delta      — average change in observation_count for surviving patterns

A pattern that disappears between snapshots (status → retired, or member
atoms dropped below threshold) is counted as not-persisting. Rising
observation counts on surviving patterns is the cumulative-learning
signal the master prompt asks us to prove.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import frontmatter

from consultant_brain.vault import VaultLayout


SNAPSHOT_FILENAME = "pattern_snapshots.jsonl"
COMPARISON_GAP_DAYS = 30


@dataclass(frozen=True, slots=True)
class PatternSnapshotRow:
    """One pattern's state at one moment in time."""

    pattern_id: str
    atom_type: str
    primary_tag: str
    observation_count: int
    unique_call_count: int
    score_impact_proxy: float
    status: str
    member_atom_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PatternSnapshot:
    """One full snapshot — every pattern's state at this moment."""

    taken_at: str  # ISO-8601 Z
    patterns: tuple[PatternSnapshotRow, ...]


@dataclass(frozen=True, slots=True)
class PatternStabilityReport:
    snapshots_total: int
    latest_snapshot_at: str | None
    older_snapshot_at: str | None
    pair_gap_days: float
    persistence_rate: float  # 0..1
    mean_observation_count_delta: float
    insufficient_data: bool = False


def snapshot_path(vault_root: Path) -> Path:
    return vault_root.expanduser().resolve() / "00_System" / SNAPSHOT_FILENAME


def take_snapshot(*, vault_root: Path, now: datetime | None = None) -> PatternSnapshot:
    """Read every pattern in 04_Patterns/, build a snapshot, append it
    to the snapshot log. Called by `distill` after every mine pass."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    layout = VaultLayout.for_root(vault_root)
    patterns_dir = layout.root / "04_Patterns"

    rows: list[PatternSnapshotRow] = []
    if patterns_dir.exists():
        for path in patterns_dir.glob("*.md"):
            try:
                post = frontmatter.load(path.open("r", encoding="utf-8"))
            except Exception:
                continue
            meta = dict(post.metadata)
            if meta.get("kind") != "pattern":
                continue
            rows.append(
                PatternSnapshotRow(
                    pattern_id=str(meta.get("id", path.stem)),
                    atom_type=str(meta.get("type", "")),
                    primary_tag=str(meta.get("primary_tag", "")),
                    observation_count=int(meta.get("observation_count", 0)),
                    unique_call_count=int(meta.get("unique_call_count", 0)),
                    score_impact_proxy=float(meta.get("score_impact_proxy", 0.5)),
                    status=str(meta.get("status", "active")),
                    member_atom_ids=tuple(meta.get("member_atom_ids", []) or []),
                )
            )

    snapshot = PatternSnapshot(taken_at=timestamp, patterns=tuple(rows))
    _append_snapshot(vault_root=vault_root, snapshot=snapshot)
    return snapshot


def load_snapshots(vault_root: Path) -> list[PatternSnapshot]:
    """Read every snapshot. Skips malformed rows."""
    path = snapshot_path(vault_root)
    if not path.exists():
        return []
    out: list[PatternSnapshot] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                out.append(
                    PatternSnapshot(
                        taken_at=raw["taken_at"],
                        patterns=tuple(
                            PatternSnapshotRow(
                                pattern_id=p["pattern_id"],
                                atom_type=p["atom_type"],
                                primary_tag=p["primary_tag"],
                                observation_count=int(p["observation_count"]),
                                unique_call_count=int(p["unique_call_count"]),
                                score_impact_proxy=float(p["score_impact_proxy"]),
                                status=p.get("status", "active"),
                                member_atom_ids=tuple(p.get("member_atom_ids", []) or []),
                            )
                            for p in raw.get("patterns", [])
                        ),
                    )
                )
            except (KeyError, json.JSONDecodeError, TypeError, ValueError):
                continue
    return out


def compute_pattern_stability(*, vault_root: Path) -> PatternStabilityReport:
    """Compare the latest snapshot with one ≥COMPARISON_GAP_DAYS old."""
    snapshots = load_snapshots(vault_root)
    if len(snapshots) < 2:
        return PatternStabilityReport(
            snapshots_total=len(snapshots),
            latest_snapshot_at=snapshots[-1].taken_at if snapshots else None,
            older_snapshot_at=None,
            pair_gap_days=0.0,
            persistence_rate=0.0,
            mean_observation_count_delta=0.0,
            insufficient_data=True,
        )

    snapshots.sort(key=lambda s: s.taken_at)
    latest = snapshots[-1]
    latest_dt = _parse(latest.taken_at)

    # Find the most-recent snapshot at least COMPARISON_GAP_DAYS before latest.
    older = None
    if latest_dt is not None:
        cutoff = latest_dt - timedelta(days=COMPARISON_GAP_DAYS)
        for snap in reversed(snapshots[:-1]):
            snap_dt = _parse(snap.taken_at)
            if snap_dt and snap_dt <= cutoff:
                older = snap
                break

    if older is None:
        # No old-enough snapshot — degrade gracefully by comparing against
        # the very first snapshot so we still surface something useful.
        older = snapshots[0]

    older_dt = _parse(older.taken_at)
    gap_days = 0.0
    if latest_dt and older_dt:
        gap_days = (latest_dt - older_dt).total_seconds() / 86400.0

    older_active = {p.pattern_id for p in older.patterns if p.status == "active"}
    latest_active = {p.pattern_id for p in latest.patterns if p.status == "active"}

    if not older_active:
        return PatternStabilityReport(
            snapshots_total=len(snapshots),
            latest_snapshot_at=latest.taken_at,
            older_snapshot_at=older.taken_at,
            pair_gap_days=gap_days,
            persistence_rate=0.0,
            mean_observation_count_delta=0.0,
            insufficient_data=True,
        )

    persisting = older_active & latest_active
    persistence = len(persisting) / len(older_active)

    older_obs = {p.pattern_id: p.observation_count for p in older.patterns}
    latest_obs = {p.pattern_id: p.observation_count for p in latest.patterns}
    deltas = [
        latest_obs[pid] - older_obs[pid]
        for pid in persisting
        if pid in latest_obs and pid in older_obs
    ]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return PatternStabilityReport(
        snapshots_total=len(snapshots),
        latest_snapshot_at=latest.taken_at,
        older_snapshot_at=older.taken_at,
        pair_gap_days=gap_days,
        persistence_rate=persistence,
        mean_observation_count_delta=mean_delta,
        insufficient_data=False,
    )


# ────────────────────────────────────────────────────────────────────────────


def _append_snapshot(*, vault_root: Path, snapshot: PatternSnapshot) -> None:
    path = snapshot_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "taken_at": snapshot.taken_at,
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "atom_type": p.atom_type,
                "primary_tag": p.primary_tag,
                "observation_count": p.observation_count,
                "unique_call_count": p.unique_call_count,
                "score_impact_proxy": p.score_impact_proxy,
                "status": p.status,
                "member_atom_ids": list(p.member_atom_ids),
            }
            for p in snapshot.patterns
        ],
    }
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _parse(iso: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
