"""Live-loop tests — throttle behavior, atom writes hit the vault + index,
errors degrade cleanly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from consultant_brain.embedder import EMBEDDING_DIM, LanceVaultIndex
from consultant_brain.live_loop import (
    MOMENT_MIN_SECONDS_BETWEEN_RUNS,
    MOMENT_MIN_TURNS_BETWEEN_RUNS,
    LiveDetectionResult,
    maybe_run_moment_detection,
)
from consultant_brain.live_state import CallState
from consultant_brain.schemas import CallType, Speaker
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


# ────────────────────────────────────────────────────────────────────────────
# Mock Anthropic
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _MockAnthropic:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0

    def messages_create(self, **kwargs) -> _Response:
        self.calls += 1
        return _Response(content=[_Block(text=self.response_text)])


VALID_RESPONSE = json.dumps(
    {
        "atoms": [
            {"type": "commitment", "body": "Reece will send 3 ads by Friday.", "confidence": 0.92, "tags": ["next_step"]},
            {"type": "client_fact", "body": "About 1000 notes in an Obsidian vault.", "confidence": 0.88, "tags": ["stack"]},
        ]
    }
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _make_state(*, started_at: datetime | None = None) -> CallState:
    return CallState(
        call_id="livetest_001",
        client="Reece",
        call_type=CallType.consulting_call,
        started_at=started_at or datetime(2026, 5, 12, 19, 0, tzinfo=timezone.utc),
    )


def _seed_turns(state: CallState, count: int) -> None:
    for i in range(count):
        state.append_turn(Speaker.them if i % 2 else Speaker.you, f"turn {i}")


def _vault(tmp_path: Path) -> Path:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    return tmp_path / "vault"


# ────────────────────────────────────────────────────────────────────────────
# Throttle
# ────────────────────────────────────────────────────────────────────────────


def test_skips_when_not_enough_new_turns(tmp_path: Path) -> None:
    state = _make_state()
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS - 1)
    client = _MockAnthropic(VALID_RESPONSE)
    result = maybe_run_moment_detection(state, vault_root=_vault(tmp_path), anthropic_client=client)
    assert result.ran is False
    assert "new turn" in result.skipped_reason
    assert client.calls == 0


def test_skips_when_too_soon_since_last_run(tmp_path: Path) -> None:
    state = _make_state()
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS + 5)
    state.last_moment_detection_at = datetime(2026, 5, 12, 19, 0, tzinfo=timezone.utc)
    state.turns_at_last_detection = 0  # plenty of new turns, but too soon
    now = state.last_moment_detection_at + timedelta(seconds=MOMENT_MIN_SECONDS_BETWEEN_RUNS - 5)

    client = _MockAnthropic(VALID_RESPONSE)
    result = maybe_run_moment_detection(
        state, vault_root=_vault(tmp_path), anthropic_client=client, now=now
    )
    assert result.ran is False
    assert "since last run" in result.skipped_reason
    assert client.calls == 0


def test_runs_when_throttle_satisfied(tmp_path: Path) -> None:
    state = _make_state()
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS + 3)
    state.last_moment_detection_at = datetime(2026, 5, 12, 19, 0, tzinfo=timezone.utc)
    state.turns_at_last_detection = 0
    now = state.last_moment_detection_at + timedelta(seconds=MOMENT_MIN_SECONDS_BETWEEN_RUNS + 5)

    client = _MockAnthropic(VALID_RESPONSE)
    result = maybe_run_moment_detection(
        state, vault_root=_vault(tmp_path), anthropic_client=client, now=now
    )
    assert result.ran is True
    assert client.calls == 1
    assert state.last_moment_detection_at == now


# ────────────────────────────────────────────────────────────────────────────
# Atom writes
# ────────────────────────────────────────────────────────────────────────────


def test_detected_atoms_get_written_to_vault(tmp_path: Path) -> None:
    state = _make_state()
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS + 1)
    vault_root = _vault(tmp_path)

    client = _MockAnthropic(VALID_RESPONSE)
    result = maybe_run_moment_detection(state, vault_root=vault_root, anthropic_client=client)
    assert result.ran is True
    assert result.atoms_written == 2

    layout = VaultLayout.for_root(vault_root)
    atom_files = list(layout.atoms_dir.glob("*.md"))
    assert len(atom_files) == 2
    # Body content from the mocked response should appear in the markdown.
    bodies = " ".join(p.read_text() for p in atom_files)
    assert "Reece will send 3 ads" in bodies
    assert "1000 notes" in bodies


def test_atom_ids_are_unique_across_multiple_passes(tmp_path: Path) -> None:
    """The throttle prevents adjacent passes, but if we step time forward
    each detection pass adds new atom files — IDs derive from a monotonic
    base index so they don't collide across passes within the same call.
    """
    vault_root = _vault(tmp_path)
    state = _make_state()
    # Pass 1: write 2 atoms
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS + 1)
    base = datetime(2026, 5, 12, 19, 0, tzinfo=timezone.utc)
    maybe_run_moment_detection(state, vault_root=vault_root, anthropic_client=_MockAnthropic(VALID_RESPONSE), now=base)
    # Pass 2: add more turns + step time, write 2 more atoms
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS + 1)
    later = base + timedelta(seconds=MOMENT_MIN_SECONDS_BETWEEN_RUNS + 5)
    maybe_run_moment_detection(state, vault_root=vault_root, anthropic_client=_MockAnthropic(VALID_RESPONSE), now=later)

    layout = VaultLayout.for_root(vault_root)
    atom_files = list(layout.atoms_dir.glob("*.md"))
    assert len(atom_files) == 4, "expected 2 atoms per pass × 2 passes = 4 distinct atoms"
    # Detected atom count tracks across passes.
    assert state.detected_atom_count == 4


def test_quiet_pass_advances_throttle_without_atoms(tmp_path: Path) -> None:
    """When the detector returns no atoms, we still update the throttle so
    we don't immediately re-attempt detection."""
    empty = json.dumps({"atoms": []})
    state = _make_state()
    _seed_turns(state, MOMENT_MIN_TURNS_BETWEEN_RUNS + 1)
    now = datetime(2026, 5, 12, 19, 0, tzinfo=timezone.utc)
    result = maybe_run_moment_detection(
        state, vault_root=_vault(tmp_path), anthropic_client=_MockAnthropic(empty), now=now
    )
    assert result.ran is True
    assert result.atoms_written == 0
    assert state.last_moment_detection_at == now
    # Subsequent immediate call is throttled.
    assert maybe_run_moment_detection(
        state, vault_root=_vault(tmp_path), anthropic_client=_MockAnthropic(VALID_RESPONSE), now=now + timedelta(seconds=1)
    ).ran is False


# ────────────────────────────────────────────────────────────────────────────
# Empty / error tolerance
# ────────────────────────────────────────────────────────────────────────────


def test_skips_with_zero_turns(tmp_path: Path) -> None:
    state = _make_state()
    result = maybe_run_moment_detection(state, vault_root=_vault(tmp_path), anthropic_client=_MockAnthropic("{}"))
    assert result.ran is False
