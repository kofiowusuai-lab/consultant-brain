"""Live call state store tests — registry lifecycle, window bounding,
thread-safety under concurrent appends.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from consultant_brain.live_state import (
    MAX_TURNS_PER_CALL,
    CallState,
    LiveCallRegistry,
)
from consultant_brain.schemas import CallType, Speaker


def test_registry_start_returns_state() -> None:
    registry = LiveCallRegistry()
    state = registry.start(call_id="abc", client="Reece", call_type=CallType.consulting_call)
    assert state.call_id == "abc"
    assert state.client == "Reece"
    assert state.call_type is CallType.consulting_call
    assert len(registry) == 1


def test_registry_get_returns_active_state() -> None:
    registry = LiveCallRegistry()
    registry.start(call_id="abc", client="Reece", call_type=CallType.consulting_call)
    state = registry.get("abc")
    assert state is not None
    assert state.client == "Reece"


def test_registry_get_unknown_returns_none() -> None:
    assert LiveCallRegistry().get("missing") is None


def test_registry_end_pops_state() -> None:
    registry = LiveCallRegistry()
    registry.start(call_id="abc", client="Reece", call_type=CallType.consulting_call)
    popped = registry.end("abc")
    assert popped is not None
    assert popped.call_id == "abc"
    assert registry.get("abc") is None
    assert len(registry) == 0


def test_registry_end_unknown_returns_none() -> None:
    assert LiveCallRegistry().end("missing") is None


def test_registry_start_replaces_duplicate_id() -> None:
    """A second /call_start with the same call_id replaces — a crash-recovery
    behavior we want to be explicit about."""
    registry = LiveCallRegistry()
    first = registry.start(call_id="abc", client="Reece", call_type=CallType.consulting_call)
    first.append_turn(Speaker.them, "old turn")
    second = registry.start(call_id="abc", client="Reece2", call_type=CallType.cold_call)
    assert second is not first
    assert second.client == "Reece2"
    assert len(second.turns) == 0


def test_transcript_window_formats_you_them_labels() -> None:
    state = CallState(
        call_id="abc",
        client="Reece",
        call_type=CallType.consulting_call,
        started_at=datetime.now(timezone.utc),
    )
    state.append_turn(Speaker.you, "Walk me through your stack.")
    state.append_turn(Speaker.them, "We use Obsidian.")
    window = state.transcript_window()
    assert "You: Walk me through your stack." in window
    assert "Them: We use Obsidian." in window


def test_transcript_window_caps_at_max_turns_per_call() -> None:
    state = CallState(
        call_id="abc",
        client="Reece",
        call_type=CallType.consulting_call,
        started_at=datetime.now(timezone.utc),
    )
    for i in range(MAX_TURNS_PER_CALL + 5):
        state.append_turn(Speaker.them, f"turn {i}")
    # Only the last MAX_TURNS_PER_CALL turns survive.
    assert len(state.turns) == MAX_TURNS_PER_CALL
    window = state.transcript_window()
    # The first 5 turns should be evicted.
    assert "turn 0" not in window
    assert "turn 4" not in window
    assert "turn 5" in window  # the (MAX+5)-MAX = 5th turn is now the oldest visible
    assert f"turn {MAX_TURNS_PER_CALL + 4}" in window


def test_concurrent_append_turn_keeps_invariants() -> None:
    """Stress test: 8 threads each writing 50 turns. After join, len(turns)
    is capped at MAX_TURNS_PER_CALL and the deque is internally consistent.
    """
    state = CallState(
        call_id="abc",
        client="Reece",
        call_type=CallType.consulting_call,
        started_at=datetime.now(timezone.utc),
    )

    def worker(tid: int) -> None:
        for i in range(50):
            state.append_turn(Speaker.them, f"t{tid}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(state.turns) == MAX_TURNS_PER_CALL
    # Sanity: window renders without error after concurrent writes.
    window = state.transcript_window()
    assert window.count("Them:") == MAX_TURNS_PER_CALL


def test_empty_state_returns_empty_window() -> None:
    state = CallState(
        call_id="abc",
        client="Reece",
        call_type=CallType.consulting_call,
        started_at=datetime.now(timezone.utc),
    )
    assert state.transcript_window() == ""
