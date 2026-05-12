"""In-memory live-call state for the FastAPI service.

One CallState per active call_id; created on /call_start, mutated on
/transcript_delta, read on /suggestions, evicted on /call_end. Lost on
service restart by design — calls are minutes-long; Phase 5+ persistent
state lands when score overrides need to survive restarts.

Concurrency: FastAPI runs request handlers on a thread pool by default
(or on the event loop for `async def`). Multiple deltas + suggestion
polls for the same call can race; we use a lock per call_id to keep
the rolling window consistent under contention.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from consultant_brain.schemas import CallType, Speaker


# A typical live call generates ~6-12 turns/minute. Keeping the last 30 turns
# corresponds to roughly the last 2-3 minutes of conversation — more than
# the 60-90 second "rolling window" the master prompt calls for, which gives
# the retrieval enough context for multi-turn topics without ballooning the
# prompt size.
MAX_TURNS_PER_CALL = 30


@dataclass(slots=True)
class CallTurn:
    """One transcribed turn the Swift app sent us."""

    speaker: Speaker
    text: str
    received_at: datetime


@dataclass(slots=True)
class CallState:
    """One active call. Bounded transcript window prevents long-running
    calls from growing memory unboundedly.
    """

    call_id: str
    client: str | None
    call_type: CallType
    started_at: datetime
    turns: deque[CallTurn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS_PER_CALL))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append_turn(self, speaker: Speaker, text: str, *, now: datetime | None = None) -> None:
        """Add a turn under lock. `deque(maxlen=N)` evicts the oldest entry
        automatically when we exceed MAX_TURNS_PER_CALL."""
        with self.lock:
            self.turns.append(
                CallTurn(speaker=speaker, text=text, received_at=now or datetime.now(timezone.utc))
            )

    def transcript_window(self) -> str:
        """The You:/Them: formatted window the retriever consumes.
        Snapshots under lock so concurrent appends don't tear the output.
        """
        with self.lock:
            turns = list(self.turns)
        if not turns:
            return ""
        lines: list[str] = []
        for turn in turns:
            label = "You" if turn.speaker is Speaker.you else "Them"
            lines.append(f"{label}: {turn.text.strip()}")
        return "\n\n".join(lines)


class LiveCallRegistry:
    """Thread-safe registry of active calls. Wrapped in its own type so the
    FastAPI app's lifespan can swap in a fresh registry per process (tests
    can pass a clean instance instead of leaking state across requests).
    """

    def __init__(self) -> None:
        self._calls: dict[str, CallState] = {}
        self._lock = threading.Lock()

    def start(self, *, call_id: str, client: str | None, call_type: CallType) -> CallState:
        """Create or replace a CallState. Replacing on duplicate call_id is
        intentional: the live copilot might re-call /call_start after a
        crash/recovery; the cleanest behavior is a fresh window."""
        state = CallState(
            call_id=call_id,
            client=client,
            call_type=call_type,
            started_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._calls[call_id] = state
        return state

    def get(self, call_id: str) -> CallState | None:
        with self._lock:
            return self._calls.get(call_id)

    def end(self, call_id: str) -> CallState | None:
        with self._lock:
            return self._calls.pop(call_id, None)

    def active_call_ids(self) -> list[str]:
        with self._lock:
            return list(self._calls.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._calls)
