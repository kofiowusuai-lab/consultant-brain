"""Session loader tests. Use the fixture transcript so they're deterministic
and don't depend on whatever calls happen to be in the user's Sessions dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consultant_brain.session_loader import load_session
from consultant_brain.schemas import Speaker


FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.json"


def test_load_session_returns_normalized_struct() -> None:
    loaded = load_session(FIXTURE)
    assert loaded.source_filename == "sample_session.json"
    assert loaded.turn_count == 10
    # First turn at 15:43:04, last at 15:47:48 — under 5 minutes.
    assert loaded.duration_minutes <= 5
    assert loaded.duration_minutes >= 4


def test_transcript_uses_you_and_them_labels() -> None:
    loaded = load_session(FIXTURE)
    assert "You:" in loaded.transcript
    assert "Them:" in loaded.transcript
    # No raw "microphone" or "systemAudio" leaking through.
    assert "microphone" not in loaded.transcript
    assert "systemAudio" not in loaded.transcript


def test_transcript_merges_consecutive_same_speaker_turns() -> None:
    # The fixture alternates speakers, so each speaker block is a single turn.
    # Build a tiny synthetic case to verify merging when two `You:` turns hit
    # back-to-back (which the VAD does produce in real calls).
    crafted = {
        "startedAt": "2026-01-01T00:00:00Z",
        "completedTurns": [
            {
                "completedAt": "2026-01-01T00:00:05Z",
                "itemID": "i1",
                "source": "microphone",
                "text": "Hey quick one,",
            },
            {
                "completedAt": "2026-01-01T00:00:08Z",
                "itemID": "i2",
                "source": "microphone",
                "text": "what's the budget look like for this?",
            },
            {
                "completedAt": "2026-01-01T00:00:15Z",
                "itemID": "i3",
                "source": "systemAudio",
                "text": "Around 12k.",
            },
        ],
        "suggestions": [],
    }
    tmp = FIXTURE.parent / "_tmp_merge.json"
    tmp.write_text(json.dumps(crafted))
    try:
        loaded = load_session(tmp)
        # The two `You:` turns should be merged into one line.
        assert loaded.transcript.count("You:") == 1
        assert loaded.transcript.count("Them:") == 1
        # And the merged content has both fragments joined.
        assert "Hey quick one, what's the budget look like" in loaded.transcript
    finally:
        tmp.unlink()


def test_load_session_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_session(Path("/nonexistent/path/session.json"))


def test_load_session_rejects_empty_turns(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"startedAt": "2026-01-01T00:00:00Z", "completedTurns": []}))
    with pytest.raises(ValueError, match="no completed turns"):
        load_session(empty)


def test_load_session_preserves_raw_for_downstream_access() -> None:
    loaded = load_session(FIXTURE)
    # First turn is the consultant (microphone).
    first = loaded.raw.completed_turns[0]
    assert first.speaker() is Speaker.you
    # Suggestions round-tripped from the fixture.
    assert len(loaded.raw.suggestions) == 2
    assert loaded.raw.suggestions[0].category == "respond"
