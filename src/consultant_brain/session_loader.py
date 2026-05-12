"""Load a Swift-app session JSON into a normalized struct the rest of the
pipeline can consume without caring about Swift-side field names or
audio-routing details.

The output `LoadedSession` carries:
  - A formatted transcript ready to drop into a Claude prompt or a call note.
  - Lifecycle metadata (start time, duration, source filename) needed by the
    call-note schema.
  - The raw `SessionJSON` so callers that need richer access (suggestions,
    item IDs, original `source` values) don't have to re-read the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from consultant_brain.schemas import SessionJSON, Speaker


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """Everything the extractor + vault writer need from one session JSON."""

    raw: SessionJSON
    source_filename: str
    started_at: datetime
    ended_at: datetime
    duration_minutes: int
    transcript: str  # "You: ...\nThem: ..." formatted

    @property
    def turn_count(self) -> int:
        return len(self.raw.completed_turns)


def load_session(path: Path) -> LoadedSession:
    """Read + validate one Sessions/*.json. Returns a LoadedSession or raises.

    Raises:
        FileNotFoundError: path doesn't exist.
        json.JSONDecodeError: file isn't valid JSON.
        pydantic.ValidationError: JSON doesn't match `SessionJSON` shape.
        ValueError: session has no completed turns (nothing to ingest).
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Session file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    raw = SessionJSON.model_validate(data)

    if not raw.completed_turns:
        raise ValueError(
            f"Session {path.name} has no completed turns — nothing to ingest. "
            "Was the call hung up before any speech was transcribed?"
        )

    started = raw.started_at
    ended = max(turn.completed_at for turn in raw.completed_turns)
    duration_minutes = max(0, round((ended - started).total_seconds() / 60))

    return LoadedSession(
        raw=raw,
        source_filename=path.name,
        started_at=started,
        ended_at=ended,
        duration_minutes=duration_minutes,
        transcript=_format_transcript(raw),
    )


def _format_transcript(session: SessionJSON) -> str:
    """Turn the speaker-tagged turns into the `You:` / `Them:` block used by
    both the extractor prompt and the call note's body.

    Consecutive turns by the same speaker are merged into one paragraph so
    the LLM doesn't see the same speaker label five times in a row from VAD
    chunking — that always degrades extraction quality.
    """
    if not session.completed_turns:
        return "(no turns)"

    lines: list[str] = []
    current_speaker: Speaker | None = None
    current_chunks: list[str] = []

    def flush() -> None:
        if current_speaker is None or not current_chunks:
            return
        label = "You" if current_speaker is Speaker.you else "Them"
        text = " ".join(chunk.strip() for chunk in current_chunks if chunk.strip())
        if text:
            lines.append(f"{label}: {text}")

    for turn in session.completed_turns:
        speaker = turn.speaker()
        if speaker is current_speaker:
            current_chunks.append(turn.text)
        else:
            flush()
            current_speaker = speaker
            current_chunks = [turn.text]
    flush()

    return "\n\n".join(lines)
