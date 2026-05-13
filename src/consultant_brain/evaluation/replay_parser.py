"""Phase 12 — replay-harness call-note parser.

Inverts `vault.py:write_call_note`'s `<details>` transcript block back
into an ordered list of (speaker, text) turns the replay harness can
feed through `retrieve()` one window at a time.

The on-disk transcript format is the same `You:/Them:` shape the live
service builds via `live_state.CallState.transcript_window()`. Turns
are separated by blank lines; each turn is a single label line of the
form `You: <text>` or `Them: <text>`, possibly wrapped over multiple
lines (continuation lines have no `You:`/`Them:` prefix).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import frontmatter

from consultant_brain.schemas import Speaker


class ReplayParserError(Exception):
    """Raised when a call note can't be parsed back into turns."""


@dataclass(frozen=True, slots=True)
class ReplayTurn:
    """One reconstructed turn. `speaker` is the canonical Speaker enum
    so downstream code that already accepts CallTurn-shaped values can
    swap in a ReplayTurn without translation."""

    speaker: Speaker
    text: str


def parse_call_note(path: Path) -> tuple[dict, list[ReplayTurn]]:
    """Read a call note markdown file at `path`. Returns:
      - the parsed YAML frontmatter as a plain dict
      - the ordered list of turns extracted from the <details>...</details>
        transcript section

    Raises `ReplayParserError` when the file is missing, the
    frontmatter won't parse, or the transcript block is missing /
    empty.
    """
    if not path.exists():
        raise ReplayParserError(f"Call note not found at {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            post = frontmatter.load(f)
    except Exception as exc:  # noqa: BLE001
        raise ReplayParserError(f"Failed to parse frontmatter at {path}: {exc}") from exc

    meta = dict(post.metadata)
    body = post.content or ""
    transcript_block = _extract_details_block(body)
    if not transcript_block.strip():
        raise ReplayParserError(
            f"Call note at {path} has no transcript block — replay needs at "
            "least one turn to walk through."
        )
    turns = list(_iter_turns(transcript_block))
    if not turns:
        raise ReplayParserError(
            f"Call note at {path} has a transcript block but no You:/Them: "
            "labeled turns to replay."
        )
    return meta, turns


def transcript_window_from_turns(turns: Iterable[ReplayTurn]) -> str:
    """Mirror `CallState.transcript_window()` so replay produces the
    exact same input string `retrieve()` saw at live-call time.

    Two blank lines between turns; You:/Them: prefix; the text is
    stripped at the edges so reconstructed windows are byte-identical
    to the live writer's output.
    """
    materialized = list(turns)
    if not materialized:
        return ""
    lines: list[str] = []
    for turn in materialized:
        label = "You" if turn.speaker is Speaker.you else "Them"
        lines.append(f"{label}: {turn.text.strip()}")
    return "\n\n".join(lines)


def _extract_details_block(body: str) -> str:
    """Pull the contents of `<details>...</details>`. The writer wraps
    the transcript in a section starting with `<summary>Full transcript</summary>`,
    but we don't depend on that — any <details> block works.
    """
    start = body.find("<details>")
    end = body.find("</details>", start) if start >= 0 else -1
    if start < 0 or end < 0:
        return ""
    inner = body[start + len("<details>") : end]
    # Strip the <summary>Full transcript</summary> line, if present.
    summary_end = inner.find("</summary>")
    if summary_end >= 0:
        inner = inner[summary_end + len("</summary>") :]
    return inner.strip()


def _iter_turns(block: str) -> Iterable[ReplayTurn]:
    """Walk the transcript block top to bottom, emitting one
    `ReplayTurn` per `You:`/`Them:` labeled paragraph.

    The block is separated into turn-paragraphs by blank lines. Each
    paragraph starts with a label. We tolerate paragraphs that wrap
    onto continuation lines without a label by appending them to the
    current turn's text.
    """
    current_speaker: Speaker | None = None
    current_chunks: list[str] = []

    def flush() -> ReplayTurn | None:
        nonlocal current_chunks, current_speaker
        if current_speaker is None or not current_chunks:
            return None
        text = " ".join(s.strip() for s in current_chunks).strip()
        if not text:
            current_chunks = []
            return None
        out = ReplayTurn(speaker=current_speaker, text=text)
        current_chunks = []
        return out

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if (turn := flush()) is not None:
                yield turn
            current_speaker = None
            continue
        label, rest = _split_label(line)
        if label is not None:
            if (turn := flush()) is not None:
                yield turn
            current_speaker = label
            current_chunks = [rest]
        elif current_speaker is not None:
            # Continuation line under the current speaker.
            current_chunks.append(line)
        # else: line before any label — ignore (probably noise).
    if (turn := flush()) is not None:
        yield turn


def _split_label(line: str) -> tuple[Speaker | None, str]:
    """If the line is `You: ...` or `Them: ...`, return (speaker, rest).
    Otherwise return (None, line).
    """
    for prefix, speaker in (("You:", Speaker.you), ("Them:", Speaker.them)):
        if line.startswith(prefix):
            return speaker, line[len(prefix):].lstrip()
    return None, line
