"""Phase 12 — replay harness tests.

Covers:
  - replay_parser inverts the <details> transcript block back into
    Speaker-labeled ReplayTurns
  - transcript_window_from_turns produces byte-identical output to
    the live CallState.transcript_window() format
  - replay_call walks turns and returns a per-turn diff
  - missing transcript / wrong call id surface ReplayError
  - graceful degradation when historical suggestion_log is empty
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.evaluation.replay import (
    ReplayError,
    render_report_markdown,
    replay_call,
)
from consultant_brain.evaluation.replay_parser import (
    ReplayParserError,
    parse_call_note,
    transcript_window_from_turns,
)
from consultant_brain.evaluation.suggestion_log import log_emit
from consultant_brain.schemas import CallNote, CallType, Speaker
from consultant_brain.vault import (
    VaultLayout,
    ensure_vault_skeleton,
    write_call_note,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


def _seed_call(vault_root: Path, *, call_id: str, transcript: str) -> Path:
    """Write a CallNote to the vault for the replay parser to load.

    `call_id` is normalized to the brain's required pattern
    `YYYY-MM-DD_<slug>_<callType>` so write_call_note's validation
    doesn't reject the fixture.
    """
    layout = VaultLayout.for_root(vault_root)
    normalized_id = f"2026-05-12_replay_{call_id.lower()}_consultingCall"
    note = CallNote(
        id=normalized_id,
        client="Reece",
        call_type=CallType.consulting_call,
        date=date(2026, 5, 12),
        duration_minutes=12,
        source_session=f"live::{normalized_id}",
        extractor_model="claude-opus-test",
        extractor_version=1,
        atom_count=0,
        created_at=datetime(2026, 5, 12, 19, 42, 10, tzinfo=timezone.utc),
        summary="A test call.",
        atom_ids=[],
        transcript=transcript,
    )
    write_call_note(layout, note)
    return layout.calls_dir / f"{normalized_id}.md"


# ────────────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────────────


def test_parse_call_note_recovers_turn_sequence(vault_root: Path) -> None:
    transcript = (
        "You: Hey, glad we connected.\n\n"
        "Them: Sure — what did you want to walk through?\n\n"
        "You: The pricing tier you proposed.\n\n"
        "Them: Specifically the team plan?"
    )
    path = _seed_call(vault_root, call_id="REPLAYTEST1", transcript=transcript)

    meta, turns = parse_call_note(path)
    assert meta["id"].endswith("_replaytest1_consultingCall")
    assert [t.speaker for t in turns] == [
        Speaker.you,
        Speaker.them,
        Speaker.you,
        Speaker.them,
    ]
    assert turns[0].text == "Hey, glad we connected."
    assert turns[-1].text == "Specifically the team plan?"


def test_parse_call_note_handles_continuation_lines(vault_root: Path) -> None:
    """Turns wrapped across newlines without a label should attach to
    the prior turn's text."""
    transcript = (
        "You: This is a longer thought\n"
        "that wraps to a second line.\n\n"
        "Them: Got it."
    )
    path = _seed_call(vault_root, call_id="REPLAYTEST2", transcript=transcript)
    _, turns = parse_call_note(path)
    assert len(turns) == 2
    assert turns[0].text == "This is a longer thought that wraps to a second line."


def test_parse_call_note_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReplayParserError):
        parse_call_note(tmp_path / "does-not-exist.md")


def test_parse_call_note_empty_transcript(vault_root: Path) -> None:
    """Call notes with a real-but-empty transcript get rejected so
    replay doesn't waste a vault load + retrieve() call producing
    nothing."""
    path = _seed_call(vault_root, call_id="REPLAYEMPTY", transcript="")
    with pytest.raises(ReplayParserError):
        parse_call_note(path)


def test_transcript_window_round_trip(vault_root: Path) -> None:
    """The reconstructed window after walking N turns should be
    byte-identical to what CallState.transcript_window() produced on
    the live side."""
    transcript = "You: alpha\n\nThem: beta\n\nYou: gamma"
    path = _seed_call(vault_root, call_id="REPLAYRT", transcript=transcript)
    _, turns = parse_call_note(path)
    rebuilt = transcript_window_from_turns(turns)
    assert rebuilt == transcript


# ────────────────────────────────────────────────────────────────────────────
# replay_call
# ────────────────────────────────────────────────────────────────────────────


def test_replay_call_walks_every_turn(vault_root: Path) -> None:
    transcript = (
        "You: First turn.\n\n"
        "Them: Second turn.\n\n"
        "You: Third turn."
    )
    _seed_call(vault_root, call_id="REPLAYWALK", transcript=transcript)

    report = replay_call(
        vault_root=vault_root,
        call_id="2026-05-12_replay_replaywalk_consultingCall",
    )
    assert report.total_turns == 3
    assert [t.speaker for t in report.turns] == ["you", "them", "you"]
    # Empty vault → no atoms to emit. The diff is fine — only_now /
    # only_then both empty.
    for turn in report.turns:
        assert turn.would_emit == ()
        assert turn.did_emit == ()


def test_replay_call_rejects_both_inputs(vault_root: Path) -> None:
    """Replay needs exactly one of --call / --call-file. Test both
    failure paths."""
    with pytest.raises(ReplayError):
        replay_call(vault_root=vault_root)
    with pytest.raises(ReplayError):
        replay_call(
            vault_root=vault_root,
            call_id="X",
            call_file=Path("/tmp/y.md"),
        )


def test_replay_call_call_file_loads_outside_vault(tmp_path: Path) -> None:
    """An ad-hoc call note file should replay successfully even when
    it lives outside the vault layout."""
    ad_hoc = tmp_path / "ad-hoc.md"
    ad_hoc.write_text(
        "---\n"
        "id: ADHOCREPLAY\n"
        "client: '[[Reece]]'\n"
        "call_type: consultingCall\n"
        "date: 2026-05-12\n"
        "duration_minutes: 5\n"
        "source_session: live::ADHOCREPLAY\n"
        "extractor_model: claude-opus-test\n"
        "extractor_version: test-1\n"
        "atom_count: 0\n"
        "created_at: 2026-05-12T19:42:10Z\n"
        "---\n\n"
        "# Test\n\n"
        "## Transcript\n"
        "<details>\n"
        "<summary>Full transcript</summary>\n\n"
        "You: hello\n\n"
        "Them: world\n\n"
        "</details>\n",
        encoding="utf-8",
    )
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    report = replay_call(vault_root=layout.root, call_file=ad_hoc)
    assert report.total_turns == 2
    assert report.call_id == "ADHOCREPLAY"


def test_replay_call_includes_historical_emits(vault_root: Path) -> None:
    """When suggestion_log has historical emits for the call_id, they
    surface in the per-turn did_emit column."""
    _seed_call(
        vault_root,
        call_id="REPLAYEMITS",
        transcript="You: First turn.\n\nThem: Second turn.",
    )
    seed_id = "2026-05-12_replay_replayemits_consultingCall"
    # Seed two historical emits for this call.
    log_emit(vault_root=vault_root, call_id=seed_id, atom_id="ATOM_A", layer="hot", score=0.82)
    log_emit(vault_root=vault_root, call_id=seed_id, atom_id="ATOM_B", layer="warm", score=0.61)

    report = replay_call(vault_root=vault_root, call_id=seed_id)
    assert report.has_historical_emits
    assert report.historical_event_count == 2
    # First turn picks the first emit; second turn picks the next.
    assert report.turns[0].did_emit == ("ATOM_A",)
    assert report.turns[1].did_emit == ("ATOM_B",)
    # Empty vault → would_emit stays empty, so all emits land in only_then.
    assert report.turns[0].only_then == ("ATOM_A",)


def test_render_report_markdown_renders_table(vault_root: Path) -> None:
    _seed_call(vault_root, call_id="REPLAYRENDER", transcript="You: hi\n\nThem: hey")
    seed_id = "2026-05-12_replay_replayrender_consultingCall"
    report = replay_call(vault_root=vault_root, call_id=seed_id)
    rendered = render_report_markdown(report)
    assert f"# Replay · {seed_id}" in rendered
    assert "| # | Speaker |" in rendered
    assert "| 0 | you |" in rendered
    assert "| 1 | them |" in rendered
