"""Loader + score-CLI tests. End-to-end: ingest a fixture → score it.

Uses a mocked Claude client for the ingest path so this test runs
offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from consultant_brain.cli import app
from consultant_brain.ingest import run_ingest
from consultant_brain.scoring.loader import (
    CallNotFoundError,
    _extract_summary_section,
    _extract_transcript_turns,
    load_call_for_scoring,
)
from consultant_brain.schemas import CallType, Speaker


FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.json"


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _MockClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def messages_create(self, **kwargs) -> _Response:
        return _Response(content=[_Block(text=self.text)])


VALID_RESPONSE = json.dumps(
    {
        "summary": "Reece wants retrieval fixed first. Strong commitment on a Friday deliverable.",
        "atoms": [
            {"type": "objection", "body": "Bot keeps grabbing wrong notes.", "confidence": 0.88, "tags": ["retrieval"]},
            {"type": "commitment", "body": "Reece will send 3 manual ads by Friday.", "confidence": 0.95, "tags": ["next_step"]},
            {"type": "client_fact", "body": "1000 notes in Obsidian vault.", "confidence": 0.9, "tags": ["stack"]},
        ],
    }
)


# ────────────────────────────────────────────────────────────────────────────
# Parser helpers
# ────────────────────────────────────────────────────────────────────────────


def test_extract_summary_section_returns_summary_body() -> None:
    body = "# title\n\n## Summary\nReece wants retrieval fixed first.\n\n## Atoms\n- [[X]]\n"
    assert _extract_summary_section(body) == "Reece wants retrieval fixed first."


def test_extract_transcript_turns_parses_you_them_inside_details() -> None:
    body = """## Transcript
<details>
<summary>Full transcript</summary>

You: Walk me through your stack.

Them: We use Obsidian.

</details>
"""
    turns = _extract_transcript_turns(body)
    assert len(turns) == 2
    assert turns[0].speaker is Speaker.you
    assert turns[1].speaker is Speaker.them
    assert "Walk me through" in turns[0].text


# ────────────────────────────────────────────────────────────────────────────
# load_call_for_scoring round-trip
# ────────────────────────────────────────────────────────────────────────────


def test_load_after_ingest_hydrates_inputs(tmp_path: Path) -> None:
    """Ingest a fixture, then load the resulting call back for scoring —
    atoms + turns + call_type all come through correctly."""
    vault_root = tmp_path / "vault"
    result = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    inputs = load_call_for_scoring(
        vault_root=vault_root,
        call_id=result.call_note_path.stem,  # "2026-05-12_reece_consultingCall"
        primary_win="ship the ad-bot retrieval fix",
    )
    assert inputs.call_type is CallType.consulting_call
    assert inputs.client == "Reece"
    assert inputs.primary_win == "ship the ad-bot retrieval fix"
    assert "wants retrieval fixed" in inputs.summary
    # All three atoms link back to this call.
    assert len(inputs.atoms) == 3
    # Transcript parsing recovers the original fixture's 10 turns.
    assert len(inputs.turns) == 10
    speakers = {turn.speaker for turn in inputs.turns}
    assert speakers == {Speaker.you, Speaker.them}


def test_load_missing_call_raises_call_not_found(tmp_path: Path) -> None:
    layout = tmp_path / "vault"
    from consultant_brain.vault import VaultLayout, ensure_vault_skeleton
    ensure_vault_skeleton(VaultLayout.for_root(layout))
    with pytest.raises(CallNotFoundError):
        load_call_for_scoring(vault_root=layout, call_id="never_existed")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


runner = CliRunner()


def test_score_cli_prints_summary_line(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    result = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    cli_result = runner.invoke(app, ["score", result.call_note_path.stem, "--vault", str(vault_root)])
    assert cli_result.exit_code == 0, cli_result.output
    # Sample fixture: 1 objection (Bot keeps grabbing wrong notes — no
    # commitment about retrieval), 1 commitment with a Friday phrase
    # (next_step_booked = 1), 1 client_fact. Expect somewhere in 50-80 range.
    assert "/100" in cli_result.output
    assert "consultingCall" in cli_result.output


def test_score_cli_explain_prints_per_feature_breakdown(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    result = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    cli_result = runner.invoke(
        app,
        ["score", result.call_note_path.stem, "--vault", str(vault_root), "--explain"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "next_step_booked" in cli_result.output
    assert "commitments_made" in cli_result.output
    assert "raw score" in cli_result.output
    assert "clamped" in cli_result.output


def test_score_cli_unknown_call_exits_2() -> None:
    cli_result = runner.invoke(app, ["score", "no-such-call", "--vault", "/tmp/never-a-vault-here"])
    assert cli_result.exit_code == 2
