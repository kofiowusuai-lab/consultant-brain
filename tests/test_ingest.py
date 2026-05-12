"""End-to-end ingest tests. The Claude client is mocked so the full
pipeline runs without network — session load → extract → write atoms +
call note → idempotent re-ingest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from consultant_brain.ingest import run_ingest
from consultant_brain.vault import VaultLayout, read_frontmatter


FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.json"


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _MockClient:
    """Records the prompts so we can assert redaction worked."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_call: dict[str, Any] | None = None

    def messages_create(self, **kwargs) -> _Response:
        self.last_call = kwargs
        return _Response(content=[_Block(text=self.response_text)])


VALID_RESPONSE = json.dumps(
    {
        "summary": "Reece wants retrieval fixed first, anxious about price.",
        "atoms": [
            {
                "type": "objection",
                "body": "Bot keeps grabbing wrong notes — no tagging system.",
                "confidence": 0.88,
                "tags": ["retrieval"],
            },
            {
                "type": "commitment",
                "body": "Reece will send the 3 best manual ads by Friday.",
                "confidence": 0.95,
                "tags": ["next_step"],
            },
            {
                "type": "client_fact",
                "body": "About 1000 notes in Obsidian; ads draft from those.",
                "confidence": 0.9,
                "tags": ["stack", "obsidian"],
            },
        ],
    }
)


def test_run_ingest_writes_atoms_and_call_note(tmp_path: Path) -> None:
    client = _MockClient(VALID_RESPONSE)
    result = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=tmp_path / "vault",
        redact=False,
        dry_run=False,
        client=client,
    )
    assert result.atom_count == 3
    assert result.dry_run is False
    assert result.call_note_path is not None
    assert result.call_note_path.exists()

    # Vault skeleton was created.
    layout = VaultLayout.for_root(tmp_path / "vault")
    assert (layout.root / "03_Atoms").is_dir()
    assert (layout.root / "02_Calls").is_dir()
    assert (layout.client_dir("reece") / "Reece.md").is_file()

    # Atom files exist + reference the call.
    atom_files = list(layout.atoms_dir.glob("*.md"))
    assert len(atom_files) == 3
    for atom_file in atom_files:
        fm = read_frontmatter(atom_file)
        assert fm["client"] == "[[Reece]]"
        assert fm["call"].startswith("[[2026-05-12_reece_")
        assert fm["call_type"] == "consultingCall"

    # Call note links to every atom.
    fm = read_frontmatter(result.call_note_path)
    assert fm["atom_count"] == 3
    call_text = result.call_note_path.read_text(encoding="utf-8")
    for atom_file in atom_files:
        atom_id = atom_file.stem
        assert f"[[{atom_id}]]" in call_text


def test_run_ingest_dry_run_writes_nothing(tmp_path: Path) -> None:
    client = _MockClient(VALID_RESPONSE)
    result = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=tmp_path / "vault",
        redact=False,
        dry_run=True,
        client=client,
    )
    assert result.dry_run is True
    assert result.atom_count == 3
    assert result.call_note_path is None
    # No vault dir created at all.
    assert not (tmp_path / "vault").exists()


def test_run_ingest_rejects_unknown_call_type(tmp_path: Path) -> None:
    import typer

    client = _MockClient(VALID_RESPONSE)
    with pytest.raises(typer.BadParameter, match="Unknown call type"):
        run_ingest(
            session_path=FIXTURE,
            client_name="Reece",
            call_type="not-a-call-type",
            vault_root=tmp_path / "vault",
            redact=False,
            dry_run=True,
            client=client,
        )


def test_run_ingest_with_redact_strips_client_name_from_prompt(tmp_path: Path) -> None:
    client = _MockClient(VALID_RESPONSE)
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=tmp_path / "vault",
        redact=True,
        dry_run=False,
        client=client,
    )
    # The transcript sent to Claude shouldn't contain the raw client name.
    user_msg = client.last_call["messages"][0]["content"]
    # Note: the fixture transcript doesn't actually mention "Reece" by name,
    # but the user-prompt's `Client:` line would. With redact=True, no client
    # line should appear.
    assert "Client:" not in user_msg or "[CLIENT_1]" in user_msg


def test_run_ingest_is_idempotent(tmp_path: Path) -> None:
    """Re-ingesting the same session produces the same atom files — no
    duplicates, atom IDs are stable.
    """
    vault_root = tmp_path / "vault"
    client = _MockClient(VALID_RESPONSE)
    first = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=client,
    )
    second = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    assert {p.name for p in first.atom_paths} == {p.name for p in second.atom_paths}
    assert first.call_note_path.name == second.call_note_path.name
    # Vault has the same number of atoms after both runs.
    layout = VaultLayout.for_root(vault_root)
    assert len(list(layout.atoms_dir.glob("*.md"))) == 3
    assert len(list(layout.calls_dir.glob("*.md"))) == 1


def test_run_ingest_re_ingest_overwrites_atom_body(tmp_path: Path) -> None:
    """If the extractor returns a different body on re-ingest (e.g. prompt
    tuning produced a sharper phrasing), the existing atom file is updated
    in place — not duplicated as a new file.
    """
    vault_root = tmp_path / "vault"
    first_response = json.dumps(
        {
            "summary": "v1 summary",
            "atoms": [
                {"type": "objection", "body": "first version body.", "confidence": 0.8, "tags": []},
            ],
        }
    )
    second_response = json.dumps(
        {
            "summary": "v2 summary, sharper.",
            "atoms": [
                {"type": "objection", "body": "second version body, much sharper.", "confidence": 0.85, "tags": []},
            ],
        }
    )
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(first_response),
    )
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(second_response),
    )
    layout = VaultLayout.for_root(vault_root)
    atom_files = list(layout.atoms_dir.glob("*.md"))
    assert len(atom_files) == 1
    body_text = atom_files[0].read_text(encoding="utf-8")
    assert "second version body, much sharper" in body_text
    assert "first version body" not in body_text
    # Call-note summary also updated.
    call_text = list(layout.calls_dir.glob("*.md"))[0].read_text(encoding="utf-8")
    assert "v2 summary, sharper" in call_text
    assert "v1 summary" not in call_text


def test_run_ingest_atom_ids_depend_on_session_filename(tmp_path: Path) -> None:
    """Different session files MUST produce different atom IDs even if the
    extractor returns identical atoms — otherwise re-ingesting call B would
    clobber call A's atoms.
    """
    fixture_a = FIXTURE
    fixture_b = tmp_path / "session-OTHER.json"
    fixture_b.write_text(fixture_a.read_text())

    first = run_ingest(
        session_path=fixture_a,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=tmp_path / "vault",
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    second = run_ingest(
        session_path=fixture_b,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=tmp_path / "vault",
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    first_ids = {p.stem for p in first.atom_paths}
    second_ids = {p.stem for p in second.atom_paths}
    assert first_ids.isdisjoint(second_ids), "Atom IDs collided across different sessions"
    # And both sets of atoms persist on disk.
    layout = VaultLayout.for_root(tmp_path / "vault")
    assert len(list(layout.atoms_dir.glob("*.md"))) == 6  # 3 + 3
