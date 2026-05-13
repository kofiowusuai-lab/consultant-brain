"""Tests for the unified chat endpoint — questions go through the
existing answer path, notes get ingested as atoms and trigger a
brief regeneration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from consultant_brain.briefs import (
    ChatProcessResult,
    ChatTurn,
    process_chat,
)
from consultant_brain.llm.provider import ChatRequest, ChatResponse
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


_BRIEF_PAYLOAD = json.dumps(
    {
        "summary": "Reece is in active scoping. Q3 deadline confirmed.",
        "key_facts": ["Team of 4 marketers."],
        "open_commitments": [
            {"by_whom": "them", "what": "Send scoping answers by EOD 2026-05-12.", "since_call": None}
        ],
        "open_objections": [],
        "recent_moves": ["May 7 coffee."],
        "next_steps": ["Confirm scoping answers received."],
        "meeting_prep_checklist": ["Have the 7 questions ready in case he hasn't sent them."],
    }
)

_NOTE_RESPONSE = json.dumps(
    {
        "intent": "note",
        "answer": "Got it — wrote a note. Reece's scoping-answers commitment is now resolved; meeting-prep can drop that line on the next refresh.",
        "ingested_facts": [
            {
                "type": "client_fact",
                "body": "Reece sent his 7 scoping answers on 2026-05-12.",
                "tags": ["chat_note", "scoping"],
            }
        ],
    }
)

_QUESTION_RESPONSE = json.dumps(
    {
        "intent": "question",
        "answer": "His deadline is the 7:20 PM Google Meet tonight (2026-05-13).",
    }
)


class _FakeProvider:
    """Returns the right canned response depending on which system
    prompt the caller used (chat vs. brief generator)."""

    name = "fake"

    def __init__(self, chat_payload: str) -> None:
        self._chat_payload = chat_payload
        self._brief_payload = _BRIEF_PAYLOAD
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if "preparing a brief" in request.system:
            return ChatResponse(text=self._brief_payload, model_used=request.model)
        return ChatResponse(text=self._chat_payload, model_used=request.model)


def _seed_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    layout = VaultLayout.for_root(vault)
    ensure_vault_skeleton(layout)
    atom = {
        "id": "01HX0000000000000000000001",
        "type": "commitment",
        "client": "[[Reece]]",
        "call": "[[2026-05-12_reece_consultingCall]]",
        "call_type": "consultingCall",
        "tags": ["timeline"],
        "confidence": 0.9,
        "evidence_count": 1,
        "last_seen": "2026-05-12",
        "created_at": "2026-05-12T19:42:10Z",
        "status": "active",
        "embedding_id": "01HX0000000000000000000001",
    }
    (layout.atoms_dir / "01HX0000000000000000000001.md").write_text(
        "---\n"
        + yaml.safe_dump(atom, sort_keys=False).rstrip()
        + "\n---\n\nReece committed to Q3 launch.",
        encoding="utf-8",
    )
    return vault


def test_process_chat_question_path(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    provider = _FakeProvider(_QUESTION_RESPONSE)

    result = process_chat(
        client_name="Reece",
        user_input="What's the deadline?",
        vault_root=vault,
        provider=provider,
    )
    assert result.intent == "question"
    assert "7:20 PM" in result.answer
    assert result.ingested_atoms == 0
    assert result.updated_brief is None


def test_process_chat_note_path_writes_atom_and_regenerates_brief(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    provider = _FakeProvider(_NOTE_RESPONSE)

    result = process_chat(
        client_name="Reece",
        user_input="He's sent the scoping answers.",
        vault_root=vault,
        provider=provider,
    )
    assert result.intent == "note"
    assert "scoping" in result.answer.lower()
    assert result.ingested_atoms == 1
    assert result.updated_brief is not None
    # Brief regeneration ran on the same fake provider — the new
    # cached brief should be on disk.
    cache_path = vault / "01_Clients" / "reece" / "brief.json"
    assert cache_path.exists()
    # The chat-note atom landed in 03_Atoms with the user's content.
    atom_files = list((vault / "03_Atoms").glob("*.md"))
    note_atom = [
        path for path in atom_files
        if "Reece sent his 7 scoping" in path.read_text(encoding="utf-8")
    ]
    assert len(note_atom) == 1


def test_process_chat_falls_back_to_question_on_malformed_json(tmp_path: Path) -> None:
    """If the LLM goes off-script (returns prose, not JSON), we treat
    it as a question answer rather than guessing intent."""
    vault = _seed_vault(tmp_path)
    provider = _FakeProvider("just some prose, not JSON at all")
    result = process_chat(
        client_name="Reece",
        user_input="anything",
        vault_root=vault,
        provider=provider,
    )
    assert result.intent == "question"
    assert result.ingested_atoms == 0


def test_endpoint_note_returns_updated_brief_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = _seed_vault(tmp_path)
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: _FakeProvider(_NOTE_RESPONSE),
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)
    resp = client.post(
        "/client_brief/ask",
        json={
            "client_name": "Reece",
            "question": "He's sent the scoping answers.",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["intent"] == "note"
    assert payload["ingested_atoms"] == 1
    assert payload["updated_brief"] is not None
    assert payload["updated_brief"]["client_slug"] == "reece"


def test_endpoint_question_does_not_regenerate_brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = _seed_vault(tmp_path)
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: _FakeProvider(_QUESTION_RESPONSE),
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)
    resp = client.post(
        "/client_brief/ask",
        json={"client_name": "Reece", "question": "What's the deadline?"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["intent"] == "question"
    assert payload["ingested_atoms"] == 0
    assert payload["updated_brief"] is None
