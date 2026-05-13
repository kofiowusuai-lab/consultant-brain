"""Tests for the brief Q&A path. Uses the same vault seeder as
test_client_brief.py + a fake LLMProvider that captures the prompts
so we can assert the right context reaches the model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from consultant_brain.briefs import (
    ChatTurn,
    ClientBriefAnswer,
    ask_about_client,
    generate_client_brief,
)
from consultant_brain.llm.provider import ChatRequest, ChatResponse
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


_BRIEF_PAYLOAD = json.dumps(
    {
        "summary": "Reece confirmed Q3 deadline; ad-bot scope agreed.",
        "key_facts": ["Team of 4 marketers."],
        "open_commitments": [
            {"by_whom": "you", "what": "Send vault sample Friday.", "since_call": "2026-05-12_reece_consultingCall"}
        ],
        "open_objections": [],
        "recent_moves": ["May 7 coffee."],
        "next_steps": ["Draft scope doc."],
        "meeting_prep_checklist": ["Bring Q3 calendar."],
    }
)


class _FakeProvider:
    name = "fake"

    def __init__(self, answer_text: str = "Reece's deadline is Q3 launch.") -> None:
        self._brief_payload = _BRIEF_PAYLOAD
        self._answer_payload = answer_text
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        # If the system prompt is the brief generator's, return the
        # JSON brief; otherwise it's the Q&A path, return free text.
        if "atom extractor" in request.system or "preparing a brief" in request.system:
            return ChatResponse(text=self._brief_payload, model_used=request.model)
        return ChatResponse(text=self._answer_payload, model_used=request.model)


def _seed_vault(tmp_path: Path) -> Path:
    """Mirror test_client_brief's seeder — one objection + one commitment
    atom plus one call note for Reece."""
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
    call_meta = {
        "id": "2026-05-12_reece_consultingCall",
        "client": "[[Reece]]",
        "call_type": "consultingCall",
        "date": "2026-05-12",
        "duration_minutes": 47,
        "source_session": "session-2026-05-12.json",
        "extractor_model": "claude-opus-4-7",
        "extractor_version": 1,
        "atom_count": 1,
        "created_at": "2026-05-12T19:42:10Z",
        "summary": "First consulting call. Agreed on Q3 launch window.",
    }
    (layout.calls_dir / "2026-05-12_reece_consultingCall.md").write_text(
        "---\n"
        + yaml.safe_dump(call_meta, sort_keys=False).rstrip()
        + "\n---\n\n## Summary\nFirst consulting call. Agreed on Q3 launch window.\n",
        encoding="utf-8",
    )
    return vault


def test_ask_about_client_returns_answer_with_provenance(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    provider = _FakeProvider("Reece's deadline is Q3 launch.")

    answer = ask_about_client(
        client_name="Reece",
        question="What is Reece's deadline?",
        vault_root=vault,
        provider=provider,
    )
    assert isinstance(answer, ClientBriefAnswer)
    assert "Q3" in answer.answer
    assert answer.atoms_consulted == 1
    assert answer.calls_consulted == 1
    assert answer.model_used  # non-empty


def test_ask_prompt_includes_cached_brief_when_present(tmp_path: Path) -> None:
    """When a brief.json exists for this client, the Q&A system prompt
    embeds it so the model doesn't re-derive the summary."""
    vault = _seed_vault(tmp_path)
    provider = _FakeProvider()
    # Generate + cache a brief first.
    generate_client_brief(client_name="Reece", vault_root=vault, provider=provider)
    # Now ask a question.
    _ = ask_about_client(
        client_name="Reece",
        question="What did Reece commit to?",
        vault_root=vault,
        provider=provider,
    )
    qa_request = provider.requests[-1]
    assert "## Cached brief" in qa_request.system
    assert "Q3 deadline" in qa_request.system or "Q3 launch" in qa_request.system
    assert "Reece" in qa_request.system


def test_ask_threads_history_into_user_message(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    provider = _FakeProvider("Yes — May 7 coffee.")
    history = [
        ChatTurn(role="user", content="What's his timeline?"),
        ChatTurn(role="assistant", content="Q3 launch."),
        ChatTurn(role="user", content="Did we have a coffee with him?"),
        ChatTurn(role="assistant", content="Yes, May 7."),
    ]
    _ = ask_about_client(
        client_name="Reece",
        question="What did we agree to bring next time?",
        history=history,
        vault_root=vault,
        provider=provider,
    )
    rendered = provider.requests[-1].user
    assert "Previous Q&A:" in rendered
    assert "Q3 launch." in rendered
    assert "What did we agree to bring next time?" in rendered


def test_ask_endpoint_validates_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _seed_vault(tmp_path)
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: _FakeProvider(),
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)

    bad_no_client = client.post("/client_brief/ask", json={"question": "x"})
    assert bad_no_client.status_code == 400
    assert "client" in bad_no_client.json()["detail"].lower()

    bad_no_question = client.post("/client_brief/ask", json={"client_name": "Reece"})
    assert bad_no_question.status_code == 400


def test_ask_endpoint_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _seed_vault(tmp_path)
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: _FakeProvider("Reece's deadline is Q3 launch."),
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)
    resp = client.post(
        "/client_brief/ask",
        json={
            "client_name": "Reece",
            "question": "What's the deadline?",
            "history": [
                {"role": "user", "content": "What did we agree on?"},
                {"role": "assistant", "content": "Q3 launch."},
            ],
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["answer"].startswith("Reece's deadline")
    assert payload["atoms_consulted"] >= 0
    assert payload["calls_consulted"] >= 0
    assert payload["model_used"]


def test_ask_endpoint_drops_malformed_history_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad history entries (wrong role, empty content) are filtered out
    — we don't 400 on them; the endpoint stays forgiving."""
    vault = _seed_vault(tmp_path)
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: _FakeProvider(),
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)
    resp = client.post(
        "/client_brief/ask",
        json={
            "client_name": "Reece",
            "question": "Anything?",
            "history": [
                {"role": "user", "content": "valid"},
                {"role": "junk", "content": "dropped"},
                {"role": "assistant", "content": ""},
                "not even a dict",
            ],
        },
    )
    assert resp.status_code == 200
