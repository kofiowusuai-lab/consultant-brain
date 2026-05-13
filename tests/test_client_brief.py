"""Unit + integration tests for the per-client AI brief generator.

Mocks the LLM through a fake ChatRequest/ChatResponse pair so the
tests stay offline. Verifies:
  - generate_client_brief loads atoms + call summaries + context
    dumps for the right client, builds a prompt that includes all of
    them, parses the JSON response correctly, and caches to disk.
  - The /client_brief endpoint returns a cached payload on default
    open and refreshes when ?refresh=true.
  - load_cached_brief gracefully returns None on missing/malformed
    cache.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from consultant_brain.briefs import (
    ClientBrief,
    generate_client_brief,
    load_cached_brief,
)
from consultant_brain.llm.provider import ChatRequest, ChatResponse
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


_OK_PAYLOAD = json.dumps(
    {
        "summary": "Reece is sold on the outcome; budget anchored before scope. Q3 deadline.",
        "key_facts": [
            "Team of 4 marketers, no engineering.",
            "Uses Notion for tickets, runs 4 cold emails per prospect per week.",
        ],
        "open_commitments": [
            {
                "by_whom": "you",
                "what": "Send vault sample by Friday.",
                "since_call": "2026-05-12_reece_consultingCall",
            },
            {
                "by_whom": "them",
                "what": "Confirm Q3 launch date by Monday.",
                "since_call": None,
            },
        ],
        "open_objections": [
            {
                "objection": "Budget came up before scope.",
                "context": "Anchor risk — number sets a ceiling on value framing.",
            }
        ],
        "recent_moves": [
            "Off-call coffee on May 7 confirmed Q3 deadline.",
            "First consulting call May 12 — established ad-bot scope.",
        ],
        "next_steps": [
            "Draft 3-line ad-bot scope doc with success metric.",
            "Send vault sample with two before/after examples.",
        ],
        "meeting_prep_checklist": [
            "Have the May 7 coffee notes open.",
            "Bring the Q3 calendar — confirm exact launch window.",
            "Lead with scope discussion, not pricing.",
        ],
    }
)


class _FakeProvider:
    """Tiny LLMProvider double — returns canned JSON every call."""

    name = "fake"

    def __init__(self, payload: str = _OK_PAYLOAD) -> None:
        self._payload = payload
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(text=self._payload, model_used=request.model)


def _seed_reece_atoms(tmp_path: Path) -> Path:
    """Plant a vault with a couple of Reece atoms + one call note +
    one context dump so the brief generator has real data to chew on."""
    vault = tmp_path / "vault"
    layout = VaultLayout.for_root(vault)
    ensure_vault_skeleton(layout)

    # Atom #1 — objection
    atom1 = {
        "id": "01HX0000000000000000000001",
        "type": "objection",
        "client": "[[Reece]]",
        "call": "[[2026-05-12_reece_consultingCall]]",
        "call_type": "consultingCall",
        "tags": ["budget"],
        "confidence": 0.85,
        "evidence_count": 1,
        "last_seen": "2026-05-12",
        "created_at": "2026-05-12T19:42:10Z",
        "status": "active",
        "embedding_id": "01HX0000000000000000000001",
    }
    (layout.atoms_dir / "01HX0000000000000000000001.md").write_text(
        "---\n"
        + yaml.safe_dump(atom1, sort_keys=False).rstrip()
        + "\n---\n\nBudget came up before scope — anchor risk.",
        encoding="utf-8",
    )

    # Atom #2 — commitment
    atom2 = {
        "id": "01HX0000000000000000000002",
        "type": "commitment",
        "client": "[[Reece]]",
        "call": "[[2026-05-12_reece_consultingCall]]",
        "call_type": "consultingCall",
        "tags": ["next_step"],
        "confidence": 0.9,
        "evidence_count": 1,
        "last_seen": "2026-05-12",
        "created_at": "2026-05-12T19:42:10Z",
        "status": "active",
        "embedding_id": "01HX0000000000000000000002",
    }
    (layout.atoms_dir / "01HX0000000000000000000002.md").write_text(
        "---\n"
        + yaml.safe_dump(atom2, sort_keys=False).rstrip()
        + "\n---\n\nWill send vault sample by Friday.",
        encoding="utf-8",
    )

    # One call note
    call_frontmatter = {
        "id": "2026-05-12_reece_consultingCall",
        "client": "[[Reece]]",
        "call_type": "consultingCall",
        "date": "2026-05-12",
        "duration_minutes": 47,
        "source_session": "session-2026-05-12.json",
        "extractor_model": "claude-opus-4-7",
        "extractor_version": 1,
        "atom_count": 2,
        "created_at": "2026-05-12T19:42:10Z",
        "summary": "First consulting call. Reece anchored budget early; agreed on Q3 launch window.",
    }
    (layout.calls_dir / "2026-05-12_reece_consultingCall.md").write_text(
        "---\n"
        + yaml.safe_dump(call_frontmatter, sort_keys=False).rstrip()
        + "\n---\n\n## Summary\nFirst consulting call. Reece anchored budget early; agreed on Q3 launch window.\n",
        encoding="utf-8",
    )

    # One context dump
    layout.context_dumps_dir.mkdir(parents=True, exist_ok=True)
    (layout.context_dumps_dir / "reece").mkdir(parents=True, exist_ok=True)
    dump_frontmatter = {
        "id": "01HX_dump_001",
        "kind": "context_dump",
        "client": "[[Reece]]",
        "observed_at": "2026-05-07",
        "uploaded_at": "2026-05-12T19:42:10Z",
        "source_filename": "coffee_notes.txt",
        "source_kind_label": "text",
        "atom_count": 0,
    }
    (layout.context_dumps_dir / "reece" / "2026-05-07_01HX_dump_001.md").write_text(
        "---\n"
        + yaml.safe_dump(dump_frontmatter, sort_keys=False).rstrip()
        + "\n---\n\n## Summary\nCoffee meeting May 7. Reece confirmed Q3 deadline + revealed team size of 4.\n",
        encoding="utf-8",
    )
    return vault


def test_generate_client_brief_produces_structured_payload(tmp_path: Path) -> None:
    vault = _seed_reece_atoms(tmp_path)
    provider = _FakeProvider()
    brief = generate_client_brief(
        client_name="Reece",
        vault_root=vault,
        provider=provider,
    )

    assert brief.client_name == "Reece"
    assert brief.client_slug == "reece"
    assert "Q3" in brief.summary
    assert len(brief.key_facts) == 2
    assert len(brief.open_commitments) == 2
    assert brief.open_commitments[0].by_whom == "you"
    assert "Friday" in brief.open_commitments[0].what
    assert len(brief.open_objections) == 1
    assert brief.open_objections[0].objection.startswith("Budget came up")
    assert len(brief.next_steps) == 2
    assert len(brief.meeting_prep_checklist) == 3
    assert brief.meta.atoms_considered == 2
    assert brief.meta.calls_considered == 1
    assert brief.meta.context_dumps_considered == 1

    # Prompt contains every input the model needs.
    prompt = provider.requests[0].user
    assert "Client: Reece" in prompt
    assert "Today is" in prompt
    assert "Budget came up before scope" in prompt
    assert "vault sample" in prompt.lower()
    assert "Coffee meeting May 7" in prompt or "Q3 deadline" in prompt


def test_generate_brief_caches_to_disk(tmp_path: Path) -> None:
    vault = _seed_reece_atoms(tmp_path)
    provider = _FakeProvider()
    generate_client_brief(client_name="Reece", vault_root=vault, provider=provider)

    cache_path = vault / "01_Clients" / "reece" / "brief.json"
    assert cache_path.exists()
    cached = load_cached_brief(client_name="Reece", vault_root=vault)
    assert cached is not None
    assert cached.summary.startswith("Reece is sold")


def test_load_cached_brief_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_cached_brief(client_name="Reece", vault_root=tmp_path / "empty") is None


def test_load_cached_brief_returns_none_on_malformed_json(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    layout = VaultLayout.for_root(vault)
    ensure_vault_skeleton(layout)
    (layout.clients_dir / "reece").mkdir(parents=True, exist_ok=True)
    (layout.clients_dir / "reece" / "brief.json").write_text("not json", encoding="utf-8")
    assert load_cached_brief(client_name="Reece", vault_root=vault) is None


def test_brief_endpoint_returns_cached_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _seed_reece_atoms(tmp_path)
    provider = _FakeProvider()
    # Generate + cache once so the endpoint can hit the cache path.
    generate_client_brief(client_name="Reece", vault_root=vault, provider=provider)

    # Now hit the endpoint with refresh=false — should NOT call the LLM.
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: provider,
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)
    resp = client.get("/client_brief?client=Reece")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["client_slug"] == "reece"
    assert payload["summary"].startswith("Reece is sold")
    # Cached request should not have hit the LLM again (only the
    # original generate_client_brief invocation logged a request).
    assert len(provider.requests) == 1


def test_brief_endpoint_refresh_runs_llm_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _seed_reece_atoms(tmp_path)
    provider = _FakeProvider()
    generate_client_brief(client_name="Reece", vault_root=vault, provider=provider)

    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: provider,
    )
    app = create_app(vault_root=vault)
    client = TestClient(app)
    resp = client.get("/client_brief?client=Reece&refresh=true")
    assert resp.status_code == 200
    assert len(provider.requests) == 2  # original + the refresh
