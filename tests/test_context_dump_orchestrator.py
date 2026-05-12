"""End-to-end orchestrator tests — run_preview + commit_preview.

The LLM is mocked via _FakeProvider (same pattern as the Phase 9
test). Parsers are exercised through parse_to_text on plain text
files so we avoid yet-more SDK mocking.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.context_dumps import (
    ContextDumpPreviewStore,
    commit_preview,
    run_preview,
)
from consultant_brain.context_dumps.preview_store import ContextDumpPreviewStore as Store
from consultant_brain.llm.provider import ChatRequest, ChatResponse


class _FakeProvider:
    """LLMProvider double that returns a canned JSON payload every call."""

    name = "fake"

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(text=self._payload, model_used=request.model)


def _txt_file(tmp_path: Path, body: str = "Coffee with Reece. He said the bot needs to ship by Q3.") -> Path:
    f = tmp_path / "note.txt"
    f.write_text(body, encoding="utf-8")
    return f


_OK_PAYLOAD = (
    '{"summary":"Coffee at Verve.","atoms":[' \
    '{"type":"commitment","body":"Reece commits to ship by Q3.","confidence":0.9,"tags":["timeline"]},'
    '{"type":"insight","body":"Reece anchors on timeline before scope.","confidence":0.7,"tags":["framing"]}'
    "]}"
)


def test_run_preview_returns_atoms_without_writing(tmp_path: Path) -> None:
    provider = _FakeProvider(_OK_PAYLOAD)
    f = _txt_file(tmp_path)

    preview = run_preview(
        file_path=f,
        client_name="Reece",
        observed_at=date(2026, 5, 12),
        notes="25-min coffee",
        vault_root=tmp_path / "vault",
        provider=provider,
    )

    assert preview.client_name == "Reece"
    assert preview.client_slug == "reece"
    assert preview.observed_at == date(2026, 5, 12)
    assert len(preview.atoms) == 2
    assert preview.summary == "Coffee at Verve."
    assert preview.notes == "25-min coffee"
    # Vault is untouched on preview.
    assert not (tmp_path / "vault" / "03_Atoms").exists()
    assert not (tmp_path / "vault" / "10_ContextDumps").exists()
    # tmp text file persisted for commit.
    assert preview.raw_text_full_path.exists()
    # Prompt the LLM saw includes the time preamble + client + notes.
    user_prompt = provider.requests[0].user
    assert "Today is" in user_prompt
    assert "context was observed on 2026-05-12" in user_prompt
    assert "Client: Reece" in user_prompt
    assert "25-min coffee" in user_prompt


def test_commit_preview_writes_atoms_with_observed_at(tmp_path: Path) -> None:
    provider = _FakeProvider(_OK_PAYLOAD)
    f = _txt_file(tmp_path)
    preview = run_preview(
        file_path=f,
        client_name="Reece",
        observed_at=date(2026, 5, 1),
        notes=None,
        vault_root=tmp_path / "vault",
        provider=provider,
    )

    result = commit_preview(preview=preview, vault_root=tmp_path / "vault")
    assert result.atom_count == 2
    # Knowledge note exists at the expected per-client subdir.
    assert result.dump_note_path.exists()
    assert result.dump_note_path.parent.name == "reece"
    assert result.dump_note_path.name.startswith("2026-05-01_")
    # Every atom file has last_seen == observed_at.
    atom_dir = tmp_path / "vault" / "03_Atoms"
    files = list(atom_dir.glob("*.md"))
    assert len(files) == 2
    for atom_file in files:
        text = atom_file.read_text(encoding="utf-8")
        assert "last_seen: '2026-05-01'" in text or "last_seen: 2026-05-01" in text
        assert "source_kind: context_dump" in text


def test_selective_commit_keeps_only_chosen_indexes(tmp_path: Path) -> None:
    provider = _FakeProvider(_OK_PAYLOAD)
    preview = run_preview(
        file_path=_txt_file(tmp_path),
        client_name="Reece",
        observed_at=date(2026, 5, 12),
        notes=None,
        vault_root=tmp_path / "vault",
        provider=provider,
    )

    result = commit_preview(
        preview=preview,
        vault_root=tmp_path / "vault",
        accepted_atom_indexes=[1],  # keep only the insight
    )
    assert result.atom_count == 1
    atoms_dir = tmp_path / "vault" / "03_Atoms"
    files = list(atoms_dir.glob("*.md"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert "Reece anchors on timeline" in body


def test_preview_store_ttl_evicts(tmp_path: Path) -> None:
    """Store with 0.1s TTL drops entries after a short sleep — covers
    the lazy eviction path."""
    store: Store = ContextDumpPreviewStore(ttl_seconds=0)
    provider = _FakeProvider(_OK_PAYLOAD)
    preview = run_preview(
        file_path=_txt_file(tmp_path),
        client_name="Reece",
        observed_at=date(2026, 5, 12),
        notes=None,
        vault_root=tmp_path / "vault",
        provider=provider,
    )
    store.put(preview)
    time.sleep(0.05)
    # TTL=0 means anything > 0s elapsed is stale.
    fetched = store.get(preview.preview_id)
    assert fetched is None


def test_preview_store_pop_removes_entry(tmp_path: Path) -> None:
    store = ContextDumpPreviewStore(ttl_seconds=60)
    provider = _FakeProvider(_OK_PAYLOAD)
    preview = run_preview(
        file_path=_txt_file(tmp_path),
        client_name="Reece",
        observed_at=date(2026, 5, 12),
        notes=None,
        vault_root=tmp_path / "vault",
        provider=provider,
    )
    store.put(preview)
    popped = store.pop(preview.preview_id)
    assert popped is not None
    assert store.get(preview.preview_id) is None


def test_run_preview_handles_empty_text(tmp_path: Path) -> None:
    """Empty file → preview returned with zero atoms + the warnings
    from the parser."""
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")

    provider = _FakeProvider(_OK_PAYLOAD)
    preview = run_preview(
        file_path=f,
        client_name="Reece",
        observed_at=date(2026, 5, 12),
        notes=None,
        vault_root=tmp_path / "vault",
        provider=provider,
    )
    assert len(preview.atoms) == 0
    assert any("empty" in w.lower() for w in preview.warnings)
    # LLM was never called.
    assert provider.requests == []
