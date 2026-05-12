"""End-to-end tests for `learn.py` — the Phase 9 ingest pipeline.

Mocks the LLM provider + injects a synthetic SourceMaterial so the
tests don't hit the network. Verifies that:
  - atoms are written with source_kind / source_url / source_title set
  - the 09_Knowledge/<id>.md note links every atom
  - dry-run skips disk writes
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.learn import run_learn
from consultant_brain.llm.provider import ChatRequest, ChatResponse
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
    ExtractedAtom,
    ExtractorResult,
    SourceKind,
)
from consultant_brain.sources import SourceMaterial


class _FakeProvider:
    """LLMProvider double — returns the canned JSON every call."""

    name = "fake"

    def __init__(self, json_payload: str) -> None:
        self._payload = json_payload
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(text=self._payload, model_used=request.model)


def _make_source() -> SourceMaterial:
    return SourceMaterial(
        kind=SourceKind.youtube,
        url="https://www.youtube.com/watch?v=HD2RU2QZxJk",
        source_id="youtube_HD2RU2QZxJk",
        title="Ad Bot Pricing",
        transcript="Speaker: pricing services is mostly about anchoring...",
        author="HormoziTalks",
        duration_seconds=420,
        published_at=date(2026, 5, 1),
        fetched_at=datetime(2026, 5, 12, 12, 0, 0),
        language="en",
        topic="ai-sales",
        for_client=None,
    )


def test_run_learn_writes_atoms_and_knowledge_note(tmp_path: Path) -> None:
    provider = _FakeProvider(
        '{"summary":"Anchor first.","atoms":[{"type":"insight","body":"Anchoring the budget before scope is the bigger leverage.","confidence":0.91,"tags":["pricing"]},{"type":"client_fact","body":"His team uses 4 emails per week per prospect.","confidence":0.88,"tags":["outbound"]}]}'
    )
    result = run_learn(
        url="https://youtu.be/HD2RU2QZxJk",
        vault_root=tmp_path,
        topic="ai-sales",
        provider=provider,
        source=_make_source(),
    )
    assert result.atom_count == 2
    assert result.source_kind == SourceKind.youtube
    assert result.knowledge_note_path is not None
    assert result.knowledge_note_path.exists()

    knowledge_text = result.knowledge_note_path.read_text(encoding="utf-8")
    assert "Ad Bot Pricing" in knowledge_text
    assert "Anchor first" in knowledge_text
    assert "## Atoms" in knowledge_text

    # Each atom file carries source_kind + source_url frontmatter
    atom_dir = tmp_path / "03_Atoms"
    md_files = list(atom_dir.glob("*.md"))
    assert len(md_files) == 2
    sample = md_files[0].read_text(encoding="utf-8")
    assert "source_kind: youtube" in sample
    assert "source_url:" in sample
    assert "source_title: Ad Bot Pricing" in sample


def test_run_learn_dry_run_skips_disk(tmp_path: Path) -> None:
    provider = _FakeProvider(
        '{"summary":"Anchor.","atoms":[{"type":"insight","body":"X.","confidence":0.9,"tags":[]}]}'
    )
    result = run_learn(
        url="https://youtu.be/HD2RU2QZxJk",
        vault_root=tmp_path,
        provider=provider,
        source=_make_source(),
        dry_run=True,
    )
    assert result.dry_run
    assert result.atom_count == 1
    assert result.knowledge_note_path is None
    # No 09_Knowledge directory was created on dry-run.
    assert not (tmp_path / "09_Knowledge").exists()
    assert not (tmp_path / "03_Atoms").exists()


def test_run_learn_propagates_topic_into_atom_tags(tmp_path: Path) -> None:
    provider = _FakeProvider(
        '{"summary":"X.","atoms":[{"type":"insight","body":"Y.","confidence":0.9,"tags":["scope"]}]}'
    )
    source = _make_source()
    object.__setattr__(source, "topic", "ai_sales")
    result = run_learn(
        url="https://youtu.be/HD2RU2QZxJk",
        vault_root=tmp_path,
        provider=provider,
        source=source,
    )
    atom_dir = tmp_path / "03_Atoms"
    atom_file = list(atom_dir.glob("*.md"))[0]
    text = atom_file.read_text(encoding="utf-8")
    assert "ai_sales" in text
    assert "youtube" in text  # source-kind tag also added
    assert "scope" in text  # original LLM-suggested tag preserved


def test_run_learn_for_client_writes_client_folder(tmp_path: Path) -> None:
    provider = _FakeProvider(
        '{"summary":"X.","atoms":[{"type":"insight","body":"Y.","confidence":0.9,"tags":[]}]}'
    )
    source = _make_source()
    object.__setattr__(source, "for_client", "Reece")
    run_learn(
        url="https://youtu.be/HD2RU2QZxJk",
        vault_root=tmp_path,
        provider=provider,
        source=source,
    )
    # client folder exists
    client_dir = tmp_path / "01_Clients" / "reece"
    assert client_dir.exists()
    # atom carries client name
    atom_dir = tmp_path / "03_Atoms"
    text = list(atom_dir.glob("*.md"))[0].read_text(encoding="utf-8")
    assert "Reece" in text
