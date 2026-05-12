"""Reindex tests — verify the index can be rebuilt from markdown alone."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from consultant_brain.embedder import LanceVaultIndex
from consultant_brain.ingest import run_ingest
from consultant_brain.reindex import _atom_from_markdown, run_reindex
from consultant_brain.vault import VaultLayout


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
        "summary": "Test summary.",
        "atoms": [
            {"type": "objection", "body": "Atom one body.", "confidence": 0.8, "tags": ["budget"]},
            {"type": "commitment", "body": "Atom two body.", "confidence": 0.9, "tags": ["next_step"]},
        ],
    }
)


def _ollama_available() -> bool:
    if os.environ.get("CI") == "true":
        return False
    try:
        import ollama
        ollama.embeddings(model="nomic-embed-text", prompt="ping")
        return True
    except Exception:
        return False


OLLAMA_AVAILABLE = _ollama_available()
requires_ollama = pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama + nomic-embed-text not available")


# ────────────────────────────────────────────────────────────────────────────
# Atom-from-markdown round trip
# ────────────────────────────────────────────────────────────────────────────


def test_atom_from_markdown_round_trips_through_vault(tmp_path: Path) -> None:
    """Ingest writes atoms → reindex parses them back → fields match."""
    vault_root = tmp_path / "vault"
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    layout = VaultLayout.for_root(vault_root)
    atom_files = list(layout.atoms_dir.glob("*.md"))
    assert atom_files, "ingest should have written atom files"

    for path in atom_files:
        atom = _atom_from_markdown(path)
        # Wikilink stripped — round-trip target is the underlying name.
        assert atom.client == "Reece"
        assert atom.call.startswith("2026-05-12_reece_")
        assert atom.type.value in {"objection", "commitment"}
        assert atom.body  # body present


# ────────────────────────────────────────────────────────────────────────────
# run_reindex behavior
# ────────────────────────────────────────────────────────────────────────────


def test_reindex_empty_vault_returns_zeros(tmp_path: Path) -> None:
    result = run_reindex(vault_root=tmp_path / "missing")
    assert result.scanned == 0 and result.embedded == 0 and result.skipped == 0


@requires_ollama
def test_reindex_rebuilds_index_after_wipe(tmp_path: Path) -> None:
    """Ingest → wipe LanceDB dir → reindex → query returns hits again."""
    vault_root = tmp_path / "vault"
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    layout = VaultLayout.for_root(vault_root)
    # Wipe the LanceDB dir; the atoms on disk still exist.
    import shutil

    shutil.rmtree(layout.system_dir / "lancedb")

    summary = run_reindex(vault_root=vault_root)
    assert summary.scanned >= 2
    assert summary.embedded >= 2
    assert summary.failed == []

    # Querying works again.
    index = LanceVaultIndex(layout)
    hits = index.query("budget", top_n=3)
    assert hits


@requires_ollama
def test_reindex_skips_when_already_indexed(tmp_path: Path) -> None:
    """Second reindex without --force shouldn't re-embed anything."""
    vault_root = tmp_path / "vault"
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    summary = run_reindex(vault_root=vault_root)
    assert summary.embedded == 0
    assert summary.skipped >= 2


@requires_ollama
def test_reindex_force_re_embeds_everything(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    summary = run_reindex(vault_root=vault_root, force=True)
    assert summary.embedded >= 2
    assert summary.skipped == 0
