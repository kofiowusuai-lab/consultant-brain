"""Embedder + LanceDB tests. Real Ollama calls require the model to be
pulled, so we gate them with the OLLAMA_AVAILABLE check — CI without the
model still gets useful coverage from the schema + index plumbing.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.embedder import (
    EMBEDDING_DIM,
    AtomHit,
    LanceVaultIndex,
    embed_text,
)
from consultant_brain.schemas import Atom, AtomStatus, AtomType, CallType
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


def _ollama_available() -> bool:
    """Cheap check that `ollama serve` is up + nomic-embed-text is pulled."""
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


def _make_atom(id_: str, body: str, type_: AtomType = AtomType.objection) -> Atom:
    return Atom(
        id=id_,
        type=type_,
        client="Reece",
        call="2026-05-12_reece_consultingCall",
        call_type=CallType.consulting_call,
        tags=[],
        confidence=0.8,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id=id_,
        body=body,
    )


# ────────────────────────────────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────────────────────────────────


def test_embed_text_rejects_empty() -> None:
    with pytest.raises(ValueError):
        embed_text("")


@requires_ollama
def test_embed_text_returns_correct_dimension() -> None:
    vector = embed_text("hello world")
    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vector)


# ────────────────────────────────────────────────────────────────────────────
# LanceDB index
# ────────────────────────────────────────────────────────────────────────────


def test_lance_index_creates_table_on_first_upsert(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)
    assert index.atom_count() == 0
    # Provide a fake vector so this test doesn't need Ollama.
    fake_vector = [0.0] * EMBEDDING_DIM
    fake_vector[0] = 1.0
    atom = _make_atom("AAAAAAAAAAAAAAAAAAAAAAAA01", "Price came up before scope.")
    index.upsert(atom, vector=fake_vector)
    assert index.atom_count() == 1


def test_lance_index_upsert_replaces_existing_row(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)
    fake_vector = [0.0] * EMBEDDING_DIM
    fake_vector[0] = 1.0

    atom_v1 = _make_atom("AAAAAAAAAAAAAAAAAAAAAAAA01", "Original body.")
    atom_v2 = _make_atom("AAAAAAAAAAAAAAAAAAAAAAAA01", "Updated body after re-ingest.")
    index.upsert(atom_v1, vector=fake_vector)
    index.upsert(atom_v2, vector=fake_vector)
    assert index.atom_count() == 1  # still one row — replaced, not appended


def test_lance_index_query_returns_empty_when_no_table(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)
    # We constructed the index but never upserted, so no table yet.
    assert index.query("anything", top_n=5) == []


@requires_ollama
def test_lance_index_round_trip(tmp_path: Path) -> None:
    """Real Ollama embedding + LanceDB write + LanceDB query — the slice the
    `consultant-brain query` CLI exercises end-to-end."""
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)

    atoms = [
        _make_atom(
            "BBBBBBBBBBBBBBBBBBBBBBBB01",
            "Price came up before scope. Reece asked about ballpark fees in the first five minutes.",
            type_=AtomType.objection,
        ),
        _make_atom(
            "BBBBBBBBBBBBBBBBBBBBBBBB02",
            "Reece will send the 3 best manual ads by Friday.",
            type_=AtomType.commitment,
        ),
        _make_atom(
            "BBBBBBBBBBBBBBBBBBBBBBBB03",
            "Bot keeps grabbing wrong notes — no tagging system, just timestamps and titles.",
            type_=AtomType.client_fact,
        ),
    ]
    for atom in atoms:
        index.upsert(atom)

    hits = index.query("how much does this cost", top_n=3)
    assert len(hits) > 0
    # The price-related atom should be the top hit.
    top = hits[0]
    assert "price" in top.body.lower() or "fee" in top.body.lower() or "cost" in top.body.lower(), top.body
    assert isinstance(top, AtomHit)
    assert 0.0 <= top.similarity <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# AtomHit formatting
# ────────────────────────────────────────────────────────────────────────────


def test_atom_hit_format_line_truncates_long_body() -> None:
    long_body = "x" * 200
    hit = AtomHit(
        id="A",
        type="objection",
        body=long_body,
        client="Reece",
        call="2026-05-12_reece_consultingCall",
        call_type="consultingCall",
        distance=0.4,
    )
    line = hit.format_line()
    assert "..." in line  # truncated
    assert "[objection" in line  # type tag visible
    assert "Reece" in line
