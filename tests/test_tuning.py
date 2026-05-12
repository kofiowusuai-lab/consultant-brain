"""Retrieval tuning corpus.

Loads tests/fixtures/tuning_corpus.json — 7 seed atoms + 5 transcript-
window queries with the expected top-1 atom for each. Builds a vault,
indexes the atoms, runs each query through the retrieve() pipeline, and
asserts that the top hit matches expectations.

This test is the *spec* for retrieval quality. If you change ranking
weights, the embedder, USE_TYPE_PREFIX, or the LanceDB schema, run this
to see whether you improved or regressed. Failing here on a CI run is a
deliberate signal — the corpus encodes "what good retrieval looks like."

Gated by @requires_ollama: the test embeds real text through Ollama and
makes real LanceDB queries. Local-only, never runs in CI.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.embedder import LanceVaultIndex
from consultant_brain.retrieve import retrieve
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
)
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


CORPUS_PATH = Path(__file__).parent / "fixtures" / "tuning_corpus.json"


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


def _seed_vault(tmp_path: Path) -> Path:
    """Build a vault, index every seed atom from the corpus, return the vault root."""
    corpus = json.loads(CORPUS_PATH.read_text())
    vault_root = tmp_path / "tuning_vault"
    layout = VaultLayout.for_root(vault_root)
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)
    for seed in corpus["atoms"]:
        atom = Atom(
            id=seed["id"],
            type=AtomType(seed["type"]),
            client=seed["client"],
            call="2026-05-12_reece_consultingCall",
            call_type=CallType.consulting_call,
            tags=list(seed.get("tags", [])),
            confidence=float(seed["confidence"]),
            evidence_count=1,
            last_seen=date(2026, 5, 12),
            created_at=datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc),
            status=AtomStatus.active,
            embedding_id=seed["id"],
            body=seed["body"],
        )
        index.upsert(atom)
    return vault_root


@requires_ollama
@pytest.mark.parametrize("query_name", [
    "price_question",
    "retrieval_complaint",
    "next_step_commitment",
    "definition_of_done",
    "stack_check",
])
def test_tuning_corpus_top1(tmp_path: Path, query_name: str) -> None:
    """For each query in the corpus, the expected atom must be the top-1
    hit across the panel slice (hot ∪ warm ∪ cold, in that order)."""
    corpus = json.loads(CORPUS_PATH.read_text())
    query = next(q for q in corpus["queries"] if q["name"] == query_name)

    vault_root = _seed_vault(tmp_path)
    result = retrieve(
        transcript_window=query["window"],
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=vault_root,
    )
    panel = result.top_for_panel(hot=1, warm=2, cold=1)
    assert panel, f"no panel hits for {query_name}"
    top = panel[0]
    expected = query["expected_top_atom_id"]
    assert top.hit.id == expected, (
        f"\nQuery: {query['name']!r} — {query['window']}\n"
        f"Expected top atom: {expected}\n"
        f"  ({query['rationale']})\n"
        f"Got:               {top.hit.id}\n"
        f"  type={top.hit.type} score={top.score:.3f}\n"
        f"  body: {top.hit.body[:120]}"
    )


@requires_ollama
def test_tuning_corpus_summary_acceptance(tmp_path: Path) -> None:
    """A single rollup test so a CI failure summary lists "tuning regressed
    on 3/5" instead of 3 separate parametrized failures. Useful for tracking
    progress as the ranker evolves."""
    corpus = json.loads(CORPUS_PATH.read_text())
    vault_root = _seed_vault(tmp_path)
    misses: list[str] = []
    for query in corpus["queries"]:
        result = retrieve(
            transcript_window=query["window"],
            client="Reece",
            call_type=CallType.consulting_call,
            vault_root=vault_root,
        )
        panel = result.top_for_panel(hot=1, warm=2, cold=1)
        if not panel or panel[0].hit.id != query["expected_top_atom_id"]:
            got = panel[0].hit.id if panel else "(no hits)"
            misses.append(f"{query['name']}: expected {query['expected_top_atom_id']}, got {got}")
    # Hard requirement: 5/5 on the corpus. Lower the bar here if we deliberately
    # accept regressions during tuning experiments.
    assert not misses, "Tuning corpus regressions:\n  " + "\n  ".join(misses)
