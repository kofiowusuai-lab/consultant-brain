"""Tests for the three-layer retrieval pipeline.

Strategy: most tests use deterministic fake vectors (no Ollama dependency)
to verify the ranking, dedupe, and layer-isolation logic. A handful of
@requires_ollama tests prove the full semantic stack works against a real
embedding model.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.embedder import EMBEDDING_DIM, LanceVaultIndex
from consultant_brain.retrieve import (
    CONFIDENCE_WEIGHT,
    RECENCY_WEIGHT,
    SIMILARITY_WEIGHT,
    RankedHit,
    RetrievalResult,
    _recency_score,
    retrieve,
)
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
)
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


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
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _make_atom(
    id_: str,
    body: str,
    *,
    client: str | None = "Reece",
    call_type: CallType = CallType.consulting_call,
    type_: AtomType = AtomType.objection,
    confidence: float = 0.8,
    last_seen: date | None = None,
    tags: tuple[str, ...] = (),
) -> Atom:
    return Atom(
        id=id_,
        type=type_,
        client=client,
        call="2026-05-12_reece_consultingCall",
        call_type=call_type,
        tags=list(tags),
        confidence=confidence,
        evidence_count=1,
        last_seen=last_seen or date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id=id_,
        body=body,
    )


def _fake_vector(seed: float) -> list[float]:
    """A unit-ish vector that varies across atoms so search ranking isn't
    arbitrary."""
    v = [0.0] * EMBEDDING_DIM
    v[0] = 1.0 - seed * 0.05
    v[1] = seed * 0.1
    return v


# ────────────────────────────────────────────────────────────────────────────
# Recency math
# ────────────────────────────────────────────────────────────────────────────


def test_recency_today_is_one() -> None:
    today = date(2026, 5, 12)
    assert _recency_score("2026-05-12", today=today) == pytest.approx(1.0)


def test_recency_at_half_life_is_one_half() -> None:
    today = date(2026, 5, 12)
    # Half-life is 30 days
    thirty_days_ago = date(2026, 4, 12)
    assert _recency_score(thirty_days_ago.isoformat(), today=today) == pytest.approx(0.5)


def test_recency_missing_iso_is_zero() -> None:
    assert _recency_score("", today=date(2026, 5, 12)) == 0.0
    assert _recency_score("not-a-date", today=date(2026, 5, 12)) == 0.0


def test_ranking_weights_sum_to_one() -> None:
    """If they don't sum to 1, the score can exceed 1.0 and break UI rendering."""
    assert SIMILARITY_WEIGHT + RECENCY_WEIGHT + CONFIDENCE_WEIGHT == pytest.approx(1.0)


# ────────────────────────────────────────────────────────────────────────────
# Empty vault
# ────────────────────────────────────────────────────────────────────────────


def test_retrieve_returns_empty_when_vault_missing(tmp_path: Path) -> None:
    result = retrieve(
        transcript_window="anything",
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "does-not-exist",
    )
    assert isinstance(result, RetrievalResult)
    assert result.hot == [] and result.warm == [] and result.cold == []


def test_retrieve_returns_empty_when_index_empty(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    # Vault exists but no LanceDB index has been opened yet.
    result = retrieve(
        transcript_window="anything",
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "vault",
    )
    assert result.all == []


# ────────────────────────────────────────────────────────────────────────────
# Hot layer
# ────────────────────────────────────────────────────────────────────────────


@requires_ollama
def test_hot_layer_pulls_atoms_for_named_client(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)

    reece = _make_atom("REECE_HOT_AAAAAAAAAAAAAAAA01", "Reece has 1000 notes in Obsidian.", tags=("stack",))
    acme = _make_atom("ACME_HOT_BBBBBBBBBBBBBBBBB02", "Acme runs on Notion not Obsidian.", tags=("stack",))
    object.__setattr__(acme, "client", "Acme")
    index.upsert(reece)
    index.upsert(acme)

    result = retrieve(
        transcript_window="how does the knowledge base look",
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "vault",
    )
    assert len(result.hot) >= 1
    assert all(rh.hit.client == "Reece" for rh in result.hot)
    assert all(rh.layer == "hot" for rh in result.hot)


@requires_ollama
def test_hot_excludes_atoms_from_warm_layer(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)

    atom = _make_atom("DUPE_AAAAAAAAAAAAAAAAAAAAA01", "Bot keeps grabbing wrong notes.", tags=("retrieval",))
    index.upsert(atom)

    result = retrieve(
        transcript_window="why does the bot pick the wrong notes",
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "vault",
    )
    hot_ids = {rh.hit.id for rh in result.hot}
    warm_ids = {rh.hit.id for rh in result.warm}
    assert hot_ids.isdisjoint(warm_ids), "an atom shouldn't appear in both layers"


def test_hot_returns_empty_when_no_client_specified(tmp_path: Path) -> None:
    """Hot is client-scoped by definition; without a client we have nothing
    to scope against."""
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)
    index.upsert(_make_atom("ANY_AAAAAAAAAAAAAAAAAAAAA01", "Some atom."), vector=_fake_vector(1))

    result = retrieve(
        transcript_window="anything",
        client=None,
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "vault",
    )
    assert result.hot == []


# ────────────────────────────────────────────────────────────────────────────
# Warm layer dedupe
# ────────────────────────────────────────────────────────────────────────────


@requires_ollama
def test_warm_dedupe_caps_per_tag_bucket(tmp_path: Path) -> None:
    """If 4 atoms all share (type=objection, primary_tag=budget), warm
    should surface at most WARM_PER_TAG_CAP of them."""
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)

    bodies = [
        "Price is way more than we expected this quarter.",
        "Budget approval is going to be a nightmare for procurement.",
        "We can't justify the cost without ROI numbers first.",
        "Fees came up before scope, anchoring concern.",
    ]
    for i, body in enumerate(bodies):
        atom = _make_atom(
            f"BUDGET_DEDUPE_AAAAAAAAAAA{i:02d}",
            body,
            client="DifferentClient",  # keep them out of hot
            tags=("budget", "fees"),
        )
        index.upsert(atom)

    result = retrieve(
        transcript_window="ballpark cost",
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "vault",
    )
    # All four atoms share (objection, budget) — dedupe caps the bucket at 2.
    budget_hits = [rh for rh in result.warm if rh.hit.primary_tag == "budget"]
    assert len(budget_hits) <= 2


# ────────────────────────────────────────────────────────────────────────────
# Cold layer (stub)
# ────────────────────────────────────────────────────────────────────────────


def test_cold_layer_is_empty_until_patterns_ship(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    index = LanceVaultIndex(layout)
    index.upsert(_make_atom("CCC_AAAAAAAAAAAAAAAAAAAAA01", "any."), vector=_fake_vector(1))

    result = retrieve(
        transcript_window="anything",
        client="Reece",
        call_type=CallType.consulting_call,
        vault_root=tmp_path / "vault",
    )
    assert result.cold == []


# ────────────────────────────────────────────────────────────────────────────
# RetrievalResult slicing
# ────────────────────────────────────────────────────────────────────────────


def test_top_for_panel_caps_each_layer() -> None:
    rh = lambda i, layer: RankedHit(  # noqa: E731
        hit=None,  # type: ignore[arg-type]
        similarity=1 - i * 0.1,
        recency=0.5,
        confidence=0.8,
        score=1 - i * 0.1,
        layer=layer,
        reason="t",
    )
    result = RetrievalResult(
        hot=[rh(i, "hot") for i in range(3)],
        warm=[rh(i, "warm") for i in range(5)],
        cold=[rh(i, "cold") for i in range(2)],
        knowledge=[rh(i, "knowledge") for i in range(4)],
    )
    panel = result.top_for_panel(hot=1, warm=2, cold=1, knowledge=2)
    assert len(panel) == 6
    assert [rh.layer for rh in panel] == [
        "hot",
        "warm",
        "warm",
        "cold",
        "knowledge",
        "knowledge",
    ]


def test_top_for_panel_defaults_knowledge_to_zero() -> None:
    """Live calls don't surface knowledge unless explicitly requested."""
    rh = RankedHit(
        hit=None,  # type: ignore[arg-type]
        similarity=0.9,
        recency=0.5,
        confidence=0.8,
        score=0.9,
        layer="knowledge",
        reason="external",
    )
    result = RetrievalResult(knowledge=[rh])
    panel = result.top_for_panel(hot=1, warm=2, cold=1)
    assert len(panel) == 0  # knowledge ignored unless caller asks for it
