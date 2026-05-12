"""FastAPI service integration tests. Use FastAPI's TestClient (httpx-
based) so the full stack runs in-process — no real network.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consultant_brain.embedder import LanceVaultIndex
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
)
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """Empty vault skeleton."""
    root = tmp_path / "vault"
    layout = VaultLayout.for_root(root)
    ensure_vault_skeleton(layout)
    return root


@pytest.fixture
def app_client(vault_root: Path) -> TestClient:
    """Fresh app + fresh registry per test — no state leaks between tests."""
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    return TestClient(app)


# ────────────────────────────────────────────────────────────────────────────
# Healthz
# ────────────────────────────────────────────────────────────────────────────


def test_healthz_returns_ok(app_client: TestClient) -> None:
    response = app_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["active_calls"] == 0
    assert "vault_root" in body


# ────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ────────────────────────────────────────────────────────────────────────────


def test_call_start_creates_state(app_client: TestClient) -> None:
    response = app_client.post(
        "/call_start",
        json={"call_id": "call_abc", "client": "Reece", "call_type": "consultingCall"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["call_id"] == "call_abc"
    assert body["client"] == "Reece"
    assert body["call_type"] == "consultingCall"

    # Healthz now reflects 1 active call.
    health = app_client.get("/healthz").json()
    assert health["active_calls"] == 1


def test_call_end_frees_state(app_client: TestClient) -> None:
    app_client.post("/call_start", json={"call_id": "call_abc", "client": "Reece", "call_type": "consultingCall"})
    response = app_client.post("/call_end", json={"call_id": "call_abc"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["call_id"] == "call_abc"
    assert body["ended"] is True
    # Phase 5.2: /call_end now writes a CallNote on disk so the score
    # endpoint can read it back immediately. The note ID is surfaced
    # for the Swift override sheet.
    assert body["call_note_id"] is not None
    assert body["call_note_id"].endswith("_consultingCall")
    assert app_client.get("/healthz").json()["active_calls"] == 0


def test_call_end_unknown_id_reports_false(app_client: TestClient) -> None:
    response = app_client.post("/call_end", json={"call_id": "never_started"})
    assert response.status_code == 200
    assert response.json()["ended"] is False


def test_call_start_invalid_call_type_returns_422(app_client: TestClient) -> None:
    response = app_client.post(
        "/call_start",
        json={"call_id": "call_abc", "client": "Reece", "call_type": "totally-bogus"},
    )
    assert response.status_code == 422
    assert "call_type" in response.text


# ────────────────────────────────────────────────────────────────────────────
# Transcript deltas
# ────────────────────────────────────────────────────────────────────────────


def test_transcript_delta_appends_turn(app_client: TestClient) -> None:
    app_client.post("/call_start", json={"call_id": "call_abc", "client": "Reece", "call_type": "consultingCall"})
    response = app_client.post(
        "/transcript_delta",
        json={"call_id": "call_abc", "speaker": "them", "text": "We have about a thousand notes."},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted"] is True
    assert body["turn_count"] == 1

    # Second delta increments the count.
    response = app_client.post(
        "/transcript_delta",
        json={"call_id": "call_abc", "speaker": "you", "text": "Walk me through your stack."},
    )
    assert response.json()["turn_count"] == 2


def test_transcript_delta_unknown_call_returns_404(app_client: TestClient) -> None:
    response = app_client.post(
        "/transcript_delta",
        json={"call_id": "never_started", "speaker": "them", "text": "hi"},
    )
    assert response.status_code == 404
    assert "POST /call_start" in response.text


def test_transcript_delta_invalid_speaker_returns_422(app_client: TestClient) -> None:
    app_client.post("/call_start", json={"call_id": "call_abc", "client": "Reece", "call_type": "consultingCall"})
    response = app_client.post(
        "/transcript_delta",
        json={"call_id": "call_abc", "speaker": "bystander", "text": "hi"},
    )
    assert response.status_code == 422


# ────────────────────────────────────────────────────────────────────────────
# Suggestions
# ────────────────────────────────────────────────────────────────────────────


def test_suggestions_unknown_call_returns_404(app_client: TestClient) -> None:
    response = app_client.get("/suggestions?call_id=never_started")
    assert response.status_code == 404


def test_suggestions_empty_window_returns_no_suggestions(app_client: TestClient) -> None:
    """A call with no transcript deltas yet returns an empty suggestion list,
    not an error. The Swift app polls every 15s, including before the first
    turn — we want a clean 200 there."""
    app_client.post("/call_start", json={"call_id": "call_abc", "client": "Reece", "call_type": "consultingCall"})
    response = app_client.get("/suggestions?call_id=call_abc")
    assert response.status_code == 200
    body = response.json()
    assert body["suggestions"] == []
    assert body["window_chars"] == 0


def _ollama_available() -> bool:
    import os
    if os.environ.get("CI") == "true":
        return False
    try:
        import ollama
        ollama.embeddings(model="nomic-embed-text", prompt="ping")
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama + nomic-embed-text not available")


def _seed_test_atoms(vault_root: Path) -> None:
    """Seed a single client with a couple of atoms so /suggestions has
    something to return."""
    layout = VaultLayout.for_root(vault_root)
    index = LanceVaultIndex(layout)

    def make(id_: str, body: str, type_: AtomType = AtomType.objection, tags=()) -> Atom:
        return Atom(
            id=id_,
            type=type_,
            client="Reece",
            call="2026-05-12_reece_consultingCall",
            call_type=CallType.consulting_call,
            tags=list(tags),
            confidence=0.85,
            evidence_count=1,
            last_seen=date(2026, 5, 12),
            created_at=datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc),
            status=AtomStatus.active,
            embedding_id=id_,
            body=body,
        )

    index.upsert(make("SVCTESTAAAAAAAAAAAAAAAAAA1", "Ad-bot retrieval keeps pulling wrong notes from the vault."))
    index.upsert(make("SVCTESTAAAAAAAAAAAAAAAAAA2", "Reece will send 3 manual ads by Friday.", type_=AtomType.commitment))


@requires_ollama
def test_full_lifecycle_returns_ranked_suggestions(app_client: TestClient, vault_root: Path) -> None:
    """Happy path the Swift app exercises every call:
       start → 3 deltas → poll suggestions → end."""
    _seed_test_atoms(vault_root)

    app_client.post(
        "/call_start",
        json={"call_id": "call_xyz", "client": "Reece", "call_type": "consultingCall"},
    )
    deltas = [
        ("you", "Walk me through your ad-bot workflow."),
        ("them", "The bot keeps grabbing the wrong sections from our Obsidian vault."),
        ("you", "Got it — so retrieval is misfiring, not generation."),
    ]
    for speaker, text in deltas:
        r = app_client.post(
            "/transcript_delta",
            json={"call_id": "call_xyz", "speaker": speaker, "text": text},
        )
        assert r.status_code == 200, r.text

    response = app_client.get("/suggestions?call_id=call_xyz")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window_chars"] > 0
    assert isinstance(body["suggestions"], list)
    # At least one suggestion should come back given the seeded atoms.
    assert body["suggestions"], "expected at least one suggestion from seeded atoms"
    for s in body["suggestions"]:
        assert s["atom_id"]
        assert s["layer"] in {"hot", "warm", "cold"}
        assert 0.0 <= s["similarity"] <= 1.0
        assert 0.0 <= s["score"] <= 1.0

    ended = app_client.post("/call_end", json={"call_id": "call_xyz"})
    assert ended.json()["ended"] is True
    # State is freed — next /suggestions returns 404.
    assert app_client.get("/suggestions?call_id=call_xyz").status_code == 404
