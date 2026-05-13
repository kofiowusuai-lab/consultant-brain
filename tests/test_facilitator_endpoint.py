"""Phase 13 — facilitator console HTTP endpoint tests.

Round-trips the new GET /facilitator/suggest_stage endpoint through
FastAPI's TestClient. Live-call setup uses /call_start +
/transcript_delta so the endpoint reads from the real registry
state, matching production.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


@pytest.fixture
def app_client(vault_root: Path) -> TestClient:
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    return TestClient(app)


def _start_call(client: TestClient, call_id: str = "CALL_FAC1") -> None:
    response = client.post(
        "/call_start",
        json={"call_id": call_id, "client": "Reece", "call_type": "consultingCall"},
    )
    assert response.status_code == 200, response.text


def _post_delta(client: TestClient, call_id: str, speaker: str, text: str) -> None:
    response = client.post(
        "/transcript_delta",
        json={"call_id": call_id, "speaker": speaker, "text": text},
    )
    assert response.status_code == 200, response.text


def test_unknown_call_id_returns_404(app_client: TestClient) -> None:
    response = app_client.get(
        "/facilitator/suggest_stage",
        params={"call_id": "DOES_NOT_EXIST", "current_stage_index": 0},
    )
    assert response.status_code == 404


def test_live_call_empty_window_abstains(app_client: TestClient) -> None:
    _start_call(app_client)
    response = app_client.get(
        "/facilitator/suggest_stage",
        params={"call_id": "CALL_FAC1", "current_stage_index": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["call_id"] == "CALL_FAC1"
    assert body["suggested_stage_index"] is None
    assert body["confidence"] == 0.0


def test_cued_transition_suggests_next_stage(app_client: TestClient) -> None:
    _start_call(app_client)
    _post_delta(app_client, "CALL_FAC1", "you", "alright, this all makes sense.")
    _post_delta(app_client, "CALL_FAC1", "them", "good.")
    _post_delta(
        app_client,
        "CALL_FAC1",
        "you",
        "let's move on to scope. what's in, what's parked?",
    )
    response = app_client.get(
        "/facilitator/suggest_stage",
        params={"call_id": "CALL_FAC1", "current_stage_index": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_stage_index"] == 3
    assert body["confidence"] > 0.0
    assert "scope" in body["reason"].lower()


def test_end_of_outline_returns_null(app_client: TestClient) -> None:
    _start_call(app_client)
    _post_delta(app_client, "CALL_FAC1", "you", "post-call memo coming")
    response = app_client.get(
        "/facilitator/suggest_stage",
        params={"call_id": "CALL_FAC1", "current_stage_index": 7},
    )
    assert response.status_code == 200
    assert response.json()["suggested_stage_index"] is None


def test_invalid_current_stage_index_rejected(app_client: TestClient) -> None:
    _start_call(app_client)
    response = app_client.get(
        "/facilitator/suggest_stage",
        params={"call_id": "CALL_FAC1", "current_stage_index": -1},
    )
    assert response.status_code == 422
