"""Tests for the /score + /score_override endpoints + the corrections log."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consultant_brain.ingest import run_ingest
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.scoring.corrections import corrections_path, load_corrections
from consultant_brain.service import create_app


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
        "summary": "Reece wants retrieval fixed first.",
        "atoms": [
            {"type": "objection", "body": "Bot grabs wrong notes.", "confidence": 0.88, "tags": ["retrieval"]},
            {"type": "commitment", "body": "Reece will send 3 ads by Friday.", "confidence": 0.95, "tags": ["next_step"]},
            {"type": "client_fact", "body": "1000 notes in Obsidian vault.", "confidence": 0.9, "tags": ["stack"]},
        ],
    }
)


@pytest.fixture
def vault_with_call(tmp_path: Path) -> tuple[Path, str]:
    """A vault that has one ingested call, ready for scoring."""
    vault_root = tmp_path / "vault"
    result = run_ingest(
        session_path=FIXTURE,
        client_name="Reece",
        call_type="consultingCall",
        vault_root=vault_root,
        redact=False,
        dry_run=False,
        client=_MockClient(VALID_RESPONSE),
    )
    return vault_root, result.call_note_path.stem


@pytest.fixture
def app_client(vault_with_call) -> TestClient:
    vault_root, _ = vault_with_call
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    return TestClient(app)


# ────────────────────────────────────────────────────────────────────────────
# GET /score
# ────────────────────────────────────────────────────────────────────────────


def test_score_returns_breakdown(app_client: TestClient, vault_with_call) -> None:
    _, call_id = vault_with_call
    response = app_client.get(f"/score?call_id={call_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["call_id"] == call_id
    assert body["call_type"] == "consultingCall"
    assert 0 <= body["score"] <= 100
    assert len(body["contributions"]) == 9
    # The commitment with "by Friday" should fire next_step_booked.
    next_step = next(c for c in body["contributions"] if c["name"] == "next_step_booked")
    assert next_step["raw_value"] == 1.0


def test_score_unknown_call_returns_404(app_client: TestClient) -> None:
    response = app_client.get("/score?call_id=never_ingested")
    assert response.status_code == 404


def test_score_accepts_primary_win_query(app_client: TestClient, vault_with_call) -> None:
    _, call_id = vault_with_call
    response = app_client.get(f"/score?call_id={call_id}&primary_win=ship%20the%20ad-bot")
    assert response.status_code == 200


# ────────────────────────────────────────────────────────────────────────────
# POST /score_override
# ────────────────────────────────────────────────────────────────────────────


def test_score_override_writes_correction(app_client: TestClient, vault_with_call) -> None:
    vault_root, call_id = vault_with_call
    response = app_client.post(
        "/score_override",
        json={"call_id": call_id, "user_score": 88.0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["call_id"] == call_id
    assert body["user_score"] == 88.0
    assert body["corrections_count"] == 1

    # File written to disk + has the correction.
    path = corrections_path(vault_root)
    assert path.exists()
    rows = load_corrections(vault_root)
    assert len(rows) == 1
    assert rows[0].user_score == 88.0
    assert rows[0].call_id == call_id
    # All 9 features captured.
    assert len(rows[0].features) == 9


def test_score_override_unknown_call_returns_404(app_client: TestClient) -> None:
    response = app_client.post(
        "/score_override",
        json={"call_id": "never_ingested", "user_score": 75.0},
    )
    assert response.status_code == 404


def test_score_override_validates_score_range(app_client: TestClient, vault_with_call) -> None:
    _, call_id = vault_with_call
    too_high = app_client.post("/score_override", json={"call_id": call_id, "user_score": 150.0})
    too_low = app_client.post("/score_override", json={"call_id": call_id, "user_score": -5.0})
    assert too_high.status_code == 422
    assert too_low.status_code == 422


def test_multiple_overrides_accumulate(app_client: TestClient, vault_with_call) -> None:
    vault_root, call_id = vault_with_call
    for user_score in (60, 70, 85):
        app_client.post("/score_override", json={"call_id": call_id, "user_score": user_score})
    rows = load_corrections(vault_root)
    assert len(rows) == 3
    assert [r.user_score for r in rows] == [60.0, 70.0, 85.0]
