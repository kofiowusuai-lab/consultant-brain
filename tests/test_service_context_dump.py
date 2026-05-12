"""FastAPI integration tests for the Phase 10 /context_dump endpoints.

Uses FastAPI's TestClient + monkeypatches the orchestrator's LLM
provider lookup so the extractor stays offline.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consultant_brain.context_dumps import orchestrator as _orchestrator
from consultant_brain.llm.provider import ChatRequest, ChatResponse
from consultant_brain.service import create_app


_OK_PAYLOAD = (
    '{"summary":"Coffee at Verve.","atoms":['
    '{"type":"commitment","body":"Reece commits to Q3.","confidence":0.9,"tags":["timeline"]},'
    '{"type":"insight","body":"Anchors on timeline before scope.","confidence":0.7,"tags":["framing"]}'
    "]}"
)


class _FakeProvider:
    name = "fake"

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(text=_OK_PAYLOAD, model_used=request.model)


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A fresh FastAPI app per test with a tmp vault + a fake LLM."""
    # Force the extractor cache + provider lookup to return our fake.
    monkeypatch.setattr(
        "consultant_brain.service._get_extractor_provider",
        lambda app: _FakeProvider(),
    )
    # Short TTL so the eviction test runs without sleeping a real 30 min.
    monkeypatch.setenv("CONSULTANT_BRAIN_CONTEXT_DUMP_TTL_SECONDS", "30")
    app = create_app(vault_root=tmp_path / "vault")
    return TestClient(app)


def _upload(client: TestClient, text: str = "Coffee with Reece, talked about Q3.") -> dict:
    return client.post(
        "/context_dump",
        data={
            "client_name": "Reece",
            "observed_at": "2026-05-12",
            "notes": "25-min coffee",
        },
        files={"file": ("note.txt", text.encode("utf-8"), "text/plain")},
    ).json()


def test_upload_returns_preview_without_vault_write(app_client: TestClient, tmp_path: Path) -> None:
    response = app_client.post(
        "/context_dump",
        data={
            "client_name": "Reece",
            "observed_at": "2026-05-12",
            "notes": "25-min coffee",
        },
        files={"file": ("note.txt", b"Coffee with Reece.", "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["client_name"] == "Reece"
    assert payload["client_slug"] == "reece"
    assert payload["observed_at"] == "2026-05-12"
    assert len(payload["atoms"]) == 2
    assert payload["summary"]
    # Atoms[*].index lets the client send back a subset on commit.
    assert payload["atoms"][0]["index"] == 0
    # Vault is untouched.
    assert not (tmp_path / "vault" / "03_Atoms").exists()
    assert not (tmp_path / "vault" / "10_ContextDumps").exists()


def test_commit_writes_atoms_and_dump_note(app_client: TestClient, tmp_path: Path) -> None:
    upload = _upload(app_client)
    preview_id = upload["preview_id"]

    commit = app_client.post(
        "/context_dump/commit",
        json={"preview_id": preview_id},
    )
    assert commit.status_code == 200
    body = commit.json()
    assert body["atom_count"] == 2
    assert body["client_slug"] == "reece"
    # Vault now has the dump note + atoms.
    assert Path(body["dump_note_path"]).exists()
    atoms = list((tmp_path / "vault" / "03_Atoms").glob("*.md"))
    assert len(atoms) == 2


def test_commit_with_subset_keeps_only_chosen(app_client: TestClient, tmp_path: Path) -> None:
    upload = _upload(app_client)
    preview_id = upload["preview_id"]

    commit = app_client.post(
        "/context_dump/commit",
        json={"preview_id": preview_id, "accepted_atom_indexes": [0]},
    )
    assert commit.status_code == 200
    assert commit.json()["atom_count"] == 1


def test_get_preview_returns_404_after_commit(app_client: TestClient) -> None:
    upload = _upload(app_client)
    pid = upload["preview_id"]
    assert app_client.get(f"/context_dump/{pid}").status_code == 200
    app_client.post("/context_dump/commit", json={"preview_id": pid})
    assert app_client.get(f"/context_dump/{pid}").status_code == 404


def test_discard_evicts_preview(app_client: TestClient) -> None:
    upload = _upload(app_client)
    pid = upload["preview_id"]
    resp = app_client.delete(f"/context_dump/{pid}")
    assert resp.status_code == 200
    assert resp.json()["discarded"] is True
    assert app_client.get(f"/context_dump/{pid}").status_code == 404


def test_upload_rejects_bad_date(app_client: TestClient) -> None:
    response = app_client.post(
        "/context_dump",
        data={
            "client_name": "Reece",
            "observed_at": "not-a-date",
        },
        files={"file": ("note.txt", b"x", "text/plain")},
    )
    assert response.status_code == 400


def test_commit_404_when_preview_unknown(app_client: TestClient) -> None:
    resp = app_client.post(
        "/context_dump/commit",
        json={"preview_id": "01HXNOTREAL00000000000000"},
    )
    assert resp.status_code == 404
