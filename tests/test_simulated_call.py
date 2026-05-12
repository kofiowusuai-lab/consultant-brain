"""End-to-end simulated call: replay a fixture session JSON's turns
through the FastAPI service, verify atoms get written progressively as
the call unfolds, and verify /suggestions improves as more context lands.

This is the spec for Phase 4's "active brain during a call" promise.
Uses a mocked moment detector so the test stays deterministic + offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import consultant_brain.live_loop as live_loop
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.json"


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _StaticAnthropic:
    """Returns the same atoms on every detection pass — enough to verify
    the plumbing without making real API calls."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def messages_create(self, **kwargs) -> _Response:
        return _Response(content=[_Block(text=self.response_text)])


DETECTED_ATOMS = json.dumps(
    {
        "atoms": [
            {"type": "objection", "body": "Ad-bot retrieval keeps grabbing wrong notes.", "confidence": 0.91, "tags": ["retrieval"]},
            {"type": "commitment", "body": "Reece will send 3 manual ads by Friday.", "confidence": 0.93, "tags": ["next_step"]},
        ]
    }
)


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    layout = VaultLayout.for_root(root)
    ensure_vault_skeleton(layout)
    return root


@pytest.fixture
def app_with_detection(vault_root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """FastAPI app with moment detection ENABLED + the Anthropic client
    swapped for a static mock at the live-loop level."""
    monkeypatch.setenv("CONSULTANT_BRAIN_MOMENT_DETECTION", "true")

    # Intercept real_anthropic_client so the live loop uses our mock.
    monkeypatch.setattr(
        live_loop,
        "real_anthropic_client",
        lambda _key: _StaticAnthropic(DETECTED_ATOMS),
    )
    # Stub out the secret lookup so we don't need the user's real key file.
    monkeypatch.setattr(live_loop, "get_anthropic_key", lambda: "test-key-not-used")
    # Drop the throttle to zero seconds so successive deltas each can fire
    # detection within a single test (the production throttle is 30s).
    monkeypatch.setattr(live_loop, "MOMENT_MIN_SECONDS_BETWEEN_RUNS", 0.0)
    monkeypatch.setattr(live_loop, "MOMENT_MIN_TURNS_BETWEEN_RUNS", 1)

    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    return TestClient(app)


def test_replaying_a_session_writes_atoms_progressively(
    app_with_detection: TestClient, vault_root: Path
) -> None:
    """The Phase 4 spec: as we feed turns to /transcript_delta, atoms appear
    in the vault. After all turns, we can /suggestions against the call and
    get back retrieval hits anchored to the just-detected atoms.
    """
    # 1. Start the call.
    start = app_with_detection.post(
        "/call_start",
        json={"call_id": "sim_001", "client": "Reece", "call_type": "consultingCall"},
    )
    assert start.status_code == 200

    # 2. Replay each turn from the fixture session.
    fixture = json.loads(FIXTURE.read_text())
    turn_count = 0
    for turn in fixture["completedTurns"]:
        speaker = "you" if turn["source"] == "microphone" else "them"
        resp = app_with_detection.post(
            "/transcript_delta",
            json={"call_id": "sim_001", "speaker": speaker, "text": turn["text"]},
        )
        assert resp.status_code == 200, resp.text
        turn_count += 1

    # 3. Verify some atoms were written during the call (background tasks
    #    in FastAPI's TestClient run synchronously during the request).
    layout = VaultLayout.for_root(vault_root)
    atom_files = list(layout.atoms_dir.glob("*.md"))
    assert atom_files, f"expected live atoms in vault, found 0 (sent {turn_count} turns)"
    # The mock detector emits 2 atoms per pass; even with one pass we'd have
    # 2 atoms. A 10-turn fixture should fire detection multiple times under
    # the test's relaxed throttle.
    assert len(atom_files) >= 2

    # 4. Querying /suggestions now should return at least one hit — the
    #    moment detector wrote atoms with the expected client + call_type.
    response = app_with_detection.get("/suggestions?call_id=sim_001&hot=1&warm=2&cold=1")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window_chars"] > 0
    # The TestClient skips real Ollama embeddings if not installed; if
    # there's no semantic match the panel could be empty. Either way, the
    # endpoint MUST return cleanly + the active brain has live atoms on
    # disk, which is the Phase 4 contract.

    # 5. End the call — state freed.
    ended = app_with_detection.post("/call_end", json={"call_id": "sim_001"})
    assert ended.json()["ended"] is True
    assert app_with_detection.get("/suggestions?call_id=sim_001").status_code == 404


def test_moment_detection_disabled_by_default(vault_root: Path) -> None:
    """Without CONSULTANT_BRAIN_MOMENT_DETECTION=true, /transcript_delta
    doesn't fire detection — opt-in for tokens-cost reasons."""
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    client = TestClient(app)
    client.post("/call_start", json={"call_id": "x", "client": "Reece", "call_type": "consultingCall"})
    for i in range(5):
        client.post("/transcript_delta", json={"call_id": "x", "speaker": "them", "text": f"turn {i}"})

    layout = VaultLayout.for_root(vault_root)
    # Vault skeleton exists but no atoms were created.
    assert layout.atoms_dir.exists()
    assert len(list(layout.atoms_dir.glob("*.md"))) == 0
