"""Phase 11 — evaluation-loop instrumentation tests.

Covers three contracts:

  1. `log_feedback()` appends a parseable row for each of the
     FEEDBACK_KINDS and rejects unknown kinds.
  2. `compute_acceptance()` aggregates the new per-kind counts and
     produces helpful_rate / used_rate / etc. denominated against
     total_emits.
  3. `/suggestion_feedback` HTTP endpoint validates the kind, persists
     the event, and surfaces in /metrics afterwards.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consultant_brain.evaluation.acceptance import compute_acceptance
from consultant_brain.evaluation.suggestion_log import (
    FEEDBACK_KINDS,
    SUGGESTION_LOG_FILENAME,
    load_events,
    log_emit,
    log_feedback,
    log_referenced,
)
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


@pytest.fixture
def app_client(vault_root: Path) -> TestClient:
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    return TestClient(app)


# ────────────────────────────────────────────────────────────────────────────
# log_feedback()
# ────────────────────────────────────────────────────────────────────────────


def test_log_feedback_appends_one_event_per_kind(vault_root: Path) -> None:
    for kind in FEEDBACK_KINDS:
        log_feedback(
            vault_root=vault_root,
            call_id="CALL1",
            atom_id=f"ATOM_{kind.upper()}",
            kind=kind,
        )
    log_path = vault_root / "00_System" / SUGGESTION_LOG_FILENAME
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    kinds_in_log = [r["kind"] for r in rows]
    assert sorted(kinds_in_log) == sorted(FEEDBACK_KINDS)
    # layer + score absent on feedback events.
    for row in rows:
        assert row["layer"] is None
        assert row["score"] is None


def test_log_feedback_rejects_unknown_kind(vault_root: Path) -> None:
    with pytest.raises(ValueError) as exc:
        log_feedback(
            vault_root=vault_root,
            call_id="CALL1",
            atom_id="ATOM",
            kind="loved_it_so_much",
        )
    assert "loved_it_so_much" in str(exc.value)


def test_load_events_roundtrip_new_kinds(vault_root: Path) -> None:
    log_emit(
        vault_root=vault_root,
        call_id="CALL1",
        atom_id="ATOM",
        layer="hot",
        score=0.91,
    )
    log_feedback(
        vault_root=vault_root,
        call_id="CALL1",
        atom_id="ATOM",
        kind="helpful",
    )
    events = load_events(vault_root)
    assert {e.kind for e in events} == {"emit", "helpful"}


# ────────────────────────────────────────────────────────────────────────────
# compute_acceptance() — aggregation
# ────────────────────────────────────────────────────────────────────────────


def test_acceptance_aggregates_all_feedback_kinds(vault_root: Path) -> None:
    # Three emits, each with a different feedback signal.
    # Mix in one legacy `referenced` event so the original metric still
    # reads cleanly.
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A_HELP", layer="hot", score=0.9)
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A_BAD", layer="warm", score=0.6)
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A_REF", layer="hot", score=0.7)
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A_DROP", layer="warm", score=0.5)

    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A_HELP", kind="helpful")
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A_BAD", kind="useless")
    log_referenced(vault_root=vault_root, call_id="C1", atom_id="A_REF")
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A_DROP", kind="dismissed")
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A_HELP", kind="used_in_call")

    report = compute_acceptance(vault_root=vault_root)
    assert report.total_emits == 4
    assert report.helpful_count == 1
    assert report.useless_count == 1
    assert report.dismissed_count == 1
    assert report.used_count == 1
    assert report.total_referenced == 1  # legacy ref signal still works
    assert report.helpful_rate == pytest.approx(0.25)
    assert report.useless_rate == pytest.approx(0.25)
    assert report.dismissed_rate == pytest.approx(0.25)
    assert report.used_rate == pytest.approx(0.25)
    assert report.acceptance_rate == pytest.approx(0.25)


def test_acceptance_dedupes_double_thumbs(vault_root: Path) -> None:
    """A user tapping thumbs-up twice on the same atom should still
    count as one helpful event so helpful_rate ≤ 1."""
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A", layer="hot", score=0.9)
    for _ in range(5):
        log_feedback(vault_root=vault_root, call_id="C1", atom_id="A", kind="helpful")

    report = compute_acceptance(vault_root=vault_root)
    assert report.helpful_count == 1
    assert report.helpful_rate == pytest.approx(1.0)


def test_acceptance_ignores_feedback_without_emit(vault_root: Path) -> None:
    """Feedback events whose (call_id, atom_id) pair never saw an emit
    are dropped — they'd otherwise produce a rate > 1 with no
    matching denominator."""
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A1", layer="hot", score=0.9)
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A1", kind="helpful")
    # Orphan feedback — no emit for this pair.
    log_feedback(vault_root=vault_root, call_id="C2", atom_id="A_GHOST", kind="helpful")

    report = compute_acceptance(vault_root=vault_root)
    assert report.total_emits == 1
    assert report.helpful_count == 1
    assert report.helpful_rate == pytest.approx(1.0)


# ────────────────────────────────────────────────────────────────────────────
# /suggestion_feedback endpoint
# ────────────────────────────────────────────────────────────────────────────


def test_suggestion_feedback_endpoint_logs_event(app_client: TestClient, vault_root: Path) -> None:
    # Seed an emit so the metric has a denominator.
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A1", layer="hot", score=0.9)

    response = app_client.post(
        "/suggestion_feedback",
        json={"call_id": "C1", "atom_id": "A1", "kind": "helpful"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"call_id": "C1", "atom_id": "A1", "kind": "helpful", "logged": True}

    events = load_events(vault_root)
    helpful_rows = [e for e in events if e.kind == "helpful"]
    assert len(helpful_rows) == 1
    assert helpful_rows[0].atom_id == "A1"


def test_suggestion_feedback_endpoint_rejects_unknown_kind(app_client: TestClient) -> None:
    response = app_client.post(
        "/suggestion_feedback",
        json={"call_id": "C1", "atom_id": "A1", "kind": "delightful"},
    )
    assert response.status_code == 400
    assert "delightful" in response.json()["detail"]


def test_suggestion_feedback_each_kind_surfaces_in_metrics(
    app_client: TestClient, vault_root: Path
) -> None:
    """End-to-end: post each feedback kind through the endpoint, then
    GET /metrics and confirm the rates show up."""
    for i, kind in enumerate(FEEDBACK_KINDS):
        atom_id = f"ATOM_{i}"
        log_emit(vault_root=vault_root, call_id="C1", atom_id=atom_id, layer="hot", score=0.9)
        post = app_client.post(
            "/suggestion_feedback",
            json={"call_id": "C1", "atom_id": atom_id, "kind": kind},
        )
        assert post.status_code == 200, post.text

    metrics = app_client.get("/metrics").json()
    accept = metrics["acceptance"]
    assert accept is not None
    assert accept["helpful_count"] == 1
    assert accept["useless_count"] == 1
    assert accept["dismissed_count"] == 1
    assert accept["used_count"] == 1
