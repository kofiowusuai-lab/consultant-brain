"""Phase 7 evaluation tests: suggestion log, precision corpus, acceptance
rate, score MAE, pattern stability, unified metrics CLI.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from consultant_brain.cli import app
from consultant_brain.evaluation.acceptance import compute_acceptance
from consultant_brain.evaluation.metrics import (
    compute_metrics,
    render_report_text,
)
from consultant_brain.evaluation.pattern_stability import (
    PatternSnapshotRow,
    compute_pattern_stability,
    load_snapshots,
    snapshot_path,
    take_snapshot,
)
from consultant_brain.evaluation.precision import compute_precision
from consultant_brain.evaluation.score_mae import (
    ACCEPTABLE_DELTA,
    compute_score_mae,
)
from consultant_brain.evaluation.suggestion_log import (
    load_events,
    log_emit,
    log_referenced,
)
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.scoring.corrections import (
    ScoreCorrection,
    append_correction,
)
from consultant_brain.scoring.features import FEATURE_NAMES
from consultant_brain.scoring.weights import DEFAULT_BIAS
from consultant_brain.service import create_app
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton, write_atom
from consultant_brain.schemas import Atom, AtomStatus, AtomType, CallType


def _vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


# ────────────────────────────────────────────────────────────────────────────
# Suggestion log
# ────────────────────────────────────────────────────────────────────────────


def test_suggestion_log_round_trip(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    log_emit(vault_root=vault_root, call_id="c1", atom_id="a1", layer="hot", score=0.85)
    log_referenced(vault_root=vault_root, call_id="c1", atom_id="a1")
    log_emit(vault_root=vault_root, call_id="c1", atom_id="a2", layer="warm", score=0.7)
    events = load_events(vault_root)
    assert len(events) == 3
    assert events[0].kind == "emit"
    assert events[0].layer == "hot"
    assert events[1].kind == "referenced"
    assert events[1].layer is None


def test_suggestion_log_skips_malformed(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    log_emit(vault_root=vault_root, call_id="c1", atom_id="a1", layer="hot", score=0.85)
    # Append a corrupt line manually.
    path = vault_root / "00_System" / "suggestion_log.jsonl"
    with path.open("a") as f:
        f.write("{not json}\n")
    log_emit(vault_root=vault_root, call_id="c1", atom_id="a2", layer="warm", score=0.6)
    events = load_events(vault_root)
    assert len(events) == 2  # corrupt line skipped


# ────────────────────────────────────────────────────────────────────────────
# /suggestion_referenced endpoint
# ────────────────────────────────────────────────────────────────────────────


def test_suggestion_referenced_endpoint_logs(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    appf = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    client = TestClient(appf)
    response = client.post(
        "/suggestion_referenced",
        json={"call_id": "c1", "atom_id": "a1"},
    )
    assert response.status_code == 200
    assert response.json()["logged"] is True
    events = load_events(vault_root)
    assert len(events) == 1
    assert events[0].kind == "referenced"


# ────────────────────────────────────────────────────────────────────────────
# Acceptance rate
# ────────────────────────────────────────────────────────────────────────────


def test_acceptance_rate_pairs_emit_with_referenced(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    t0 = datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc)
    # 3 emits, 2 referenced within window, 1 referenced outside window.
    log_emit(vault_root=vault_root, call_id="c1", atom_id="a1", layer="hot", score=0.9, now=t0)
    log_referenced(vault_root=vault_root, call_id="c1", atom_id="a1", now=t0 + timedelta(seconds=30))

    log_emit(vault_root=vault_root, call_id="c1", atom_id="a2", layer="warm", score=0.7, now=t0)
    log_referenced(vault_root=vault_root, call_id="c1", atom_id="a2", now=t0 + timedelta(seconds=60))

    log_emit(vault_root=vault_root, call_id="c1", atom_id="a3", layer="cold", score=0.5, now=t0)
    # Referenced 5 minutes later — outside the 90s window.
    log_referenced(vault_root=vault_root, call_id="c1", atom_id="a3", now=t0 + timedelta(minutes=5))

    report = compute_acceptance(vault_root=vault_root)
    assert report.total_emits == 3
    assert report.total_referenced == 2  # a1 + a2 only
    assert report.acceptance_rate == pytest.approx(2 / 3)
    # By-layer breakdown.
    assert report.by_layer_counts["hot"] == (1, 1)
    assert report.by_layer_counts["warm"] == (1, 1)
    assert report.by_layer_counts["cold"] == (0, 1)


def test_acceptance_empty_log_returns_zero(tmp_path: Path) -> None:
    report = compute_acceptance(vault_root=_vault(tmp_path))
    assert report.total_emits == 0
    assert report.acceptance_rate == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Precision corpus
# ────────────────────────────────────────────────────────────────────────────


def _atom(*, id_: str, body: str, tags: tuple[str, ...] = ("budget",), client: str = "Reece") -> Atom:
    return Atom(
        id=id_,
        type=AtomType.objection,
        client=client,
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


@requires_ollama
def test_precision_reports_per_query_breakdown(tmp_path: Path) -> None:
    """Build a tiny vault + corpus + verify precision computes."""
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    from consultant_brain.embedder import LanceVaultIndex
    index = LanceVaultIndex(layout)
    a1 = _atom(id_="PREC_BUDGETAAAAAAAAAAAAAAAAA1", body="Price came up before scope.")
    a2 = _atom(id_="PREC_RETRIEVALAAAAAAAAAAAAAAA2", body="Ad-bot grabs wrong notes.", tags=("retrieval",))
    write_atom(layout, a1)
    write_atom(layout, a2)
    index.upsert(a1)
    index.upsert(a2)

    corpus = {
        "queries": [
            {
                "name": "price",
                "window": "what does this cost",
                "client": "Reece",
                "call_type": "consultingCall",
                "relevant_atom_ids": ["PREC_BUDGETAAAAAAAAAAAAAAAAA1"],
            }
        ]
    }
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))

    report = compute_precision(vault_root=vault_root, corpus_path=corpus_path)
    assert report.corpus_size == 1
    # The price atom should win the top-1 slot.
    assert report.queries[0].p_at_1 == 1.0


def test_precision_empty_corpus_returns_zeroes(tmp_path: Path) -> None:
    corpus_path = tmp_path / "empty.json"
    corpus_path.write_text(json.dumps({"queries": []}))
    report = compute_precision(vault_root=_vault(tmp_path), corpus_path=corpus_path)
    assert report.corpus_size == 0
    assert report.mean_p_at_1 == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Score MAE
# ────────────────────────────────────────────────────────────────────────────


def _correction(*, call_type: str, predicted: float, user: float, idx: int = 0) -> ScoreCorrection:
    return ScoreCorrection(
        call_id=f"c{idx}",
        call_type=call_type,
        predicted_score=predicted,
        user_score=user,
        features={name: 0.0 for name in FEATURE_NAMES},
        bias=DEFAULT_BIAS,
        created_at="2026-05-12T17:00:00Z",
    )


def test_score_mae_basic(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    # 3 corrections: errors of 5, 15, 25 → MAE = 15, 1/3 within ±10.
    append_correction(vault_root=vault_root, correction=_correction(call_type="consultingCall", predicted=60, user=65, idx=1))
    append_correction(vault_root=vault_root, correction=_correction(call_type="consultingCall", predicted=60, user=75, idx=2))
    append_correction(vault_root=vault_root, correction=_correction(call_type="consultingCall", predicted=60, user=85, idx=3))
    report = compute_score_mae(vault_root=vault_root)
    assert report.sample_count == 3
    assert report.mean_absolute_error == pytest.approx(15.0)
    assert report.within_acceptable_delta == pytest.approx(1 / 3)
    by_type = report.by_call_type["consultingCall"]
    assert by_type[0] == 3  # n
    assert by_type[1] == pytest.approx(15.0)  # mae


def test_score_mae_empty_returns_insufficient(tmp_path: Path) -> None:
    report = compute_score_mae(vault_root=_vault(tmp_path))
    assert report.insufficient_data is True


# ────────────────────────────────────────────────────────────────────────────
# Pattern stability
# ────────────────────────────────────────────────────────────────────────────


def _seed_pattern(vault_root: Path, pattern_id: str, observation_count: int, status: str = "active") -> None:
    """Write a minimal pattern note so `take_snapshot` picks it up."""
    import yaml
    patterns_dir = vault_root / "04_Patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": pattern_id,
        "kind": "pattern",
        "type": "objection",
        "primary_tag": pattern_id.split("__")[-1],
        "observation_count": observation_count,
        "unique_call_count": observation_count,
        "score_impact_proxy": 0.82,
        "status": status,
        "member_atom_ids": [],
        "created_at": "2026-05-12T17:00:00Z",
        "updated_at": "2026-05-12T17:00:00Z",
    }
    yaml_block = yaml.safe_dump(meta, sort_keys=False).rstrip()
    (patterns_dir / f"{pattern_id}.md").write_text(f"---\n{yaml_block}\n---\n\nbody\n")


def test_stability_take_snapshot_and_compute(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    # T0: 2 patterns active
    _seed_pattern(vault_root, "objection__budget", observation_count=3)
    _seed_pattern(vault_root, "objection__scope", observation_count=3)
    take_snapshot(vault_root=vault_root, now=datetime(2026, 4, 1, tzinfo=timezone.utc))

    # T1 (35 days later): budget grew, scope retired
    _seed_pattern(vault_root, "objection__budget", observation_count=7)
    _seed_pattern(vault_root, "objection__scope", observation_count=3, status="retired")
    take_snapshot(vault_root=vault_root, now=datetime(2026, 5, 6, tzinfo=timezone.utc))

    snapshots = load_snapshots(vault_root)
    assert len(snapshots) == 2

    report = compute_pattern_stability(vault_root=vault_root)
    assert not report.insufficient_data
    # 1 of 2 patterns persisted (budget) → 0.5
    assert report.persistence_rate == 0.5
    # Budget grew by 4 observations.
    assert report.mean_observation_count_delta == 4.0


def test_stability_insufficient_with_one_snapshot(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    _seed_pattern(vault_root, "x", 3)
    take_snapshot(vault_root=vault_root)
    report = compute_pattern_stability(vault_root=vault_root)
    assert report.insufficient_data is True


# ────────────────────────────────────────────────────────────────────────────
# Unified metrics
# ────────────────────────────────────────────────────────────────────────────


def test_metrics_text_renders_missing_sources_as_dashes(tmp_path: Path) -> None:
    """An empty vault produces a sensible text report — no false positives."""
    report = compute_metrics(vault_root=_vault(tmp_path), corpus_path=None)
    text = render_report_text(report)
    # All four metrics row should render with a friendly '—' fallback.
    assert "Retrieval precision" in text
    assert "Suggestion acceptance" in text
    assert "Score prediction MAE" in text
    assert "Pattern stability" in text
    # No actual numerical lies.
    assert "0.0%" not in text
    assert "100%" not in text


def test_metrics_cli_runs_clean_on_empty_vault(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["metrics", "--vault", str(_vault(tmp_path))])
    assert result.exit_code == 0
    assert "Consultant Brain Metrics" in result.output


def test_metrics_cli_emits_json_with_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["metrics", "--vault", str(_vault(tmp_path)), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "generated_at" in payload
    # Each metric key present; values may be None for an empty vault.
    for key in ("precision", "acceptance", "score_mae", "pattern_stability"):
        assert key in payload
