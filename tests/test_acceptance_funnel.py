"""Phase 12 — exposure-funnel acceptance tests.

Covers:
  - shown / hidden / expanded / copied / followup_created round-trip
    through `log_feedback`
  - compute_acceptance aggregates the new counts
  - rate denominators compose correctly (shown / emits, then expand /
    shown, copy / shown, followup / shown)
  - hidden tracked but not surfaced as a rate (sanity only)
  - exposure events for atoms that were never emitted are dropped
"""

from __future__ import annotations

from pathlib import Path

import pytest

from consultant_brain.evaluation.acceptance import compute_acceptance
from consultant_brain.evaluation.suggestion_log import (
    FEEDBACK_KINDS,
    log_emit,
    log_feedback,
)
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


def test_feedback_kinds_include_funnel_events() -> None:
    """All five Phase 12 kinds are in FEEDBACK_KINDS so the
    /suggestion_feedback endpoint accepts them without a 400."""
    for kind in ("shown", "hidden", "expanded", "copied", "followup_created"):
        assert kind in FEEDBACK_KINDS


def test_funnel_rates_compose(vault_root: Path) -> None:
    """10 emits, 8 shown, 4 expanded, 2 copied, 1 followup_created.
    Expected: shown_rate 0.8, expand 0.5, copy 0.25, followup 0.125."""
    for i in range(10):
        log_emit(
            vault_root=vault_root,
            call_id="C1",
            atom_id=f"A{i}",
            layer="hot",
            score=0.5,
        )
    for i in range(8):
        log_feedback(vault_root=vault_root, call_id="C1", atom_id=f"A{i}", kind="shown")
    for i in range(4):
        log_feedback(vault_root=vault_root, call_id="C1", atom_id=f"A{i}", kind="expanded")
    for i in range(2):
        log_feedback(vault_root=vault_root, call_id="C1", atom_id=f"A{i}", kind="copied")
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A0", kind="followup_created")

    report = compute_acceptance(vault_root=vault_root)
    assert report.total_emits == 10
    assert report.shown_count == 8
    assert report.expanded_count == 4
    assert report.copied_count == 2
    assert report.followup_count == 1
    assert report.shown_rate == pytest.approx(0.8)
    assert report.expand_rate == pytest.approx(0.5)
    assert report.copy_rate == pytest.approx(0.25)
    assert report.followup_rate == pytest.approx(0.125)


def test_hidden_tracked_but_no_rate(vault_root: Path) -> None:
    """hidden is recorded but doesn't get its own published rate.
    It exists so a future "shown ≈ hidden" sanity check has data."""
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A1", layer="hot", score=0.5)
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A1", kind="shown")
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A1", kind="hidden")

    report = compute_acceptance(vault_root=vault_root)
    assert report.shown_count == 1
    assert report.hidden_count == 1


def test_funnel_drops_orphan_events(vault_root: Path) -> None:
    """A `shown` event for an atom that never had an `emit` row
    must NOT count toward shown_count — otherwise shown_rate could
    exceed 1.0 in pathological cases."""
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A1", layer="hot", score=0.5)
    log_feedback(vault_root=vault_root, call_id="C1", atom_id="A1", kind="shown")
    # Orphan: no emit for this pair.
    log_feedback(vault_root=vault_root, call_id="C9", atom_id="A_GHOST", kind="shown")

    report = compute_acceptance(vault_root=vault_root)
    assert report.total_emits == 1
    assert report.shown_count == 1
    assert report.shown_rate == pytest.approx(1.0)


def test_funnel_dedupes_repeat_events(vault_root: Path) -> None:
    """SwiftUI re-renders can fire .onAppear multiple times. A
    deduplicating Set in the aggregator keeps the funnel honest."""
    log_emit(vault_root=vault_root, call_id="C1", atom_id="A1", layer="hot", score=0.5)
    for _ in range(7):
        log_feedback(vault_root=vault_root, call_id="C1", atom_id="A1", kind="shown")
    for _ in range(3):
        log_feedback(vault_root=vault_root, call_id="C1", atom_id="A1", kind="expanded")

    report = compute_acceptance(vault_root=vault_root)
    assert report.shown_count == 1
    assert report.expanded_count == 1
    assert report.expand_rate == pytest.approx(1.0)


def test_funnel_empty_log_returns_zero_rates(vault_root: Path) -> None:
    """A fresh vault produces zero rates without raising."""
    report = compute_acceptance(vault_root=vault_root)
    assert report.shown_count == 0
    assert report.shown_rate == 0.0
    assert report.expand_rate == 0.0
    assert report.copy_rate == 0.0
    assert report.followup_rate == 0.0
