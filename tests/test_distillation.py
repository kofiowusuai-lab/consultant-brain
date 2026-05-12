"""Phase 6 distillation tests: pattern mining, cold-layer retrieval,
client context regeneration, distill CLI.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter
import pytest
from typer.testing import CliRunner

from consultant_brain.cli import app
from consultant_brain.distillation.client_context import (
    PIN_PATTERN,
    regenerate_all_clients,
    regenerate_client_context,
)
from consultant_brain.distillation.cold_layer import (
    PatternHit,
    find_matching_patterns,
    pattern_hit_as_atom_hit,
)
from consultant_brain.distillation.patterns import (
    MIN_OBSERVATIONS,
    mine_patterns,
)
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
)
from consultant_brain.vault import (
    VaultLayout,
    ensure_client_folder,
    ensure_vault_skeleton,
    write_atom,
)


def _atom(
    *,
    id_: str,
    type_: AtomType,
    body: str,
    tags: tuple[str, ...] = (),
    client: str = "Reece",
    call: str = "2026-05-12_reece_consultingCall",
    status: AtomStatus = AtomStatus.active,
) -> Atom:
    return Atom(
        id=id_,
        type=type_,
        client=client,
        call=call,
        call_type=CallType.consulting_call,
        tags=list(tags),
        confidence=0.85,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc),
        status=status,
        embedding_id=id_,
        body=body,
    )


def _vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(root))
    return root


# ────────────────────────────────────────────────────────────────────────────
# Pattern miner
# ────────────────────────────────────────────────────────────────────────────


def test_mine_patterns_promotes_when_three_distinct_calls(tmp_path: Path) -> None:
    """3 objection atoms tagged 'budget' across 3 different calls → 1 pattern."""
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    for i in range(3):
        write_atom(layout, _atom(
            id_=f"BUDGETAAAAAAAAAAAAAAAA{i:04d}",
            type_=AtomType.objection,
            body=f"Price came up early ({i}).",
            tags=("budget",),
            call=f"2026-05-1{i+1}_reece_consultingCall",
        ))
    result = mine_patterns(vault_root=vault_root)
    assert result.patterns_written == 1
    assert result.patterns_updated == 0
    pattern_path = vault_root / "04_Patterns" / "objection__budget.md"
    assert pattern_path.exists()
    post = frontmatter.load(pattern_path.open("r", encoding="utf-8"))
    assert post.metadata["observation_count"] == 3
    assert post.metadata["unique_call_count"] == 3
    assert post.metadata["primary_tag"] == "budget"


def test_mine_patterns_skips_when_same_call_only(tmp_path: Path) -> None:
    """3 atoms from the SAME call don't qualify — needs distinct calls.
    Master prompt's promotion threshold is observation-based, but
    redundancy within one call shouldn't qualify; we count unique calls."""
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    for i in range(3):
        write_atom(layout, _atom(
            id_=f"DUPECALLAAAAAAAAAAAAAA{i:04d}",
            type_=AtomType.objection,
            body=f"price came up ({i}).",
            tags=("budget",),
            call="2026-05-12_reece_consultingCall",  # SAME call
        ))
    result = mine_patterns(vault_root=vault_root)
    assert result.patterns_written == 0


def test_mine_patterns_re_run_updates_in_place(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    for i in range(3):
        write_atom(layout, _atom(
            id_=f"IDEMAAAAAAAAAAAAAAAAAA{i:04d}",
            type_=AtomType.commitment,
            body=f"Will ship by Friday ({i}).",
            tags=("next_step",),
            call=f"2026-05-1{i+1}_reece_consultingCall",
        ))
    first = mine_patterns(vault_root=vault_root)
    assert first.patterns_written == 1
    # Add a 4th observation across a 4th call.
    write_atom(layout, _atom(
        id_="IDEMAAAAAAAAAAAAAAAAAA0099",
        type_=AtomType.commitment,
        body="Will deliver by Wednesday.",
        tags=("next_step",),
        call="2026-05-15_reece_consultingCall",
    ))
    second = mine_patterns(vault_root=vault_root)
    assert second.patterns_written == 0
    assert second.patterns_updated == 1
    pattern_path = vault_root / "04_Patterns" / "commitment__next_step.md"
    post = frontmatter.load(pattern_path.open("r", encoding="utf-8"))
    assert post.metadata["observation_count"] == 4


def test_mine_patterns_skips_retired_atoms(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    for i in range(3):
        write_atom(layout, _atom(
            id_=f"DEADAAAAAAAAAAAAAAAAAA{i:04d}",
            type_=AtomType.objection,
            body=f"old.",
            tags=("legacy",),
            call=f"2026-05-1{i+1}_x_consultingCall",
            status=AtomStatus.retired,
        ))
    result = mine_patterns(vault_root=vault_root)
    assert result.patterns_written == 0


# ────────────────────────────────────────────────────────────────────────────
# Cold-layer pattern retrieval
# ────────────────────────────────────────────────────────────────────────────


def test_find_matching_patterns_fires_on_primary_tag_token(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    for i in range(3):
        write_atom(layout, _atom(
            id_=f"PATAAAAAAAAAAAAAAAAAA{i:04d}",
            type_=AtomType.objection,
            body=f"Price hit early.",
            tags=("budget", "anchor_risk"),
            call=f"2026-05-1{i+1}_reece_consultingCall",
        ))
    mine_patterns(vault_root=vault_root)
    hits = find_matching_patterns(
        vault_root=vault_root,
        window="You: what's the budget look like?",
    )
    assert hits, "expected at least one pattern hit"
    assert any(h.primary_tag == "budget" for h in hits)


def test_find_matching_patterns_returns_empty_without_patterns(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    hits = find_matching_patterns(
        vault_root=vault_root,
        window="anything goes here",
    )
    assert hits == []


def test_pattern_hit_as_atom_hit_round_trips_similarity() -> None:
    pattern_hit = PatternHit(
        pattern_id="objection__budget",
        atom_type="objection",
        primary_tag="budget",
        observation_count=4,
        score_impact=0.82,
        matched_phrase="budget",
        summary="Price came up before scope.",
    )
    atom_hit = pattern_hit_as_atom_hit(pattern_hit)
    # similarity should round-trip back to score_impact (within float epsilon)
    assert abs(atom_hit.similarity - 0.82) < 0.01


# ────────────────────────────────────────────────────────────────────────────
# Client context regeneration
# ────────────────────────────────────────────────────────────────────────────


def test_regenerate_client_context_writes_context_md(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    ensure_client_folder(layout, "Reece")
    write_atom(layout, _atom(
        id_="CTXAAAAAAAAAAAAAAAAAA0001",
        type_=AtomType.objection,
        body="Price came up before scope.",
        tags=("budget",),
    ))
    write_atom(layout, _atom(
        id_="CTXBBBBBBBBBBBBBBBBBB0002",
        type_=AtomType.commitment,
        body="Reece will send ads by Friday.",
        tags=("next_step",),
    ))
    result = regenerate_client_context(vault_root=vault_root, client_name="Reece")
    assert result is not None
    assert result.atom_count == 2
    text = result.context_path.read_text(encoding="utf-8")
    assert "# Reece" in text
    assert "## Atoms by type" in text
    assert "objection" in text
    assert "commitment" in text


def test_regenerate_preserves_pinned_block(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    ensure_client_folder(layout, "Reece")
    write_atom(layout, _atom(
        id_="PIN_AAAAAAAAAAAAAAAAAA0001",
        type_=AtomType.objection,
        body="x",
        tags=("budget",),
    ))
    # First pass — writes context.md with the default empty pin section.
    regenerate_client_context(vault_root=vault_root, client_name="Reece")
    context_path = layout.client_dir("reece") / "context.md"
    # User hand-edits the pin section.
    existing = context_path.read_text(encoding="utf-8")
    edited = existing.replace(
        "_Pinned notes go here. This block is preserved across regenerations._",
        "Reece prefers Mondays. Cash flow problems Q2.",
    )
    context_path.write_text(edited)
    # Second pass — must preserve the pinned block.
    regenerate_client_context(vault_root=vault_root, client_name="Reece")
    after = context_path.read_text(encoding="utf-8")
    assert "Reece prefers Mondays. Cash flow problems Q2." in after


def test_regenerate_all_clients_walks_directory(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    ensure_client_folder(layout, "Reece")
    ensure_client_folder(layout, "Acme Industries")
    write_atom(layout, _atom(id_="R01AAAAAAAAAAAAAAAAAAAAAA", type_=AtomType.objection, body="x", tags=("a",), client="Reece"))
    write_atom(layout, _atom(id_="A01AAAAAAAAAAAAAAAAAAAAAA", type_=AtomType.commitment, body="y", tags=("b",), client="Acme Industries"))
    results = regenerate_all_clients(vault_root=vault_root)
    slugs = {r.client_slug for r in results}
    assert "reece" in slugs
    assert "acme_industries" in slugs


def test_regenerate_returns_none_when_client_has_no_atoms(tmp_path: Path) -> None:
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    ensure_client_folder(layout, "Empty")
    result = regenerate_client_context(vault_root=vault_root, client_name="Empty")
    assert result is None


# ────────────────────────────────────────────────────────────────────────────
# distill CLI
# ────────────────────────────────────────────────────────────────────────────


def test_distill_cli_runs_clean_on_empty_vault(tmp_path: Path) -> None:
    runner = CliRunner()
    vault_root = _vault(tmp_path)
    result = runner.invoke(app, ["distill", "--vault", str(vault_root)])
    assert result.exit_code == 0
    assert "Patterns" in result.output
    assert "Context" in result.output


def test_distill_cli_writes_pattern_and_context(tmp_path: Path) -> None:
    runner = CliRunner()
    vault_root = _vault(tmp_path)
    layout = VaultLayout.for_root(vault_root)
    ensure_client_folder(layout, "Reece")
    for i in range(3):
        write_atom(layout, _atom(
            id_=f"DISTILLAAAAAAAAAAAAA{i:06d}",
            type_=AtomType.objection,
            body=f"Price hit early ({i}).",
            tags=("budget",),
            call=f"2026-05-1{i+1}_reece_consultingCall",
        ))
    result = runner.invoke(app, ["distill", "--vault", str(vault_root)])
    assert result.exit_code == 0, result.output
    assert (vault_root / "04_Patterns" / "objection__budget.md").exists()
    assert (vault_root / "01_Clients" / "reece" / "context.md").exists()
