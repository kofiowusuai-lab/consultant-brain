"""Tests for the Phase 8 distillation modules:
  - tags (normalize_tags)
  - retirement (run_retirement)
  - plays (mine_plays + cold_layer play matching)
  - people (regenerate_people)
  - reviews (generate_weekly_review)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from consultant_brain.distillation.cold_layer import find_matching_plays
from consultant_brain.distillation.people import regenerate_people
from consultant_brain.distillation.plays import mine_plays
from consultant_brain.distillation.retirement import run_retirement
from consultant_brain.distillation.reviews import generate_weekly_review
from consultant_brain.distillation.tags import (
    CANONICAL_ALIASES,
    normalize_tags,
)
from consultant_brain.schemas import AtomStatus, AtomType
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


def _write_atom(
    vault: Path,
    *,
    atom_id: str,
    atom_type: str = "client_fact",
    client: str = "Acme",
    call: str = "2026-05-12_acme_consultingCall",
    body: str = "Reece said the team uses Notion for ticketing.",
    tags: list[str] | None = None,
    last_seen: str = "2026-05-12",
    confidence: float = 0.8,
    status: str = "active",
) -> Path:
    """Drop a single atom markdown into 03_Atoms/ via raw YAML write so
    we don't have to drag the whole ingest pipeline in."""
    layout = VaultLayout.for_root(vault)
    ensure_vault_skeleton(layout)
    fm = {
        "id": atom_id,
        "type": atom_type,
        "client": client,
        "call": f"[[{call}]]",
        "call_type": "consultingCall",
        "tags": tags or [],
        "confidence": confidence,
        "evidence_count": 1,
        "last_seen": last_seen,
        "created_at": "2026-05-12T19:42:10Z",
        "status": status,
        "embedding_id": atom_id,
    }
    yaml_block = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{body}\n"
    path = layout.atoms_dir / f"{atom_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ────────────────────────────────────────────────────────────────────────────
# tags
# ────────────────────────────────────────────────────────────────────────────


def test_normalize_tags_folds_canonical_aliases(tmp_path: Path) -> None:
    _write_atom(tmp_path, atom_id="01HX0000000000000000000001", tags=["nextstep", "budget"])
    _write_atom(tmp_path, atom_id="01HX0000000000000000000002", tags=["followup", "scope"])
    _write_atom(tmp_path, atom_id="01HX0000000000000000000003", tags=["next_step", "scope-creep"])

    result = normalize_tags(vault_root=tmp_path)
    assert result.atoms_scanned == 3
    assert result.atoms_rewritten >= 2
    # Canonical aliases hit deterministically.
    assert "nextstep" in result.tag_replacements
    assert result.tag_replacements["nextstep"] == "next_step"
    assert result.tag_replacements["followup"] == "next_step"

    # Reload atoms and confirm tags rewritten on disk.
    import frontmatter as fm
    layout = VaultLayout.for_root(tmp_path)
    for path in layout.atoms_dir.glob("*.md"):
        meta = fm.load(path.open("r", encoding="utf-8")).metadata
        for tag in meta["tags"]:
            assert tag not in CANONICAL_ALIASES, f"alias {tag} survived"


def test_normalize_tags_is_idempotent(tmp_path: Path) -> None:
    _write_atom(tmp_path, atom_id="01HX0000000000000000000001", tags=["nextstep"])
    first = normalize_tags(vault_root=tmp_path)
    assert first.atoms_rewritten == 1
    second = normalize_tags(vault_root=tmp_path)
    assert second.atoms_rewritten == 0


def test_normalize_tags_levenshtein_merges_typos(tmp_path: Path) -> None:
    # Five atoms with the canonical spelling, one with a typo.
    for i in range(5):
        _write_atom(tmp_path, atom_id=f"01HX000000000000000000000{i}", tags=["timeline"])
    _write_atom(tmp_path, atom_id="01HX0000000000000000000099", tags=["timelinr"])
    result = normalize_tags(vault_root=tmp_path)
    assert result.atoms_rewritten == 1
    assert result.tag_replacements.get("timelinr") == "timeline"


# ────────────────────────────────────────────────────────────────────────────
# retirement
# ────────────────────────────────────────────────────────────────────────────


def test_retirement_flips_old_active_to_needs_review(tmp_path: Path) -> None:
    today = date(2026, 5, 12)
    long_ago = (today - timedelta(days=150)).isoformat()
    _write_atom(tmp_path, atom_id="01HX0000000000000000000001", last_seen=long_ago)
    _write_atom(tmp_path, atom_id="01HX0000000000000000000002", last_seen=today.isoformat())

    result = run_retirement(vault_root=tmp_path, today=today)
    assert result.atoms_scanned == 2
    assert result.flagged_needs_review == 1
    assert result.retired == 0


def test_retirement_flips_old_needs_review_to_retired(tmp_path: Path) -> None:
    today = date(2026, 5, 12)
    very_long_ago = (today - timedelta(days=200)).isoformat()
    _write_atom(
        tmp_path,
        atom_id="01HX0000000000000000000001",
        last_seen=very_long_ago,
        status="needs_review",
    )
    result = run_retirement(vault_root=tmp_path, today=today)
    assert result.retired == 1


def test_retirement_revives_recently_observed_needs_review(tmp_path: Path) -> None:
    today = date(2026, 5, 12)
    _write_atom(
        tmp_path,
        atom_id="01HX0000000000000000000001",
        last_seen=today.isoformat(),
        status="needs_review",
    )
    result = run_retirement(vault_root=tmp_path, today=today)
    assert result.revived == 1
    assert result.flagged_needs_review == 0


# ────────────────────────────────────────────────────────────────────────────
# plays
# ────────────────────────────────────────────────────────────────────────────


def test_mine_plays_promotes_cross_client_repeat(tmp_path: Path) -> None:
    body = "We need this to land before Q3 — budget hesitation is real."
    for i, client in enumerate(["Acme", "Beta Corp", "Gamma Co"]):
        _write_atom(
            tmp_path,
            atom_id=f"01HX000000000000000000000{i}",
            atom_type="objection",
            client=client,
            call=f"2026-05-{10+i:02d}_{client.lower().replace(' ', '_')}_consultingCall",
            body=body,
            tags=["budget"],
        )
    result = mine_plays(vault_root=tmp_path)
    assert result.plays_written >= 1
    plays_dir = tmp_path / "05_Plays"
    assert plays_dir.exists()
    md_files = list(plays_dir.glob("*.md"))
    assert md_files
    text = md_files[0].read_text(encoding="utf-8")
    assert "## Four-frame template" in text
    assert "budget" in text.lower()


def test_mine_plays_skips_when_only_one_client(tmp_path: Path) -> None:
    body = "We need this to land before Q3 — budget hesitation is real."
    for i in range(4):
        _write_atom(
            tmp_path,
            atom_id=f"01HX000000000000000000000{i}",
            atom_type="objection",
            client="Acme",
            call=f"2026-05-{10+i:02d}_acme_consultingCall",
            body=body,
        )
    result = mine_plays(vault_root=tmp_path)
    assert result.plays_written == 0


def test_cold_layer_play_matches_window(tmp_path: Path) -> None:
    body = "We need an ad bot for Reece's funnel before Q3 launch."
    for i, client in enumerate(["Acme", "Beta", "Gamma"]):
        _write_atom(
            tmp_path,
            atom_id=f"01HX000000000000000000000{i}",
            atom_type="win_signal",
            client=client,
            call=f"2026-05-{10+i:02d}_{client.lower()}_consultingCall",
            body=body,
            tags=["adbot"],
        )
    mine_plays(vault_root=tmp_path)
    hits = find_matching_plays(
        vault_root=tmp_path,
        window="You: I'm running a funnel for the new launch.",
    )
    assert hits, "expected at least one play hit"
    assert hits[0].matched_phrase == "funnel"


# ────────────────────────────────────────────────────────────────────────────
# people
# ────────────────────────────────────────────────────────────────────────────


def _crm_with_contact(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            domain TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE contacts (
            id TEXT PRIMARY KEY,
            organization_id TEXT,
            name TEXT NOT NULL,
            email TEXT,
            role TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE client_contexts (
            organization_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            goals TEXT NOT NULL DEFAULT '',
            pain_points TEXT NOT NULL DEFAULT '',
            current_tools TEXT NOT NULL DEFAULT '',
            objections TEXT NOT NULL DEFAULT '',
            tone_preferences TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            industry TEXT NOT NULL DEFAULT '',
            engagement_stage TEXT NOT NULL DEFAULT '',
            engagement_type TEXT NOT NULL DEFAULT '',
            primary_goal TEXT NOT NULL DEFAULT ''
        );
        """
    )
    conn.execute(
        "INSERT INTO organizations VALUES (?, ?, ?, '2026-05-12T00:00:00Z', '2026-05-12T00:00:00Z')",
        ("11111111-1111-1111-1111-111111111111", "Reece", None),
    )
    conn.execute(
        "INSERT INTO contacts VALUES (?, ?, ?, ?, ?, '2026-05-12T00:00:00Z', '2026-05-12T00:00:00Z')",
        (
            "22222222-2222-2222-2222-222222222222",
            "11111111-1111-1111-1111-111111111111",
            "Pat Lee",
            "pat@reece.io",
            "CEO",
        ),
    )
    conn.commit()
    conn.close()


def test_regenerate_people_writes_one_file_per_contact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crm_path = tmp_path / "crm.sqlite"
    _crm_with_contact(crm_path)
    monkeypatch.setattr(
        "consultant_brain.crm.sqlite_reader.DEFAULT_CRM_PATH", crm_path
    )

    vault = tmp_path / "vault"
    _write_atom(
        vault,
        atom_id="01HX0000000000000000000001",
        client="Reece",
        body="Pat Lee mentioned they're hiring two more engineers.",
    )
    result = regenerate_people(vault_root=vault)
    assert result.people_written == 1
    file = vault / "07_People" / "pat_lee.md"
    assert file.exists()
    text = file.read_text(encoding="utf-8")
    assert "Pat Lee" in text
    assert "CEO" in text
    assert "01HX0000000000000000000001" in text


def test_regenerate_people_is_noop_when_no_crm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.crm.sqlite_reader.DEFAULT_CRM_PATH",
        tmp_path / "missing.sqlite",
    )
    monkeypatch.setattr(
        "consultant_brain.crm.sqlite_reader.ALT_CRM_PATH",
        tmp_path / "also-missing.sqlite",
    )
    result = regenerate_people(vault_root=tmp_path / "vault")
    assert result.people_written == 0
    assert any("CRM database not found" in n for n in result.notes)


# ────────────────────────────────────────────────────────────────────────────
# reviews
# ────────────────────────────────────────────────────────────────────────────


def test_generate_weekly_review_summarizes_atoms_and_calls(tmp_path: Path) -> None:
    # Two atoms inside the target week, one outside.
    _write_atom(
        tmp_path,
        atom_id="01HX0000000000000000000001",
        atom_type="commitment",
        last_seen="2026-05-12",
    )
    _write_atom(
        tmp_path,
        atom_id="01HX0000000000000000000002",
        atom_type="objection",
        last_seen="2026-05-13",
    )
    _write_atom(
        tmp_path,
        atom_id="01HX0000000000000000000003",
        atom_type="client_fact",
        last_seen="2026-04-01",
    )

    result = generate_weekly_review(vault_root=tmp_path, week_iso="2026-W20")
    assert result.atom_count == 2
    text = result.path.read_text(encoding="utf-8")
    assert "## At a glance" in text
    assert "commitment" in text or "Commitment" in text
    # Pinned section template is included.
    assert "## Lessons + commitments for next week" in text


def test_weekly_review_preserves_pinned_section(tmp_path: Path) -> None:
    _write_atom(
        tmp_path,
        atom_id="01HX0000000000000000000001",
        atom_type="commitment",
        last_seen="2026-05-12",
    )
    first = generate_weekly_review(vault_root=tmp_path, week_iso="2026-W20")
    # User hand-edits the pinned section.
    text = first.path.read_text(encoding="utf-8")
    text = text.replace(
        "_(Add your own commitments + lessons here. Auto-generated sections above will re-render.)_",
        "MY HAND-EDITED NOTE",
    )
    first.path.write_text(text, encoding="utf-8")

    # Re-run; auto sections refresh but pinned content survives.
    second = generate_weekly_review(vault_root=tmp_path, week_iso="2026-W20")
    new_text = second.path.read_text(encoding="utf-8")
    assert "MY HAND-EDITED NOTE" in new_text


def test_weekly_review_empty_week(tmp_path: Path) -> None:
    result = generate_weekly_review(vault_root=tmp_path, week_iso="2025-W01")
    assert result.atom_count == 0
    text = result.path.read_text(encoding="utf-8")
    assert "No activity recorded" in text
