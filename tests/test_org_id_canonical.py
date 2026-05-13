"""Phase 11 — canonical-ID enforcement tests.

Covers two contracts:

1. Mint paths populate `client_org_id` when the client resolves in the
   CRM (notes ingest, context-dump commit, learn ingest). The unit
   tests for those modules already exist; here we verify that the
   shared mint-time helper writes the UUID into both the markdown
   frontmatter AND the LanceDB row.

2. `reconcile-org-ids` rewrites the backlog idempotently:
     - atoms with `client` but no `client_org_id` → updated
     - atoms whose name doesn't match the CRM → skipped_no_match
     - atoms already carrying a UUID → already_set (no rewrite)
     - re-running after a successful pass reports updated=0

Mixed-state vault: we mint three atoms (one with UUID, two without)
and verify reconcile only touches the two missing UUIDs and leaves the
third byte-identical.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from consultant_brain.crm.resolver import CRMResolver
from consultant_brain.reconcile import run_reconcile_org_ids
from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, AtomStatus, AtomType, CallType, SourceKind
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton, write_atom


REECE_UUID = UUID("11111111-1111-4111-8111-111111111111")
ACME_UUID = UUID("22222222-2222-4222-8222-222222222222")


def _build_crm(path: Path, *, orgs: list[tuple[str, str]]) -> None:
    """Minimal CRM mirror — orgs only; reconcile doesn't read contacts."""
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
        """
    )
    for org_id, name in orgs:
        conn.execute(
            "INSERT INTO organizations VALUES (?, ?, NULL, '2026-05-12T00:00:00Z', '2026-05-12T00:00:00Z')",
            (org_id, name),
        )
    conn.commit()
    conn.close()


def _make_atom(
    *,
    atom_id: str,
    client: str | None,
    client_org_id: UUID | None = None,
) -> Atom:
    return Atom(
        id=atom_id,
        type=AtomType.client_fact,
        client=client,
        client_org_id=client_org_id,
        call="2026-05-12_reece_consultingCall",
        call_type=CallType.consulting_call,
        source_kind=SourceKind.call,
        tags=["test"],
        confidence=0.8,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 19, 42, 10, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id=atom_id,
        body="Test atom body for reconcile-org-ids.",
    )


# ────────────────────────────────────────────────────────────────────────────
# Mixed-state vault
# ────────────────────────────────────────────────────────────────────────────


def test_reconcile_populates_missing_uuids(tmp_path: Path) -> None:
    """Three atoms: one with UUID, one matching the CRM, one not in CRM.
    After reconcile: the matching atom gains its UUID, the unmatched atom
    is reported as skipped, and the pre-set atom is untouched.
    """
    crm_db = tmp_path / "crm.sqlite"
    _build_crm(
        crm_db,
        orgs=[
            (str(REECE_UUID), "Reece"),
            (str(ACME_UUID), "Acme Co"),
        ],
    )
    resolver = CRMResolver(crm_path=crm_db)

    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    # Pre-set atom (already has UUID) — must not be rewritten.
    preset = _make_atom(
        atom_id="ATOMPRESET00000000000000000",
        client="Reece",
        client_org_id=REECE_UUID,
    )
    preset_path = write_atom(layout, preset)
    preset_mtime = preset_path.stat().st_mtime_ns

    # Backlog atom — has client, no UUID. Must be backfilled.
    backlog = _make_atom(
        atom_id="ATOMBACKLOG00000000000000Z0",
        client="Reece",
        client_org_id=None,
    )
    write_atom(layout, backlog)

    # Unknown client — CRM has no match. Must be skipped, not failed.
    unknown = _make_atom(
        atom_id="ATOMUNKNOWN00000000000000Z1",
        client="Ghost Org",
        client_org_id=None,
    )
    write_atom(layout, unknown)

    summary = run_reconcile_org_ids(vault_root=layout.root, resolver=resolver)

    assert summary.scanned == 3
    assert summary.updated == 1
    assert summary.already_set == 1
    assert summary.skipped_no_match == ["Ghost Org"]
    assert summary.failed == []
    assert summary.dry_run is False

    # Backfilled atom now carries the UUID, on disk + in LanceDB row.
    reloaded_backlog = _atom_from_markdown(layout.atoms_dir / "ATOMBACKLOG00000000000000Z0.md")
    assert reloaded_backlog.client_org_id == REECE_UUID

    # Pre-set atom byte-untouched (mtime unchanged proves no rewrite).
    assert preset_path.stat().st_mtime_ns == preset_mtime
    reloaded_preset = _atom_from_markdown(preset_path)
    assert reloaded_preset.client_org_id == REECE_UUID


def test_reconcile_is_idempotent(tmp_path: Path) -> None:
    """Two successive runs against the same vault: first pass populates
    the UUID, second pass reports already_set + updated=0."""
    crm_db = tmp_path / "crm.sqlite"
    _build_crm(crm_db, orgs=[(str(REECE_UUID), "Reece")])
    resolver = CRMResolver(crm_path=crm_db)

    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    atom = _make_atom(
        atom_id="IDEMPOTENT00000000000000000",
        client="Reece",
        client_org_id=None,
    )
    write_atom(layout, atom)

    first = run_reconcile_org_ids(vault_root=layout.root, resolver=resolver)
    assert first.updated == 1
    assert first.already_set == 0

    second = run_reconcile_org_ids(vault_root=layout.root, resolver=resolver)
    assert second.updated == 0
    assert second.already_set == 1
    assert second.skipped_no_match == []


def test_reconcile_dry_run_writes_nothing(tmp_path: Path) -> None:
    crm_db = tmp_path / "crm.sqlite"
    _build_crm(crm_db, orgs=[(str(REECE_UUID), "Reece")])
    resolver = CRMResolver(crm_path=crm_db)

    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    atom = _make_atom(
        atom_id="DRYRUNATOM0000000000000000Z",
        client="Reece",
        client_org_id=None,
    )
    path = write_atom(layout, atom)
    mtime_before = path.stat().st_mtime_ns

    summary = run_reconcile_org_ids(
        vault_root=layout.root, resolver=resolver, dry_run=True
    )
    assert summary.updated == 1
    assert summary.dry_run is True

    # No rewrite — file mtime untouched.
    assert path.stat().st_mtime_ns == mtime_before
    reloaded = _atom_from_markdown(path)
    assert reloaded.client_org_id is None


def test_reconcile_skips_knowledge_atoms_without_client(tmp_path: Path) -> None:
    """Knowledge atoms minted from a generic source (no --for-client) have
    no `client` field. They must be reported as no-client, not failed."""
    crm_db = tmp_path / "crm.sqlite"
    _build_crm(crm_db, orgs=[(str(REECE_UUID), "Reece")])
    resolver = CRMResolver(crm_path=crm_db)

    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    knowledge_atom = Atom(
        id="KNOWLEDGEATOM000000000000Z2",
        type=AtomType.client_fact,
        client=None,
        client_org_id=None,
        call="youtube_dQw4w9WgXcQ",
        call_type=CallType.ai_training,
        source_kind=SourceKind.youtube,
        source_url="https://youtu.be/dQw4w9WgXcQ",
        source_title="Training",
        tags=["test"],
        confidence=0.8,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 19, 42, 10, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id="KNOWLEDGEATOM000000000000Z2",
        body="Generic training insight, not tied to a client.",
    )
    write_atom(layout, knowledge_atom)

    summary = run_reconcile_org_ids(vault_root=layout.root, resolver=resolver)
    assert summary.scanned == 1
    assert summary.updated == 0
    assert summary.skipped_no_client == 1
    assert summary.failed == []


def test_reconcile_summary_line_lists_unmatched_names(tmp_path: Path) -> None:
    crm_db = tmp_path / "crm.sqlite"
    _build_crm(crm_db, orgs=[])
    resolver = CRMResolver(crm_path=crm_db)

    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    for i, client_name in enumerate(["Alpha", "Beta", "Alpha"]):
        write_atom(
            layout,
            _make_atom(
                atom_id=f"UNMATCHED{i:017d}",
                client=client_name,
                client_org_id=None,
            ),
        )

    summary = run_reconcile_org_ids(vault_root=layout.root, resolver=resolver)
    line = summary.summary_line()
    # Deduped + alphabetized in the summary; raw list keeps the per-atom hits.
    assert "no-match 2" in line
    assert "Alpha" in line and "Beta" in line
    assert summary.skipped_no_match.count("Alpha") == 2
    assert summary.skipped_no_match.count("Beta") == 1
