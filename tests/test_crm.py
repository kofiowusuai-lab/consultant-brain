"""CRM SQLite reader + resolver tests against a fixture database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from consultant_brain.crm.resolver import CRMResolver
from consultant_brain.crm.sqlite_reader import (
    CRMNotFoundError,
    find_client_context,
    find_organization_by_id,
    find_organization_by_name,
    list_all_contacts,
    list_contacts_for_org,
    list_organizations,
    open_read_only,
)


def _build_crm(path: Path, *, orgs: list[tuple[str, str, str | None]], contacts: list[tuple[str, str, str, str | None, str | None]] = None, contexts: list[tuple[str, str]] = None) -> None:
    """Spin up a fresh fixture CRM SQLite mirroring the Swift schema."""
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
    for org_id, name, domain in orgs:
        conn.execute(
            "INSERT INTO organizations VALUES (?, ?, ?, '2026-05-12T00:00:00Z', '2026-05-12T00:00:00Z')",
            (org_id, name, domain),
        )
    for cid, org_id, name, email, role in (contacts or []):
        conn.execute(
            "INSERT INTO contacts VALUES (?, ?, ?, ?, ?, '2026-05-12T00:00:00Z', '2026-05-12T00:00:00Z')",
            (cid, org_id, name, email, role),
        )
    for org_id, primary_goal in (contexts or []):
        conn.execute(
            "INSERT INTO client_contexts (organization_id, summary, goals, pain_points, current_tools, objections, tone_preferences, notes, updated_at, industry, engagement_stage, engagement_type, primary_goal) VALUES (?, '', '', '', '', '', '', '', '2026-05-12T00:00:00Z', '', '', '', ?)",
            (org_id, primary_goal),
        )
    conn.commit()
    conn.close()


# ────────────────────────────────────────────────────────────────────────────
# sqlite_reader
# ────────────────────────────────────────────────────────────────────────────


def test_open_read_only_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(CRMNotFoundError):
        open_read_only(tmp_path / "no-such.sqlite")


def test_open_read_only_blocks_writes(tmp_path: Path) -> None:
    """A write attempt should fail because we opened mode=ro."""
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    conn = open_read_only(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO organizations VALUES ('x', 'y', NULL, '', '')"
        )


def test_list_organizations_returns_alphabetical(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(
        db,
        orgs=[
            ("00000000-0000-0000-0000-000000000a01", "Zoo Industries", None),
            ("00000000-0000-0000-0000-000000000a02", "Acme AI", "acme.com"),
            ("00000000-0000-0000-0000-000000000a03", "reece consulting", None),
        ],
    )
    conn = open_read_only(db)
    orgs = list_organizations(conn)
    assert [o.name for o in orgs] == ["Acme AI", "reece consulting", "Zoo Industries"]
    assert orgs[0].domain == "acme.com"


def test_find_organization_by_name_is_case_insensitive(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    conn = open_read_only(db)
    hit = find_organization_by_name(conn, "reece")
    assert hit is not None
    assert hit.id == UUID("00000000-0000-0000-0000-000000000a01")
    assert find_organization_by_name(conn, "missing") is None


def test_find_organization_by_id(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    org_uuid = "00000000-0000-0000-0000-000000000a01"
    _build_crm(db, orgs=[(org_uuid, "Reece", None)])
    conn = open_read_only(db)
    assert find_organization_by_id(conn, UUID(org_uuid)).name == "Reece"
    assert find_organization_by_id(conn, uuid4()) is None


def test_list_contacts_for_org_filters_correctly(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    o1 = "00000000-0000-0000-0000-000000000001"
    o2 = "00000000-0000-0000-0000-000000000002"
    _build_crm(
        db,
        orgs=[(o1, "Acme", None), (o2, "Reece", None)],
        contacts=[
            ("00000000-0000-0000-0000-00000000c001", o1, "Pat Lee", "pat@acme.com", "CEO"),
            ("00000000-0000-0000-0000-00000000c002", o1, "Sam West", None, None),
            ("00000000-0000-0000-0000-00000000c003", o2, "Reece himself", None, "Founder"),
        ],
    )
    conn = open_read_only(db)
    acme_contacts = list_contacts_for_org(conn, UUID(o1))
    assert [c.name for c in acme_contacts] == ["Pat Lee", "Sam West"]
    assert acme_contacts[0].email == "pat@acme.com"
    assert acme_contacts[0].role == "CEO"
    assert acme_contacts[1].email is None
    all_contacts = list_all_contacts(conn)
    assert len(all_contacts) == 3


def test_find_client_context_returns_primary_goal(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    org_id = "00000000-0000-0000-0000-000000000a01"
    _build_crm(
        db,
        orgs=[(org_id, "Reece", None)],
        contexts=[(org_id, "Ship the ad-bot in 30 days")],
    )
    conn = open_read_only(db)
    ctx = find_client_context(conn, UUID(org_id))
    assert ctx is not None
    assert ctx.primary_goal == "Ship the ad-bot in 30 days"
    assert find_client_context(conn, uuid4()) is None


# ────────────────────────────────────────────────────────────────────────────
# resolver
# ────────────────────────────────────────────────────────────────────────────


def test_resolver_returns_org(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    r = CRMResolver(crm_path=db)
    org = r.resolve("Reece")
    assert org is not None
    assert org.id == UUID("00000000-0000-0000-0000-000000000a01")


def test_resolver_returns_none_for_missing_name(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    r = CRMResolver(crm_path=db)
    assert r.resolve("never-existed") is None


def test_resolver_returns_none_when_crm_missing(tmp_path: Path) -> None:
    """If the Swift app hasn't run yet, the CRM file won't exist. The
    resolver shouldn't crash — it should return None and let downstream
    code treat that as 'no link yet'."""
    r = CRMResolver(crm_path=tmp_path / "doesnt-exist.sqlite")
    assert r.resolve("Reece") is None


def test_resolver_caches_lookups(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    r = CRMResolver(crm_path=db)
    first = r.resolve("Reece")
    # Mutate the underlying DB; the cached result shouldn't change.
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE organizations SET name = 'Renamed' WHERE id = '00000000-0000-0000-0000-000000000a01'")
    conn.commit()
    conn.close()
    second = r.resolve("Reece")  # cached
    assert first.name == "Reece"
    assert second.name == "Reece"
    r.forget("Reece")
    after_forget = r.resolve("Reece")
    # After forget, "Reece" no longer matches anything (the row is now "Renamed").
    assert after_forget is None


def test_resolver_resolve_uuid_shortcut(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    r = CRMResolver(crm_path=db)
    assert r.resolve_uuid("Reece") == UUID("00000000-0000-0000-0000-000000000a01")
    assert r.resolve_uuid("missing") is None
    assert r.resolve_uuid(None) is None


def test_resolver_caches_negative_results(tmp_path: Path) -> None:
    db = tmp_path / "crm.sqlite"
    _build_crm(db, orgs=[("00000000-0000-0000-0000-000000000a01", "Reece", None)])
    r = CRMResolver(crm_path=db)
    assert r.resolve("Acme") is None
    # Add Acme to the DB after the negative cache hit.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO organizations VALUES ('00000000-0000-0000-0000-000000000a02', 'Acme', NULL, '2026-05-12T00:00:00Z', '2026-05-12T00:00:00Z')"
    )
    conn.commit()
    conn.close()
    # Without forget, the negative cache hides the new row — that's the
    # intentional behavior so we don't slam SQLite every atom.
    assert r.resolve("Acme") is None
    r.forget("Acme")
    assert r.resolve("Acme") is not None
