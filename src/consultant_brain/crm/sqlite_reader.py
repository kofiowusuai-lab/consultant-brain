"""Read-only access to the Swift app's CRM SQLite database.

Lives at `~/Library/Application Support/Consultant Copilot/crm.sqlite`,
schema defined by `SQLiteCRMStore.swift`. We never write — the Swift app
owns this database. Opening in URI mode with `mode=ro` so accidental
writes raise rather than corrupting CRM data while the user is on a call.

Phase 8 only needs reads of:
  - organizations  (id, name, domain)
  - contacts       (id, organization_id, name, email, role)
  - client_contexts (organization_id, primary_goal — for the scoring's
                     primary-win judge to consume)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


DEFAULT_CRM_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Consultant Copilot"
    / "CRM"
    / "crm.sqlite"
)


# Fallback if the user is on an older Swift build that wrote the db one
# level shallower.
ALT_CRM_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Consultant Copilot"
    / "crm.sqlite"
)


@dataclass(frozen=True, slots=True)
class CRMOrganization:
    """One organization from the Swift app's CRM."""

    id: UUID
    name: str
    domain: str | None


@dataclass(frozen=True, slots=True)
class CRMContact:
    """One contact (stakeholder) from the Swift app's CRM."""

    id: UUID
    organization_id: UUID | None
    name: str
    email: str | None
    role: str | None


@dataclass(frozen=True, slots=True)
class CRMClientContext:
    """The Swift app's per-org context block — we read primary_goal for
    the primary-win-progress scoring feature."""

    organization_id: UUID
    primary_goal: str


class CRMNotFoundError(FileNotFoundError):
    """Raised when no CRM database exists at the expected path."""


def resolve_crm_path(path: Path | None = None) -> Path:
    """Pick the right CRM path with reasonable fallbacks. Tests pass an
    explicit path; production code calls with no args."""
    if path is not None:
        return path
    if DEFAULT_CRM_PATH.exists():
        return DEFAULT_CRM_PATH
    if ALT_CRM_PATH.exists():
        return ALT_CRM_PATH
    return DEFAULT_CRM_PATH  # surface the canonical missing-path error


def open_read_only(path: Path | None = None) -> sqlite3.Connection:
    """Open the CRM database in read-only URI mode."""
    target = resolve_crm_path(path)
    if not target.exists():
        raise CRMNotFoundError(
            f"CRM database not found at {target}. The Swift app must have "
            "run at least once and have at least one client added."
        )
    conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_organizations(conn: sqlite3.Connection) -> list[CRMOrganization]:
    """Return every organization. Sorted alphabetically by name."""
    rows = conn.execute(
        "SELECT id, name, domain FROM organizations ORDER BY name COLLATE NOCASE"
    ).fetchall()
    return [_org_from_row(r) for r in rows]


def find_organization_by_name(
    conn: sqlite3.Connection, name: str
) -> CRMOrganization | None:
    """Case-insensitive exact match on the org's display name. Returns
    None when no row matches."""
    row = conn.execute(
        "SELECT id, name, domain FROM organizations WHERE LOWER(name) = LOWER(?) LIMIT 1",
        (name,),
    ).fetchone()
    return _org_from_row(row) if row else None


def find_organization_by_id(
    conn: sqlite3.Connection, org_id: UUID
) -> CRMOrganization | None:
    row = conn.execute(
        "SELECT id, name, domain FROM organizations WHERE id = ? LIMIT 1",
        (str(org_id),),
    ).fetchone()
    return _org_from_row(row) if row else None


def list_contacts_for_org(
    conn: sqlite3.Connection, org_id: UUID
) -> list[CRMContact]:
    """Every contact attached to one org."""
    rows = conn.execute(
        "SELECT id, organization_id, name, email, role FROM contacts WHERE organization_id = ? ORDER BY name COLLATE NOCASE",
        (str(org_id),),
    ).fetchall()
    return [_contact_from_row(r) for r in rows]


def list_all_contacts(conn: sqlite3.Connection) -> list[CRMContact]:
    """Every contact in the CRM. Used by the stakeholder graph builder."""
    rows = conn.execute(
        "SELECT id, organization_id, name, email, role FROM contacts ORDER BY name COLLATE NOCASE"
    ).fetchall()
    return [_contact_from_row(r) for r in rows]


def find_client_context(
    conn: sqlite3.Connection, org_id: UUID
) -> CRMClientContext | None:
    """The Swift app's per-org context row — currently we only consume
    primary_goal for scoring's primary-win-progress feature."""
    row = conn.execute(
        "SELECT organization_id, primary_goal FROM client_contexts WHERE organization_id = ? LIMIT 1",
        (str(org_id),),
    ).fetchone()
    if row is None:
        return None
    try:
        return CRMClientContext(
            organization_id=UUID(row["organization_id"]),
            primary_goal=row["primary_goal"] or "",
        )
    except (ValueError, TypeError):
        return None


# ────────────────────────────────────────────────────────────────────────────


def _org_from_row(row: sqlite3.Row | None) -> CRMOrganization | None:
    if row is None:
        return None
    try:
        return CRMOrganization(
            id=UUID(row["id"]),
            name=row["name"],
            domain=(row["domain"] or None),
        )
    except (ValueError, TypeError):
        return None


def _contact_from_row(row: sqlite3.Row | None) -> CRMContact | None:
    if row is None:
        return None
    try:
        org_id_raw = row["organization_id"]
        org_id = UUID(org_id_raw) if org_id_raw else None
        return CRMContact(
            id=UUID(row["id"]),
            organization_id=org_id,
            name=row["name"],
            email=(row["email"] or None),
            role=(row["role"] or None),
        )
    except (ValueError, TypeError):
        return None
