"""Stakeholder graph generator — populates 07_People/.

Walks the Swift CRM's contacts table + scans atom bodies for mentions,
writes one markdown file per person with:
  - their role + email (from CRM)
  - a backlink to the organization
  - every atom that mentions them by name

Mention detection is simple substring match for the first name and the
full name. False positives ("John" matching "Johnson") are accepted —
the consultant's eyes can spot them in Obsidian's graph view.

Idempotent: re-running re-renders every file. Person files are
auto-managed; manual notes belong in a pinned section the regenerator
preserves (mirrors client_context.py's pattern).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import frontmatter as fm
import yaml

from consultant_brain.crm.resolver import CRMResolver
from consultant_brain.crm.sqlite_reader import (
    CRMContact,
    CRMNotFoundError,
    list_all_contacts,
    list_organizations,
    open_read_only,
)
from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, AtomStatus
from consultant_brain.vault import VaultLayout


PEOPLE_DIR_NAME = "07_People"
PINNED_SECTION_HEADER = "## Pinned notes"


@dataclass(frozen=True, slots=True)
class PersonRecord:
    """One person row in the stakeholder graph."""

    slug: str
    name: str
    email: str | None
    role: str | None
    org_name: str | None
    mentioning_atoms: tuple[Atom, ...]


@dataclass(frozen=True, slots=True)
class PeopleBuildResult:
    """Outcome of one regenerate pass."""

    people_written: int
    people_updated: int
    people_dir: Path
    notes: list[str] = field(default_factory=list)


def regenerate_people(*, vault_root: Path, now: datetime | None = None) -> PeopleBuildResult:
    """Read the Swift CRM, scan atoms, write 07_People/<slug>.md files.

    When the CRM database doesn't exist yet (Swift app never launched),
    the function is a no-op — atoms still hold the client name, so a
    later run picks them up.
    """
    layout = VaultLayout.for_root(vault_root)
    people_dir = layout.root / PEOPLE_DIR_NAME
    people_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now or datetime.now(timezone.utc)

    notes: list[str] = []
    try:
        conn = open_read_only()
    except CRMNotFoundError:
        notes.append("CRM database not found — skipped.")
        return PeopleBuildResult(0, 0, people_dir, notes)

    try:
        contacts = list_all_contacts(conn)
        org_lookup = {org.id: org.name for org in list_organizations(conn)}
    finally:
        conn.close()

    if not contacts:
        notes.append("CRM has zero contacts — skipped.")
        return PeopleBuildResult(0, 0, people_dir, notes)

    all_atoms = _load_atoms(layout)
    records: list[PersonRecord] = []
    for contact in contacts:
        record = PersonRecord(
            slug=_slugify(contact.name),
            name=contact.name,
            email=contact.email,
            role=contact.role,
            org_name=org_lookup.get(contact.organization_id) if contact.organization_id else None,
            mentioning_atoms=tuple(_atoms_mentioning(contact, all_atoms)),
        )
        records.append(record)

    written = 0
    updated = 0
    for record in records:
        path = people_dir / f"{record.slug}.md"
        existed = path.exists()
        _write_person_note(path=path, record=record, now=timestamp)
        if existed:
            updated += 1
        else:
            written += 1
    return PeopleBuildResult(
        people_written=written,
        people_updated=updated,
        people_dir=people_dir,
        notes=notes,
    )


def _load_atoms(layout: VaultLayout) -> list[Atom]:
    atoms: list[Atom] = []
    if not layout.atoms_dir.exists():
        return atoms
    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        if atom.status is AtomStatus.retired:
            continue
        atoms.append(atom)
    return atoms


def _atoms_mentioning(contact: CRMContact, atoms: list[Atom]) -> list[Atom]:
    """Substring-match the contact's name (and first name) against atom
    bodies. Case-insensitive. Returns matches sorted by recency."""
    needles = {contact.name.strip().lower()}
    first_name = contact.name.strip().split()[0].lower() if contact.name.strip() else ""
    if first_name and len(first_name) >= 3:
        needles.add(first_name)
    matches: list[Atom] = []
    for atom in atoms:
        body_lower = atom.body.lower()
        if any(needle in body_lower for needle in needles):
            matches.append(atom)
    matches.sort(key=lambda a: a.last_seen, reverse=True)
    return matches


def _write_person_note(
    *,
    path: Path,
    record: PersonRecord,
    now: datetime,
) -> None:
    """Write/update one person markdown. Preserves any pinned section
    a human added (mirrors client_context.py's pinned-block contract)."""
    existing_pinned = ""
    existing_meta: dict = {}
    if path.exists():
        try:
            post = fm.load(path.open("r", encoding="utf-8"))
            existing_meta = dict(post.metadata)
            existing_pinned = _extract_pinned_section(post.content or "")
        except Exception:
            pass

    created_at_iso = existing_meta.get("created_at") or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    frontmatter_data = {
        "id": record.slug,
        "kind": "person",
        "name": record.name,
        "role": record.role or "",
        "email": record.email or "",
        "org": record.org_name or "",
        "mention_count": len(record.mentioning_atoms),
        "created_at": created_at_iso,
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    org_link = f"[[clients/{_slugify(record.org_name)}]]" if record.org_name else ""
    role_line = f"**Role**: {record.role}" if record.role else ""
    email_line = f"**Email**: {record.email}" if record.email else ""
    head_lines = [line for line in [role_line, email_line, f"**Org**: {org_link}" if org_link else ""] if line]

    atom_links: list[str] = []
    for atom in record.mentioning_atoms[:50]:
        excerpt = atom.body.strip().replace("\n", " ")
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."
        atom_links.append(f"- [[{atom.id}]] — {atom.type.value} · {excerpt}")

    auto_body = f"""# {record.name}

{chr(10).join(head_lines)}

## Atom mentions ({len(record.mentioning_atoms)})
{chr(10).join(atom_links) if atom_links else "_No atom mentions yet._"}
"""

    pinned_block = f"\n\n{PINNED_SECTION_HEADER}\n{existing_pinned.rstrip()}\n" if existing_pinned else ""

    yaml_block = yaml.safe_dump(frontmatter_data, sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{auto_body.rstrip()}{pinned_block}"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    tmp.replace(path)


def _extract_pinned_section(body: str) -> str:
    """Return everything from `## Pinned notes` to the end. Empty when no
    pinned block exists yet."""
    idx = body.find(PINNED_SECTION_HEADER)
    if idx < 0:
        return ""
    return body[idx + len(PINNED_SECTION_HEADER):].strip()


_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _slugify(text: str | None) -> str:
    if not text:
        return "unknown"
    return _SLUG_RE.sub("_", text.lower()).strip("_") or "unknown"
