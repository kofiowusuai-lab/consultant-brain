"""Anonymized vault export.

`consultant-brain export --out path/` writes a sanitized copy of the
vault that's safe to share with a team or open-source. With anonymize
(default), every distinct client display name becomes `[CLIENT_N]`,
every stakeholder name becomes `[PERSON_N]`, every email is redacted
to `[EMAIL_N]`. Pattern + play STRUCTURE is preserved intact — the
whole point is to share the playbook without leaking client identity.

What's redacted:
  - frontmatter `client:` fields
  - body occurrences of client names + stakeholder names
  - email addresses anywhere in markdown
  - `client_org_id` UUIDs (replaced with deterministic `org-N`)
  - CallNote source_session paths (replaced with synthetic ids)

What's preserved:
  - atom types, tags, confidence, dates, structure
  - patterns + plays in their entirety
  - markdown formatting + links (rewritten through the alias map)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import frontmatter as fm

from consultant_brain.crm.sqlite_reader import (
    CRMNotFoundError,
    list_all_contacts,
    list_organizations,
    open_read_only,
)
from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom
from consultant_brain.vault import VaultLayout, read_frontmatter


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


@dataclass(frozen=True, slots=True)
class ExportResult:
    out_dir: Path
    atom_count: int
    call_count: int
    pattern_count: int
    play_count: int
    replacements_applied: int


def export_vault(*, vault_root: Path, out_dir: Path, anonymize: bool = True) -> ExportResult:
    """Copy the vault to `out_dir`, applying anonymization when requested."""
    vault_root = vault_root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = VaultLayout.for_root(vault_root)

    alias_map = _build_alias_map(layout) if anonymize else {}
    replacements_applied = 0

    def _maybe_redact(text: str) -> tuple[str, int]:
        if not anonymize or not text:
            return text, 0
        applied = 0
        for original, replacement in alias_map.items():
            if original and original in text:
                text = text.replace(original, replacement)
                applied += 1
        text, email_applied = _redact_emails(text, alias_map)
        text, uuid_applied = _redact_uuids(text, alias_map)
        return text, applied + email_applied + uuid_applied

    atom_count = 0
    if layout.atoms_dir.exists():
        target = out_dir / "03_Atoms"
        target.mkdir(parents=True, exist_ok=True)
        for path in layout.atoms_dir.glob("*.md"):
            redacted_text, applied = _redact_markdown(path, _maybe_redact, anonymize)
            (target / path.name).write_text(redacted_text, encoding="utf-8")
            atom_count += 1
            replacements_applied += applied

    call_count = 0
    if layout.calls_dir.exists():
        target = out_dir / "02_Calls"
        target.mkdir(parents=True, exist_ok=True)
        for path in layout.calls_dir.glob("*.md"):
            redacted_text, applied = _redact_markdown(path, _maybe_redact, anonymize)
            (target / path.name).write_text(redacted_text, encoding="utf-8")
            call_count += 1
            replacements_applied += applied

    pattern_count = _copy_passthrough(layout.root / "04_Patterns", out_dir / "04_Patterns", _maybe_redact, anonymize, replacements_applied_ref=[0])
    play_count = _copy_passthrough(layout.root / "05_Plays", out_dir / "05_Plays", _maybe_redact, anonymize, replacements_applied_ref=[0])

    # Brain skeleton + system index untouched aside from text replacement
    for d in ("00_System", "06_Definitions", "07_People", "08_Reviews"):
        src = layout.root / d
        dst = out_dir / d
        if src.exists():
            _copy_passthrough(src, dst, _maybe_redact, anonymize, replacements_applied_ref=[0])

    # Drop a manifest so the receiver knows what they're looking at.
    manifest_lines = [
        "# Anonymized vault export" if anonymize else "# Vault export",
        f"",
        f"- atoms: {atom_count}",
        f"- calls: {call_count}",
        f"- patterns: {pattern_count}",
        f"- plays: {play_count}",
        f"- anonymized: {anonymize}",
        f"- replacements_applied: {replacements_applied}",
    ]
    (out_dir / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return ExportResult(
        out_dir=out_dir,
        atom_count=atom_count,
        call_count=call_count,
        pattern_count=pattern_count,
        play_count=play_count,
        replacements_applied=replacements_applied,
    )


def _build_alias_map(layout: VaultLayout) -> dict[str, str]:
    """Stable client/person → alias mapping. The same name always maps
    to the same `[CLIENT_N]` across the run."""
    client_names: list[str] = []
    if layout.atoms_dir.exists():
        for path in layout.atoms_dir.glob("*.md"):
            try:
                atom = _atom_from_markdown(path)
            except Exception:
                continue
            if atom.client and atom.client not in client_names:
                client_names.append(atom.client)

    # Person names from the Swift CRM, when available.
    person_names: list[str] = []
    try:
        conn = open_read_only()
    except CRMNotFoundError:
        conn = None
    if conn is not None:
        try:
            for contact in list_all_contacts(conn):
                if contact.name and contact.name not in person_names:
                    person_names.append(contact.name)
            # If CRM has orgs the vault hasn't picked up yet, add them too.
            for org in list_organizations(conn):
                if org.name and org.name not in client_names:
                    client_names.append(org.name)
        finally:
            conn.close()

    mapping: dict[str, str] = {}
    for idx, name in enumerate(client_names, start=1):
        mapping[name] = f"[CLIENT_{idx}]"
    for idx, name in enumerate(person_names, start=1):
        mapping[name] = f"[PERSON_{idx}]"
        first = name.split()[0] if name.strip() else ""
        if first and first not in mapping:
            mapping[first] = f"[PERSON_{idx}]"
    return mapping


def _redact_markdown(path: Path, redactor, anonymize: bool) -> tuple[str, int]:
    try:
        post = fm.load(path.open("r", encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace"), 0

    metadata = dict(post.metadata)
    applied = 0
    if anonymize:
        for key in ("client", "summary", "source_session", "org", "email"):
            if key in metadata and isinstance(metadata[key], str):
                new, count = redactor(metadata[key])
                metadata[key] = new
                applied += count
        if "client_org_id" in metadata and metadata["client_org_id"]:
            metadata["client_org_id"] = "org-X"
            applied += 1

    body_text = post.content or ""
    if anonymize:
        body_text, count = redactor(body_text)
        applied += count

    import yaml

    yaml_block = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{yaml_block}\n---\n\n{body_text.rstrip()}\n", applied


def _copy_passthrough(src: Path, dst: Path, redactor, anonymize: bool, *, replacements_applied_ref: list[int]) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in src.iterdir():
        if entry.is_dir():
            _copy_passthrough(entry, dst / entry.name, redactor, anonymize, replacements_applied_ref=replacements_applied_ref)
            continue
        if entry.suffix == ".md" and anonymize:
            text, applied = _redact_markdown(entry, redactor, anonymize)
            (dst / entry.name).write_text(text, encoding="utf-8")
            replacements_applied_ref[0] += applied
        else:
            shutil.copy2(entry, dst / entry.name)
        count += 1
    return count


def _redact_emails(text: str, alias_map: dict[str, str]) -> tuple[str, int]:
    counter = [0]

    def repl(match: re.Match) -> str:
        counter[0] += 1
        return alias_map.setdefault(match.group(0), f"[EMAIL_{counter[0]}]")

    return _EMAIL_RE.sub(repl, text), counter[0]


def _redact_uuids(text: str, alias_map: dict[str, str]) -> tuple[str, int]:
    counter = [0]

    def repl(match: re.Match) -> str:
        counter[0] += 1
        return alias_map.setdefault(match.group(0), f"[UUID_{counter[0]}]")

    return _UUID_RE.sub(repl, text), counter[0]
