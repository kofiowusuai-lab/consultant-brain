"""Auto-regenerate `01_Clients/<slug>/context.md` from atoms + recent calls.

The master prompt says: "context.md is auto-regenerated — never hand-edit
beyond the pinned section at top". This module honors that:

  - Reads every active atom whose `client` field == this client.
  - Reads the N most recent call notes for this client.
  - Writes a context.md that groups atoms by type + lists call summaries
    with their score (when scoreable).
  - PRESERVES any text inside `<!-- pin -->` ... `<!-- /pin -->` blocks
    so the user CAN hand-edit pinned notes without losing them on re-run.

Idempotent: running distill twice in a row produces identical files.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import yaml

from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, AtomStatus, AtomType
from consultant_brain.vault import VaultLayout, slugify_client


PIN_PATTERN = re.compile(r"<!--\s*pin\s*-->(.*?)<!--\s*/pin\s*-->", re.DOTALL)
RECENT_CALLS_MAX = 5


@dataclass(frozen=True, slots=True)
class ContextRegenerationResult:
    """Outcome of one regeneration pass for one client."""

    client_slug: str
    context_path: Path
    atom_count: int
    call_count: int
    preserved_pinned: bool


def regenerate_client_context(
    *,
    vault_root: Path,
    client_name: str,
    now: datetime | None = None,
) -> ContextRegenerationResult | None:
    """Rebuild context.md for one client. Returns None when the client has
    no atoms in the vault (nothing to summarize)."""
    timestamp = now or datetime.now(timezone.utc)
    layout = VaultLayout.for_root(vault_root)
    slug = slugify_client(client_name)
    client_dir = layout.client_dir(slug)
    client_dir.mkdir(parents=True, exist_ok=True)

    atoms = _atoms_for_client(layout=layout, client_name=client_name)
    if not atoms:
        return None

    calls = _recent_calls_for_client(layout=layout, client_name=client_name, limit=RECENT_CALLS_MAX)
    context_path = client_dir / "context.md"
    pinned = _read_pinned_block(context_path)
    body = _render_body(
        client_name=client_name,
        atoms=atoms,
        calls=calls,
        pinned=pinned,
        now=timestamp,
    )
    frontmatter_data = {
        "client": client_name,
        "client_slug": slug,
        "auto_generated": True,
        "atom_count": len(atoms),
        "call_count": len(calls),
        "regenerated_at": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    content = _wrap_with_frontmatter(metadata=frontmatter_data, body=body)
    tmp = context_path.with_suffix(context_path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(context_path)
    return ContextRegenerationResult(
        client_slug=slug,
        context_path=context_path,
        atom_count=len(atoms),
        call_count=len(calls),
        preserved_pinned=bool(pinned),
    )


def regenerate_all_clients(*, vault_root: Path, now: datetime | None = None) -> list[ContextRegenerationResult]:
    """Walk every subdirectory under 01_Clients/, regenerate context.md
    for each. Returns the list of successful regenerations."""
    layout = VaultLayout.for_root(vault_root)
    if not layout.clients_dir.exists():
        return []
    results: list[ContextRegenerationResult] = []
    for child in sorted(layout.clients_dir.iterdir()):
        if not child.is_dir():
            continue
        # The directory name is the slug; recover the display name from any
        # atom or just title-case the slug.
        client_name = _client_name_from_slug(layout=layout, slug=child.name) or child.name.replace("_", " ").title()
        try:
            result = regenerate_client_context(
                vault_root=vault_root,
                client_name=client_name,
                now=now,
            )
        except ValueError:
            # Bad slug — skip rather than crash a whole distill pass.
            continue
        if result is not None:
            results.append(result)
    return results


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _atoms_for_client(*, layout: VaultLayout, client_name: str) -> list[Atom]:
    """Read every atom whose `client` frontmatter resolves to this name.
    Skips retired / needs_review atoms."""
    matches: list[Atom] = []
    expected = f"[[{client_name}]]"
    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        if atom.status is not AtomStatus.active:
            continue
        if atom.client != client_name and not _matches_wikilink_form(path=path, expected_link=expected):
            continue
        matches.append(atom)
    # Sort newest-first by created_at so the body groupings show the most
    # recent observations first.
    matches.sort(key=lambda a: a.created_at, reverse=True)
    return matches


def _matches_wikilink_form(*, path: Path, expected_link: str) -> bool:
    """Backup match for atom files whose frontmatter keeps the wikilink
    form ([[Name]]) — _atom_from_markdown strips it, but defensive check."""
    try:
        post = frontmatter.load(path.open("r", encoding="utf-8"))
        return post.metadata.get("client") == expected_link
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class CallNoteSummary:
    """Lightweight call-note view for the context.md timeline."""

    id: str
    date: str
    call_type: str
    atom_count: int
    summary: str
    duration_minutes: int


def _recent_calls_for_client(*, layout: VaultLayout, client_name: str, limit: int) -> list[CallNoteSummary]:
    if not layout.calls_dir.exists():
        return []
    expected_link = f"[[{client_name}]]"
    summaries: list[CallNoteSummary] = []
    for path in layout.calls_dir.glob("*.md"):
        try:
            post = frontmatter.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        meta = dict(post.metadata)
        if meta.get("client") != expected_link:
            continue
        # Pull the summary section out of the body.
        body_summary = ""
        match = re.search(r"##\s*Summary\s*\n+(.+?)(?=\n##|\Z)", post.content or "", re.DOTALL)
        if match:
            body_summary = match.group(1).strip()
        summaries.append(
            CallNoteSummary(
                id=str(meta.get("id", path.stem)),
                date=str(meta.get("date", "")),
                call_type=str(meta.get("call_type", "")),
                atom_count=int(meta.get("atom_count", 0)),
                summary=body_summary,
                duration_minutes=int(meta.get("duration_minutes", 0)),
            )
        )
    summaries.sort(key=lambda s: s.date, reverse=True)
    return summaries[:limit]


def _read_pinned_block(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    match = PIN_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def _wrap_with_frontmatter(*, metadata: dict, body: str) -> str:
    yaml_block = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n"


def _render_body(
    *,
    client_name: str,
    atoms: list[Atom],
    calls: list[CallNoteSummary],
    pinned: str,
    now: datetime,
) -> str:
    """Compose the context.md body. Layout:

      # <Client>
      <!-- pin -->  <user-editable region preserved across regenerations  -->
      <!-- /pin -->
      ## Profile
      ## Recent calls
      ## Atoms by type
    """
    by_type: dict[AtomType, list[Atom]] = defaultdict(list)
    for atom in atoms:
        by_type[atom.type].append(atom)

    sections: list[str] = []
    sections.append(f"# {client_name}")
    sections.append("")
    sections.append("<!-- pin -->")
    if pinned:
        sections.append(pinned)
    else:
        sections.append("_Pinned notes go here. This block is preserved across regenerations._")
    sections.append("<!-- /pin -->")
    sections.append("")

    sections.append("## Profile")
    sections.append(f"Auto-regenerated from {len(atoms)} active atom(s) across {len(calls)} recent call(s).")
    sections.append("")

    if calls:
        sections.append("## Recent calls")
        for call in calls:
            label = f"{call.date} · {call.call_type} · {call.atom_count} atom(s)"
            if call.duration_minutes:
                label += f" · {call.duration_minutes} min"
            sections.append(f"- [[{call.id}]] — {label}")
            if call.summary:
                sections.append(f"  > {call.summary}")
        sections.append("")

    if by_type:
        sections.append("## Atoms by type")
        # Stable ordering — most-impactful types first.
        ordered = [
            AtomType.objection,
            AtomType.commitment,
            AtomType.win_signal,
            AtomType.loss_signal,
            AtomType.confusion,
            AtomType.insight,
            AtomType.client_fact,
        ]
        for atom_type in ordered:
            atoms_of_type = by_type.get(atom_type) or []
            if not atoms_of_type:
                continue
            sections.append(f"### {atom_type.value} ({len(atoms_of_type)})")
            for atom in atoms_of_type[:8]:
                tag_str = (", ".join(atom.tags[:3])) if atom.tags else ""
                tag_part = f"  ·  {tag_str}" if tag_str else ""
                sections.append(f"- [[{atom.id}]] {atom.body}{tag_part}")
            if len(atoms_of_type) > 8:
                sections.append(f"  _… and {len(atoms_of_type) - 8} more_")
            sections.append("")

    sections.append(f"_Last regenerated: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}_")
    return "\n".join(sections)


def _client_name_from_slug(*, layout: VaultLayout, slug: str) -> str | None:
    """Recover the display name from any atom in the vault that uses it.
    Falls back to None — caller title-cases the slug as a last resort.
    """
    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        if atom.client and slugify_client(atom.client) == slug:
            return atom.client
    return None
