"""Vault writer — turns Atom / CallNote Pydantic models into markdown files
with YAML frontmatter at the right vault paths, using atomic write semantics
(write-to-tmp + fsync + rename) so a crash mid-write never corrupts a note.

Also owns:
  - Vault skeleton creation (the 9 top-level dirs + .gitignore + README).
  - Deterministic atom ID derivation (sha256(session + version + index)).
  - Slug generation for client display names and call-note IDs.

No reads — everything that needs to query the vault later (Phase 6 distillation,
watchdog re-indexing) gets its own module. This file is write-only by design.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import frontmatter
import yaml

from consultant_brain.schemas import Atom, CallNote, CallType


VAULT_SUBDIRS: tuple[str, ...] = (
    "00_System",
    "01_Clients",
    "02_Calls",
    "03_Atoms",
    "04_Patterns",
    "05_Plays",
    "06_Definitions",
    "07_People",
    "08_Reviews",
    "_Indexes",
)


# ────────────────────────────────────────────────────────────────────────────
# Vault layout
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VaultLayout:
    """Resolved paths for one vault root. Use `for_root()` instead of building
    by hand so the subdir set stays consistent across the codebase.
    """

    root: Path

    @classmethod
    def for_root(cls, root: Path) -> "VaultLayout":
        return cls(root=Path(root).expanduser().resolve())

    @property
    def system_dir(self) -> Path:
        return self.root / "00_System"

    @property
    def clients_dir(self) -> Path:
        return self.root / "01_Clients"

    @property
    def calls_dir(self) -> Path:
        return self.root / "02_Calls"

    @property
    def atoms_dir(self) -> Path:
        return self.root / "03_Atoms"

    def atom_file(self, atom_id: str) -> Path:
        return self.atoms_dir / f"{atom_id}.md"

    def call_file(self, call_id: str) -> Path:
        return self.calls_dir / f"{call_id}.md"

    def client_dir(self, client_slug: str) -> Path:
        return self.clients_dir / client_slug


def ensure_vault_skeleton(layout: VaultLayout) -> None:
    """Create the 9-subdir skeleton + a root README + .gitignore if any are
    missing. Safe to call repeatedly — it's idempotent."""
    layout.root.mkdir(parents=True, exist_ok=True)
    for sub in VAULT_SUBDIRS:
        (layout.root / sub).mkdir(parents=True, exist_ok=True)

    readme = layout.root / "README.md"
    if not readme.exists():
        readme.write_text(_ROOT_README, encoding="utf-8")

    gitignore = layout.root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_ROOT_GITIGNORE, encoding="utf-8")


_ROOT_README = """\
# ConsultantBrain

Auto-managed vault for the Consultant Copilot active brain. Each top-level
folder has a specific role:

- **00_System/** — schema docs, LanceDB index, scoring weights. Owned by code, not humans.
- **01_Clients/<Name>/** — per-client folder. `context.md` is auto-regenerated; don't hand-edit.
- **02_Calls/** — one note per ingested call. Each links to the atoms extracted.
- **03_Atoms/** — atomic typed notes. 1-3 sentences each. Wikilinked freely.
- **04_Patterns/** — distilled rules (Phase 6). Empty in Phase 1.
- **05_Plays/** — reusable openers, frames, objection handles (Phase 6). Empty in Phase 1.
- **06_Definitions/** — surfaceable terms (Phase 6). Empty in Phase 1.
- **07_People/** — stakeholder graph (Phase 6). Empty in Phase 1.
- **08_Reviews/** — weekly summaries (Phase 6). Empty in Phase 1.
- **_Indexes/** — auto-generated MOCs (Phase 6).

Open this folder in Obsidian. Backlinks + graph view will reveal the call → atom structure.
"""

_ROOT_GITIGNORE = """\
00_System/lancedb/
.obsidian/workspace*
.obsidian/cache
*.lance/
"""


# ────────────────────────────────────────────────────────────────────────────
# Deterministic atom IDs
# ────────────────────────────────────────────────────────────────────────────


def derive_atom_id(*, session_filename: str, extractor_version: int, atom_index: int) -> str:
    """SHA-256 of (session, version, index) truncated to 26 chars.

    Why deterministic: re-ingesting the same session with the same extractor
    version produces the same atom IDs, so the vault writer can overwrite in
    place rather than duplicate. Bumping `extractor_version` invalidates all
    old derivations and forces fresh atoms — useful when the extractor prompt
    changes meaningfully.
    """
    seed = f"{session_filename}|v{extractor_version}|atom{atom_index}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    # 26 chars matches ULID length so Obsidian's note-title display stays consistent.
    return digest[:26].upper()


def derive_call_id(*, client_name: str, call_type: CallType, call_date: datetime) -> str:
    """`YYYY-MM-DD_<client-slug>_<callType>`. Matches CallNote.id pattern."""
    return f"{call_date:%Y-%m-%d}_{slugify_client(client_name)}_{call_type.value}"


def slugify_client(name: str) -> str:
    """Lowercase + ascii-only + non-alphanumerics → underscores. Empty input
    raises rather than producing an empty slug (which would collide).
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not cleaned:
        raise ValueError(f"Client name does not produce a valid slug: {name!r}")
    return cleaned


# ────────────────────────────────────────────────────────────────────────────
# Atomic writes
# ────────────────────────────────────────────────────────────────────────────


def _atomic_write(path: Path, content: str) -> None:
    """write-to-tmp + fsync + rename. Crash safety: a half-written file never
    appears at `path` — either the old file is there, or the fully-written
    new file is. Same pattern the Swift app uses for its session JSONs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile in the same directory so rename() is atomic on APFS.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix + ".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the .tmp; swallow the unlink error so the
        # original exception bubbles up clearly.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ────────────────────────────────────────────────────────────────────────────


def _render_atom_markdown(atom: Atom) -> str:
    """Atom frontmatter + body. Client and call rendered as wikilinks
    (`[[Reece]]`, `[[2026-05-12_reece_consultingCall]]`) so Obsidian's
    backlinks panel resolves them automatically.
    """
    fm: dict = {
        "id": atom.id,
        "type": atom.type.value,
        "client": f"[[{atom.client}]]" if atom.client else None,
        "call": f"[[{atom.call}]]",
        "call_type": atom.call_type.value,
        "tags": list(atom.tags),
        "confidence": atom.confidence,
        "evidence_count": atom.evidence_count,
        "last_seen": atom.last_seen.isoformat(),
        "created_at": _iso_utc(atom.created_at),
        "status": atom.status.value,
        "embedding_id": atom.embedding_id,
    }
    # Drop None-valued keys so the frontmatter doesn't show `client: null`.
    fm = {k: v for k, v in fm.items() if v is not None}
    return _dump_frontmatter(fm, atom.body)


def _render_call_note_markdown(note: CallNote) -> str:
    """Call-note frontmatter + body. Body has Summary section, atom-link
    list, and the full transcript inside `<details>`.
    """
    fm: dict = {
        "id": note.id,
        "client": f"[[{note.client}]]" if note.client else None,
        "call_type": note.call_type.value,
        "date": note.date.isoformat(),
        "duration_minutes": note.duration_minutes,
        "source_session": note.source_session,
        "extractor_model": note.extractor_model,
        "extractor_version": note.extractor_version,
        "atom_count": note.atom_count,
        "created_at": _iso_utc(note.created_at),
    }
    fm = {k: v for k, v in fm.items() if v is not None}

    title = _call_note_title(note)
    atom_links_section = (
        "\n".join(f"- [[{atom_id}]]" for atom_id in note.atom_ids)
        if note.atom_ids
        else "_No atoms extracted from this call._"
    )

    body = f"""# {title}

## Summary
{note.summary.strip()}

## Atoms
{atom_links_section}

## Transcript
<details>
<summary>Full transcript</summary>

{note.transcript.strip()}

</details>
"""
    return _dump_frontmatter(fm, body)


def _call_note_title(note: CallNote) -> str:
    client_label = note.client or "Untitled client"
    return f"{client_label} — {note.call_type.value}, {note.date.strftime('%-d %b %Y')}"


def _dump_frontmatter(metadata: dict, body: str) -> str:
    """Wrap body + metadata as a YAML-frontmatter markdown doc. Uses
    `yaml.safe_dump` directly so we control key order (insertion order, the
    same order Pydantic gives) instead of frontmatter's defaults.
    """
    yaml_block = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n"


def _iso_utc(dt: datetime) -> str:
    """ISO-8601 in UTC with a trailing Z. Naive datetimes are assumed UTC
    (warned via the type system — we never construct them in this codebase)."""
    from datetime import timezone as _tz
    if dt.tzinfo is None:
        utc = dt.replace(tzinfo=_tz.utc)
    else:
        utc = dt.astimezone(_tz.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────────────
# Public write API
# ────────────────────────────────────────────────────────────────────────────


def write_atom(layout: VaultLayout, atom: Atom) -> Path:
    """Write one atom to `<vault>/03_Atoms/<id>.md`. Overwrites if it exists
    (idempotent by atom ID, which is itself deterministic)."""
    layout.atoms_dir.mkdir(parents=True, exist_ok=True)
    path = layout.atom_file(atom.id)
    _atomic_write(path, _render_atom_markdown(atom))
    return path


def write_call_note(layout: VaultLayout, note: CallNote) -> Path:
    """Write one call note to `<vault>/02_Calls/<id>.md`. Overwrites if it
    exists (idempotent by call ID)."""
    layout.calls_dir.mkdir(parents=True, exist_ok=True)
    path = layout.call_file(note.id)
    _atomic_write(path, _render_call_note_markdown(note))
    return path


def ensure_client_folder(layout: VaultLayout, client_name: str) -> Path:
    """Create `<vault>/01_Clients/<slug>/` if missing. Returns the path. The
    folder gets a placeholder context.md in Phase 6; for now it stays empty
    so Obsidian-side wikilinks `[[<Name>]]` resolve."""
    slug = slugify_client(client_name)
    client_dir = layout.client_dir(slug)
    client_dir.mkdir(parents=True, exist_ok=True)
    # The wikilink `[[Reece]]` resolves to either a file named Reece.md or a
    # folder named Reece/ with an index file. We write a minimal placeholder
    # so Obsidian's "create on click" doesn't lose state.
    placeholder = client_dir / f"{client_name}.md"
    if not placeholder.exists():
        _atomic_write(
            placeholder,
            _dump_frontmatter(
                {"client_slug": slug, "auto_generated": True},
                f"# {client_name}\n\n_Auto-managed placeholder. Phase 6 will populate this with a summary._",
            ),
        )
    return client_dir


# ────────────────────────────────────────────────────────────────────────────
# Read-back helpers (for tests + idempotency checks)
# ────────────────────────────────────────────────────────────────────────────


def read_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a vault file. Returns the metadata dict.

    Used by tests to assert what we wrote, and by Phase 6 distillation to
    inspect existing atoms without re-reading every body.
    """
    with path.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)
    return dict(post.metadata)
