"""Run-preview + commit for context dumps.

`run_preview()` does the heavy lift (parse → extract) and returns a
`ContextDumpPreview` the user can inspect. `commit_preview()` writes
the dump note + atoms to the vault.

The preview is intentionally separate from the commit — the user
should see what the brain extracted before any vault state changes.
This is the core Phase-10 contract.
"""

from __future__ import annotations

import tempfile
import typing
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import ulid

from consultant_brain.context_dumps.parsers import (
    ParsedDoc,
    UnsupportedFileError,
    parse_to_text,
)
from consultant_brain.extractor import (
    ExtractorError,
    extract_from_context_dump,
)
from consultant_brain.llm.provider import LLMProvider
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    CallType,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTOR_VERSION,
    ExtractedAtom,
    ExtractorResult,
    SourceKind,
)
from consultant_brain.vault import (
    VaultLayout,
    derive_atom_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    slugify_client,
    write_atom,
    write_context_dump_note,
)


RAW_EXCERPT_CHARS = 4000


@dataclass(frozen=True, slots=True)
class ContextDumpPreview:
    """One in-flight context dump waiting for the user's confirmation.

    `atoms` are `ExtractedAtom` (semantic-only fields) — the orchestrator
    converts them to full `Atom` records at commit time so atom IDs
    stay deterministic from the preview ID.
    """

    preview_id: str
    client_name: str
    client_slug: str
    observed_at: date
    uploaded_at: datetime
    source_filename: str
    source_kind_label: str
    raw_text_excerpt: str
    raw_text_full_path: Path
    summary: str
    atoms: tuple[ExtractedAtom, ...]
    notes: Optional[str]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContextDumpCommitResult:
    """Outcome of committing a preview."""

    preview_id: str
    dump_note_path: Path
    atom_paths: list[Path]
    atom_count: int
    client_slug: str


def run_preview(
    *,
    file_path: Path,
    client_name: str,
    observed_at: date,
    notes: Optional[str],
    vault_root: Path,
    provider: Optional[LLMProvider] = None,
    client=None,
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
    today: Optional[date] = None,
) -> ContextDumpPreview:
    """Parse the file, run the context-dump extractor, return a
    preview — vault is NOT touched.

    The original text is stashed in a long-lived tmp file referenced
    by the preview so commit can re-read it without re-running parsers.
    The store's TTL eviction sweeps the file on expiry.
    """
    if provider is None and client is None:
        raise ExtractorError("run_preview() requires `provider` or `client`")
    if not file_path.exists():
        raise FileNotFoundError(f"Context-dump file not found: {file_path}")

    parsed = parse_to_text(file_path)

    if not parsed.text.strip():
        # Nothing to extract — still return a preview so the user can
        # see the warnings and decide what to do next.
        return _empty_preview(
            file_path=file_path,
            client_name=client_name,
            observed_at=observed_at,
            notes=notes,
            parsed=parsed,
        )

    summary, atoms = _run_extractor(
        parsed=parsed,
        file_path=file_path,
        client_name=client_name,
        notes=notes,
        observed_at=observed_at,
        today=today,
        provider=provider,
        client=client,
        model=extractor_model,
    )

    raw_path = _persist_raw_text(parsed.text)
    return ContextDumpPreview(
        preview_id=str(ulid.ULID()),
        client_name=client_name,
        client_slug=slugify_client(client_name),
        observed_at=observed_at,
        uploaded_at=datetime.now(timezone.utc),
        source_filename=file_path.name,
        source_kind_label=parsed.kind_label,
        raw_text_excerpt=parsed.text[:RAW_EXCERPT_CHARS],
        raw_text_full_path=raw_path,
        summary=summary,
        atoms=tuple(atoms),
        notes=notes,
        warnings=tuple(parsed.warnings),
    )


def commit_preview(
    *,
    preview: ContextDumpPreview,
    vault_root: Path,
    accepted_atom_indexes: Optional[list[int]] = None,
) -> ContextDumpCommitResult:
    """Materialize the preview's atoms + dump note into the vault.

    `accepted_atom_indexes` defaults to every atom; pass a subset
    (e.g. `[0, 2, 4]`) to drop the others before commit. The dropped
    atoms never touch disk.
    """
    layout = VaultLayout.for_root(vault_root)
    ensure_vault_skeleton(layout)
    ensure_client_folder(layout, preview.client_name)

    full_text = _load_raw_text(preview.raw_text_full_path)

    indexes = (
        accepted_atom_indexes
        if accepted_atom_indexes is not None
        else list(range(len(preview.atoms)))
    )
    kept_extracted = [
        preview.atoms[i]
        for i in indexes
        if 0 <= i < len(preview.atoms)
    ]

    atoms: list[Atom] = []
    for index, extracted in enumerate(kept_extracted):
        atom_id = derive_atom_id(
            session_filename=f"context_dump::{preview.preview_id}",
            extractor_version=EXTRACTOR_VERSION,
            atom_index=index,
        )
        atoms.append(
            Atom(
                id=atom_id,
                type=extracted.type,
                client=preview.client_name,
                client_org_id=None,
                call=preview.preview_id,
                call_type=CallType.consulting_call,  # closest analog; not a real call
                source_kind=SourceKind.context_dump,
                source_url=None,
                source_title=preview.source_filename,
                tags=list(dict.fromkeys([*extracted.tags, "context_dump"])),
                confidence=extracted.confidence,
                evidence_count=1,
                last_seen=preview.observed_at,
                created_at=preview.uploaded_at,
                status=AtomStatus.active,
                embedding_id=atom_id,
                body=extracted.body,
            )
        )

    atom_paths = [write_atom(layout, atom) for atom in atoms]
    dump_note_path = write_context_dump_note(
        layout,
        dump_id=preview.preview_id,
        client_name=preview.client_name,
        client_slug=preview.client_slug,
        observed_at=preview.observed_at,
        uploaded_at_iso=_iso_z(preview.uploaded_at),
        source_filename=preview.source_filename,
        source_kind_label=preview.source_kind_label,
        summary=preview.summary,
        atom_ids=[a.id for a in atoms],
        raw_text=full_text,
        notes=preview.notes,
        warnings=list(preview.warnings),
    )

    # Embed — non-fatal on failure (same pattern as ingest.py).
    try:
        from consultant_brain.embedder import LanceVaultIndex

        index = LanceVaultIndex(layout)
        index.upsert_many(atoms)
    except Exception:
        # Markdown survives; reindex will pick these up later.
        pass

    # Tmp file isn't needed once we've written the note's <details>
    # section — clean up immediately on successful commit.
    try:
        preview.raw_text_full_path.unlink(missing_ok=True)
    except Exception:
        pass

    return ContextDumpCommitResult(
        preview_id=preview.preview_id,
        dump_note_path=dump_note_path,
        atom_paths=atom_paths,
        atom_count=len(atoms),
        client_slug=preview.client_slug,
    )


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _run_extractor(
    *,
    parsed: ParsedDoc,
    file_path: Path,
    client_name: str,
    notes: Optional[str],
    observed_at: date,
    today: Optional[date],
    provider: Optional[LLMProvider],
    client,
    model: str,
) -> tuple[str, list[ExtractedAtom]]:
    result: ExtractorResult = extract_from_context_dump(
        text=parsed.text,
        client_name=client_name,
        source_kind_label=parsed.kind_label,
        source_filename=file_path.name,
        notes=notes,
        observed_at=observed_at,
        today=today,
        provider=provider,
        client=client,
        model=model,
    )
    return result.summary, list(result.atoms)


def _empty_preview(
    *,
    file_path: Path,
    client_name: str,
    observed_at: date,
    notes: Optional[str],
    parsed: ParsedDoc,
) -> ContextDumpPreview:
    """Build a preview with zero atoms — for when parsing yielded no
    text (image-only PDF, empty audio, all-garbage zip). The UI surfaces
    the warnings so the user knows why."""
    return ContextDumpPreview(
        preview_id=str(ulid.ULID()),
        client_name=client_name,
        client_slug=slugify_client(client_name),
        observed_at=observed_at,
        uploaded_at=datetime.now(timezone.utc),
        source_filename=file_path.name,
        source_kind_label=parsed.kind_label,
        raw_text_excerpt="",
        raw_text_full_path=_persist_raw_text(""),
        summary="No content extracted from this file.",
        atoms=tuple(),
        notes=notes,
        warnings=tuple(parsed.warnings),
    )


def _persist_raw_text(text: str) -> Path:
    """Write the parsed text to a tmp file we can read back at commit
    time. Tmp dir is `tempfile.gettempdir()/consultant-brain-dumps/`
    so the files share a parent for easy cleanup."""
    base = Path(tempfile.gettempdir()) / "consultant-brain-dumps"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{ulid.ULID()}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _load_raw_text(path: Path) -> str:
    """Read the tmp file back. Missing file = empty content (the
    preview was created without text, e.g. an empty image)."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _iso_z(dt: datetime) -> str:
    """ISO with trailing Z for the dump note's frontmatter."""
    aware = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return aware.strftime("%Y-%m-%dT%H:%M:%SZ")
