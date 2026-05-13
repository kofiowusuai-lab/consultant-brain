"""Ingest user-typed notes about a client into the vault, then
regenerate the brief.

Flow when the user types a declarative statement in the dashboard's
brief chat ("he's sent the scoping answers"):

  1. The classifier in `qa.process_chat()` decides intent=note.
  2. The LLM also structures the user's free text into one or more
     atoms (type + body + tags).
  3. `ingest_chat_note()` writes those atoms into the vault as
     `source_kind=context_dump` records with `last_seen` = today and
     a synthetic call_id like `chat_note_<ULID>`.
  4. The brief is regenerated against the updated atom set so the
     dashboard renders the new state immediately.

We treat chat notes as context_dump-shaped atoms (not call-shaped)
because they're off-call. They get the user's typed text verbatim as
the atom body, so the audit trail in 03_Atoms/ shows exactly what the
user told the brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ulid

from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
    EXTRACTOR_VERSION,
    SourceKind,
)
from consultant_brain.vault import (
    VaultLayout,
    derive_atom_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    slugify_client,
    write_atom,
)


@dataclass(frozen=True, slots=True)
class StructuredNoteFact:
    """One LLM-structured fact extracted from a free-text chat note.
    Mirrors `ExtractedAtom`'s semantic fields — the orchestrator
    fills in the bookkeeping (id, dates, embedding) on write."""

    type: AtomType
    body: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestedNoteResult:
    """Outcome of one note ingestion. Returned to the chat endpoint
    so it can wire the atom IDs into the response payload + decide
    whether to regenerate the brief."""

    note_id: str
    atom_paths: list[Path]
    atom_count: int


def ingest_chat_note(
    *,
    client_name: str,
    raw_text: str,
    facts: list[StructuredNoteFact],
    vault_root: Path,
    observed_at: Optional[datetime] = None,
) -> IngestedNoteResult:
    """Persist the user's note as atoms in the vault.

    `facts` is the LLM-structured form of `raw_text`; if it's empty
    we still mint a single client_fact atom carrying the verbatim
    text so the chat-note ALWAYS produces something traceable.
    """
    if not facts:
        facts = [
            StructuredNoteFact(
                type=AtomType.client_fact,
                body=raw_text.strip(),
                tags=("chat_note",),
            )
        ]

    timestamp = observed_at or datetime.now(timezone.utc)
    note_id = f"chat_note_{ulid.ULID()}"
    layout = VaultLayout.for_root(vault_root)
    ensure_vault_skeleton(layout)
    ensure_client_folder(layout, client_name)

    slug = slugify_client(client_name)
    seen_on = timestamp.date()
    atom_paths: list[Path] = []
    base_tags = ("chat_note", "context_dump")

    for index, fact in enumerate(facts):
        atom_id = derive_atom_id(
            session_filename=note_id,
            extractor_version=EXTRACTOR_VERSION,
            atom_index=index,
        )
        combined_tags = list(
            dict.fromkeys(list(fact.tags) + list(base_tags))
        )
        atom = Atom(
            id=atom_id,
            type=fact.type,
            client=client_name,
            client_org_id=None,
            call=note_id,
            # Closest analog — chat notes aren't real calls so we use
            # consulting_call as the bucket. Retrieval's call_type
            # filter never wants to surface a chat note when the user
            # is on, say, a coldCall, so this stays consistent.
            call_type=CallType.consulting_call,
            source_kind=SourceKind.context_dump,
            source_url=None,
            source_title=f"chat note · {client_name}",
            tags=combined_tags,
            confidence=0.9,  # the user typed it themselves — trust it
            evidence_count=1,
            last_seen=seen_on,
            created_at=timestamp,
            status=AtomStatus.active,
            embedding_id=atom_id,
            body=fact.body.strip(),
        )
        atom_paths.append(write_atom(layout, atom))

    # Embed atoms into LanceDB. Same non-fatal pattern as ingest +
    # learn — markdown is on disk; a missing Ollama means embeddings
    # backfill on the next `reindex`.
    try:
        from consultant_brain.embedder import LanceVaultIndex

        index = LanceVaultIndex(layout)
        for atom_path in atom_paths:
            # Re-read via the same pipeline the writer just produced so
            # the embedder gets the exact frontmatter we just wrote.
            from consultant_brain.reindex import _atom_from_markdown

            atom = _atom_from_markdown(atom_path)
            index.upsert(atom)
    except Exception:
        pass

    return IngestedNoteResult(
        note_id=note_id,
        atom_paths=atom_paths,
        atom_count=len(atom_paths),
    )
