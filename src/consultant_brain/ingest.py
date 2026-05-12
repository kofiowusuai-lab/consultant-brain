"""End-to-end ingest pipeline. Glues session_loader → extractor → vault.

`run_ingest()` is the single function the CLI calls. Returns an
`IngestResult` so the CLI can print a one-line summary and exit cleanly.

Phase 1 keeps this file thin — no embeddings yet (those land in
embedder.py with the query subcommand), no idempotency check beyond what
derive_atom_id already guarantees structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer

from consultant_brain.crm.resolver import CRMResolver
from consultant_brain.extractor import (
    AnthropicClient,
    extract,
    real_anthropic_client,
)
from consultant_brain.llm.provider import LLMProvider, ProviderError
from consultant_brain.llm.registry import build_provider, default_model_for
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    CallNote,
    CallType,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTOR_VERSION,
    IngestConfig,
)
from consultant_brain.secrets import get_anthropic_key
from consultant_brain.session_loader import LoadedSession, load_session
from consultant_brain.vault import (
    VaultLayout,
    derive_atom_id,
    derive_call_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    write_atom,
    write_call_note,
)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of one ingest run. Printed by the CLI; consumed by Phase 7
    metrics in the future."""

    call_note_path: Path | None
    atom_paths: list[Path]
    atom_count: int
    summary: str
    vault_root: Path
    dry_run: bool

    def summary_line(self) -> str:
        verb = "Would extract" if self.dry_run else "Extracted"
        if self.call_note_path:
            return (
                f"{verb} {self.atom_count} atoms · "
                f"call note: {self.call_note_path.relative_to(self.vault_root)} · "
                f"vault: {self.vault_root}"
            )
        return f"{verb} {self.atom_count} atoms (dry-run, nothing written)"


def run_ingest(
    *,
    session_path: Path,
    client_name: str,
    call_type: str,
    vault_root: Path,
    redact: bool,
    dry_run: bool,
    client: AnthropicClient | None = None,
    provider: LLMProvider | None = None,
    extractor_provider_name: str | None = None,
    extractor_model: str | None = None,
    crm_resolver: CRMResolver | None = None,
) -> IngestResult:
    """Ingest one session JSON into the vault.

    Steps:
      1. Validate call_type against the CallType enum (CLI flag is a string).
      2. Load + normalize the session JSON.
      3. Optionally redact client name from the transcript before extraction.
      4. Send to Claude → ExtractorResult.
      5. Materialize Atom + CallNote models (vault writer attaches bookkeeping).
      6. Write to vault (unless dry-run).

    Raises:
      - typer.BadParameter for bad CLI args (invalid call_type, missing key, etc.).
      - FileNotFoundError / ValueError from session_loader on bad session JSON.
      - ExtractorError on Claude failures.
    """
    try:
        call_type_enum = CallType(call_type)
    except ValueError as exc:
        valid = ", ".join(t.value for t in CallType)
        raise typer.BadParameter(
            f"Unknown call type: {call_type!r}. Valid: {valid}"
        ) from exc

    config = IngestConfig(client_name=client_name, call_type=call_type_enum, redact=redact, dry_run=dry_run)

    # Phase 8: try to link this client to the Swift CRM. Resolver returns
    # None when the CRM doesn't have the client yet — atoms still write,
    # just without a stable UUID. A later distill pass can backfill if
    # the client gets added in the CRM.
    resolver = crm_resolver or CRMResolver()
    org = resolver.resolve(client_name)
    client_org_id = org.id if org else None

    loaded = load_session(session_path)
    transcript_for_extraction = _maybe_redact(loaded.transcript, client_name) if redact else loaded.transcript

    # Phase 8 wiring: prefer the provider path, fall back to the legacy
    # AnthropicClient one. Resolution order for `provider`:
    #   1. caller passed one explicitly
    #   2. caller passed --extractor-provider name
    #   3. BRAIN_LLM_PROVIDER env var
    #   4. anthropic default
    active_provider: LLMProvider | None = provider
    if active_provider is None and (extractor_provider_name or "anthropic") != "anthropic":
        try:
            active_provider = build_provider(extractor_provider_name)
        except ProviderError as exc:
            raise typer.BadParameter(str(exc)) from exc

    model = extractor_model or config.extractor_model
    if active_provider is not None and extractor_model is None and extractor_provider_name:
        # Use the provider's sensible default model unless caller pinned one.
        model = default_model_for(extractor_provider_name) or config.extractor_model

    if active_provider is not None:
        extractor_result = extract(
            transcript=transcript_for_extraction,
            call_type=call_type_enum,
            client_name=None if redact else client_name,
            provider=active_provider,
            model=model,
        )
    else:
        anthropic_client = client or real_anthropic_client(get_anthropic_key())
        extractor_result = extract(
            transcript=transcript_for_extraction,
            call_type=call_type_enum,
            client_name=None if redact else client_name,
            client=anthropic_client,
            model=model,
        )

    layout = VaultLayout.for_root(vault_root)
    call_id = derive_call_id(
        client_name=client_name,
        call_type=call_type_enum,
        call_date=loaded.started_at,
    )

    atoms: list[Atom] = []
    for index, extracted in enumerate(extractor_result.atoms):
        atom_id = derive_atom_id(
            session_filename=loaded.source_filename,
            extractor_version=config.extractor_version,
            atom_index=index,
        )
        atoms.append(
            Atom(
                id=atom_id,
                type=extracted.type,
                client=client_name,
                client_org_id=client_org_id,
                call=call_id,
                call_type=call_type_enum,
                tags=list(extracted.tags),
                confidence=extracted.confidence,
                evidence_count=1,
                last_seen=loaded.started_at.date(),
                created_at=config.now,
                status=AtomStatus.active,
                embedding_id=atom_id,
                body=extracted.body,
            )
        )

    call_note = CallNote(
        id=call_id,
        client=client_name,
        call_type=call_type_enum,
        date=loaded.started_at.date(),
        duration_minutes=loaded.duration_minutes,
        source_session=loaded.source_filename,
        extractor_model=config.extractor_model,
        extractor_version=config.extractor_version,
        atom_count=len(atoms),
        created_at=config.now,
        summary=extractor_result.summary,
        atom_ids=[atom.id for atom in atoms],
        transcript=loaded.transcript,
    )

    if dry_run:
        return IngestResult(
            call_note_path=None,
            atom_paths=[],
            atom_count=len(atoms),
            summary=call_note.summary,
            vault_root=layout.root,
            dry_run=True,
        )

    ensure_vault_skeleton(layout)
    ensure_client_folder(layout, client_name)
    atom_paths = [write_atom(layout, atom) for atom in atoms]
    call_note_path = write_call_note(layout, call_note)

    # Embed every atom into LanceDB. Lazy import so unit tests that mock the
    # whole pipeline don't have to spin up Ollama just to verify file writes.
    # On failure (Ollama down, model not pulled) we log + continue — markdown
    # has already been written; re-running ingest will pick up the embedding.
    try:
        from consultant_brain.embedder import LanceVaultIndex

        index = LanceVaultIndex(layout)
        index.upsert_many(atoms)
    except Exception as exc:  # noqa: BLE001 — surface to operator without losing the write
        typer.secho(
            f"[warning] Embedding skipped: {exc}. Markdown is on disk; "
            "re-run ingest after `ollama serve && ollama pull nomic-embed-text`.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    return IngestResult(
        call_note_path=call_note_path,
        atom_paths=atom_paths,
        atom_count=len(atoms),
        summary=call_note.summary,
        vault_root=layout.root,
        dry_run=False,
    )


def _maybe_redact(transcript: str, client_name: str) -> str:
    """Replace `client_name` in the transcript with `[CLIENT_1]` so the
    transcript that leaves the machine doesn't carry the client's identity.
    Case-insensitive whole-word match — won't touch substrings.
    """
    import re

    pattern = re.compile(rf"\b{re.escape(client_name)}\b", flags=re.IGNORECASE)
    return pattern.sub("[CLIENT_1]", transcript)
