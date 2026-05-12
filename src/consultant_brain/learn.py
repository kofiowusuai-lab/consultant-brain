"""Phase 9 — knowledge ingest pipeline.

`consultant-brain learn --url <youtube|instagram URL>` runs this end
to end:
  1. Dispatch URL → `SourceMaterial` (transcript + metadata).
  2. Run the knowledge-tuned extractor against the transcript.
  3. Write a `09_Knowledge/<source_id>.md` source note.
  4. Write one Atom per extracted entry into `03_Atoms/`. Atoms carry
     `source_kind=youtube|instagram`, `source_url`, `source_title`,
     and a synthetic `call` field of `<source_id>`.
  5. Embed each atom into LanceDB (when Ollama is up).

`source_kind != "call"` is the marker retrieval uses to keep this
knowledge OUT of live-call Memory by default. The Phase 9 retrieval
toggle (`include_knowledge=True`) flips them in for prep mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from consultant_brain.extractor import (
    AnthropicClient,
    extract_from_source,
    real_anthropic_client,
)
from consultant_brain.llm.provider import LLMProvider, ProviderError
from consultant_brain.llm.registry import build_provider
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    CallType,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTOR_VERSION,
    SourceKind,
)
from consultant_brain.secrets import SecretNotFoundError, get_anthropic_key
from consultant_brain.sources import (
    KnowledgeFetchError,
    SourceMaterial,
    fetch_source,
)
from consultant_brain.vault import (
    VaultLayout,
    derive_atom_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    write_atom,
    write_knowledge_note,
)


@dataclass(frozen=True, slots=True)
class LearnResult:
    """Outcome of one `learn` run. Returned to the CLI for the summary
    line + to the HTTP service for the response payload."""

    source_id: str
    source_url: str
    source_kind: SourceKind
    knowledge_note_path: Optional[Path]
    atom_paths: list[Path]
    atom_count: int
    summary: str
    vault_root: Path
    dry_run: bool

    def summary_line(self) -> str:
        verb = "Would learn" if self.dry_run else "Learned"
        if self.knowledge_note_path:
            try:
                rel = self.knowledge_note_path.relative_to(self.vault_root)
            except ValueError:
                rel = self.knowledge_note_path
            return (
                f"{verb} {self.atom_count} atoms from "
                f"{self.source_kind.value} · note: {rel}"
            )
        return f"{verb} {self.atom_count} atoms (dry-run, nothing written)"


def run_learn(
    *,
    url: str,
    vault_root: Path,
    topic: Optional[str] = None,
    for_client: Optional[str] = None,
    allow_whisper: bool = False,
    cookies_from_browser: Optional[str] = None,
    dry_run: bool = False,
    provider: Optional[LLMProvider] = None,
    extractor_provider_name: Optional[str] = None,
    extractor_model: Optional[str] = None,
    client: Optional[AnthropicClient] = None,
    # Tests inject a pre-built SourceMaterial to skip the network leg.
    source: Optional[SourceMaterial] = None,
) -> LearnResult:
    """Ingest one source URL into the brain.

    Errors:
      - `typer.BadParameter` when the URL has no fetcher AND no
        injected `source`.
      - `KnowledgeFetchError` when the fetcher fails (no captions,
        Whisper down, private content without cookies, etc.).
      - `ExtractorError` when the LLM returns malformed JSON.
    """
    if source is None:
        try:
            source = fetch_source(
                url,
                allow_whisper=allow_whisper,
                topic=topic,
                for_client=for_client,
                cookies_from_browser=cookies_from_browser,
            )
        except KnowledgeFetchError as exc:
            raise typer.BadParameter(str(exc)) from exc

    layout = VaultLayout.for_root(vault_root)
    timestamp = datetime.now(timezone.utc)

    # Build provider (Phase 8 abstraction). Falls back to the legacy
    # AnthropicClient path when neither provider nor extractor name set
    # — keeps tests that inject a mock client working unchanged.
    active_provider: Optional[LLMProvider] = provider
    if active_provider is None and extractor_provider_name and extractor_provider_name != "anthropic":
        try:
            active_provider = build_provider(extractor_provider_name)
        except ProviderError as exc:
            raise typer.BadParameter(str(exc)) from exc

    model = extractor_model or DEFAULT_EXTRACTOR_MODEL

    if active_provider is not None:
        extractor_result = extract_from_source(
            transcript=source.transcript,
            source_title=source.title,
            author=source.author,
            topic=source.topic,
            for_client=source.for_client,
            provider=active_provider,
            model=model,
        )
    else:
        anthropic_client = client
        if anthropic_client is None:
            try:
                anthropic_client = real_anthropic_client(get_anthropic_key())
            except SecretNotFoundError as exc:
                raise typer.BadParameter(str(exc)) from exc
        extractor_result = extract_from_source(
            transcript=source.transcript,
            source_title=source.title,
            author=source.author,
            topic=source.topic,
            for_client=source.for_client,
            client=anthropic_client,
            model=model,
        )

    # Materialize atoms. `call_type` is required by the Atom schema but
    # for external knowledge it doesn't really fit — pick the closest
    # call-type analog (training calls = learning-focused) so existing
    # filters don't ignore them.
    atoms: list[Atom] = []
    base_tags: list[str] = []
    if source.topic:
        base_tags.append(source.topic.lower().replace(" ", "_"))
    base_tags.append(source.kind.value)

    seen_last = timestamp.date()
    for index, extracted in enumerate(extractor_result.atoms):
        atom_id = derive_atom_id(
            session_filename=source.source_id,
            extractor_version=EXTRACTOR_VERSION,
            atom_index=index,
        )
        combined_tags = list(dict.fromkeys([*base_tags, *extracted.tags]))
        atoms.append(
            Atom(
                id=atom_id,
                type=extracted.type,
                client=source.for_client,
                client_org_id=None,
                call=source.source_id,
                call_type=CallType.ai_training,  # closest analog; not a real call
                source_kind=source.kind,
                source_url=source.url,
                source_title=source.title,
                tags=combined_tags,
                confidence=extracted.confidence,
                evidence_count=1,
                last_seen=seen_last,
                created_at=timestamp,
                status=AtomStatus.active,
                embedding_id=atom_id,
                body=extracted.body,
            )
        )

    if dry_run:
        return LearnResult(
            source_id=source.source_id,
            source_url=source.url,
            source_kind=source.kind,
            knowledge_note_path=None,
            atom_paths=[],
            atom_count=len(atoms),
            summary=extractor_result.summary,
            vault_root=layout.root,
            dry_run=True,
        )

    ensure_vault_skeleton(layout)
    if source.for_client:
        ensure_client_folder(layout, source.for_client)

    atom_paths = [write_atom(layout, atom) for atom in atoms]
    knowledge_path = write_knowledge_note(
        layout,
        source_id=source.source_id,
        title=source.title,
        kind=source.kind.value,
        url=source.url,
        author=source.author,
        duration_seconds=source.duration_seconds,
        published_at_iso=source.published_at.isoformat() if source.published_at else None,
        fetched_at_iso=source.fetched_at.replace(microsecond=0).isoformat() + "Z",
        language=source.language,
        topic=source.topic,
        for_client=source.for_client,
        summary=extractor_result.summary,
        atom_ids=[a.id for a in atoms],
        transcript=source.transcript,
    )

    # Embed atoms. Failure here is non-fatal — markdown survives, and a
    # `reindex` pass picks up the missing rows later. Same pattern as
    # ingest.py.
    try:
        from consultant_brain.embedder import LanceVaultIndex

        index = LanceVaultIndex(layout)
        index.upsert_many(atoms)
    except Exception as exc:  # noqa: BLE001
        typer.secho(
            f"[warning] Embedding skipped: {exc}. Markdown is on disk; "
            "re-run `consultant-brain reindex` after `ollama serve && ollama pull nomic-embed-text`.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    return LearnResult(
        source_id=source.source_id,
        source_url=source.url,
        source_kind=source.kind,
        knowledge_note_path=knowledge_path,
        atom_paths=atom_paths,
        atom_count=len(atoms),
        summary=extractor_result.summary,
        vault_root=layout.root,
        dry_run=False,
    )
