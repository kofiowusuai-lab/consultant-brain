"""Convert a CallState into a CallNote on disk.

Called by /call_end so the post-call scoring + retrieval surfaces can
read the call back as a normal vault entry. Bridges the Phase 4 live
state (in-memory transcript + opportunistically-detected atoms) into the
Phase 1 vault model (markdown call note + atom files + LanceDB rows).

Phase 8 item 5 upgrade: when a `provider` is supplied, finalization runs
the full atom extractor on the live transcript so the call note carries
a real Claude-written summary plus any atoms the live moment-detector
missed. The legacy placeholder string is kept only as the fallback when
extraction fails or no provider is configured — never the happy path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from consultant_brain.extractor import ExtractorError, extract
from consultant_brain.live_state import CallState
from consultant_brain.llm.provider import LLMProvider, ProviderError
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    CallNote,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTOR_VERSION,
    ExtractorResult,
    Speaker,
)
from consultant_brain.vault import (
    VaultLayout,
    derive_atom_id,
    derive_call_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    read_frontmatter,
    write_atom,
    write_call_note,
)


PLACEHOLDER_SUMMARY = (
    "(Live-finalized summary placeholder. Run `consultant-brain ingest` against "
    "the saved Sessions/*.json for a full Claude-extracted summary.)"
)


@dataclass(frozen=True, slots=True)
class FinalizeResult:
    """Outcome of /call_end's finalize step."""

    call_note_id: str
    call_note_path: Path
    atom_count: int
    transcript_chars: int
    summary: str
    extracted_post_call: bool


def finalize_live_call(
    *,
    state: CallState,
    vault_root: Path,
    now: datetime | None = None,
    provider: LLMProvider | None = None,
    client_org_id=None,
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
    crm_org_id_resolver=None,
) -> FinalizeResult:
    """Write a CallNote markdown file for one ended live call.

    Steps:
      1. Resolve the canonical call_note_id from state's client + call_type
         + start date (same derivation Phase 4 used when stamping live atoms).
      2. Format the rolling transcript into the You:/Them: block.
      3. Collect the atom IDs the live loop wrote with this call_note_id.
      4. If `provider` is set, run the full atom extractor on the live
         transcript — real summary + any atoms the live loop missed.
         When extraction fails (network blip, transcript too short),
         degrade to the placeholder summary; never crash /call_end.
      5. Write the call note via vault.py's atomic markdown writer.

    Returns the resolved IDs/paths + summary so /call_end can surface
    them to the Swift app for the score-override sheet.
    """
    completed_at = now or datetime.now(timezone.utc)
    layout = VaultLayout.for_root(vault_root)
    ensure_vault_skeleton(layout)
    if state.client:
        ensure_client_folder(layout, state.client)

    call_note_id = derive_call_id(
        client_name=state.client or "unknown",
        call_type=state.call_type,
        call_date=state.started_at,
    )

    transcript = state.transcript_window()
    atom_ids = _atom_ids_for_call(layout=layout, call_note_id=call_note_id)
    duration_minutes = max(0, int((completed_at - state.started_at).total_seconds() // 60))

    # Resolve org id (may be passed by caller or resolved here via the
    # optional resolver). Falling back to None is fine — atoms without
    # a UUID still write; distill can backfill later.
    org_uuid = client_org_id
    if org_uuid is None and crm_org_id_resolver is not None and state.client:
        try:
            org_uuid = crm_org_id_resolver(state.client)
        except Exception:
            org_uuid = None

    summary, new_atom_ids, extracted = _maybe_extract_post_call(
        state=state,
        transcript=transcript,
        provider=provider,
        layout=layout,
        call_note_id=call_note_id,
        completed_at=completed_at,
        extractor_model=extractor_model,
        already_atom_ids=set(atom_ids),
        client_org_id=org_uuid,
    )
    atom_ids = sorted(set(atom_ids) | set(new_atom_ids))

    call_note = CallNote(
        id=call_note_id,
        client=state.client,
        call_type=state.call_type,
        date=state.started_at.date(),
        duration_minutes=duration_minutes,
        source_session=f"live::{state.call_id}",
        extractor_model=extractor_model,
        extractor_version=EXTRACTOR_VERSION,
        atom_count=len(atom_ids),
        created_at=completed_at,
        summary=summary,
        atom_ids=atom_ids,
        transcript=transcript,
    )
    path = write_call_note(layout, call_note)

    return FinalizeResult(
        call_note_id=call_note_id,
        call_note_path=path,
        atom_count=len(atom_ids),
        transcript_chars=len(transcript),
        summary=summary,
        extracted_post_call=extracted,
    )


def _maybe_extract_post_call(
    *,
    state: CallState,
    transcript: str,
    provider: LLMProvider | None,
    layout: VaultLayout,
    call_note_id: str,
    completed_at: datetime,
    extractor_model: str,
    already_atom_ids: set[str],
    client_org_id,
) -> tuple[str, list[str], bool]:
    """Run the post-call atom extractor and persist any newly minted
    atoms. Returns (summary, new_atom_ids, did_extract).
    """
    if provider is None or not transcript.strip():
        return PLACEHOLDER_SUMMARY, [], False
    try:
        result: ExtractorResult = extract(
            transcript=transcript,
            call_type=state.call_type,
            client_name=state.client,
            provider=provider,
            model=extractor_model,
        )
    except (ExtractorError, ProviderError):
        return PLACEHOLDER_SUMMARY, [], False
    except Exception:
        return PLACEHOLDER_SUMMARY, [], False

    new_atom_ids: list[str] = []
    for index, extracted in enumerate(result.atoms):
        atom_id = derive_atom_id(
            session_filename=f"live::{state.call_id}",
            extractor_version=EXTRACTOR_VERSION,
            atom_index=index,
        )
        if atom_id in already_atom_ids:
            continue
        atom = Atom(
            id=atom_id,
            type=extracted.type,
            client=state.client or "unknown",
            client_org_id=client_org_id,
            call=call_note_id,
            call_type=state.call_type,
            tags=list(extracted.tags),
            confidence=extracted.confidence,
            evidence_count=1,
            last_seen=state.started_at.date(),
            created_at=completed_at,
            status=AtomStatus.active,
            embedding_id=atom_id,
            body=extracted.body,
        )
        write_atom(layout, atom)
        new_atom_ids.append(atom_id)

    return result.summary, new_atom_ids, True


def _atom_ids_for_call(*, layout: VaultLayout, call_note_id: str) -> list[str]:
    """Scan 03_Atoms/ for atoms whose frontmatter `call` field points at
    this call note. Sorted by id so output is deterministic.
    """
    matches: list[str] = []
    expected_link = f"[[{call_note_id}]]"
    if not layout.atoms_dir.exists():
        return matches
    for atom_path in layout.atoms_dir.glob("*.md"):
        try:
            meta = read_frontmatter(atom_path)
        except Exception:
            continue
        if meta.get("call") == expected_link:
            atom_id = meta.get("id")
            if isinstance(atom_id, str):
                matches.append(atom_id)
    return sorted(matches)
