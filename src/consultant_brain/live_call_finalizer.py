"""Convert a CallState into a CallNote on disk.

Called by /call_end so the post-call scoring + retrieval surfaces can
read the call back as a normal vault entry. Bridges the Phase 4 live
state (in-memory transcript + opportunistically-detected atoms) into the
Phase 1 vault model (markdown call note + atom files + LanceDB rows).

The summary field is left as "(generated live; run `consultant-brain ingest`
for the full Claude-extracted summary)" so callers know the post-call
extractor didn't run yet. Phase 6 distillation can replace this with a
richer summary later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from consultant_brain.live_state import CallState
from consultant_brain.schemas import (
    CallNote,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTOR_VERSION,
    Speaker,
)
from consultant_brain.vault import (
    VaultLayout,
    derive_call_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    read_frontmatter,
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


def finalize_live_call(
    *,
    state: CallState,
    vault_root: Path,
    now: datetime | None = None,
) -> FinalizeResult:
    """Write a CallNote markdown file for one ended live call.

    Steps:
      1. Resolve the canonical call_note_id from state's client + call_type
         + start date (same derivation Phase 4 used when stamping live atoms).
      2. Format the rolling transcript into the You:/Them: block.
      3. Collect the atom IDs the live loop wrote with this call_note_id.
      4. Write the call note via vault.py's atomic markdown writer.

    Returns the resolved IDs/paths so /call_end can surface them to the
    Swift app for the score-override sheet.
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

    call_note = CallNote(
        id=call_note_id,
        client=state.client,
        call_type=state.call_type,
        date=state.started_at.date(),
        duration_minutes=duration_minutes,
        source_session=f"live::{state.call_id}",
        extractor_model=DEFAULT_EXTRACTOR_MODEL,
        extractor_version=EXTRACTOR_VERSION,
        atom_count=len(atom_ids),
        created_at=completed_at,
        summary=PLACEHOLDER_SUMMARY,
        atom_ids=atom_ids,
        transcript=transcript,
    )
    path = write_call_note(layout, call_note)

    return FinalizeResult(
        call_note_id=call_note_id,
        call_note_path=path,
        atom_count=len(atom_ids),
        transcript_chars=len(transcript),
    )


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
