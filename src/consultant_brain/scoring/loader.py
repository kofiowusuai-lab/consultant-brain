"""Load a finished call's scoring inputs from the vault.

`load_call_for_scoring(call_id)` parses the call note + all linked atoms
and returns the bundle the feature extractor consumes. Used by both the
`score` CLI and the FastAPI `/score` endpoint.

The transcript inside the call note lives between `<details>` tags; we
parse it back into `TranscriptTurnLite` objects so the talk-ratio
feature has real data to work with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, CallType, Speaker
from consultant_brain.scoring.features import TranscriptTurnLite
from consultant_brain.vault import VaultLayout


@dataclass(frozen=True, slots=True)
class CallScoringInputs:
    """Everything the scorer needs for one call."""

    call_id: str
    call_type: CallType
    client: str | None
    atoms: list[Atom]
    turns: list[TranscriptTurnLite]
    primary_win: str | None
    summary: str  # the extractor's 3-line summary; surfaced in the CLI


class CallNotFoundError(LookupError):
    """Call note doesn't exist in the vault — surface a clear message."""


def load_call_for_scoring(
    *,
    vault_root: Path,
    call_id: str,
    primary_win: str | None = None,
) -> CallScoringInputs:
    """Resolve a call note + every atom it linked, plus parse the transcript
    block back into turn structs."""
    layout = VaultLayout.for_root(vault_root)
    call_path = layout.call_file(call_id)
    if not call_path.exists():
        raise CallNotFoundError(
            f"Call note {call_id!r} not found at {call_path}. "
            "Either it hasn't been ingested yet or you're pointing at the wrong vault."
        )
    post = frontmatter.load(call_path.open("r", encoding="utf-8"))
    meta = dict(post.metadata)
    body = post.content or ""

    call_type = CallType(meta["call_type"])
    client_raw = meta.get("client")
    client = _strip_wikilink(client_raw)
    summary = _extract_summary_section(body)
    turns = _extract_transcript_turns(body)
    atoms = _load_linked_atoms(layout=layout, call_note_id=call_id, body=body)

    return CallScoringInputs(
        call_id=call_id,
        call_type=call_type,
        client=client,
        atoms=atoms,
        turns=turns,
        primary_win=primary_win,
        summary=summary,
    )


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


_SUMMARY_RE = re.compile(r"##\s*Summary\s*\n+(.+?)(?=\n##|\Z)", re.DOTALL)


def _extract_summary_section(body: str) -> str:
    match = _SUMMARY_RE.search(body)
    if not match:
        return ""
    return match.group(1).strip()


# Pulls "You:" / "Them:" lines out of the call note's <details> block.
# Bullet-list atom links (`- [[01HX...]]`) are skipped — the atoms list
# section uses the same line shape but with leading `- [[`.
_TURN_LINE_RE = re.compile(r"^(You|Them):\s+(.+)$", re.MULTILINE)


def _extract_transcript_turns(body: str) -> list[TranscriptTurnLite]:
    # Restrict the regex to text inside <details> ... </details> so we never
    # accidentally pick up a "You:" appearing elsewhere in the body.
    details_match = re.search(r"<details>(.+?)</details>", body, re.DOTALL)
    target = details_match.group(1) if details_match else body
    out: list[TranscriptTurnLite] = []
    for match in _TURN_LINE_RE.finditer(target):
        label, text = match.group(1), match.group(2).strip()
        speaker = Speaker.you if label == "You" else Speaker.them
        out.append(TranscriptTurnLite(speaker=speaker, text=text))
    return out


_ATOM_LINK_RE = re.compile(r"\[\[([0-9A-Z]{20,32})\]\]")


def _load_linked_atoms(*, layout: VaultLayout, call_note_id: str, body: str) -> list[Atom]:
    """Pull atom IDs from the call note's `## Atoms` section + read each
    atom markdown file. Tolerates missing atom files (skips them) — a
    detached call note still scores."""
    atoms: list[Atom] = []
    for match in _ATOM_LINK_RE.finditer(body):
        atom_id = match.group(1)
        atom_path = layout.atom_file(atom_id)
        if not atom_path.exists():
            continue
        try:
            atom = _atom_from_markdown(atom_path)
        except Exception:
            continue
        # Only include atoms that link back to THIS call note (defensive —
        # in case the regex over-matches inside the transcript block).
        if atom.call == call_note_id:
            atoms.append(atom)
    return atoms


def _strip_wikilink(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s.startswith("[[") and s.endswith("]]"):
        return s[2:-2]
    return s or None
