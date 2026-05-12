"""Pydantic schemas for every persistent type in consultant-brain.

One module so the canonical shape lives in one place. Splits by concern:
  - Inputs: the Swift app's session JSON format (SessionJSON, CompletedTurn, ...).
  - Outputs: vault notes (Atom, CallNote).
  - Internal: extractor I/O contracts (ExtractedAtom, ExtractorResult).

See SCHEMAS.md at the repo root for the human-readable spec.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ────────────────────────────────────────────────────────────────────────────
# Shared enums
# ────────────────────────────────────────────────────────────────────────────


class CallType(str, Enum):
    """Mirrors the Swift `CallType` enum. Values are the raw strings the Swift
    app uses, so a session JSON's `call_type` field round-trips by reference.
    """

    consulting_call = "consultingCall"
    ai_training = "aiTraining"
    cold_call = "coldCall"
    closing_call = "closingCall"
    follow_up = "followUp"
    implementation = "implementation"


class AtomType(str, Enum):
    """The 7 atom types Phase 1 extracts. Anything broader (patterns, plays,
    definitions) lives in different vault folders, not as atoms.
    """

    insight = "insight"
    objection = "objection"
    commitment = "commitment"
    win_signal = "win_signal"
    loss_signal = "loss_signal"
    confusion = "confusion"
    client_fact = "client_fact"


class AtomStatus(str, Enum):
    """Lifecycle status. Phase 1 only writes `active`; later phases handle
    promotion / demotion / archival."""

    active = "active"
    retired = "retired"
    needs_review = "needs_review"


class Speaker(str, Enum):
    """Normalized speaker identity. The Swift app's `source` field uses
    `systemAudio` (= the other person) and `microphone` (= the consultant) —
    we translate at load time so the schema stays semantic, not audio-routing.
    """

    them = "them"  # client / prospect
    you = "you"  # the consultant (Casper)


# ────────────────────────────────────────────────────────────────────────────
# Session JSON (input from Swift app)
# ────────────────────────────────────────────────────────────────────────────


class SessionSuggestion(BaseModel):
    """One suggestion emitted by the Swift app's live suggestion engine during
    the call. We preserve them on ingest so the brain can learn what the
    consultant was being shown at each moment (not used as atoms in Phase 1).
    """

    model_config = ConfigDict(extra="ignore")

    category: str
    created_at: datetime = Field(alias="createdAt")
    text: str


class CompletedTurn(BaseModel):
    """One finalized transcript turn from the Swift app. The Swift app commits
    a turn when the realtime transcription's VAD detects ~900ms of silence."""

    model_config = ConfigDict(extra="ignore")

    completed_at: datetime = Field(alias="completedAt")
    item_id: str = Field(alias="itemID")
    source: str  # "systemAudio" or "microphone" — translated to Speaker by load_session()
    text: str

    def speaker(self) -> Speaker:
        return Speaker.them if self.source == "systemAudio" else Speaker.you


class SessionJSON(BaseModel):
    """Top-level shape of `~/Library/Application Support/Consultant Copilot/Sessions/*.json`.

    We tolerate unknown keys via `extra="ignore"` because the Swift app evolves
    its session schema faster than this repo would catch up.
    """

    model_config = ConfigDict(extra="ignore")

    started_at: datetime = Field(alias="startedAt")
    completed_turns: list[CompletedTurn] = Field(alias="completedTurns")
    suggestions: list[SessionSuggestion] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Vault notes (output)
# ────────────────────────────────────────────────────────────────────────────


class Atom(BaseModel):
    """One typed note in the vault — the smallest meaningful unit the brain
    operates on. Persisted as a markdown file with YAML frontmatter at
    `03_Atoms/<id>.md`. Body holds 1-3 sentences of human-readable content
    plus `[[wikilinks]]` to related atoms / clients / plays.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=8, max_length=64)
    type: AtomType
    client: Optional[str] = None  # display name; vault writer renders as `[[<name>]]`
    call: str  # call note ID this atom was extracted from; renders as `[[<call>]]`
    call_type: CallType
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(default=1, ge=1)
    last_seen: date
    created_at: datetime
    status: AtomStatus = AtomStatus.active
    embedding_id: str
    body: str = Field(min_length=1)

    @field_validator("body")
    @classmethod
    def body_is_one_paragraph(cls, value: str) -> str:
        # Atoms are atomic. Reject obvious multi-paragraph drift early so the
        # extractor can't silently produce essays disguised as atoms.
        stripped = value.strip()
        if stripped.count("\n\n") > 1:
            raise ValueError("Atom body must be a single paragraph (≤1 blank line)")
        return stripped

    @field_validator("embedding_id")
    @classmethod
    def embedding_mirrors_id(cls, value: str, info) -> str:
        # In Phase 1 the embedding_id is just the atom id. Keeping it as a
        # separate field lets future phases swap embedding storage without
        # touching the atom schema (e.g. one atom → multiple embeddings).
        return value


class CallNote(BaseModel):
    """One call note in the vault — represents the entire call. Persisted at
    `02_Calls/<id>.md`. The body lists every atom extracted from this call as
    wikilinks + holds a 3-line summary + collapsed full transcript.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9_-]+_[a-zA-Z]+$")
    client: Optional[str] = None  # rendered as wikilink
    call_type: CallType
    date: date  # noqa: A003 — `date` is the field name in the vault frontmatter
    duration_minutes: int = Field(ge=0)
    source_session: str  # filename of the session JSON we ingested
    extractor_model: str
    extractor_version: int = Field(ge=1)
    atom_count: int = Field(ge=0)
    created_at: datetime
    summary: str  # 1-3 sentence summary, written into the body
    atom_ids: list[str] = Field(default_factory=list)  # ULIDs of every linked atom
    transcript: str  # raw transcript, embedded inside <details> tags in the body


# ────────────────────────────────────────────────────────────────────────────
# Extractor I/O contracts
# ────────────────────────────────────────────────────────────────────────────


class ExtractedAtom(BaseModel):
    """What the extractor returns per atom — the LLM only fills the
    semantically-meaningful fields; the rest (id, embedding_id, client, call,
    call_type, last_seen, created_at, status) are added by the vault writer.
    """

    model_config = ConfigDict(extra="forbid")

    type: AtomType
    body: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class ExtractorResult(BaseModel):
    """What `extractor.extract()` returns — atoms + a call-level summary
    sentence. The summary lands in the call note's body; the atoms become
    individual files."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    atoms: list[ExtractedAtom]


# ────────────────────────────────────────────────────────────────────────────
# Run-time config
# ────────────────────────────────────────────────────────────────────────────


# Bumping this number = re-ingesting old sessions will overwrite their atoms
# with the new extraction. Used in the deterministic atom-ID derivation so old
# versions don't collide with new ones in the vault.
EXTRACTOR_VERSION = 1

# Default Claude model. Pinned to the cheapest capable Sonnet so each ingest
# stays cost-effective. Bumping requires re-running ingests to pick up the
# new model's atoms; the extractor_version above usually bumps in tandem.
DEFAULT_EXTRACTOR_MODEL = "claude-sonnet-4-6"


class IngestConfig(BaseModel):
    """Everything one ingest run needs. Built by the CLI; passed to the
    pipeline. Keeping it explicit (not env-driven) makes tests deterministic.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    client_name: str
    call_type: CallType
    redact: bool = False
    dry_run: bool = False
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL
    extractor_version: int = EXTRACTOR_VERSION
    now: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
