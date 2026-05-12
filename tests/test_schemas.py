"""Round-trip tests for every persistent schema.

The goal: catch shape regressions (extra fields, wrong types, missing
validators) before they corrupt the vault. Cheap to write, prevents
expensive forensic work later.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallNote,
    CallType,
    CompletedTurn,
    ExtractedAtom,
    ExtractorResult,
    IngestConfig,
    SessionJSON,
    Speaker,
)


# ────────────────────────────────────────────────────────────────────────────
# Input schemas — SessionJSON, CompletedTurn
# ────────────────────────────────────────────────────────────────────────────


def test_completed_turn_translates_source_to_speaker() -> None:
    them = CompletedTurn.model_validate(
        {
            "completedAt": "2026-05-12T15:43:04Z",
            "itemID": "item_abc",
            "source": "systemAudio",
            "text": "I tried that before",
        }
    )
    assert them.speaker() is Speaker.them

    you = CompletedTurn.model_validate(
        {
            "completedAt": "2026-05-12T15:43:14Z",
            "itemID": "item_xyz",
            "source": "microphone",
            "text": "Quick one — when you say tried before",
        }
    )
    assert you.speaker() is Speaker.you


def test_session_json_tolerates_unknown_keys() -> None:
    # Real session JSONs carry partialSources / partialTranscripts /
    # processedSegmentIDs / partialSpeakers — Phase 1 ignores them.
    raw = {
        "startedAt": "2026-05-12T15:42:29Z",
        "completedTurns": [
            {
                "completedAt": "2026-05-12T15:43:04Z",
                "itemID": "item_a",
                "source": "systemAudio",
                "text": "hello",
            }
        ],
        "partialSources": {"item_b": "systemAudio"},
        "partialSpeakers": {"item_b": "them"},
        "partialTranscripts": {"item_b": "incomplete"},
        "processedSegmentIDs": [],
        "suggestions": [],
        "unknown_future_field": "should not break parsing",
    }
    session = SessionJSON.model_validate(raw)
    assert len(session.completed_turns) == 1
    assert session.completed_turns[0].text == "hello"


def test_session_json_accepts_suggestions() -> None:
    raw = {
        "startedAt": "2026-05-12T15:42:29Z",
        "completedTurns": [],
        "suggestions": [
            {
                "category": "askThis",
                "createdAt": "2026-05-12T15:45:00Z",
                "text": "What's blocking the ad-bot rollout right now?",
            }
        ],
    }
    session = SessionJSON.model_validate(raw)
    assert session.suggestions[0].category == "askThis"
    assert session.suggestions[0].text.startswith("What's blocking")


# ────────────────────────────────────────────────────────────────────────────
# Output schemas — Atom, CallNote
# ────────────────────────────────────────────────────────────────────────────


def _atom_payload(**overrides) -> dict:
    base = {
        "id": "01HXAVQR8N0F7P5K2C4M9B6S3D",
        "type": "objection",
        "client": "Reece",
        "call": "2026-05-12_reece_consultingCall",
        "call_type": "consultingCall",
        "tags": ["budget", "scope"],
        "confidence": 0.82,
        "evidence_count": 1,
        "last_seen": "2026-05-12",
        "created_at": "2026-05-12T19:42:10Z",
        "status": "active",
        "embedding_id": "01HXAVQR8N0F7P5K2C4M9B6S3D",
        "body": "Price came up before scope. Anchor risk.",
    }
    base.update(overrides)
    return base


def test_atom_round_trip() -> None:
    atom = Atom.model_validate(_atom_payload())
    assert atom.type is AtomType.objection
    assert atom.call_type is CallType.consulting_call
    assert atom.status is AtomStatus.active
    assert atom.confidence == 0.82
    # Dump → re-validate must produce an identical object.
    redumped = Atom.model_validate(atom.model_dump(mode="json"))
    assert redumped == atom


def test_atom_rejects_extra_fields() -> None:
    payload = _atom_payload(extra_unknown_field="should not be silently accepted")
    with pytest.raises(ValidationError):
        Atom.model_validate(payload)


def test_atom_rejects_multi_paragraph_body() -> None:
    long = "First paragraph.\n\nSecond.\n\nThird paragraph — atoms aren't essays."
    with pytest.raises(ValidationError, match="single paragraph"):
        Atom.model_validate(_atom_payload(body=long))


def test_atom_confidence_must_be_in_range() -> None:
    with pytest.raises(ValidationError):
        Atom.model_validate(_atom_payload(confidence=1.4))
    with pytest.raises(ValidationError):
        Atom.model_validate(_atom_payload(confidence=-0.1))


def test_atom_evidence_count_minimum_is_one() -> None:
    with pytest.raises(ValidationError):
        Atom.model_validate(_atom_payload(evidence_count=0))


def test_atom_client_org_id_round_trips_uuid() -> None:
    """Phase 8: stable UUID linking atom → Swift CRMOrganization row."""
    from uuid import UUID
    payload = _atom_payload(client_org_id="00000000-0000-0000-0000-000000000a01")
    atom = Atom.model_validate(payload)
    assert atom.client_org_id == UUID("00000000-0000-0000-0000-000000000a01")
    # Dump + re-validate to confirm it serializes back cleanly.
    dumped = atom.model_dump(mode="json")
    assert dumped["client_org_id"] == "00000000-0000-0000-0000-000000000a01"
    assert Atom.model_validate(dumped) == atom


def test_atom_client_org_id_optional_defaults_none() -> None:
    payload = _atom_payload()
    # No client_org_id key in the payload at all.
    payload.pop("client_org_id", None)
    atom = Atom.model_validate(payload)
    assert atom.client_org_id is None


def test_call_note_id_format_is_enforced() -> None:
    base = {
        "client": "Reece",
        "call_type": "consultingCall",
        "date": "2026-05-12",
        "duration_minutes": 47,
        "source_session": "session-2026-05-12T05-42-10Z.json",
        "extractor_model": "claude-sonnet-4-6",
        "extractor_version": 1,
        "atom_count": 19,
        "created_at": "2026-05-12T19:42:10Z",
        "summary": "Reece is sold on the outcome but anxious about price.",
        "atom_ids": [],
        "transcript": "...",
    }
    # Good id — passes
    note = CallNote.model_validate({**base, "id": "2026-05-12_reece_consultingCall"})
    assert note.call_type is CallType.consulting_call

    # Bad id — fails (no date prefix)
    with pytest.raises(ValidationError):
        CallNote.model_validate({**base, "id": "reece_consultingCall"})

    # Bad id — fails (uppercase client slug)
    with pytest.raises(ValidationError):
        CallNote.model_validate({**base, "id": "2026-05-12_REECE_consultingCall"})


# ────────────────────────────────────────────────────────────────────────────
# Extractor I/O
# ────────────────────────────────────────────────────────────────────────────


def test_extracted_atom_has_only_semantic_fields() -> None:
    raw = {"type": "win_signal", "body": "Reece said this is exactly what we need.", "confidence": 0.91, "tags": ["enthusiasm"]}
    atom = ExtractedAtom.model_validate(raw)
    assert atom.type is AtomType.win_signal
    assert atom.confidence == 0.91

    # Bookkeeping fields are NOT accepted from the LLM — vault writer adds them.
    with pytest.raises(ValidationError):
        ExtractedAtom.model_validate({**raw, "id": "should_not_be_set_by_llm"})


def test_extractor_result_packages_summary_and_atoms() -> None:
    result = ExtractorResult.model_validate(
        {
            "summary": "Call landed well; one open objection on price.",
            "atoms": [
                {"type": "objection", "body": "Price hesitation.", "confidence": 0.7, "tags": []},
                {"type": "commitment", "body": "Send vault sample Friday.", "confidence": 0.95, "tags": ["next_step"]},
            ],
        }
    )
    assert len(result.atoms) == 2
    assert result.atoms[1].type is AtomType.commitment


# ────────────────────────────────────────────────────────────────────────────
# IngestConfig
# ────────────────────────────────────────────────────────────────────────────


def test_ingest_config_defaults() -> None:
    cfg = IngestConfig(client_name="Reece", call_type=CallType.consulting_call)
    assert cfg.redact is False
    assert cfg.dry_run is False
    assert cfg.extractor_version >= 1
    assert cfg.now.tzinfo is timezone.utc
