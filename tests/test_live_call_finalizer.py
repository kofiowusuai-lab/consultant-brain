"""Tests for finalize_live_call + the /call_end finalize behavior."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consultant_brain.live_call_finalizer import (
    PLACEHOLDER_SUMMARY,
    finalize_live_call,
)
from consultant_brain.live_state import CallState, LiveCallRegistry
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
    Speaker,
)
from consultant_brain.service import create_app
from consultant_brain.vault import (
    VaultLayout,
    derive_call_id,
    ensure_vault_skeleton,
    read_frontmatter,
    write_atom,
)


def _make_state(*, call_id="live-001", client="Reece") -> CallState:
    return CallState(
        call_id=call_id,
        client=client,
        call_type=CallType.consulting_call,
        started_at=datetime(2026, 5, 12, 19, 0, tzinfo=timezone.utc),
    )


def _seed_atom(*, layout: VaultLayout, atom_id: str, call_id: str) -> None:
    atom = Atom(
        id=atom_id,
        type=AtomType.objection,
        client="Reece",
        call=call_id,
        call_type=CallType.consulting_call,
        tags=["budget"],
        confidence=0.85,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 19, 5, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id=atom_id,
        body="Price came up before scope.",
    )
    write_atom(layout, atom)


# ────────────────────────────────────────────────────────────────────────────
# Finalizer direct
# ────────────────────────────────────────────────────────────────────────────


def test_finalize_writes_call_note_with_correct_metadata(tmp_path: Path) -> None:
    state = _make_state()
    state.append_turn(Speaker.you, "Walk me through your stack.")
    state.append_turn(Speaker.them, "We use Obsidian.")
    vault_root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(vault_root))

    result = finalize_live_call(state=state, vault_root=vault_root)
    expected_id = derive_call_id(
        client_name="Reece",
        call_type=CallType.consulting_call,
        call_date=state.started_at,
    )
    assert result.call_note_id == expected_id
    assert result.call_note_path.exists()
    fm = read_frontmatter(result.call_note_path)
    assert fm["id"] == expected_id
    assert fm["client"] == "[[Reece]]"
    assert fm["call_type"] == "consultingCall"
    assert fm["source_session"].startswith("live::live-001")


def test_finalize_carries_transcript_into_call_note(tmp_path: Path) -> None:
    state = _make_state()
    state.append_turn(Speaker.you, "Walk me through your stack.")
    state.append_turn(Speaker.them, "We use Obsidian, 1000 notes.")
    vault_root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(vault_root))

    result = finalize_live_call(state=state, vault_root=vault_root)
    text = result.call_note_path.read_text(encoding="utf-8")
    assert "You: Walk me through your stack." in text
    assert "Them: We use Obsidian, 1000 notes." in text
    assert PLACEHOLDER_SUMMARY in text


def test_finalize_links_atoms_written_during_call(tmp_path: Path) -> None:
    """The Phase 4 loop writes atoms with `call: [[<call_note_id>]]` BEFORE
    the call ends. Finalize discovers them by scanning + links them in
    the call note's atom_ids list."""
    state = _make_state()
    state.append_turn(Speaker.them, "Price came up before scope.")
    vault_root = tmp_path / "vault"
    layout = VaultLayout.for_root(vault_root)
    ensure_vault_skeleton(layout)

    call_id = derive_call_id(
        client_name="Reece",
        call_type=CallType.consulting_call,
        call_date=state.started_at,
    )
    _seed_atom(layout=layout, atom_id="LIVEAAAAAAAAAAAAAAAAAAA001", call_id=call_id)
    _seed_atom(layout=layout, atom_id="LIVEAAAAAAAAAAAAAAAAAAA002", call_id=call_id)
    # And one atom pointing to a DIFFERENT call — must NOT be linked.
    _seed_atom(layout=layout, atom_id="OTHERBBBBBBBBBBBBBBBBBB003", call_id="some-other-call")

    result = finalize_live_call(state=state, vault_root=vault_root)
    assert result.atom_count == 2
    fm = read_frontmatter(result.call_note_path)
    assert fm["atom_count"] == 2


def test_finalize_empty_state_still_writes_call_note(tmp_path: Path) -> None:
    """A call that ended with zero turns still produces a (mostly empty)
    call note — so /score can return something coherent."""
    state = _make_state()
    vault_root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(vault_root))

    result = finalize_live_call(state=state, vault_root=vault_root)
    assert result.call_note_path.exists()
    assert result.atom_count == 0
    assert result.transcript_chars == 0


# ────────────────────────────────────────────────────────────────────────────
# /call_end behavior
# ────────────────────────────────────────────────────────────────────────────


def test_call_end_returns_call_note_id_when_call_was_active(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(vault_root))
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    client = TestClient(app)

    client.post(
        "/call_start",
        json={"call_id": "lc_001", "client": "Reece", "call_type": "consultingCall"},
    )
    client.post(
        "/transcript_delta",
        json={"call_id": "lc_001", "speaker": "you", "text": "Walk me through your stack."},
    )

    response = client.post("/call_end", json={"call_id": "lc_001"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ended"] is True
    assert body["call_note_id"]
    assert body["call_note_id"].endswith("_consultingCall")
    assert body["atom_count"] == 0  # no live atoms written in this test


def test_call_end_unknown_call_skips_finalize(tmp_path: Path) -> None:
    """When the call_id was never started, /call_end returns ended=False
    and skips finalize. The Swift app sees a clean signal that there's
    nothing to score."""
    vault_root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(vault_root))
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    client = TestClient(app)

    response = client.post("/call_end", json={"call_id": "never_started"})
    assert response.status_code == 200
    body = response.json()
    assert body["ended"] is False
    assert body["call_note_id"] is None
    # No call note file written.
    layout = VaultLayout.for_root(vault_root)
    assert not list(layout.calls_dir.glob("*.md"))


def test_call_end_then_score_round_trip(tmp_path: Path) -> None:
    """The integration the Swift app actually runs: start call → deltas →
    end call → score by the returned call_note_id."""
    vault_root = tmp_path / "vault"
    ensure_vault_skeleton(VaultLayout.for_root(vault_root))
    app = create_app(vault_root=vault_root, registry=LiveCallRegistry())
    client = TestClient(app)

    client.post(
        "/call_start",
        json={"call_id": "lc_001", "client": "Reece", "call_type": "consultingCall"},
    )
    client.post(
        "/transcript_delta",
        json={"call_id": "lc_001", "speaker": "you", "text": "What's blocking you?"},
    )
    client.post(
        "/transcript_delta",
        json={"call_id": "lc_001", "speaker": "them", "text": "Price was a concern."},
    )

    ended = client.post("/call_end", json={"call_id": "lc_001"})
    call_note_id = ended.json()["call_note_id"]
    assert call_note_id

    scored = client.get(f"/score?call_id={call_note_id}")
    assert scored.status_code == 200, scored.text
    body = scored.json()
    assert body["call_id"] == call_note_id
    assert 0 <= body["score"] <= 100
    assert len(body["contributions"]) == 9
