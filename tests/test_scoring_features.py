"""Feature extractor tests — verify each of the 9 signals is computed
correctly across the call-type matrix.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallType,
    Speaker,
)
from consultant_brain.scoring.features import (
    FEATURE_NAMES,
    NeutralWinJudge,
    ScoreFeatures,
    TALK_RATIO_TARGETS,
    TranscriptTurnLite,
    compute_features,
)


def _atom(
    *,
    type_: AtomType,
    body: str,
    created_at: datetime | None = None,
    id_suffix: str = "01",
) -> Atom:
    return Atom(
        id=f"FEATURE_TEST_AAAAAAAAA{id_suffix:>4}"[-26:].ljust(26, "A"),
        type=type_,
        client="Reece",
        call="2026-05-12_reece_consultingCall",
        call_type=CallType.consulting_call,
        tags=[],
        confidence=0.85,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=created_at or datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id="dummy",
        body=body,
    )


def _turn(speaker: Speaker, text: str) -> TranscriptTurnLite:
    return TranscriptTurnLite(speaker=speaker, text=text)


# ────────────────────────────────────────────────────────────────────────────
# Shape
# ────────────────────────────────────────────────────────────────────────────


def test_feature_order_matches_names_tuple() -> None:
    f = ScoreFeatures(
        next_step_booked=1, objections_resolved=0, talk_ratio_balance=0,
        commitments_made=0, win_signals=0, loss_signals=0,
        confusion_events=0, completion_of_agenda=0, primary_win_progress=0,
    )
    assert len(f.as_vector()) == len(FEATURE_NAMES) == 9


# ────────────────────────────────────────────────────────────────────────────
# next_step_booked
# ────────────────────────────────────────────────────────────────────────────


def test_next_step_booked_fires_on_friday_phrasing() -> None:
    atoms = [_atom(type_=AtomType.commitment, body="Reece will send 3 ads by Friday.")]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.next_step_booked == 1.0


def test_next_step_booked_fires_on_explicit_time() -> None:
    atoms = [_atom(type_=AtomType.commitment, body="We'll meet tomorrow at 9am.")]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.next_step_booked == 1.0


def test_next_step_booked_zero_when_commitment_lacks_time() -> None:
    atoms = [_atom(type_=AtomType.commitment, body="I'll follow up at some point.")]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.next_step_booked == 0.0


def test_next_step_booked_zero_when_no_commitments() -> None:
    atoms = [_atom(type_=AtomType.objection, body="Tried that before.")]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.next_step_booked == 0.0


# ────────────────────────────────────────────────────────────────────────────
# objections_resolved
# ────────────────────────────────────────────────────────────────────────────


def test_objections_resolved_one_for_one() -> None:
    base = datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc)
    atoms = [
        _atom(type_=AtomType.objection, body="Price hesitation.", created_at=base, id_suffix="01"),
        _atom(type_=AtomType.commitment, body="Reece will send 3 by Friday.", created_at=base + timedelta(minutes=2), id_suffix="02"),
    ]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.objections_resolved == 1.0


def test_objections_resolved_half_when_one_of_two_resolved() -> None:
    base = datetime(2026, 5, 12, 17, 0, tzinfo=timezone.utc)
    atoms = [
        _atom(type_=AtomType.objection, body="Price.", created_at=base, id_suffix="01"),
        _atom(type_=AtomType.commitment, body="Will send by Friday.", created_at=base + timedelta(minutes=1), id_suffix="02"),
        _atom(type_=AtomType.objection, body="Adoption risk.", created_at=base + timedelta(minutes=10), id_suffix="03"),
    ]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.objections_resolved == 0.5


def test_objections_resolved_one_when_no_objections() -> None:
    atoms = [_atom(type_=AtomType.commitment, body="Will send by Friday.")]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    # Vacuously resolved — no objections to leave unhandled.
    assert f.objections_resolved == 1.0


# ────────────────────────────────────────────────────────────────────────────
# talk_ratio_balance
# ────────────────────────────────────────────────────────────────────────────


def test_talk_ratio_perfect_match_returns_one() -> None:
    # Consulting target: 40/60 you/them. Make exactly that ratio.
    you_chars = "x" * 40
    them_chars = "x" * 60
    turns = [_turn(Speaker.you, you_chars), _turn(Speaker.them, them_chars)]
    f = compute_features(atoms=[], turns=turns, call_type=CallType.consulting_call)
    assert f.talk_ratio_balance == pytest.approx(1.0)


def test_talk_ratio_too_much_consultant_talking_drops() -> None:
    you_chars = "x" * 90
    them_chars = "x" * 10
    turns = [_turn(Speaker.you, you_chars), _turn(Speaker.them, them_chars)]
    f = compute_features(atoms=[], turns=turns, call_type=CallType.consulting_call)
    # 0.90 actual vs 0.40 target = 0.50 off = should rate 0 (capped at 0.30).
    assert f.talk_ratio_balance == 0.0


def test_talk_ratio_uses_call_type_specific_target() -> None:
    """Cold call target is 30/70 — same actual ratio of 50/50 should rate
    DIFFERENTLY across call types because targets differ."""
    you_chars = "x" * 50
    them_chars = "x" * 50
    turns = [_turn(Speaker.you, you_chars), _turn(Speaker.them, them_chars)]
    consulting_score = compute_features(atoms=[], turns=turns, call_type=CallType.consulting_call).talk_ratio_balance
    cold_score = compute_features(atoms=[], turns=turns, call_type=CallType.cold_call).talk_ratio_balance
    closing_score = compute_features(atoms=[], turns=turns, call_type=CallType.closing_call).talk_ratio_balance
    # Closing's target IS 50/50 → highest score for 50/50 actual.
    assert closing_score > consulting_score
    assert closing_score > cold_score


def test_talk_ratio_empty_transcript_returns_neutral() -> None:
    f = compute_features(atoms=[], turns=[], call_type=CallType.consulting_call)
    assert f.talk_ratio_balance == 0.5


# ────────────────────────────────────────────────────────────────────────────
# Saturating counts
# ────────────────────────────────────────────────────────────────────────────


def test_commitments_saturate_at_five() -> None:
    atoms = [
        _atom(type_=AtomType.commitment, body=f"commit {i}", id_suffix=str(i).zfill(2))
        for i in range(7)
    ]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.commitments_made == 1.0  # capped


def test_loss_signals_count_above_cap_caps() -> None:
    atoms = [
        _atom(type_=AtomType.loss_signal, body=f"deflect {i}", id_suffix=str(i).zfill(2))
        for i in range(5)
    ]
    f = compute_features(atoms=atoms, turns=[], call_type=CallType.consulting_call)
    assert f.loss_signals == 1.0


def test_confusion_zero_when_no_atoms() -> None:
    f = compute_features(atoms=[], turns=[], call_type=CallType.consulting_call)
    assert f.confusion_events == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Primary win progress
# ────────────────────────────────────────────────────────────────────────────


def test_primary_win_neutral_judge_returns_half() -> None:
    f = compute_features(
        atoms=[],
        turns=[_turn(Speaker.them, "hello")],
        call_type=CallType.consulting_call,
        primary_win="ship the ad-bot",
    )
    assert f.primary_win_progress == 0.5


def test_primary_win_uses_custom_judge() -> None:
    class _AlwaysHigh:
        def judge(self, *, primary_win, transcript, call_type):
            return 0.93

    f = compute_features(
        atoms=[],
        turns=[_turn(Speaker.you, "anything")],
        call_type=CallType.consulting_call,
        primary_win="ship the ad-bot",
        judge=_AlwaysHigh(),
    )
    assert f.primary_win_progress == 0.93


def test_primary_win_clamps_to_unit_interval() -> None:
    class _OutOfBounds:
        def judge(self, *, primary_win, transcript, call_type):
            return 1.7

    f = compute_features(
        atoms=[],
        turns=[_turn(Speaker.you, "hi")],
        call_type=CallType.consulting_call,
        primary_win="ship it",
        judge=_OutOfBounds(),
    )
    assert f.primary_win_progress == 1.0


def test_primary_win_returns_neutral_without_win_field() -> None:
    f = compute_features(
        atoms=[],
        turns=[_turn(Speaker.you, "hi")],
        call_type=CallType.consulting_call,
        primary_win=None,
    )
    assert f.primary_win_progress == 0.5


# ────────────────────────────────────────────────────────────────────────────
# Constants sanity
# ────────────────────────────────────────────────────────────────────────────


def test_talk_ratio_targets_cover_all_call_types() -> None:
    for ct in CallType:
        assert ct in TALK_RATIO_TARGETS, f"missing target for {ct.value}"
