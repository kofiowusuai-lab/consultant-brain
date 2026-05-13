"""Phase 13 — facilitator stage suggestor tests."""

from __future__ import annotations

import pytest

from consultant_brain.facilitator.outline import REECE_OUTLINE
from consultant_brain.facilitator.suggestor import (
    MAX_LOOKAHEAD,
    suggest_next_stage,
)


def test_empty_transcript_abstains() -> None:
    result = suggest_next_stage(transcript_window="", current_stage_index=0)
    assert result.abstains
    assert "empty" in result.reason.lower()


def test_end_of_outline_abstains() -> None:
    """At stage 8 (index 7) there's nowhere to go forward."""
    result = suggest_next_stage(
        transcript_window="You: post-call memo next step\n\nThem: ok",
        current_stage_index=len(REECE_OUTLINE) - 1,
    )
    assert result.abstains
    assert "end of outline" in result.reason.lower()


def test_transition_phrase_plus_named_cue_suggests_next() -> None:
    """Classic happy path: 'next section' is the transition signal,
    'scope' is the stage 04 cue. Should land on index 3."""
    window = (
        "You: alright, this all makes sense.\n\n"
        "Them: cool.\n\n"
        "You: let's move on to scope. what's in, what's parked?"
    )
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=2,
    )
    assert result.suggested_index == 3  # stage 4 = scope
    assert result.confidence > 0.0
    assert "scope" in result.reason.lower()
    assert "transition" in result.reason.lower()


def test_two_cue_hits_without_transition_phrase_suggests() -> None:
    """No 'next section' but two cue phrases for stage 05 in the
    window — enough to suggest without the transition cue."""
    window = (
        "You: the avatar slice goes here, awareness lock there.\n\n"
        "Them: and how does the bot decide?"
    )
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=3,  # stage 04
    )
    assert result.suggested_index == 4  # stage 05 engine


def test_single_low_signal_cue_abstains() -> None:
    """Only one cue hit + no transition phrase = noise. Abstain so
    the toast doesn't fire on every casual mention."""
    window = "You: what's the agenda for tomorrow?"
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=1,
    )
    # 'agenda' cues stage 03 (index 2) but only one cue; without
    # the transition phrase, suggestor abstains.
    assert result.abstains


def test_lookahead_cap_clamps_at_plus_two() -> None:
    """Cue phrase for stage 8 in the window while on stage 1.
    Suggestor must NOT recommend stage 8 — it clamps at +2 (stage 3
    in this case)."""
    window = "You: let's move on, post-call memo time, three doors."
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=0,
    )
    # Suggestion must be within current+1 and current+MAX_LOOKAHEAD.
    if not result.abstains:
        assert result.suggested_index is not None
        assert result.suggested_index <= 0 + MAX_LOOKAHEAD


def test_closer_candidate_wins_ties() -> None:
    """When current+1 AND current+2 both match cue phrases, current+1
    wins. Prevents accidental over-jump."""
    # Both stage 4 (scope) and stage 5 (engine) have cue phrases in
    # the window. Suggestor must pick the closer one.
    window = (
        "You: let's move on. lock scope and start the angle engine."
    )
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=2,
    )
    assert result.suggested_index == 3  # closer match (scope)


def test_deterministic_repeated_calls() -> None:
    """Same input → same output. No hidden randomness or LLM calls."""
    window = "You: let's move on to decision questions."
    r1 = suggest_next_stage(transcript_window=window, current_stage_index=5)
    r2 = suggest_next_stage(transcript_window=window, current_stage_index=5)
    assert r1 == r2


def test_recent_chars_window_drops_old_context() -> None:
    """When the cue lives outside the recent window, the suggestor
    can't see it and abstains."""
    cue = "let's move on to scope"
    padding = "You: ok ok ok\n\n" * 200
    window = cue + "\n\n" + padding
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=2,
        recent_chars=200,
    )
    # 'scope' is buried > 200 chars into the past; suggestor sees
    # only the recent padding, no cues.
    assert result.abstains


def test_confidence_bounded() -> None:
    """Confidence never escapes [0.05, 0.95] for any non-abstain
    return path."""
    window = (
        "You: let's move on. scope. parking lot. in scope. out of scope. lock scope. v1 scope."
    )
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=2,
    )
    assert result.suggested_index == 3
    assert 0.05 <= result.confidence <= 0.95


def test_current_stage_cue_alone_doesnt_suggest() -> None:
    """Cue phrases for the CURRENT stage don't trigger a self-suggest."""
    # 'agenda' is a stage-03 cue. If user is on stage 03 (index 2),
    # mentioning the agenda shouldn't suggest stage 03 again.
    window = "You: looking at the agenda, the timeline says 90 minutes."
    result = suggest_next_stage(
        transcript_window=window,
        current_stage_index=2,
    )
    # Suggestor only looks at stages > current. With no cues for
    # stages 4 or 5, it abstains.
    assert result.abstains
