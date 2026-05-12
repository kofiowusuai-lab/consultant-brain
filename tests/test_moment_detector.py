"""Moment detector tests — mock Anthropic client; verify confidence
floor, cap-at-3, empty-input handling, error tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from consultant_brain.moment_detector import (
    MOMENT_SYSTEM_PROMPT,
    build_user_prompt,
    detect_moments,
)
from consultant_brain.schemas import AtomType, CallType


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _MockClient:
    """Returns a canned response. last_call lets tests assert what was sent."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_call: dict[str, Any] | None = None

    def messages_create(self, **kwargs) -> _Response:
        self.last_call = kwargs
        return _Response(content=[_Block(text=self.response_text)])


class _ThrowingClient:
    """Simulates an Anthropic call that raised (network down, rate limit)."""

    def messages_create(self, **kwargs):
        raise RuntimeError("boom")


# ────────────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────────────


def test_moment_system_prompt_lists_all_atom_types() -> None:
    for atom_type in AtomType:
        assert atom_type.value in MOMENT_SYSTEM_PROMPT, f"missing {atom_type.value}"


def test_moment_user_prompt_carries_window_and_call_type() -> None:
    prompt = build_user_prompt(
        window="You: hi\nThem: 1000 notes in Obsidian.",
        call_type=CallType.consulting_call,
        client_name="Reece",
    )
    assert "consultingCall" in prompt
    assert "Reece" in prompt
    assert "1000 notes" in prompt


def test_moment_user_prompt_omits_client_when_none() -> None:
    prompt = build_user_prompt(
        window="x",
        call_type=CallType.cold_call,
        client_name=None,
    )
    assert "Client:" not in prompt


# ────────────────────────────────────────────────────────────────────────────
# detect_moments() behavior
# ────────────────────────────────────────────────────────────────────────────


def test_detect_returns_high_confidence_atoms() -> None:
    response = json.dumps(
        {
            "atoms": [
                {"type": "commitment", "body": "Reece will send 3 ads by Friday.", "confidence": 0.92, "tags": ["next_step"]},
                {"type": "objection", "body": "Tried that with another vendor.", "confidence": 0.78, "tags": ["past_attempt"]},
            ]
        }
    )
    atoms = detect_moments(
        window="You: anything else?\nThem: tried that with another vendor and it didn't stick.",
        call_type=CallType.consulting_call,
        client_name="Reece",
        client=_MockClient(response),
    )
    assert len(atoms) == 2
    assert atoms[0].type is AtomType.commitment
    assert atoms[1].type is AtomType.objection


def test_detect_filters_below_confidence_floor() -> None:
    response = json.dumps(
        {
            "atoms": [
                {"type": "win_signal", "body": "Sounded mildly interested.", "confidence": 0.55, "tags": []},
                {"type": "commitment", "body": "Reece will send 3 ads by Friday.", "confidence": 0.92, "tags": ["next_step"]},
            ]
        }
    )
    atoms = detect_moments(
        window="x", call_type=CallType.consulting_call, client_name=None, client=_MockClient(response)
    )
    # The 0.55 atom is dropped; the 0.92 is kept.
    assert len(atoms) == 1
    assert atoms[0].type is AtomType.commitment


def test_detect_caps_at_three_atoms() -> None:
    response = json.dumps(
        {
            "atoms": [
                {"type": "objection", "body": f"a {i}", "confidence": 0.9, "tags": []}
                for i in range(7)
            ]
        }
    )
    atoms = detect_moments(
        window="x", call_type=CallType.consulting_call, client_name=None, client=_MockClient(response)
    )
    assert len(atoms) == 3


def test_detect_returns_empty_on_empty_window() -> None:
    atoms = detect_moments(
        window="   \n  ",
        call_type=CallType.consulting_call,
        client_name="Reece",
        client=_MockClient(json.dumps({"atoms": []})),
    )
    assert atoms == []


def test_detect_returns_empty_on_anthropic_exception() -> None:
    """Live loop must NEVER take down the service. Network errors return []."""
    atoms = detect_moments(
        window="something happened",
        call_type=CallType.consulting_call,
        client_name=None,
        client=_ThrowingClient(),
    )
    assert atoms == []


def test_detect_returns_empty_on_invalid_json() -> None:
    atoms = detect_moments(
        window="something happened",
        call_type=CallType.consulting_call,
        client_name=None,
        client=_MockClient("this is not json"),
    )
    assert atoms == []


def test_detect_returns_empty_when_model_returns_no_atoms() -> None:
    atoms = detect_moments(
        window="You: cool. Them: cool.",
        call_type=CallType.consulting_call,
        client_name=None,
        client=_MockClient(json.dumps({"atoms": []})),
    )
    assert atoms == []


def test_detect_uses_overridable_confidence_floor() -> None:
    """A lower floor surfaces more atoms — useful for testing the pipeline
    with less stringent thresholds."""
    response = json.dumps(
        {"atoms": [{"type": "insight", "body": "Vague signal.", "confidence": 0.55, "tags": []}]}
    )
    atoms = detect_moments(
        window="x",
        call_type=CallType.consulting_call,
        client_name=None,
        client=_MockClient(response),
        confidence_floor=0.5,
    )
    assert len(atoms) == 1
