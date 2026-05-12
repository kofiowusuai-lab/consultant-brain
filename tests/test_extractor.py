"""Extractor tests. Mock the Anthropic client so CI doesn't burn tokens
(and so tests pass without network access).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from consultant_brain.extractor import (
    SYSTEM_PROMPT,
    ExtractorError,
    build_user_prompt,
    extract,
)
from consultant_brain.schemas import AtomType, CallType


# ────────────────────────────────────────────────────────────────────────────
# Mock client
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]


class _MockAnthropicClient:
    """Returns a pre-canned response text. Records the last call args so
    tests can assert what was sent."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_call: dict[str, Any] | None = None

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> _Response:
        self.last_call = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        return _Response(content=[_Block(text=self.response_text)])


# ────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────────────────────────────────────


def test_system_prompt_lists_all_seven_atom_types() -> None:
    for atom_type in AtomType:
        assert atom_type.value in SYSTEM_PROMPT, f"missing {atom_type.value} in system prompt"


def test_build_user_prompt_includes_call_type_and_transcript() -> None:
    prompt = build_user_prompt(
        transcript="You: hello\nThem: hi",
        call_type=CallType.consulting_call,
        client_name="Reece",
    )
    assert "consultingCall" in prompt
    assert "Reece" in prompt
    assert "You: hello" in prompt
    assert "Them: hi" in prompt


def test_build_user_prompt_omits_client_line_when_none() -> None:
    prompt = build_user_prompt(
        transcript="You: hello",
        call_type=CallType.cold_call,
        client_name=None,
    )
    assert "Client:" not in prompt
    assert "coldCall" in prompt


# ────────────────────────────────────────────────────────────────────────────
# extract() end-to-end with mock client
# ────────────────────────────────────────────────────────────────────────────


VALID_RESPONSE = json.dumps(
    {
        "summary": "Reece wants ad-bot retrieval fixed first.",
        "atoms": [
            {
                "type": "objection",
                "body": "Bot keeps grabbing wrong notes — no tagging system, just timestamps.",
                "confidence": 0.88,
                "tags": ["retrieval", "obsidian"],
            },
            {
                "type": "commitment",
                "body": "Reece will send the 3 best manual ads by Friday.",
                "confidence": 0.95,
                "tags": ["next_step", "deliverable"],
            },
        ],
    }
)


def test_extract_returns_validated_result() -> None:
    client = _MockAnthropicClient(VALID_RESPONSE)
    result = extract(
        transcript="You: hi\nThem: hi",
        call_type=CallType.consulting_call,
        client_name="Reece",
        client=client,
    )
    assert result.summary.startswith("Reece wants")
    assert len(result.atoms) == 2
    assert result.atoms[0].type is AtomType.objection
    assert result.atoms[1].confidence == 0.95


def test_extract_propagates_model_and_system_prompt_to_client() -> None:
    client = _MockAnthropicClient(VALID_RESPONSE)
    extract(
        transcript="You: hi\nThem: hi",
        call_type=CallType.consulting_call,
        client_name="Reece",
        client=client,
        model="claude-fake-model",
        max_tokens=2048,
    )
    assert client.last_call is not None
    assert client.last_call["model"] == "claude-fake-model"
    assert client.last_call["max_tokens"] == 2048
    assert client.last_call["system"] == SYSTEM_PROMPT
    user_msg = client.last_call["messages"][0]
    assert user_msg["role"] == "user"
    assert "consultingCall" in user_msg["content"]


def test_extract_strips_markdown_fences_around_json() -> None:
    fenced = f"```json\n{VALID_RESPONSE}\n```"
    client = _MockAnthropicClient(fenced)
    result = extract(
        transcript="x", call_type=CallType.cold_call, client_name=None, client=client
    )
    assert len(result.atoms) == 2


def test_extract_raises_on_invalid_json() -> None:
    client = _MockAnthropicClient("this is not json at all")
    with pytest.raises(ExtractorError, match="invalid JSON"):
        extract(transcript="x", call_type=CallType.cold_call, client_name=None, client=client)


def test_extract_raises_on_missing_required_field() -> None:
    bad = json.dumps({"atoms": []})  # missing "summary"
    client = _MockAnthropicClient(bad)
    with pytest.raises(Exception):  # pydantic ValidationError
        extract(transcript="x", call_type=CallType.cold_call, client_name=None, client=client)


def test_extract_raises_on_empty_response() -> None:
    class _EmptyClient:
        def messages_create(self, **kwargs):
            return _Response(content=[])

    with pytest.raises(ExtractorError, match="no text blocks"):
        extract(transcript="x", call_type=CallType.cold_call, client_name=None, client=_EmptyClient())
