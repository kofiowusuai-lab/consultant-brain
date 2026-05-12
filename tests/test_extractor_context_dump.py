"""Tests for the extractor's Phase 10 additions:
  - build_context_dump_user_prompt() time preamble
  - extract_from_context_dump() routing through ChatRequest
  - build_user_prompt() + build_knowledge_user_prompt() now emit the
    "Today is ..." preamble
"""

from __future__ import annotations

from datetime import date

import pytest

from consultant_brain.extractor import (
    CONTEXT_DUMP_SYSTEM_PROMPT,
    build_context_dump_user_prompt,
    build_knowledge_user_prompt,
    build_user_prompt,
    extract_from_context_dump,
)
from consultant_brain.llm.provider import ChatRequest, ChatResponse
from consultant_brain.schemas import CallType


class _FakeProvider:
    name = "fake"

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(text=self._payload, model_used=request.model)


# ────────────────────────────────────────────────────────────────────────────
# Time preamble
# ────────────────────────────────────────────────────────────────────────────


def test_build_user_prompt_emits_today_line() -> None:
    prompt = build_user_prompt(
        "You: hi. Them: hello.",
        CallType.consulting_call,
        "Reece",
        today=date(2026, 5, 12),
    )
    assert prompt.startswith("Today is 2026-05-12.\n")


def test_build_user_prompt_emits_observed_when_given() -> None:
    prompt = build_user_prompt(
        "transcript",
        CallType.cold_call,
        None,
        observed_at=date(2026, 5, 1),
        today=date(2026, 5, 12),
    )
    assert "Today is 2026-05-12. this call was observed on 2026-05-01." in prompt


def test_build_knowledge_user_prompt_includes_time() -> None:
    prompt = build_knowledge_user_prompt(
        transcript="...",
        source_title="Hormozi rant",
        author="Alex",
        topic=None,
        for_client=None,
        observed_at=date(2026, 5, 1),
        today=date(2026, 5, 12),
    )
    assert "Today is 2026-05-12" in prompt
    assert "this source was observed on 2026-05-01" in prompt


def test_build_context_dump_user_prompt_bundles_metadata() -> None:
    prompt = build_context_dump_user_prompt(
        text="Coffee notes about pricing.",
        client_name="Reece",
        source_kind_label="audio",
        source_filename="coffee.m4a",
        notes="25-min recording, mostly pricing",
        observed_at=date(2026, 5, 12),
        today=date(2026, 5, 12),
    )
    assert "Client: Reece" in prompt
    assert "Source kind: audio" in prompt
    assert "Source filename: coffee.m4a" in prompt
    assert "25-min recording" in prompt
    assert "context was observed on 2026-05-12" in prompt


# ────────────────────────────────────────────────────────────────────────────
# extract_from_context_dump
# ────────────────────────────────────────────────────────────────────────────


_OK_PAYLOAD = (
    '{"summary":"Pricing talk.","atoms":['
    '{"type":"commitment","body":"Reece commits to Q3.","confidence":0.9,"tags":["timeline"]}'
    "]}"
)


def test_extract_from_context_dump_routes_through_provider() -> None:
    provider = _FakeProvider(_OK_PAYLOAD)
    result = extract_from_context_dump(
        text="Coffee notes.",
        client_name="Reece",
        source_kind_label="audio",
        source_filename="coffee.m4a",
        notes=None,
        observed_at=date(2026, 5, 1),
        today=date(2026, 5, 12),
        provider=provider,
    )
    assert result.summary == "Pricing talk."
    assert len(result.atoms) == 1
    # System prompt is the context-dump one, not the call extractor.
    assert provider.requests[0].system == CONTEXT_DUMP_SYSTEM_PROMPT
    # Time preamble + metadata both reached the model.
    assert "Today is 2026-05-12" in provider.requests[0].user
    assert "Client: Reece" in provider.requests[0].user


def test_extract_from_context_dump_requires_provider_or_client() -> None:
    with pytest.raises(Exception, match="requires"):
        extract_from_context_dump(
            text="x",
            client_name="Reece",
            source_kind_label="text",
            source_filename="x.txt",
        )
