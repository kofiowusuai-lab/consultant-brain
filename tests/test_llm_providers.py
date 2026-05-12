"""Tests for the LLM provider abstraction.

Each provider's `chat()` is exercised with a fake SDK client so we can
assert the exact wire-shape that hits the provider's API. We never
make real network calls in unit tests — those go through manual smoke
tests in scripts/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from consultant_brain.llm.anthropic_provider import AnthropicProvider
from consultant_brain.llm.deepseek_provider import DeepSeekProvider
from consultant_brain.llm.kimi_provider import KimiProvider
from consultant_brain.llm.openai_compatible import OpenAICompatibleProvider
from consultant_brain.llm.openai_provider import OpenAIProvider
from consultant_brain.llm.openrouter_provider import OpenRouterProvider
from consultant_brain.llm.provider import ChatRequest, ProviderError


# ────────────────────────────────────────────────────────────────────────────
# Anthropic fakes
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeAnthropicBlock:
    text: str


@dataclass
class _FakeAnthropicUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass
class _FakeAnthropicResponse:
    content: list[_FakeAnthropicBlock]
    model: str
    usage: _FakeAnthropicUsage | None = None


class _FakeAnthropicMessages:
    def __init__(self, recorder: dict) -> None:
        self.recorder = recorder

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.recorder.update(kwargs)
        return _FakeAnthropicResponse(
            content=[_FakeAnthropicBlock(text="ok")],
            model=kwargs["model"],
            usage=_FakeAnthropicUsage(
                input_tokens=12,
                output_tokens=34,
                cache_read_input_tokens=10,
                cache_creation_input_tokens=2,
            ),
        )


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.recorder: dict = {}
        self.messages = _FakeAnthropicMessages(self.recorder)


def test_anthropic_provider_attaches_cache_control_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """When enable_prompt_cache is True, the system arg becomes a list of
    text blocks with cache_control: ephemeral attached."""
    fake = _FakeAnthropicClient()
    monkeypatch.setattr(
        "consultant_brain.llm.anthropic_provider.Anthropic",
        lambda api_key: fake,
    )
    p = AnthropicProvider(api_key="sk-test")
    resp = p.chat(
        ChatRequest(
            system="You are a senior consultant.",
            user="Hello",
            model="claude-opus-4-7",
            max_tokens=128,
            enable_prompt_cache=True,
        )
    )
    assert resp.text == "ok"
    assert resp.model_used == "claude-opus-4-7"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 34
    assert resp.cache_read_tokens == 10
    assert resp.cache_creation_tokens == 2

    # The system arg is the list form with cache_control.
    sent = fake.recorder
    assert sent["model"] == "claude-opus-4-7"
    assert sent["max_tokens"] == 128
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["type"] == "text"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["system"][0]["text"] == "You are a senior consultant."
    assert sent["messages"] == [{"role": "user", "content": "Hello"}]


def test_anthropic_provider_drops_cache_control_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAnthropicClient()
    monkeypatch.setattr(
        "consultant_brain.llm.anthropic_provider.Anthropic",
        lambda api_key: fake,
    )
    p = AnthropicProvider(api_key="sk-test")
    p.chat(
        ChatRequest(
            system="prompt",
            user="hi",
            model="claude-opus-4-7",
            enable_prompt_cache=False,
        )
    )
    # When disabled, the system arg is the plain string.
    assert fake.recorder["system"] == "prompt"


def test_anthropic_provider_rejects_empty_key() -> None:
    with pytest.raises(ProviderError):
        AnthropicProvider(api_key="")


# ────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible fakes
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakePromptTokensDetails:
    cached_tokens: int


@dataclass
class _FakeOpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _FakePromptTokensDetails | None = None


@dataclass
class _FakeOpenAIResponse:
    choices: list[_FakeChoice]
    model: str
    usage: _FakeOpenAIUsage | None = None


class _FakeChatCompletions:
    def __init__(self, recorder: dict) -> None:
        self.recorder = recorder

    def create(self, **kwargs: Any) -> _FakeOpenAIResponse:
        self.recorder.update(kwargs)
        return _FakeOpenAIResponse(
            choices=[_FakeChoice(message=_FakeMessage(content="ok"))],
            model=kwargs["model"],
            usage=_FakeOpenAIUsage(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=_FakePromptTokensDetails(cached_tokens=80),
            ),
        )


class _FakeOpenAIClient:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.recorder: dict = {}
        self.chat = type(
            "Chat",
            (),
            {"completions": _FakeChatCompletions(self.recorder)},
        )()


@pytest.fixture
def _patched_openai(monkeypatch: pytest.MonkeyPatch):
    """Replace `openai.OpenAI` constructor so providers see our fake."""
    captured: dict = {}

    def factory(**kwargs: Any) -> _FakeOpenAIClient:
        client = _FakeOpenAIClient(**kwargs)
        captured["last"] = client
        return client

    monkeypatch.setattr(
        "consultant_brain.llm.openai_compatible.OpenAI",
        factory,
    )
    return captured


def test_openai_provider_uses_default_base_url(_patched_openai) -> None:
    p = OpenAIProvider(api_key="sk-openai")
    p.chat(ChatRequest(system="s", user="u", model="gpt-5"))
    client = _patched_openai["last"]
    assert client.init_kwargs["api_key"] == "sk-openai"
    # Default base_url for OpenAI is None — kwargs shouldn't even pass it.
    assert "base_url" not in client.init_kwargs


def test_openrouter_provider_sets_base_url_and_headers(_patched_openai) -> None:
    p = OpenRouterProvider(api_key="sk-or")
    p.chat(ChatRequest(system="s", user="u", model="anthropic/claude-opus-4-7"))
    client = _patched_openai["last"]
    assert client.init_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert "HTTP-Referer" in client.init_kwargs["default_headers"]
    assert "X-Title" in client.init_kwargs["default_headers"]


def test_deepseek_provider_uses_deepseek_base_url(_patched_openai) -> None:
    p = DeepSeekProvider(api_key="sk-deepseek")
    p.chat(ChatRequest(system="s", user="u", model="deepseek-chat"))
    client = _patched_openai["last"]
    assert client.init_kwargs["base_url"] == "https://api.deepseek.com/v1"


def test_kimi_provider_uses_moonshot_base_url(_patched_openai) -> None:
    p = KimiProvider(api_key="sk-kimi")
    p.chat(ChatRequest(system="s", user="u", model="kimi-k2"))
    client = _patched_openai["last"]
    assert client.init_kwargs["base_url"] == "https://api.moonshot.ai/v1"


def test_openai_compatible_serializes_system_role_in_messages(_patched_openai) -> None:
    """OpenAI-style APIs receive the system prompt as a 'system' role
    message, not as a separate kwarg the way Anthropic does."""
    p = OpenAIProvider(api_key="sk-openai")
    p.chat(
        ChatRequest(
            system="extract atoms",
            user="here is the transcript",
            model="gpt-5",
            max_tokens=512,
        )
    )
    client = _patched_openai["last"]
    sent = client.chat.completions.recorder
    assert sent["model"] == "gpt-5"
    assert sent["max_tokens"] == 512
    assert sent["messages"] == [
        {"role": "system", "content": "extract atoms"},
        {"role": "user", "content": "here is the transcript"},
    ]


def test_openai_compatible_returns_cached_tokens(_patched_openai) -> None:
    p = OpenAIProvider(api_key="sk-openai")
    resp = p.chat(ChatRequest(system="s", user="u", model="gpt-5"))
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.cache_read_tokens == 80
    assert resp.cache_creation_tokens is None  # not exposed by OpenAI


def test_openai_compatible_rejects_empty_key() -> None:
    with pytest.raises(ProviderError):
        OpenAIProvider(api_key="")
    with pytest.raises(ProviderError):
        DeepSeekProvider(api_key="")
