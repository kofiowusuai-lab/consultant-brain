"""LLM provider abstraction.

One Protocol (`LLMProvider`) that every concrete provider (Anthropic,
OpenAI, OpenRouter, DeepSeek, Kimi) implements. The extractor + moment
detector + future LLM-driven features call providers through this
contract; provider switching becomes a one-line registry lookup.

Design constraints inherited from the existing extractor:
  - Synchronous (`def`, not `async def`) — the existing pipelines call
    these from FastAPI background tasks + the CLI; sync is simpler.
  - System + user message split — every LLM we target supports this.
  - max_tokens + model_id are per-call so individual call sites can tune.
  - cache_control is an optional hint — Anthropic uses it explicitly,
    OpenAI auto-caches identical prefixes ≥1024 tokens, others ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One inbound LLM call. Same shape across all providers."""

    system: str
    user: str
    model: str
    max_tokens: int = 4096
    # When True, the provider should mark the system block as cacheable
    # if its API supports prompt caching. Anthropic uses cache_control
    # blocks; OpenAI auto-caches on prefix match ≥1024 tokens; others
    # silently ignore.
    enable_prompt_cache: bool = True


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """One outbound LLM call response. Normalized across providers."""

    text: str
    model_used: str
    # Usage tokens. Optional because some providers don't return them.
    input_tokens: int | None = None
    output_tokens: int | None = None
    # When the provider exposes cache stats (Anthropic does, OpenAI does
    # in cached_tokens), surface them so the eval dashboard can graph
    # cache effectiveness.
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


class ProviderError(RuntimeError):
    """Raised when a provider can't fulfill a request (network down,
    auth failed, model unknown). Callers catch this — the brain never
    crashes on a single LLM failure."""


class LLMProvider(Protocol):
    """The contract every concrete provider implements."""

    name: str  # short identifier — "anthropic", "openai", "openrouter", etc.

    def chat(self, request: ChatRequest) -> ChatResponse:
        """One synchronous chat completion. Raises ProviderError on failure."""
        ...
