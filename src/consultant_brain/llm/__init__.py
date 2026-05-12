"""LLM provider abstraction — one Protocol, five concrete providers.

Call sites should import from this module rather than reaching into the
concrete provider modules; the public surface here stays stable even
when individual providers change.
"""

from __future__ import annotations

from consultant_brain.llm.anthropic_provider import AnthropicProvider
from consultant_brain.llm.deepseek_provider import DeepSeekProvider
from consultant_brain.llm.kimi_provider import KimiProvider
from consultant_brain.llm.openai_provider import OpenAIProvider
from consultant_brain.llm.openrouter_provider import OpenRouterProvider
from consultant_brain.llm.provider import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    ProviderError,
)
from consultant_brain.llm.registry import (
    DEFAULT_MODEL,
    KNOWN_PROVIDERS,
    build_provider,
    default_model_for,
    resolve_provider_name,
)

__all__ = [
    "AnthropicProvider",
    "ChatRequest",
    "ChatResponse",
    "DEFAULT_MODEL",
    "DeepSeekProvider",
    "KimiProvider",
    "KNOWN_PROVIDERS",
    "LLMProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderError",
    "build_provider",
    "default_model_for",
    "resolve_provider_name",
]
