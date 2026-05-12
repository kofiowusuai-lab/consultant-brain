"""Provider registry — string name → LLMProvider instance.

Three layers of provider selection, in order of precedence:
  1. Explicit kwarg passed to `build_provider(name=...)`.
  2. `BRAIN_LLM_PROVIDER` (extractor) / `BRAIN_MOMENT_PROVIDER` (moment
     detector) env vars — these let the Swift app pick providers per
     task via UserDefaults → env.
  3. Falls back to anthropic (the build default, lowest surprise).

Per-task overrides (extractor vs moment-detector) are wired in
extractor.py and moment_detector.py — they call `build_provider` with
the right env-var hint at construction time.

API keys are pulled from the centralized `secrets` module; a missing
key for the selected provider raises `ProviderError` (callers catch
and degrade — the brain never crashes a call over one missing key).
"""

from __future__ import annotations

import os
from typing import Literal

from consultant_brain.llm.anthropic_provider import AnthropicProvider
from consultant_brain.llm.deepseek_provider import DeepSeekProvider
from consultant_brain.llm.kimi_provider import KimiProvider
from consultant_brain.llm.openai_provider import OpenAIProvider
from consultant_brain.llm.openrouter_provider import OpenRouterProvider
from consultant_brain.llm.provider import LLMProvider, ProviderError
from consultant_brain.secrets import (
    SecretNotFoundError,
    get_anthropic_key,
    get_deepseek_key,
    get_kimi_key,
    get_openai_key,
    get_openrouter_key,
)


ProviderName = Literal["anthropic", "openai", "openrouter", "deepseek", "kimi"]

KNOWN_PROVIDERS: tuple[ProviderName, ...] = (
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
    "kimi",
)

# Default model per provider. Keeps the call sites from having to know
# every provider's "good default" — extractor / moment detector can
# still override with their own model strings, but if they don't this
# is what fires.
DEFAULT_MODEL: dict[str, str] = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-5",
    "openrouter": "anthropic/claude-opus-4-7",
    "deepseek": "deepseek-chat",
    "kimi": "kimi-k2",
}


def resolve_provider_name(
    explicit: str | None = None,
    *,
    env_var: str = "BRAIN_LLM_PROVIDER",
    default: ProviderName = "anthropic",
) -> ProviderName:
    """Pick which provider to use, with explicit > env > default."""
    raw = explicit or os.environ.get(env_var) or default
    raw = raw.strip().lower()
    if raw not in KNOWN_PROVIDERS:
        raise ProviderError(
            f"Unknown LLM provider: {raw!r}. Valid: {', '.join(KNOWN_PROVIDERS)}"
        )
    return raw  # type: ignore[return-value]


def build_provider(
    name: str | None = None,
    *,
    env_var: str = "BRAIN_LLM_PROVIDER",
    default: ProviderName = "anthropic",
) -> LLMProvider:
    """Instantiate the named provider with its API key resolved from
    secrets.json + env. Raises ProviderError if the key is missing.
    """
    resolved = resolve_provider_name(name, env_var=env_var, default=default)
    try:
        if resolved == "anthropic":
            return AnthropicProvider(api_key=get_anthropic_key())
        if resolved == "openai":
            return OpenAIProvider(api_key=get_openai_key())
        if resolved == "openrouter":
            return OpenRouterProvider(api_key=get_openrouter_key())
        if resolved == "deepseek":
            return DeepSeekProvider(api_key=get_deepseek_key())
        if resolved == "kimi":
            return KimiProvider(api_key=get_kimi_key())
    except SecretNotFoundError as exc:
        raise ProviderError(str(exc)) from exc
    # Unreachable thanks to resolve_provider_name's validation, but the
    # type checker doesn't know that.
    raise ProviderError(f"Provider not wired into registry: {resolved!r}")


def default_model_for(provider_name: str) -> str:
    """Sensible default model string for a given provider."""
    return DEFAULT_MODEL.get(provider_name, "")
