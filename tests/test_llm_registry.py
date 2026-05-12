"""Tests for the LLM provider registry + secrets extensions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consultant_brain.llm.anthropic_provider import AnthropicProvider
from consultant_brain.llm.deepseek_provider import DeepSeekProvider
from consultant_brain.llm.kimi_provider import KimiProvider
from consultant_brain.llm.openai_provider import OpenAIProvider
from consultant_brain.llm.openrouter_provider import OpenRouterProvider
from consultant_brain.llm.provider import ProviderError
from consultant_brain.llm.registry import (
    DEFAULT_MODEL,
    KNOWN_PROVIDERS,
    build_provider,
    default_model_for,
    resolve_provider_name,
)
from consultant_brain.secrets import (
    SecretNotFoundError,
    get_deepseek_key,
    get_kimi_key,
    get_openai_key,
    get_openrouter_key,
    has_key,
)


# ────────────────────────────────────────────────────────────────────────────
# resolve_provider_name
# ────────────────────────────────────────────────────────────────────────────


def test_resolve_provider_name_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "openai")
    assert resolve_provider_name("deepseek") == "deepseek"


def test_resolve_provider_name_env_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_LLM_PROVIDER", "kimi")
    assert resolve_provider_name() == "kimi"


def test_resolve_provider_name_default_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_LLM_PROVIDER", raising=False)
    assert resolve_provider_name() == "anthropic"


def test_resolve_provider_name_rejects_unknown() -> None:
    with pytest.raises(ProviderError, match="Unknown LLM provider"):
        resolve_provider_name("not-a-real-provider")


def test_resolve_provider_name_uses_custom_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_MOMENT_PROVIDER", "deepseek")
    monkeypatch.delenv("BRAIN_LLM_PROVIDER", raising=False)
    assert resolve_provider_name(env_var="BRAIN_MOMENT_PROVIDER") == "deepseek"


def test_default_model_for_each_provider() -> None:
    for name in KNOWN_PROVIDERS:
        assert default_model_for(name) == DEFAULT_MODEL[name]


# ────────────────────────────────────────────────────────────────────────────
# build_provider — exercise every branch
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def _all_keys_in_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a secrets.json with every key + point env to it. Returns the
    path so individual tests can mutate it."""
    secrets = tmp_path / "secrets.json"
    secrets.write_text(
        json.dumps(
            {
                "anthropic-api-key": "sk-anthropic",
                "openai-api-key": "sk-openai",
                "openrouter-api-key": "sk-openrouter",
                "deepseek-api-key": "sk-deepseek",
                "kimi-api-key": "sk-kimi",
            }
        )
    )
    # Patch the default secrets path module-wide so every secret getter
    # sees the fixture file.
    monkeypatch.setattr(
        "consultant_brain.secrets.DEFAULT_SECRETS_PATH", secrets
    )
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "KIMI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return secrets


def test_build_provider_anthropic(_all_keys_in_secrets, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.llm.anthropic_provider.Anthropic",
        lambda api_key: object(),
    )
    p = build_provider("anthropic")
    assert isinstance(p, AnthropicProvider)


def test_build_provider_openai(_all_keys_in_secrets, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.llm.openai_compatible.OpenAI",
        lambda **kw: object(),
    )
    p = build_provider("openai")
    assert isinstance(p, OpenAIProvider)


def test_build_provider_openrouter(_all_keys_in_secrets, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.llm.openai_compatible.OpenAI",
        lambda **kw: object(),
    )
    p = build_provider("openrouter")
    assert isinstance(p, OpenRouterProvider)


def test_build_provider_deepseek(_all_keys_in_secrets, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.llm.openai_compatible.OpenAI",
        lambda **kw: object(),
    )
    p = build_provider("deepseek")
    assert isinstance(p, DeepSeekProvider)


def test_build_provider_kimi(_all_keys_in_secrets, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.llm.openai_compatible.OpenAI",
        lambda **kw: object(),
    )
    p = build_provider("kimi")
    assert isinstance(p, KimiProvider)


def test_build_provider_raises_when_secret_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty secrets.json + no env vars → ProviderError, NOT a leaky
    SecretNotFoundError."""
    empty = tmp_path / "secrets.json"
    empty.write_text("{}")
    monkeypatch.setattr("consultant_brain.secrets.DEFAULT_SECRETS_PATH", empty)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "KIMI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ProviderError, match="not found"):
        build_provider("anthropic")


# ────────────────────────────────────────────────────────────────────────────
# secrets — new key getters
# ────────────────────────────────────────────────────────────────────────────


def test_get_openai_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert get_openai_key() == "env-key"


def test_get_openrouter_key_from_secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"openrouter-api-key": "sk-or-fromfile"}))
    assert get_openrouter_key(secrets_path=secrets) == "sk-or-fromfile"


def test_get_deepseek_key_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    empty = tmp_path / "secrets.json"
    empty.write_text("{}")
    with pytest.raises(SecretNotFoundError, match="DeepSeek"):
        get_deepseek_key(secrets_path=empty)


def test_get_kimi_key_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"kimi-api-key": "sk-kimi"}))
    assert get_kimi_key(secrets_path=secrets) == "sk-kimi"


def test_has_key_returns_bool_for_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"anthropic-api-key": "k"}))
    assert has_key(
        env_var="ANTHROPIC_API_KEY",
        secrets_account="anthropic-api-key",
        secrets_path=secrets,
    )
    assert not has_key(
        env_var="OPENAI_API_KEY",
        secrets_account="openai-api-key",
        secrets_path=secrets,
    )
