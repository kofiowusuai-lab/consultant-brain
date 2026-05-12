"""Read API keys from the Swift app's FileSecretStore JSON.

The Swift app at `~/ai-consultant-copilot/` stores Anthropic + OpenAI +
OpenRouter + DeepSeek + Kimi keys in `~/Library/Application Support/
Consultant Copilot/secrets.json` (mode 0600, plain JSON dict). We
piggy-back on that file instead of asking the user to duplicate the
key into a `.env` or a separate keychain entry.

Falls through to env vars if the file is missing — useful for CI.

Key names mirror the Swift app's account namespace exactly:
  anthropic-api-key
  openai-api-key
  openrouter-api-key
  deepseek-api-key
  kimi-api-key
"""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_SECRETS_PATH = (
    Path.home() / "Library" / "Application Support" / "Consultant Copilot" / "secrets.json"
)


class SecretNotFoundError(LookupError):
    """Raised when a requested account isn't present in any of the lookups."""


def _get_key(
    *,
    env_var: str,
    secrets_account: str,
    human_label: str,
    secrets_path: Path | None,
) -> str:
    """Shared lookup: env var first, then secrets.json. Raises if neither
    has the key."""
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    path = secrets_path or DEFAULT_SECRETS_PATH
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SecretNotFoundError(
                f"Could not parse {path} as JSON: {exc.msg}"
            ) from exc
        key = data.get(secrets_account)
        if isinstance(key, str) and key:
            return key

    raise SecretNotFoundError(
        f"{human_label} API key not found. Set {env_var} or add "
        f"'{secrets_account}' to {path}."
    )


def get_anthropic_key(secrets_path: Path | None = None) -> str:
    return _get_key(
        env_var="ANTHROPIC_API_KEY",
        secrets_account="anthropic-api-key",
        human_label="Anthropic",
        secrets_path=secrets_path,
    )


def get_openai_key(secrets_path: Path | None = None) -> str:
    return _get_key(
        env_var="OPENAI_API_KEY",
        secrets_account="openai-api-key",
        human_label="OpenAI",
        secrets_path=secrets_path,
    )


def get_openrouter_key(secrets_path: Path | None = None) -> str:
    return _get_key(
        env_var="OPENROUTER_API_KEY",
        secrets_account="openrouter-api-key",
        human_label="OpenRouter",
        secrets_path=secrets_path,
    )


def get_deepseek_key(secrets_path: Path | None = None) -> str:
    return _get_key(
        env_var="DEEPSEEK_API_KEY",
        secrets_account="deepseek-api-key",
        human_label="DeepSeek",
        secrets_path=secrets_path,
    )


def get_kimi_key(secrets_path: Path | None = None) -> str:
    return _get_key(
        env_var="KIMI_API_KEY",
        secrets_account="kimi-api-key",
        human_label="Kimi",
        secrets_path=secrets_path,
    )


def has_key(
    *,
    env_var: str,
    secrets_account: str,
    secrets_path: Path | None = None,
) -> bool:
    """Cheap presence check for /diagnostics. Doesn't raise on miss."""
    try:
        _get_key(
            env_var=env_var,
            secrets_account=secrets_account,
            human_label=secrets_account,
            secrets_path=secrets_path,
        )
        return True
    except SecretNotFoundError:
        return False
