"""Read API keys from the Swift app's FileSecretStore JSON.

The Swift app at `~/ai-consultant-copilot/` stores Anthropic + OpenAI keys in
`~/Library/Application Support/Consultant Copilot/secrets.json` (mode 0600,
plain JSON dict). We piggy-back on that file instead of asking the user to
duplicate the key into a `.env` or a separate keychain entry.

Falls through to env vars if the file is missing — useful for CI.
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


def get_anthropic_key(secrets_path: Path | None = None) -> str:
    """Read the Anthropic API key. Lookup order: explicit env var,
    then the Swift app's secrets.json. Raises SecretNotFoundError if
    neither source has it.
    """
    env_value = os.environ.get("ANTHROPIC_API_KEY")
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
        key = data.get("anthropic-api-key")
        if isinstance(key, str) and key:
            return key

    raise SecretNotFoundError(
        "Anthropic API key not found. Set ANTHROPIC_API_KEY or add "
        f"'anthropic-api-key' to {path}."
    )
