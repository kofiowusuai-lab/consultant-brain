"""OpenAI provider — direct to api.openai.com."""

from __future__ import annotations

from consultant_brain.llm.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    # Use the SDK's built-in default. OpenAI auto-caches prefix tokens
    # ≥1024 tokens — no special header / flag needed.
    default_base_url = None
