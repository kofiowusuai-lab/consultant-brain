"""OpenRouter provider — one key, dozens of routed models.

OpenRouter's API is OpenAI-compatible at https://openrouter.ai/api/v1.
We attach the recommended `HTTP-Referer` + `X-Title` headers so requests
show up nicely in the OpenRouter dashboard analytics. Model strings
follow the OpenRouter slug convention (e.g. `anthropic/claude-sonnet-4`,
`google/gemini-2.0-pro`, `meta-llama/llama-3.1-70b-instruct`).
"""

from __future__ import annotations

from consultant_brain.llm.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_headers = {
        "HTTP-Referer": "https://github.com/kofiowusuai-lab/consultant-brain",
        "X-Title": "Consultant Brain",
    }
