"""Kimi (Moonshot AI) provider — direct to api.moonshot.ai.

OpenAI-compatible. Model strings: `kimi-k2`, `moonshot-v1-8k`,
`moonshot-v1-32k`, `moonshot-v1-128k`. The K2 model is exceptional at
long-context tool use; useful when the extractor needs to read a 60-min
transcript without chunking.
"""

from __future__ import annotations

from consultant_brain.llm.openai_compatible import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    name = "kimi"
    default_base_url = "https://api.moonshot.ai/v1"
