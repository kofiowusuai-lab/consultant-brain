"""DeepSeek provider — direct to api.deepseek.com.

DeepSeek's API is OpenAI-compatible. Model strings: `deepseek-chat` for
the general-purpose model (DeepSeek-V3), `deepseek-reasoner` for the
reasoning model (R1). Dramatically cheaper than Sonnet for
moment-detection-style fast classification.
"""

from __future__ import annotations

from consultant_brain.llm.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
