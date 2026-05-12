"""Anthropic provider with prompt-caching support.

Adds Anthropic's `cache_control: {"type": "ephemeral"}` to the system
message when `request.enable_prompt_cache` is True. The system prompt
becomes a cached prefix; subsequent calls within 5 minutes that share
that prefix only pay for the user message. ~70% cost cut on our
extractor + moment-detector loops, which reuse system prompts verbatim.
"""

from __future__ import annotations

from anthropic import Anthropic

from consultant_brain.llm.provider import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    ProviderError,
)


class AnthropicProvider:
    """Implements LLMProvider. Wraps the official anthropic SDK."""

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ProviderError("AnthropicProvider requires a non-empty API key")
        self._client = Anthropic(api_key=api_key)

    def chat(self, request: ChatRequest) -> ChatResponse:
        # System message can be a string OR a list of blocks. We use the
        # list form to attach cache_control. Both shapes are valid.
        if request.enable_prompt_cache:
            system_param = [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_param = request.system

        try:
            response = self._client.messages.create(
                model=request.model,
                system=system_param,
                messages=[{"role": "user", "content": request.user}],
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic call failed: {exc}") from exc

        text = _concat_text_blocks(response)
        if not text:
            raise ProviderError("Anthropic returned no text blocks")

        usage = getattr(response, "usage", None)
        return ChatResponse(
            text=text,
            model_used=getattr(response, "model", request.model),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) if usage else None,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None) if usage else None,
        )


def _concat_text_blocks(response) -> str:
    """Anthropic responses are a list of content blocks; concatenate the
    `.text` of each. Tolerant of mock objects in tests."""
    blocks = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()
