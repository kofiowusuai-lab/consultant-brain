"""Shared base class for OpenAI-compatible providers.

OpenAI, OpenRouter, DeepSeek, and Kimi all expose Chat Completions
endpoints with the same request shape. Only the base URL + headers
differ. Centralizing the wire shape here means one place to fix
streaming bugs, retry logic, etc.
"""

from __future__ import annotations

from openai import OpenAI

from consultant_brain.llm.provider import (
    ChatRequest,
    ChatResponse,
    ProviderError,
)


class OpenAICompatibleProvider:
    """Common implementation. Subclasses fix `name`, `base_url`, and
    optionally `default_headers`."""

    name: str = "openai_compatible"
    default_base_url: str | None = None  # None = OpenAI's hosted endpoint
    default_headers: dict[str, str] = {}

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(f"{self.name} requires a non-empty API key")
        kwargs: dict = {"api_key": api_key}
        url = base_url or self.default_base_url
        if url:
            kwargs["base_url"] = url
        if self.default_headers or extra_headers:
            merged = {**self.default_headers, **(extra_headers or {})}
            kwargs["default_headers"] = merged
        self._client = OpenAI(**kwargs)

    def chat(self, request: ChatRequest) -> ChatResponse:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        try:
            response = self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"{self.name} call failed: {exc}") from exc

        if not response.choices:
            raise ProviderError(f"{self.name} returned no choices")
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ProviderError(f"{self.name} returned empty text")

        usage = getattr(response, "usage", None)
        cached = None
        if usage and hasattr(usage, "prompt_tokens_details"):
            details = usage.prompt_tokens_details
            cached = getattr(details, "cached_tokens", None) if details else None
        return ChatResponse(
            text=text,
            model_used=getattr(response, "model", request.model),
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            cache_read_tokens=cached,
            cache_creation_tokens=None,  # not exposed by OpenAI-style APIs
        )
