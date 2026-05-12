"""Dispatcher: URL → SourceMaterial.

Picks the right fetcher (YouTube / Instagram / generic article-someday)
based on the URL shape. Each fetcher returns a normalized
`SourceMaterial`, raises `KnowledgeFetchError` on failure.
"""

from __future__ import annotations

from typing import Optional

from consultant_brain.sources.instagram import (
    InstagramFetchError,
    fetch_instagram,
    is_instagram_url,
)
from consultant_brain.sources.types import SourceMaterial
from consultant_brain.sources.youtube import (
    YouTubeFetchError,
    fetch_youtube,
    is_youtube_url,
)


class KnowledgeFetchError(RuntimeError):
    """Single error type the orchestrator catches. Wraps the
    source-specific exception with context."""


def fetch_source(
    url: str,
    *,
    allow_whisper: bool = False,
    topic: Optional[str] = None,
    for_client: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> SourceMaterial:
    """Dispatch on URL shape. Adds defensive context to whatever the
    concrete fetcher raises."""
    if is_youtube_url(url):
        try:
            return fetch_youtube(
                url,
                allow_whisper=allow_whisper,
                topic=topic,
                for_client=for_client,
            )
        except YouTubeFetchError as exc:
            raise KnowledgeFetchError(f"YouTube fetch failed: {exc}") from exc

    if is_instagram_url(url):
        try:
            return fetch_instagram(
                url,
                allow_whisper=allow_whisper,
                topic=topic,
                for_client=for_client,
                cookies_from_browser=cookies_from_browser,
            )
        except InstagramFetchError as exc:
            raise KnowledgeFetchError(f"Instagram fetch failed: {exc}") from exc

    raise KnowledgeFetchError(
        f"No fetcher knows how to handle this URL yet: {url}. "
        "Supported: youtube.com, youtu.be, instagram.com/(reel|p|tv)."
    )
