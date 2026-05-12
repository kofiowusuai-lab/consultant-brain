"""Phase 9 — source materials.

External knowledge ingestion. Each source (YouTube video, Instagram
Reel, blog article, podcast episode) is normalized into a
`SourceMaterial` dataclass so the rest of the pipeline (extractor,
vault writer) doesn't care where the transcript came from.

Public API:
  - `SourceMaterial` — the normalized in-memory shape.
  - `fetch_source(url)` — dispatcher that picks a kind-specific fetcher.
  - `KnowledgeFetchError` — raised when a transcript can't be obtained.
"""

from __future__ import annotations

from consultant_brain.sources.fetcher import (
    KnowledgeFetchError,
    fetch_source,
)
from consultant_brain.sources.types import SourceMaterial

__all__ = [
    "KnowledgeFetchError",
    "SourceMaterial",
    "fetch_source",
]
