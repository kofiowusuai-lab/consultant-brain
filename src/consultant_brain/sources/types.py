"""Shared data shapes for the sources subpackage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from consultant_brain.schemas import SourceKind


@dataclass(frozen=True, slots=True)
class SourceMaterial:
    """One fetched external source, normalized.

    Every `fetch_source(url)` call returns this regardless of source type.
    The extractor reads it as a generic "transcript with metadata"; the
    vault writer renders the 09_Knowledge/ note from it.
    """

    kind: SourceKind
    url: str
    # Stable id used as the atom's `call` field — `youtube_<id>`,
    # `instagram_<id>`, etc. Derived deterministically from the URL so
    # re-running `learn` on the same source overwrites in place.
    source_id: str
    title: str
    transcript: str  # plain text, one paragraph per turn / segment
    author: Optional[str] = None  # channel name / handle
    duration_seconds: Optional[int] = None
    published_at: Optional[date] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.utcnow())
    language: str = "en"
    # Caller-supplied. `--topic ai-sales` becomes a tag on every minted
    # atom; `--for-client Reece` ties the knowledge to that client (so
    # retrieval can prefer it for that client).
    topic: Optional[str] = None
    for_client: Optional[str] = None

    @property
    def slug(self) -> str:
        """Filename-safe slug for the 09_Knowledge/ markdown note."""
        return self.source_id
