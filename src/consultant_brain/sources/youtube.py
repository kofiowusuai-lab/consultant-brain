"""YouTube transcript fetcher.

Two strategies, tried in order:

  1. **youtube-transcript-api** — fastest path, no auth required, hits the
     captions API directly. Works for any public video with manual or
     auto-generated captions.

  2. **yt-dlp + Whisper fallback** — when a video has no captions at all,
     or when the captions API returns garbage, download the audio with
     yt-dlp and transcribe with the Whisper module's preferred backend
     (OpenAI hosted, or a local whisper.cpp build if the user wires it).

The fallback is gated behind a flag — most videos have captions, and
Whisper costs money / time. Callers opt in with `allow_whisper=True`.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from consultant_brain.schemas import SourceKind
from consultant_brain.sources.types import SourceMaterial


YOUTUBE_URL_RE = re.compile(
    r"""
    (?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)
    (?P<id>[A-Za-z0-9_\-]{11})
    """,
    re.VERBOSE,
)


class YouTubeFetchError(RuntimeError):
    """Raised when neither the captions API nor Whisper can produce text."""


def is_youtube_url(url: str) -> bool:
    return YOUTUBE_URL_RE.search(url) is not None


def extract_video_id(url: str) -> Optional[str]:
    """Pull the 11-char video id from any YouTube URL shape we know."""
    match = YOUTUBE_URL_RE.search(url)
    return match.group("id") if match else None


def fetch_youtube(
    url: str,
    *,
    allow_whisper: bool = False,
    topic: Optional[str] = None,
    for_client: Optional[str] = None,
    transcript_fetcher=None,
    metadata_fetcher=None,
    whisper_transcriber=None,
) -> SourceMaterial:
    """Fetch a transcript + metadata for one YouTube URL.

    Network calls are routed through optional injected functions so
    tests can mock the whole flow without monkey-patching urllib3.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise YouTubeFetchError(f"Not a YouTube URL: {url}")

    transcript_text = ""
    transcript_lang = "en"

    fetcher = transcript_fetcher or _default_transcript_fetcher
    try:
        transcript_text, transcript_lang = fetcher(video_id)
    except Exception as exc:
        if not allow_whisper:
            raise YouTubeFetchError(
                f"No captions for {video_id} and --whisper not enabled: {exc}"
            ) from exc
        transcribe = whisper_transcriber or _default_whisper_transcriber
        try:
            transcript_text = transcribe(url)
        except Exception as inner:
            raise YouTubeFetchError(
                f"Captions unavailable and Whisper transcription failed: {inner}"
            ) from inner

    if not transcript_text.strip():
        raise YouTubeFetchError(f"Empty transcript for {video_id}")

    metadata = (metadata_fetcher or _default_metadata_fetcher)(video_id)
    return SourceMaterial(
        kind=SourceKind.youtube,
        url=f"https://www.youtube.com/watch?v={video_id}",
        source_id=f"youtube_{video_id}",
        title=metadata.title or f"YouTube video {video_id}",
        transcript=transcript_text.strip(),
        author=metadata.author,
        duration_seconds=metadata.duration_seconds,
        published_at=metadata.published_at,
        language=transcript_lang,
        topic=topic,
        for_client=for_client,
    )


# ────────────────────────────────────────────────────────────────────────────
# Metadata
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class YouTubeMetadata:
    title: Optional[str]
    author: Optional[str]
    duration_seconds: Optional[int]
    published_at: Optional[date]


def _default_metadata_fetcher(video_id: str) -> YouTubeMetadata:
    """Best-effort metadata via yt-dlp's --skip-download path. yt-dlp is
    a soft dependency: if it isn't installed we still return something
    usable (just title-less)."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-download",
                "--print",
                "%(title)s||%(uploader)s||%(duration)s||%(upload_date)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return YouTubeMetadata(None, None, None, None)
        parts = result.stdout.strip().split("||")
        if len(parts) != 4:
            return YouTubeMetadata(None, None, None, None)
        title, author, duration, upload_date = parts
        published = None
        if upload_date and upload_date.isdigit() and len(upload_date) == 8:
            try:
                published = datetime.strptime(upload_date, "%Y%m%d").date()
            except ValueError:
                published = None
        try:
            duration_int: Optional[int] = int(float(duration)) if duration not in ("NA", "") else None
        except (TypeError, ValueError):
            duration_int = None
        return YouTubeMetadata(
            title=title or None,
            author=author or None,
            duration_seconds=duration_int,
            published_at=published,
        )
    except FileNotFoundError:
        return YouTubeMetadata(None, None, None, None)
    except subprocess.TimeoutExpired:
        return YouTubeMetadata(None, None, None, None)


# ────────────────────────────────────────────────────────────────────────────
# Transcript paths
# ────────────────────────────────────────────────────────────────────────────


def _default_transcript_fetcher(video_id: str) -> tuple[str, str]:
    """Lazy import + best-language selection via youtube-transcript-api.

    Returns (plain-text transcript, language code). Raises if no
    transcript is available — caller catches and decides whether to
    fall back to Whisper.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api not installed. Run `uv add youtube-transcript-api`."
        ) from exc

    # The API exposes a transcript-list view we can pick from. Prefer
    # manual English captions; fall back to auto-generated English; fall
    # back to whatever the first available language is.
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception as exc:
        raise RuntimeError(f"No transcripts available for {video_id}: {exc}") from exc

    preferred_languages = ["en", "en-US", "en-GB"]
    transcript = None
    for lang in preferred_languages:
        try:
            transcript = listing.find_manually_created_transcript([lang])
            break
        except Exception:
            continue
    if transcript is None:
        for lang in preferred_languages:
            try:
                transcript = listing.find_generated_transcript([lang])
                break
            except Exception:
                continue
    if transcript is None:
        # Last-resort: pull the first transcript in any language and let
        # the extractor work in non-English.
        transcript = next(iter(listing), None)
    if transcript is None:
        raise RuntimeError(f"No transcripts at all for {video_id}")

    segments = transcript.fetch()
    text = " ".join(s.get("text", "").strip() for s in segments if s.get("text"))
    return text, getattr(transcript, "language_code", "en")


def _default_whisper_transcriber(url: str) -> str:
    """yt-dlp → audio mp3 → Whisper. Cheap path uses OpenAI's hosted
    Whisper API when the OpenAI key is configured."""
    from consultant_brain.sources.whisper import transcribe_url_with_whisper

    return transcribe_url_with_whisper(url)
