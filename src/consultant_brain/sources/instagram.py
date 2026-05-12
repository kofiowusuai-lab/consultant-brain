"""Instagram Reel / video fetcher.

Instagram has no public transcript API. The pipeline:
  1. yt-dlp downloads the Reel's audio track (works for public Reels;
     private accounts need cookies via `--cookies-from-browser chrome`).
  2. Whisper transcribes the audio.
  3. yt-dlp also writes the post's caption + author metadata, which
     we keep so the knowledge note carries context.

If the user has *not* enabled Whisper (no OpenAI key wired up, no
local whisper binary), `fetch_instagram` raises — we don't have a
captions API fallback like YouTube does.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from consultant_brain.schemas import SourceKind
from consultant_brain.sources.types import SourceMaterial


INSTAGRAM_URL_RE = re.compile(
    r"instagram\.com/(?:reel|p|tv)/(?P<id>[A-Za-z0-9_\-]+)"
)


class InstagramFetchError(RuntimeError):
    """Raised when transcription or metadata extraction fails."""


def is_instagram_url(url: str) -> bool:
    return INSTAGRAM_URL_RE.search(url) is not None


def extract_post_id(url: str) -> Optional[str]:
    match = INSTAGRAM_URL_RE.search(url)
    return match.group("id") if match else None


def fetch_instagram(
    url: str,
    *,
    allow_whisper: bool = True,
    topic: Optional[str] = None,
    for_client: Optional[str] = None,
    audio_downloader=None,
    metadata_fetcher=None,
    whisper_transcriber=None,
    cookies_from_browser: Optional[str] = None,
) -> SourceMaterial:
    """Fetch + transcribe an Instagram Reel.

    Whisper is mandatory here (no captions API). When `allow_whisper`
    is False we raise immediately so the caller can surface a useful
    error rather than chewing through yt-dlp before failing.
    """
    if not allow_whisper:
        raise InstagramFetchError(
            "Instagram requires Whisper transcription; pass --whisper or wire OPENAI_API_KEY."
        )

    post_id = extract_post_id(url)
    if not post_id:
        raise InstagramFetchError(f"Not an Instagram URL: {url}")

    metadata = (metadata_fetcher or _default_metadata_fetcher)(
        url, cookies_from_browser=cookies_from_browser
    )
    transcribe = whisper_transcriber or _default_whisper_transcriber
    try:
        transcript_text = transcribe(url, cookies_from_browser=cookies_from_browser)
    except Exception as exc:
        raise InstagramFetchError(f"Whisper transcription failed: {exc}") from exc
    if not transcript_text.strip():
        raise InstagramFetchError("Empty transcript")

    body_lines = [transcript_text.strip()]
    if metadata.caption:
        body_lines.append("")
        body_lines.append(f"## Post caption\n{metadata.caption.strip()}")
    combined = "\n\n".join(body_lines)

    return SourceMaterial(
        kind=SourceKind.instagram,
        url=url,
        source_id=f"instagram_{post_id}",
        title=metadata.title or f"Instagram post {post_id}",
        transcript=combined,
        author=metadata.author,
        duration_seconds=metadata.duration_seconds,
        published_at=metadata.published_at,
        topic=topic,
        for_client=for_client,
    )


# ────────────────────────────────────────────────────────────────────────────
# Metadata
# ────────────────────────────────────────────────────────────────────────────


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InstagramMetadata:
    title: Optional[str]
    author: Optional[str]
    caption: Optional[str]
    duration_seconds: Optional[int]
    published_at: Optional[date]


def _default_metadata_fetcher(
    url: str,
    *,
    cookies_from_browser: Optional[str] = None,
) -> InstagramMetadata:
    """yt-dlp --dump-json gives us the full info block. We pull the bits
    that matter and discard the rest."""
    cmd = ["yt-dlp", "--no-download", "--dump-single-json"]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except FileNotFoundError:
        return InstagramMetadata(None, None, None, None, None)
    except subprocess.TimeoutExpired:
        return InstagramMetadata(None, None, None, None, None)
    if result.returncode != 0:
        return InstagramMetadata(None, None, None, None, None)
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return InstagramMetadata(None, None, None, None, None)
    published = None
    upload_date = info.get("upload_date") or ""
    if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
        try:
            published = datetime.strptime(upload_date, "%Y%m%d").date()
        except ValueError:
            published = None
    duration = info.get("duration")
    try:
        duration_int = int(duration) if duration else None
    except (TypeError, ValueError):
        duration_int = None
    return InstagramMetadata(
        title=info.get("title") or None,
        author=(info.get("uploader") or info.get("channel") or None),
        caption=info.get("description") or None,
        duration_seconds=duration_int,
        published_at=published,
    )


def _default_whisper_transcriber(
    url: str,
    *,
    cookies_from_browser: Optional[str] = None,
) -> str:
    from consultant_brain.sources.whisper import transcribe_url_with_whisper

    return transcribe_url_with_whisper(url, cookies_from_browser=cookies_from_browser)
