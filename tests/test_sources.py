"""Source fetcher tests — YouTube URL parsing, Instagram dispatch, and
the dispatcher. Network calls are mocked through the injected fetcher
hooks so unit tests never hit the real APIs.
"""

from __future__ import annotations

from datetime import date

import pytest

from consultant_brain.schemas import SourceKind
from consultant_brain.sources import KnowledgeFetchError, fetch_source
from consultant_brain.sources.instagram import (
    InstagramFetchError,
    InstagramMetadata,
    extract_post_id,
    fetch_instagram,
    is_instagram_url,
)
from consultant_brain.sources.youtube import (
    YouTubeFetchError,
    YouTubeMetadata,
    extract_video_id,
    fetch_youtube,
    is_youtube_url,
)


# ────────────────────────────────────────────────────────────────────────────
# URL parsers
# ────────────────────────────────────────────────────────────────────────────


def test_extract_video_id_handles_every_youtube_url_shape() -> None:
    cases = {
        "https://www.youtube.com/watch?v=HD2RU2QZxJk": "HD2RU2QZxJk",
        "https://youtu.be/HD2RU2QZxJk?si=abc": "HD2RU2QZxJk",
        "https://www.youtube.com/embed/HD2RU2QZxJk": "HD2RU2QZxJk",
        "https://www.youtube.com/shorts/HD2RU2QZxJk": "HD2RU2QZxJk",
        "https://www.youtube.com/v/HD2RU2QZxJk?fs=1": "HD2RU2QZxJk",
    }
    for url, expected in cases.items():
        assert extract_video_id(url) == expected, url


def test_extract_video_id_rejects_non_youtube() -> None:
    assert extract_video_id("https://vimeo.com/123456") is None
    assert extract_video_id("not a url at all") is None


def test_is_youtube_url() -> None:
    assert is_youtube_url("https://youtu.be/HD2RU2QZxJk")
    assert not is_youtube_url("https://twitter.com/x/status/123")


def test_is_instagram_url() -> None:
    assert is_instagram_url("https://www.instagram.com/reel/Cabc1234/")
    assert is_instagram_url("https://instagram.com/p/Cxyz/")
    assert is_instagram_url("https://instagram.com/tv/CTV01/")
    assert not is_instagram_url("https://twitter.com/x/status/123")


def test_extract_post_id() -> None:
    assert extract_post_id("https://www.instagram.com/reel/Cabc1234/") == "Cabc1234"
    assert extract_post_id("https://instagram.com/p/Cxyz/?utm=1") == "Cxyz"
    assert extract_post_id("nope") is None


# ────────────────────────────────────────────────────────────────────────────
# YouTube fetch — happy path + Whisper fallback
# ────────────────────────────────────────────────────────────────────────────


def test_fetch_youtube_uses_captions_when_available() -> None:
    transcript_calls: list[str] = []

    def fake_transcript(video_id: str) -> tuple[str, str]:
        transcript_calls.append(video_id)
        return "We talk about ad bots and budget anchors today.", "en"

    def fake_meta(video_id: str) -> YouTubeMetadata:
        return YouTubeMetadata(
            title="Ad Bot Pricing",
            author="HormoziTalks",
            duration_seconds=423,
            published_at=date(2026, 5, 1),
        )

    result = fetch_youtube(
        "https://youtu.be/HD2RU2QZxJk?si=abc",
        transcript_fetcher=fake_transcript,
        metadata_fetcher=fake_meta,
    )
    assert result.kind == SourceKind.youtube
    assert result.source_id == "youtube_HD2RU2QZxJk"
    assert result.title == "Ad Bot Pricing"
    assert result.author == "HormoziTalks"
    assert result.duration_seconds == 423
    assert result.published_at == date(2026, 5, 1)
    assert "ad bots" in result.transcript
    assert transcript_calls == ["HD2RU2QZxJk"]


def test_fetch_youtube_raises_when_captions_missing_and_no_whisper() -> None:
    def fake_transcript(video_id: str) -> tuple[str, str]:
        raise RuntimeError("no captions")

    def fake_meta(video_id: str) -> YouTubeMetadata:
        return YouTubeMetadata(None, None, None, None)

    with pytest.raises(YouTubeFetchError, match="No captions"):
        fetch_youtube(
            "https://youtu.be/HD2RU2QZxJk",
            allow_whisper=False,
            transcript_fetcher=fake_transcript,
            metadata_fetcher=fake_meta,
        )


def test_fetch_youtube_falls_back_to_whisper_when_allowed() -> None:
    def fake_transcript(video_id: str) -> tuple[str, str]:
        raise RuntimeError("no captions")

    def fake_meta(video_id: str) -> YouTubeMetadata:
        return YouTubeMetadata(title="Long Video", author=None, duration_seconds=None, published_at=None)

    def fake_whisper(url: str) -> str:
        return "Whisper says this is the transcript."

    result = fetch_youtube(
        "https://www.youtube.com/watch?v=HD2RU2QZxJk",
        allow_whisper=True,
        transcript_fetcher=fake_transcript,
        metadata_fetcher=fake_meta,
        whisper_transcriber=fake_whisper,
    )
    assert "Whisper says" in result.transcript
    assert result.title == "Long Video"


def test_fetch_youtube_rejects_non_youtube_url() -> None:
    with pytest.raises(YouTubeFetchError):
        fetch_youtube("https://vimeo.com/whatever")


# ────────────────────────────────────────────────────────────────────────────
# Instagram fetch
# ────────────────────────────────────────────────────────────────────────────


def test_fetch_instagram_requires_whisper() -> None:
    with pytest.raises(InstagramFetchError, match="requires Whisper"):
        fetch_instagram(
            "https://www.instagram.com/reel/Cabc1234/",
            allow_whisper=False,
        )


def test_fetch_instagram_happy_path() -> None:
    def fake_meta(url: str, *, cookies_from_browser=None) -> InstagramMetadata:
        return InstagramMetadata(
            title="Reel about pricing",
            author="@hormozi",
            caption="3 mistakes when pricing services",
            duration_seconds=58,
            published_at=date(2026, 4, 30),
        )

    def fake_whisper(url: str, *, cookies_from_browser=None) -> str:
        return "Voice over: three mistakes ..."

    result = fetch_instagram(
        "https://www.instagram.com/reel/Cabc1234/",
        allow_whisper=True,
        metadata_fetcher=fake_meta,
        whisper_transcriber=fake_whisper,
    )
    assert result.kind == SourceKind.instagram
    assert result.source_id == "instagram_Cabc1234"
    assert result.author == "@hormozi"
    assert "Voice over" in result.transcript
    assert "## Post caption" in result.transcript


def test_fetch_instagram_propagates_whisper_failure() -> None:
    def fake_meta(url: str, *, cookies_from_browser=None) -> InstagramMetadata:
        return InstagramMetadata(None, None, None, None, None)

    def boom(url: str, *, cookies_from_browser=None) -> str:
        raise RuntimeError("network down")

    with pytest.raises(InstagramFetchError, match="Whisper"):
        fetch_instagram(
            "https://www.instagram.com/reel/Cabc1234/",
            allow_whisper=True,
            metadata_fetcher=fake_meta,
            whisper_transcriber=boom,
        )


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────────


def test_fetch_source_rejects_unknown_url() -> None:
    with pytest.raises(KnowledgeFetchError, match="No fetcher"):
        fetch_source("https://vimeo.com/12345")
