"""Whisper transcription wrapper.

Downloads the audio for any yt-dlp-supported URL into a tmpdir,
then transcribes it via the OpenAI hosted Whisper API. Local whisper.cpp
isn't wired in v1 — the dependency footprint isn't worth it when the
OpenAI key is already required for other parts of the brain.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from consultant_brain.secrets import SecretNotFoundError, get_openai_key


class WhisperError(RuntimeError):
    """Raised when audio download or transcription fails."""


def transcribe_local_audio(
    audio_path: Path,
    *,
    model: str = "whisper-1",
) -> str:
    """Phase 10: transcribe a local audio file (mp3 / wav / m4a / ogg /
    flac / mp4 / mov) via the OpenAI hosted Whisper API.

    No yt-dlp step — the caller already has the file on disk. Used by
    the context-dump parser and any future "drop an audio file"
    pipeline. Same OpenAI key lookup + error shape as
    `transcribe_url_with_whisper`.
    """
    try:
        api_key = get_openai_key()
    except SecretNotFoundError as exc:
        raise WhisperError(
            "OPENAI_API_KEY missing — Whisper transcription requires the OpenAI key in "
            "secrets.json (`openai-api-key`)."
        ) from exc
    if not audio_path.exists() or not audio_path.is_file():
        raise WhisperError(f"Audio file not found: {audio_path}")
    return _whisper_transcribe(audio_path, api_key=api_key, model=model)


def transcribe_url_with_whisper(
    url: str,
    *,
    cookies_from_browser: Optional[str] = None,
    model: str = "whisper-1",
) -> str:
    """Download the URL's audio with yt-dlp + send to OpenAI Whisper.

    The audio file is downloaded to a tmp dir that's cleaned up
    automatically. We use whisper-1 (the hosted Whisper-v3) — fixed
    cheap model; not worth exposing as a knob until someone needs a
    different one.
    """
    try:
        api_key = get_openai_key()
    except SecretNotFoundError as exc:
        raise WhisperError(
            "OPENAI_API_KEY missing — Whisper transcription requires the OpenAI key in "
            "secrets.json (`openai-api-key`)."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="brain-whisper-") as tmpdir:
        audio_path = _download_audio(url, Path(tmpdir), cookies_from_browser)
        return _whisper_transcribe(audio_path, api_key=api_key, model=model)


def _download_audio(url: str, tmpdir: Path, cookies_from_browser: Optional[str]) -> Path:
    """yt-dlp downloads audio as mp3 (re-encoded) so the OpenAI API
    accepts it without further conversion."""
    output_template = str(tmpdir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
    ]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as exc:
        raise WhisperError(
            "yt-dlp not installed. Run `brew install yt-dlp` then retry."
        ) from exc
    if result.returncode != 0:
        raise WhisperError(
            f"yt-dlp failed (exit {result.returncode}): {result.stderr[-400:]}"
        )

    mp3s = list(tmpdir.glob("*.mp3"))
    if not mp3s:
        raise WhisperError("yt-dlp returned 0 but no mp3 file was produced.")
    return mp3s[0]


def _whisper_transcribe(audio_path: Path, *, api_key: str, model: str) -> str:
    """Stream the file to OpenAI's audio transcription endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise WhisperError(
            "openai package missing. Run `uv add openai`."
        ) from exc

    client = OpenAI(api_key=api_key)
    try:
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )
    except Exception as exc:
        raise WhisperError(f"OpenAI Whisper call failed: {exc}") from exc

    if isinstance(response, str):
        return response.strip()
    # SDK v2 returns an object whose .text holds the transcript.
    text = getattr(response, "text", None)
    if not text:
        raise WhisperError("OpenAI Whisper returned empty text.")
    return text.strip()
