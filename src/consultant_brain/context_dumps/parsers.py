"""File → plain text dispatcher for context dumps.

One public function: `parse_to_text(path) -> ParsedDoc`. The right
backend gets picked off the file extension. Optional fetcher /
transcriber hooks let tests skip real SDK calls.

Per extension:
  .pdf                                  → pypdf
  .docx                                 → python-docx
  .txt / .md / .markdown                → stdlib read
  .jpg / .jpeg / .png / .heic / .webp   → Claude vision (anthropic SDK)
  .mp3 / .wav / .m4a / .ogg / .flac /
  .mp4 / .mov                           → OpenAI Whisper (Phase 9 reuse)
  .zip                                  → recurse depth-1
"""

from __future__ import annotations

import base64
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from consultant_brain.secrets import SecretNotFoundError, get_anthropic_key
from consultant_brain.sources.whisper import (
    WhisperError,
    transcribe_local_audio,
)


PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".text"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".mp4", ".mov", ".aac", ".webm"}
ZIP_EXTS = {".zip"}

# Anthropic accepts image media-types as exact MIME strings. Map our
# extensions onto them; falls back to image/jpeg when uncertain.
_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Claude doesn't accept HEIC directly — caller can convert ahead of
# time or we surface a warning. Keep the mapping so the warning is
# the only failure mode (not a 400).
_IMAGE_PROMPT = (
    "Transcribe any text visible in this image verbatim. Then describe "
    "the rest of the image in 2-4 factual sentences — who/what is "
    "shown, what context is suggested by signage/timestamps/handwriting. "
    "If this looks like a screenshot of a chat, render the conversation "
    "as You: / Them: lines. Do NOT speculate about meaning."
)


class UnsupportedFileError(RuntimeError):
    """File extension we don't have a parser for."""


@dataclass(frozen=True, slots=True)
class ParsedDoc:
    """Normalized output of every parser. `text` is the searchable
    representation we feed the extractor; `kind_label` is the human
    string the UI surfaces ('pdf', 'audio', 'zip')."""

    text: str
    kind_label: str
    page_count: Optional[int] = None
    duration_seconds: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


def parse_to_text(
    path: Path,
    *,
    pdf_reader: Optional[Callable[[Path], tuple[str, int, list[str]]]] = None,
    docx_reader: Optional[Callable[[Path], tuple[str, list[str]]]] = None,
    image_transcriber: Optional[Callable[[Path], tuple[str, list[str]]]] = None,
    audio_transcriber: Optional[Callable[[Path], tuple[str, list[str]]]] = None,
) -> ParsedDoc:
    """Dispatch on file extension.

    Every backend is optional + injectable so tests can short-circuit
    real SDK calls. In production, omitted hooks fall through to the
    real implementations.
    """
    if not path.exists():
        raise FileNotFoundError(f"Context-dump file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        warnings = [] if text else ["File was empty."]
        return ParsedDoc(text=text, kind_label="text", warnings=warnings)

    if suffix in PDF_EXTS:
        reader = pdf_reader or _default_pdf_reader
        text, page_count, warnings = reader(path)
        return ParsedDoc(
            text=text,
            kind_label="pdf",
            page_count=page_count,
            warnings=warnings,
        )

    if suffix in DOCX_EXTS:
        reader = docx_reader or _default_docx_reader
        text, warnings = reader(path)
        return ParsedDoc(text=text, kind_label="docx", warnings=warnings)

    if suffix in IMAGE_EXTS:
        transcriber = image_transcriber or _default_image_transcriber
        text, warnings = transcriber(path)
        return ParsedDoc(text=text, kind_label="image", warnings=warnings)

    if suffix in AUDIO_EXTS:
        transcriber = audio_transcriber or _default_audio_transcriber
        text, warnings = transcriber(path)
        return ParsedDoc(text=text, kind_label="audio", warnings=warnings)

    if suffix in ZIP_EXTS:
        return _parse_zip(
            path,
            pdf_reader=pdf_reader,
            docx_reader=docx_reader,
            image_transcriber=image_transcriber,
            audio_transcriber=audio_transcriber,
        )

    raise UnsupportedFileError(
        f"No parser for {suffix or '(no extension)'}. Supported: "
        f"text, pdf, docx, image (jpg/png/etc.), audio (mp3/wav/etc.), zip."
    )


# ────────────────────────────────────────────────────────────────────────────
# Default backends
# ────────────────────────────────────────────────────────────────────────────


def _default_pdf_reader(path: Path) -> tuple[str, int, list[str]]:
    """pypdf — page-by-page text extraction. Image-only PDFs return
    empty text + a warning; v1 doesn't auto-OCR them."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pypdf missing. Run `uv add pypdf` then retry."
        ) from exc

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    text = "\n\n".join(c.strip() for c in chunks if c.strip())
    warnings: list[str] = []
    if not text:
        warnings.append(
            "PDF had no extractable text — looks like a scanned/image PDF. "
            "Drop the page in as a JPG/PNG instead so the vision path can OCR it."
        )
    return text, len(reader.pages), warnings


def _default_docx_reader(path: Path) -> tuple[str, list[str]]:
    """python-docx — paragraph-by-paragraph join."""
    try:
        from docx import Document  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "python-docx missing. Run `uv add python-docx` then retry."
        ) from exc

    document = Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    warnings = [] if text else ["DOCX had no extractable paragraphs."]
    return text, warnings


def _default_image_transcriber(path: Path) -> tuple[str, list[str]]:
    """Claude vision — one-shot OCR + factual description."""
    try:
        api_key = get_anthropic_key()
    except SecretNotFoundError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing — image parsing needs the Anthropic key in "
            "secrets.json (`anthropic-api-key`)."
        ) from exc

    suffix = path.suffix.lower()
    media_type = _IMAGE_MEDIA_TYPES.get(suffix)
    warnings: list[str] = []
    if media_type is None:
        if suffix in {".heic", ".heif"}:
            warnings.append(
                "Claude vision doesn't accept HEIC directly. Convert to JPG/PNG "
                "(macOS: open in Preview → File → Export As → JPEG) and re-upload."
            )
            return "", warnings
        # Unknown image suffix → try jpeg + warn
        media_type = "image/jpeg"
        warnings.append(f"Unknown image suffix {suffix}; trying as image/jpeg.")

    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "anthropic SDK missing. Run `uv add anthropic` then retry."
        ) from exc

    image_b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": _IMAGE_PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Claude vision call failed: {exc}") from exc

    text_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str):
            text_parts.append(block_text)
    text = "\n".join(p.strip() for p in text_parts if p.strip()).strip()
    if not text:
        warnings.append("Claude vision returned no text.")
    return text, warnings


def _default_audio_transcriber(path: Path) -> tuple[str, list[str]]:
    """OpenAI Whisper, local-file path (Phase 9 module, refactored)."""
    try:
        text = transcribe_local_audio(path)
    except WhisperError as exc:
        raise RuntimeError(str(exc)) from exc
    warnings = [] if text else ["Whisper returned empty text."]
    return text, warnings


def _parse_zip(
    path: Path,
    *,
    pdf_reader,
    docx_reader,
    image_transcriber,
    audio_transcriber,
) -> ParsedDoc:
    """Depth-1 zip handling — extract every direct child, parse each
    individually, stitch the results into one ParsedDoc with section
    dividers so the extractor can see what came from which file."""
    import tempfile

    sections: list[str] = []
    warnings: list[str] = []
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory(prefix="brain-zip-") as tmpdir:
        tmp = Path(tmpdir)
        for entry in zf.namelist():
            if entry.endswith("/") or entry.startswith("__MACOSX/"):
                continue
            entry_path = tmp / Path(entry).name
            try:
                with zf.open(entry) as src, entry_path.open("wb") as dst:
                    dst.write(src.read())
            except Exception as exc:
                warnings.append(f"Could not extract {entry}: {exc}")
                continue

            try:
                parsed = parse_to_text(
                    entry_path,
                    pdf_reader=pdf_reader,
                    docx_reader=docx_reader,
                    image_transcriber=image_transcriber,
                    audio_transcriber=audio_transcriber,
                )
            except UnsupportedFileError as exc:
                warnings.append(f"Skipped {entry}: {exc}")
                continue
            except Exception as exc:
                warnings.append(f"Failed to parse {entry}: {exc}")
                continue

            warnings.extend(f"{entry}: {w}" for w in parsed.warnings)
            if parsed.text:
                sections.append(
                    f"### {entry} ({parsed.kind_label})\n\n{parsed.text}"
                )

    combined = "\n\n---\n\n".join(sections)
    if not combined:
        warnings.insert(0, "ZIP archive yielded no parseable content.")
    return ParsedDoc(text=combined, kind_label="zip", warnings=warnings)
