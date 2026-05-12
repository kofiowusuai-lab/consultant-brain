"""Tests for context_dumps/parsers.py — file → text dispatcher.

SDK calls (pypdf, python-docx, anthropic vision, Whisper) are mocked
through the injected fetcher/transcriber hooks so the tests stay
offline and fast.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from consultant_brain.context_dumps.parsers import (
    UnsupportedFileError,
    parse_to_text,
)


# ────────────────────────────────────────────────────────────────────────────
# Plain text
# ────────────────────────────────────────────────────────────────────────────


def test_parse_text_file(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("Hello world. Coffee with Reece was great.", encoding="utf-8")
    parsed = parse_to_text(f)
    assert parsed.kind_label == "text"
    assert "Coffee with Reece" in parsed.text
    assert parsed.warnings == []


def test_parse_markdown_file(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("# Heading\n\nBody", encoding="utf-8")
    parsed = parse_to_text(f)
    assert parsed.kind_label == "text"
    assert "Heading" in parsed.text


def test_parse_text_empty_warns(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    parsed = parse_to_text(f)
    assert parsed.text == ""
    assert "empty" in parsed.warnings[0].lower()


# ────────────────────────────────────────────────────────────────────────────
# PDF / DOCX via injected mocks
# ────────────────────────────────────────────────────────────────────────────


def test_parse_pdf_with_mock_reader(tmp_path: Path) -> None:
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    def fake_reader(path: Path) -> tuple[str, int, list[str]]:
        return "Page one text. Page two text.", 2, []

    parsed = parse_to_text(f, pdf_reader=fake_reader)
    assert parsed.kind_label == "pdf"
    assert parsed.page_count == 2
    assert "Page two text" in parsed.text


def test_parse_pdf_scanned_warns(tmp_path: Path) -> None:
    f = tmp_path / "scanned.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    def fake_reader(path: Path) -> tuple[str, int, list[str]]:
        return "", 3, ["PDF had no extractable text — image-only?"]

    parsed = parse_to_text(f, pdf_reader=fake_reader)
    assert parsed.text == ""
    assert any("no extractable" in w for w in parsed.warnings)


def test_parse_docx_with_mock_reader(tmp_path: Path) -> None:
    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK fake")

    def fake_reader(path: Path) -> tuple[str, list[str]]:
        return "Paragraph one.\n\nParagraph two.", []

    parsed = parse_to_text(f, docx_reader=fake_reader)
    assert parsed.kind_label == "docx"
    assert "Paragraph two" in parsed.text


# ────────────────────────────────────────────────────────────────────────────
# Image (Claude vision) via mocked transcriber
# ────────────────────────────────────────────────────────────────────────────


def test_parse_image_with_mock_vision(tmp_path: Path) -> None:
    f = tmp_path / "whiteboard.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")

    def fake_vision(path: Path) -> tuple[str, list[str]]:
        return "WHITEBOARD: Q3 targets · 200 emails / week · close 5", []

    parsed = parse_to_text(f, image_transcriber=fake_vision)
    assert parsed.kind_label == "image"
    assert "200 emails" in parsed.text


def test_parse_image_heic_warns_without_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real (un-mocked) image transcriber surfaces a clean HEIC warning
    instead of crashing. We patch the secrets getter so the auth check
    succeeds before the warning."""
    monkeypatch.setattr(
        "consultant_brain.context_dumps.parsers.get_anthropic_key",
        lambda: "fake-key",
    )
    f = tmp_path / "photo.heic"
    f.write_bytes(b"fake heic")
    parsed = parse_to_text(f)
    assert parsed.text == ""
    assert any("HEIC" in w for w in parsed.warnings)


# ────────────────────────────────────────────────────────────────────────────
# Audio via mocked transcriber
# ────────────────────────────────────────────────────────────────────────────


def test_parse_audio_with_mock_whisper(tmp_path: Path) -> None:
    f = tmp_path / "coffee.m4a"
    f.write_bytes(b"fake audio")

    def fake_whisper(path: Path) -> tuple[str, list[str]]:
        return "You: pricing. Them: We can do six figures.", []

    parsed = parse_to_text(f, audio_transcriber=fake_whisper)
    assert parsed.kind_label == "audio"
    assert "six figures" in parsed.text


# ────────────────────────────────────────────────────────────────────────────
# ZIP — depth-1 recursion
# ────────────────────────────────────────────────────────────────────────────


def test_parse_zip_recurses_into_entries(tmp_path: Path) -> None:
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("notes.txt", "Notes file content. Reece + AI.")
        zf.writestr("photo.jpg", b"\xff\xd8\xff\xe0 fake jpeg")

    def fake_vision(path: Path) -> tuple[str, list[str]]:
        return "Photo shows handwritten 'AI strategy'.", []

    parsed = parse_to_text(zpath, image_transcriber=fake_vision)
    assert parsed.kind_label == "zip"
    # Both entries' content should be present.
    assert "Notes file content" in parsed.text
    assert "AI strategy" in parsed.text
    # Section dividers + entry names embedded.
    assert "notes.txt" in parsed.text
    assert "photo.jpg" in parsed.text


def test_parse_zip_skips_macos_metadata(tmp_path: Path) -> None:
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("__MACOSX/.DS_Store", "junk")
        zf.writestr("note.md", "Real content")
    parsed = parse_to_text(zpath)
    assert "junk" not in parsed.text
    assert "Real content" in parsed.text


def test_parse_zip_warns_on_unsupported_entries(tmp_path: Path) -> None:
    zpath = tmp_path / "mixed.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data.xyz", "bytes")
        zf.writestr("note.txt", "kept")
    parsed = parse_to_text(zpath)
    assert "kept" in parsed.text
    assert any("data.xyz" in w for w in parsed.warnings)


# ────────────────────────────────────────────────────────────────────────────
# Unsupported
# ────────────────────────────────────────────────────────────────────────────


def test_parse_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "what.xyz"
    f.write_bytes(b"???")
    with pytest.raises(UnsupportedFileError):
        parse_to_text(f)


def test_parse_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_to_text(tmp_path / "nope.txt")
