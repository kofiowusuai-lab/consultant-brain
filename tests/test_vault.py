"""Vault writer tests — atomic writes, frontmatter round-trips, slug rules,
deterministic atom IDs, and idempotency.

Uses `tmp_path` everywhere so tests never touch the real ~/ConsultantBrain.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    AtomType,
    CallNote,
    CallType,
)
from consultant_brain.vault import (
    VAULT_SUBDIRS,
    VaultLayout,
    derive_atom_id,
    derive_call_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    read_frontmatter,
    slugify_client,
    write_atom,
    write_call_note,
)


# ────────────────────────────────────────────────────────────────────────────
# Skeleton
# ────────────────────────────────────────────────────────────────────────────


def test_ensure_vault_skeleton_creates_all_subdirs(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    for sub in VAULT_SUBDIRS:
        assert (layout.root / sub).is_dir(), f"missing subdir: {sub}"
    assert (layout.root / "README.md").is_file()
    assert (layout.root / ".gitignore").is_file()


def test_ensure_vault_skeleton_is_idempotent(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    # Write something into one subdir to detect destructive re-runs.
    canary = layout.atoms_dir / "canary.md"
    canary.write_text("don't touch me")
    ensure_vault_skeleton(layout)
    assert canary.read_text() == "don't touch me"


# ────────────────────────────────────────────────────────────────────────────
# Slugs + IDs
# ────────────────────────────────────────────────────────────────────────────


def test_slugify_client_lowercases_and_underscores() -> None:
    assert slugify_client("Reece") == "reece"
    assert slugify_client("Acme Industries") == "acme_industries"
    assert slugify_client("ABC-123 Co.") == "abc_123_co"
    assert slugify_client("  Trailing  ") == "trailing"


def test_slugify_client_rejects_empty_after_clean() -> None:
    with pytest.raises(ValueError):
        slugify_client("!!!")
    with pytest.raises(ValueError):
        slugify_client("")


def test_derive_atom_id_is_deterministic() -> None:
    args = dict(session_filename="session-2026-05-12T00.json", extractor_version=1, atom_index=3)
    a = derive_atom_id(**args)
    b = derive_atom_id(**args)
    assert a == b
    assert len(a) == 26
    # Different inputs → different IDs
    assert derive_atom_id(**{**args, "atom_index": 4}) != a
    assert derive_atom_id(**{**args, "extractor_version": 2}) != a
    assert derive_atom_id(**{**args, "session_filename": "other.json"}) != a


def test_derive_call_id_matches_callnote_pattern() -> None:
    call_id = derive_call_id(
        client_name="Reece",
        call_type=CallType.consulting_call,
        call_date=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )
    assert call_id == "2026-05-12_reece_consultingCall"


# ────────────────────────────────────────────────────────────────────────────
# Atom write/read
# ────────────────────────────────────────────────────────────────────────────


def _make_atom(**overrides) -> Atom:
    base = dict(
        id="0123456789ABCDEFGHIJKLMNOP",
        type=AtomType.objection,
        client="Reece",
        call="2026-05-12_reece_consultingCall",
        call_type=CallType.consulting_call,
        tags=["budget", "scope"],
        confidence=0.82,
        evidence_count=1,
        last_seen=date(2026, 5, 12),
        created_at=datetime(2026, 5, 12, 19, 42, 10, tzinfo=timezone.utc),
        status=AtomStatus.active,
        embedding_id="0123456789ABCDEFGHIJKLMNOP",
        body="Price came up before scope. Anchor risk for the rest of the call.",
    )
    base.update(overrides)
    return Atom(**base)


def test_write_atom_creates_file_with_frontmatter(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    atom = _make_atom()
    path = write_atom(layout, atom)
    assert path == layout.atom_file(atom.id)
    assert path.exists()

    fm = read_frontmatter(path)
    assert fm["id"] == atom.id
    assert fm["type"] == "objection"
    assert fm["call_type"] == "consultingCall"
    # Client + call rendered as wikilinks so Obsidian backlinks resolve.
    assert fm["client"] == "[[Reece]]"
    assert fm["call"] == "[[2026-05-12_reece_consultingCall]]"
    assert fm["tags"] == ["budget", "scope"]
    assert fm["confidence"] == 0.82

    # Body present.
    content = path.read_text(encoding="utf-8")
    assert "Price came up before scope" in content


def test_write_atom_overwrites_when_same_id(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)

    first = _make_atom(body="First body.")
    write_atom(layout, first)
    second = _make_atom(body="Updated body after re-ingest.")
    write_atom(layout, second)

    assert "Updated body after re-ingest" in layout.atom_file(first.id).read_text()
    assert "First body" not in layout.atom_file(first.id).read_text()
    # And exactly one atom file exists.
    assert len(list(layout.atoms_dir.glob("*.md"))) == 1


def test_atom_with_no_client_omits_client_key(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    atom = _make_atom(client=None)
    path = write_atom(layout, atom)
    fm = read_frontmatter(path)
    assert "client" not in fm


# ────────────────────────────────────────────────────────────────────────────
# Call note write/read
# ────────────────────────────────────────────────────────────────────────────


def _make_call_note(**overrides) -> CallNote:
    base = dict(
        id="2026-05-12_reece_consultingCall",
        client="Reece",
        call_type=CallType.consulting_call,
        date=date(2026, 5, 12),
        duration_minutes=47,
        source_session="session-2026-05-12T05-42-10Z.json",
        extractor_model="claude-sonnet-4-6",
        extractor_version=1,
        atom_count=2,
        created_at=datetime(2026, 5, 12, 19, 42, 10, tzinfo=timezone.utc),
        summary="Reece is sold on the outcome but anxious about price.",
        atom_ids=["AAAAAAAAAAAAAAAAAAAAAAAA01", "AAAAAAAAAAAAAAAAAAAAAAAA02"],
        transcript="You: ...\n\nThem: ...",
    )
    base.update(overrides)
    return CallNote(**base)


def test_write_call_note_renders_summary_atoms_and_transcript(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    note = _make_call_note()
    path = write_call_note(layout, note)
    text = path.read_text(encoding="utf-8")

    fm = read_frontmatter(path)
    assert fm["id"] == note.id
    assert fm["atom_count"] == 2

    assert "## Summary" in text
    assert "sold on the outcome" in text
    assert "## Atoms" in text
    for atom_id in note.atom_ids:
        assert f"[[{atom_id}]]" in text
    assert "<details>" in text
    assert "<summary>Full transcript</summary>" in text


def test_call_note_with_no_atoms_has_placeholder(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    note = _make_call_note(atom_ids=[], atom_count=0)
    path = write_call_note(layout, note)
    text = path.read_text(encoding="utf-8")
    assert "No atoms extracted" in text


# ────────────────────────────────────────────────────────────────────────────
# Client folder
# ────────────────────────────────────────────────────────────────────────────


def test_ensure_client_folder_creates_dir_and_placeholder(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    client_dir = ensure_client_folder(layout, "Reece")
    assert client_dir.is_dir()
    assert client_dir.name == "reece"
    placeholder = client_dir / "Reece.md"
    assert placeholder.is_file()
    fm = read_frontmatter(placeholder)
    assert fm["auto_generated"] is True


def test_ensure_client_folder_is_idempotent(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    ensure_client_folder(layout, "Reece")
    placeholder = layout.client_dir("reece") / "Reece.md"
    placeholder.write_text("USER EDITED — don't clobber")
    ensure_client_folder(layout, "Reece")
    assert placeholder.read_text() == "USER EDITED — don't clobber"


# ────────────────────────────────────────────────────────────────────────────
# Atomicity
# ────────────────────────────────────────────────────────────────────────────


def test_created_at_renders_as_utc_with_z_suffix(tmp_path: Path) -> None:
    """Regression: an earlier version called .astimezone() with no argument,
    which converts to system local time. We always want UTC."""
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    atom = _make_atom()
    path = write_atom(layout, atom)
    text = path.read_text(encoding="utf-8")
    # YAML may quote the value; accept either form.
    assert ("created_at: 2026-05-12T19:42:10Z" in text
            or "created_at: '2026-05-12T19:42:10Z'" in text), text
    # Must NOT contain a local-time offset like -0300 / +0100.
    for offset in ("-0300", "-0400", "-0500", "+0100", "+0900"):
        assert offset not in text, f"local-time offset {offset} leaked into output"


def test_atomic_write_does_not_leave_tmp_files(tmp_path: Path) -> None:
    layout = VaultLayout.for_root(tmp_path / "vault")
    ensure_vault_skeleton(layout)
    atom = _make_atom()
    write_atom(layout, atom)
    # No leftover .tmp files in the atoms dir.
    tmps = list(layout.atoms_dir.glob(".*.tmp"))
    assert tmps == []
