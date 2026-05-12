"""Phase 8 items 10 + 11 — backup/restore round-trip + anonymized export."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from consultant_brain.backup import create_backup, restore_backup
from consultant_brain.export import export_vault
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


def _make_vault_with_one_atom(vault: Path) -> Path:
    layout = VaultLayout.for_root(vault)
    ensure_vault_skeleton(layout)
    atom_path = layout.atoms_dir / "01HX0000000000000000000001.md"
    fm_dict = {
        "id": "01HX0000000000000000000001",
        "type": "client_fact",
        "client": "Reece",
        "call": "[[2026-05-12_reece_consultingCall]]",
        "call_type": "consultingCall",
        "tags": ["funnel"],
        "confidence": 0.9,
        "evidence_count": 1,
        "last_seen": "2026-05-12",
        "created_at": "2026-05-12T19:42:10Z",
        "status": "active",
        "embedding_id": "01HX0000000000000000000001",
        "client_org_id": "11111111-1111-1111-1111-111111111111",
    }
    yaml_block = yaml.safe_dump(fm_dict, sort_keys=False, allow_unicode=True).rstrip()
    atom_path.write_text(
        f"---\n{yaml_block}\n---\n\nReece said they're running a funnel. Email pat@reece.io for the rollout.\n",
        encoding="utf-8",
    )
    return atom_path


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    src_vault = tmp_path / "vault"
    _make_vault_with_one_atom(src_vault)

    archive_dir = tmp_path / "backups"
    archive_dir.mkdir()
    archive_path = archive_dir / "v.tar.gz"
    backup = create_backup(vault_root=src_vault, out_path=archive_path)
    assert backup.archive_path == archive_path
    assert backup.archive_size_bytes > 0
    assert backup.file_count >= 1

    target = tmp_path / "restored"
    result = restore_backup(archive_path=archive_path, vault_root=target)
    assert result.file_count == backup.file_count
    restored_atom = target / "03_Atoms" / "01HX0000000000000000000001.md"
    assert restored_atom.exists()
    original = (src_vault / "03_Atoms" / "01HX0000000000000000000001.md").read_text(encoding="utf-8")
    assert restored_atom.read_text(encoding="utf-8") == original


def test_restore_refuses_non_empty_target_without_force(tmp_path: Path) -> None:
    src_vault = tmp_path / "vault"
    _make_vault_with_one_atom(src_vault)
    archive_path = tmp_path / "v.tar.gz"
    create_backup(vault_root=src_vault, out_path=archive_path)

    target = tmp_path / "restored"
    target.mkdir()
    (target / "marker.txt").write_text("don't clobber me")

    with pytest.raises(FileExistsError):
        restore_backup(archive_path=archive_path, vault_root=target)

    # --force overrides.
    restore_backup(archive_path=archive_path, vault_root=target, force=True)


def test_export_anonymize_redacts_client_names_and_emails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "consultant_brain.export.open_read_only",
        lambda: (_ for _ in ()).throw(__import__("consultant_brain.crm.sqlite_reader", fromlist=["CRMNotFoundError"]).CRMNotFoundError("no crm")),
    )

    vault = tmp_path / "vault"
    _make_vault_with_one_atom(vault)
    out = tmp_path / "export"
    result = export_vault(vault_root=vault, out_dir=out, anonymize=True)

    assert result.atom_count == 1
    exported = (out / "03_Atoms" / "01HX0000000000000000000001.md").read_text(encoding="utf-8")
    # Real client name is gone; alias is present.
    assert "Reece" not in exported
    assert "[CLIENT_1]" in exported
    # Email redacted.
    assert "pat@reece.io" not in exported
    assert "[EMAIL_" in exported
    # Org UUID stripped.
    assert "11111111-1111-1111-1111-111111111111" not in exported
    manifest = (out / "MANIFEST.md").read_text(encoding="utf-8")
    assert "anonymized: True" in manifest


def test_export_no_anonymize_preserves_identity(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _make_vault_with_one_atom(vault)
    out = tmp_path / "export"
    result = export_vault(vault_root=vault, out_dir=out, anonymize=False)
    exported = (out / "03_Atoms" / "01HX0000000000000000000001.md").read_text(encoding="utf-8")
    assert "Reece" in exported
    assert "pat@reece.io" in exported
    assert result.replacements_applied == 0
