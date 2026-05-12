"""Vault backup + restore.

`consultant-brain backup` tars the vault root into a gzip-compressed
archive at `~/Documents/ConsultantBrain-backups/<ISO-timestamp>.tar.gz`.
The LanceDB index lives inside `~/ConsultantBrain/.lancedb/`, so the
single tar of the vault root captures it without special handling.

`consultant-brain restore --from <tar> --vault <target>` reverses the
operation. Refuses to overwrite a non-empty target unless `--force` is
passed.

Tar.gz over tar.zst: gzip is universally available, the vault is
small (typically <50MB for a year of calls), and the dependency footprint
matters more than the few extra MB.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BACKUP_DIR = Path.home() / "Documents" / "ConsultantBrain-backups"


@dataclass(frozen=True, slots=True)
class BackupResult:
    archive_path: Path
    archive_size_bytes: int
    file_count: int


@dataclass(frozen=True, slots=True)
class RestoreResult:
    vault_root: Path
    file_count: int
    bytes_written: int


def create_backup(*, vault_root: Path, out_path: Path | None = None) -> BackupResult:
    """Tar the vault into a gzip archive. Creates the backups directory
    if it doesn't exist."""
    vault_root = vault_root.expanduser().resolve()
    if not vault_root.is_dir():
        raise FileNotFoundError(f"Vault not found: {vault_root}")

    if out_path is None:
        DEFAULT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        out_path = DEFAULT_BACKUP_DIR / f"vault-{stamp}.tar.gz"
    else:
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with tarfile.open(out_path, "w:gz") as tar:
        for entry in sorted(vault_root.rglob("*")):
            if entry.is_file():
                arcname = entry.relative_to(vault_root.parent)
                tar.add(entry, arcname=str(arcname), recursive=False)
                file_count += 1

    size = out_path.stat().st_size
    return BackupResult(archive_path=out_path, archive_size_bytes=size, file_count=file_count)


def restore_backup(*, archive_path: Path, vault_root: Path, force: bool = False) -> RestoreResult:
    """Extract a backup tarball into the target vault root.

    Refuses to overwrite a non-empty vault unless `force=True`. After
    extraction the LanceDB index is preserved in place since it was
    captured as plain files inside the vault tree.
    """
    archive_path = archive_path.expanduser().resolve()
    vault_root = vault_root.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if vault_root.exists() and any(vault_root.iterdir()) and not force:
        raise FileExistsError(
            f"Target vault {vault_root} is not empty. Pass --force to overwrite."
        )

    vault_root.mkdir(parents=True, exist_ok=True)
    file_count = 0
    bytes_written = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Drop the leading directory in the archive (the vault name);
            # rewrite paths into the target vault_root.
            parts = Path(member.name).parts
            if not parts:
                continue
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            target = vault_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            data = src.read()
            target.write_bytes(data)
            bytes_written += len(data)
            file_count += 1
    return RestoreResult(vault_root=vault_root, file_count=file_count, bytes_written=bytes_written)
