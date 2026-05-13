"""Phase 11 — backfill `client_org_id` on existing atoms.

Atoms minted before Phase 11 carry a `client` display name but no
`client_org_id` UUID. After the Phase 11 mint paths went live, new
atoms write the UUID at creation time, but the historical tail
(Reece's 84 atoms in the production vault, plus anything ingested
during the early ConsultantBrain phases) still need a one-shot
backfill.

`run_reconcile_org_ids()` walks `03_Atoms/*.md`, resolves each
atom's display name through `CRMResolver`, and rewrites the
frontmatter (and LanceDB row) with the matching UUID. Atoms whose
display name doesn't appear in the CRM yet are reported as
`skipped_no_match` so the caller can decide whether to add the org
and re-run, and atoms that already carry a UUID are reported as
`already_set` so the command is idempotent — re-running after the
first pass should report `updated: 0`.

Knowledge atoms (`source_kind != "call"`) often have no client at
all (e.g., generic AI training videos with no `--for-client`); we
skip them silently via `skipped_no_client`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from consultant_brain.crm.resolver import CRMResolver
from consultant_brain.embedder import LanceVaultIndex
from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom
from consultant_brain.vault import VaultLayout, write_atom


@dataclass(frozen=True, slots=True)
class ReconcileSummary:
    """Outcome of one reconcile-org-ids run. Returned to the CLI for
    the summary line + to tests for assertions."""

    scanned: int
    updated: int
    already_set: int
    skipped_no_client: int
    skipped_no_match: list[str]  # client display names with no CRM hit
    failed: list[Path]
    dry_run: bool

    def summary_line(self) -> str:
        verb = "Would update" if self.dry_run else "Updated"
        parts = [
            f"Reconcile: scanned {self.scanned}",
            f"{verb.lower()} {self.updated}",
            f"already-set {self.already_set}",
        ]
        if self.skipped_no_client:
            parts.append(f"no-client {self.skipped_no_client}")
        if self.skipped_no_match:
            names = sorted(set(self.skipped_no_match))
            parts.append(
                f"no-match {len(names)} ({', '.join(names[:5])}"
                + ("…)" if len(names) > 5 else ")")
            )
        if self.failed:
            parts.append(f"failed {len(self.failed)}")
        return ", ".join(parts)


def run_reconcile_org_ids(
    *,
    vault_root: Path,
    resolver: CRMResolver | None = None,
    dry_run: bool = False,
) -> ReconcileSummary:
    """Walk the atoms dir; for each atom with `client` but no
    `client_org_id`, resolve a UUID via the CRM and rewrite both the
    markdown file and the LanceDB row.

    The function is idempotent: subsequent calls find a populated
    `client_org_id` and report `already_set` instead of touching the
    file.
    """
    layout = VaultLayout.for_root(vault_root)
    if not layout.atoms_dir.exists():
        return ReconcileSummary(
            scanned=0,
            updated=0,
            already_set=0,
            skipped_no_client=0,
            skipped_no_match=[],
            failed=[],
            dry_run=dry_run,
        )

    if resolver is None:
        resolver = CRMResolver()

    # Open the index lazily — a vault that's never been embedded won't
    # have a 00_System/lancedb/ dir yet and that's fine, the markdown
    # rewrites still proceed. LanceVaultIndex.upsert() creates the
    # table on first write.
    index = LanceVaultIndex(layout) if not dry_run else None

    scanned = 0
    updated = 0
    already_set = 0
    skipped_no_client = 0
    skipped_no_match: list[str] = []
    failed: list[Path] = []

    for atom_path in sorted(layout.atoms_dir.glob("*.md")):
        scanned += 1
        try:
            atom = _atom_from_markdown(atom_path)
        except Exception:
            failed.append(atom_path)
            continue

        if atom.client_org_id is not None:
            already_set += 1
            continue
        if not atom.client:
            skipped_no_client += 1
            continue

        uuid = resolver.resolve_uuid(atom.client)
        if uuid is None:
            skipped_no_match.append(atom.client)
            continue

        if dry_run:
            updated += 1
            continue

        # Atom is frozen — clone with the populated UUID, then rewrite
        # the markdown + re-upsert the index row so retrieval can use
        # the new column.
        try:
            new_atom = atom.model_copy(update={"client_org_id": uuid})
            write_atom(layout, new_atom)
            if index is not None:
                try:
                    index.upsert(new_atom)
                except Exception:
                    # Embedding failure is non-fatal — the markdown
                    # carries the truth and a later `reindex` pass
                    # repairs the row.
                    pass
            updated += 1
        except Exception:
            failed.append(atom_path)

    return ReconcileSummary(
        scanned=scanned,
        updated=updated,
        already_set=already_set,
        skipped_no_client=skipped_no_client,
        skipped_no_match=skipped_no_match,
        failed=failed,
        dry_run=dry_run,
    )
