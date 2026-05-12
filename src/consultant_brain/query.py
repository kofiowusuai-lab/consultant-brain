"""Public surface for the `consultant-brain query` CLI subcommand.

Single function `run_query()` that the CLI calls. Kept as a thin wrapper
over `LanceVaultIndex.query` so the CLI stays decoupled from LanceDB's
APIs — easier to swap retrieval backends in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

from consultant_brain.embedder import AtomHit, LanceVaultIndex
from consultant_brain.vault import VaultLayout


def run_query(*, query_text: str, vault_root: Path, top_n: int = 5) -> list[AtomHit]:
    """Vector-search the vault's atom index. Returns ranked hits; empty
    list if the vault has no index yet.
    """
    layout = VaultLayout.for_root(vault_root)
    if not layout.system_dir.exists():
        return []
    index = LanceVaultIndex(layout)
    return index.query(query_text, top_n=top_n)
