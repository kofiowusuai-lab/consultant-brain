"""Local embeddings + vector index.

Ollama's `nomic-embed-text` produces 768-dim vectors. LanceDB stores them
at `<vault>/00_System/lancedb/atoms/` — file-backed, no server. The
schema mirrors the Atom frontmatter just closely enough for query results
to show useful context (id, type, body excerpt, client, call) without
having to re-read the markdown files.

Phase 1 ships:
  - `embed_text(text)` for one-off embeddings
  - `LanceVaultIndex` with `upsert(atom)` + `query(text, top_n)`
  - A `consultant-brain query "..."` CLI subcommand
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lancedb
import ollama
import pyarrow as pa

from consultant_brain.schemas import Atom
from consultant_brain.vault import VaultLayout


EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
ATOMS_TABLE = "atoms"


# ────────────────────────────────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────────────────────────────────


def embed_text(text: str, *, model: str = EMBEDDING_MODEL) -> list[float]:
    """Synchronous Ollama embedding call. Returns a 768-dim float vector.

    Raises:
        RuntimeError: if Ollama isn't running, the model isn't pulled, or
            the returned vector has the wrong dimension.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    try:
        response = ollama.embeddings(model=model, prompt=text)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama embedding failed for model={model!r}. "
            "Is `ollama serve` running and the model pulled?"
        ) from exc
    vector = response.get("embedding")
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIM}-dim embedding, got {type(vector).__name__} "
            f"of length {len(vector) if isinstance(vector, list) else 'n/a'}"
        )
    return vector


# ────────────────────────────────────────────────────────────────────────────
# LanceDB index
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AtomHit:
    """One query result. Includes enough fields to render a useful CLI line
    without re-reading the markdown."""

    id: str
    type: str
    body: str
    client: str | None
    call: str
    call_type: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine distance → similarity. LanceDB's cosine distance is in
        [0, 2] (0 = identical direction, 2 = opposite). Convert to a [0,1]
        score where 1 is best."""
        return max(0.0, min(1.0, 1.0 - self.distance / 2.0))

    def format_line(self) -> str:
        body_excerpt = self.body if len(self.body) <= 120 else self.body[:117] + "..."
        client = f" {self.client}" if self.client else ""
        return f"[{self.type:<11}] {self.similarity:.2f}  {body_excerpt}{client}"


def _atoms_schema() -> pa.Schema:
    """PyArrow schema for the atoms LanceDB table. Keeps query results
    self-contained so callers don't have to round-trip back through the
    vault markdown for common fields.
    """
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("type", pa.string()),
            pa.field("body", pa.string()),
            pa.field("client", pa.string()),  # empty string for None
            pa.field("call", pa.string()),
            pa.field("call_type", pa.string()),
            pa.field("confidence", pa.float32()),
        ]
    )


class LanceVaultIndex:
    """Wraps the LanceDB table at `<vault>/00_System/lancedb/`. Created on
    first upsert; safe to open repeatedly.
    """

    def __init__(self, layout: VaultLayout) -> None:
        self.layout = layout
        self.lancedb_path = layout.system_dir / "lancedb"
        self.lancedb_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.lancedb_path))
        self._schema = _atoms_schema()

    def _existing_tables(self) -> set[str]:
        """LanceDB 0.13 changed list_tables() to return a paginated response
        object with a `.tables` attribute. Older versions returned a plain
        list. This helper papers over both shapes."""
        result = self._db.list_tables()
        names = getattr(result, "tables", result)
        return set(names)

    def _table(self):
        if ATOMS_TABLE in self._existing_tables():
            return self._db.open_table(ATOMS_TABLE)
        return self._db.create_table(ATOMS_TABLE, schema=self._schema)

    def upsert(self, atom: Atom, *, vector: list[float] | None = None) -> None:
        """Embed (or use the provided vector) + write one atom. Replaces
        existing rows with the same atom ID, so re-ingest is idempotent.
        """
        if vector is None:
            vector = embed_text(atom.body)
        row = {
            "id": atom.id,
            "vector": vector,
            "type": atom.type.value,
            "body": atom.body,
            "client": atom.client or "",
            "call": atom.call,
            "call_type": atom.call_type.value,
            "confidence": float(atom.confidence),
        }
        table = self._table()
        # Delete-then-insert because LanceDB's merge_insert is unstable across
        # versions. The atom ID is a deterministic hash, so duplicates are
        # rare in practice — this stays fast.
        table.delete(f"id = '{atom.id}'")
        table.add([row])

    def upsert_many(self, atoms: Iterable[Atom]) -> int:
        count = 0
        for atom in atoms:
            self.upsert(atom)
            count += 1
        return count

    def query(self, text: str, *, top_n: int = 5) -> list[AtomHit]:
        if not text or not text.strip():
            return []
        if ATOMS_TABLE not in self._existing_tables():
            return []
        table = self._db.open_table(ATOMS_TABLE)
        vector = embed_text(text)
        # Cosine because nomic-embed-text vectors are unit-normalized;
        # L2 (the LanceDB default) produces 100+ distances on these.
        results = table.search(vector).metric("cosine").limit(top_n).to_list()
        hits: list[AtomHit] = []
        for row in results:
            client = row.get("client") or None
            hits.append(
                AtomHit(
                    id=row["id"],
                    type=row["type"],
                    body=row["body"],
                    client=client if client else None,
                    call=row["call"],
                    call_type=row["call_type"],
                    distance=float(row.get("_distance", 0.0)),
                )
            )
        return hits

    def atom_count(self) -> int:
        if ATOMS_TABLE not in self._existing_tables():
            return 0
        return self._db.open_table(ATOMS_TABLE).count_rows()
