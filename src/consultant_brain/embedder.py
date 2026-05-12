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

# Phase 2 evaluation found that prepending the atom type as a bracketed tag
# improves clustering — objections cluster near other objections, commitments
# near other commitments — which helps the warm-layer ranker disambiguate.
# Toggleable so we can A/B test rapidly without rewriting embeddings.
USE_TYPE_PREFIX = True


def embedding_input_for(atom: Atom) -> str:
    """The string passed to the embedder for a given atom. Centralized so the
    embed-at-write path and the reindex path produce identical vectors.
    """
    if USE_TYPE_PREFIX:
        return f"[{atom.type.value.upper()}] {atom.body}"
    return atom.body


def embedding_input_for_query(text: str, *, type_hint: str | None = None) -> str:
    """Mirror image of embedding_input_for() for query strings. If we biased
    atoms with a type prefix, queries should match the same convention.
    A `type_hint` lets callers explicitly ask "find objections like this";
    otherwise we omit the prefix so queries match all types.
    """
    if USE_TYPE_PREFIX and type_hint:
        return f"[{type_hint.upper()}] {text}"
    return text


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
    + run warm-layer ranking + tag-based dedupe without re-reading markdown."""

    id: str
    type: str
    body: str
    client: str | None
    call: str
    call_type: str
    confidence: float
    last_seen: str  # ISO date string; "" if missing from the index
    tags: tuple[str, ...]
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine distance → similarity. LanceDB's cosine distance is in
        [0, 2] (0 = identical direction, 2 = opposite). Convert to a [0,1]
        score where 1 is best."""
        return max(0.0, min(1.0, 1.0 - self.distance / 2.0))

    @property
    def primary_tag(self) -> str | None:
        """First tag — used by warm-layer dedupe so we don't show 3 atoms
        about the same topic in the suggestion panel."""
        return self.tags[0] if self.tags else None

    def format_line(self) -> str:
        body_excerpt = self.body if len(self.body) <= 120 else self.body[:117] + "..."
        client = f" {self.client}" if self.client else ""
        return f"[{self.type:<11}] {self.similarity:.2f}  {body_excerpt}{client}"


def _atoms_schema() -> pa.Schema:
    """PyArrow schema for the atoms LanceDB table. Keeps query results
    self-contained so callers don't have to round-trip back through the
    vault markdown for common fields. Phase 2 added last_seen + tags for
    recency-boosted ranking + primary-tag dedupe.
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
            pa.field("last_seen", pa.string()),  # ISO date YYYY-MM-DD
            pa.field("tags", pa.list_(pa.string())),
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
            vector = embed_text(embedding_input_for(atom))
        row = {
            "id": atom.id,
            "vector": vector,
            "type": atom.type.value,
            "body": atom.body,
            "client": atom.client or "",
            "call": atom.call,
            "call_type": atom.call_type.value,
            "confidence": float(atom.confidence),
            "last_seen": atom.last_seen.isoformat(),
            "tags": list(atom.tags),
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

    def query(
        self,
        text: str,
        *,
        top_n: int = 5,
        client_filter: str | None = None,
        call_type_filter: str | None = None,
        exclude_atom_ids: set[str] | None = None,
        type_hint: str | None = None,
    ) -> list[AtomHit]:
        """Semantic search the index. Optional `client_filter` /
        `call_type_filter` push the constraint into LanceDB's SQL-ish where
        clause so filtered queries don't pay for ranking atoms we'd throw
        away. `exclude_atom_ids` lets the retrieval pipeline drop hot-layer
        atoms before the warm layer surfaces them again.
        """
        if not text or not text.strip():
            return []
        if ATOMS_TABLE not in self._existing_tables():
            return []
        table = self._db.open_table(ATOMS_TABLE)
        vector = embed_text(embedding_input_for_query(text, type_hint=type_hint))
        # Over-fetch by 3x when filters or excludes are active so the post-
        # filter top_n is still a real top_n.
        fetch_n = top_n * 3 if (client_filter or call_type_filter or exclude_atom_ids) else top_n

        search = table.search(vector).metric("cosine")
        where_clauses: list[str] = []
        if client_filter:
            where_clauses.append(f"client = '{_sql_escape(client_filter)}'")
        if call_type_filter:
            where_clauses.append(f"call_type = '{_sql_escape(call_type_filter)}'")
        if where_clauses:
            search = search.where(" AND ".join(where_clauses))

        rows = search.limit(fetch_n).to_list()

        excluded = exclude_atom_ids or set()
        hits: list[AtomHit] = []
        for row in rows:
            if row["id"] in excluded:
                continue
            client = row.get("client") or None
            hits.append(
                AtomHit(
                    id=row["id"],
                    type=row["type"],
                    body=row["body"],
                    client=client if client else None,
                    call=row["call"],
                    call_type=row["call_type"],
                    confidence=float(row.get("confidence", 0.5)),
                    last_seen=row.get("last_seen", "") or "",
                    tags=tuple(row.get("tags") or ()),
                    distance=float(row.get("_distance", 0.0)),
                )
            )
            if len(hits) >= top_n:
                break
        return hits

    def list_atom_ids(self) -> set[str]:
        """All atom IDs currently in the index. Used by `reindex` to figure
        out which markdown files still need embedding. Uses to_arrow() so
        we don't pull pandas into the dependency tree just for this lookup.
        """
        if ATOMS_TABLE not in self._existing_tables():
            return set()
        table = self._db.open_table(ATOMS_TABLE)
        arrow_table = table.search().limit(0).to_arrow()  # gets schema
        # Empty search returns 0 rows — switch to a full scan via head().
        full = table.head(table.count_rows() or 1)
        return {row["id"] for row in full.to_pylist()}

    def atom_count(self) -> int:
        if ATOMS_TABLE not in self._existing_tables():
            return 0
        return self._db.open_table(ATOMS_TABLE).count_rows()


def _sql_escape(value: str) -> str:
    """Escape single quotes for LanceDB's filter strings. The where()
    builder doesn't expose parameter binding, so we sanitize the values
    ourselves. The only escape character LanceDB needs is `'` → `''`."""
    return value.replace("'", "''")
