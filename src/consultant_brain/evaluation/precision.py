"""Retrieval precision metric.

Loads a hand-labeled corpus JSON with N queries, each carrying a list of
`relevant_atom_ids`. Runs `retrieve()` against each query and computes
precision@k — the fraction of the top-k results that are in the
relevance set.

Corpus format (JSON):

    {
      "queries": [
        {
          "name": "price_question",
          "window": "what does this cost like ballpark",
          "client": "Reece",
          "call_type": "consultingCall",
          "relevant_atom_ids": ["TUNE_PRICE_ANCHOR_AAAAAAAAA1", ...]
        },
        ...
      ]
    }

We DON'T require the user to label every atom in the vault — only the
ones they consider relevant. Precision computed against this allowlist
is the right metric for "did the retriever surface a good one in the
top k", which is what the live copilot's UI cares about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from consultant_brain.retrieve import retrieve
from consultant_brain.schemas import CallType


@dataclass(frozen=True, slots=True)
class PrecisionQueryResult:
    """Per-query precision breakdown."""

    name: str
    relevant_ids: tuple[str, ...]
    returned_ids: tuple[str, ...]  # top-k from retrieve() in rank order
    p_at_1: float
    p_at_3: float
    p_at_5: float
    top_rank_of_relevant: int | None  # 1-indexed; None if no relevant atom appeared


@dataclass(frozen=True, slots=True)
class PrecisionReport:
    """Aggregate precision across the whole corpus."""

    corpus_size: int
    mean_p_at_1: float
    mean_p_at_3: float
    mean_p_at_5: float
    perfect_top1_count: int  # queries where the top hit was relevant
    queries: list[PrecisionQueryResult] = field(default_factory=list)


def compute_precision(*, vault_root: Path, corpus_path: Path) -> PrecisionReport:
    """Run every query in the corpus through retrieve() + compute precision."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])

    per_query: list[PrecisionQueryResult] = []
    for q in queries:
        result = retrieve(
            transcript_window=q["window"],
            client=q.get("client"),
            call_type=CallType(q.get("call_type", "consultingCall")),
            vault_root=vault_root,
        )
        # Use the FULL ranked list (all three layers concatenated), not the
        # panel slice — precision@k is a retrieval metric not a UI slice metric.
        all_hits = result.all
        returned_ids = tuple(rh.hit.id for rh in all_hits)
        relevant_ids = tuple(q.get("relevant_atom_ids", []))

        per_query.append(
            PrecisionQueryResult(
                name=q.get("name", "(unnamed)"),
                relevant_ids=relevant_ids,
                returned_ids=returned_ids,
                p_at_1=_precision_at_k(returned_ids, relevant_ids, 1),
                p_at_3=_precision_at_k(returned_ids, relevant_ids, 3),
                p_at_5=_precision_at_k(returned_ids, relevant_ids, 5),
                top_rank_of_relevant=_first_relevant_rank(returned_ids, relevant_ids),
            )
        )

    if not per_query:
        return PrecisionReport(
            corpus_size=0,
            mean_p_at_1=0.0,
            mean_p_at_3=0.0,
            mean_p_at_5=0.0,
            perfect_top1_count=0,
            queries=[],
        )

    n = len(per_query)
    return PrecisionReport(
        corpus_size=n,
        mean_p_at_1=sum(q.p_at_1 for q in per_query) / n,
        mean_p_at_3=sum(q.p_at_3 for q in per_query) / n,
        mean_p_at_5=sum(q.p_at_5 for q in per_query) / n,
        perfect_top1_count=sum(1 for q in per_query if q.p_at_1 == 1.0),
        queries=per_query,
    )


# ────────────────────────────────────────────────────────────────────────────


def _precision_at_k(returned: tuple[str, ...], relevant: tuple[str, ...], k: int) -> float:
    if k <= 0:
        return 0.0
    if not returned:
        return 0.0
    top_k = returned[:k]
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for atom_id in top_k if atom_id in relevant_set)
    return hits / k


def _first_relevant_rank(returned: tuple[str, ...], relevant: tuple[str, ...]) -> Optional[int]:
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    for i, atom_id in enumerate(returned, start=1):
        if atom_id in relevant_set:
            return i
    return None
