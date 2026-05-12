"""Three-layer retrieval for the consultant copilot's live suggestion panel.

Given a transcript window + the current client + the call type, returns
a small ranked list of atoms the consultant should be reminded of right
now. Three layers from the master prompt:

  Layer 1 (Hot)  — atoms already known about this client; always loaded.
  Layer 2 (Warm) — vector search on the rolling transcript window,
                   filtered to current call_type, boosted by recency +
                   confidence, deduped by primary tag.
  Layer 3 (Cold) — pattern matching against 04_Patterns/. Phase 1/2 have
                   no patterns yet, so this layer is a stub that returns
                   []. Phase 6 distillation will populate it.

The public surface is `retrieve()` returning a `RetrievalResult`. The
`suggest` CLI subcommand and (later) the FastAPI service both call this
single function — keeps ranking logic out of the request handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from consultant_brain.embedder import AtomHit, LanceVaultIndex
from consultant_brain.schemas import CallType
from consultant_brain.vault import VaultLayout


# Phase 2 tuning constants. Each is a single number that, when changed,
# moves retrieval quality in a documented direction. Keeping them at the
# module level (vs scattered through code) makes the tuning corpus useful —
# you can flip a constant + re-run the corpus to see what shifts.

# Max atoms emitted per layer. Master prompt says "1 hot + 2 warm + 1 cold"
# for the suggestion panel; we surface the full ranking via the
# RetrievalResult and let the caller (CLI / FastAPI) slice as needed.
HOT_LAYER_MAX = 4
WARM_LAYER_MAX = 8
COLD_LAYER_MAX = 4

# Warm-layer ranking knobs.
RECENCY_HALF_LIFE_DAYS = 30.0  # atoms older than this halve in recency weight
CONFIDENCE_WEIGHT = 0.20  # how much extractor confidence biases ranking
RECENCY_WEIGHT = 0.15  # how much "last seen recently" biases ranking
SIMILARITY_WEIGHT = 0.65  # raw semantic similarity is still the dominant signal

# Dedupe: max atoms per (type, primary_tag) bucket in the warm layer.
WARM_PER_TAG_CAP = 2


@dataclass(frozen=True, slots=True)
class RankedHit:
    """An AtomHit plus the components of its rank score. Surfacing the
    components is intentional — `suggest --explain` can show why an atom
    ranked where it did, which makes the tuning corpus debuggable.
    """

    hit: AtomHit
    similarity: float
    recency: float  # [0, 1]
    confidence: float  # [0, 1]
    score: float  # weighted blend used to rank
    layer: str  # "hot" | "warm" | "cold"
    reason: str  # human-readable hint for --explain mode


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Output of one retrieval call. Layers are kept separate so the UI can
    render them with distinct styling (Hot = chip, Warm = card, Cold = play).
    """

    hot: list[RankedHit] = field(default_factory=list)
    warm: list[RankedHit] = field(default_factory=list)
    cold: list[RankedHit] = field(default_factory=list)

    @property
    def all(self) -> list[RankedHit]:
        return [*self.hot, *self.warm, *self.cold]

    def top_for_panel(self, *, hot: int = 1, warm: int = 2, cold: int = 1) -> list[RankedHit]:
        """The exact slice the live suggestion panel renders."""
        return [*self.hot[:hot], *self.warm[:warm], *self.cold[:cold]]


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────


def retrieve(
    *,
    transcript_window: str,
    client: str | None,
    call_type: CallType,
    vault_root: Path,
    now: Optional[date] = None,
) -> RetrievalResult:
    """The function the CLI + future FastAPI service both call.

    Doesn't open the LanceDB index if the vault has no `00_System/` dir —
    returns an empty result so the live copilot stays alive on a fresh
    machine where no ingests have happened yet.
    """
    layout = VaultLayout.for_root(vault_root)
    if not layout.system_dir.exists():
        return RetrievalResult()

    index = LanceVaultIndex(layout)
    if index.atom_count() == 0:
        return RetrievalResult()

    today = now or datetime.now(timezone.utc).date()

    hot = _hot_layer(index=index, client=client, transcript_window=transcript_window, today=today)
    hot_ids = {hit.hit.id for hit in hot}

    warm = _warm_layer(
        index=index,
        transcript_window=transcript_window,
        call_type=call_type,
        excluded_ids=hot_ids,
        today=today,
    )

    cold = _cold_layer(vault_root=vault_root, transcript_window=transcript_window, today=today)

    return RetrievalResult(hot=hot, warm=warm, cold=cold)


# ────────────────────────────────────────────────────────────────────────────
# Layer 1 — Hot
# ────────────────────────────────────────────────────────────────────────────


def _hot_layer(
    *, index: LanceVaultIndex, client: str | None, transcript_window: str, today: date
) -> list[RankedHit]:
    """Atoms already on file about this client, ranked by semantic relevance
    to the current transcript window so the panel surfaces the most-likely-
    useful one first. If no client is given, hot is empty — those atoms only
    matter when we know who's on the call.
    """
    if not client or not transcript_window.strip():
        return []
    raw_hits = index.query(
        text=transcript_window,
        client_filter=client,
        top_n=HOT_LAYER_MAX,
    )
    return [
        _rank_hit(hit, layer="hot", today=today, reason="known about this client")
        for hit in raw_hits
    ]


# ────────────────────────────────────────────────────────────────────────────
# Layer 2 — Warm
# ────────────────────────────────────────────────────────────────────────────


def _warm_layer(
    *,
    index: LanceVaultIndex,
    transcript_window: str,
    call_type: CallType,
    excluded_ids: set[str],
    today: date,
) -> list[RankedHit]:
    """Vector search across all atoms (this client + others), filtered by
    the current call_type so we don't surface a cold-call opener during a
    closing call. Hot-layer hits are excluded so we don't double-count.

    After raw vector ranking we re-rank with a weighted blend of similarity
    + recency + confidence, then dedupe by (type, primary_tag) bucket so the
    panel doesn't show three "objection: budget" atoms in a row.
    """
    if not transcript_window.strip():
        return []
    # Pull twice the cap so dedupe still leaves us with WARM_LAYER_MAX atoms.
    raw_hits = index.query(
        text=transcript_window,
        call_type_filter=call_type.value,
        exclude_atom_ids=excluded_ids,
        top_n=WARM_LAYER_MAX * 2,
    )
    ranked = [
        _rank_hit(hit, layer="warm", today=today, reason="semantic match on transcript")
        for hit in raw_hits
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    deduped = _dedupe_by_tag(ranked, per_bucket=WARM_PER_TAG_CAP)
    return deduped[:WARM_LAYER_MAX]


def _dedupe_by_tag(ranked: list[RankedHit], *, per_bucket: int) -> list[RankedHit]:
    """Cap how many atoms share a (type, primary_tag) bucket so the panel
    doesn't repeat the same topic. Stable wrt input order so the top-ranked
    atom in each bucket always wins.
    """
    buckets: dict[tuple[str, str | None], int] = {}
    kept: list[RankedHit] = []
    for r in ranked:
        key = (r.hit.type, r.hit.primary_tag)
        if buckets.get(key, 0) >= per_bucket:
            continue
        buckets[key] = buckets.get(key, 0) + 1
        kept.append(r)
    return kept


# ────────────────────────────────────────────────────────────────────────────
# Layer 3 — Cold
# ────────────────────────────────────────────────────────────────────────────


def _cold_layer(*, vault_root: Path, transcript_window: str, today: date) -> list[RankedHit]:
    """Pattern matching against `04_Patterns/` populated by the Phase 6
    distiller. Patterns whose primary_tag or example phrases echo in the
    transcript window fire. Falls back to an empty list when (a) no
    patterns exist yet, (b) the window is empty.
    """
    if not transcript_window.strip():
        return []
    # Lazy import — keeps retrieve.py importable without distillation deps.
    from consultant_brain.distillation.cold_layer import (
        find_matching_patterns,
        pattern_hit_as_atom_hit,
    )

    hits = find_matching_patterns(vault_root=vault_root, window=transcript_window, top_n=COLD_LAYER_MAX)
    out: list[RankedHit] = []
    for hit in hits:
        atom_hit = pattern_hit_as_atom_hit(hit)
        out.append(
            _rank_hit(
                atom_hit,
                layer="cold",
                today=today,
                reason=f"pattern fire: {hit.matched_phrase}",
            )
        )
    return out


# ────────────────────────────────────────────────────────────────────────────
# Ranking helpers
# ────────────────────────────────────────────────────────────────────────────


def _rank_hit(hit: AtomHit, *, layer: str, today: date, reason: str) -> RankedHit:
    similarity = hit.similarity
    recency = _recency_score(hit.last_seen, today=today)
    confidence = max(0.0, min(1.0, hit.confidence))
    score = (
        SIMILARITY_WEIGHT * similarity
        + RECENCY_WEIGHT * recency
        + CONFIDENCE_WEIGHT * confidence
    )
    return RankedHit(
        hit=hit,
        similarity=similarity,
        recency=recency,
        confidence=confidence,
        score=score,
        layer=layer,
        reason=reason,
    )


def _recency_score(last_seen_iso: str, *, today: date) -> float:
    """Exponential decay over RECENCY_HALF_LIFE_DAYS. Returns 1.0 for an atom
    seen today, ~0.5 at the half-life, near 0 for atoms several half-lives old.
    """
    if not last_seen_iso:
        return 0.0
    try:
        last_seen = date.fromisoformat(last_seen_iso)
    except ValueError:
        return 0.0
    days_ago = max(0, (today - last_seen).days)
    return 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
