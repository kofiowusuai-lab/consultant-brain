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
    render them with distinct styling (Hot = chip, Warm = card, Cold = play,
    Knowledge = library scroll, Phase 9).
    """

    hot: list[RankedHit] = field(default_factory=list)
    warm: list[RankedHit] = field(default_factory=list)
    cold: list[RankedHit] = field(default_factory=list)
    knowledge: list[RankedHit] = field(default_factory=list)

    @property
    def all(self) -> list[RankedHit]:
        return [*self.hot, *self.warm, *self.cold, *self.knowledge]

    def top_for_panel(
        self,
        *,
        hot: int = 1,
        warm: int = 2,
        cold: int = 1,
        knowledge: int = 0,
    ) -> list[RankedHit]:
        """The exact slice the live suggestion panel renders. `knowledge`
        defaults to 0 — prep view passes a positive number to opt in."""
        return [
            *self.hot[:hot],
            *self.warm[:warm],
            *self.cold[:cold],
            *self.knowledge[:knowledge],
        ]


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
    include_knowledge: bool = False,
    client_org_id: str | None = None,
) -> RetrievalResult:
    """The function the CLI + future FastAPI service both call.

    Doesn't open the LanceDB index if the vault has no `00_System/` dir —
    returns an empty result so the live copilot stays alive on a fresh
    machine where no ingests have happened yet.

    Phase 9: `include_knowledge=False` (the default for live calls)
    keeps externally-sourced atoms out of Hot + Warm so Memory
    surfaces only the consultant's own call history. Prep mode flips
    it to True; the knowledge layer surfaces beside warm.

    Phase 11: when `client_org_id` is set, the hot layer prefers a UUID
    match (survives display-name changes in the CRM) and falls back to
    the name filter for pre-Phase-11 atoms that have no UUID yet.
    """
    layout = VaultLayout.for_root(vault_root)
    if not layout.system_dir.exists():
        return RetrievalResult()

    index = LanceVaultIndex(layout)
    if index.atom_count() == 0:
        return RetrievalResult()

    today = now or datetime.now(timezone.utc).date()

    hot = _hot_layer(
        index=index,
        client=client,
        client_org_id=client_org_id,
        transcript_window=transcript_window,
        today=today,
        include_knowledge=include_knowledge,
    )
    hot_ids = {hit.hit.id for hit in hot}

    warm = _warm_layer(
        index=index,
        transcript_window=transcript_window,
        call_type=call_type,
        excluded_ids=hot_ids,
        today=today,
        include_knowledge=include_knowledge,
    )

    cold = _cold_layer(vault_root=vault_root, transcript_window=transcript_window, today=today)

    knowledge: list[RankedHit] = []
    if include_knowledge:
        excluded = hot_ids | {hit.hit.id for hit in warm}
        knowledge = _knowledge_layer(
            index=index,
            transcript_window=transcript_window,
            client=client,
            excluded_ids=excluded,
            today=today,
        )

    return RetrievalResult(hot=hot, warm=warm, cold=cold, knowledge=knowledge)


# ────────────────────────────────────────────────────────────────────────────
# Layer 1 — Hot
# ────────────────────────────────────────────────────────────────────────────


def _hot_layer(
    *,
    index: LanceVaultIndex,
    client: str | None,
    transcript_window: str,
    today: date,
    include_knowledge: bool = False,
    client_org_id: str | None = None,
) -> list[RankedHit]:
    """Atoms already on file about this client, ranked by semantic relevance
    to the current transcript window so the panel surfaces the most-likely-
    useful one first. If no client is given, hot is empty — those atoms only
    matter when we know who's on the call.

    Phase 9: hot defaults to `source_kind=call` only — external knowledge
    tagged for this client lives in the Knowledge layer, not Memory.

    Phase 11: when a `client_org_id` is supplied we run a UUID-pinned query
    first so display-name renames don't orphan an atom from its client. The
    name query then back-fills the pre-Phase-11 tail (atoms whose rows have
    an empty `client_org_id` column). Hits are deduped by atom id and the
    top HOT_LAYER_MAX wins.
    """
    if not client and not client_org_id:
        return []
    if not transcript_window.strip():
        return []
    source_filter = None if include_knowledge else ("call",)

    primary: list[AtomHit] = []
    if client_org_id:
        primary = index.query(
            text=transcript_window,
            client_org_id_filter=client_org_id,
            source_kind_filter=source_filter,
            top_n=HOT_LAYER_MAX,
        )

    fallback: list[AtomHit] = []
    if client and len(primary) < HOT_LAYER_MAX:
        # Cover the legacy tail: atoms minted before client_org_id was
        # populated will only match on display name. Exclude the UUID
        # winners so we don't rank the same atom twice.
        seen_ids = {hit.id for hit in primary}
        fallback = index.query(
            text=transcript_window,
            client_filter=client,
            source_kind_filter=source_filter,
            exclude_atom_ids=seen_ids,
            top_n=HOT_LAYER_MAX - len(primary),
        )

    raw_hits = [*primary, *fallback][:HOT_LAYER_MAX]
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
    include_knowledge: bool = False,
) -> list[RankedHit]:
    """Vector search across all atoms (this client + others), filtered by
    the current call_type so we don't surface a cold-call opener during a
    closing call. Hot-layer hits are excluded so we don't double-count.

    After raw vector ranking we re-rank with a weighted blend of similarity
    + recency + confidence, then dedupe by (type, primary_tag) bucket so the
    panel doesn't show three "objection: budget" atoms in a row.

    Phase 9: warm stays `source_kind=call` only by default. Prep flow flips
    `include_knowledge=True` to let the same window pull in external
    insights alongside the consultant's call history.
    """
    if not transcript_window.strip():
        return []
    source_filter = None if include_knowledge else ("call",)
    # Pull twice the cap so dedupe still leaves us with WARM_LAYER_MAX atoms.
    raw_hits = index.query(
        text=transcript_window,
        call_type_filter=call_type.value,
        source_kind_filter=source_filter,
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
# Phase 9 — Knowledge layer (external sources)
# ────────────────────────────────────────────────────────────────────────────


# Max atoms emitted by the knowledge layer in one retrieval call.
KNOWLEDGE_LAYER_MAX = 6
# `source_kind` values that count as external knowledge (everything
# except `call`). Tuple form because the embedder accepts tuples.
KNOWLEDGE_SOURCE_KINDS: tuple[str, ...] = ("youtube", "instagram", "article", "podcast")


def _knowledge_layer(
    *,
    index: LanceVaultIndex,
    transcript_window: str,
    client: str | None,
    excluded_ids: set[str],
    today: date,
) -> list[RankedHit]:
    """Atoms minted from external sources (videos, articles, podcasts).

    Only fires when the caller opts in (`include_knowledge=True`) —
    prep view does, the live overlay does not by default. Atoms tagged
    with `for-client` on `learn` get a confidence boost when the
    current client matches, so "studied with Reece in mind" surfaces
    for Reece's calls first.
    """
    if not transcript_window.strip():
        return []
    raw_hits = index.query(
        text=transcript_window,
        source_kind_filter=KNOWLEDGE_SOURCE_KINDS,
        exclude_atom_ids=excluded_ids,
        top_n=KNOWLEDGE_LAYER_MAX * 2,
    )
    out: list[RankedHit] = []
    for hit in raw_hits:
        reason_bits = [f"{_kind_label(hit)} knowledge"]
        if client and hit.client == client:
            reason_bits.append(f"tagged for {client}")
        out.append(
            _rank_hit(
                hit,
                layer="knowledge",
                today=today,
                reason=" · ".join(reason_bits),
            )
        )
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:KNOWLEDGE_LAYER_MAX]


def _kind_label(hit) -> str:
    """Best-effort source-kind label for the reason string. The
    embedder's AtomHit doesn't include source_kind today (we'd need to
    extend the schema) — fall back to the call-id prefix which is
    deterministic from `learn`."""
    call = (hit.call or "").lower()
    if call.startswith("youtube_"):
        return "YouTube"
    if call.startswith("instagram_"):
        return "Instagram"
    if call.startswith("podcast_"):
        return "Podcast"
    if call.startswith("article_"):
        return "Article"
    return "External"


# ────────────────────────────────────────────────────────────────────────────
# Layer 3 — Cold
# ────────────────────────────────────────────────────────────────────────────


def _cold_layer(*, vault_root: Path, transcript_window: str, today: date) -> list[RankedHit]:
    """Pattern + play matching for the cold layer.

    Patterns (`04_Patterns/`) fire when their primary_tag or example
    phrases echo in the window. Plays (`05_Plays/`) fire when the
    transcript echoes any of the promoted play's seed body words.
    Plays rank above patterns because they require more evidence
    (≥3 distinct clients vs ≥3 distinct calls) and so represent a more
    proven move.
    """
    if not transcript_window.strip():
        return []
    # Lazy import — keeps retrieve.py importable without distillation deps.
    from consultant_brain.distillation.cold_layer import (
        find_matching_patterns,
        find_matching_plays,
        pattern_hit_as_atom_hit,
        play_hit_as_atom_hit,
    )

    out: list[RankedHit] = []

    # Plays first — higher-leverage by construction.
    play_hits = find_matching_plays(vault_root=vault_root, window=transcript_window, top_n=COLD_LAYER_MAX)
    for hit in play_hits:
        atom_hit = play_hit_as_atom_hit(hit)
        out.append(
            _rank_hit(
                atom_hit,
                layer="cold",
                today=today,
                reason=f"play fire: {hit.matched_phrase}",
            )
        )

    pattern_hits = find_matching_patterns(vault_root=vault_root, window=transcript_window, top_n=COLD_LAYER_MAX)
    for hit in pattern_hits:
        atom_hit = pattern_hit_as_atom_hit(hit)
        out.append(
            _rank_hit(
                atom_hit,
                layer="cold",
                today=today,
                reason=f"pattern fire: {hit.matched_phrase}",
            )
        )
    return out[:COLD_LAYER_MAX]


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
