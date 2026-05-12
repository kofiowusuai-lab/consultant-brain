"""Cold-layer retrieval: pattern + play matching during a live call.

The Phase 2 retrieve.py shipped Layer 3 as an empty stub. Phase 6
populated it for patterns. Phase 8 item 7 extends it: when the live
transcript window echoes the seed body of a promoted Play (05_Plays/),
surface that play alongside any matching patterns.

V1 uses pure text-match against pattern primary_tags + member bodies,
and against play seed bodies + member excerpts. Fast, deterministic,
no Ollama call required. A future revision can add semantic
embedding-based pattern retrieval if text-match misses too many fires.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from consultant_brain.embedder import AtomHit
from consultant_brain.vault import VaultLayout


# Max patterns to surface per retrieve() call. Master prompt says "1 cold"
# in the panel slice; we over-fetch slightly so the dedupe still leaves
# a meaningful top-1.
MAX_PATTERNS_PER_QUERY = 4


@dataclass(frozen=True, slots=True)
class PatternHit:
    """One pattern that fired against the transcript window."""

    pattern_id: str
    atom_type: str
    primary_tag: str
    observation_count: int
    score_impact: float
    matched_phrase: str
    summary: str  # short body line from the pattern note


def find_matching_patterns(*, vault_root: Path, window: str, top_n: int = MAX_PATTERNS_PER_QUERY) -> list[PatternHit]:
    """Walk 04_Patterns/, return the patterns whose primary_tag or example
    bodies appear (case-insensitive) in the rolling transcript window.

    Ranked by:
      1. number of distinct trigger words matched (more = stronger fire)
      2. observation_count (more-evidenced patterns rank higher)
      3. score_impact_proxy (higher confidence rank higher)
    """
    if not window.strip():
        return []
    layout = VaultLayout.for_root(vault_root)
    patterns_dir = layout.root / "04_Patterns"
    if not patterns_dir.exists():
        return []

    window_lower = window.lower()
    hits: list[tuple[int, PatternHit]] = []  # (match_strength, hit)

    for path in patterns_dir.glob("*.md"):
        try:
            post = frontmatter.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        meta = dict(post.metadata)
        if meta.get("status") != "active":
            continue

        primary_tag = str(meta.get("primary_tag", "") or "")
        atom_type = str(meta.get("type", "") or "")
        obs_count = int(meta.get("observation_count", 0))
        score_impact = float(meta.get("score_impact_proxy", 0.5))

        # Trigger candidates: the primary tag (split on underscores so
        # `next_step` matches "next step") + any whole-word noun in the
        # first three example bodies. Cheap heuristic; tune later.
        triggers = _trigger_terms(primary_tag=primary_tag, body=post.content or "")
        matched_words: list[str] = []
        for term in triggers:
            if _contains_whole_word(window_lower, term.lower()):
                matched_words.append(term)
        if not matched_words:
            continue

        summary = _first_example_line(post.content or "") or primary_tag
        hit = PatternHit(
            pattern_id=str(meta.get("id", path.stem)),
            atom_type=atom_type,
            primary_tag=primary_tag,
            observation_count=obs_count,
            score_impact=score_impact,
            matched_phrase=matched_words[0],
            summary=summary,
        )
        # Rank by (#matched terms, observation_count, score_impact)
        strength = len(matched_words) * 1000 + obs_count * 10 + int(score_impact * 10)
        hits.append((strength, hit))

    hits.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in hits[:top_n]]


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _trigger_terms(*, primary_tag: str, body: str) -> list[str]:
    """Words that, if present in the transcript window, indicate this
    pattern fires. The primary tag (split on `_`) is the main one; we
    also pull the first 5 lowercase nouns from the body for fuzzy fires.
    """
    terms: set[str] = set()
    for chunk in primary_tag.split("_"):
        chunk = chunk.strip()
        if len(chunk) >= 3:
            terms.add(chunk)
    # Body keywords — strip section headers + list markers, drop stopwords.
    stop = {"the", "this", "that", "with", "from", "what", "were", "your",
            "their", "these", "those", "into", "have", "been", "across",
            "above", "below", "patterns"}
    for word in re.findall(r"[A-Za-z]{4,}", body):
        word_l = word.lower()
        if word_l in stop:
            continue
        terms.add(word_l)
        if len(terms) >= 12:
            break
    return sorted(terms)


def _contains_whole_word(text: str, word: str) -> bool:
    """Whole-word match — avoids 'price' matching 'priceless'."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


_EXAMPLE_LINE_RE = re.compile(r"^-\s+(.+)$", re.MULTILINE)


def _first_example_line(body: str) -> str:
    """Pull the first '- ...' line from the Examples section so the
    suggestion has something concrete to render."""
    match = _EXAMPLE_LINE_RE.search(body)
    if not match:
        return ""
    return match.group(1).strip()


def pattern_hit_as_atom_hit(hit: PatternHit) -> AtomHit:
    """Adapter so the existing retrieval pipeline can carry pattern hits
    through the same RankedHit machinery. similarity≈score_impact so the
    pattern's score is interpretable in the same units."""
    return AtomHit(
        id=hit.pattern_id,
        type=hit.atom_type,
        body=hit.summary or hit.primary_tag,
        client=None,
        call="(pattern)",
        call_type="",
        confidence=hit.score_impact,
        last_seen="",
        tags=(hit.primary_tag,),
        # Convert our "match strength" back into a fake cosine-distance so
        # similarity ≈ score_impact. similarity = 1 - distance/2 →
        # distance = 2 * (1 - score_impact).
        distance=2.0 * (1.0 - max(0.0, min(1.0, hit.score_impact))),
    )


# ────────────────────────────────────────────────────────────────────────────
# Phase 8 item 7 — Play matching (05_Plays/)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PlayHit:
    """One promoted Play that fired against the transcript window."""

    play_id: str
    atom_type: str
    call_count: int
    client_count: int
    matched_phrase: str
    summary: str


def find_matching_plays(
    *,
    vault_root: Path,
    window: str,
    top_n: int = MAX_PATTERNS_PER_QUERY,
) -> list[PlayHit]:
    """Walk 05_Plays/, surface plays whose seed body or member excerpts
    appear (case-insensitive, whole-word) in the transcript window."""
    if not window.strip():
        return []
    layout = VaultLayout.for_root(vault_root)
    plays_dir = layout.root / "05_Plays"
    if not plays_dir.exists():
        return []

    window_lower = window.lower()
    hits: list[tuple[int, PlayHit]] = []
    for path in plays_dir.glob("*.md"):
        try:
            post = frontmatter.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        meta = dict(post.metadata)
        if meta.get("status") != "active":
            continue

        body = post.content or ""
        atom_type = str(meta.get("type", "") or "")
        call_count = int(meta.get("call_count", 0))
        client_count = int(meta.get("client_count", 0))

        terms = _play_trigger_terms(body)
        matched = [t for t in terms if _contains_whole_word(window_lower, t.lower())]
        if not matched:
            continue
        summary = _first_example_line(body) or atom_type or "play"
        hit = PlayHit(
            play_id=str(meta.get("id", path.stem)),
            atom_type=atom_type,
            call_count=call_count,
            client_count=client_count,
            matched_phrase=matched[0],
            summary=summary,
        )
        strength = len(matched) * 1000 + client_count * 100 + call_count * 10
        hits.append((strength, hit))

    hits.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in hits[:top_n]]


_PLAY_TRIGGER_STOPWORDS = {
    "play", "four", "frame", "template", "opener", "frame", "response",
    "handle", "examples", "atoms", "member", "fill", "_(fill", "in)_",
    "promoted", "from", "across", "calls", "clients", "the", "with",
    "this", "that", "from", "into",
}


def _play_trigger_terms(body: str) -> list[str]:
    """Pull words ≥4 chars from the example excerpts of a play; drop
    template/header chrome via a small stoplist."""
    terms: set[str] = set()
    in_examples = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Observed examples"):
            in_examples = True
            continue
        if stripped.startswith("##"):
            in_examples = False
            continue
        if not in_examples:
            continue
        for word in re.findall(r"[A-Za-z]{4,}", stripped):
            word_l = word.lower()
            if word_l in _PLAY_TRIGGER_STOPWORDS:
                continue
            terms.add(word_l)
            if len(terms) >= 12:
                break
        if len(terms) >= 12:
            break
    return sorted(terms)


def play_hit_as_atom_hit(hit: PlayHit) -> AtomHit:
    """Adapter so the retrieval pipeline can carry play hits through the
    same RankedHit machinery as patterns + atoms."""
    # Plays are higher-leverage than patterns by construction (need ≥3
    # distinct clients to promote), so we bias their confidence proxy up.
    confidence = min(1.0, 0.6 + 0.05 * hit.client_count)
    return AtomHit(
        id=hit.play_id,
        type=hit.atom_type,
        body=hit.summary,
        client=None,
        call="(play)",
        call_type="",
        confidence=confidence,
        last_seen="",
        tags=("play",),
        distance=2.0 * (1.0 - confidence),
    )
