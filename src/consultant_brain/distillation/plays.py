"""Plays auto-promotion.

Patterns are abstractions; Plays are reusable lines. When the same
verbatim-ish atom body (or the same opening turn / framing / closing
move) shows up across calls with different clients, it's earned the
right to be remembered as a play — not as a sentence to copy, but as a
proven move to lean on.

Promotion rule (v1):
  - same atom `type` (commitment / objection / win_signal)
  - body shingles overlap (Jaccard ≥0.4) across ≥3 calls in ≥3 DIFFERENT
    clients (so the same play repeated in one client's renewals doesn't
    fool the promoter)

A Play has 4 frames: opener / frame / response / handle. V1 surfaces
the play with body excerpts; the consultant fills in the four-frame
template by hand during the weekly review. The auto-promoter creates
the skeleton.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, AtomStatus, AtomType
from consultant_brain.vault import VaultLayout


PLAYS_DIR_NAME = "05_Plays"
MIN_CALLS_FOR_PLAY = 3
MIN_CLIENTS_FOR_PLAY = 3
SHINGLE_K = 3  # 3-word shingles for Jaccard similarity
SHINGLE_SIMILARITY_FLOOR = 0.4


_PLAY_PROMOTABLE_TYPES: tuple[AtomType, ...] = (
    AtomType.commitment,
    AtomType.objection,
    AtomType.win_signal,
)


@dataclass(frozen=True, slots=True)
class PlayCandidate:
    """One cluster of similar atoms eligible for play promotion."""

    atom_type: AtomType
    seed_body: str
    member_atoms: tuple[Atom, ...]
    distinct_clients: tuple[str, ...]

    def play_id(self) -> str:
        slug = _slugify(self.seed_body)[:48] or "play"
        return f"{self.atom_type.value}__{slug}"

    def call_count(self) -> int:
        return len({a.call for a in self.member_atoms})


@dataclass(frozen=True, slots=True)
class PlaysMiningResult:
    """Outcome of one mining pass."""

    plays_written: int
    plays_updated: int
    plays_skipped: int
    play_ids: list[str] = field(default_factory=list)


def mine_plays(*, vault_root: Path, now: datetime | None = None) -> PlaysMiningResult:
    """Walk active atoms, find cross-client recurring bodies, write
    05_Plays/<type>__<seed>.md files."""
    layout = VaultLayout.for_root(vault_root)
    if not layout.atoms_dir.exists():
        return PlaysMiningResult(0, 0, 0)

    candidates = _cluster_candidates(layout)
    eligible = [
        c
        for c in candidates
        if c.call_count() >= MIN_CALLS_FOR_PLAY
        and len(c.distinct_clients) >= MIN_CLIENTS_FOR_PLAY
    ]

    plays_dir = layout.root / PLAYS_DIR_NAME
    plays_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now or datetime.now(timezone.utc)

    written = 0
    updated = 0
    play_ids: list[str] = []
    for candidate in eligible:
        path = plays_dir / f"{candidate.play_id()}.md"
        existed = path.exists()
        _write_play_note(path=path, candidate=candidate, now=timestamp, existed=existed)
        play_ids.append(candidate.play_id())
        if existed:
            updated += 1
        else:
            written += 1

    return PlaysMiningResult(
        plays_written=written,
        plays_updated=updated,
        plays_skipped=len(candidates) - len(eligible),
        play_ids=play_ids,
    )


def _cluster_candidates(layout: VaultLayout) -> list[PlayCandidate]:
    """Group atoms by type, then cluster by body similarity. Greedy
    seed-based clustering: each atom either joins the first existing
    cluster within shingle-similarity floor, or starts a new one."""
    by_type: dict[AtomType, list[Atom]] = defaultdict(list)
    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        if atom.status is not AtomStatus.active:
            continue
        if atom.type not in _PLAY_PROMOTABLE_TYPES:
            continue
        by_type[atom.type].append(atom)

    candidates: list[PlayCandidate] = []
    for atom_type, atoms in by_type.items():
        clusters: list[list[Atom]] = []
        cluster_shingles: list[frozenset[str]] = []
        for atom in atoms:
            shingles = _shingles(atom.body, SHINGLE_K)
            if not shingles:
                continue
            joined = False
            for index, existing in enumerate(cluster_shingles):
                if _jaccard(shingles, existing) >= SHINGLE_SIMILARITY_FLOOR:
                    clusters[index].append(atom)
                    cluster_shingles[index] = existing | shingles
                    joined = True
                    break
            if not joined:
                clusters.append([atom])
                cluster_shingles.append(shingles)

        for cluster in clusters:
            distinct_clients = tuple(sorted({a.client for a in cluster}))
            candidates.append(
                PlayCandidate(
                    atom_type=atom_type,
                    seed_body=cluster[0].body,
                    member_atoms=tuple(cluster),
                    distinct_clients=distinct_clients,
                )
            )
    return candidates


def _shingles(text: str, k: int) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < k:
        return frozenset()
    return frozenset(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _write_play_note(
    *,
    path: Path,
    candidate: PlayCandidate,
    now: datetime,
    existed: bool,
) -> None:
    """Write/update one play .md file. The four-frame template is the
    skeleton; the consultant fills in opener/frame/response/handle by
    hand during the weekly review."""
    existing_meta: dict = {}
    if existed:
        try:
            import frontmatter as fm

            existing_meta = dict(fm.load(path.open("r", encoding="utf-8")).metadata)
        except Exception:
            existing_meta = {}

    created_at_iso = existing_meta.get("created_at") or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    member_ids = sorted([a.id for a in candidate.member_atoms])

    frontmatter_data = {
        "id": candidate.play_id(),
        "kind": "play",
        "type": candidate.atom_type.value,
        "call_count": candidate.call_count(),
        "client_count": len(candidate.distinct_clients),
        "clients": list(candidate.distinct_clients),
        "status": "active",
        "member_atom_ids": member_ids,
        "created_at": created_at_iso,
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    examples = [f"- {a.body}" for a in candidate.member_atoms[:3]]
    member_links = [f"- [[{atom_id}]]" for atom_id in member_ids]
    body = f"""# Play: {candidate.atom_type.value}

Promoted from {len(candidate.member_atoms)} atoms across
{candidate.call_count()} calls in {len(candidate.distinct_clients)}
clients ({", ".join(candidate.distinct_clients)}).

## Four-frame template

- **Opener** — _(fill in)_
- **Frame** — _(fill in)_
- **Response** — _(fill in)_
- **Handle** — _(fill in)_

## Observed examples
{chr(10).join(examples)}

## Member atoms
{chr(10).join(member_links)}
"""
    # Preserve any hand-edited four-frame body across re-promotion. If
    # the file already has filled-in frames, keep them; otherwise write
    # the template.
    if existed:
        try:
            import frontmatter as fm

            prior = fm.load(path.open("r", encoding="utf-8")).content or ""
            if "_(fill in)_" not in prior and "## Four-frame template" in prior:
                body = prior
        except Exception:
            pass

    yaml_block = yaml.safe_dump(frontmatter_data, sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
