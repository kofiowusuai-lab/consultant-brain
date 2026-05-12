"""Pattern miner — promotes recurring atoms to reusable patterns.

When an atom of a given (type, primary_tag) shows up ≥3× across calls,
it's no longer an isolated observation — it's a pattern. Phase 6
distillation walks the vault, finds these clusters, and writes a
Pattern note to `04_Patterns/` that the cold retrieval layer can use
during future live calls.

A Pattern is NOT itself an atom — it's a higher-level abstraction:
  - Lives in 04_Patterns/
  - Has its own frontmatter (observation_count, score_impact_proxy,
    member_atom_ids, status)
  - References the atoms that contributed to it

Promotion rule (v1):
  same atom `type` × same `primary_tag` (first tag) observed ≥ MIN_OBSERVATIONS
  across DIFFERENT calls → promote.

Why "different calls": three atoms from the same call describing the
same objection isn't a pattern, it's redundancy. The promotion
requires evidence across multiple sessions.
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


MIN_OBSERVATIONS = 3  # master prompt's threshold


@dataclass(frozen=True, slots=True)
class PatternCandidate:
    """One (type, primary_tag) bucket that's eligible for promotion."""

    atom_type: AtomType
    primary_tag: str
    member_atoms: tuple[Atom, ...]

    @property
    def observation_count(self) -> int:
        return len(self.member_atoms)

    @property
    def unique_call_count(self) -> int:
        return len({a.call for a in self.member_atoms})

    def pattern_id(self) -> str:
        """Stable, human-readable ID. Filename-safe; matches `[a-z0-9_]+`."""
        return f"{self.atom_type.value}__{_slugify_tag(self.primary_tag)}"


@dataclass(frozen=True, slots=True)
class PatternMiningResult:
    """Outcome of one mining pass."""

    patterns_written: int
    patterns_updated: int
    candidates_skipped: int
    pattern_ids: list[str] = field(default_factory=list)


def mine_patterns(*, vault_root: Path, now: datetime | None = None) -> PatternMiningResult:
    """Walk 03_Atoms/, identify pattern candidates, write/update them in
    04_Patterns/. Idempotent: re-running updates `observation_count` and
    `member_atom_ids` rather than creating duplicate files.
    """
    layout = VaultLayout.for_root(vault_root)
    if not layout.atoms_dir.exists():
        return PatternMiningResult(0, 0, 0)

    timestamp = now or datetime.now(timezone.utc)
    candidates = _collect_candidates(layout)

    eligible = [c for c in candidates if c.unique_call_count >= MIN_OBSERVATIONS]
    skipped = len(candidates) - len(eligible)

    patterns_dir = layout.root / "04_Patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    updated = 0
    pattern_ids: list[str] = []
    for candidate in eligible:
        path = patterns_dir / f"{candidate.pattern_id()}.md"
        existed = path.exists()
        _write_pattern_note(
            path=path,
            candidate=candidate,
            now=timestamp,
            previously_existed=existed,
        )
        pattern_ids.append(candidate.pattern_id())
        if existed:
            updated += 1
        else:
            written += 1

    return PatternMiningResult(
        patterns_written=written,
        patterns_updated=updated,
        candidates_skipped=skipped,
        pattern_ids=pattern_ids,
    )


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _collect_candidates(layout: VaultLayout) -> list[PatternCandidate]:
    """Bucket active atoms by (type, primary_tag) — atoms with no tags get
    a synthetic `__untagged__` bucket so they can still cluster.
    """
    buckets: dict[tuple[AtomType, str], list[Atom]] = defaultdict(list)
    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        if atom.status is not AtomStatus.active:
            continue
        primary_tag = atom.tags[0] if atom.tags else "__untagged__"
        buckets[(atom.type, primary_tag)].append(atom)

    return [
        PatternCandidate(atom_type=t, primary_tag=tag, member_atoms=tuple(atoms))
        for (t, tag), atoms in buckets.items()
    ]


_TAG_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _slugify_tag(tag: str) -> str:
    """Tags are already lower-snake by convention; this just defends
    against the rare special-character tag the LLM emits."""
    return _TAG_SLUG_RE.sub("_", tag.lower()).strip("_") or "untagged"


def _write_pattern_note(
    *,
    path: Path,
    candidate: PatternCandidate,
    now: datetime,
    previously_existed: bool,
) -> None:
    """Write/update one pattern .md file. Updates preserve the original
    `created_at` so the pattern's history isn't lost."""
    # Preserve created_at across re-runs.
    existing_meta = _read_existing_pattern_meta(path) if previously_existed else {}
    created_at_iso = existing_meta.get("created_at") or now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Quick proxy: average confidence of the members. Phase 5+ retrain can
    # eventually use a learned score_impact; this is fine for v1 ranking.
    avg_confidence = sum(a.confidence for a in candidate.member_atoms) / candidate.observation_count

    member_ids = sorted([a.id for a in candidate.member_atoms])

    frontmatter = {
        "id": candidate.pattern_id(),
        "kind": "pattern",
        "type": candidate.atom_type.value,
        "primary_tag": candidate.primary_tag,
        "observation_count": candidate.observation_count,
        "unique_call_count": candidate.unique_call_count,
        "score_impact_proxy": round(avg_confidence, 3),
        "status": "active",
        "member_atom_ids": member_ids,
        "created_at": created_at_iso,
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # The body is a brief auto-summary: list the first 3 member bodies so a
    # human reading the pattern note in Obsidian sees concrete examples,
    # then the wikilinks to every member atom.
    example_lines = [f"- {a.body}" for a in candidate.member_atoms[:3]]
    member_links = [f"- [[{atom_id}]]" for atom_id in member_ids]

    body = f"""# Pattern: {candidate.atom_type.value} · {candidate.primary_tag}

This pattern fires when the live transcript echoes any of the observed
moments below. Promoted from {candidate.observation_count} atoms across
{candidate.unique_call_count} distinct calls.

## Examples
{chr(10).join(example_lines)}

## Member atoms
{chr(10).join(member_links)}
"""

    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _read_existing_pattern_meta(path: Path) -> dict:
    """Best-effort parse of an existing pattern's frontmatter so we can
    preserve fields like created_at on re-mine."""
    try:
        import frontmatter as fm
        post = fm.load(path.open("r", encoding="utf-8"))
        return dict(post.metadata)
    except Exception:
        return {}
