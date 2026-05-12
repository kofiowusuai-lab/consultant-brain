"""Tag normalization — fold near-duplicate tags onto a canonical form.

The atom extractor is consistent within one call but drifts across
sessions: `next_step` / `nextstep` / `next-step` / `followup` / `follow_up`
all describe the same concept. Left alone they fragment retrieval (a
pattern keyed on `next_step` misses every `nextstep` atom) and pattern
mining (each spelling forms its own under-evidenced bucket).

Strategy:
  1. Pull every distinct tag across all atoms, count occurrences.
  2. Build a canonical map: the most-frequent spelling wins, plus any
     hand-curated aliases (the master prompt's "next_step is one tag,
     not three" rule lives here).
  3. Group near-duplicates by Levenshtein distance (radius 2 on tags ≥4
     chars long, exact match only on shorter tags so we don't merge
     `vip` into `via`).
  4. Rewrite every affected atom's frontmatter in place via the
     atomic-write path vault.py already uses.

Idempotent: running twice produces no further changes once everything
is canonical. Triggered by `distill`, not its own CLI subcommand.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import frontmatter as fm
import yaml

from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom
from consultant_brain.vault import VaultLayout


# Hand-curated aliases that frequency alone can't catch — e.g. when both
# variants are equally common in the corpus, we pin the canonical form
# rather than letting the tiebreaker pick randomly.
CANONICAL_ALIASES: dict[str, str] = {
    "nextstep": "next_step",
    "next-step": "next_step",
    "followup": "next_step",
    "follow_up": "next_step",
    "follow-up": "next_step",
    "todo": "next_step",
    "action_item": "next_step",
    "actionitem": "next_step",
    "team-size": "team_size",
    "teamsize": "team_size",
    "headcount": "team_size",
    "budget_range": "budget",
    "budgeting": "budget",
    "pricing": "budget",
    "price": "budget",
    "scope_creep": "scope",
    "scope-creep": "scope",
    "decisionmaker": "decision_maker",
    "decision-maker": "decision_maker",
}


@dataclass(frozen=True, slots=True)
class TagNormalizationResult:
    """Outcome of one normalization pass. Surfaced by the CLI."""

    atoms_scanned: int
    atoms_rewritten: int
    tag_replacements: dict[str, str] = field(default_factory=dict)


def normalize_tags(*, vault_root: Path) -> TagNormalizationResult:
    """Walk 03_Atoms/, fold every non-canonical tag onto its canonical
    form, rewrite the markdown in place. Returns a count + the mapping
    actually applied so the CLI can print what changed.
    """
    layout = VaultLayout.for_root(vault_root)
    if not layout.atoms_dir.exists():
        return TagNormalizationResult(0, 0, {})

    atom_paths = list(layout.atoms_dir.glob("*.md"))
    atoms_by_path: dict[Path, Atom] = {}
    for path in atom_paths:
        try:
            atoms_by_path[path] = _atom_from_markdown(path)
        except Exception:
            continue

    canonical_map = _build_canonical_map(atoms_by_path.values())

    rewritten = 0
    applied: dict[str, str] = {}
    for path, atom in atoms_by_path.items():
        new_tags = _apply_map(atom.tags, canonical_map)
        if new_tags == list(atom.tags):
            continue
        _rewrite_tags(path, new_tags)
        for old, new in zip(atom.tags, new_tags):
            if old != new:
                applied[old] = new
        rewritten += 1

    return TagNormalizationResult(
        atoms_scanned=len(atoms_by_path),
        atoms_rewritten=rewritten,
        tag_replacements=applied,
    )


def _build_canonical_map(atoms: Iterable[Atom]) -> dict[str, str]:
    """Tag-frequency-driven canonicalization. The most-common spelling
    in the corpus wins; aliases override frequency."""
    counter: Counter[str] = Counter()
    for atom in atoms:
        for tag in atom.tags:
            counter[tag] += 1

    mapping: dict[str, str] = dict(CANONICAL_ALIASES)

    # Levenshtein-radius merge across remaining tags. We iterate in
    # frequency order so the most-common spelling becomes the canonical
    # for its cluster.
    remaining = [tag for tag, _ in counter.most_common() if tag not in mapping]
    for index, tag in enumerate(remaining):
        if tag in mapping:
            continue
        for other in remaining[index + 1:]:
            if other in mapping:
                continue
            if _close_enough(tag, other):
                mapping[other] = mapping.get(tag, tag)

    # Apply alias chains so `nextstep` -> `next_step` resolves directly,
    # not via two hops.
    for src in list(mapping.keys()):
        seen = {src}
        target = mapping[src]
        while target in mapping and mapping[target] != target and target not in seen:
            seen.add(target)
            target = mapping[target]
        mapping[src] = target
    return mapping


def _apply_map(tags: list[str] | tuple[str, ...], canonical: dict[str, str]) -> list[str]:
    """Apply canonical map, preserve original order, drop duplicates that
    arose because two tags collapsed onto the same canonical form."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        new = canonical.get(tag, tag)
        if new in seen:
            continue
        seen.add(new)
        out.append(new)
    return out


def _rewrite_tags(path: Path, new_tags: list[str]) -> None:
    """Rewrite one atom's tags in place. Uses the atomic write pattern
    (write to .tmp, rename) so a crash mid-write leaves the original
    file intact.
    """
    try:
        post = fm.load(path.open("r", encoding="utf-8"))
    except Exception:
        return
    post["tags"] = new_tags
    yaml_block = yaml.safe_dump(dict(post.metadata), sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{(post.content or '').rstrip()}\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _close_enough(a: str, b: str) -> bool:
    """Levenshtein radius 2 for tags ≥4 chars long, exact match otherwise.

    No external dep — small DP table. We bail out fast on length deltas
    bigger than the radius and on tags that share no prefix character.
    """
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    if abs(len(a) - len(b)) > 2:
        return False
    if a[0] != b[0]:
        # First-char anchor catches the common typos but avoids merging
        # genuinely-different concepts that happen to be near in edit
        # distance ("scope" vs "scape" stays separate; "scope" vs
        # "scopes" merges).
        return False
    return _levenshtein(a, b) <= 2


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            ins = current[j - 1] + 1
            dele = previous[j] + 1
            sub = previous[j - 1] + (ca != cb)
            current.append(min(ins, dele, sub))
        previous = current
    return previous[-1]
