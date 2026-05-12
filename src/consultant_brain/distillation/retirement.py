"""Atom lifecycle management.

The vault accumulates atoms forever. Most stay relevant for months
(client_facts that anchor retrieval), but call-specific atoms
(commitments, objections from a one-off discovery) become stale fast.
Retention transitions atoms through three states:

  active        — written by the extractor, used by retrieval + patterns.
  needs_review  — not re-observed within RETENTION_DAYS (default 120).
                  Retrieval skips it; surfaced in the weekly review so the
                  consultant can manually re-confirm or let it die.
  retired       — needs_review for another 30 days without intervention.
                  Skipped by retrieval + pattern mining; kept on disk for
                  audit (the master prompt's vault-as-system-of-record).

The "re-observed" signal: when a new atom matches an existing atom by
(client, type, primary_tag) AND embedding similarity ≥0.85, the older
atom's `last_seen` is bumped to today. This logic lives on the ingest
path; the retirement pass here just looks at `last_seen` vs today.

Triggered by `distill`, not its own CLI command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import frontmatter as fm
import yaml

from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, AtomStatus
from consultant_brain.vault import VaultLayout


DEFAULT_RETENTION_DAYS = 120  # active → needs_review
DEFAULT_REVIEW_GRACE_DAYS = 30  # needs_review → retired


@dataclass(frozen=True, slots=True)
class RetirementResult:
    """Outcome of one retirement pass."""

    atoms_scanned: int
    flagged_needs_review: int
    retired: int
    revived: int  # needs_review → active because last_seen got bumped
    transitions: dict[str, str] = field(default_factory=dict)  # atom_id → new status


def run_retirement(
    *,
    vault_root: Path,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    grace_days: int = DEFAULT_REVIEW_GRACE_DAYS,
    today: date | None = None,
) -> RetirementResult:
    """Walk active + needs_review atoms, transition by age."""
    layout = VaultLayout.for_root(vault_root)
    if not layout.atoms_dir.exists():
        return RetirementResult(0, 0, 0, 0)

    today_d = today or datetime.now(timezone.utc).date()
    cutoff_review = today_d - timedelta(days=retention_days)
    cutoff_retire = today_d - timedelta(days=retention_days + grace_days)

    scanned = 0
    flagged = 0
    retired = 0
    revived = 0
    transitions: dict[str, str] = {}

    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        scanned += 1

        new_status = _evaluate(
            atom=atom,
            cutoff_review=cutoff_review,
            cutoff_retire=cutoff_retire,
        )
        if new_status is None or new_status is atom.status:
            continue

        _rewrite_status(path, new_status)
        transitions[atom.id] = new_status.value
        if new_status is AtomStatus.needs_review:
            flagged += 1
        elif new_status is AtomStatus.retired:
            retired += 1
        elif new_status is AtomStatus.active:
            revived += 1

    return RetirementResult(
        atoms_scanned=scanned,
        flagged_needs_review=flagged,
        retired=retired,
        revived=revived,
        transitions=transitions,
    )


def _evaluate(
    *,
    atom: Atom,
    cutoff_review: date,
    cutoff_retire: date,
) -> AtomStatus | None:
    """Return the new status if the atom should transition, else None."""
    if atom.status is AtomStatus.retired:
        return None  # retired is terminal
    last_seen = atom.last_seen

    if atom.status is AtomStatus.needs_review:
        if last_seen >= cutoff_review:
            # Got re-observed during the review window — revive.
            return AtomStatus.active
        if last_seen < cutoff_retire:
            return AtomStatus.retired
        return None

    # status is active
    if last_seen < cutoff_review:
        return AtomStatus.needs_review
    return None


def _rewrite_status(path: Path, new_status: AtomStatus) -> None:
    """Atomically rewrite one atom's `status` frontmatter field."""
    try:
        post = fm.load(path.open("r", encoding="utf-8"))
    except Exception:
        return
    post["status"] = new_status.value
    yaml_block = yaml.safe_dump(dict(post.metadata), sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{(post.content or '').rstrip()}\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
