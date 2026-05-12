"""Weekly review generator — populates 08_Reviews/.

`consultant-brain weekly [--week 2026-W19]` writes one markdown file
covering: clients touched, atoms minted by type, patterns promoted,
plays auto-promoted, scores trended, top moves recommended for next
week. Cron-friendly: no interactive prompts.

Same regeneration discipline as client_context.py: the auto-generated
sections rewrite every run, a pinned section the user can edit is
preserved verbatim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import frontmatter as fm
import yaml

from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom, AtomType, AtomStatus, CallNote
from consultant_brain.vault import (
    VaultLayout,
    read_frontmatter,
)


REVIEWS_DIR_NAME = "08_Reviews"
PINNED_SECTION_HEADER = "## Lessons + commitments for next week"


@dataclass(frozen=True, slots=True)
class WeeklyReviewResult:
    """Outcome of one weekly-review build."""

    path: Path
    week_id: str
    atom_count: int
    call_count: int
    client_count: int
    notes: list[str] = field(default_factory=list)


def generate_weekly_review(
    *,
    vault_root: Path,
    week_iso: str | None = None,
    now: datetime | None = None,
) -> WeeklyReviewResult:
    """Build the markdown for one ISO week and write it to 08_Reviews/.

    `week_iso` looks like `"2026-W19"`. When omitted, defaults to the
    current ISO week of `now` (or today). The function never raises on
    empty data — an empty week gets a one-line "no activity" note.
    """
    layout = VaultLayout.for_root(vault_root)
    reviews_dir = layout.root / REVIEWS_DIR_NAME
    reviews_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now or datetime.now(timezone.utc)
    today_d = timestamp.date()
    week_id = week_iso or _iso_week_id(today_d)
    week_start, week_end = _iso_week_bounds(week_id)

    atoms_this_week, calls_this_week = _collect_weekly(layout, week_start, week_end)

    path = reviews_dir / f"{week_id}.md"
    _write_review(
        path=path,
        week_id=week_id,
        week_start=week_start,
        week_end=week_end,
        atoms=atoms_this_week,
        calls=calls_this_week,
        now=timestamp,
    )

    return WeeklyReviewResult(
        path=path,
        week_id=week_id,
        atom_count=len(atoms_this_week),
        call_count=len(calls_this_week),
        client_count=len({a.client for a in atoms_this_week}),
    )


def _iso_week_id(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def _iso_week_bounds(week_id: str) -> tuple[date, date]:
    """Convert `"2026-W19"` to (Monday-date, Sunday-date)."""
    try:
        year_str, week_str = week_id.split("-W")
        year = int(year_str)
        week = int(week_str)
    except ValueError as exc:
        raise ValueError(f"Bad week id: {week_id!r}. Use 'YYYY-Www'.") from exc
    monday = date.fromisocalendar(year, week, 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _collect_weekly(
    layout: VaultLayout,
    week_start: date,
    week_end: date,
) -> tuple[list[Atom], list[dict]]:
    atoms: list[Atom] = []
    if layout.atoms_dir.exists():
        for path in layout.atoms_dir.glob("*.md"):
            try:
                atom = _atom_from_markdown(path)
            except Exception:
                continue
            if week_start <= atom.last_seen <= week_end:
                atoms.append(atom)

    calls: list[dict] = []
    if layout.calls_dir.exists():
        for path in layout.calls_dir.glob("*.md"):
            try:
                meta = read_frontmatter(path)
            except Exception:
                continue
            date_value = meta.get("date")
            if not isinstance(date_value, (date, str)):
                continue
            if isinstance(date_value, str):
                try:
                    date_value = date.fromisoformat(date_value)
                except ValueError:
                    continue
            if week_start <= date_value <= week_end:
                meta["_path"] = path
                calls.append(meta)
    return atoms, calls


def _write_review(
    *,
    path: Path,
    week_id: str,
    week_start: date,
    week_end: date,
    atoms: list[Atom],
    calls: list[dict],
    now: datetime,
) -> None:
    existing_pinned = ""
    existing_meta: dict = {}
    if path.exists():
        try:
            post = fm.load(path.open("r", encoding="utf-8"))
            existing_meta = dict(post.metadata)
            existing_pinned = _extract_pinned(post.content or "")
        except Exception:
            pass

    created_at_iso = existing_meta.get("created_at") or now.strftime("%Y-%m-%dT%H:%M:%SZ")

    by_type: Counter[str] = Counter(atom.type.value for atom in atoms)
    by_client: Counter[str] = Counter(atom.client for atom in atoms)
    top_clients = by_client.most_common(5)
    top_moves = _suggest_moves(atoms)

    frontmatter_data = {
        "id": week_id,
        "kind": "weekly_review",
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "atom_count": len(atoms),
        "call_count": len(calls),
        "client_count": len({a.client for a in atoms}),
        "created_at": created_at_iso,
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if not atoms and not calls:
        body = f"""# Week {week_id} ({week_start.isoformat()} → {week_end.isoformat()})

_No activity recorded this week. Either the brain wasn't running or no
calls were ingested. Run `consultant-brain ingest` against any session
JSONs you might have missed._
"""
    else:
        type_lines = [f"- **{t}** — {count}" for t, count in by_type.most_common()]
        client_lines = [f"- **{c}** — {count} atoms" for c, count in top_clients]
        call_lines = []
        for call in calls:
            call_id = str(call.get("id", call.get("_path", "")))
            client_name = call.get("client", "")
            summary = call.get("summary", "") or ""
            if len(summary) > 200:
                summary = summary[:197] + "..."
            call_lines.append(f"- [[{call_id}]] · {client_name} · {summary}")

        move_lines = [f"- {m}" for m in top_moves] or ["- (no commitments captured this week)"]

        body = f"""# Week {week_id} ({week_start.isoformat()} → {week_end.isoformat()})

## At a glance
- Calls: {len(calls)}
- Atoms: {len(atoms)}
- Clients touched: {len({a.client for a in atoms})}

## Atoms by type
{chr(10).join(type_lines) if type_lines else "- (none)"}

## Top clients
{chr(10).join(client_lines) if client_lines else "- (none)"}

## Calls
{chr(10).join(call_lines) if call_lines else "- (no calls)"}

## Suggested next moves
{chr(10).join(move_lines)}
"""

    pinned_block = ""
    if existing_pinned:
        pinned_block = f"\n\n{PINNED_SECTION_HEADER}\n{existing_pinned.rstrip()}\n"
    else:
        pinned_block = f"\n\n{PINNED_SECTION_HEADER}\n_(Add your own commitments + lessons here. Auto-generated sections above will re-render.)_\n"

    yaml_block = yaml.safe_dump(frontmatter_data, sort_keys=False, allow_unicode=True).rstrip()
    content = f"---\n{yaml_block}\n---\n\n{body.rstrip()}{pinned_block}"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    tmp.replace(path)


def _suggest_moves(atoms: list[Atom]) -> list[str]:
    """Top-3 actionable next moves derived from this week's atoms.

    Heuristic v1: surface every distinct unresolved commitment + every
    objection that didn't have a matching commitment in the same call.
    Sorted by atom confidence descending. Capped at 3 — the weekly
    review should be skimmable.
    """
    commitments_by_call: dict[str, list[Atom]] = defaultdict(list)
    objections_by_call: dict[str, list[Atom]] = defaultdict(list)
    for atom in atoms:
        if atom.type is AtomType.commitment:
            commitments_by_call[atom.call].append(atom)
        elif atom.type is AtomType.objection:
            objections_by_call[atom.call].append(atom)

    moves: list[tuple[float, str]] = []
    for call_id, commits in commitments_by_call.items():
        for atom in commits:
            moves.append((atom.confidence, f"Follow up on commitment ({atom.client}): {atom.body}"))
    for call_id, objections in objections_by_call.items():
        if call_id in commitments_by_call:
            continue
        for atom in objections:
            moves.append(
                (
                    atom.confidence,
                    f"Address unhandled objection ({atom.client}): {atom.body}",
                )
            )
    moves.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in moves[:3]]


def _extract_pinned(body: str) -> str:
    idx = body.find(PINNED_SECTION_HEADER)
    if idx < 0:
        return ""
    return body[idx + len(PINNED_SECTION_HEADER):].strip()
