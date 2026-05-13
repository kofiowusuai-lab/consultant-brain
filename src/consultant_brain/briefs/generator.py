"""Generate per-client AI briefs.

`generate_client_brief()` collects every atom, call summary, and
context-dump note tied to one client, sends them to the configured
LLM provider with a brief-shaped prompt, and returns a structured
`ClientBrief` dataclass the FastAPI service serializes for the Swift
dashboard.

Cached at `01_Clients/<slug>/brief.json` so opening a client's profile
in the dashboard renders the last brief instantly; the user explicitly
refreshes (or the service decides the cache is stale) to regenerate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter as fm

from consultant_brain.extractor import (
    AnthropicClient,
    ExtractorError,
    real_anthropic_client,
)
from consultant_brain.llm.provider import (
    ChatRequest,
    LLMProvider,
    ProviderError,
)
from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import (
    Atom,
    AtomStatus,
    DEFAULT_EXTRACTOR_MODEL,
)
from consultant_brain.secrets import SecretNotFoundError, get_anthropic_key
from consultant_brain.vault import VaultLayout, slugify_client


# ────────────────────────────────────────────────────────────────────────────
# Public shapes
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClientBriefCommitment:
    by_whom: str       # "you" | "them"
    what: str
    since_call: Optional[str] = None  # call_id (e.g. "2026-05-12_reece_consultingCall")


@dataclass(frozen=True)
class ClientBriefObjection:
    objection: str
    context: str = ""


@dataclass(frozen=True)
class ClientBriefMeta:
    """Counts + timestamps the UI surfaces under the brief so the user
    knows the brief reflects real signal vs being hallucinated from
    nothing."""

    atoms_considered: int
    calls_considered: int
    context_dumps_considered: int
    generated_at: str    # ISO-8601 UTC
    extractor_model: str


@dataclass(frozen=True)
class ClientBrief:
    client_name: str
    client_slug: str
    summary: str
    key_facts: list[str]
    open_commitments: list[ClientBriefCommitment]
    open_objections: list[ClientBriefObjection]
    recent_moves: list[str]
    next_steps: list[str]
    meeting_prep_checklist: list[str]
    meta: ClientBriefMeta

    def to_dict(self) -> dict:
        return {
            "client_name": self.client_name,
            "client_slug": self.client_slug,
            "summary": self.summary,
            "key_facts": list(self.key_facts),
            "open_commitments": [asdict(c) for c in self.open_commitments],
            "open_objections": [asdict(o) for o in self.open_objections],
            "recent_moves": list(self.recent_moves),
            "next_steps": list(self.next_steps),
            "meeting_prep_checklist": list(self.meeting_prep_checklist),
            "meta": asdict(self.meta),
        }


# ────────────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────────────


BRIEF_SYSTEM_PROMPT = """\
You are preparing a brief for a senior consultant about to sit down
with a real client. Read the atoms + call summaries + off-call context
provided and produce a structured, actionable brief the consultant
can scan in 60 seconds before walking into the next conversation.

You return STRICT JSON only — no commentary, no markdown fences. The
schema (every field is required, lists may be empty):

{
  "summary": "2-4 sentences. State of the relationship right now. What
              matters most about this client today, in the consultant's
              voice (concise, no hedging).",
  "key_facts": [
    "5-8 atomic, concrete facts the consultant must remember about this
     client. Numbers, tools, team size, deadlines, decision-makers.
     One sentence each. NO hedging language."
  ],
  "open_commitments": [
    {
      "by_whom": "you" or "them",
      "what": "What was promised, in a single sentence",
      "since_call": "call_id of the call where it was made, e.g.
                     2026-05-12_reece_consultingCall, or null if unknown"
    }
  ],
  "open_objections": [
    {
      "objection": "The hesitation / pushback as the client phrased it",
      "context": "1 sentence on what's underneath the objection"
    }
  ],
  "recent_moves": [
    "What happened in the last 2-3 sessions, one line each. The
     'changed since last brief' view."
  ],
  "next_steps": [
    "What the CONSULTANT should DO before the next session. Specific.
     Action-verb-led. 'Draft 3-line ad-bot scope doc' beats
     'work on scope clarity'."
  ],
  "meeting_prep_checklist": [
    "Specific things to review or have on hand for the next sit-down.
     Mix of: numbers to bring up, questions to ask, references to pull."
  ]
}

Hard rules:
  1. Every entry is anchored to something specific from the input —
     a date, a number, a name, a quote. Generic platitudes are FORBIDDEN.
  2. The brief is for the consultant's eyes only. Do not pad. Do not
     hedge. "There may be considerations around..." is forbidden.
  3. If the input is sparse, return shorter lists — better 3 sharp
     items than 8 vague ones.
  4. Output STRICT JSON. No prose outside the JSON object.
"""


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def generate_client_brief(
    *,
    client_name: str,
    vault_root: Path,
    provider: Optional[LLMProvider] = None,
    client: Optional[AnthropicClient] = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    today: Optional[datetime] = None,
) -> ClientBrief:
    """Build a fresh brief for one client. Caches the result to
    `01_Clients/<slug>/brief.json` on success.

    Raises:
      - ExtractorError on LLM failure
      - typer.BadParameter equivalents if no auth wired
    """
    if provider is None and client is None:
        try:
            client = real_anthropic_client(get_anthropic_key())
        except SecretNotFoundError as exc:
            raise ExtractorError(
                f"Brief generation needs an LLM auth token: {exc}"
            ) from exc

    layout = VaultLayout.for_root(vault_root)
    slug = slugify_client(client_name)
    atoms = _load_client_atoms(layout=layout, client_name=client_name)
    call_summaries = _load_client_call_summaries(layout=layout, client_name=client_name)
    dump_summaries = _load_client_dump_summaries(layout=layout, client_slug=slug)

    user_prompt = _build_user_prompt(
        client_name=client_name,
        atoms=atoms,
        call_summaries=call_summaries,
        dump_summaries=dump_summaries,
        today=today or datetime.now(timezone.utc),
    )

    payload_text = _run_llm(
        system=BRIEF_SYSTEM_PROMPT,
        user=user_prompt,
        provider=provider,
        client=client,
        model=model,
    )
    payload = _parse_json_strict(payload_text)

    brief = _payload_to_brief(
        payload=payload,
        client_name=client_name,
        client_slug=slug,
        atom_count=len(atoms),
        call_count=len(call_summaries),
        dump_count=len(dump_summaries),
        model=model,
        now_iso=(today or datetime.now(timezone.utc))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
    )

    _persist_brief(layout=layout, slug=slug, brief=brief)
    return brief


def load_cached_brief(*, client_name: str, vault_root: Path) -> Optional[ClientBrief]:
    """Read the last generated brief from disk. Returns None when no
    brief has been generated yet, or when the cached file is malformed
    (we never raise — the caller falls through to a fresh generate)."""
    layout = VaultLayout.for_root(vault_root)
    slug = slugify_client(client_name)
    path = layout.clients_dir / slug / "brief.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return _dict_to_brief(data)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _load_client_atoms(*, layout: VaultLayout, client_name: str) -> list[Atom]:
    """Every active or needs_review atom whose `client` matches.
    Sorted recency-desc so the prompt gets recent context first."""
    if not layout.atoms_dir.exists():
        return []
    atoms: list[Atom] = []
    for path in layout.atoms_dir.glob("*.md"):
        try:
            atom = _atom_from_markdown(path)
        except Exception:
            continue
        if atom.client and atom.client.casefold() != client_name.casefold():
            continue
        if atom.status is AtomStatus.retired:
            continue
        atoms.append(atom)
    atoms.sort(key=lambda a: a.last_seen, reverse=True)
    return atoms


def _load_client_call_summaries(*, layout: VaultLayout, client_name: str) -> list[tuple[str, str, str]]:
    """Most-recent-first list of (call_id, date_iso, summary) tuples
    for calls whose `client` frontmatter matches. Caps at the 8 most
    recent to keep the prompt manageable."""
    if not layout.calls_dir.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for path in layout.calls_dir.glob("*.md"):
        try:
            post = fm.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        client_value = post.metadata.get("client") or ""
        # CallNote writer renders client as `[[Reece]]` — strip the brackets.
        if isinstance(client_value, str):
            client_clean = client_value.strip().lstrip("[").rstrip("]").strip()
        else:
            client_clean = ""
        if client_clean.casefold() != client_name.casefold():
            continue
        call_id = str(post.metadata.get("id", path.stem))
        date_iso = str(post.metadata.get("date", ""))
        summary = str(post.metadata.get("summary", "") or "")
        if not summary:
            # Fallback: pull "## Summary\n<text>" out of the body
            summary = _extract_summary_section(post.content or "")
        rows.append((call_id, date_iso, summary))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:8]


def _load_client_dump_summaries(*, layout: VaultLayout, client_slug: str) -> list[tuple[str, str, str]]:
    """(dump_id, observed_at, summary) for off-call context dumps."""
    dumps_dir = layout.context_dumps_dir / client_slug
    if not dumps_dir.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for path in dumps_dir.glob("*.md"):
        try:
            post = fm.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        dump_id = str(post.metadata.get("id", path.stem))
        observed_at = str(post.metadata.get("observed_at", ""))
        summary = _extract_summary_section(post.content or "")
        rows.append((dump_id, observed_at, summary))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:6]


def _extract_summary_section(body: str) -> str:
    """Pull the text under a `## Summary` heading. Used as a fallback
    when CallNote / ContextDump notes don't expose a summary in their
    frontmatter."""
    marker = "## Summary"
    idx = body.find(marker)
    if idx < 0:
        return ""
    chunk = body[idx + len(marker):].lstrip()
    # Stop at the next H2.
    end = chunk.find("\n## ")
    return chunk[:end].strip() if end > 0 else chunk.strip()


def _build_user_prompt(
    *,
    client_name: str,
    atoms: list[Atom],
    call_summaries: list[tuple[str, str, str]],
    dump_summaries: list[tuple[str, str, str]],
    today: datetime,
) -> str:
    """Assemble the user message with date awareness + every piece of
    context the model needs to generate a real brief (not a hallucinated
    one)."""
    lines: list[str] = []
    lines.append(f"Today is {today.date().isoformat()}.")
    lines.append(f"Client: {client_name}")
    lines.append("")

    if call_summaries:
        lines.append("## Past call summaries (most recent first)")
        for call_id, date_iso, summary in call_summaries:
            lines.append(f"- [{date_iso} · {call_id}] {summary}")
        lines.append("")

    if dump_summaries:
        lines.append("## Off-call context drops (most recent first)")
        for dump_id, observed_at, summary in dump_summaries:
            lines.append(f"- [{observed_at} · {dump_id}] {summary}")
        lines.append("")

    if atoms:
        lines.append("## Atoms (most recent first)")
        for atom in atoms[:80]:  # cap so we don't blow the context window
            tag_str = ", ".join(atom.tags) if atom.tags else ""
            tags_segment = f" [{tag_str}]" if tag_str else ""
            lines.append(
                f"- {atom.last_seen} · {atom.type.value} · {atom.body}{tags_segment}"
            )
        lines.append("")

    if not (atoms or call_summaries or dump_summaries):
        lines.append("(No atoms, call notes, or context dumps exist for this client yet.)")

    lines.append(
        "Produce the brief now. Strict JSON only, matching the schema "
        "specified in the system prompt."
    )
    return "\n".join(lines)


def _run_llm(
    *,
    system: str,
    user: str,
    provider: Optional[LLMProvider],
    client: Optional[AnthropicClient],
    model: str,
) -> str:
    if provider is not None:
        try:
            response = provider.chat(
                ChatRequest(
                    system=system,
                    user=user,
                    model=model,
                    max_tokens=4096,
                    enable_prompt_cache=True,
                )
            )
        except ProviderError as exc:
            raise ExtractorError(str(exc)) from exc
        return response.text

    assert client is not None
    resp = client.messages_create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=4096,
    )
    blocks = getattr(resp, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _parse_json_strict(text: str) -> dict:
    cleaned = text.strip()
    # Tolerate accidental code fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop opening + closing fences.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractorError(
            f"Brief LLM returned invalid JSON at line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ExtractorError("Brief LLM did not return a JSON object")
    return data


def _payload_to_brief(
    *,
    payload: dict,
    client_name: str,
    client_slug: str,
    atom_count: int,
    call_count: int,
    dump_count: int,
    model: str,
    now_iso: str,
) -> ClientBrief:
    commitments = [
        ClientBriefCommitment(
            by_whom=str(item.get("by_whom", "")).strip() or "you",
            what=str(item.get("what", "")).strip(),
            since_call=(
                str(item["since_call"]).strip()
                if item.get("since_call") not in (None, "", "null")
                else None
            ),
        )
        for item in (payload.get("open_commitments") or [])
        if isinstance(item, dict)
    ]
    objections = [
        ClientBriefObjection(
            objection=str(item.get("objection", "")).strip(),
            context=str(item.get("context", "") or "").strip(),
        )
        for item in (payload.get("open_objections") or [])
        if isinstance(item, dict)
    ]
    return ClientBrief(
        client_name=client_name,
        client_slug=client_slug,
        summary=str(payload.get("summary", "")).strip(),
        key_facts=[str(x).strip() for x in (payload.get("key_facts") or []) if isinstance(x, str)],
        open_commitments=[c for c in commitments if c.what],
        open_objections=[o for o in objections if o.objection],
        recent_moves=[str(x).strip() for x in (payload.get("recent_moves") or []) if isinstance(x, str)],
        next_steps=[str(x).strip() for x in (payload.get("next_steps") or []) if isinstance(x, str)],
        meeting_prep_checklist=[
            str(x).strip()
            for x in (payload.get("meeting_prep_checklist") or [])
            if isinstance(x, str)
        ],
        meta=ClientBriefMeta(
            atoms_considered=atom_count,
            calls_considered=call_count,
            context_dumps_considered=dump_count,
            generated_at=now_iso,
            extractor_model=model,
        ),
    )


def _dict_to_brief(data: dict) -> ClientBrief:
    """Re-hydrate a cached brief from JSON. Used by load_cached_brief —
    schema-tolerant so a v2 brief loaded by older code falls through
    to fresh generation instead of crashing."""
    meta_raw = data.get("meta") or {}
    meta = ClientBriefMeta(
        atoms_considered=int(meta_raw.get("atoms_considered", 0) or 0),
        calls_considered=int(meta_raw.get("calls_considered", 0) or 0),
        context_dumps_considered=int(meta_raw.get("context_dumps_considered", 0) or 0),
        generated_at=str(meta_raw.get("generated_at", "")),
        extractor_model=str(meta_raw.get("extractor_model", "")),
    )
    return ClientBrief(
        client_name=str(data.get("client_name", "")),
        client_slug=str(data.get("client_slug", "")),
        summary=str(data.get("summary", "")),
        key_facts=[str(x) for x in (data.get("key_facts") or [])],
        open_commitments=[
            ClientBriefCommitment(
                by_whom=str(c.get("by_whom", "you")),
                what=str(c.get("what", "")),
                since_call=c.get("since_call"),
            )
            for c in (data.get("open_commitments") or [])
            if isinstance(c, dict)
        ],
        open_objections=[
            ClientBriefObjection(
                objection=str(o.get("objection", "")),
                context=str(o.get("context", "")),
            )
            for o in (data.get("open_objections") or [])
            if isinstance(o, dict)
        ],
        recent_moves=[str(x) for x in (data.get("recent_moves") or [])],
        next_steps=[str(x) for x in (data.get("next_steps") or [])],
        meeting_prep_checklist=[str(x) for x in (data.get("meeting_prep_checklist") or [])],
        meta=meta,
    )


def _persist_brief(*, layout: VaultLayout, slug: str, brief: ClientBrief) -> None:
    """Cache the brief at `01_Clients/<slug>/brief.json` so the UI can
    re-render instantly on next open without re-running the LLM."""
    client_dir = layout.clients_dir / slug
    client_dir.mkdir(parents=True, exist_ok=True)
    target = client_dir / "brief.json"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(brief.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(target)
