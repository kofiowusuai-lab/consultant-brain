"""Q&A on top of a per-client brief.

`ask_about_client()` takes a natural-language question + optional
chat history, loads the same atoms/calls/context-dumps the brief
generator uses, and asks the LLM to answer concisely. The cached
brief.json (when present) is included in the system prompt so the
model doesn't re-derive the summary on every turn.

Designed for follow-up questions in the dashboard's AI Brief section:
"What was the deadline Reece mentioned?", "Has he confirmed the ad
bot scope?", "What did we agree to send before next call?".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from consultant_brain.briefs.generator import (
    ClientBrief,
    _build_user_prompt,
    _load_client_atoms,
    _load_client_call_summaries,
    _load_client_dump_summaries,
    load_cached_brief,
)
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
from consultant_brain.schemas import DEFAULT_EXTRACTOR_MODEL
from consultant_brain.secrets import SecretNotFoundError, get_anthropic_key
from consultant_brain.vault import VaultLayout


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One Q or A turn in the brief chat. Role is "user" or
    "assistant"; same shape both LLM SDKs accept."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ClientBriefAnswer:
    """One assistant answer. Includes provenance so the UI can show
    "answered from N atoms, latest call X" rather than a bare reply."""

    answer: str
    atoms_consulted: int
    calls_consulted: int
    dumps_consulted: int
    model_used: str


QA_SYSTEM_PROMPT_TEMPLATE = """\
You are answering follow-up questions about ONE specific consulting
client. You have access to:
  - the AI brief generated for this client
  - every active/needs_review atom about them
  - every recent call summary
  - every off-call context dump

Rules for every answer:
  1. Be concise. 1-4 short paragraphs max, or a tight bulleted list.
     This is a senior consultant about to walk into a conversation —
     they need ammunition, not essays.
  2. Anchor every claim to specific evidence. Quote dates, numbers,
     or atom types ("on 2026-05-12 Reece said...", "his ad-bot
     commitment from the May 7 coffee...").
  3. If the answer isn't supported by the data you have, say so
     directly: "I don't see that in the brief or atoms — last update
     on this topic was 2026-05-08." Never speculate. Never hedge.
  4. Use the consultant's voice. Drop "as an AI" disclaimers.
  5. If a follow-up question references something not yet in the
     data, answer "Not in the data yet — closest match is X." Do not
     invent.

Today is {today}.

Client: {client_name}

---

{brief_section}

---

{evidence_section}
"""


def ask_about_client(
    *,
    client_name: str,
    question: str,
    history: Optional[list[ChatTurn]] = None,
    vault_root: Path,
    provider: Optional[LLMProvider] = None,
    client: Optional[AnthropicClient] = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    today: Optional[datetime] = None,
) -> ClientBriefAnswer:
    """Answer one follow-up question about a client.

    `history` lets the dashboard pass prior Q&A turns so multi-turn
    follow-ups ("what about the price he mentioned?") have context.
    Recent-first; we cap the history at the most-recent 6 turns so
    the prompt stays bounded.
    """
    if provider is None and client is None:
        try:
            client = real_anthropic_client(get_anthropic_key())
        except SecretNotFoundError as exc:
            raise ExtractorError(
                f"Brief Q&A needs an LLM auth token: {exc}"
            ) from exc

    layout = VaultLayout.for_root(vault_root)
    today_dt = today or datetime.now(timezone.utc)

    atoms = _load_client_atoms(layout=layout, client_name=client_name)
    call_summaries = _load_client_call_summaries(
        layout=layout, client_name=client_name
    )
    dump_summaries = _load_client_dump_summaries(
        layout=layout, client_slug=client_name.lower().replace(" ", "_")
    )
    cached_brief = load_cached_brief(client_name=client_name, vault_root=vault_root)

    evidence_section = _build_user_prompt(
        client_name=client_name,
        atoms=atoms,
        call_summaries=call_summaries,
        dump_summaries=dump_summaries,
        today=today_dt,
    )

    brief_section = _format_brief_block(cached_brief)

    system_prompt = QA_SYSTEM_PROMPT_TEMPLATE.format(
        today=today_dt.date().isoformat(),
        client_name=client_name,
        brief_section=brief_section,
        evidence_section=evidence_section,
    )

    # Build the conversation: prior turns (capped) + the new question.
    # The provider's ChatRequest only supports a single user message
    # today, so we pre-render multi-turn context into the user content
    # rather than expanding the abstraction. Cleaner refactor lands
    # when more endpoints need multi-turn.
    user_content = _render_conversation(history, question)

    answer_text = _call_llm(
        system=system_prompt,
        user=user_content,
        provider=provider,
        client=client,
        model=model,
    )

    return ClientBriefAnswer(
        answer=answer_text.strip(),
        atoms_consulted=len(atoms),
        calls_consulted=len(call_summaries),
        dumps_consulted=len(dump_summaries),
        model_used=model,
    )


# ────────────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────────────


def _format_brief_block(brief: Optional[ClientBrief]) -> str:
    """Render the cached brief as Markdown so the LLM can reference it
    by section. Returns an empty-ish placeholder when no brief has
    been generated yet."""
    if brief is None:
        return "## Cached brief\n_No brief generated yet — answer from atoms + calls only._"

    lines: list[str] = ["## Cached brief"]
    if brief.summary:
        lines.append(f"**Summary**: {brief.summary}")
    if brief.key_facts:
        lines.append("**Key facts**:")
        lines.extend(f"- {fact}" for fact in brief.key_facts)
    if brief.open_commitments:
        lines.append("**Open commitments**:")
        for c in brief.open_commitments:
            since = f" (since {c.since_call})" if c.since_call else ""
            lines.append(f"- {c.by_whom.upper()}: {c.what}{since}")
    if brief.open_objections:
        lines.append("**Open objections**:")
        for o in brief.open_objections:
            ctx = f" — {o.context}" if o.context else ""
            lines.append(f"- {o.objection}{ctx}")
    if brief.recent_moves:
        lines.append("**Recent moves**:")
        lines.extend(f"- {m}" for m in brief.recent_moves)
    if brief.next_steps:
        lines.append("**Next steps**:")
        lines.extend(f"- {n}" for n in brief.next_steps)
    if brief.meeting_prep_checklist:
        lines.append("**Meeting prep**:")
        lines.extend(f"- {p}" for p in brief.meeting_prep_checklist)
    return "\n".join(lines)


def _render_conversation(history: Optional[list[ChatTurn]], question: str) -> str:
    """Flatten chat history + new question into a single user message.

    Cap at the 6 most-recent turns so a long conversation doesn't
    drift the model and doesn't blow the prompt budget. Format:

      Previous Q&A:
      Q: ...
      A: ...

      New question:
      <the current question>
    """
    if not history:
        return f"Question: {question.strip()}"

    capped = history[-6:]
    lines = ["Previous Q&A:"]
    for turn in capped:
        prefix = "Q" if turn.role == "user" else "A"
        lines.append(f"{prefix}: {turn.content.strip()}")
    lines.append("")
    lines.append(f"New question: {question.strip()}")
    return "\n".join(lines)


def _call_llm(
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
                    max_tokens=2048,
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
        max_tokens=2048,
    )
    blocks = getattr(resp, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()
