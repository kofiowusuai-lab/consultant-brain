"""Claude-driven atom extractor.

Given a transcript and the call's call-type, returns an `ExtractorResult`
with a 1-3 sentence summary + a list of `ExtractedAtom` records. The
extractor only fills semantically-meaningful fields; the vault writer
attaches bookkeeping (id, client, call, embedding_id, dates).

Why Claude and not local: Sonnet 4.6 hits ≥0.85 F1 on the 7-type taxonomy
on a single-shot prompt (master prompt explicitly authorizes cloud LLM
for reasoning). Local Ollama models cost 30-60s + multi-GB downloads and
miss subtle moments (conditional commitments, sarcasm, indirect confusion
signals). We re-evaluate when an Ollama 14B+ matches.

Tests cover the prompt builder, JSON parsing, and a mocked client. Real
Anthropic calls are guarded behind the --live flag in the CLI; CI runs
without API access.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from anthropic import Anthropic

from consultant_brain.llm.provider import (
    ChatRequest,
    LLMProvider,
    ProviderError,
)
from datetime import date

from consultant_brain.schemas import (
    DEFAULT_EXTRACTOR_MODEL,
    AtomType,
    CallType,
    ExtractedAtom,
    ExtractorResult,
)


# ────────────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """\
You are an atom extractor for a senior consultant's knowledge graph.

Your job: read one call transcript and return a JSON object with:
  - "summary": 1-3 sentences capturing the call's substance.
  - "atoms": a list of typed atomic notes the consultant should remember.

Atoms are atomic: each is one idea, 1-3 sentences max, no headers or bullets
inside the body. Reject the urge to write essays. If a thought needs more
than 3 sentences, it's two atoms.

The 7 atom types:

  insight       — observation about call dynamics, patterns, or framing.
                  Example: "Reece keeps saying 'I think' about scope —
                  hasn't decided what 'winning ad' means yet."

  objection     — concern, hesitation, or pushback raised by the client.
                  Example: "Tried that before with another vendor and it
                  didn't stick."

  commitment    — concrete promise made by either side. Includes scheduling,
                  deliverables, scope agreements.
                  Example: "Reece will send the 3 best manual ads by Friday."

  win_signal    — enthusiasm, scope expansion, 'this is exactly what we
                  need', concrete buying signals.

  loss_signal   — deflection, going silent on questions, 'let me think about
                  it', pushing decisions out, unanswered questions.

  confusion     — client asked a clarifying question, repeated something back
                  wrong, or signaled they didn't follow.

  client_fact   — verifiable fact the client stated about their business —
                  numbers, tools, team size, current stack, workflow. Anchors
                  future retrieval.

Hard rules:
  1. Every atom must be anchored to something specific the client said or
     did. Quote a noun, number, tool, person, or phrase from the transcript.
     If an atom could be pasted into any other call, you have failed.
  2. Tag each atom with 1-5 lowercase snake_case keywords for retrieval
     (e.g. "budget", "scope", "retrieval", "next_step", "team_size").
  3. Confidence is 0.0-1.0 based on how unambiguously the transcript
     supports the atom. 0.9+ for direct quotes; 0.5-0.7 for inferred
     dynamics. Below 0.5: don't emit.
  4. Don't extract more than ~30 atoms per call. If you find more, keep the
     most consequential ones.
  5. NEVER include filler ("um", "uh"), broken VAD fragments, or off-topic
     side comments.
  6. NEVER reveal that you are an AI or reference the tool that produced
     this transcript.

Output strict JSON only. No commentary, no markdown fences. Schema:

{
  "summary": "...",
  "atoms": [
    {
      "type": "objection",
      "body": "...",
      "confidence": 0.82,
      "tags": ["budget", "scope"]
    }
  ]
}
"""


def build_user_prompt(
    transcript: str,
    call_type: CallType,
    client_name: str | None,
    *,
    observed_at: "date | None" = None,
    today: "date | None" = None,
) -> str:
    """One-shot prompt: transcript + call type + (optional) client name.

    Phase 10: optionally include a `Today is …. Observed on ….` preamble
    so the LLM can resolve relative dates ("yesterday", "next Friday")
    against the right reference points instead of training-data prior.
    """
    client_line = f"Client: {client_name}\n" if client_name else ""
    time_line = _build_time_preamble(observed_at=observed_at, today=today, kind="call")
    return f"""\
{time_line}{client_line}Call type: {call_type.value}

Transcript:
{transcript}
"""


def _build_time_preamble(
    *,
    observed_at: "date | None",
    today: "date | None",
    kind: str,
) -> str:
    """Single time-awareness line every extractor flavor shares.

    `kind` is "call" / "source" / "context_dump" — appears in the line
    so the LLM has the right frame for interpreting the date. Returns
    "" when no dates are supplied (keeps existing call sites that
    don't pass dates indistinguishable from before).
    """
    from datetime import datetime, timezone

    parts: list[str] = []
    today_d = today or datetime.now(timezone.utc).date()
    parts.append(f"Today is {today_d.isoformat()}")
    if observed_at is not None:
        parts.append(f"this {kind} was observed on {observed_at.isoformat()}")
    return ". ".join(parts) + ".\n"


# ────────────────────────────────────────────────────────────────────────────
# Phase 9 — knowledge extraction (videos, articles, podcasts)
# ────────────────────────────────────────────────────────────────────────────


KNOWLEDGE_SYSTEM_PROMPT = """\
You are a knowledge extractor for a senior consultant's reference library.

You read transcripts from EXTERNAL SOURCES (YouTube videos, Instagram
Reels, podcasts, articles) — not the consultant's own calls. Your job
is to mine timeless insights, reusable frameworks, and concrete
verifiable facts the consultant can lean on later. You are NOT mining
client commitments / objections / win-or-loss signals — those are
call-specific.

Return a JSON object:
  - "summary": 1-3 sentences. What is this source about? Who is the
    speaker / author? Why might a consultant care?
  - "atoms": a list of typed atomic notes.

You may only emit these atom types from external sources:

  insight       — a non-obvious observation, principle, or pattern the
                  speaker articulates. The kind of thing the consultant
                  would scribble in a margin and reuse.
                  Example: "The mistake most agencies make is selling
                  the deliverable instead of the outcome."

  client_fact   — a verifiable, specific fact: a number, a tool, a
                  technique, a workflow detail. Anchored to something
                  concrete in the source.
                  Example: "Hormozi's team runs 4 cold emails per
                  prospect per week and stops at 6 if no reply."

  win_signal    — a pattern of WHAT WORKS, articulated as a claim.
                  Reusable language for the consultant's own calls.
                  Example: "When a prospect mentions a deadline first,
                  ask 'what happens if you miss it?' — turns vague
                  urgency into concrete loss aversion."

Hard rules:
  1. Atoms are atomic: 1-3 sentences max. No essays.
  2. Every atom is anchored to a specific moment, number, name, or
     phrase in the source. If a generic "always be closing"-flavor
     truism could fit any context, skip it.
  3. Tag each atom with 1-5 lowercase snake_case keywords. Include the
     source's topic when caller supplies one.
  4. Confidence: 0.9+ for direct quotes or specific numbers; 0.6-0.8
     for clearly-articulated principles. Below 0.5: don't emit.
  5. Cap at ~20 atoms per source. Quality over quantity.
  6. NEVER fabricate a number, name, or claim. If the source is vague
     about a fact, don't emit a client_fact for it.

Output strict JSON only. No commentary, no markdown fences. Schema:

{
  "summary": "...",
  "atoms": [
    {
      "type": "insight",
      "body": "...",
      "confidence": 0.82,
      "tags": ["sales", "framing"]
    }
  ]
}
"""


def build_knowledge_user_prompt(
    *,
    transcript: str,
    source_title: str,
    author: str | None,
    topic: str | None,
    for_client: str | None,
    observed_at: "date | None" = None,
    today: "date | None" = None,
) -> str:
    """Prompt for `extract_from_source` — emphasizes that the source
    is external context, not a client call.

    Phase 10: optional `observed_at` + `today` weave a time preamble in
    so relative dates resolve correctly.
    """
    time_line = _build_time_preamble(
        observed_at=observed_at, today=today, kind="source"
    )
    lines: list[str] = []
    if time_line:
        lines.append(time_line.rstrip())
    lines.append(f"Source title: {source_title}")
    if author:
        lines.append(f"Author / speaker: {author}")
    if topic:
        lines.append(f"Topic (caller-supplied): {topic}")
    if for_client:
        lines.append(
            f"Studied with this client in mind: {for_client}. Atoms may "
            f"reference relevance to {for_client} if obvious."
        )
    lines.append("")
    lines.append("Transcript:")
    lines.append(transcript)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Phase 10 — context-dump extraction (off-call drops, in-person convos)
# ────────────────────────────────────────────────────────────────────────────


CONTEXT_DUMP_SYSTEM_PROMPT = """\
You are a context extractor for a senior consultant's knowledge graph.

The user is dropping in OFF-CALL CONTEXT about ONE specific client —
a document they sent, a photo of a whiteboard, a recording of an
in-person conversation, a screenshot of a Slack thread. The consultant
wants the brain to remember this so it can surface relevant moments
during future calls with this client.

Return a JSON object:
  - "summary": 1-3 sentences. What is this context about, and what
    should the consultant remember from it before the next call?
  - "atoms": a list of typed atomic notes.

All seven atom types are valid here — context dumps often carry the
same shape of moments calls do:

  insight       — observation about how the client thinks / what
                  matters to them, anchored to something specific they
                  said or wrote.
  objection     — concern, hesitation, pushback from the client.
  commitment    — concrete promise either side made (email exchange,
                  signed proposal, in-person agreement).
  win_signal    — enthusiasm, scope expansion, buying signal.
  loss_signal   — deflection, going silent, "let me think about it".
  confusion     — clarifying question, repeated something back wrong.
  client_fact   — specific verifiable fact about the client's business
                  — numbers, tools, team size, workflow.

Hard rules:
  1. Every atom anchors to a specific date, number, name, phrase, or
     visual detail from the input. If it could fit any other client,
     you failed.
  2. DO NOT invent facts. If the input is ambiguous, drop the atom.
  3. Tag each atom with 1-5 lowercase snake_case keywords for retrieval.
  4. Confidence 0.9+ for direct quotes / clear numbers; 0.6-0.8 for
     clearly-stated principles; below 0.5 = don't emit.
  5. Cap at ~25 atoms. Quality over quantity.
  6. NEVER reference the extraction tool or that this came from a file.

Output strict JSON only. No commentary, no markdown fences. Schema:

{
  "summary": "...",
  "atoms": [
    {
      "type": "commitment",
      "body": "...",
      "confidence": 0.85,
      "tags": ["pricing", "in_person"]
    }
  ]
}
"""


def build_context_dump_user_prompt(
    *,
    text: str,
    client_name: str,
    source_kind_label: str,
    source_filename: str,
    notes: str | None,
    observed_at: "date | None" = None,
    today: "date | None" = None,
) -> str:
    """Prompt for `extract_from_context_dump`. Bundles the dump's
    metadata into the user message so the model has full context."""
    time_line = _build_time_preamble(
        observed_at=observed_at, today=today, kind="context"
    )
    lines: list[str] = []
    if time_line:
        lines.append(time_line.rstrip())
    lines.append(f"Client: {client_name}")
    lines.append(f"Source kind: {source_kind_label}")
    lines.append(f"Source filename: {source_filename}")
    if notes:
        lines.append("")
        lines.append("User notes:")
        lines.append(notes.strip())
    lines.append("")
    lines.append("Context:")
    lines.append(text)
    return "\n".join(lines)


def extract_from_context_dump(
    *,
    text: str,
    client_name: str,
    source_kind_label: str,
    source_filename: str,
    notes: str | None = None,
    observed_at: "date | None" = None,
    today: "date | None" = None,
    client: AnthropicClient | None = None,
    provider: "LLMProvider | None" = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 8192,
) -> ExtractorResult:
    """Context-dump-tuned extractor. Same return shape as `extract()`
    but with the off-call system prompt that permits the full 7-atom
    palette (commitments/objections/etc. are valid here, unlike the
    knowledge extractor)."""
    if provider is None and client is None:
        raise ExtractorError("extract_from_context_dump() requires `provider` or `client`")

    user_prompt = build_context_dump_user_prompt(
        text=text,
        client_name=client_name,
        source_kind_label=source_kind_label,
        source_filename=source_filename,
        notes=notes,
        observed_at=observed_at,
        today=today,
    )

    if provider is not None:
        try:
            response = provider.chat(
                ChatRequest(
                    system=CONTEXT_DUMP_SYSTEM_PROMPT,
                    user=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    enable_prompt_cache=True,
                )
            )
        except ProviderError as exc:
            raise ExtractorError(str(exc)) from exc
        raw_text = response.text
    else:
        assert client is not None
        resp = client.messages_create(
            model=model,
            system=CONTEXT_DUMP_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )
        raw_text = _concat_text(resp)

    payload = _parse_json_strict(raw_text)
    return ExtractorResult.model_validate(payload)


def extract_from_source(
    *,
    transcript: str,
    source_title: str,
    author: str | None,
    topic: str | None = None,
    for_client: str | None = None,
    client: AnthropicClient | None = None,
    provider: "LLMProvider | None" = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 8192,
) -> ExtractorResult:
    """Knowledge-tuned extractor. Same return shape as `extract()`, but
    biased toward `insight` / `client_fact` / `win_signal` atoms with
    the call-specific types (commitment / objection / etc.) suppressed.

    Pass either `provider` (Phase 8 LLMProvider abstraction) or `client`
    (legacy AnthropicClient mock). Mirrors `extract()`'s contract."""
    if provider is None and client is None:
        raise ExtractorError("extract_from_source() requires `provider` or `client`")

    user_prompt = build_knowledge_user_prompt(
        transcript=transcript,
        source_title=source_title,
        author=author,
        topic=topic,
        for_client=for_client,
    )

    if provider is not None:
        try:
            response = provider.chat(
                ChatRequest(
                    system=KNOWLEDGE_SYSTEM_PROMPT,
                    user=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    enable_prompt_cache=True,
                )
            )
        except ProviderError as exc:
            raise ExtractorError(str(exc)) from exc
        raw_text = response.text
    else:
        assert client is not None
        resp = client.messages_create(
            model=model,
            system=KNOWLEDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )
        raw_text = _concat_text(resp)

    payload = _parse_json_strict(raw_text)
    return ExtractorResult.model_validate(payload)


# ────────────────────────────────────────────────────────────────────────────
# Anthropic client wrapper
# ────────────────────────────────────────────────────────────────────────────


class _MessageContent(Protocol):
    text: str


class _Message(Protocol):
    content: list[_MessageContent]


class AnthropicClient(Protocol):
    """Minimal Protocol so tests can pass in a mock without monkey-patching
    the real `anthropic.Anthropic` class.
    """

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> _Message:
        ...


class _RealAnthropicClient:
    """Thin wrapper around `anthropic.Anthropic.messages.create` matching the
    `AnthropicClient` Protocol — so production code uses the same surface
    that tests mock.
    """

    def __init__(self, api_key: str) -> None:
        self._client = Anthropic(api_key=api_key)

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ):
        return self._client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )


def real_anthropic_client(api_key: str) -> AnthropicClient:
    """Public factory so the CLI doesn't import the private wrapper class."""
    return _RealAnthropicClient(api_key)


# ────────────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────────────


def extract(
    *,
    transcript: str,
    call_type: CallType,
    client_name: str | None,
    client: AnthropicClient | None = None,
    provider: LLMProvider | None = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 8192,
) -> ExtractorResult:
    """Send the transcript to an LLM, parse the response, validate against
    `ExtractorResult`. Raises on any deviation — never returns partial data.

    Either `provider` (new path) or `client` (legacy / test path) must be
    supplied. When both are passed, `provider` wins — call sites in
    transition can keep `client` while we migrate them.
    """
    if provider is None and client is None:
        raise ExtractorError("extract() requires either `provider` or `client`")

    user_prompt = build_user_prompt(
        transcript=transcript,
        call_type=call_type,
        client_name=client_name,
    )

    if provider is not None:
        try:
            response = provider.chat(
                ChatRequest(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    enable_prompt_cache=True,
                )
            )
        except ProviderError as exc:
            raise ExtractorError(str(exc)) from exc
        raw_text = response.text
    else:
        assert client is not None  # narrowed by the guard above
        resp = client.messages_create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )
        raw_text = _concat_text(resp)

    payload = _parse_json_strict(raw_text)
    return ExtractorResult.model_validate(payload)


def _concat_text(response: Any) -> str:
    """Anthropic returns a list of content blocks; we concatenate the .text
    of each. Defensive in case the response shape is mocked in tests."""
    blocks = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        raise ExtractorError("Anthropic response had no text blocks")
    return "".join(parts).strip()


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.DOTALL)


def _parse_json_strict(text: str) -> dict[str, Any]:
    """Tolerate accidental ```json fences but reject everything else."""
    cleaned = text.strip()
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractorError(
            f"Extractor returned invalid JSON at line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc


# ────────────────────────────────────────────────────────────────────────────
# Errors
# ────────────────────────────────────────────────────────────────────────────


class ExtractorError(RuntimeError):
    """Raised when Claude's response can't be parsed or validated."""
