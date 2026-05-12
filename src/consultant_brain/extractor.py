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


def build_user_prompt(transcript: str, call_type: CallType, client_name: str | None) -> str:
    """One-shot prompt: transcript + call type + (optional) client name."""
    client_line = f"Client: {client_name}\n" if client_name else ""
    return f"""\
{client_line}Call type: {call_type.value}

Transcript:
{transcript}
"""


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
    client: AnthropicClient,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 8192,
) -> ExtractorResult:
    """Send the transcript to Claude, parse the response, validate against
    `ExtractorResult`. Raises on any deviation — never returns partial data.
    """
    user_prompt = build_user_prompt(transcript=transcript, call_type=call_type, client_name=client_name)
    response = client.messages_create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
    )

    raw_text = _concat_text(response)
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
