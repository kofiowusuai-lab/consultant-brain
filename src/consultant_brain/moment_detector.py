"""Online moment detector — the live-call counterpart to the post-call
atom extractor.

Where `extractor.extract()` reads a full transcript at call end and
emits 15-30 atoms, `detect_moments()` reads the last 3-5 turns of an
in-flight call and emits 0-3 high-confidence atoms describing what just
happened semantically (an objection got raised, a commitment got made,
the client signaled confusion, etc.).

Key differences from the post-call extractor:
  - Much smaller input window — last 3-5 turns instead of whole call.
  - Much higher confidence floor (0.7 vs 0.5) — false positives during
    a live call clutter the overlay; we only fire when we're sure.
  - Returns 0-3 atoms max per detection pass — the live loop runs this
    every few turns, so we want to surface "what's new" not "everything".
  - Same `ExtractedAtom` schema as the post-call path — so the vault
    writer / embedder / retrieval pipeline are unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

from consultant_brain.extractor import (
    AnthropicClient,
    ExtractorError,
)
from consultant_brain.llm.provider import (
    ChatRequest,
    LLMProvider,
    ProviderError,
)
from consultant_brain.schemas import (
    DEFAULT_EXTRACTOR_MODEL,
    AtomType,
    CallType,
    ExtractedAtom,
)


# ────────────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────────────


MOMENT_SYSTEM_PROMPT = """\
You are a live-call moment detector. Read the most recent 3-5 turns of a
consulting call IN PROGRESS and identify any meaningful moment that just
occurred — moments that are worth remembering even before the call is
over.

The 7 atom types (same as the post-call extractor):

  insight       — observation about call dynamics (rare during live;
                  prefer client_fact for concrete things said).
  objection     — concern or pushback the client just raised.
  commitment    — concrete promise either side just made.
  win_signal    — enthusiasm, scope expansion, strong buying signals.
  loss_signal   — deflection, "let me think about it", going silent.
  confusion    — client asked a clarifying question or repeated something
                  back wrong.
  client_fact   — specific verifiable fact the client just stated about
                  their business (numbers, tools, team size, workflow).

Hard rules:
  1. CONFIDENCE FLOOR IS 0.7. During a live call, false positives are
     worse than missing a moment — the consultant will see this surface
     in the overlay; spurious noise erodes trust. Below 0.7: emit nothing.
  2. Anchor every atom to a SPECIFIC noun/number/tool/phrase from the
     last turn. If the same atom could fit any other call, you failed.
  3. Return AT MOST 3 atoms per detection pass. If you find more, keep
     the highest-confidence + most-actionable ones.
  4. Skip if the latest turn is small talk, filler, or VAD fragments.
  5. Skip if nothing meaningful happened in the latest turn — return an
     empty list. Quietness is a valid answer.
  6. Body is 1-3 sentences. One idea per atom. No essays.
  7. Tag with 1-5 lowercase snake_case keywords for retrieval.

Output strict JSON only — no commentary, no markdown fences:

{
  "atoms": [
    { "type": "objection", "body": "...", "confidence": 0.85, "tags": ["budget"] }
  ]
}

When nothing new is meaningful: { "atoms": [] }
"""


def build_user_prompt(window: str, call_type: CallType, client_name: str | None) -> str:
    """Compact prompt — just the latest window + the call type, no extra
    chrome. Speed matters during live calls."""
    client_line = f"Client: {client_name}\n" if client_name else ""
    return f"""\
{client_line}Call type: {call_type.value}

Latest turns (most recent at the bottom):
{window}
"""


# ────────────────────────────────────────────────────────────────────────────
# Detection
# ────────────────────────────────────────────────────────────────────────────


def detect_moments(
    *,
    window: str,
    call_type: CallType,
    client_name: str | None,
    client: AnthropicClient | None = None,
    provider: LLMProvider | None = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 1024,
    confidence_floor: float = 0.7,
) -> list[ExtractedAtom]:
    """Run one detection pass. Returns 0-3 atoms above `confidence_floor`.

    Errors return `[]` rather than raising — the live loop should never
    take down the FastAPI service. Logged failures are visible in the
    service's stderr.

    Either `provider` (new path) or `client` (legacy / test path) must
    be supplied. Passing both = provider wins.
    """
    if not window.strip():
        return []
    if provider is None and client is None:
        return []  # never crash the live loop

    user_prompt = build_user_prompt(
        window=window, call_type=call_type, client_name=client_name
    )

    raw: str = ""
    if provider is not None:
        try:
            resp = provider.chat(
                ChatRequest(
                    system=MOMENT_SYSTEM_PROMPT,
                    user=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    enable_prompt_cache=True,
                )
            )
            raw = resp.text
        except ProviderError:
            return []
        except Exception:
            return []
    else:
        try:
            response = client.messages_create(  # type: ignore[union-attr]
                model=model,
                system=MOMENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
            )
        except Exception:
            return []
        raw = _concat_text(response)

    if not raw:
        return []

    try:
        payload = _parse_json_strict(raw)
        atoms_raw = payload.get("atoms") or []
        atoms: list[ExtractedAtom] = []
        for entry in atoms_raw:
            atom = ExtractedAtom.model_validate(entry)
            if atom.confidence >= confidence_floor:
                atoms.append(atom)
        return atoms[:3]
    except Exception:
        return []


def _concat_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.DOTALL)


def _parse_json_strict(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractorError(
            f"Moment detector returned invalid JSON at line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc
