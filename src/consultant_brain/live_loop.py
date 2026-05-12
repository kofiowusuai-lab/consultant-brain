"""Live-loop orchestration — the function /transcript_delta calls after
appending a turn, to opportunistically detect moments + write atoms to
the vault during the call (not just at the end).

`maybe_run_moment_detection` is the single entry point. It enforces a
throttle (at least N new turns AND M seconds since the last run) so a
rapid burst of short turns can't melt the Anthropic bill, then hands a
small transcript window to `moment_detector.detect_moments()` and writes
any high-confidence atoms via vault.py + LanceVaultIndex.

Per-call atom ID derivation reuses the existing
`derive_atom_id(session_filename=..., extractor_version=..., atom_index=...)`
helper — we treat the live `call_id` as the session_filename so atoms
written during a live call collide cleanly when the post-call extractor
re-runs against the same session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from consultant_brain.embedder import LanceVaultIndex
from consultant_brain.extractor import AnthropicClient, real_anthropic_client
from consultant_brain.live_state import CallState
from consultant_brain.moment_detector import detect_moments
from consultant_brain.schemas import (
    EXTRACTOR_VERSION,
    Atom,
    AtomStatus,
    DEFAULT_EXTRACTOR_MODEL,
)
from consultant_brain.secrets import SecretNotFoundError, get_anthropic_key
from consultant_brain.vault import (
    VaultLayout,
    derive_atom_id,
    derive_call_id,
    ensure_client_folder,
    ensure_vault_skeleton,
    write_atom,
)


logger = logging.getLogger(__name__)


# Throttle knobs. Live-call moment detection is opt-in expensive — one
# Anthropic call per detection pass. Tuned so a 30-minute call produces
# roughly 30-50 detection passes ≈ same order as the post-call extractor.
MOMENT_MIN_TURNS_BETWEEN_RUNS = 2
MOMENT_MIN_SECONDS_BETWEEN_RUNS = 30.0

# How much of the rolling window we send to the detector. Smaller than the
# CallState's full 30-turn deque so the detector focuses on "what just
# happened" rather than the whole call.
MOMENT_DETECTOR_WINDOW_TURNS = 5


@dataclass(frozen=True, slots=True)
class LiveDetectionResult:
    """One detection pass's outcome — surfaced to /transcript_delta so the
    HTTP layer can choose whether to include it in the response."""

    ran: bool
    atoms_written: int
    skipped_reason: str | None = None


def maybe_run_moment_detection(
    state: CallState,
    *,
    vault_root: Path,
    anthropic_client: AnthropicClient | None = None,
    now: datetime | None = None,
) -> LiveDetectionResult:
    """Run a detection pass if the throttle allows. Best-effort — failures
    return a skipped LiveDetectionResult; they never raise."""
    now = now or datetime.now(timezone.utc)

    # ---- throttle gate ----
    turns_total = len(state.turns)
    new_turns = turns_total - state.turns_at_last_detection
    if new_turns < MOMENT_MIN_TURNS_BETWEEN_RUNS:
        return LiveDetectionResult(
            ran=False,
            atoms_written=0,
            skipped_reason=f"only {new_turns} new turn(s) since last run",
        )
    if state.last_moment_detection_at is not None:
        elapsed = (now - state.last_moment_detection_at).total_seconds()
        if elapsed < MOMENT_MIN_SECONDS_BETWEEN_RUNS:
            return LiveDetectionResult(
                ran=False,
                atoms_written=0,
                skipped_reason=f"only {elapsed:.0f}s since last run",
            )

    # ---- build the detection window (last N turns, formatted) ----
    with state.lock:
        recent = list(state.turns)[-MOMENT_DETECTOR_WINDOW_TURNS:]
    if not recent:
        return LiveDetectionResult(ran=False, atoms_written=0, skipped_reason="no turns yet")
    window_lines: list[str] = []
    for turn in recent:
        label = "You" if turn.speaker.value == "you" else "Them"
        window_lines.append(f"{label}: {turn.text.strip()}")
    window = "\n\n".join(window_lines)

    # ---- pick the LLM ----
    # Phase 8 wiring + Phase ~ knob: read `BRAIN_MOMENT_PROVIDER` so the
    # Swift Brain Settings picker can flip the moment detector to a
    # cheaper/faster model (DeepSeek, Kimi, Haiku via OpenRouter) without
    # touching the extractor. Falls back gracefully to the legacy
    # AnthropicClient path when the env is unset OR when building the
    # chosen provider fails (e.g. its key is missing). That fallback is
    # important: a missing DeepSeek key should NOT kill live moment
    # detection — Anthropic still runs.
    from consultant_brain.llm.provider import LLMProvider, ProviderError
    from consultant_brain.llm.registry import build_provider, resolve_provider_name

    provider: LLMProvider | None = None
    provider_name = resolve_provider_name(env_var="BRAIN_MOMENT_PROVIDER")
    if provider_name != "anthropic":
        try:
            provider = build_provider(env_var="BRAIN_MOMENT_PROVIDER")
        except ProviderError as exc:
            logger.warning(
                "moment detection: %s provider unavailable (%s) — falling back to Anthropic",
                provider_name, exc,
            )
            provider = None

    client = anthropic_client
    if provider is None and client is None:
        try:
            key = get_anthropic_key()
        except SecretNotFoundError as exc:
            logger.warning("moment detection skipped: %s", exc)
            return LiveDetectionResult(ran=False, atoms_written=0, skipped_reason="no anthropic key")
        client = real_anthropic_client(key)

    # ---- run detection ----
    extracted = detect_moments(
        window=window,
        call_type=state.call_type,
        client_name=state.client,
        provider=provider,
        client=client if provider is None else None,
    )

    # Update throttle even when we got zero atoms — a quiet pass still
    # counted against the throttle so we don't immediately retry.
    state.last_moment_detection_at = now
    state.turns_at_last_detection = turns_total

    if not extracted:
        return LiveDetectionResult(ran=True, atoms_written=0, skipped_reason=None)

    # ---- write each atom to the vault + LanceDB ----
    layout = VaultLayout.for_root(vault_root)
    ensure_vault_skeleton(layout)
    if state.client:
        ensure_client_folder(layout, state.client)

    # Use the live call_id as the deterministic session-filename for atom
    # IDs. When the post-call extractor re-runs against the same session
    # later, those IDs collide cleanly and the writer overwrites in place.
    call_note_id = derive_call_id(
        client_name=state.client or "unknown",
        call_type=state.call_type,
        call_date=state.started_at,
    )
    today = state.started_at.date()

    # Offset atom_index by the live count-so-far so successive detection
    # passes produce unique IDs even within the same call.
    base_index = state.detected_atom_count
    written = 0
    index = LanceVaultIndex(layout)
    for offset, atom_seed in enumerate(extracted):
        atom_id = derive_atom_id(
            session_filename=f"live::{state.call_id}",
            extractor_version=EXTRACTOR_VERSION,
            atom_index=base_index + offset,
        )
        atom = Atom(
            id=atom_id,
            type=atom_seed.type,
            client=state.client,
            call=call_note_id,
            call_type=state.call_type,
            tags=list(atom_seed.tags),
            confidence=atom_seed.confidence,
            evidence_count=1,
            last_seen=today,
            created_at=now,
            status=AtomStatus.active,
            embedding_id=atom_id,
            body=atom_seed.body,
        )
        try:
            write_atom(layout, atom)
            try:
                index.upsert(atom)
            except Exception as exc:  # noqa: BLE001 — atom is on disk; embedding is optional
                logger.warning("live moment embed failed for %s: %s", atom_id, exc)
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("live moment write failed for %s: %s", atom_id, exc)

    state.detected_atom_count = base_index + written
    return LiveDetectionResult(ran=True, atoms_written=written)
