"""FastAPI service that exposes the brain to the Swift app + Phase 4's
live loop. All endpoints sit on localhost only — the trust model is
"this user owns the machine"; Phase 5+ adds auth when there's a reason.

Endpoints:
  GET  /healthz                         — liveness
  POST /call_start                      — begin a call
  POST /call_end                        — end a call, free state
  POST /transcript_delta                — append one turn to a call's window
  GET  /suggestions?call_id=X           — three-layer retrieval against the window

State is in-memory via LiveCallRegistry. Suggestions are computed on
demand against the rolling window (we don't cache because Phase 4 polls
every 15s anyway, and the retrieval pipeline is sub-second).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from consultant_brain.live_loop import maybe_run_moment_detection
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.retrieve import RankedHit, retrieve
from consultant_brain.schemas import CallType, Speaker
from consultant_brain.scoring.corrections import (
    ScoreCorrection,
    append_correction,
)
from consultant_brain.scoring.features import FEATURE_NAMES, compute_features
from consultant_brain.scoring.loader import CallNotFoundError, load_call_for_scoring
from consultant_brain.scoring.score import compute_score
from consultant_brain.scoring.weights import ScoringWeightsTable


DEFAULT_VAULT = Path.home() / "ConsultantBrain"


# ────────────────────────────────────────────────────────────────────────────
# Request / response models
# ────────────────────────────────────────────────────────────────────────────


class CallStartRequest(BaseModel):
    call_id: str = Field(min_length=1)
    client: str | None = Field(default=None)
    call_type: CallType = Field(default=CallType.consulting_call)


class CallStartResponse(BaseModel):
    call_id: str
    client: str | None
    call_type: CallType
    started_at: datetime


class CallEndRequest(BaseModel):
    call_id: str = Field(min_length=1)


class CallEndResponse(BaseModel):
    call_id: str
    ended: bool  # False if the call_id wasn't active (caller might have already ended it)


class TranscriptDeltaRequest(BaseModel):
    call_id: str = Field(min_length=1)
    speaker: Speaker
    text: str = Field(min_length=1)


class TranscriptDeltaResponse(BaseModel):
    call_id: str
    turn_count: int
    accepted: bool


class SuggestionDTO(BaseModel):
    """Wire format for one suggestion. Matches what the Swift overlay
    expects to render — layer, score, body, and the rank components so
    --explain works over HTTP too.
    """

    atom_id: str
    type: str
    layer: str
    body: str
    client: str | None
    call_type: str
    confidence: float
    similarity: float
    recency: float
    score: float
    reason: str

    @classmethod
    def from_ranked_hit(cls, rh: RankedHit) -> "SuggestionDTO":
        return cls(
            atom_id=rh.hit.id,
            type=rh.hit.type,
            layer=rh.layer,
            body=rh.hit.body,
            client=rh.hit.client,
            call_type=rh.hit.call_type,
            confidence=rh.confidence,
            similarity=rh.similarity,
            recency=rh.recency,
            score=rh.score,
            reason=rh.reason,
        )


class SuggestionsResponse(BaseModel):
    call_id: str
    suggestions: list[SuggestionDTO]
    window_chars: int  # how much transcript was in the rolling window
    generated_at: datetime


# ────────────────────────────────────────────────────────────────────────────
# Scoring DTOs (Phase 5)
# ────────────────────────────────────────────────────────────────────────────


class FeatureContributionDTO(BaseModel):
    name: str
    raw_value: float
    weight: float
    contribution: float


class ScoreResponse(BaseModel):
    call_id: str
    call_type: str
    score: float
    raw_score: float
    bias: float
    contributions: list[FeatureContributionDTO]
    summary: str  # extractor's 3-line summary, surfaced for context


class ScoreOverrideRequest(BaseModel):
    call_id: str = Field(min_length=1)
    user_score: float = Field(ge=0.0, le=100.0)
    primary_win: str | None = Field(default=None)


class ScoreOverrideResponse(BaseModel):
    call_id: str
    predicted_score: float
    user_score: float
    delta: float
    corrections_count: int  # total corrections in the log AFTER this write


# ────────────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────────────


def create_app(*, vault_root: Path | None = None, registry: LiveCallRegistry | None = None) -> FastAPI:
    """Build a FastAPI app bound to a specific vault + registry. Tests use
    their own vault path + a fresh registry; the production CLI launcher
    uses DEFAULT_VAULT + a singleton registry created at startup.
    """
    resolved_vault = (vault_root or DEFAULT_VAULT).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Lifespan hook is the place future Phase 5+ persistence code would
        # warm caches / flush at shutdown. Phase 3 has no such state, so
        # it's a clean pass-through with a doc comment.
        yield

    app = FastAPI(
        title="Consultant Brain",
        description="Local FastAPI service wrapping the Phase 1/2 retrieval pipeline.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.registry = registry or LiveCallRegistry()
    app.state.vault_root = resolved_vault
    # Phase 4: live moment detection runs after each /transcript_delta when
    # enabled. Off by default in tests (set via env in production). The env
    # var lets the Swift app launch the service with detection enabled
    # without code changes here.
    app.state.moment_detection_enabled = os.environ.get(
        "CONSULTANT_BRAIN_MOMENT_DETECTION", ""
    ).lower() in ("1", "true", "yes", "on")

    _register_routes(app)
    return app


def _compute_score_response(*, vault: Path, call_id: str, primary_win: str | None):
    """Shared between GET /score and POST /score_override — loads the call,
    runs feature extraction + the scorer, and returns a ScoreResponse DTO.

    Raises HTTPException(404) when the call note doesn't exist. The
    primary-win field is optional; when missing the feature stays at the
    neutral 0.5 baseline.
    """
    try:
        inputs = load_call_for_scoring(
            vault_root=vault,
            call_id=call_id,
            primary_win=primary_win or None,
        )
    except CallNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    weights_path = vault.expanduser().resolve() / "00_System" / "scoring_weights.yaml"
    weights_table = ScoringWeightsTable.load(weights_path)
    weights = weights_table.for_call_type(inputs.call_type)

    features = compute_features(
        atoms=inputs.atoms,
        turns=inputs.turns,
        call_type=inputs.call_type,
        primary_win=inputs.primary_win,
    )
    result = compute_score(features=features, weights=weights)

    return ScoreResponse(
        call_id=call_id,
        call_type=inputs.call_type.value,
        score=result.score,
        raw_score=result.raw_score,
        bias=result.bias,
        contributions=[
            FeatureContributionDTO(
                name=c.name,
                raw_value=c.raw_value,
                weight=c.weight,
                contribution=c.contribution,
            )
            for c in result.contributions
        ],
        summary=inputs.summary,
    )


def _run_moment_detection_safely(state, vault_root: Path) -> None:
    """Run the live-loop detection pass, swallowing all errors so a single
    background-task failure can never crash the worker / break /transcript_delta.
    """
    import logging

    try:
        maybe_run_moment_detection(state, vault_root=vault_root)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("background moment detection failed")


def reload_app() -> FastAPI:
    """Import-string entrypoint for `uvicorn --reload`. Reads the vault path
    from `CONSULTANT_BRAIN_VAULT` env var (the `serve` CLI sets it before
    spawning uvicorn). Plain factory pattern so uvicorn can reload the
    module on file changes without losing the vault binding.
    """
    import os
    vault = os.environ.get("CONSULTANT_BRAIN_VAULT")
    return create_app(vault_root=Path(vault) if vault else None)


def _register_routes(app: FastAPI) -> None:
    def get_registry(request: Request) -> LiveCallRegistry:
        return request.app.state.registry

    def get_vault(request: Request) -> Path:
        return request.app.state.vault_root

    @app.get("/healthz")
    def healthz(
        request: Request,
        registry: LiveCallRegistry = Depends(get_registry),
        vault: Path = Depends(get_vault),
    ) -> dict:
        """Liveness check. Returns active-call count + vault path so a
        misconfigured client can spot the wrong-vault footgun immediately.
        """
        return {
            "ok": True,
            "active_calls": len(registry),
            "vault_root": str(vault),
        }

    @app.post("/call_start", response_model=CallStartResponse)
    def call_start(
        body: CallStartRequest,
        registry: LiveCallRegistry = Depends(get_registry),
    ) -> CallStartResponse:
        state = registry.start(call_id=body.call_id, client=body.client, call_type=body.call_type)
        return CallStartResponse(
            call_id=state.call_id,
            client=state.client,
            call_type=state.call_type,
            started_at=state.started_at,
        )

    @app.post("/call_end", response_model=CallEndResponse)
    def call_end(
        body: CallEndRequest,
        registry: LiveCallRegistry = Depends(get_registry),
    ) -> CallEndResponse:
        popped = registry.end(body.call_id)
        return CallEndResponse(call_id=body.call_id, ended=popped is not None)

    @app.post("/transcript_delta", response_model=TranscriptDeltaResponse)
    def transcript_delta(
        body: TranscriptDeltaRequest,
        background: BackgroundTasks,
        request: Request,
        registry: LiveCallRegistry = Depends(get_registry),
    ) -> TranscriptDeltaResponse:
        state = registry.get(body.call_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"No active call with id={body.call_id!r}. POST /call_start first.",
            )
        state.append_turn(body.speaker, body.text)

        # Phase 4: opportunistic live moment detection — scheduled as a
        # background task so the HTTP response stays under 50ms. The
        # throttle inside `maybe_run_moment_detection` enforces "at most
        # one Anthropic call per N new turns + M seconds" so a chatty call
        # doesn't melt the bill.
        if request.app.state.moment_detection_enabled:
            vault_root: Path = request.app.state.vault_root
            background.add_task(_run_moment_detection_safely, state, vault_root)

        return TranscriptDeltaResponse(
            call_id=body.call_id, turn_count=len(state.turns), accepted=True
        )

    @app.get("/score", response_model=ScoreResponse)
    def score(
        request: Request,
        call_id: str = Query(..., min_length=1),
        primary_win: str | None = Query(default=None),
    ) -> ScoreResponse:
        """Compute the 0-100 post-call score for a finished call. Reads the
        call note + linked atoms from the vault — does NOT touch the live
        call registry. Phase 5 v1 uses the neutral primary-win judge.
        """
        vault: Path = request.app.state.vault_root
        return _compute_score_response(vault=vault, call_id=call_id, primary_win=primary_win)

    @app.post("/score_override", response_model=ScoreOverrideResponse)
    def score_override(
        body: ScoreOverrideRequest,
        request: Request,
    ) -> ScoreOverrideResponse:
        """Record the user's corrected score for a call. Re-computes the
        predicted score + features at write time so the correction log has
        everything `retrain` needs (no later schema migration to merge
        features that were computed in the past)."""
        vault: Path = request.app.state.vault_root
        predicted = _compute_score_response(
            vault=vault,
            call_id=body.call_id,
            primary_win=body.primary_win,
        )
        feature_values = {c.name: c.raw_value for c in predicted.contributions}
        correction = ScoreCorrection.make(
            call_id=body.call_id,
            call_type=predicted.call_type,
            predicted_score=predicted.score,
            user_score=body.user_score,
            features=feature_values,
            bias=predicted.bias,
        )
        append_correction(vault_root=vault, correction=correction)
        from consultant_brain.scoring.corrections import load_corrections
        total = len(load_corrections(vault))
        return ScoreOverrideResponse(
            call_id=body.call_id,
            predicted_score=predicted.score,
            user_score=body.user_score,
            delta=body.user_score - predicted.score,
            corrections_count=total,
        )

    @app.get("/suggestions", response_model=SuggestionsResponse)
    def suggestions(
        request: Request,
        call_id: str = Query(..., min_length=1),
        hot: int = Query(1, ge=0, le=10),
        warm: int = Query(2, ge=0, le=10),
        cold: int = Query(1, ge=0, le=10),
    ) -> SuggestionsResponse:
        registry: LiveCallRegistry = request.app.state.registry
        vault: Path = request.app.state.vault_root
        state = registry.get(call_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"No active call with id={call_id!r}. POST /call_start first.",
            )
        window = state.transcript_window()
        if not window:
            return SuggestionsResponse(
                call_id=call_id,
                suggestions=[],
                window_chars=0,
                generated_at=datetime.now(timezone.utc),
            )

        result = retrieve(
            transcript_window=window,
            client=state.client,
            call_type=state.call_type,
            vault_root=vault,
        )
        panel = result.top_for_panel(hot=hot, warm=warm, cold=cold)
        return SuggestionsResponse(
            call_id=call_id,
            suggestions=[SuggestionDTO.from_ranked_hit(rh) for rh in panel],
            window_chars=len(window),
            generated_at=datetime.now(timezone.utc),
        )
