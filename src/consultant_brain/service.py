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

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.retrieve import RankedHit, retrieve
from consultant_brain.schemas import CallType, Speaker


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

    _register_routes(app)
    return app


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
        registry: LiveCallRegistry = Depends(get_registry),
    ) -> TranscriptDeltaResponse:
        state = registry.get(body.call_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"No active call with id={body.call_id!r}. POST /call_start first.",
            )
        state.append_turn(body.speaker, body.text)
        return TranscriptDeltaResponse(
            call_id=body.call_id, turn_count=len(state.turns), accepted=True
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
