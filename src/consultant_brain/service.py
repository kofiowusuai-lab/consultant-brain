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

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field

from consultant_brain.briefs import (
    ChatTurn as _ChatTurn,
    ClientBrief,
    ClientBriefAnswer,
    ask_about_client as _ask_about_client,
    generate_client_brief as _generate_client_brief,
    load_cached_brief as _load_cached_brief,
)
from consultant_brain.context_dumps import (
    ContextDumpPreview,
    ContextDumpPreviewStore,
    commit_preview as _commit_dump_preview,
    run_preview as _run_dump_preview,
)
from consultant_brain.crm.resolver import CRMResolver
from consultant_brain.live_call_finalizer import finalize_live_call
from consultant_brain.live_loop import maybe_run_moment_detection
from consultant_brain.live_state import LiveCallRegistry
from consultant_brain.llm.provider import LLMProvider, ProviderError
from consultant_brain.llm.registry import build_provider
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
from consultant_brain.evaluation.suggestion_log import (
    log_emit as _log_suggestion_emit,
    log_referenced as _log_suggestion_referenced,
)


DEFAULT_VAULT = Path.home() / "ConsultantBrain"


# ────────────────────────────────────────────────────────────────────────────
# Request / response models
# ────────────────────────────────────────────────────────────────────────────


class CallStartRequest(BaseModel):
    call_id: str = Field(min_length=1)
    client: str | None = Field(default=None)
    call_type: CallType = Field(default=CallType.consulting_call)
    # Phase 8 item 1: Swift sends the CRM organization UUID so atoms
    # written during this call carry a stable client identifier even if
    # the display name is later renamed. Optional — falls back to
    # name-based resolution when missing.
    org_id: str | None = Field(default=None)


class CallStartResponse(BaseModel):
    call_id: str
    client: str | None
    call_type: CallType
    started_at: datetime
    org_id: str | None = None


class CallEndRequest(BaseModel):
    call_id: str = Field(min_length=1)


class CallEndResponse(BaseModel):
    call_id: str
    ended: bool  # False if the call_id wasn't active (caller might have already ended it)
    # Phase 5.2: when the call was active + had any state, /call_end now
    # writes a CallNote markdown file so the post-call /score endpoint
    # can read it immediately. The Swift override sheet uses this ID.
    call_note_id: str | None = None
    atom_count: int = 0


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


class SuggestionReferencedRequest(BaseModel):
    """Phase 7: Swift posts this when the consultant acts on (says, paraphrases,
    references) a suggestion. Powers the acceptance-rate metric."""

    call_id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)


class SuggestionReferencedResponse(BaseModel):
    call_id: str
    atom_id: str
    logged: bool


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

    # Phase 8 item 1 + 5: shared CRM resolver + post-call extractor provider.
    # Both are lazy — built on first use, cached on app.state so tests can
    # inject doubles via create_app's kwargs in the future.
    app.state.crm_resolver = CRMResolver()
    app.state.crm_org_id_for_call = {}
    app.state.post_call_extraction_enabled = os.environ.get(
        "CONSULTANT_BRAIN_POST_CALL_EXTRACT", ""
    ).lower() in ("1", "true", "yes", "on")
    app.state.extractor_provider_cache = None

    # Phase 10: per-client context-dump previews. In-memory store keyed
    # by ULID; the upload endpoint pushes, the commit endpoint pops.
    # TTL configurable via env so tests can dial it down for eviction
    # checks without sleeping 30 min.
    _ttl = int(os.environ.get("CONSULTANT_BRAIN_CONTEXT_DUMP_TTL_SECONDS", "1800"))
    app.state.context_dump_previews = ContextDumpPreviewStore(ttl_seconds=_ttl)

    _register_routes(app)
    return app


def _preview_to_dto(preview: ContextDumpPreview) -> dict:
    """Serialize a ContextDumpPreview into the JSON the Swift client
    (or curl test) consumes. Atoms are exposed as plain dicts so the
    UI can render + toggle them without needing the full ExtractedAtom
    schema on the client side."""
    return {
        "preview_id": preview.preview_id,
        "client_name": preview.client_name,
        "client_slug": preview.client_slug,
        "observed_at": preview.observed_at.isoformat(),
        "uploaded_at": preview.uploaded_at.replace(microsecond=0).isoformat() + "Z",
        "source_filename": preview.source_filename,
        "source_kind_label": preview.source_kind_label,
        "summary": preview.summary,
        "raw_text_excerpt": preview.raw_text_excerpt,
        "warnings": list(preview.warnings),
        "notes": preview.notes,
        "atoms": [
            {
                "index": index,
                "type": atom.type.value,
                "body": atom.body,
                "confidence": atom.confidence,
                "tags": list(atom.tags),
            }
            for index, atom in enumerate(preview.atoms)
        ],
    }


def _get_extractor_provider(app) -> LLMProvider | None:
    """Lazy-build the extractor provider — cached on the app once it
    succeeds. Returns None if construction fails (missing API key) so
    /call_end degrades to the placeholder summary instead of crashing.
    """
    cached = app.state.extractor_provider_cache
    if cached is not None:
        return cached
    try:
        provider = build_provider()
    except ProviderError:
        return None
    app.state.extractor_provider_cache = provider
    return provider


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
        request: Request,
        registry: LiveCallRegistry = Depends(get_registry),
    ) -> CallStartResponse:
        state = registry.start(call_id=body.call_id, client=body.client, call_type=body.call_type)

        # Resolve the client → org UUID. Swift either sends it explicitly
        # (the fast path — already in the CRM) or we resolve it here from
        # the display name via CRMResolver. Cache the result on app.state
        # so /call_end can stamp newly-extracted atoms with it.
        from uuid import UUID
        org_uuid: UUID | None = None
        if body.org_id:
            try:
                org_uuid = UUID(body.org_id)
            except ValueError:
                org_uuid = None
        if org_uuid is None and body.client:
            try:
                org_uuid = request.app.state.crm_resolver.resolve_uuid(body.client)
            except Exception:
                org_uuid = None
        if org_uuid is not None:
            request.app.state.crm_org_id_for_call[body.call_id] = org_uuid

        return CallStartResponse(
            call_id=state.call_id,
            client=state.client,
            call_type=state.call_type,
            started_at=state.started_at,
            org_id=str(org_uuid) if org_uuid else None,
        )

    @app.post("/call_end", response_model=CallEndResponse)
    def call_end(
        body: CallEndRequest,
        request: Request,
        registry: LiveCallRegistry = Depends(get_registry),
    ) -> CallEndResponse:
        popped = registry.end(body.call_id)
        if popped is None:
            return CallEndResponse(call_id=body.call_id, ended=False)

        # Finalize: write a CallNote on disk so /score can read it without
        # waiting for the post-call Claude ingestion. Live-detected atoms
        # were already linked to this call_note_id by the Phase 4 loop.
        # Phase 8 item 5: when post-call extraction is enabled and we can
        # build an LLM provider, run the full extractor for a real summary.
        provider: LLMProvider | None = None
        if request.app.state.post_call_extraction_enabled:
            provider = _get_extractor_provider(request.app)
        org_uuid = request.app.state.crm_org_id_for_call.pop(body.call_id, None)
        try:
            result = finalize_live_call(
                state=popped,
                vault_root=request.app.state.vault_root,
                provider=provider,
                client_org_id=org_uuid,
            )
        except Exception as exc:  # noqa: BLE001 — never let finalize crash /call_end
            import logging
            logging.getLogger(__name__).exception("finalize_live_call failed: %s", exc)
            return CallEndResponse(call_id=body.call_id, ended=True)

        return CallEndResponse(
            call_id=body.call_id,
            ended=True,
            call_note_id=result.call_note_id,
            atom_count=result.atom_count,
        )

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
        knowledge: int = Query(0, ge=0, le=10),
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
            include_knowledge=knowledge > 0,
        )
        panel = result.top_for_panel(hot=hot, warm=warm, cold=cold, knowledge=knowledge)
        suggestions = [SuggestionDTO.from_ranked_hit(rh) for rh in panel]
        # Phase 7: log each emit so acceptance-rate can match against
        # POST /suggestion_referenced events later. Errors here never
        # propagate — the metric is non-critical.
        for s in suggestions:
            try:
                _log_suggestion_emit(
                    vault_root=vault,
                    call_id=call_id,
                    atom_id=s.atom_id,
                    layer=s.layer,
                    score=s.score,
                )
            except Exception:  # noqa: BLE001
                pass
        return SuggestionsResponse(
            call_id=call_id,
            suggestions=suggestions,
            window_chars=len(window),
            generated_at=datetime.now(timezone.utc),
        )

    # ──────────────────────────────────────────────────────────────────
    # Phase 10 — per-client context dumps (PDF / DOCX / image / audio / ZIP)
    # ──────────────────────────────────────────────────────────────────

    @app.post("/context_dump")
    async def context_dump_upload(
        request: Request,
        file: UploadFile = File(...),
        client_name: str = Form(...),
        observed_at: str = Form(...),
        notes: str | None = Form(default=None),
        extractor_provider: str | None = Form(default=None),
    ) -> dict:
        """Phase 10: upload one file → preview (no vault write yet).

        Returns a `ContextDumpPreviewDTO` with the proposed atoms +
        warnings the user reviews before committing. The user POSTs
        `/context_dump/commit` with the preview_id to actually write.
        """
        from datetime import date as _date

        import tempfile

        if not client_name.strip():
            raise HTTPException(status_code=400, detail="`client_name` is required")
        try:
            observed_at_date = _date.fromisoformat(observed_at.strip())
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"`observed_at` must be YYYY-MM-DD: {exc}",
            ) from exc

        # Pick provider (LLMProvider if configured, otherwise the legacy
        # AnthropicClient path so the call works with just the Anthropic
        # key in secrets.json).
        provider = None
        legacy_client = None
        if extractor_provider and extractor_provider != "anthropic":
            try:
                provider = build_provider(extractor_provider)
            except ProviderError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            try:
                provider = _get_extractor_provider(request.app)
            except Exception:
                provider = None
            if provider is None:
                from consultant_brain.extractor import real_anthropic_client
                from consultant_brain.secrets import (
                    SecretNotFoundError,
                    get_anthropic_key,
                )

                try:
                    legacy_client = real_anthropic_client(get_anthropic_key())
                except SecretNotFoundError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc

        # Spool the upload to a tmp file the parser can mmap; cleaned
        # up by the OS once we exit the context.
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(
            prefix="brain-dump-",
            suffix=suffix,
            delete=False,
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            preview = _run_dump_preview(
                file_path=tmp_path,
                client_name=client_name.strip(),
                observed_at=observed_at_date,
                notes=notes.strip() if notes else None,
                vault_root=request.app.state.vault_root,
                provider=provider,
                client=legacy_client,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        request.app.state.context_dump_previews.put(preview)
        return _preview_to_dto(preview)

    @app.get("/context_dump/{preview_id}")
    def context_dump_get(preview_id: str, request: Request) -> dict:
        """Fetch a stashed preview by id. Returns 404 if it's gone
        (committed, discarded, or evicted by TTL)."""
        preview = request.app.state.context_dump_previews.get(preview_id)
        if preview is None:
            raise HTTPException(status_code=404, detail="Preview not found or expired")
        return _preview_to_dto(preview)

    @app.post("/context_dump/commit")
    def context_dump_commit(body: dict, request: Request) -> dict:
        """Commit a preview to the vault. Body:

          { "preview_id": "01HX...", "accepted_atom_indexes": [0, 2, 4] }

        `accepted_atom_indexes` is optional; omit to commit every atom
        in the preview."""
        preview_id = (body or {}).get("preview_id")
        if not isinstance(preview_id, str) or not preview_id.strip():
            raise HTTPException(status_code=400, detail="`preview_id` is required")

        store = request.app.state.context_dump_previews
        preview = store.pop(preview_id)
        if preview is None:
            raise HTTPException(status_code=404, detail="Preview not found or expired")

        accepted = body.get("accepted_atom_indexes")
        if accepted is not None and not isinstance(accepted, list):
            raise HTTPException(
                status_code=400,
                detail="`accepted_atom_indexes` must be a list of ints.",
            )

        try:
            result = _commit_dump_preview(
                preview=preview,
                vault_root=request.app.state.vault_root,
                accepted_atom_indexes=accepted,
            )
        except Exception as exc:  # noqa: BLE001
            # Re-stash the preview so the user can retry without
            # re-uploading.
            store.put(preview)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return {
            "preview_id": result.preview_id,
            "dump_note_path": str(result.dump_note_path),
            "atom_count": result.atom_count,
            "client_slug": result.client_slug,
        }

    @app.delete("/context_dump/{preview_id}")
    def context_dump_discard(preview_id: str, request: Request) -> dict:
        """Drop a preview without committing — cleans up its tmp file.
        Useful when the user reviews the proposed atoms and decides
        not to keep them."""
        store = request.app.state.context_dump_previews
        dropped = store.discard(preview_id)
        return {"preview_id": preview_id, "discarded": dropped}

    # ──────────────────────────────────────────────────────────────────
    # Per-client AI brief — structured summary the dashboard renders
    # under each client's Context section. Reads atoms + call notes +
    # context dumps for one client, sends to the configured LLM,
    # returns named sections (summary / key_facts / open_commitments /
    # open_objections / recent_moves / next_steps / meeting_prep).
    # ──────────────────────────────────────────────────────────────────

    @app.get("/client_brief")
    def client_brief_endpoint(
        request: Request,
        client: str = Query(..., min_length=1),
        refresh: bool = Query(default=False),
    ) -> dict:
        """Return a per-client brief. When `refresh=true` OR no cached
        brief exists, run the LLM. Otherwise return the cached version
        (instant). The Swift UI sets `refresh=true` from the refresh
        button + on first open of a client whose atom count has
        changed since the cached brief.
        """
        vault: Path = request.app.state.vault_root

        # Cache hit on default open — keeps the dashboard snappy.
        if not refresh:
            cached = _load_cached_brief(client_name=client, vault_root=vault)
            if cached is not None:
                return cached.to_dict()

        # Cache miss or explicit refresh — build a new brief.
        provider = None
        try:
            provider = _get_extractor_provider(request.app)
        except Exception:
            provider = None

        legacy_client = None
        if provider is None:
            from consultant_brain.extractor import real_anthropic_client
            from consultant_brain.secrets import (
                SecretNotFoundError,
                get_anthropic_key,
            )

            try:
                legacy_client = real_anthropic_client(get_anthropic_key())
            except SecretNotFoundError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        try:
            brief = _generate_client_brief(
                client_name=client,
                vault_root=vault,
                provider=provider,
                client=legacy_client,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return brief.to_dict()

    @app.post("/client_brief/ask")
    def client_brief_ask_endpoint(body: dict, request: Request) -> dict:
        """Answer one follow-up question about a client.

        Body shape:
          {
            "client_name": "Reece",
            "question": "What was the deadline he mentioned?",
            "history": [
              { "role": "user", "content": "..." },
              { "role": "assistant", "content": "..." }
            ]
          }

        The history field is optional. The brain caps it at 6 most-
        recent turns server-side so long threads don't blow the
        prompt budget.
        """
        client_name = (body or {}).get("client_name") or (body or {}).get("client")
        question = (body or {}).get("question")
        if not isinstance(client_name, str) or not client_name.strip():
            raise HTTPException(status_code=400, detail="`client_name` is required")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=400, detail="`question` is required")

        history_raw = body.get("history") or []
        if not isinstance(history_raw, list):
            raise HTTPException(status_code=400, detail="`history` must be a list")
        history: list[_ChatTurn] = []
        for entry in history_raw:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role", "")).strip().lower()
            content = str(entry.get("content", "")).strip()
            if role not in ("user", "assistant") or not content:
                continue
            history.append(_ChatTurn(role=role, content=content))

        # Provider selection mirrors /client_brief.
        provider = None
        try:
            provider = _get_extractor_provider(request.app)
        except Exception:
            provider = None

        legacy_client = None
        if provider is None:
            from consultant_brain.extractor import real_anthropic_client
            from consultant_brain.secrets import (
                SecretNotFoundError,
                get_anthropic_key,
            )

            try:
                legacy_client = real_anthropic_client(get_anthropic_key())
            except SecretNotFoundError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        vault: Path = request.app.state.vault_root
        try:
            answer: ClientBriefAnswer = _ask_about_client(
                client_name=client_name.strip(),
                question=question.strip(),
                history=history,
                vault_root=vault,
                provider=provider,
                client=legacy_client,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return {
            "answer": answer.answer,
            "atoms_consulted": answer.atoms_consulted,
            "calls_consulted": answer.calls_consulted,
            "dumps_consulted": answer.dumps_consulted,
            "model_used": answer.model_used,
        }

    # ──────────────────────────────────────────────────────────────────
    # Phase 9 — learn from external sources (YouTube / Instagram)
    # ──────────────────────────────────────────────────────────────────

    @app.post("/learn")
    def learn_endpoint(body: dict, request: Request) -> dict:
        """Ingest a YouTube or Instagram URL into the knowledge layer.

        Body: { "url": "...", "topic": "ai-sales", "for_client": "Reece",
                "allow_whisper": true, "cookies_from_browser": "chrome",
                "extractor_provider": "anthropic" }
        Only `url` is required.

        Returns a summary payload. Fire-and-forget from the Swift app
        is fine — the long-running fetch happens inline so the caller
        can choose to poll or wait.
        """
        url = (body or {}).get("url")
        if not isinstance(url, str) or not url.strip():
            raise HTTPException(status_code=400, detail="`url` is required")

        from consultant_brain.learn import run_learn

        vault: Path = request.app.state.vault_root
        try:
            result = run_learn(
                url=url,
                vault_root=vault,
                topic=body.get("topic"),
                for_client=body.get("for_client"),
                allow_whisper=bool(body.get("allow_whisper", False)),
                cookies_from_browser=body.get("cookies_from_browser"),
                extractor_provider_name=body.get("extractor_provider"),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return {
            "source_id": result.source_id,
            "source_url": result.source_url,
            "source_kind": result.source_kind.value,
            "atom_count": result.atom_count,
            "summary": result.summary,
            "knowledge_note": str(result.knowledge_note_path) if result.knowledge_note_path else None,
        }

    @app.post("/suggestion_referenced", response_model=SuggestionReferencedResponse)
    def suggestion_referenced(
        body: SuggestionReferencedRequest,
        request: Request,
    ) -> SuggestionReferencedResponse:
        """Swift app POSTs here when the consultant acts on a brain
        suggestion (says it / paraphrases it / clicks "got it" in the
        overlay). Feeds the acceptance-rate metric."""
        vault: Path = request.app.state.vault_root
        try:
            _log_suggestion_referenced(
                vault_root=vault,
                call_id=body.call_id,
                atom_id=body.atom_id,
            )
        except Exception:
            return SuggestionReferencedResponse(call_id=body.call_id, atom_id=body.atom_id, logged=False)
        return SuggestionReferencedResponse(call_id=body.call_id, atom_id=body.atom_id, logged=True)

    # ──────────────────────────────────────────────────────────────────
    # Phase 8 item 12 + 14 — metrics + diagnostics
    # ──────────────────────────────────────────────────────────────────

    @app.get("/metrics")
    def metrics_endpoint(request: Request) -> dict:
        """Return the Phase 7 dashboard as JSON.

        Wraps `compute_metrics()` so the Swift Brain Status window can
        render the same numbers the CLI prints. Missing data points stay
        None — the dashboard renders them as `—`.
        """
        from consultant_brain.evaluation.metrics import compute_metrics

        vault: Path = request.app.state.vault_root
        report = compute_metrics(vault_root=vault)
        return report.to_dict()

    @app.get("/diagnostics")
    def diagnostics_endpoint(request: Request) -> dict:
        """Deep readiness check — checks every dependency the brain
        relies on. Returns a structured response the Swift health
        indicator parses to pick green/yellow/red.

        - vault_root exists + is writable
        - LanceDB index path exists (or can be created)
        - Ollama is reachable on localhost (warning, not error)
        - LLM provider's API key is present
        """
        import os

        from consultant_brain.embedder import LanceVaultIndex
        from consultant_brain.secrets import has_key

        vault: Path = request.app.state.vault_root
        checks: dict[str, dict] = {}

        # Vault path
        checks["vault"] = {
            "ok": vault.exists() and vault.is_dir() and os.access(vault, os.W_OK),
            "path": str(vault),
        }

        # LanceDB index
        try:
            from consultant_brain.vault import VaultLayout
            layout = VaultLayout.for_root(vault)
            LanceVaultIndex(layout)  # builds dir if missing
            checks["lancedb"] = {"ok": True}
        except Exception as exc:
            checks["lancedb"] = {"ok": False, "error": str(exc)[:200]}

        # Ollama
        ollama_ok = _ping_ollama()
        checks["ollama"] = ollama_ok

        # LLM provider key
        provider_name = (
            os.environ.get("BRAIN_LLM_PROVIDER", "").strip().lower() or "anthropic"
        )
        key_account = {
            "anthropic": ("ANTHROPIC_API_KEY", "anthropic-api-key"),
            "openai": ("OPENAI_API_KEY", "openai-api-key"),
            "openrouter": ("OPENROUTER_API_KEY", "openrouter-api-key"),
            "deepseek": ("DEEPSEEK_API_KEY", "deepseek-api-key"),
            "kimi": ("KIMI_API_KEY", "kimi-api-key"),
        }
        env_var, secrets_acct = key_account.get(
            provider_name, ("ANTHROPIC_API_KEY", "anthropic-api-key")
        )
        checks["llm_provider"] = {
            "name": provider_name,
            "key_present": has_key(env_var=env_var, secrets_account=secrets_acct),
        }

        # Aggregate status: green = all green, yellow = ollama or llm_provider
        # missing (degraded but usable), red = vault or lancedb broken.
        status = "green"
        if not checks["vault"]["ok"] or not checks["lancedb"]["ok"]:
            status = "red"
        elif not checks["ollama"]["ok"] or not checks["llm_provider"]["key_present"]:
            status = "yellow"

        return {
            "status": status,
            "checks": checks,
            "vault_root": str(vault),
            "active_calls": len(request.app.state.registry),
        }


def _ping_ollama() -> dict:
    """Best-effort liveness probe for Ollama. Two-second timeout so a
    dead/missing Ollama doesn't slow the diagnostics endpoint."""
    try:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code != 200:
                return {"ok": False, "status_code": response.status_code}
            data = response.json()
            tags = [m.get("name", "") for m in data.get("models", [])]
            has_embed_model = any("nomic-embed-text" in tag for tag in tags)
            return {
                "ok": True,
                "has_nomic_embed_text": has_embed_model,
                "models": tags[:20],
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
