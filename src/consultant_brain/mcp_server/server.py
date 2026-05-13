"""MCP server exposing the consultant-brain as tools any MCP-aware
agent (Hermes, Claude Desktop, Cursor, OpenAI Agents SDK, etc.) can
attach to remotely.

The tool surface is intentionally a slim layer over the existing
Python functions and FastAPI endpoints — every tool maps to one
existing capability the dashboard already uses. Two reasons:

  1. Single source of truth. Behavior changes propagate everywhere
     without two implementations to keep in sync.
  2. The MCP server adds zero new business logic. It's a transport
     adapter, nothing more.

Transport
---------
SSE (Server-Sent Events) over HTTP. Mounts at:
  GET  /mcp/sse           — agent opens the event stream here
  POST /mcp/messages/?... — agent posts tool calls back

Auth
----
Optional bearer token via the `CONSULTANT_BRAIN_MCP_API_KEY` env var.
When set, requests without `Authorization: Bearer <key>` return 401.
For local Hermes (same machine) you can leave it unset and rely on
the loopback bind. For cloudflared-exposed remote use, set it.

Tools
-----
list_clients              — every client name in the vault
get_client_brief          — structured AI brief for one client
refresh_client_brief      — force-regenerate the brief
ask_client                — Q&A or note-ingest on a client
note_about_client         — explicit note → atom + brief refresh
search_vault              — semantic search across atoms
get_atom                  — pull one atom by id
list_recent_calls         — recent call notes for a client (or all)
get_call                  — full call note + transcript
list_context_dumps        — dumps for one client
list_patterns             — promoted (type, tag) patterns
list_plays                — cross-client recurring moves
vault_diagnostics         — vault/LanceDB/Ollama/LLM-key health
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import frontmatter as fm
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from consultant_brain.briefs import (
    ChatTurn,
    ask_about_client,
    generate_client_brief,
    load_cached_brief,
    process_chat,
)
from consultant_brain.embedder import LanceVaultIndex
from consultant_brain.extractor import real_anthropic_client
from consultant_brain.llm.provider import LLMProvider, ProviderError
from consultant_brain.llm.registry import build_provider
from consultant_brain.reindex import _atom_from_markdown
from consultant_brain.schemas import Atom
from consultant_brain.secrets import SecretNotFoundError, get_anthropic_key
from consultant_brain.vault import VaultLayout, slugify_client


# ────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ────────────────────────────────────────────────────────────────────────────


TOOLS: list[Tool] = [
    Tool(
        name="list_clients",
        description=(
            "Return every client the consultant brain knows about. Each "
            "entry has the display name + slug + counts (atoms, calls, "
            "dumps). Use this first to discover what clients are "
            "available before calling client-scoped tools."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_client_brief",
        description=(
            "Return the structured AI brief for one client. Reads the "
            "cached brief.json — fast (~50ms). For a fresh LLM-derived "
            "brief use refresh_client_brief instead."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "client": {
                    "type": "string",
                    "description": "Client display name, e.g. 'Reece'.",
                }
            },
            "required": ["client"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="refresh_client_brief",
        description=(
            "Force a fresh LLM pass to regenerate the brief from current "
            "atoms + call notes + context dumps. Takes 10-30s; use only "
            "when the cached brief is stale or you need the latest "
            "synthesis. The new brief is written to disk and returned."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "client": {"type": "string"},
            },
            "required": ["client"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="ask_client",
        description=(
            "Conversational interface for one client. The brain "
            "classifies the message as a question (gets a focused "
            "answer anchored in the atoms) or a note (writes a fact to "
            "the vault and returns the updated brief inline). Pass "
            "prior turns via `history` for multi-turn follow-ups."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "client": {"type": "string"},
                "message": {
                    "type": "string",
                    "description": "Question or declarative note about the client.",
                },
                "history": {
                    "type": "array",
                    "description": "Previous Q&A turns (most recent at the end).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant"]},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["client", "message"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="note_about_client",
        description=(
            "Explicit note path. Writes the text as a context-dump-shaped "
            "atom for the client with today's date, then regenerates the "
            "brief. Use when you know you're recording a fact rather "
            "than asking — skips the classification step ask_client uses."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "client": {"type": "string"},
                "text": {
                    "type": "string",
                    "description": "The fact / observation in plain English.",
                },
            },
            "required": ["client", "text"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="search_vault",
        description=(
            "Semantic search across every atom in the vault. Returns "
            "the top-N matches with their body, type, client, and "
            "similarity score. Optional `client` filter restricts to "
            "one client's atoms."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5},
                "client": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_atom",
        description=(
            "Fetch one atom by its 26-char ID. Returns the full atom "
            "(frontmatter + body) — useful when search_vault returns a "
            "hit and the agent wants the canonical record."
        ),
        inputSchema={
            "type": "object",
            "properties": {"atom_id": {"type": "string"}},
            "required": ["atom_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_recent_calls",
        description=(
            "Recent call notes (date, client, summary). Optional `client` "
            "filter and `limit` (default 10)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "client": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_call",
        description=(
            "Full call note (frontmatter + summary + transcript). "
            "Pass the call_id from list_recent_calls."
        ),
        inputSchema={
            "type": "object",
            "properties": {"call_id": {"type": "string"}},
            "required": ["call_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_context_dumps",
        description=(
            "Off-call context dumps for one client (PDF / DOCX / audio "
            "transcripts / image OCR). Returns (id, observed_at, "
            "source_kind, summary, atom_count) tuples."
        ),
        inputSchema={
            "type": "object",
            "properties": {"client": {"type": "string"}},
            "required": ["client"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_patterns",
        description=(
            "Every promoted (atom_type, primary_tag) pattern in 04_Patterns/. "
            "Use this to see what cross-call rules the brain has learned."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="list_plays",
        description=(
            "Every Play in 05_Plays/ — cross-client recurring moves that "
            "have been observed in ≥3 distinct clients."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="vault_diagnostics",
        description=(
            "Health check matching the dashboard's brain-status pill: "
            "vault path / LanceDB index / Ollama / LLM provider key. "
            "Returns status='green'|'yellow'|'red' plus per-check detail."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ────────────────────────────────────────────────────────────────────────────


def _vault_root(server_state: dict) -> Path:
    """Pull the vault root the parent FastAPI service passed in. Falls
    back to the user's home `~/ConsultantBrain` so a standalone
    invocation of the MCP server works without the FastAPI host."""
    vault = server_state.get("vault_root") if server_state else None
    if isinstance(vault, Path):
        return vault
    return Path.home() / "ConsultantBrain"


def _get_llm_provider() -> tuple[Optional[LLMProvider], Optional[Any]]:
    """Same provider-selection pattern as the FastAPI endpoints — try
    the configured BRAIN_LLM_PROVIDER first, fall back to a plain
    Anthropic client from secrets.json. Returns (provider, client) so
    callers thread whichever is non-None into the brief helpers."""
    try:
        provider = build_provider()
    except ProviderError:
        provider = None
    if provider is not None:
        return provider, None
    try:
        return None, real_anthropic_client(get_anthropic_key())
    except SecretNotFoundError:
        return None, None


async def _call_tool(name: str, arguments: dict, server_state: dict) -> list[TextContent]:
    """Dispatch one tool call. Wraps every implementation in a try/
    except so a bug in one tool doesn't kill the whole MCP session —
    error gets surfaced to the agent as a text response prefixed with
    'Error:' so the agent can decide how to handle it."""
    try:
        result_text = await _dispatch(name, arguments, server_state)
    except Exception as exc:  # noqa: BLE001
        result_text = f"Error in {name}: {exc}"
    return [TextContent(type="text", text=result_text)]


async def _dispatch(name: str, arguments: dict, server_state: dict) -> str:
    vault = _vault_root(server_state)
    layout = VaultLayout.for_root(vault)

    if name == "list_clients":
        return json.dumps(_list_clients(layout), indent=2)

    if name == "get_client_brief":
        client = arguments.get("client", "")
        brief = load_cached_brief(client_name=client, vault_root=vault)
        if brief is None:
            return json.dumps(
                {
                    "client": client,
                    "cached": False,
                    "note": "No cached brief — call refresh_client_brief to generate one.",
                },
                indent=2,
            )
        payload = brief.to_dict()
        payload["cached"] = True
        return json.dumps(payload, indent=2)

    if name == "refresh_client_brief":
        client = arguments.get("client", "")
        provider, legacy_client = _get_llm_provider()
        brief = generate_client_brief(
            client_name=client,
            vault_root=vault,
            provider=provider,
            client=legacy_client,
        )
        return json.dumps(brief.to_dict(), indent=2)

    if name == "ask_client":
        client = arguments.get("client", "")
        message = arguments.get("message", "")
        history_raw = arguments.get("history") or []
        history = [
            ChatTurn(role=str(h.get("role", "")), content=str(h.get("content", "")))
            for h in history_raw
            if isinstance(h, dict)
        ]
        provider, legacy_client = _get_llm_provider()
        result = process_chat(
            client_name=client,
            user_input=message,
            history=history,
            vault_root=vault,
            provider=provider,
            client=legacy_client,
        )
        payload = {
            "intent": result.intent,
            "answer": result.answer,
            "atoms_consulted": result.atoms_consulted,
            "calls_consulted": result.calls_consulted,
            "dumps_consulted": result.dumps_consulted,
            "ingested_atoms": result.ingested_atoms,
            "updated_brief": result.updated_brief.to_dict() if result.updated_brief else None,
        }
        return json.dumps(payload, indent=2)

    if name == "note_about_client":
        # Force the note path by phrasing the input as a declarative
        # statement; process_chat will classify it as a note (no `?`).
        client = arguments.get("client", "")
        text = arguments.get("text", "").strip()
        provider, legacy_client = _get_llm_provider()
        result = process_chat(
            client_name=client,
            user_input=text,
            history=[],
            vault_root=vault,
            provider=provider,
            client=legacy_client,
        )
        return json.dumps(
            {
                "intent": result.intent,
                "ingested_atoms": result.ingested_atoms,
                "answer": result.answer,
                "updated_brief": (
                    result.updated_brief.to_dict() if result.updated_brief else None
                ),
            },
            indent=2,
        )

    if name == "search_vault":
        query = arguments.get("query", "")
        top_n = int(arguments.get("top_n", 5))
        client_filter = arguments.get("client")
        return json.dumps(
            _search_vault(
                layout=layout,
                query=query,
                top_n=top_n,
                client_filter=client_filter,
            ),
            indent=2,
        )

    if name == "get_atom":
        atom_id = arguments.get("atom_id", "")
        return json.dumps(_get_atom(layout, atom_id), indent=2)

    if name == "list_recent_calls":
        return json.dumps(
            _list_recent_calls(
                layout=layout,
                client=arguments.get("client"),
                limit=int(arguments.get("limit", 10)),
            ),
            indent=2,
        )

    if name == "get_call":
        return json.dumps(_get_call(layout, arguments.get("call_id", "")), indent=2)

    if name == "list_context_dumps":
        return json.dumps(
            _list_context_dumps(layout=layout, client=arguments.get("client", "")),
            indent=2,
        )

    if name == "list_patterns":
        return json.dumps(_list_md_dir(layout.root / "04_Patterns"), indent=2)

    if name == "list_plays":
        return json.dumps(_list_md_dir(layout.root / "05_Plays"), indent=2)

    if name == "vault_diagnostics":
        return json.dumps(_diagnostics(layout), indent=2)

    return f"Unknown tool: {name}"


# ────────────────────────────────────────────────────────────────────────────
# Per-tool helpers (kept compact — every one reuses existing brain code)
# ────────────────────────────────────────────────────────────────────────────


def _list_clients(layout: VaultLayout) -> list[dict]:
    """Enumerate clients from the 01_Clients/ subdir plus anyone with
    atoms but no client folder yet."""
    clients_dir = layout.clients_dir
    seen: dict[str, dict] = {}
    if clients_dir.exists():
        for sub in clients_dir.iterdir():
            if not sub.is_dir():
                continue
            seen[sub.name] = {
                "slug": sub.name,
                "display_name": sub.name.replace("_", " ").title(),
                "has_folder": True,
                "atom_count": 0,
                "call_count": 0,
                "dump_count": 0,
            }
    # Scan atoms to fill in counts + discover atom-only clients
    if layout.atoms_dir.exists():
        for path in layout.atoms_dir.glob("*.md"):
            try:
                atom = _atom_from_markdown(path)
            except Exception:
                continue
            if not atom.client:
                continue
            slug = slugify_client(atom.client)
            entry = seen.setdefault(
                slug,
                {
                    "slug": slug,
                    "display_name": atom.client,
                    "has_folder": False,
                    "atom_count": 0,
                    "call_count": 0,
                    "dump_count": 0,
                },
            )
            entry["display_name"] = atom.client  # prefer the actual case
            entry["atom_count"] += 1
    # Scan call notes for call counts
    if layout.calls_dir.exists():
        for path in layout.calls_dir.glob("*.md"):
            try:
                meta = fm.load(path.open("r", encoding="utf-8")).metadata
            except Exception:
                continue
            client_raw = meta.get("client") or ""
            client_clean = (
                str(client_raw).strip().lstrip("[").rstrip("]").strip()
                if isinstance(client_raw, str)
                else ""
            )
            if not client_clean:
                continue
            slug = slugify_client(client_clean)
            entry = seen.setdefault(
                slug,
                {
                    "slug": slug,
                    "display_name": client_clean,
                    "has_folder": False,
                    "atom_count": 0,
                    "call_count": 0,
                    "dump_count": 0,
                },
            )
            entry["call_count"] += 1
    # Context dump counts from 10_ContextDumps/<slug>/
    if layout.context_dumps_dir.exists():
        for slug_dir in layout.context_dumps_dir.iterdir():
            if not slug_dir.is_dir():
                continue
            entry = seen.get(slug_dir.name)
            if entry is None:
                continue
            entry["dump_count"] = sum(1 for _ in slug_dir.glob("*.md"))
    return sorted(seen.values(), key=lambda e: e["display_name"].lower())


def _search_vault(
    *, layout: VaultLayout, query: str, top_n: int, client_filter: Optional[str]
) -> list[dict]:
    if not query.strip():
        return []
    try:
        index = LanceVaultIndex(layout)
    except Exception as exc:
        return [{"error": f"LanceDB unavailable: {exc}"}]
    hits = index.query(
        text=query,
        top_n=top_n,
        client_filter=client_filter,
    )
    return [
        {
            "id": h.id,
            "type": h.type,
            "body": h.body,
            "client": h.client,
            "call": h.call,
            "call_type": h.call_type,
            "confidence": h.confidence,
            "last_seen": h.last_seen,
            "tags": list(h.tags),
            "similarity": h.similarity,
        }
        for h in hits
    ]


def _get_atom(layout: VaultLayout, atom_id: str) -> dict:
    path = layout.atom_file(atom_id)
    if not path.is_file():
        return {"error": f"Atom not found: {atom_id}"}
    try:
        atom = _atom_from_markdown(path)
    except Exception as exc:
        return {"error": f"Could not parse {atom_id}: {exc}"}
    return {
        "id": atom.id,
        "type": atom.type.value,
        "client": atom.client,
        "call": atom.call,
        "call_type": atom.call_type.value,
        "source_kind": atom.source_kind.value,
        "source_url": atom.source_url,
        "source_title": atom.source_title,
        "tags": list(atom.tags),
        "confidence": atom.confidence,
        "last_seen": atom.last_seen.isoformat(),
        "created_at": atom.created_at.isoformat(),
        "status": atom.status.value,
        "body": atom.body,
    }


def _list_recent_calls(
    *, layout: VaultLayout, client: Optional[str], limit: int
) -> list[dict]:
    if not layout.calls_dir.exists():
        return []
    rows: list[dict] = []
    target = client.casefold() if client else None
    for path in layout.calls_dir.glob("*.md"):
        try:
            post = fm.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        meta = post.metadata
        client_raw = meta.get("client") or ""
        client_clean = (
            str(client_raw).strip().lstrip("[").rstrip("]").strip()
            if isinstance(client_raw, str)
            else ""
        )
        if target and client_clean.casefold() != target:
            continue
        rows.append(
            {
                "id": str(meta.get("id", path.stem)),
                "client": client_clean,
                "call_type": str(meta.get("call_type", "")),
                "date": str(meta.get("date", "")),
                "duration_minutes": int(meta.get("duration_minutes", 0) or 0),
                "atom_count": int(meta.get("atom_count", 0) or 0),
                "summary": str(meta.get("summary", "") or "").strip(),
            }
        )
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:limit]


def _get_call(layout: VaultLayout, call_id: str) -> dict:
    path = layout.call_file(call_id)
    if not path.is_file():
        return {"error": f"Call note not found: {call_id}"}
    try:
        post = fm.load(path.open("r", encoding="utf-8"))
    except Exception as exc:
        return {"error": f"Could not parse call: {exc}"}
    return {
        "id": str(post.metadata.get("id", call_id)),
        "frontmatter": dict(post.metadata),
        "body": post.content or "",
    }


def _list_context_dumps(*, layout: VaultLayout, client: str) -> list[dict]:
    if not client:
        return []
    slug = slugify_client(client)
    target = layout.context_dumps_dir / slug
    if not target.exists():
        return []
    rows: list[dict] = []
    for path in target.glob("*.md"):
        try:
            post = fm.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "id": str(post.metadata.get("id", path.stem)),
                "observed_at": str(post.metadata.get("observed_at", "")),
                "source_kind_label": str(post.metadata.get("source_kind_label", "")),
                "source_filename": str(post.metadata.get("source_filename", "")),
                "atom_count": int(post.metadata.get("atom_count", 0) or 0),
                "notes": str(post.metadata.get("notes", "") or ""),
            }
        )
    rows.sort(key=lambda r: r["observed_at"], reverse=True)
    return rows


def _list_md_dir(dir_path: Path) -> list[dict]:
    if not dir_path.exists():
        return []
    rows: list[dict] = []
    for path in dir_path.glob("*.md"):
        try:
            meta = fm.load(path.open("r", encoding="utf-8")).metadata
        except Exception:
            continue
        rows.append({"id": str(meta.get("id", path.stem)), "frontmatter": dict(meta)})
    return rows


def _diagnostics(layout: VaultLayout) -> dict:
    """Mirror of the FastAPI /diagnostics endpoint's logic. Inlined so
    the MCP server stays self-contained — same shape on the wire."""
    from consultant_brain.embedder import LanceVaultIndex as _LDB
    from consultant_brain.secrets import has_key as _has_key

    checks: dict[str, dict] = {}
    checks["vault"] = {"ok": layout.root.exists() and layout.root.is_dir(), "path": str(layout.root)}
    try:
        _LDB(layout)
        checks["lancedb"] = {"ok": True}
    except Exception as exc:
        checks["lancedb"] = {"ok": False, "error": str(exc)[:200]}

    # Ollama probe — short timeout
    try:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            tags = [m.get("name", "") for m in response.json().get("models", [])]
            checks["ollama"] = {
                "ok": response.status_code == 200,
                "has_nomic_embed_text": any("nomic-embed-text" in t for t in tags),
            }
    except Exception as exc:
        checks["ollama"] = {"ok": False, "error": str(exc)[:200]}

    provider_name = os.environ.get("BRAIN_LLM_PROVIDER", "anthropic").lower()
    env_key = f"{provider_name.upper()}_API_KEY"
    secrets_key = f"{provider_name}-api-key"
    checks["llm_provider"] = {
        "name": provider_name,
        "key_present": _has_key(env_var=env_key, secrets_account=secrets_key),
    }
    status = "green"
    if not checks["vault"]["ok"] or not checks["lancedb"]["ok"]:
        status = "red"
    elif not checks["ollama"]["ok"] or not checks["llm_provider"]["key_present"]:
        status = "yellow"
    return {"status": status, "checks": checks, "vault_root": str(layout.root)}


# ────────────────────────────────────────────────────────────────────────────
# Server construction
# ────────────────────────────────────────────────────────────────────────────


def _build_server(vault_root: Path) -> Server:
    """Create an MCP Server instance with the tool list + dispatcher
    wired in. `vault_root` is captured so every call sees the right
    vault even when the brain process is launched with a non-default
    `--vault` flag."""
    server = Server("consultant-brain")
    state = {"vault_root": vault_root}

    @server.list_tools()
    async def _list() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def _call(name: str, arguments: dict) -> list[TextContent]:
        return await _call_tool(name, arguments, state)

    return server


def build_mcp_starlette_app(vault_root: Path) -> Starlette:
    """Return a Starlette ASGI app that the parent FastAPI service can
    mount at `/mcp`. Two routes:
      GET  /mcp/sse           — agent opens the event stream
      POST /mcp/messages/...  — agent posts tool calls back

    Auth is enforced via Starlette middleware reading
    `CONSULTANT_BRAIN_MCP_API_KEY`. Leave unset to disable auth
    (loopback-only use cases); set it for any cloudflared-exposed
    deployment.
    """
    server = _build_server(vault_root)
    transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        if not _auth_ok(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
        return Response()

    async def handle_post_message(request: Request) -> Response:
        if not _auth_ok(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await transport.handle_post_message(request.scope, request.receive, request._send)

    app = Starlette(
        debug=False,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=handle_post_message),
        ],
    )
    return app


def _auth_ok(request: Request) -> bool:
    """Bearer-token check against `CONSULTANT_BRAIN_MCP_API_KEY`. No
    key configured → auth is disabled (intentional, for loopback-only
    use). When set, the header must match exactly."""
    expected = os.environ.get("CONSULTANT_BRAIN_MCP_API_KEY", "").strip()
    if not expected:
        return True
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return False
    return header.split(" ", 1)[1].strip() == expected
