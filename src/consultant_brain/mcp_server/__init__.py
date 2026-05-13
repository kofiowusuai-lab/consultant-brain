"""MCP-server harness — exposes every brain capability as
Model-Context-Protocol tools so a different agent runtime (Hermes,
Claude Desktop, Cursor, any MCP-compatible client) can attach
remotely and search the vault, get briefs, ask follow-up questions,
add notes.

Mounts at `/mcp/sse` + `/mcp/messages` inside the existing FastAPI
service — one process, one port, one cloudflared tunnel.
"""

from __future__ import annotations

from consultant_brain.mcp_server.server import build_mcp_starlette_app

__all__ = ["build_mcp_starlette_app"]
