"""Smoke tests for the MCP server. Verifies the tool registry shape
+ that key tools dispatch correctly against a seeded vault."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from consultant_brain.mcp_server.server import TOOLS, _dispatch
from consultant_brain.vault import VaultLayout, ensure_vault_skeleton


def _seed_vault(tmp_path: Path) -> Path:
    """One Reece atom + one call note + one context dump folder so
    the list_clients tool has real counts to return."""
    vault = tmp_path / "vault"
    layout = VaultLayout.for_root(vault)
    ensure_vault_skeleton(layout)
    atom = {
        "id": "01HX0000000000000000000001",
        "type": "client_fact",
        "client": "[[Reece]]",
        "call": "[[2026-05-12_reece_consultingCall]]",
        "call_type": "consultingCall",
        "tags": ["scoping"],
        "confidence": 0.9,
        "evidence_count": 1,
        "last_seen": "2026-05-12",
        "created_at": "2026-05-12T19:42:10Z",
        "status": "active",
        "embedding_id": "01HX0000000000000000000001",
    }
    (layout.atoms_dir / "01HX0000000000000000000001.md").write_text(
        "---\n" + yaml.safe_dump(atom, sort_keys=False).rstrip() + "\n---\n\nReece sent his scoping answers.",
        encoding="utf-8",
    )
    call_meta = {
        "id": "2026-05-12_reece_consultingCall",
        "client": "[[Reece]]",
        "call_type": "consultingCall",
        "date": "2026-05-12",
        "duration_minutes": 47,
        "source_session": "session-2026-05-12.json",
        "extractor_model": "claude-opus-4-7",
        "extractor_version": 1,
        "atom_count": 1,
        "created_at": "2026-05-12T19:42:10Z",
        "summary": "Working session.",
    }
    (layout.calls_dir / "2026-05-12_reece_consultingCall.md").write_text(
        "---\n" + yaml.safe_dump(call_meta, sort_keys=False).rstrip() + "\n---\n\n## Summary\nWorking session.\n",
        encoding="utf-8",
    )
    return vault


def test_tool_registry_shape() -> None:
    """Every tool has a name, description, and JSON-schema input."""
    seen_names: set[str] = set()
    for tool in TOOLS:
        assert tool.name, "every tool needs a name"
        assert tool.name not in seen_names, f"duplicate tool name: {tool.name}"
        seen_names.add(tool.name)
        assert tool.description and len(tool.description) > 20, f"{tool.name}: description too short"
        assert isinstance(tool.inputSchema, dict)
        assert tool.inputSchema.get("type") == "object"


def test_list_clients_finds_seeded_data(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    state = {"vault_root": vault}
    raw = asyncio.run(_dispatch("list_clients", {}, state))
    clients = json.loads(raw)
    assert len(clients) == 1
    entry = clients[0]
    assert entry["display_name"] == "Reece"
    assert entry["slug"] == "reece"
    assert entry["atom_count"] == 1
    assert entry["call_count"] == 1


def test_list_recent_calls_filters_by_client(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    state = {"vault_root": vault}
    rows = json.loads(
        asyncio.run(_dispatch("list_recent_calls", {"client": "Reece", "limit": 5}, state))
    )
    assert len(rows) == 1
    assert rows[0]["id"] == "2026-05-12_reece_consultingCall"


def test_get_call_returns_frontmatter_and_body(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    state = {"vault_root": vault}
    payload = json.loads(
        asyncio.run(
            _dispatch("get_call", {"call_id": "2026-05-12_reece_consultingCall"}, state)
        )
    )
    assert payload["id"] == "2026-05-12_reece_consultingCall"
    assert "Working session" in payload["body"]


def test_get_atom_returns_typed_payload(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    state = {"vault_root": vault}
    payload = json.loads(
        asyncio.run(_dispatch("get_atom", {"atom_id": "01HX0000000000000000000001"}, state))
    )
    assert payload["type"] == "client_fact"
    assert payload["client"] == "Reece"
    assert "scoping answers" in payload["body"]


def test_get_atom_missing_returns_error(tmp_path: Path) -> None:
    vault = _seed_vault(tmp_path)
    state = {"vault_root": vault}
    payload = json.loads(
        asyncio.run(_dispatch("get_atom", {"atom_id": "DOES_NOT_EXIST"}, state))
    )
    assert "error" in payload


def test_unknown_tool_does_not_crash(tmp_path: Path) -> None:
    state = {"vault_root": tmp_path / "vault"}
    response = asyncio.run(_dispatch("not_a_real_tool", {}, state))
    assert "Unknown tool" in response


def test_vault_diagnostics_runs(tmp_path: Path) -> None:
    state = {"vault_root": _seed_vault(tmp_path)}
    payload = json.loads(asyncio.run(_dispatch("vault_diagnostics", {}, state)))
    assert payload["status"] in ("green", "yellow", "red")
    assert "checks" in payload
    assert "vault" in payload["checks"]
