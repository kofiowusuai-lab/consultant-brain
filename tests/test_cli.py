"""Smoke tests for the CLI surface. Verifies Typer wired everything up before
we layer real subcommand logic on top.
"""

from __future__ import annotations

from typer.testing import CliRunner

from consultant_brain.cli import app


runner = CliRunner()


def test_help_lists_both_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "ingest" in result.output
    assert "query" in result.output


def test_ingest_help_shows_required_flags() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--session", "--client", "--call-type", "--vault", "--redact", "--dry-run"):
        assert flag in result.output, f"missing flag: {flag}"


def test_query_help_shows_top_flag() -> None:
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0, result.output
    assert "--top" in result.output
    assert "TEXT" in result.output  # the positional `text` argument
