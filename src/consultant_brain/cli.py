"""Typer entry point for `consultant-brain`.

Two subcommands shipped in Phase 1:
- `ingest`: parse a Sessions/*.json from the Swift app, extract atoms via
  Claude, write the atoms + a call note to the Obsidian vault, embed each
  atom into LanceDB.
- `query`: vector search the LanceDB index for the top-N atoms by semantic
  similarity to a query string.

Later phases extend the CLI surface; this file is intentionally thin so
each subcommand body delegates to a dedicated module.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Consultant Copilot Active Brain — Obsidian-backed atom graph.",
)


DEFAULT_VAULT = Path.home() / "ConsultantBrain"


@app.command()
def ingest(
    session: Path = typer.Option(
        ...,
        "--session",
        "-s",
        help="Path to a Sessions/*.json file from the Swift app.",
        exists=True,
        readable=True,
    ),
    client: str = typer.Option(
        ...,
        "--client",
        "-c",
        help="Client display name (becomes an Obsidian wikilink target).",
    ),
    call_type: str = typer.Option(
        ...,
        "--call-type",
        "-t",
        help="One of: consultingCall | aiTraining | coldCall | closingCall | followUp | implementation",
    ),
    vault: Path = typer.Option(
        DEFAULT_VAULT,
        "--vault",
        "-v",
        help="Vault root directory.",
    ),
    redact: bool = typer.Option(
        False,
        "--redact",
        help="Replace client names with [CLIENT_1] etc. before sending the transcript to Claude.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Extract + validate atoms but do not write to disk.",
    ),
) -> None:
    """Ingest a session JSON into the vault as a graph of typed atoms."""
    # Lazy import — heavy modules (anthropic, lancedb) shouldn't slow `--help`.
    from consultant_brain.ingest import run_ingest

    result = run_ingest(
        session_path=session,
        client_name=client,
        call_type=call_type,
        vault_root=vault,
        redact=redact,
        dry_run=dry_run,
    )
    typer.echo(result.summary_line())


@app.command()
def query(
    text: str = typer.Argument(..., help="Semantic search query."),
    vault: Path = typer.Option(
        DEFAULT_VAULT,
        "--vault",
        "-v",
        help="Vault root directory.",
    ),
    top: int = typer.Option(5, "--top", "-n", help="Number of results to return."),
) -> None:
    """Vector-search the atom index. Returns the top-N atoms by similarity."""
    from consultant_brain.query import run_query

    hits = run_query(query_text=text, vault_root=vault, top_n=top)
    if not hits:
        typer.echo("No atoms found. Has anything been ingested yet?")
        raise typer.Exit(code=1)
    for hit in hits:
        typer.echo(hit.format_line())


if __name__ == "__main__":
    app()
