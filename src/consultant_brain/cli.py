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


@app.command()
def suggest(
    window: str = typer.Option(
        ...,
        "--window",
        "-w",
        help="Live transcript window (last 60-90s of conversation). The model retrieves atoms semantically relevant to this.",
    ),
    client: str = typer.Option(
        None,
        "--client",
        "-c",
        help="Client display name (enables the hot layer; omit for cross-client warm-only retrieval).",
    ),
    call_type: str = typer.Option(
        "consultingCall",
        "--call-type",
        "-t",
        help="One of: consultingCall | aiTraining | coldCall | closingCall | followUp | implementation",
    ),
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Print rank components (similarity / recency / confidence / final score) per atom.",
    ),
    hot_n: int = typer.Option(1, "--hot", help="Max atoms from the hot layer."),
    warm_n: int = typer.Option(2, "--warm", help="Max atoms from the warm layer."),
    cold_n: int = typer.Option(1, "--cold", help="Max atoms from the cold layer."),
) -> None:
    """Run three-layer retrieval against a transcript window and print the
    ranked suggestions the live copilot would surface.
    """
    from consultant_brain.schemas import CallType
    from consultant_brain.retrieve import retrieve

    try:
        call_type_enum = CallType(call_type)
    except ValueError as exc:
        valid = ", ".join(t.value for t in CallType)
        raise typer.BadParameter(f"Unknown call type: {call_type!r}. Valid: {valid}") from exc

    result = retrieve(
        transcript_window=window,
        client=client,
        call_type=call_type_enum,
        vault_root=vault,
    )
    panel = result.top_for_panel(hot=hot_n, warm=warm_n, cold=cold_n)
    if not panel:
        typer.echo("No suggestions — vault empty or transcript window unmatched.")
        raise typer.Exit(code=1)

    for rh in panel:
        line = f"{rh.layer.upper():4}  {rh.hit.format_line()}"
        typer.echo(line)
        if explain:
            typer.echo(
                f"      ↳ sim={rh.similarity:.2f}  rec={rh.recency:.2f}  "
                f"conf={rh.confidence:.2f}  score={rh.score:.3f}  · {rh.reason}"
            )


@app.command()
def reindex(
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-embed every atom in the vault, even ones already in the index.",
    ),
) -> None:
    """Re-embed atoms missing from LanceDB. Safe recovery from index wipes,
    Obsidian-side manual atom adds, or model swaps. Does NOT re-run Claude
    extraction — just rebuilds the vector index from the existing markdown.
    """
    from consultant_brain.reindex import run_reindex

    summary = run_reindex(vault_root=vault, force=force)
    typer.echo(summary.summary_line())


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Defaults to localhost — never expose the brain to the LAN without auth (none in Phase 3).",
    ),
    port: int = typer.Option(
        8787,
        "--port",
        "-p",
        help="TCP port. Default 8787 — staying off 8765 which is shared by other OpenClaw services.",
    ),
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable uvicorn auto-reload on file changes. Dev-only — slower start, restarts on save.",
    ),
) -> None:
    """Launch the FastAPI brain service. The Swift app + Phase 4's live loop
    talk to this over HTTP at http://127.0.0.1:<port>.
    """
    import uvicorn

    from consultant_brain.service import create_app

    if reload:
        # uvicorn.run with --reload needs an import string, not an instance,
        # so the worker process can re-import after a file change.
        import os

        os.environ["CONSULTANT_BRAIN_VAULT"] = str(vault.expanduser().resolve())
        typer.echo(f"Reload mode — vault: {os.environ['CONSULTANT_BRAIN_VAULT']}")
        uvicorn.run(
            "consultant_brain.service:reload_app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        # Build the app once + hand it to uvicorn — fastest startup.
        api = create_app(vault_root=vault)
        typer.echo(f"Consultant Brain ready on http://{host}:{port}  ·  vault: {vault}")
        uvicorn.run(api, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
