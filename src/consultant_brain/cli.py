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
def metrics(
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    corpus: Path = typer.Option(
        None,
        "--corpus",
        help="Path to a labeled retrieval-precision corpus (see SCHEMAS.md). When omitted, the precision row reports '—'.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of the text summary."),
) -> None:
    """Phase 7: print the self-evaluation dashboard.

    Four metrics, each only computed when its source data exists:
      - Retrieval precision  (requires --corpus pointing at labeled queries)
      - Suggestion acceptance rate  (requires /suggestions emits + /suggestion_referenced events)
      - Score-prediction MAE  (requires user overrides in score_corrections.jsonl)
      - Pattern stability  (requires ≥2 pattern snapshots written by `distill`)
    """
    from consultant_brain.evaluation.metrics import (
        compute_metrics,
        render_report_json,
        render_report_text,
    )

    report = compute_metrics(vault_root=vault, corpus_path=corpus)
    if json_output:
        typer.echo(render_report_json(report))
    else:
        typer.echo(render_report_text(report))


@app.command()
def score(
    call_id: str = typer.Argument(..., help="Call note ID, e.g. 2026-05-12_reece_consultingCall."),
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    primary_win: str = typer.Option(
        "",
        "--primary-win",
        help="The client's primary_win statement. Drives the primary_win_progress feature when an LLM judge is wired (Phase 5 v1: judge is the neutral 0.5 stub).",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Print per-feature contributions (raw × weight = contribution).",
    ),
    weights_path: Path = typer.Option(
        None,
        "--weights",
        help="Override the weights YAML path. Defaults to <vault>/00_System/scoring_weights.yaml.",
    ),
) -> None:
    """Compute the post-call 0-100 score for one call. Reads the call note +
    its linked atoms from the vault; no Anthropic calls in v1 (the
    primary-win judge uses the neutral 0.5 stub until the LLM judge is wired).
    """
    from consultant_brain.scoring.features import compute_features
    from consultant_brain.scoring.loader import CallNotFoundError, load_call_for_scoring
    from consultant_brain.scoring.score import compute_score
    from consultant_brain.scoring.weights import ScoringWeightsTable

    try:
        inputs = load_call_for_scoring(
            vault_root=vault,
            call_id=call_id,
            primary_win=primary_win or None,
        )
    except CallNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    weights_table_path = weights_path or (vault.expanduser() / "00_System" / "scoring_weights.yaml")
    weights_table = ScoringWeightsTable.load(weights_table_path)
    weights = weights_table.for_call_type(inputs.call_type)

    features = compute_features(
        atoms=inputs.atoms,
        turns=inputs.turns,
        call_type=inputs.call_type,
        primary_win=inputs.primary_win,
    )
    result = compute_score(features=features, weights=weights)

    typer.echo(result.summary_line())
    if explain:
        typer.echo("")
        typer.echo(f"  bias              {weights.bias:>6.1f}")
        for c in result.contributions:
            typer.echo(
                f"  {c.name:<22} {c.raw_value:>5.2f} × {c.weight:>+6.1f} = {c.contribution:>+6.1f}"
            )
        typer.echo(f"  raw score         {result.raw_score:>+6.1f}")
        typer.echo(f"  clamped           {result.score:>6.1f}")


@app.command()
def distill(
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    skip_context: bool = typer.Option(
        False,
        "--skip-context",
        help="Skip the context.md regeneration pass.",
    ),
    skip_patterns: bool = typer.Option(
        False,
        "--skip-patterns",
        help="Skip the pattern miner.",
    ),
    skip_plays: bool = typer.Option(
        False,
        "--skip-plays",
        help="Skip the plays auto-promoter.",
    ),
    skip_people: bool = typer.Option(
        False,
        "--skip-people",
        help="Skip the stakeholder graph build.",
    ),
    skip_tags: bool = typer.Option(
        False,
        "--skip-tags",
        help="Skip tag normalization.",
    ),
    skip_retirement: bool = typer.Option(
        False,
        "--skip-retirement",
        help="Skip atom lifecycle transitions.",
    ),
) -> None:
    """Phase 6+8 distillation pipeline.

    Runs every active distiller unless skipped:
      tags         — fold near-duplicate tags onto canonical forms
      retirement   — flip stale atoms active → needs_review → retired
      patterns     — promote recurring (type, tag) clusters → 04_Patterns/
      plays        — promote cross-client recurring bodies → 05_Plays/
      people       — regenerate 07_People/ stakeholder files from the CRM
      context      — rebuild every client's context.md with pinned-note
                     preservation
    """
    from consultant_brain.distillation.client_context import regenerate_all_clients
    from consultant_brain.distillation.patterns import mine_patterns
    from consultant_brain.distillation.plays import mine_plays
    from consultant_brain.distillation.people import regenerate_people
    from consultant_brain.distillation.retirement import run_retirement
    from consultant_brain.distillation.tags import normalize_tags

    if not skip_tags:
        tag_result = normalize_tags(vault_root=vault)
        if tag_result.atoms_rewritten:
            typer.echo(
                f"Tags: rewrote {tag_result.atoms_rewritten} of "
                f"{tag_result.atoms_scanned} atoms · "
                f"{len(tag_result.tag_replacements)} unique replacements."
            )
        else:
            typer.echo(f"Tags: {tag_result.atoms_scanned} scanned, none needed rewriting.")

    if not skip_retirement:
        ret_result = run_retirement(vault_root=vault)
        typer.echo(
            f"Retirement: scanned {ret_result.atoms_scanned} · "
            f"flagged {ret_result.flagged_needs_review} · "
            f"retired {ret_result.retired} · revived {ret_result.revived}."
        )

    if not skip_patterns:
        pattern_result = mine_patterns(vault_root=vault)
        typer.echo(
            f"Patterns: wrote {pattern_result.patterns_written}, "
            f"updated {pattern_result.patterns_updated}, "
            f"skipped {pattern_result.candidates_skipped} below threshold."
        )
        from consultant_brain.evaluation.pattern_stability import take_snapshot
        take_snapshot(vault_root=vault)

    if not skip_plays:
        play_result = mine_plays(vault_root=vault)
        typer.echo(
            f"Plays: wrote {play_result.plays_written}, "
            f"updated {play_result.plays_updated}, "
            f"skipped {play_result.plays_skipped} below threshold."
        )

    if not skip_people:
        people_result = regenerate_people(vault_root=vault)
        typer.echo(
            f"People: wrote {people_result.people_written}, "
            f"updated {people_result.people_updated} · "
            f"dir={people_result.people_dir}"
        )
        for note in people_result.notes:
            typer.echo(f"  · {note}")

    if not skip_context:
        context_results = regenerate_all_clients(vault_root=vault)
        if context_results:
            for r in context_results:
                pinned_note = " (pinned preserved)" if r.preserved_pinned else ""
                typer.echo(
                    f"Context: {r.client_slug} · {r.atom_count} atoms · "
                    f"{r.call_count} calls{pinned_note}"
                )
        else:
            typer.echo("Context: no clients to regenerate.")


@app.command()
def weekly(
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    week: str | None = typer.Option(
        None,
        "--week",
        "-w",
        help="ISO week id like 2026-W19. Defaults to the current ISO week.",
    ),
) -> None:
    """Generate a weekly review at 08_Reviews/<YYYY-Www>.md."""
    from consultant_brain.distillation.reviews import generate_weekly_review

    result = generate_weekly_review(vault_root=vault, week_iso=week)
    typer.echo(
        f"Weekly review {result.week_id}: {result.atom_count} atoms · "
        f"{result.call_count} calls · {result.client_count} clients · "
        f"{result.path}"
    )


@app.command()
def backup(
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output tarball path. Defaults to ~/Documents/ConsultantBrain-backups/<timestamp>.tar.gz",
    ),
) -> None:
    """Snapshot the vault + LanceDB index to a tarball.

    Output format is .tar.gz (universal) — the index is small enough
    that zstd isn't worth the dependency. Restore with `consultant-brain
    restore --from <tarball> --vault <target>`.
    """
    from consultant_brain.backup import create_backup

    result = create_backup(vault_root=vault, out_path=out)
    typer.echo(
        f"Backup → {result.archive_path} · {result.archive_size_bytes} bytes · "
        f"{result.file_count} files"
    )


@app.command()
def restore(
    archive: Path = typer.Option(
        ...,
        "--from",
        "-f",
        help="Path to a tarball produced by `consultant-brain backup`.",
        exists=True,
        readable=True,
    ),
    vault: Path = typer.Option(
        ...,
        "--vault",
        "-v",
        help="Target vault directory. Must be empty or non-existent.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing non-empty vault directory.",
    ),
) -> None:
    """Restore a vault from a backup tarball."""
    from consultant_brain.backup import restore_backup

    result = restore_backup(archive_path=archive, vault_root=vault, force=force)
    typer.echo(
        f"Restored {result.file_count} files · {result.bytes_written} bytes · "
        f"vault={result.vault_root}"
    )


@app.command()
def export(
    out: Path = typer.Option(..., "--out", "-o", help="Output directory."),
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
    anonymize: bool = typer.Option(
        True,
        "--anonymize/--no-anonymize",
        help="Strip client + person names + emails. Default ON.",
    ),
) -> None:
    """Export the vault to a shareable directory.

    With --anonymize (default), every client display name becomes
    [CLIENT_N], every stakeholder becomes [PERSON_N], every email is
    redacted. Patterns + plays preserve their structure intact so a team
    can share playbooks without leaking client identity.
    """
    from consultant_brain.export import export_vault

    result = export_vault(vault_root=vault, out_dir=out, anonymize=anonymize)
    typer.echo(
        f"Exported {result.atom_count} atoms · {result.call_count} call notes · "
        f"{result.pattern_count} patterns · {result.play_count} plays · "
        f"{result.replacements_applied} identifiers redacted · → {result.out_dir}"
    )


@app.command()
def retrain(
    vault: Path = typer.Option(DEFAULT_VAULT, "--vault", "-v", help="Vault root."),
) -> None:
    """Re-fit the scoring weights from accumulated user corrections.

    Reads vault/00_System/score_corrections.jsonl, runs a per-call-type
    linear regression once total corrections ≥20 (and per-type corrections
    ≥5), writes new weights to vault/00_System/scoring_weights.yaml with a
    timestamped .bak of the prior file.
    """
    from consultant_brain.scoring.retrain import retrain_weights

    report = retrain_weights(vault_root=vault)
    typer.echo(report.summary_line())


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
