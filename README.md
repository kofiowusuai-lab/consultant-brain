# consultant-brain

Active brain for [Consultant Copilot](../ai-consultant-copilot/) — an Obsidian-backed knowledge graph that every live call writes into and every live call reads from.

## What this is (Phase 1)

A CLI that ingests one of the Swift app's saved `Sessions/*.json` transcripts and writes a graph of typed atoms (objections, commitments, win_signals, etc.) plus a call note into an Obsidian vault at `~/ConsultantBrain/`. Each atom is also embedded into a local LanceDB index so they're queryable by semantic similarity.

Phase 1 is **CLI-only** — no FastAPI service, no live transcript deltas, no Swift integration. Those land in later phases.

## Architecture

```
~/ai-consultant-copilot/      Swift app                    [unchanged]
                                    │
                                    │ reads Sessions/*.json
                                    ▼
~/code/consultant-brain/      Python CLI                   [this repo]
  src/consultant_brain/
    cli.py               ← `consultant-brain` Typer entry
    schemas.py           ← Pydantic models (Atom, CallNote, ...)
    session_loader.py    ← parse Sessions/*.json → struct
    extractor.py         ← Claude Sonnet → atoms[]
    vault.py             ← atomic markdown + YAML frontmatter writer
    embedder.py          ← Ollama nomic-embed-text → LanceDB
                                    │
                                    │ writes
                                    ▼
~/ConsultantBrain/            Obsidian vault              [created on first run]
  00_System/                  ← schema docs, lancedb/
  01_Clients/                 ← per-client folder
  02_Calls/                   ← one .md per ingested call
  03_Atoms/                   ← one .md per atom
  04_Patterns/   05_Plays/   06_Definitions/   07_People/   08_Reviews/
```

The seven `04_..08_` dirs stay empty in Phase 1. Later phases populate them.

## Setup (one-time)

```bash
# 1. Install uv (skip if already installed)
brew install uv

# 2. Pull the Ollama embedding model (~270 MB, one-time)
ollama pull nomic-embed-text

# 3. Sync dependencies
cd ~/code/consultant-brain
uv sync
```

The Anthropic API key is read from the Swift app's `~/Library/Application Support/Consultant Copilot/secrets.json` — no second copy needed.

## Usage

```bash
# Ingest a real call
uv run consultant-brain ingest \
  --session "$HOME/Library/Application Support/Consultant Copilot/Sessions/session-2026-05-12T05-42-10Z.json" \
  --client "Reece" \
  --call-type consultingCall

# Query the brain
uv run consultant-brain query "what did the client say about price"
```

After an ingest, open `~/ConsultantBrain/` in Obsidian — the new call note shows ~15-30 wikilinked atoms.

## Tests

```bash
uv run pytest
```

Tests with the `live` marker actually hit the Anthropic API; CI runs without `--run-live` by default.

## Out of scope this phase

FastAPI service · live 15s loop · auto-scoring · pattern distillation · context.md auto-regen · stakeholder graph · weekly reviews · CRM linking. Each gets its own plan when we ship it.
