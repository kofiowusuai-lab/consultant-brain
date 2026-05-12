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

### Ingest + query (Phase 1)

```bash
# Ingest a real call
uv run consultant-brain ingest \
  --session "$HOME/Library/Application Support/Consultant Copilot/Sessions/session-2026-05-12T05-42-10Z.json" \
  --client "Reece" --call-type consultingCall

# Vector search the vault
uv run consultant-brain query "what did the client say about price"

# Re-embed atoms missing from LanceDB (recovers from index wipes)
uv run consultant-brain reindex
```

After an ingest, open `~/ConsultantBrain/` in Obsidian — the new call note shows ~15-30 wikilinked atoms.

### Three-layer suggest (Phase 2)

```bash
uv run consultant-brain suggest \
  --window "the bot keeps grabbing the wrong notes from our vault" \
  --client "Reece" --call-type consultingCall \
  --explain
```

Returns the panel slice the live copilot would show: 1 hot + 2 warm + 1 cold atom by default. `--explain` prints rank components (similarity / recency / confidence / final score).

### Self-evaluation (Phase 7)

```bash
uv run consultant-brain metrics --vault ~/ConsultantBrain
# With a labeled corpus for retrieval precision:
uv run consultant-brain metrics --vault ~/ConsultantBrain --corpus ~/labels.json
# Machine-readable:
uv run consultant-brain metrics --vault ~/ConsultantBrain --json
```

Four metrics, each only computed when its source data exists:

- **Retrieval precision** — pass `--corpus path/to/labels.json` with hand-labeled `{queries: [{name, window, client, call_type, relevant_atom_ids}, ...]}`. Reports P@1/P@3/P@5 + perfect-top-1 count.
- **Suggestion acceptance rate** — `/suggestions` logs every emit; `POST /suggestion_referenced` from the Swift app logs each user-referenced atom. Pairs emits with referenced events within a 90s window; reports overall + per-layer.
- **Score-prediction MAE** — mean absolute error between predicted and user-overridden scores, plus the master prompt's calibration target ("% within ±10"). Per-call-type breakdown.
- **Pattern stability** — `distill` snapshots pattern state on every run; metric compares the latest with the most recent snapshot ≥30 days old, reports % patterns that persisted + mean observation-count delta per surviving pattern.

When a metric's source data isn't ready (no corpus / no override yet / one snapshot only), the report shows `—` instead of fake zeros. Honest dashboard by design.

### Distillation (Phase 6)

```bash
uv run consultant-brain distill --vault ~/ConsultantBrain
```

Two passes:
- **Pattern mining:** any (atom_type, primary_tag) cluster observed across ≥3 distinct calls gets promoted to `04_Patterns/<type>__<tag>.md` with frontmatter (observation_count, member_atom_ids, score_impact_proxy). Re-runs are idempotent — updates an existing pattern in place, preserving `created_at`.
- **`context.md` auto-regeneration:** every client folder's `context.md` gets rebuilt from active atoms grouped by type + the 5 most recent call summaries. The `<!-- pin --> ... <!-- /pin -->` block preserves user-edited notes across regenerations.

Once patterns exist, the **cold retrieval layer fires automatically** during live calls — `consultant-brain suggest` returns matched patterns alongside hot+warm atoms. The Swift overlay's `Memory` panel shows them in purple (`COLD` tag).

### Post-call scoring + override (Phase 5)

```bash
# Score a finished call
uv run consultant-brain score 2026-05-12_reece_consultingCall --vault ~/ConsultantBrain --explain

# Once 20+ user overrides have accumulated, refit the weights
uv run consultant-brain retrain --vault ~/ConsultantBrain
```

Scoring is a weighted sum of 9 observable features (next-step booked, objections resolved, talk-ratio balance, commitments made, win/loss/confusion signal counts, completion of agenda, LLM-judged primary-win progress). Default weights live in `~/ConsultantBrain/00_System/scoring_weights.yaml` with per-call-type profiles — closing calls weight commitments heavily, cold calls weight next-step heaviest, training calls treat confusion as positive (questions are good when learning).

User overrides hit `POST /score_override`, append to `00_System/score_corrections.jsonl`, and feed the `retrain` linear-regression job once you have ≥20 corrections (≥5 per call type).

### FastAPI service (Phase 3)

The Swift app + Phase 4's live loop talk to the brain over HTTP.

```bash
uv run consultant-brain serve            # localhost:8787 by default

curl http://127.0.0.1:8787/healthz
curl -X POST http://127.0.0.1:8787/call_start \
  -H 'Content-Type: application/json' \
  -d '{"call_id":"abc","client":"Reece","call_type":"consultingCall"}'
curl -X POST http://127.0.0.1:8787/transcript_delta \
  -H 'Content-Type: application/json' \
  -d '{"call_id":"abc","speaker":"them","text":"the bot keeps grabbing wrong notes"}'
curl 'http://127.0.0.1:8787/suggestions?call_id=abc' | jq
curl -X POST http://127.0.0.1:8787/call_end \
  -H 'Content-Type: application/json' \
  -d '{"call_id":"abc"}'
```

Endpoints:
- `GET /healthz` — liveness + active-call count + vault path
- `POST /call_start` — `{ call_id, client, call_type }`
- `POST /transcript_delta` — `{ call_id, speaker: "you"|"them", text }`
- `GET /suggestions?call_id=X[&hot=1&warm=2&cold=1]` — panel slice
- `POST /call_end` — `{ call_id }`
- `GET /score?call_id=X[&primary_win=...]` — 0–100 score + per-feature breakdown
- `POST /score_override` — `{ call_id, user_score, primary_win? }`
- `POST /suggestion_referenced` — `{ call_id, atom_id }` — Phase 7 acceptance-rate hook

Port 8787 by default (not 8765 — that's OpenClaw's shared swap port). `--reload` for dev auto-reload.

## Tests

```bash
uv run pytest
```

`@requires_ollama` tests skip cleanly when Ollama isn't running locally. CI without Ollama still gets 90+ tests covering schemas, vault writes, ingest plumbing, retrieval ranking, and the service.

## What's next

The master prompt's 7 phases are complete. Future work that didn't make the master prompt:

- **Stakeholder graph (`07_People/`)** — auto-built from every named individual across atoms + contacts. Useful for "who said this" / "what does this person care about" lookups.
- **Weekly reviews (`08_Reviews/`)** — auto-generated per-week summaries: which clients moved, which patterns fired, which scores trended up.
- **CRM linking** — atom `client` field → `CRMOrganization` UUID lookup against the Swift app's SQLite. Eliminates the "two namespaces" issue (live UUID call_id vs canonical call_note_id) by joining at the org level.
- **Swift suggestion-referenced UI** — tappable "got it" / "said it" button on each Memory-panel atom in the overlay so the acceptance-rate metric gets real data.
