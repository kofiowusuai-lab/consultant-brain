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

## Phase 8 — completing the brain

Phase 8 closes every empty folder in the vault and ships the production polish the master prompt deferred. Fourteen items in one batch:

### CRM linking (item 1)

Every atom now carries `client_org_id: UUID | null` linking back to a row in the Swift CRM's SQLite. The brain reads `~/Library/Application Support/Consultant Copilot/CRM/crm.sqlite` in read-only mode, resolves client display name → UUID at ingest time, and stamps the UUID on every atom + call note. Display-name renames in the Swift CRM no longer fragment the atom graph.

### Plays auto-promotion (`05_Plays/`) + cold layer (item 7)

Atoms of type `commitment` / `objection` / `win_signal` whose body shingles overlap (Jaccard ≥0.4) across ≥3 distinct clients get auto-promoted to a Play. Each play file carries a four-frame template (opener / frame / response / handle) the consultant fills in by hand. Plays now fire alongside patterns from the cold retrieval layer — surfaced first because they require more evidence to promote.

### Stakeholder graph (`07_People/`) + Weekly reviews (`08_Reviews/`) (items 2, 3)

`distill` regenerates one markdown per CRM contact, listing every atom mentioning them by name; click the file in Obsidian's graph view to see the cross-org connections. `consultant-brain weekly --week 2026-W19` writes a per-ISO-week review of clients touched, atoms minted by type, top moves recommended for next week. Both files preserve hand-edited pinned sections across regeneration.

### Tag normalization + atom retirement (items 9, 8)

`distill` runs two more passes on every invocation:
- **Tag normalization** folds near-duplicate tags (`nextstep`, `followup`, `follow_up`, `next-step` → `next_step`) using a canonical alias map plus Levenshtein-radius-2 merge.
- **Retirement lifecycle** flips atoms `active → needs_review` after 120 days without re-observation, then `needs_review → retired` after another 30. Retired atoms are kept on disk but excluded from retrieval + pattern mining.

### Multi-LLM provider abstraction (item 13)

The atom extractor + live moment detector now route through an `LLMProvider` Protocol with concrete implementations for **Anthropic, OpenAI, OpenRouter, DeepSeek, and Kimi (Moonshot)**. Pick a provider per-task via:

```bash
# Per call
uv run consultant-brain ingest --extractor-provider openrouter

# Or pin globally
export BRAIN_LLM_PROVIDER=deepseek
export BRAIN_MOMENT_PROVIDER=kimi
```

Or via the Swift app: **Settings → Brain → LLM Providers** picks the extractor + moment-detector providers independently. API keys live in the same secrets.json as the Anthropic + OpenAI keys (`openai-api-key`, `openrouter-api-key`, `deepseek-api-key`, `kimi-api-key`).

The Anthropic provider attaches `cache_control: {"type": "ephemeral"}` to the system prompt — ~70% token-cost cut after the first cache hit per 5-minute TTL. OpenAI auto-caches prefixes ≥1024 tokens; OpenRouter routes the cache_control through to upstream providers that support it.

### Real post-call extraction (item 5)

`/call_end` now optionally runs the full atom extractor against the live transcript. The placeholder summary string is gone; call notes carry real Claude-written prose, and any atoms the live moment detector missed get backfilled. Toggle via `CONSULTANT_BRAIN_POST_CALL_EXTRACT=1` (Swift app sets it from the Brain Settings toggle).

### Backup + restore + anonymized export (items 10, 11)

```bash
uv run consultant-brain backup                                  # → ~/Documents/ConsultantBrain-backups/<ts>.tar.gz
uv run consultant-brain restore --from <tar> --vault /tmp/v
uv run consultant-brain export --out ~/Desktop/anon --anonymize # → safe to share
```

The export rewrites client + stakeholder names + emails to `[CLIENT_N]` / `[PERSON_N]` / `[EMAIL_N]` while preserving every pattern + play structurally intact.

### `/metrics` + `/diagnostics` endpoints (items 12, 14)

```bash
curl http://127.0.0.1:8787/metrics     # Phase 7 dashboard as JSON
curl http://127.0.0.1:8787/diagnostics # vault + LanceDB + Ollama + LLM-key readiness
```

The Swift app's new **Brain Status** window renders both live (refresh every 5s). The dashboard sidebar's always-visible **Brain** indicator (green/yellow/red dot) polls `/diagnostics` every 10s — click to open Brain Status, right-click for **Run Health Check Now / Open Brain Settings / Restart Brain Service**.

### Swift "Reference this" UI (item 4)

Tap any Memory atom in the overlay → POSTs `/suggestion_referenced` and the row flashes green for 1.5s. Powers the acceptance-rate metric with real signal.

### New CLI subcommands

| Subcommand | Purpose |
|---|---|
| `weekly [--week 2026-W19]` | Generate `08_Reviews/<week>.md` |
| `backup [--out path]` | Snapshot vault + LanceDB to a tarball |
| `restore --from <tar> --vault <target>` | Restore a vault from a backup |
| `export --out <dir> [--no-anonymize]` | Shareable export of patterns + plays |

`distill` itself now runs all six passes (tags → retirement → patterns → plays → people → context) — skip any with `--skip-tags`, `--skip-retirement`, `--skip-plays`, `--skip-people`, `--skip-patterns`, `--skip-context`.

### Acceptance per item

| # | Item | Pass test |
|---|---|---|
| 1 | CRM linking | New atom on a known client has `client_org_id` matching the CRM row |
| 2 | Stakeholder graph | `distill` produces `07_People/<slug>.md` linking back to org + listing every atom mentioning them |
| 3 | Weekly reviews | `consultant-brain weekly` produces `08_Reviews/<YYYY-Www>.md` |
| 4 | Swift Reference UI | Tap Memory atom → `/suggestion_referenced` event logged in <200ms |
| 5 | Real call-end extraction | Call note carries Claude-written summary, not the placeholder |
| 6 | Prompt caching | Anthropic responses report `cache_read_input_tokens > 0` on call 2+ |
| 7 | Plays | After 3 calls with same opener in 3 clients, `05_Plays/<seed>.md` exists |
| 8 | Atom retirement | Atom with `last_seen` 130d ago → `status: needs_review` after `distill` |
| 9 | Tag normalization | Atoms tagged `nextstep`/`followup`/`next_step` all rewritten to `next_step` |
| 10 | Backup | Tarball round-trips byte-identically |
| 11 | Export | No real client/email/UUID survives anonymized export |
| 12 | `/metrics` + Brain Status window | Window renders live data, refreshes every 5s |
| 13 | Multi-LLM providers | `--extractor-provider deepseek` succeeds; unit tests assert each provider's wire shape |
| 14 | Brain health indicator | Dashboard dot turns green → yellow → red as deps fail; click opens Brain Status |

## Phase 9 — learning from videos

The brain isn't limited to your own calls anymore. Point it at a YouTube video or Instagram Reel and it absorbs the content into a separate knowledge layer.

```bash
# YouTube (uses captions API, free + fast)
uv run consultant-brain learn --url 'https://youtu.be/HD2RU2QZxJk' --topic ai-sales

# YouTube without captions (falls back to yt-dlp + Whisper, costs ~$0.006/min)
uv run consultant-brain learn --url 'https://youtu.be/...' --whisper

# Instagram Reel (Whisper required; --cookies-from-browser for private content)
uv run consultant-brain learn --url 'https://instagram.com/reel/Cabc/' --whisper

# Tie knowledge to one client so retrieval prefers it for that client's prep
uv run consultant-brain learn --url 'https://youtu.be/...' --for-client Reece
```

### Why it stays out of Memory (by default)

Each external atom is stamped `source_kind=youtube|instagram|...`. The retrieval pipeline knows the difference:

- **Live calls** (the default): Hot + Warm layers filter on `source_kind=call`. A Hormozi rant about $1M ad budgets never blurs into Reece's real $10k objection. The Memory panel stays clean.
- **Prep mode**: pass `?knowledge=3` to `/suggestions` (or call `retrieve(include_knowledge=True)`) and a new Knowledge layer fires alongside Warm, surfacing the best matches from `09_Knowledge/`.

### What gets extracted

External sources use a knowledge-tuned extractor (`extractor.extract_from_source`) with a system prompt that biases toward `insight`, `client_fact`, and `win_signal` atoms — not commitments or objections. Atoms get tagged with the source kind + your `--topic` flag, so a quick `consultant-brain query` filtered by tag returns just the things you learned from videos.

### Where it lives on disk

```
~/ConsultantBrain/
  09_Knowledge/
    youtube_HD2RU2QZxJk.md          ← source note: title, summary, atom links, full transcript
    instagram_Cabc1234.md
  03_Atoms/
    01HX...md  ← source_kind: youtube · source_url: https://youtu.be/... · source_title: ...
```

Each knowledge file is idempotent — re-running `learn` on the same URL overwrites in place (atom IDs are deterministic from the source ID + extractor version).

### Transcription stack

| Provider | What it does | Cost | When to use |
|---|---|---|---|
| `youtube-transcript-api` | Pulls auto + manual captions directly | Free | Default for YouTube |
| `yt-dlp` (audio) + OpenAI Whisper | Downloads audio, transcribes via OpenAI's hosted Whisper-1 | ~$0.006/min | Videos without captions, all Instagram Reels |

Whisper needs the OpenAI key in the same `secrets.json` (`openai-api-key`). Private Instagram content needs `--cookies-from-browser chrome` so yt-dlp can authenticate.

### Service endpoint

```bash
curl -X POST http://127.0.0.1:8787/learn -H 'Content-Type: application/json' -d '{
  "url": "https://youtu.be/HD2RU2QZxJk",
  "topic": "ai-sales",
  "for_client": "Reece",
  "allow_whisper": false
}'
```
