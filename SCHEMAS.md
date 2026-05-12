# Schemas — consultant-brain

Canonical reference for every persistent type. The Pydantic source of truth is `src/consultant_brain/schemas.py`; this doc explains the shapes, the reasoning behind them, and the on-disk representations.

If you change a schema, update both. The atom schema is especially load-bearing — every later phase reads atoms; a sloppy schema rots the brain.

---

## Inputs

### `SessionJSON` — the Swift app's saved call

Read-only source. Lives at `~/Library/Application Support/Consultant Copilot/Sessions/session-<ISO8601>.json`.

```jsonc
{
  "startedAt": "2026-05-12T15:42:29Z",
  "completedTurns": [
    {
      "completedAt": "2026-05-12T15:43:04Z",
      "itemID": "item_DejXrVRtjPEcwF9tG8pyC",
      "source": "systemAudio" | "microphone",
      "text": "..."
    }
  ],
  "suggestions": [
    {
      "category": "askThis" | "sayThis" | "respond" | "angle" | "private_note" | "...",
      "createdAt": "2026-05-11T22:08:37Z",
      "text": "..."
    }
  ]
  // unknown keys (partialSources, partialTranscripts, processedSegmentIDs, partialSpeakers) ignored
}
```

**Source → Speaker mapping:**
- `source: "systemAudio"` → `Speaker.them` (client / prospect)
- `source: "microphone"` → `Speaker.you` (the consultant)

Suggestions are preserved on ingest for future analysis (Phase 4+ will mine acceptance rate) but Phase 1 doesn't promote them to atoms.

---

## Vault outputs

### `Atom` — `~/ConsultantBrain/03_Atoms/<id>.md`

```markdown
---
id: 01HXAVQR8N0F7P5K2C4M9B6S3D
type: objection
client: "[[Reece]]"
call: "[[2026-05-12_reece_consultingCall]]"
call_type: consultingCall
tags: [budget, scope]
confidence: 0.82
evidence_count: 1
last_seen: 2026-05-12
created_at: 2026-05-12T19:42:10Z
status: active
embedding_id: 01HXAVQR8N0F7P5K2C4M9B6S3D
---

Price came up before scope. Reece asked about ballpark fees in the first five
minutes, before we discussed deliverables. Anchor risk: budget number sets
mental ceiling regardless of value framing.
```

**Field reference:**

| Field | Type | Why it's here |
|---|---|---|
| `id` | str (8-64 chars) | Deterministic — `sha256(session_id + extractor_version + atom_index)[:26]`. Makes re-ingest idempotent. |
| `type` | `AtomType` enum (7 values) | Insight, objection, commitment, win_signal, loss_signal, confusion, client_fact. See "Atom types" below. |
| `client` | str \| null | Display name (e.g. `"Reece"`). Vault writer renders as `[[Reece]]` so Obsidian backlinks resolve. |
| `call` | str | Call note ID — vault writer renders as `[[<id>]]`. Always set; an atom always belongs to a call. |
| `call_type` | `CallType` enum (6 values) | Mirrors the Swift `CallType`. |
| `tags` | list[str] | Free-form keywords for filtering / Obsidian's tag pane. Lower-snake by convention. |
| `confidence` | float [0,1] | Extractor's confidence. Phase 2+ retrieval boosts high-confidence atoms. |
| `evidence_count` | int ≥1 | Times this atom has been observed across calls. Always 1 in Phase 1; Phase 6 promotion logic increments. |
| `last_seen` | date | Last call this atom (or a sibling) appeared in. Lets Phase 6 decay stale atoms. |
| `created_at` | datetime (UTC) | First time the atom was minted. Immutable after. |
| `status` | `AtomStatus` enum | `active` (default), `retired`, `needs_review`. Phase 6 demotes; Phase 1 always writes `active`. |
| `embedding_id` | str | Vector DB lookup key. Mirrors `id` in Phase 1; separated so future phases can map one atom → many embeddings (e.g. multilingual). |
| `client_org_id` | UUID \| null | **Phase 8.** Stable identifier from the Swift CRM's `organizations.id`. Lets us survive display-name renames without fragmenting the atom graph. Null when the client isn't in the CRM yet. |

**Body** = the atomic content. 1–3 sentences. Wikilinks freely. Validator rejects multi-paragraph bodies (atoms are atomic).

### Atom types

| Type | When to mint one |
|---|---|
| `insight` | An observation about the dynamic / pattern in the call that's worth remembering. "Reece keeps saying 'I think' about scope — hasn't decided what 'winning ad' means." |
| `objection` | A concern, hesitation, or pushback raised by the client. "Tried that before with another vendor and it didn't stick." |
| `commitment` | A concrete promise either side made. "I'll send you the vault sample by Friday." Includes scheduling, deliverables, scope agreements. |
| `win_signal` | Enthusiasm, scope expansion, "this is exactly what we need", buying signals. |
| `loss_signal` | Deflection, going silent on questions, "let me think about it", pushing decisions out. |
| `confusion` | Client asked a clarifying question, repeated something back wrong, or signaled they didn't follow. Negative weight on training calls (questions are good there); negative on consulting (they shouldn't be confused at the end). |
| `client_fact` | Verifiable facts the client stated about their business — numbers, tools, team size, current stack, workflow. Anchors future retrieval. |

Dropped from Phase 1 (return in later phases):
- `question` / `frame` → live in `05_Plays/` (reusable, not call-extracted) — Phase 6/8.
- `definition` → lives in `06_Definitions/` — Phase 6.
- `win_pattern` / `loss_pattern` → promoted from repeated atoms — Phase 6 distillation.

---

## Phase 8 schemas

### `Play` — `~/ConsultantBrain/05_Plays/<id>.md`

```markdown
---
id: objection__budget_came_up_before_scope
kind: play
type: objection
call_count: 5
client_count: 3
clients: [Acme, BetaCorp, Gamma]
status: active
member_atom_ids: [01HXAVQR..., 01HXAVQS..., ...]
created_at: 2026-05-12T19:42:10Z
updated_at: 2026-05-12T19:42:10Z
---

# Play: objection

Promoted from 5 atoms across 5 calls in 3 clients.

## Four-frame template

- **Opener** — _(fill in)_
- **Frame** — _(fill in)_
- **Response** — _(fill in)_
- **Handle** — _(fill in)_

## Observed examples
- Price came up before scope on Reece...
- ...

## Member atoms
- [[01HXAVQR...]]
```

Auto-promoted by `distill` when an atom body's 3-word shingles overlap (Jaccard ≥0.4) across ≥3 distinct clients and ≥3 calls. The four-frame body is hand-edited by the consultant during weekly reviews — the promoter only writes the template the first time; subsequent re-promotions preserve filled-in frames.

### `Person` — `~/ConsultantBrain/07_People/<slug>.md`

```markdown
---
id: pat_lee
kind: person
name: Pat Lee
role: CEO
email: pat@acme.com
org: Acme AI
mention_count: 12
created_at: 2026-05-12T19:42:10Z
updated_at: 2026-05-12T19:42:10Z
---

# Pat Lee

**Role**: CEO
**Email**: pat@acme.com
**Org**: [[clients/acme_ai]]

## Atom mentions (12)
- [[01HXAVQR...]] — commitment · Pat will sign off on the rollout by Friday
- ...

## Pinned notes
<!-- User-edited section, preserved across regeneration -->
```

Auto-regenerated by `distill` from the Swift CRM's contacts table + atom-body substring match for first-name and full-name.

### `WeeklyReview` — `~/ConsultantBrain/08_Reviews/<YYYY-Www>.md`

```markdown
---
id: 2026-W19
kind: weekly_review
week_start: 2026-05-11
week_end: 2026-05-17
atom_count: 87
call_count: 6
client_count: 3
created_at: 2026-05-17T22:00:00Z
updated_at: 2026-05-17T22:00:00Z
---

# Week 2026-W19 (2026-05-11 → 2026-05-17)

## At a glance
- Calls: 6
- Atoms: 87
- Clients touched: 3

## Atoms by type
- **commitment** — 21
- ...

## Top clients
- ...

## Suggested next moves
- Follow up on commitment (Reece): ...

## Lessons + commitments for next week
<!-- User-edited, preserved across regeneration -->
```

Generated by `consultant-brain weekly [--week 2026-W19]`. Atoms inside the week's date range get bucketed; calls in the same range get summary excerpts. The pinned section at the bottom survives regeneration.

### LLM provider contracts

All five providers implement one Protocol:

```python
@dataclass(frozen=True)
class ChatRequest:
    system: str
    user: str
    model: str
    max_tokens: int = 4096
    enable_prompt_cache: bool = True

@dataclass(frozen=True)
class ChatResponse:
    text: str
    model_used: str
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_creation_tokens: int | None

class LLMProvider(Protocol):
    name: str
    def chat(self, request: ChatRequest) -> ChatResponse: ...
```

Concrete providers (`AnthropicProvider`, `OpenAIProvider`, `OpenRouterProvider`, `DeepSeekProvider`, `KimiProvider`) translate ChatRequest into their respective wire formats. Caching: Anthropic via explicit `cache_control` blocks, OpenAI/OpenRouter via automatic prefix-match (≥1024 tokens), DeepSeek/Kimi pass through without caching.

---

### `CallNote` — `~/ConsultantBrain/02_Calls/<id>.md`

```markdown
---
id: 2026-05-12_reece_consultingCall
client: "[[Reece]]"
call_type: consultingCall
date: 2026-05-12
duration_minutes: 47
source_session: session-2026-05-12T05-42-10Z.json
extractor_model: claude-sonnet-4-6
extractor_version: 1
atom_count: 19
created_at: 2026-05-12T19:42:10Z
---

# Reece — consultingCall, 12 May 2026

## Summary
3-line summary of the call written by the extractor.

## Atoms
- [[01HXAVQR8N...]] — objection: price came up before scope
- [[01HXAVQR9P...]] — commitment: Reece will send the ad vault sample by Friday
- ...

## Transcript
<details>
<summary>Full transcript</summary>

You: ...
Them: ...

</details>
```

**`id` format**: `YYYY-MM-DD_<client-slug>_<callType>`. Lowercase, kebab/snake the client name. Stable across re-ingests.

**Why transcript embedded inside `<details>`**: Obsidian renders it as a collapsible disclosure in preview mode, so the file stays scannable but the source is never lost.

---

## Extractor I/O

### `ExtractedAtom` — what the LLM returns per atom

The LLM only fills semantically-meaningful fields. The vault writer adds the bookkeeping (`id`, `client`, `call`, `call_type`, `last_seen`, `created_at`, `status`, `embedding_id`).

```json
{
  "type": "objection",
  "body": "Price came up before scope. Reece asked about ballpark fees...",
  "confidence": 0.82,
  "tags": ["budget", "scope"]
}
```

### `ExtractorResult` — full response from one Claude call

```json
{
  "summary": "Reece is sold on the outcome but anxious about price.",
  "atoms": [
    { "type": "objection", "body": "...", "confidence": 0.82, "tags": ["budget"] }
  ]
}
```

---

## Determinism

**Atom IDs are deterministic.** `id = sha256(f"{session_id}|{extractor_version}|{atom_index}")[:26]`. Re-ingesting the same session JSON with the same extractor version produces the same atom IDs. The vault writer overwrites in place — never duplicates.

**`extractor_version`** lives in `schemas.py` as a module constant. Bump it when you change the extractor prompt or model in a way that should regenerate old atoms. Existing atom files will overwrite with new content; old atom IDs vanish (they're a function of `(session, old_version, index)`).

This is the only place in the system where IDs aren't randomly generated. Worth the simplicity tax.
