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
- `question` / `frame` → live in `05_Plays/` (reusable, not call-extracted) — Phase 6.
- `definition` → lives in `06_Definitions/` — Phase 6.
- `win_pattern` / `loss_pattern` → promoted from repeated atoms — Phase 6 distillation.

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
