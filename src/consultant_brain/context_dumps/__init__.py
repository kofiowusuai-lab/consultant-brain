"""Phase 10 — per-client context dumps.

User flow: pick a file (PDF / DOCX / TXT / image / audio / ZIP) →
brain parses + transcribes → extractor returns a preview of proposed
atoms → user confirms → atoms commit to the vault with
`last_seen = user-supplied observed_at`.

Public surface:
  - `parse_to_text(path) -> ParsedDoc` — file → text via the right
    backend for each extension.
  - `run_preview(...)` — run parsing + extraction, returning a
    `ContextDumpPreview` without writing anything.
  - `commit_preview(...)` — write the dump note + atoms to the vault.
  - `ContextDumpPreviewStore` — in-memory preview state with TTL.
"""

from __future__ import annotations

from consultant_brain.context_dumps.orchestrator import (
    ContextDumpCommitResult,
    ContextDumpPreview,
    commit_preview,
    run_preview,
)
from consultant_brain.context_dumps.parsers import (
    ParsedDoc,
    UnsupportedFileError,
    parse_to_text,
)
from consultant_brain.context_dumps.preview_store import (
    ContextDumpPreviewStore,
)

__all__ = [
    "ContextDumpCommitResult",
    "ContextDumpPreview",
    "ContextDumpPreviewStore",
    "ParsedDoc",
    "UnsupportedFileError",
    "commit_preview",
    "parse_to_text",
    "run_preview",
]
