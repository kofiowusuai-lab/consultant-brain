"""In-memory preview state for context dumps.

The upload endpoint stashes a `ContextDumpPreview` keyed by ULID, the
UI fetches/edits/commits it, then it's evicted on commit OR after a
TTL (so abandoned uploads don't leak tmp files forever).

Thread-safe — multiple FastAPI worker threads can hit the store
concurrently when several uploads are in flight.

The store is one of the `app.state.*` attributes seeded in
`service.create_app()`, mirroring the Phase 8 pattern
(`app.state.crm_resolver`, `app.state.extractor_provider_cache`).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes


class ContextDumpPreviewStore:
    """ULID → preview, with lazy TTL eviction on every read."""

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._entries: dict[str, tuple[datetime, "ContextDumpPreview"]] = {}
        self._lock = threading.Lock()

    def put(self, preview: "ContextDumpPreview") -> None:
        """Stash a preview. Replaces an existing entry under the same
        preview_id (re-upload after edit, etc.)."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._entries[preview.preview_id] = (now, preview)

    def get(self, preview_id: str) -> "ContextDumpPreview | None":
        """Fetch a preview, evicting it if it's past TTL. Returns None
        for unknown OR expired IDs — callers can treat 404 identically."""
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._entries.get(preview_id)
            if entry is None:
                return None
            stored_at, preview = entry
            if now - stored_at > self._ttl:
                # Stale — drop + clean its tmp file if any.
                self._entries.pop(preview_id, None)
                _cleanup_tmp(preview)
                return None
            return preview

    def pop(self, preview_id: str) -> "ContextDumpPreview | None":
        """Fetch + remove the preview atomically. Used on commit."""
        with self._lock:
            entry = self._entries.pop(preview_id, None)
        if entry is None:
            return None
        _, preview = entry
        return preview

    def discard(self, preview_id: str) -> bool:
        """Drop the preview without returning it. Cleans up its tmp."""
        with self._lock:
            entry = self._entries.pop(preview_id, None)
        if entry is None:
            return False
        _cleanup_tmp(entry[1])
        return True

    def evict_expired(self) -> int:
        """Sweep stale entries. Called periodically by callers that
        want a tidy memory footprint; the lazy path already removes
        them on read."""
        now = datetime.now(timezone.utc)
        evicted = 0
        with self._lock:
            for pid in list(self._entries.keys()):
                stored_at, preview = self._entries[pid]
                if now - stored_at > self._ttl:
                    self._entries.pop(pid)
                    _cleanup_tmp(preview)
                    evicted += 1
        return evicted

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _cleanup_tmp(preview) -> None:
    """Best-effort: remove the tmp text file the preview pinned."""
    path = getattr(preview, "raw_text_full_path", None)
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# Import at end to dodge the import-order cycle (orchestrator imports
# from this module too).
from consultant_brain.context_dumps.orchestrator import ContextDumpPreview  # noqa: E402,F401
