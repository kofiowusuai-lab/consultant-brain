"""Resolves client display names → CRMOrganization UUIDs.

Wraps `sqlite_reader` with an in-process LRU cache so the ingest path
doesn't slam SQLite once per atom. The cache is keyed by lowercased
client name; misses fall through to a real lookup, results are stored
even for the negative case (no match found → cached as None).

Production code calls `CRMResolver().resolve("Reece")`. Tests pass an
explicit `crm_path` so they hit a fixture DB.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional
from uuid import UUID

from consultant_brain.crm.sqlite_reader import (
    CRMOrganization,
    CRMNotFoundError,
    find_organization_by_name,
    open_read_only,
)


class CRMResolver:
    """Caches name→UUID lookups across the lifetime of one process.

    Thread-safe — the FastAPI service can race on this from multiple
    request handlers, especially during a busy ingest run.
    """

    def __init__(self, crm_path: Path | None = None) -> None:
        self._crm_path = crm_path
        self._cache: dict[str, CRMOrganization | None] = {}
        self._lock = threading.Lock()

    def resolve(self, client_name: str | None) -> CRMOrganization | None:
        """Return the matching CRMOrganization or None.

        Negative results ARE cached — that's deliberate so we don't
        slam SQLite for every atom written when the client isn't in
        the CRM yet. Call `forget(name)` after the user adds the
        missing org if you need a fresh lookup.
        """
        if not client_name:
            return None
        key = client_name.strip().lower()
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        # Lookup outside the lock so a slow SQLite read doesn't block
        # other resolver users.
        org = self._lookup(client_name)
        with self._lock:
            self._cache[key] = org
        return org

    def resolve_uuid(self, client_name: str | None) -> UUID | None:
        """Convenience: same as resolve() but returns just the UUID."""
        org = self.resolve(client_name)
        return org.id if org else None

    def forget(self, client_name: str) -> None:
        """Drop the cached result for one name (positive or negative).
        Call this after the user adds an org to the CRM mid-session."""
        key = client_name.strip().lower()
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def _lookup(self, client_name: str) -> CRMOrganization | None:
        try:
            conn = open_read_only(self._crm_path)
        except CRMNotFoundError:
            return None
        try:
            return find_organization_by_name(conn, client_name)
        finally:
            conn.close()
