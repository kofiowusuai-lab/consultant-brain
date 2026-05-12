"""Active brain for Consultant Copilot.

Ingests live-call session JSONs from the Swift app into an Obsidian-backed
atom graph at `~/ConsultantBrain/`, then makes those atoms queryable via a
local LanceDB vector index. Phase 1 ships the ingest + query CLI; later
phases add the FastAPI service, live transcript deltas, scoring, and
post-call distillation.
"""

__version__ = "0.1.0"
