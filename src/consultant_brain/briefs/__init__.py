"""Per-client AI briefs — structured summaries the dashboard renders
under each client's Context section.

The brief pulls every atom + call note + context dump for one client,
sends them to the post-call LLM (typically Anthropic Opus for atom
quality), and returns named sections: state summary, key facts, open
commitments, open objections, recent moves, next steps, meeting-prep
checklist. The dashboard renders each block as its own card.
"""

from __future__ import annotations

from consultant_brain.briefs.generator import (
    ClientBrief,
    ClientBriefCommitment,
    ClientBriefObjection,
    ClientBriefMeta,
    generate_client_brief,
    load_cached_brief,
)
from consultant_brain.briefs.notes import (
    IngestedNoteResult,
    StructuredNoteFact,
    ingest_chat_note,
)
from consultant_brain.briefs.qa import (
    ChatProcessResult,
    ChatTurn,
    ClientBriefAnswer,
    ask_about_client,
    process_chat,
)

__all__ = [
    "ChatProcessResult",
    "ChatTurn",
    "ClientBrief",
    "ClientBriefAnswer",
    "ClientBriefCommitment",
    "ClientBriefMeta",
    "ClientBriefObjection",
    "IngestedNoteResult",
    "StructuredNoteFact",
    "ask_about_client",
    "generate_client_brief",
    "ingest_chat_note",
    "load_cached_brief",
    "process_chat",
]
