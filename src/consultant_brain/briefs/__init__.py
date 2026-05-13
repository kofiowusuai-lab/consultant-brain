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
from consultant_brain.briefs.qa import (
    ChatTurn,
    ClientBriefAnswer,
    ask_about_client,
)

__all__ = [
    "ChatTurn",
    "ClientBrief",
    "ClientBriefAnswer",
    "ClientBriefCommitment",
    "ClientBriefMeta",
    "ClientBriefObjection",
    "ask_about_client",
    "generate_client_brief",
    "load_cached_brief",
]
