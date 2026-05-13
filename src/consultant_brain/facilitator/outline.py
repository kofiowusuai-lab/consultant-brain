"""Canonical facilitator outline.

Mirrors `Sources/ConsultantCopilotCore/Facilitator/FacilitatorOutline.swift`
in the Swift repo. Both files must stay in lockstep: if a cue phrase
moves between stages here, the same edit lands in the Swift outline,
otherwise the brain's `suggest_stage` endpoint will recommend stages
the Swift UI doesn't know how to render.

Phase 13 ships a single outline (Reece's Ad Engine v1 working
session). Future phases will swap the outline by call_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StageGroup(str, Enum):
    """The three top-level sections rendered in the HTML sidebar."""

    session = "session"
    build = "build"
    convergence = "convergence"


@dataclass(frozen=True, slots=True)
class FacilitatorStage:
    """One stage in the meeting outline. Mirror of the Swift struct.

    `id` is 1-based for display ("01"-"08" in the HTML sidebar);
    `cue_phrases` are lowercase substrings the suggestor matches
    against the rolling transcript window. Phrases stay short so the
    suggestor doesn't over-fit to one specific moment of one specific
    call.
    """

    id: int
    group: StageGroup
    anchor: str
    title: str
    duration: Optional[str]
    cue_phrases: tuple[str, ...]

    @property
    def index(self) -> int:
        """0-based index used everywhere except in the rendered label."""
        return self.id - 1


REECE_OUTLINE: tuple[FacilitatorStage, ...] = (
    FacilitatorStage(
        id=1,
        group=StageGroup.session,
        anchor="objective",
        title="Session objective",
        duration="8 min",
        cue_phrases=(
            "objective",
            "frame the session",
            "smallest useful version",
            "build path",
            "park everything else",
        ),
    ),
    FacilitatorStage(
        id=2,
        group=StageGroup.session,
        anchor="choreography",
        title="Screen-share choreography",
        duration="when to show what",
        cue_phrases=(
            "choreography",
            "share this console",
            "stop sharing",
            "screen share",
            "walk me through your morning",
        ),
    ),
    FacilitatorStage(
        id=3,
        group=StageGroup.session,
        anchor="agenda",
        title="Agenda timeline",
        duration="90 min total",
        cue_phrases=(
            "agenda",
            "timeline",
            "ninety minutes",
            "90 min",
            "frame the session",
            "map current workflow",
        ),
    ),
    FacilitatorStage(
        id=4,
        group=StageGroup.build,
        anchor="scope",
        title="v1 scope boundary",
        duration="in vs parked",
        cue_phrases=(
            "scope",
            "v1 scope",
            "parking lot",
            "out of scope",
            "in scope",
            "lock scope",
            "park modules",
        ),
    ),
    FacilitatorStage(
        id=5,
        group=StageGroup.build,
        anchor="engine",
        title="Angle-selection engine",
        duration="how the bot decides",
        cue_phrases=(
            "angle engine",
            "angle selection",
            "core angle",
            "avatar slice",
            "awareness lock",
            "belief gap",
            "emotional entry",
            "how the bot decides",
        ),
    ),
    FacilitatorStage(
        id=6,
        group=StageGroup.build,
        anchor="canvas",
        title="Live workflow canvas",
        duration="today vs v1",
        cue_phrases=(
            "workflow canvas",
            "today vs v1",
            "current workflow",
            "future workflow",
            "workflow map",
        ),
    ),
    FacilitatorStage(
        id=7,
        group=StageGroup.convergence,
        anchor="decisions",
        title="Decision questions",
        duration="15 min",
        cue_phrases=(
            "decision questions",
            "decision mode",
            "decisions",
            "eight questions",
            "question 1",
            "question 2",
        ),
    ),
    FacilitatorStage(
        id=8,
        group=StageGroup.convergence,
        anchor="close",
        title="Close and next step",
        duration="5 min",
        cue_phrases=(
            "close",
            "wrap up",
            "next step",
            "three doors",
            "post-call memo",
            "end on a choice",
        ),
    ),
)


def get_outline(outline_id: str = "reece-ad-engine-v1") -> tuple[FacilitatorStage, ...]:
    """Resolve an outline by id. Phase 13 only knows about Reece's
    session; future phases extend the registry. Unknown ids fall back
    to the Reece outline so older Swift clients pinning an unknown id
    still get a sensible response."""
    if outline_id == "reece-ad-engine-v1":
        return REECE_OUTLINE
    return REECE_OUTLINE
