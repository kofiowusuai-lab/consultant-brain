"""Phase 13 — facilitator console support.

The brain side of the meeting-outline feature. The Swift app pins a
floating window to the bottom of the screen and renders a hand-built
HTML facilitator console in a WKWebView. The brain provides a
deterministic transcript-driven stage suggestor so the user gets a
"next stage" toast when the conversation cues a transition.

Submodules:
  outline   — canonical 8-stage outline mirroring the Swift
              FacilitatorOutline.reece corpus
  suggestor — pure function over (window, current_stage_index,
              outline) → SuggestionResult
"""
