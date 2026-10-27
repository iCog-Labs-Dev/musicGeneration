"""MIDI analysis helpers (fallback when structure.json is absent)."""

from __future__ import annotations

from aimusic.audio.analysis.from_midi import analyze_midi, build_analysis_document

__all__ = ["analyze_midi", "build_analysis_document"]
