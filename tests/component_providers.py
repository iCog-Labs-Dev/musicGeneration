"""Installed-style constructors used by config-driven workflow tests."""

from dataclasses import dataclass, replace
from pathlib import Path

import mido

from aimusic.core.component_types import RenderRequest, RenderResult
from aimusic.core.core_types import Score
from aimusic.core.protocols import ScoreDecoder, ScoreRenderer, TonalMetric
from aimusic.planning.protocols import CandidateGenerator
from aimusic.render.midi_adapter import MidiRenderer, MidiRenderSettings
from tests.test_artifact_injection import KnownScoreDecoder, ThirdHeadCandidates


def make_candidates(*, tonal_metric: TonalMetric) -> CandidateGenerator:
    return ThirdHeadCandidates(tonal_metric)


def make_decoder(*, fail: bool = False) -> ScoreDecoder:
    return KnownScoreDecoder(fail=fail)


@dataclass(frozen=True)
class MarkedMidiRenderer:
    midi_settings: MidiRenderSettings
    marker: str

    def __call__(self, score: Score, request: RenderRequest) -> RenderResult:
        result = MidiRenderer(self.midi_settings)(score, request)
        midi = mido.MidiFile(request.output_path)
        midi.tracks[0].insert(0, mido.MetaMessage("text", text=self.marker, time=0))
        midi.save(request.output_path)
        return replace(result, size_bytes=Path(request.output_path).stat().st_size)


def make_renderer(*, midi_settings: MidiRenderSettings, marker: str) -> ScoreRenderer:
    return MarkedMidiRenderer(midi_settings, marker)
