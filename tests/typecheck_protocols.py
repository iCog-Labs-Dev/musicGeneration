"""Mypy checks real defaults and independently supplied replacements here."""

from __future__ import annotations

from typing import Mapping, Sequence

from aimusic.core.component_types import (
    CandidateGenerationResult,
    SymbolicNote,
    TrackInstrumentConfig,
)
from aimusic.core.config import DecodeConfig, StyleConfig
from aimusic.core.core_types import BeatState, Score
from aimusic.core.protocols import ReportingScoreRenderer, ScoreDecoder, ScoreRenderer, TonalMetric
from aimusic.render.adapters import RenderResultAdapter
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies
from aimusic.decode import decode_path_to_score
from aimusic.planning.candidates import MetricBoundCandidates, get_valid_next_states
from aimusic.planning.protocols import CandidateGenerator
from aimusic.render.midi_render import render_midi
from aimusic.scoring.priors import Prior, PriorContext
from aimusic.theory.edo import EDO
from aimusic.theory.tonal import ExistingTonalMetric


class FixedMetric:
    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
        return 0.0 if a_root % edo == b_root % edo else 2.0

    def basic_space_distance(
        self,
        a_root: int,
        a_quality: str,
        b_root: int,
        b_quality: str,
        edo: int = 12,
    ) -> float:
        return self.tonal_distance(a_root, b_root, edo)


class EmptyCandidates:
    def __call__(
        self,
        prev_state: BeatState,
        t: int,
        key: RNGKey,
        d_max: int,
        style_config: StyleConfig | None = None,
        vocabularies: Vocabularies | None = None,
        prior: Prior | None = None,
        context: PriorContext | None = None,
        edo: int | None = None,
        proposal_budget: int | None = None,
        prior_guided_proposals: bool = False,
    ) -> tuple[CandidateGenerationResult, RNGKey]:
        return CandidateGenerationResult(t, prev_state, ()), key.next_key()


class EmptyScoreDecoder:
    def __call__(
        self,
        path: Sequence[BeatState],
        *,
        key: RNGKey,
        decode_config: DecodeConfig | None = None,
        vocabularies: Vocabularies,
        edo: int = 12,
        ticks_per_beat: int = 480,
        tempo_bpm: float = 120.0,
        include_terminal_state: bool = False,
    ) -> tuple[Score, RNGKey]:
        return Score(ticks_per_beat=ticks_per_beat, tempo_bpm=tempo_bpm), key.next_key()


class RecordingRenderer:
    def __init__(self) -> None:
        self.destinations: list[str] = []

    def __call__(
        self,
        notes: Sequence[SymbolicNote] | Score,
        edo: EDO,
        output_path: str,
        ticks_per_beat: int = 480,
        *,
        track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
    ) -> None:
        self.destinations.append(output_path)


candidate: CandidateGenerator = get_valid_next_states
decoder: ScoreDecoder = decode_path_to_score
renderer: ScoreRenderer = render_midi
metric: TonalMetric = ExistingTonalMetric()
metric_bound_candidate: CandidateGenerator = MetricBoundCandidates(metric)

candidate_double: CandidateGenerator = EmptyCandidates()
decoder_double: ScoreDecoder = EmptyScoreDecoder()
renderer_double: ScoreRenderer = RecordingRenderer()
metric_double: TonalMetric = FixedMetric()

reporting_renderer: ReportingScoreRenderer = RenderResultAdapter(render_midi)
reporting_double: ReportingScoreRenderer = RenderResultAdapter(RecordingRenderer())
