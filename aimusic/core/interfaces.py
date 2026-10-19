"""Public protocol seams for replaceable musical system components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, Tuple, runtime_checkable

from aimusic.core.config import DecodeConfig, EDOConfig, StyleConfig
from aimusic.core.core_types import BeatState, NoteEvent, Score
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies


@runtime_checkable
class TonalMetric(Protocol):
    def distance(self, left_root: int, right_root: int, edo: int) -> float: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    def propose(
        self,
        state: BeatState,
        time_index: int,
        *,
        style_config: StyleConfig,
        vocabularies: Vocabularies,
        rng: RNGKey,
    ) -> Tuple[Tuple[BeatState, ...], RNGKey]: ...


@runtime_checkable
class TrackDecoder(Protocol):
    def decode(
        self,
        path: Sequence[BeatState],
        *,
        config: DecodeConfig,
        vocabularies: Vocabularies,
        edo: int,
        rng: RNGKey,
    ) -> Tuple[Tuple[NoteEvent, ...], RNGKey]: ...


@runtime_checkable
class ScoreDecoder(Protocol):
    def decode(
        self,
        path: Sequence[BeatState],
        *,
        config: DecodeConfig,
        vocabularies: Vocabularies,
        edo: int,
        rng: RNGKey,
    ) -> Tuple[Score, RNGKey]: ...


@runtime_checkable
class ScoreRenderer(Protocol):
    def render(
        self,
        score: Score,
        output_path: str,
        *,
        options: Mapping[str, object],
    ) -> None: ...


@dataclass(frozen=True)
class RuleBasedTonalMetric:
    """Default EDO-generic tonal metric implementation."""

    def distance(self, left_root: int, right_root: int, edo: int) -> float:
        from aimusic.theory.tonal import tonal_distance

        return float(tonal_distance(left_root, right_root, edo))


@dataclass(frozen=True)
class AlgorithmicCandidateGenerator:
    """Functional adapter over the default bounded candidate generator."""

    d_max: int = 8

    def propose(
        self,
        state: BeatState,
        time_index: int,
        *,
        style_config: StyleConfig,
        vocabularies: Vocabularies,
        rng: RNGKey,
    ) -> Tuple[Tuple[BeatState, ...], RNGKey]:
        import numpy as np

        from aimusic.planning.candidates import get_valid_next_states

        seed = rng.generator().randrange(0, 2**63)
        result = get_valid_next_states(
            state,
            time_index,
            np.random.default_rng(seed),
            self.d_max,
            style_config=style_config,
            vocabularies=vocabularies,
        )
        return result.states, rng.next_key()


@dataclass(frozen=True)
class AlgorithmicScoreDecoder:
    """Protocol-compatible adapter for the pure rule-based decoder."""

    def decode(
        self,
        path: Sequence[BeatState],
        *,
        config: DecodeConfig,
        vocabularies: Vocabularies,
        edo: int,
        rng: RNGKey,
    ) -> Tuple[Score, RNGKey]:
        from aimusic.decode import decode_path_to_score

        score = decode_path_to_score(
            path,
            decode_config=config,
            vocabularies=vocabularies,
            edo=edo,
        )
        # The default decoder is deterministic and therefore consumes no key.
        return score, rng


@dataclass(frozen=True)
class MidiScoreRenderer:
    """Default renderer adapter that keeps MIDI concerns outside planning."""

    edo_config: EDOConfig

    def render(
        self,
        score: Score,
        output_path: str,
        *,
        options: Mapping[str, object],
    ) -> None:
        from aimusic.render.midi_render import render_midi
        from aimusic.theory.edo import EDO

        track_instruments = options.get("track_instruments")
        render_midi(
            score,
            EDO(self.edo_config),
            output_path,
            track_instruments=track_instruments,  # type: ignore[arg-type]
        )
