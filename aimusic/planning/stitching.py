"""Pure boundary costs and exact shared-endpoint path assembly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aimusic.core.config import StitchingConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import Vocabularies


@dataclass(frozen=True)
class StitchCost:
    key: float
    chord: float
    meter: float
    groove: float
    role: float
    head: float
    boundary: float
    total: float


def stitch_cost(
    left: BeatState,
    right: BeatState,
    config: StitchingConfig,
    vocabularies: Vocabularies,
    edo: int,
) -> StitchCost:
    """Pitch distances use EDO roots, never arithmetic on vocabulary IDs.

    Chords combine root displacement and quality mismatch equally. Other
    categorical features use 0/1 mismatch; boundary levels use |delta|/3.
    Every component is in [0, 1] for the project's boundary vocabulary.
    """
    def pitch_distance(a: int, b: int) -> float:
        delta = (a - b) % edo
        return min(delta, edo - delta) / max(1, edo // 2)

    left_chord = vocabularies.chords.token_for_id(left.chord_id)
    right_chord = vocabularies.chords.token_for_id(right.chord_id)
    components = (
        pitch_distance(vocabularies.keys.token_for_id(left.key_id).root_pc,
                       vocabularies.keys.token_for_id(right.key_id).root_pc),
        0.5 * (pitch_distance(left_chord.root_pc, right_chord.root_pc)
               + float(left_chord.quality != right_chord.quality)),
        float(left.meter_id != right.meter_id),
        float(left.groove_id != right.groove_id),
        float(left.role_id != right.role_id),
        float(left.head_id != right.head_id),
        min(1.0, abs(left.boundary_lvl - right.boundary_lvl) / 3.0),
    )
    weights = (
        config.key_weight, config.chord_weight, config.meter_weight,
        config.groove_weight, config.role_weight, config.head_weight,
        config.boundary_weight,
    )
    return StitchCost(*components, total=sum(a * b for a, b in zip(components, weights)))


@dataclass(frozen=True)
class StitchPolicy:
    """Apply costs only to edges touching an internal section boundary."""

    config: StitchingConfig
    vocabularies: Vocabularies
    edo: int
    edge_times: tuple[int, ...]

    def cost(self, left: BeatState, right: BeatState, time_index: int) -> float:
        if time_index not in self.edge_times:
            return 0.0
        return stitch_cost(left, right, self.config, self.vocabularies, self.edo).total


@dataclass(frozen=True)
class SectionJoinDiagnostics:
    left_section: str
    right_section: str
    time_index: int
    shared_state: BeatState
    incoming: StitchCost
    outgoing: StitchCost
    tolerance: float


def concatenate_section_paths(paths: Sequence[Sequence[BeatState]]) -> tuple[BeatState, ...]:
    combined: list[BeatState] = []
    for path in paths:
        if len(path) < 2:
            raise ValueError("Each section must have at least one transition.")
        if combined and combined[-1] != path[0]:
            raise ValueError("Adjacent sections do not share their endpoint.")
        combined.extend(path[1:] if combined else path)
    return tuple(combined)
