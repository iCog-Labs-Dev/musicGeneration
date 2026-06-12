from __future__ import annotations

import unittest
import numpy as np

from aimusic.core.config import SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Layer
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.graph import build_sparse_graph


VOCABS = DEFAULT_VOCABULARIES


def state(
    *,
    meter: str = "4/4",
    beat: int = 0,
    boundary: str = "none",
    key: str = "C",
    chord: str = "Cmaj",
    role: str = "hold",
    head: str = "root",
    groove: str = "straight_8ths",
) -> BeatState:
    return BeatState(
        meter_id=VOCABS.meters.token_for_label(meter).id,
        beat_in_bar=beat,
        boundary_lvl=VOCABS.boundaries.token_for_label(boundary).id,
        key_id=VOCABS.keys.token_for_label(key).id,
        chord_id=VOCABS.chords.token_for_label(chord).id,
        role_id=VOCABS.roles.token_for_label(role).id,
        head_id=VOCABS.heads.token_for_label(head).id,
        groove_id=VOCABS.grooves.token_for_label(groove).id,
    )


class RecordingBatchedPrior:
    def __init__(self) -> None:
        self.scalar_calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context=None,
    ) -> float:
        del prev_state, next_state, t, context
        self.scalar_calls += 1
        return 0.0

    def logp_next_batch(self, queries) -> tuple[float, ...]:
        items = tuple(queries)
        self.batch_calls += 1
        self.batch_sizes.append(len(items))
        return tuple(0.0 for _ in items)


class TestSparseGraphBatchScoring(unittest.TestCase):
    def test_build_sparse_graph_uses_batched_prior_scoring_for_edge_weights(self) -> None:
        style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        sb_config = SBConfig(horizon_t=3, k_max=3, d_max=2)
        prior = RecordingBatchedPrior()
        start_state = state(
            beat=1,
            key="C",
            chord="G7",
            role="prep",
            head="upper_approach",
            groove="straight_8ths",
        )
        end_state = state(
            beat=0,
            boundary="phrase",
            key="C",
            chord="Cmaj",
            role="cad",
            head="root",
            groove="straight_8ths",
        )
        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=3, states=(end_state,))

        build_sparse_graph(
            start_layer,
            end_layer,
            3,
            sb_config=sb_config,
            style_config=style,
            vocabularies=VOCABS,
            prior=prior,
            rng=np.random.default_rng(42),
            d_max=sb_config.d_max,
        )

        self.assertGreater(prior.batch_calls, 0)
        self.assertTrue(all(size > 0 for size in prior.batch_sizes))
        self.assertEqual(prior.scalar_calls, 0)


if __name__ == "__main__":
    unittest.main()
