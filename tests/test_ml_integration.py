import math
import tempfile
import unittest
from pathlib import Path

from aimusic.core.config import PriorWeights, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.core.rng import RNGKey
from aimusic.scoring.priors import NullPrior, calculate_transition_log_weight
from tests.conftest import HAS_JAX, requires_jax, skip_unless_jax
from tests.test_midi_ingest import write_simple_c_g_progression

if HAS_JAX:
    from aimusic.ml.inference import load_trained_neural_prior
    from aimusic.ml.train import train_prior_from_corpus


def _state(chord: str = "Cmaj") -> BeatState:
    return BeatState(
        meter_id=DEFAULT_VOCABULARIES.meters.token_for_label("4/4").id,
        beat_in_bar=0,
        boundary_lvl=0,
        key_id=DEFAULT_VOCABULARIES.keys.token_for_label("C").id,
        chord_id=DEFAULT_VOCABULARIES.chords.token_for_label(chord).id,
        role_id=DEFAULT_VOCABULARIES.roles.token_for_label("hold").id,
        head_id=DEFAULT_VOCABULARIES.heads.token_for_label("root").id,
        groove_id=DEFAULT_VOCABULARIES.grooves.token_for_label("straight_8ths").id,
    )


@requires_jax
@skip_unless_jax
class TestMLIntegration(unittest.TestCase):
    def test_loaded_prior_runs_method_a(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_dir = Path(tmp_dir) / "corpus"
            midi_dir.mkdir()
            write_simple_c_g_progression(midi_dir / "piece.mid")
            bundle_dir = Path(tmp_dir) / "bundle"
            train_prior_from_corpus(
                midi_dir,
                bundle_dir,
                style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated", "swing")),
            )
            prior = load_trained_neural_prior(str(bundle_dir))
            run_config = MethodARunConfig(
                total_beats=4,
                seed=7,
                style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated", "swing")),
            )
            result, _ = run_method_a(run_config, key=RNGKey(seed=7), prior=prior)

        self.assertGreater(len(result.path), 1)

    def test_trained_prior_changes_transition_weight(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_dir = Path(tmp_dir) / "corpus"
            midi_dir.mkdir()
            write_simple_c_g_progression(midi_dir / "piece.mid")
            bundle_dir = Path(tmp_dir) / "bundle"
            train_prior_from_corpus(
                midi_dir,
                bundle_dir,
                style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated", "swing")),
            )
            prior = load_trained_neural_prior(str(bundle_dir))

        prev = _state("Cmaj")
        nxt = _state("G7")
        weights = PriorWeights(lambda_data=1.0, lambda_gttm=0.0)
        null_weight = calculate_transition_log_weight(
            prev, nxt, 0, prior=NullPrior(), weights=weights
        )
        trained_weight = calculate_transition_log_weight(
            prev, nxt, 0, prior=prior, weights=weights
        )
        self.assertNotEqual(null_weight, trained_weight)
        self.assertTrue(math.isfinite(trained_weight))


if __name__ == "__main__":
    unittest.main()
