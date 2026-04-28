import tempfile
import unittest

from config import (
    NeuralPriorConfig,
    PlaceholderPriorMode,
    PriorFactorization,
    PriorWeights,
)
from core_types import BeatState
from gttm_features import TransitionWindow, calculate_gttm_energy
from priors import (
    NeuralPrior,
    NeuralPriorManifest,
    NullPrior,
    PriorContext,
    PriorQuery,
    StructuralEventTokens,
    StructuralTokenSequence,
    TokenizedPriorQuery,
    build_neural_prior_manifest,
    calculate_transition_log_weight,
    calculate_transition_log_weights,
    load_neural_prior_manifest,
    prior_logps,
    save_neural_prior_manifest,
)
from vocab import DEFAULT_VOCABULARIES


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


class DummyScalarModel:
    def score_transition(self, query: TokenizedPriorQuery) -> float:
        return float(query.next_event.chord_id - query.prev_event.chord_id)


class DummyBatchModel:
    def score_transition(self, query: TokenizedPriorQuery) -> float:
        return float(query.time_index)

    def score_transition_batch(self, queries):
        return tuple(self.score_transition(query) + 0.5 for query in queries)


class TestStructuralTokenContracts(unittest.TestCase):
    def test_structural_event_tokens_follow_beatstate_fields(self):
        beat_state = state(beat=2, boundary="phrase", chord="G7", role="prep")
        tokens = StructuralEventTokens.from_state(beat_state)

        self.assertEqual(tokens.beat_position, beat_state.beat_in_bar)
        self.assertEqual(tokens.boundary_level, beat_state.boundary_lvl)
        self.assertEqual(tokens.chord_id, beat_state.chord_id)

    def test_structural_token_sequence_factorizes_state_history(self):
        states = (
            state(beat=0, chord="Cmaj"),
            state(beat=1, chord="G7", role="prep"),
            state(beat=2, chord="Cmaj", role="cad"),
        )
        sequence = StructuralTokenSequence.from_states(states)

        self.assertEqual(len(sequence), 3)
        self.assertEqual(sequence.chord_ids, tuple(item.chord_id for item in states))
        self.assertEqual(sequence.event_at(1).role_id, states[1].role_id)

    def test_prior_context_derives_history_and_future_tokens(self):
        history = (state(beat=0, chord="Cmaj"), state(beat=1, chord="G7"))
        future = (state(beat=0, boundary="phrase", chord="Cmaj"),)
        context = PriorContext(
            history=history,
            future_hints=future,
            section_name="intro",
            metadata=(("plan", "method_a"),),
        )

        self.assertEqual(context.history_tokens.chord_ids, (history[0].chord_id, history[1].chord_id))
        self.assertEqual(context.future_hint_tokens.boundary_levels, (future[0].boundary_lvl,))
        self.assertEqual(context.section_name, "intro")


class TestManifestIO(unittest.TestCase):
    def test_manifest_round_trips_through_json(self):
        manifest = NeuralPriorManifest(
            model_family="flax_transformer",
            model_version="v2",
            factorization_mode=PriorFactorization.FACTORIZED,
            checkpoint_path="artifacts/model.ckpt",
            tokenizer_path="artifacts/tokens.json",
            expected_edo=19,
            metadata=(("owner", "corpus-team"),),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = f"{tmp_dir}/prior_manifest.json"
            save_neural_prior_manifest(manifest, manifest_path)
            loaded = load_neural_prior_manifest(manifest_path)

        self.assertEqual(loaded, manifest)

    def test_manifest_can_be_built_from_runtime_config(self):
        config = NeuralPriorConfig(
            model_family="torch_transformer",
            model_version="placeholder-v2",
            checkpoint_path="artifacts/model.pt",
            tokenizer_path="artifacts/tokens.json",
            supports_batch_scoring=False,
        )
        manifest = build_neural_prior_manifest(config)

        self.assertEqual(manifest.model_family, "torch_transformer")
        self.assertFalse(manifest.supports_batch_scoring)


class TestNullPrior(unittest.TestCase):
    def test_null_prior_is_constant_for_scalar_and_batch_queries(self):
        prior = NullPrior(neutral_logp=-0.125)
        query = PriorQuery(state(chord="G7"), state(chord="Cmaj"), 4)

        self.assertEqual(
            prior.logp_next(query.prev_state, query.next_state, query.time_index),
            -0.125,
        )
        self.assertEqual(prior_logps(prior, (query, query)), (-0.125, -0.125))


class TestNeuralPriorPlaceholder(unittest.TestCase):
    def setUp(self):
        self.history = (state(beat=0, chord="Cmaj"), state(beat=1, chord="G7", role="prep"))
        self.context = PriorContext(history=self.history, section_name="intro")
        self.prev = state(beat=3, chord="G7", role="prep", head="upper_approach")
        self.next_state = state(
            beat=0,
            boundary="phrase",
            chord="Cmaj",
            role="cad",
            head="root",
        )
        self.query = PriorQuery(self.prev, self.next_state, 8, context=self.context)

    def test_placeholder_scores_are_deterministic_and_batch_matches_scalar(self):
        prior = NeuralPrior(
            config=NeuralPriorConfig(
                placeholder_mode=PlaceholderPriorMode.STRUCTURED,
                default_logp=-0.2,
            )
        )

        scalar = prior.logp_next(self.prev, self.next_state, 8, self.context)
        batched = prior.logp_next_batch((self.query, self.query))

        self.assertEqual(batched, (scalar, scalar))
        self.assertAlmostEqual(scalar, prior.logp_next(self.prev, self.next_state, 8, self.context))

    def test_neutral_placeholder_returns_constant_default_logp(self):
        prior = NeuralPrior(
            config=NeuralPriorConfig(
                placeholder_mode=PlaceholderPriorMode.NEUTRAL,
                default_logp=-0.33,
            )
        )

        self.assertEqual(prior.logp_next(self.prev, self.next_state, 8, self.context), -0.33)


class TestNeuralPriorModelWrapping(unittest.TestCase):
    def test_wrapper_uses_external_scalar_model_when_present(self):
        prior = NeuralPrior(
            config=NeuralPriorConfig(supports_batch_scoring=False),
            model=DummyScalarModel(),
        )
        prev = state(chord="Cmaj")
        next_state = state(chord="G7")

        self.assertEqual(
            prior.logp_next(prev, next_state, 3),
            float(next_state.chord_id - prev.chord_id),
        )

    def test_wrapper_prefers_external_batch_model_for_batch_queries(self):
        prior = NeuralPrior(
            config=NeuralPriorConfig(supports_batch_scoring=True),
            model=DummyBatchModel(),
        )
        queries = (
            PriorQuery(state(chord="Cmaj"), state(chord="G7"), 3),
            PriorQuery(state(chord="G7"), state(chord="Cmaj"), 5),
        )

        self.assertEqual(prior.logp_next_batch(queries), (3.5, 5.5))


class TestPriorScoringIntegration(unittest.TestCase):
    def setUp(self):
        self.prev = state(beat=3, chord="G7", role="prep", head="upper_approach")
        self.next_state = state(
            beat=0,
            boundary="phrase",
            chord="Cmaj",
            role="cad",
            head="root",
        )
        self.context = PriorContext(history=(state(beat=0, chord="Cmaj"), self.prev))
        self.window = TransitionWindow(right_state=state(beat=1, chord="Cmaj", role="hold"))
        self.weights = PriorWeights(lambda_data=0.75, lambda_gttm=1.25, harmonic=1.5)

    def test_transition_log_weight_matches_explicit_formula_for_null_prior(self):
        prior = NullPrior(neutral_logp=0.0)
        expected = -self.weights.lambda_gttm * calculate_gttm_energy(
            self.prev,
            self.next_state,
            8,
            window=self.window,
            weights=self.weights,
        )

        self.assertAlmostEqual(
            calculate_transition_log_weight(
                self.prev,
                self.next_state,
                8,
                prior=prior,
                context=self.context,
                window=self.window,
                weights=self.weights,
            ),
            expected,
        )

    def test_transition_log_weight_swaps_priors_without_api_changes(self):
        null_prior = NullPrior()
        neural_prior = NeuralPrior(
            config=NeuralPriorConfig(
                placeholder_mode=PlaceholderPriorMode.STRUCTURED,
                default_logp=-0.1,
            )
        )

        null_weight = calculate_transition_log_weight(
            self.prev,
            self.next_state,
            8,
            prior=null_prior,
            context=self.context,
            window=self.window,
            weights=self.weights,
        )
        neural_weight = calculate_transition_log_weight(
            self.prev,
            self.next_state,
            8,
            prior=neural_prior,
            context=self.context,
            window=self.window,
            weights=self.weights,
        )

        self.assertNotEqual(null_weight, neural_weight)

    def test_batch_log_weight_matches_scalar_calls(self):
        prior = NeuralPrior()
        queries = (
            PriorQuery(self.prev, self.next_state, 8, self.context),
            PriorQuery(state(chord="Cmaj"), state(chord="Cmaj", beat=1), 9, self.context),
        )
        windows = (
            self.window,
            TransitionWindow(right_state=state(beat=2, chord="Cmaj")),
        )

        batched = calculate_transition_log_weights(
            queries,
            prior=prior,
            windows=windows,
            weights=self.weights,
        )
        scalar = tuple(
            calculate_transition_log_weight(
                query.prev_state,
                query.next_state,
                query.time_index,
                prior=prior,
                context=query.context,
                window=windows[idx],
                weights=self.weights,
            )
            for idx, query in enumerate(queries)
        )

        self.assertEqual(batched, scalar)


if __name__ == "__main__":
    unittest.main()
