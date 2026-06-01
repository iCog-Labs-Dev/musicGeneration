import tempfile
import unittest
import math

from aimusic.core.config import (
    NeuralPriorConfig,
    PlaceholderPriorMode,
    PriorFactorization,
    PriorWeights,
)
from aimusic.core.core_types import BeatState
from aimusic.scoring.gttm_features import TransitionWindow, calculate_gttm_energy
from aimusic.scoring.priors import (
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
from aimusic.core.vocab import DEFAULT_VOCABULARIES

from aimusic.scoring.priors import (
    NeuralPriorScoringStats,
    NeuralPriorSession,
    FactorizedStreamScorer,
    open_neural_prior_session,
    load_neural_prior,
    _chunk,
)
from unittest.mock import MagicMock

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


class TestNeuralPriorScoringStats(unittest.TestCase):
    def test_model_fraction_zero_when_no_queries(self):
        stats = NeuralPriorScoringStats()
        self.assertAlmostEqual(stats.model_fraction, 0.0)

    def test_model_fraction_correct(self):
        stats = NeuralPriorScoringStats(total_queries=10, model_calls=7, placeholder_calls=3)
        self.assertAlmostEqual(stats.model_fraction, 0.7)

    def test_all_placeholder_gives_zero_fraction(self):
        stats = NeuralPriorScoringStats(total_queries=8, model_calls=0, placeholder_calls=8)
        self.assertAlmostEqual(stats.model_fraction, 0.0)

    def test_all_model_gives_one(self):
        stats = NeuralPriorScoringStats(total_queries=10, model_calls=10, placeholder_calls=0)
        self.assertAlmostEqual(stats.model_fraction, 1.0)

    def test_to_dict_has_required_keys(self):
        d = NeuralPriorScoringStats(total_queries=5, model_calls=4, fallback_count=1).to_dict()
        for key in ("total_queries", "model_calls", "placeholder_calls",
                    "batch_count", "fallback_count", "model_fraction"):
            self.assertIn(key, d)

    def test_rejects_negative_counts(self):
        with self.assertRaises((ValueError, TypeError)):
            NeuralPriorScoringStats(total_queries=-1)


class TestChunk(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(_chunk(list(range(6)), 2), [[0, 1], [2, 3], [4, 5]])

    def test_uneven_split(self):
        self.assertEqual(_chunk(list(range(5)), 2), [[0, 1], [2, 3], [4]])

    def test_empty_input(self):
        self.assertEqual(_chunk([], 3), [])

    def test_size_larger_than_input(self):
        self.assertEqual(_chunk([1, 2], 10), [[1, 2]])

    def test_size_one(self):
        self.assertEqual(_chunk([10, 20, 30], 1), [[10], [20], [30]])

    def test_splits_12_into_3_batches_of_4(self):
        result = _chunk(list(range(12)), 4)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(len(c) == 4 for c in result))

    def test_splits_10_into_3_batches(self):
        result = _chunk(list(range(10)), 4)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2], list(range(8, 10)))


class TestNeuralPriorSession(unittest.TestCase):
    def _make_session(self):
        prior = NeuralPrior(config=NeuralPriorConfig())
        return NeuralPriorSession(prior)

    def test_finalize_seals_session(self):
        session = self._make_session()
        session.finalize()
        s = state()
        with self.assertRaises(RuntimeError):
            session.score(s, s, 0)

    def test_finalize_twice_raises(self):
        session = self._make_session()
        session.finalize()
        with self.assertRaises(RuntimeError):
            session.finalize()

    def test_placeholder_mode_returns_finite_score(self):
        session = self._make_session()
        s = state()
        logp = session.score(s, s, 0)
        self.assertTrue(math.isfinite(logp))

    def test_stats_total_matches_call_count(self):
        session = self._make_session()
        s = state()
        for _ in range(5):
            session.score(s, s, 0)
        stats = session.finalize()
        self.assertEqual(stats.total_queries, 5)

    def test_batch_score_returns_correct_count(self):
        from aimusic.scoring.priors import PriorQuery
        session = self._make_session()
        s = state()
        queries = [PriorQuery(prev_state=s, next_state=s, time_index=i) for i in range(8)]
        logps = session.score_batch(queries)
        self.assertEqual(len(logps), 8)
        self.assertTrue(all(math.isfinite(lp) for lp in logps))

    def test_fallback_not_double_counted_in_factorized_mode(self):
        from aimusic.scoring.priors import NeuralPriorManifest
        bad_model = MagicMock()
        bad_model.score_transition = MagicMock(side_effect=RuntimeError("bad"))
        manifest = NeuralPriorManifest(
            factorization_mode=PriorFactorization.FACTORIZED,
            token_streams=("meter", "key"),
        )
        prior = NeuralPrior(
            config=NeuralPriorConfig(factorization_mode=PriorFactorization.FACTORIZED),
            manifest=manifest,
            model=bad_model,
        )
        session = NeuralPriorSession(prior)
        s = state()
        session.score(s, s, 0)
        stats = session.finalize()
        # 2 streams both failed → fallback_count should be 2, not 4
        self.assertEqual(stats.fallback_count, 2)

    def test_score_and_batch_cannot_be_used_after_finalize(self):
        session = self._make_session()
        session.finalize()
        s = state()
        with self.assertRaises(RuntimeError):
            session.score(s, s, 0)
        session2 = self._make_session()
        session2.finalize()
        with self.assertRaises(RuntimeError):
            session2.score_batch([])


class TestLoadNeuralPrior(unittest.TestCase):
    def test_returns_neural_prior_instance(self):
        prior = load_neural_prior(NeuralPriorConfig())
        self.assertIsInstance(prior, NeuralPrior)

    def test_placeholder_when_no_model(self):
        prior = load_neural_prior(NeuralPriorConfig(), model=None)
        self.assertIsNone(prior.model)

    def test_model_attached_when_provided(self):
        model = MagicMock(spec=["score_transition"])
        model.score_transition = MagicMock(return_value=0.0)
        prior = load_neural_prior(NeuralPriorConfig(), model=model)
        self.assertIs(prior.model, model)

    def test_factorization_mismatch_raises(self):
        from aimusic.scoring.priors import NeuralPriorManifest
        config = NeuralPriorConfig(factorization_mode=PriorFactorization.FACTORIZED)
        manifest = NeuralPriorManifest(factorization_mode=PriorFactorization.WHOLE_STATE)
        with self.assertRaises(ValueError):
            load_neural_prior(config, manifest_override=manifest)

    def test_matching_modes_do_not_raise(self):
        from aimusic.scoring.priors import NeuralPriorManifest
        for mode in (PriorFactorization.FACTORIZED, PriorFactorization.WHOLE_STATE, PriorFactorization.MIXED):
            config = NeuralPriorConfig(factorization_mode=mode)
            manifest = NeuralPriorManifest(factorization_mode=mode)
            prior = load_neural_prior(config, manifest_override=manifest)
            self.assertIs(prior.manifest.factorization_mode, mode)


class TestOpenNeuralPriorSession(unittest.TestCase):
    def test_returns_session_instance(self):
        prior = NeuralPrior(config=NeuralPriorConfig())
        session = open_neural_prior_session(prior)
        self.assertIsInstance(session, NeuralPriorSession)

    def test_session_is_not_sealed(self):
        prior = NeuralPrior(config=NeuralPriorConfig())
        session = open_neural_prior_session(prior)
        logp = session.score(state(), state(), 0)
        self.assertTrue(math.isfinite(logp))

    def test_each_call_returns_independent_session(self):
        prior = NeuralPrior(config=NeuralPriorConfig())
        self.assertIsNot(open_neural_prior_session(prior), open_neural_prior_session(prior))


class TestFactorizedStreamScorer(unittest.TestCase):
    def _make_query(self):
        from aimusic.scoring.priors import (
            TokenizedPriorQuery, StructuralEventTokens, StructuralTokenSequence,
        )
        event = StructuralEventTokens(0, 0, 0, 0, 0, 0, 0, 0)
        return TokenizedPriorQuery(
            prev_event=event, next_event=event, time_index=0,
            history_tokens=StructuralTokenSequence(),
            future_hint_tokens=StructuralTokenSequence(),
            factorization_mode=PriorFactorization.FACTORIZED,
        )

    def test_sums_per_stream_scores(self):
        model = MagicMock()
        model.score_transition = MagicMock(return_value=0.5)
        scorer = FactorizedStreamScorer(
            model=model, active_streams=("meter", "key", "chord"), default_logp=0.0,
        )
        fallback = NeuralPrior(config=NeuralPriorConfig())
        total, fb = scorer.score(self._make_query(), fallback_scorer=fallback)
        self.assertAlmostEqual(total, 1.5)
        self.assertEqual(fb, 0)
        self.assertEqual(model.score_transition.call_count, 3)

    def test_fallback_on_stream_exception(self):
        model = MagicMock()
        model.score_transition = MagicMock(side_effect=RuntimeError("dead"))
        scorer = FactorizedStreamScorer(
            model=model, active_streams=("meter", "key"), default_logp=0.0,
        )
        fallback = NeuralPrior(config=NeuralPriorConfig())
        total, fb = scorer.score(self._make_query(), fallback_scorer=fallback)
        self.assertEqual(fb, 2)
        self.assertTrue(math.isfinite(total))

    def test_rejects_empty_streams(self):
        model = MagicMock(spec=["score_transition"])
        model.score_transition = MagicMock(return_value=0.0)
        with self.assertRaises(ValueError):
            FactorizedStreamScorer(model=model, active_streams=())


class TestPlaceholderAllFactorizationModes(unittest.TestCase):
    """Placeholder-only (model=None) must return finite scores for every mode."""

    def _run(self, mode):
        from aimusic.scoring.priors import NeuralPriorManifest
        config = NeuralPriorConfig(factorization_mode=mode)
        manifest = NeuralPriorManifest(factorization_mode=mode)
        prior = NeuralPrior(config=config, manifest=manifest, model=None)
        session = NeuralPriorSession(prior)
        s = state()
        logp = session.score(s, s, 0)
        stats = session.finalize()
        self.assertTrue(math.isfinite(logp))
        self.assertEqual(stats.total_queries, 1)
        self.assertEqual(stats.model_calls, 0)
        self.assertEqual(stats.placeholder_calls, 1)
        self.assertAlmostEqual(stats.model_fraction, 0.0)

    def test_placeholder_factorized(self):
        self._run(PriorFactorization.FACTORIZED)

    def test_placeholder_whole_state(self):
        self._run(PriorFactorization.WHOLE_STATE)

    def test_placeholder_mixed(self):
        self._run(PriorFactorization.MIXED)

if __name__ == "__main__":
    unittest.main()
