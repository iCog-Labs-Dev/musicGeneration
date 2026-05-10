import unittest
import numpy as np

from config import StyleConfig, SBConfig, PriorWeights
from core_types import BeatState, Layer, EndpointDistribution
from graph import build_sparse_graph
from priors import NullPrior
from sb import build_sb_problem, solve_sb, extract_transition_probabilities
from vocab import build_default_vocabularies

class TestPipelineIntegration(unittest.TestCase):
    """
    End-to-end integration test that links concepts from Tonal Theory, 
    Rhythm/Vocabularies, and Candidate Generation up to the construction 
    of the Schrödinger Bridge sparse graph.
    """

    def test_end_to_end_sparse_graph_generation(self):
        # 1. Initialize core configuration and vocabulary mapping
        style = StyleConfig()
        sb_config = SBConfig(k_max=20, d_max=10) # Keep sizes manageable for tests
        vocabs = build_default_vocabularies(style)
        rng = np.random.default_rng(42)
        prior = NullPrior()
        weights = PriorWeights()

        # 2. Define the start constraints (e.g. beginning of a phrase)
        start_state = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=0,
            boundary_lvl=vocabs.boundaries.token_for_label("phrase").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("Cmaj").id,
            role_id=vocabs.roles.token_for_label("change").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        # 3. Define the target constraints (e.g. cadence at the end of the phrase)
        end_state = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=0,
            boundary_lvl=vocabs.boundaries.token_for_label("phrase").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("G7").id,
            role_id=vocabs.roles.token_for_label("cad").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        # 4. Wrap them in layer definitions
        total_beats = 4 # We want to generate a 4-beat transition (1 full bar in 4/4)
        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=total_beats, states=(end_state,))

        # 5. Build the sparse graph
        # This implicitly exercises candidates.py (get_valid_next_states, is_legal_transition),
        # tonal.py (harmonic distances, EDO properties), gttm_features.py (energy scoring), 
        # and priors.py (calculating transition log weights).
        sparse_graph = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=total_beats,
            sb_config=sb_config,
            style_config=style,
            vocabularies=vocabs,
            prior=prior,
            weights=weights,
            rng=rng,
            d_max=sb_config.d_max
        )

        # 6. Verify pipeline outputs
        self.assertIsNotNone(sparse_graph)
        print(sparse_graph.layers)
        
        # We expect a layer for time_index 0, 1, 2, 3, 4 -> 5 layers total
        self.assertEqual(len(sparse_graph.layers), total_beats + 1)
        self.assertEqual(sparse_graph.layers[0].time_index, 0)
        self.assertEqual(sparse_graph.layers[-1].time_index, total_beats)

        # Check diagnostics to ensure candidates were generated and evaluated
        self.assertEqual(len(sparse_graph.diagnostics.layer_diagnostics), total_beats)
        
        first_step_diag = sparse_graph.diagnostics.layer_diagnostics[0]
        self.assertGreater(first_step_diag.raw_candidate_count, 0, "Should have proposed candidates.")
        
        # Verify that edges exist between layers
        self.assertEqual(len(sparse_graph.edges_by_time), total_beats)
        self.assertGreater(len(sparse_graph.edges_by_time[0]), 0, "Should have edges connecting the first step.")

        # 7. Solve Schrödinger Bridge and extract transition probabilities
        pi0 = EndpointDistribution(layer=sparse_graph.layers[0], probabilities=(1.0,))
        piT = EndpointDistribution(layer=sparse_graph.layers[-1], probabilities=(1.0,))
        
        sb_config_for_solver = SBConfig(k_max=sb_config.k_max, d_max=sb_config.d_max, horizon_t=total_beats)
        sb_problem = build_sb_problem(sparse_graph, pi0, piT, sb_config_for_solver)
        sb_solution = solve_sb(sb_problem)
        transition_model = extract_transition_probabilities(sb_solution)
        
        self.assertIsNotNone(transition_model)
        self.assertEqual(len(transition_model.edge_probabilities_by_time), total_beats)
        self.assertTrue(sb_solution.trace.converged, "SB solver should converge")
        
        # Verify that the extracted probabilities sum to 1.0 for the start state
        first_step_probs = transition_model.edge_probabilities_by_time[0]
        self.assertAlmostEqual(sum(first_step_probs), 1.0, places=5)

    def test_impossible_path_rejection(self):
        style = StyleConfig()
        sb_config = SBConfig(k_max=20, d_max=10)
        vocabs = build_default_vocabularies(style)
        rng = np.random.default_rng(42)

        # Start on beat 1 (weak beat) with no boundary
        start_state = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=1,
            boundary_lvl=vocabs.boundaries.token_for_label("none").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("Cmaj").id,
            role_id=vocabs.roles.token_for_label("hold").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        # Target a meter change to 3/4 (illegal without a phrase boundary or downbeat)
        end_state = BeatState(
            meter_id=vocabs.meters.token_for_label("5/4").id,
            beat_in_bar=2,
            boundary_lvl=vocabs.boundaries.token_for_label("none").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("Cmaj").id,
            role_id=vocabs.roles.token_for_label("hold").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=1, states=(end_state,))

        sparse_graph = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=1,
            sb_config=sb_config,
            style_config=style,
            vocabularies=vocabs,
            prior=NullPrior(),
            rng=rng,
            d_max=sb_config.d_max
        )

        # The end_state is unreachable due to constraints, so it should be pruned.
        # The first layer diagnostics should show the candidate was rejected or pruned as unreachable endpoint.
        first_step_diag = sparse_graph.diagnostics.layer_diagnostics[0]
        pruned_reasons = [p.reason for p in first_step_diag.pruned_states]
        
        self.assertIn("unreachable_endpoint", pruned_reasons)
        self.assertEqual(len(sparse_graph.layers[-1]), 0, "Target layer should be empty due to impossible path")

    def test_scoring_accuracy_and_pruning(self):
        style = StyleConfig()
        vocabs = build_default_vocabularies(style)
        
        from gttm_features import calculate_gttm_energy
        from priors import calculate_transition_log_weight
        
        state_Cmaj = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=3,
            boundary_lvl=vocabs.boundaries.token_for_label("none").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("Cmaj").id,
            role_id=vocabs.roles.token_for_label("prep").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )
        
        state_G7_cad = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=0, # downbeat
            boundary_lvl=vocabs.boundaries.token_for_label("phrase").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("G7").id,
            role_id=vocabs.roles.token_for_label("cad").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        state_Fsharp_bad = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=0, 
            boundary_lvl=vocabs.boundaries.token_for_label("phrase").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("F#maj").id,
            role_id=vocabs.roles.token_for_label("cad").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )
        
        prior = NullPrior()
        weights = PriorWeights(lambda_data=1.0, lambda_gttm=1.0)
        
        # Calculate transition log weights
        weight_good = calculate_transition_log_weight(
            state_Cmaj, state_G7_cad, 0, prior=prior, weights=weights, vocabularies=vocabs
        )
        weight_bad = calculate_transition_log_weight(
            state_Cmaj, state_Fsharp_bad, 0, prior=prior, weights=weights, vocabularies=vocabs
        )
        
        # In GTTM, lower energy is better, and log_weight = data - gttm_energy.
        # So a good transition should have a HIGHER log_weight (less negative energy).
        self.assertGreater(weight_good, weight_bad, "Cmaj -> G7 should score higher than Cmaj -> F#maj")

    def test_microtonal_19edo(self):
        style_19 = StyleConfig(key_vocabulary_size=19, chord_vocabulary_size=76)
        sb_config = SBConfig(k_max=10, d_max=5)
        vocabs_19 = build_default_vocabularies(style_19)
        rng = np.random.default_rng(42)

        start_state = BeatState(
            meter_id=vocabs_19.meters.token_for_label("4/4").id,
            beat_in_bar=0,
            boundary_lvl=vocabs_19.boundaries.token_for_label("phrase").id,
            key_id=0,
            chord_id=0,
            role_id=vocabs_19.roles.token_for_label("hold").id,
            head_id=vocabs_19.heads.token_for_label("root").id,
            groove_id=vocabs_19.grooves.token_for_label("straight_8ths").id,
        )

        end_state = BeatState(
            meter_id=vocabs_19.meters.token_for_label("4/4").id,
            beat_in_bar=0,
            boundary_lvl=vocabs_19.boundaries.token_for_label("phrase").id,
            key_id=0,
            chord_id=0,
            role_id=vocabs_19.roles.token_for_label("hold").id,
            head_id=vocabs_19.heads.token_for_label("root").id,
            groove_id=vocabs_19.grooves.token_for_label("straight_8ths").id,
        )

        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=1, states=(end_state,))

        sparse_graph = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=1,
            sb_config=sb_config,
            style_config=style_19,
            vocabularies=vocabs_19,
            prior=NullPrior(),
            rng=rng,
            d_max=sb_config.d_max,
            edo=19
        )
        
        self.assertIsNotNone(sparse_graph)

    def test_neural_prior_integration(self):
        from priors import NeuralPrior
        from config import NeuralPriorConfig, PlaceholderPriorMode
        
        style = StyleConfig()
        vocabs = build_default_vocabularies(style)
        
        prior_config = NeuralPriorConfig(placeholder_mode=PlaceholderPriorMode.STRUCTURED)
        prior = NeuralPrior(config=prior_config)

        start_state = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=0,
            boundary_lvl=vocabs.boundaries.token_for_label("phrase").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("Cmaj").id,
            role_id=vocabs.roles.token_for_label("hold").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        end_state = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=1,
            boundary_lvl=vocabs.boundaries.token_for_label("none").id,
            key_id=vocabs.keys.token_for_label("C").id,
            chord_id=vocabs.chords.token_for_label("G7").id,
            role_id=vocabs.roles.token_for_label("cad").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )

        logp = prior.logp_next(start_state, end_state, 1)
        self.assertNotEqual(logp, 0.0, "Structured placeholder prior should return non-zero logp")

if __name__ == "__main__":
    unittest.main()
