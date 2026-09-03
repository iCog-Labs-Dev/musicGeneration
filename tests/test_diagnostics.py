import tempfile
import json
import copy
from pathlib import Path
import unittest
import dataclasses
import math
from typing import Any
from unittest.mock import MagicMock

# --- Diagnostics Imports ---
from aimusic.core.diagnostics import (
    ManifestValidationError,
    TimelineEvent,
    StructuralDiagnostics,
    compute_tension_curve,
    RunManifest,
    SBDiagnostics,
    build_run_manifest,
)
# --- Math Pipeline Imports ---
from aimusic.core.config import SBConfig, SBBackend
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.planning.graph import SparseGraph
from aimusic.planning.sb import (
    build_sb_problem,
    solve_sb,
    map_bridge_path,
    solved_bridge_from_solution
)
from aimusic.render.midi_render import SymbolicNote, render_midi
from aimusic.theory.edo import EDO, EDOConfig

class TestDiagnostics(unittest.TestCase):
    def test_timeline_event_serialization(self):
        """Ensures timeline events serialize properly using the standard asdict."""
        event = TimelineEvent(start_time=0.0, end_time=4.0, label="C Major")
        serialized = dataclasses.asdict(event)

        self.assertEqual(serialized["start_time"], 0.0)
        self.assertEqual(serialized["end_time"], 4.0)
        self.assertEqual(serialized["label"], "C Major")

    def test_structural_diagnostics_to_dict(self):
        """Verifies that EVERY timeline array converts safely to JSON structures."""
        struct = StructuralDiagnostics(
            key_timeline=[TimelineEvent(0.0, 4.0, "C Major")],
            chord_timeline=[TimelineEvent(0.0, 2.0, "Cmaj7")],
            role_timeline=[TimelineEvent(0.0, 2.0, "Tonic")],
            groove_timeline=[TimelineEvent(0.0, 4.0, "Swing")],
            boundaries=[0.0, 4.0],
            tension_curve=[(0.0, 0.1), (4.0, 0.9)]
        )

        data = struct.to_dict()

        # Exhaustively checking every single key to prevent silent failures
        self.assertIn("key_timeline", data)
        self.assertIn("chord_timeline", data)
        self.assertIn("role_timeline", data)
        self.assertIn("groove_timeline", data)
        self.assertIn("boundaries", data)
        self.assertIn("tension_curve", data)

        # Verify nested data is accurate
        self.assertEqual(data["key_timeline"][0]["label"], "C Major")
        self.assertEqual(data["chord_timeline"][0]["label"], "Cmaj7")
        self.assertEqual(data["role_timeline"][0]["label"], "Tonic")
        self.assertEqual(data["groove_timeline"][0]["label"], "Swing")
        self.assertEqual(data["boundaries"], [0.0, 4.0])

    def test_compute_tension_curve(self):
        """Tests the heuristic math mapping musical roles to tension floats."""
        roles = [
            TimelineEvent(0.0, 4.0, "Tonic"),
            TimelineEvent(4.0, 8.0, "Subdominant"),
            TimelineEvent(8.0, 12.0, "Dominant"),
            TimelineEvent(12.0, 16.0, "Unknown")
        ]

        curve = compute_tension_curve(roles)

        self.assertEqual(len(curve), 4)
        self.assertEqual(curve[0], (0.0, 0.1))
        self.assertEqual(curve[1], (4.0, 0.5))
        self.assertEqual(curve[2], (8.0, 0.9))
        self.assertEqual(curve[3], (12.0, 0.5))

    def test_sb_diagnostics_extraction(self):
        """Tests that SB logs and Effective Entropy are correctly calculated from a solution."""
        # Mock the SBSolution object returned by aimusic.planning.sb
        mock_solution = MagicMock()
        mock_solution.trace.iterations = 42
        mock_solution.trace.converged = True
        mock_solution.trace.final_max_delta = 1e-6

        mock_solution.problem.diagnostics.layer_sizes = (5, 10, 5)
        mock_solution.problem.diagnostics.zero_outdegree_count = 2
        mock_solution.problem.diagnostics.zero_indegree_count = 1

        # Layer 1: Confident (entropy = 0)
        # Layer 2: 50/50 Split (entropy = approx 0.693)
        mock_solution.marginals.node_marginals_by_layer = [
            (1.0, 0.0),
            (0.5, 0.5)
        ]

        #Extract Data
        stats = SBDiagnostics.from_solution(mock_solution)

        #Verify Basic Stats
        self.assertEqual(stats.iterations_run, 42)
        self.assertTrue(stats.converged)
        self.assertEqual(stats.final_max_delta, 1e-6)
        self.assertEqual(stats.layer_sizes, [5, 10, 5])
        self.assertEqual(stats.pruned_nodes, 3) # 2 out + 1 in

        # Verify Shannon Entropy Math
        expected_layer_2_entropy = -(0.5 * math.log(0.5)) * 2
        expected_average_entropy = (0.0 + expected_layer_2_entropy) / 2
        self.assertAlmostEqual(stats.effective_entropy, expected_average_entropy, places=5)

    def test_run_manifest_generation(self):
        """Ensures the top-level manifest generates valid UUIDs and timestamps."""
        manifest = RunManifest(seed=42, config_dump={"edo": 12})
        data = manifest.to_dict()

        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["config"]["edo"], 12)
        self.assertIsNotNone(data["run_id"])
        self.assertIsNotNone(data["timestamp"])
        self.assertIn("structure", data)

    # END-TO-END PASSAGE FIXTURE
    def test_e2e_produce_stable_short_passage(self):
        """
        True E2E Fixture:
        1. Runs the real math engine to get a deterministic path.
        2. Translates it into a musical passage (SymbolicNotes).
        3. PRODUCES physical output files (MIDI and Manifest).
        4. Asserts the production is identical every time.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # SETUP THE MATH FIXTURE (The Short Passage)
            def _make_state(time_idx: int, var_id: int) -> BeatState:
                return BeatState(
                    meter_id=0,
                    beat_in_bar=time_idx,
                    boundary_lvl=0,
                    key_id=0,
                    chord_id=var_id,
                    role_id=0,
                    head_id=0,
                    groove_id=0
                )

            state_start = _make_state(0, 0)
            state_mid_a = _make_state(1, 1)
            state_mid_b = _make_state(1, 2)
            state_end = _make_state(2, 3)

            layer_0 = Layer(time_index=0, states=(state_start,))
            layer_1 = Layer(time_index=1, states=(state_mid_a, state_mid_b))
            layer_2 = Layer(time_index=2, states=(state_end,))

            edges_0 = (
                Edge(source=state_start, target=state_mid_a, log_weight=math.log(0.9), time_index=0),
                Edge(source=state_start, target=state_mid_b, log_weight=math.log(0.1), time_index=0),
            )
            edges_1 = (
                Edge(source=state_mid_a, target=state_end, log_weight=math.log(1.0), time_index=1),
                Edge(source=state_mid_b, target=state_end, log_weight=math.log(1.0), time_index=1),
            )

            graph = SparseGraph(
                layers=(layer_0, layer_1, layer_2),
                edges_by_time=(edges_0, edges_1),
                diagnostics=MagicMock()
            )

            pi0 = EndpointDistribution(layer=layer_0, probabilities=(1.0,))
            piT = EndpointDistribution(layer=layer_2, probabilities=(1.0,))
            config = SBConfig(horizon_t=2, max_iterations=10, tolerance=1e-5, backend_selection=SBBackend.NUMPY)

            # EXECUTE MATH PIPELINE
            problem = build_sb_problem(graph, pi0, piT, config)
            solution = solve_sb(problem)
            bridge = solved_bridge_from_solution(solution)
            path, best_score = map_bridge_path(bridge)

            # Strict math assertion
            self.assertEqual(path, (state_start, state_mid_a, state_end))

            # TRANSLATE TO MUSICAL PASSAGE (Connecting the pipeline)
            state_to_pitch = {
                state_start: 60, # C4
                state_mid_a: 64, # E4
                state_mid_b: 65, # F4 (Should not be picked)
                state_end: 67    # G4
            }

            notes = []
            for i, state in enumerate(path):
                notes.append(SymbolicNote(
                    pitch_height=state_to_pitch[state],
                    start_time=float(i),
                    end_time=float(i + 1)
                ))

            # PRODUCE PHYSICAL OUTPUTS (MIDI and Manifest)
            midi_path = out_dir / "stable_passage.mid"
            manifest_path = out_dir / "stable_passage_manifest.json"

            # Produce MIDI
            edo_12 = EDO(EDOConfig(n=12, base_tuning=60, pitch_bend_range=48))
            render_midi(notes, edo_12, str(midi_path))

            # Produce Manifest
            manifest = RunManifest(
                seed=42,
                config_dump={"edo": 12},
                sb_stats=SBDiagnostics.from_solution(solution)
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest.to_dict(), f)

            # REGRESSION TRAPS ON PRODUCED FILES
            self.assertTrue(midi_path.exists(), "Pipeline failed to produce MIDI file.")
            self.assertTrue(manifest_path.exists(), "Pipeline failed to produce Manifest file.")

            with open(manifest_path, "r") as f:
                saved_manifest = json.load(f)
            self.assertTrue(saved_manifest["sb_stats"]["converged"])
            self.assertEqual(saved_manifest["sb_stats"]["layer_sizes"], [1, 2, 1])

            self.assertEqual(len(notes), 3)
            self.assertEqual(notes[1].pitch_height, 64, "Regression: Pipeline picked wrong structural path")

class TestVersionedRunManifest(unittest.TestCase):
    def _manifest(self, *, sample: bool = False) -> tuple[Any, RunManifest]:
        from aimusic.planning.plans import MethodARunConfig, run_method_a

        plan_result = run_method_a(
            MethodARunConfig(total_beats=4, seed=123, use_sampling=sample)
        )
        return plan_result, build_run_manifest(
            plan_result,
            seed=123,
            config_dump={"test": True},
        )

    def test_complete_manifest_round_trip_is_lossless(self):
        _, manifest = self._manifest()
        self.assertEqual(RunManifest.from_dict(manifest.to_dict()), manifest)

    def test_graph_totals_match_source_records(self):
        plan_result, manifest = self._manifest()
        source_layers = plan_result.graph.diagnostics.layer_diagnostics
        graph = manifest.graph_stats

        self.assertEqual(
            graph.total_pruned,
            sum(len(layer.pruned_states) for layer in source_layers),
        )
        self.assertEqual(
            graph.total_pruned_edges,
            sum(
                layer.outdegree_pruned_count + layer.state_pruned_edge_count
                for layer in source_layers
            ),
        )
        self.assertEqual(
            graph.total_candidate_pruned,
            sum(layer.d_max_pruned_candidate_count for layer in source_layers),
        )
        self.assertEqual(
            graph.total_prune_operations,
            sum(graph.pruning_summary.values()),
        )
        for source, serialized in zip(source_layers, graph.per_layer_stats):
            self.assertEqual(serialized.proposed, source.raw_candidate_count)
            self.assertEqual(serialized.legal, source.legal_candidate_count)
            self.assertEqual(serialized.scored, source.scored_candidate_count)
            self.assertEqual(serialized.retained, source.kept_candidate_count)
            self.assertEqual(serialized.pruned, len(source.pruned_states))

    def test_candidate_flow_accounts_for_duplicates_and_state_pruning(self):
        _, manifest = self._manifest()
        graph = manifest.graph_stats
        self.assertEqual(
            graph.total_legal,
            graph.total_scored
            + graph.total_duplicate_proposals
            + graph.total_candidate_pruned,
        )
        self.assertEqual(
            graph.total_proposed - graph.total_scored,
            sum(graph.rejection_summary.values()),
        )
        for layer in graph.per_layer_stats:
            self.assertEqual(
                layer.legal,
                layer.scored + layer.duplicate_proposals + layer.candidate_pruned,
            )
            self.assertLessEqual(layer.scored_unique_states, layer.scored)

    def test_original_and_solver_endpoint_support_are_both_preserved(self):
        _, manifest = self._manifest()
        endpoints = manifest.endpoint_stats
        self.assertGreater(endpoints.original_pi0_support_size, endpoints.solver_pi0_support_size)
        self.assertGreater(endpoints.original_piT_support_size, endpoints.solver_piT_support_size)
        self.assertGreater(endpoints.unreachable_pi0_mass, 0.0)
        self.assertGreater(endpoints.unreachable_piT_mass, 0.0)

    def test_sampled_path_has_reproducible_log_probability(self):
        first_result, first = self._manifest(sample=True)
        second_result, second = self._manifest(sample=True)
        self.assertEqual(first_result.path, second_result.path)
        self.assertEqual(first.path_stats.path_mode, "sample")
        self.assertIsNotNone(first.path_stats.path_score)
        self.assertEqual(first.path_stats.path_score, second.path_stats.path_score)
        self.assertEqual(
            first.path_stats.path_score,
            first_result.sampled_path.log_probability,
        )

    def test_strict_parser_rejects_malformed_nested_data(self):
        _, manifest = self._manifest()
        valid = manifest.to_dict()
        cases = []

        wrong_seed = copy.deepcopy(valid)
        wrong_seed["seed"] = "123"
        cases.append((wrong_seed, "manifest.seed"))

        wrong_config = copy.deepcopy(valid)
        wrong_config["config"] = []
        cases.append((wrong_config, "manifest.config"))

        wrong_boolean = copy.deepcopy(valid)
        wrong_boolean["sb_stats"]["converged"] = "false"
        cases.append((wrong_boolean, "sb_stats.converged"))

        negative_count = copy.deepcopy(valid)
        negative_count["graph_stats"]["per_layer_stats"][0]["proposed"] = -1
        cases.append((negative_count, "proposed"))

        inconsistent_total = copy.deepcopy(valid)
        inconsistent_total["graph_stats"]["total_pruned"] += 1
        cases.append((inconsistent_total, "total_pruned"))

        non_finite = copy.deepcopy(valid)
        non_finite["sb_stats"]["final_max_delta"] = float("nan")
        cases.append((non_finite, "final_max_delta"))

        missing_nested = copy.deepcopy(valid)
        del missing_nested["path_stats"]["path_mode"]
        cases.append((missing_nested, "path_stats"))

        unsupported = copy.deepcopy(valid)
        unsupported["schema_version"] = "1.99.0"
        cases.append((unsupported, "schema_version"))

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ManifestValidationError, message):
                    RunManifest.from_dict(data)

    def test_unversioned_manifest_migrates_with_warnings(self):
        legacy = {
            "run_id": "legacy-run",
            "timestamp": 1.0,
            "version": "0.1.0",
            "seed": 7,
            "config": {"edo": 12},
            "structure": StructuralDiagnostics().to_dict(),
            "sb_stats": {
                "iterations_run": 2,
                "converged": True,
                "final_max_delta": 0.01,
                "layer_sizes": [1, 2, 1],
                "pruned_nodes": 3,
                "effective_entropy": 0.4,
            },
        }
        manifest = RunManifest.from_dict(legacy)
        self.assertEqual(manifest.schema_version, "1.0.0")
        self.assertEqual(manifest.sb_stats.disconnected_nodes, 3)
        self.assertEqual(manifest.path_stats.path_mode, "unknown")
        self.assertIsNone(manifest.path_stats.path_score)
        self.assertTrue(manifest.migration_warnings)


if __name__ == "__main__":
    unittest.main()
