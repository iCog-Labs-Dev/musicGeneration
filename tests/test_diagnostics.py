import tempfile
import json
from pathlib import Path
import unittest
import dataclasses
import math
from unittest.mock import MagicMock

# --- Diagnostics Imports ---
from aimusic.core.diagnostics import (
    TimelineEvent, 
    StructuralDiagnostics, 
    compute_tension_curve,
    RunManifest,
    SBDiagnostics
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

if __name__ == "__main__":
    unittest.main()