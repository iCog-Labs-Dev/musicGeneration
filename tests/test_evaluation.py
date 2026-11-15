"""Tests for the MUS-28 evaluation suite (aimusic.evaluation)."""
from __future__ import annotations

import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from aimusic.core.core_types import BeatState, Edge, Layer
from aimusic.core.rng import RNGKey
from aimusic.evaluation.baselines import coreachable_states, reachable_uniform_walk
from aimusic.evaluation.config import expand_matrix, parse_config, select_specs
from aimusic.evaluation.matrix import load_results, run_matrix, write_outputs
from aimusic.evaluation.report import render_report
from aimusic.evaluation.runner import evaluate_run
from aimusic.evaluation.schema import RESULT_SCHEMA_VERSION, RunSpec, new_result
from aimusic.evaluation.stats import (
    aggregate,
    flatten_metrics,
    paired_difference,
    summarize,
    t_critical_95,
    wilson_interval,
)
from aimusic.planning.graph import GraphDiagnostics, SparseGraph

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_config(**overrides: Any) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "schema_version": "mus28-config/1",
        "name": "unit",
        "config_version": "0.0.1",
        "horizons": [8],
        "seeds": [0, 1],
        "edos": [12],
        "path_modes": ["sample"],
        "conditions": ["no_prior"],
        "graph_grid": [{"k_max": 64, "d_max": 8, "proposal_budget": 256}],
        "prior_guided_proposals": False,
        "sb": {"max_iterations": 200, "tolerance": 1e-6, "temperature": 1.0},
        "timeout_s": 600,
    }
    raw.update(overrides)
    return raw


def _spec(**overrides: Any) -> RunSpec:
    base: Dict[str, Any] = dict(
        condition="custom", horizon=8, seed=0, edo=12, path_mode="sample",
        prior_mode="null", lambda_data=1.0, lambda_gttm=1.0, bridge_mode="sb",
        k_max=64, d_max=8, proposal_budget=256, prior_guided_proposals=False,
        max_iterations=200, tolerance=1e-6, temperature=1.0,
    )
    base.update(overrides)
    return RunSpec(**base)


class TestConfig(unittest.TestCase):
    def test_expansion_counts_and_unique_ids(self) -> None:
        cfg = parse_config(_raw_config(
            horizons=[64, 128], seeds=[0, 1, 2], edos=[12, 19],
            conditions=["full", "no_gttm", "no_prior", "no_sb"],
        ))
        specs = expand_matrix(cfg)
        self.assertEqual(len(specs), 2 * 3 * 2 * 4)
        self.assertEqual(len({s.run_id for s in specs}), len(specs))

    def test_no_sb_is_not_expanded_for_map_paths(self) -> None:
        cfg = parse_config(_raw_config(path_modes=["map", "sample"], conditions=["full", "no_sb"], seeds=[0]))
        pairs = {(s.condition, s.path_mode) for s in expand_matrix(cfg)}
        self.assertEqual(pairs, {("full", "map"), ("full", "sample"), ("no_sb", "sample")})

    def test_ablations_change_exactly_one_factor_from_full(self) -> None:
        cfg = parse_config(_raw_config(conditions=["full", "no_gttm", "no_prior", "no_sb"], seeds=[0]))
        by_cond = {s.condition: s.to_dict() for s in expand_matrix(cfg)}
        ignored = {"condition", "config_name", "config_hash"}
        for cond, factor_keys in (
            ("no_gttm", {"lambda_gttm"}),
            ("no_prior", {"prior_mode"}),
            ("no_sb", {"bridge_mode"}),
        ):
            diff = {
                k for k in by_cond["full"]
                if k not in ignored and by_cond["full"][k] != by_cond[cond][k]
            }
            self.assertEqual(diff, factor_keys, cond)

    def test_rejects_bad_configs(self) -> None:
        with self.assertRaises(ValueError):
            parse_config(_raw_config(conditions=["nonsense"]))
        with self.assertRaises(ValueError):
            parse_config(_raw_config(seeds=[1, 1]))
        raw = _raw_config()
        del raw["timeout_s"]
        with self.assertRaises(ValueError):
            parse_config(raw)
        with self.assertRaises(ValueError):
            parse_config(_raw_config(schema_version="other/1"))

    def test_select_specs_filters_without_changing_ids(self) -> None:
        cfg = parse_config(_raw_config(seeds=[0, 1, 2], conditions=["full", "no_prior"]))
        specs = expand_matrix(cfg)
        sub = select_specs(specs, conditions=["no_prior"], seeds=[1])
        self.assertEqual([s.seed for s in sub], [1])
        self.assertIn(sub[0].run_id, {s.run_id for s in specs})

    def test_shipped_configs_parse(self) -> None:
        for path in sorted((REPO_ROOT / "config" / "benchmarks").glob("*.json")):
            cfg = parse_config(json.loads(path.read_text()))
            self.assertTrue(expand_matrix(cfg), path.name)


class TestRunSpec(unittest.TestCase):
    def test_uniform_bridge_requires_sampling(self) -> None:
        with self.assertRaises(ValueError):
            _spec(bridge_mode="uniform", path_mode="map")

    def test_run_id_ignores_config_identity_but_not_knobs(self) -> None:
        a = _spec(config_name="a", config_hash="1")
        b = _spec(config_name="b", config_hash="2")
        self.assertEqual(a.run_id, b.run_id)
        self.assertNotEqual(a.run_id, _spec(k_max=32).run_id)
        self.assertNotEqual(a.run_id, _spec(lambda_gttm=0.0).run_id)

    def test_round_trip(self) -> None:
        spec = _spec(condition="x", seed=3)
        self.assertEqual(RunSpec.from_dict(spec.to_dict()), spec)


class TestStats(unittest.TestCase):
    def test_summarize_known_values(self) -> None:
        s = summarize([1, 2, 3, 4, 5])
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 3.0)
        self.assertAlmostEqual(s["sd"], math.sqrt(2.5))
        half = t_critical_95(4) * math.sqrt(2.5) / math.sqrt(5)
        self.assertAlmostEqual(s["ci95_low"], 3.0 - half)
        self.assertAlmostEqual(s["ci95_high"], 3.0 + half)
        self.assertAlmostEqual(s["median"], 3.0)

    def test_single_value_has_no_interval(self) -> None:
        s = summarize([2.0])
        self.assertIsNone(s["ci95_low"])
        self.assertEqual(summarize([])["n"], 0)

    def test_wilson_bounds(self) -> None:
        lo, hi = wilson_interval(0, 5)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.3)
        lo, hi = wilson_interval(5, 5)
        self.assertEqual(hi, 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_paired_difference_uses_only_shared_seeds(self) -> None:
        d = paired_difference({0: 3.0, 1: 5.0, 2: 9.0}, {0: 2.0, 1: 3.0, 7: 100.0})
        self.assertEqual(d["seeds"], [0, 1])
        self.assertAlmostEqual(d["mean"], 1.5)


def _fake_result(condition: str, seed: int, value: float, status: str = "ok") -> Dict[str, Any]:
    spec = _spec(condition=condition, seed=seed)
    res = new_result(spec)
    res["status"] = status
    res["runtime"] = {"wall_s": 10.0 + seed, "peak_rss_mb": 50.0}
    res["metrics"]["tension"] = {"mean_absolute_error": value}
    res["metrics"]["path"] = {"length_ok": True}
    res["reproducibility"] = {"git_commit": "abc", "git_branch": "b", "python": "3", "numpy": "2"}
    if status != "ok":
        res["failure"] = {"kind": "boom", "stage": "generate", "message": "x", "traceback_tail": ""}
    return res


class TestAggregation(unittest.TestCase):
    def test_flatten_metrics_includes_runtime_and_bools(self) -> None:
        flat = flatten_metrics(_fake_result("full", 0, 0.5))
        self.assertEqual(flat["tension.mean_absolute_error"], 0.5)
        self.assertEqual(flat["path.length_ok"], 1.0)
        self.assertEqual(flat["runtime.wall_s"], 10.0)
        self.assertNotIn("runtime.rss_growth_mb", flat)  # needs both peak and baseline
        res = _fake_result("full", 0, 0.5)
        res["runtime"]["baseline_rss_mb"] = 40.0
        self.assertEqual(flatten_metrics(res)["runtime.rss_growth_mb"], 10.0)

    def test_cells_failures_and_paired_deltas(self) -> None:
        results = [_fake_result("full", s, 0.4) for s in range(3)]
        results += [_fake_result("no_gttm", s, 0.4 + 0.1 * (s + 1)) for s in range(3)]
        results.append(_fake_result("no_prior", 0, 0.0, status="failed"))
        agg = aggregate(results)
        cells = {c["condition"]: c for c in agg["cells"].values()}
        self.assertEqual(cells["no_prior"]["failure_rate"], 1.0)
        self.assertEqual(cells["no_prior"]["failure_kinds"], {"boom": 1})
        self.assertEqual(cells["full"]["n_ok"], 3)
        paired = next(v for k, v in agg["paired_vs_control"].items() if k.startswith("no_gttm"))
        delta = paired["metrics"]["tension.mean_absolute_error"]
        self.assertAlmostEqual(delta["mean"], 0.2)
        self.assertEqual(delta["n"], 3)
        # An all-failed cell has nothing to pair, so it is absent from the paired section.
        failed_key = next(k for k in agg["cells"] if k.startswith("no_prior"))
        self.assertNotIn(failed_key, agg["paired_vs_control"])

    def test_cells_never_mix_edos_or_horizons(self) -> None:
        a = _fake_result("full", 0, 0.1)
        b = _fake_result("full", 1, 0.2)
        b["spec"]["edo"] = 19
        agg = aggregate([a, b])
        self.assertEqual(len(agg["cells"]), 2)

    def test_report_renders_with_deferred_section(self) -> None:
        results = [_fake_result("full", s, 0.4) for s in range(2)]
        agg = aggregate(results)
        agg.update({"completed_runs": 2, "expected_runs": 3, "missing_run_ids": ["x"]})
        manifest = {"config_name": "unit", "config_version": "1", "config_hash": "0" * 64}
        text = render_report(manifest, results, agg)
        for needle in ("## Success and failure", "## Not measured / deferred", "MUS-15", "Incomplete"):
            self.assertIn(needle, text)


def _state(chord: int, beat: int = 0) -> BeatState:
    return BeatState(0, beat, 0, 0, chord, 0, 0, 0)


class TestReachableUniformWalk(unittest.TestCase):
    def _graph(self) -> SparseGraph:
        # layer1 has a reachable state A and a dead-end D; D -> X, but X is not the terminal.
        s, a, d, x, t = _state(0), _state(1), _state(2), _state(3), _state(4)
        return SparseGraph(
            layers=(Layer(0, (s,)), Layer(1, (a, d)), Layer(2, (t, x))),
            edges_by_time=(
                (Edge(0, s, a, 0.0), Edge(0, s, d, 0.0)),
                (Edge(1, a, t, 0.0), Edge(1, d, x, 0.0)),
            ),
            diagnostics=GraphDiagnostics((), ()),
        )

    def test_coreachability_excludes_dead_ends(self) -> None:
        graph = self._graph()
        # Only layer-2 state T is the terminal here; emulate a singleton terminal layer.
        terminal_only = SparseGraph(
            layers=(graph.layers[0], graph.layers[1], Layer(2, (_state(4),))),
            edges_by_time=(graph.edges_by_time[0], (graph.edges_by_time[1][0],)),
            diagnostics=graph.diagnostics,
        )
        reach = coreachable_states(terminal_only)
        self.assertEqual(reach[1], {_state(1)})

    def test_walk_reaches_terminal_and_is_deterministic(self) -> None:
        s, a, d, t = _state(0), _state(1), _state(2), _state(4)
        graph = SparseGraph(
            layers=(Layer(0, (s,)), Layer(1, (a, d)), Layer(2, (t,))),
            edges_by_time=(
                (Edge(0, s, a, 0.0), Edge(0, s, d, 0.0)),
                (Edge(1, a, t, 0.0), Edge(1, d, t, 0.0)),
            ),
            diagnostics=GraphDiagnostics((), ()),
        )
        seen = set()
        for seed in range(20):
            path = reachable_uniform_walk(graph, RNGKey(seed=seed))
            self.assertEqual((path[0], path[-1], len(path)), (s, t, 3))
            seen.add(path[1])
        self.assertEqual(seen, {a, d})  # both branches are reachable, so both get sampled
        self.assertEqual(
            reachable_uniform_walk(graph, RNGKey(seed=3)),
            reachable_uniform_walk(graph, RNGKey(seed=3)),
        )

    def test_dead_end_branch_is_never_taken(self) -> None:
        s, a, d, x, t = _state(0), _state(1), _state(2), _state(3), _state(4)
        graph = SparseGraph(
            layers=(Layer(0, (s,)), Layer(1, (a, d)), Layer(2, (t,))),
            edges_by_time=(
                (Edge(0, s, a, 0.0), Edge(0, s, d, 0.0)),
                (Edge(1, a, t, 0.0),),  # d has no edge to the terminal
            ),
            diagnostics=GraphDiagnostics((), ()),
        )
        del x
        for seed in range(20):
            path = reachable_uniform_walk(graph, RNGKey(seed=seed))
            self.assertEqual(path, (s, a, t))

    def test_unreachable_terminal_raises(self) -> None:
        s, x, t = _state(0), _state(1), _state(9)
        graph = SparseGraph(
            layers=(Layer(0, (s,)), Layer(1, (x,)), Layer(2, (t,))),
            edges_by_time=((Edge(0, s, x, 0.0),), (Edge(1, x, _state(7), 0.0),)),
            diagnostics=GraphDiagnostics((), ()),
        )
        with self.assertRaises(ValueError):
            reachable_uniform_walk(graph, RNGKey(seed=0))


def _stable(result: Dict[str, Any]) -> Dict[str, Any]:
    """Result minus wall-clock/memory/environment fields."""
    out = json.loads(json.dumps(result))
    out.pop("runtime")
    out.pop("reproducibility")
    return out


class TestEvaluateRun(unittest.TestCase):
    def test_repeated_runs_are_deterministic(self) -> None:
        spec = _spec(condition="det", bridge_mode="uniform")  # exercises the no-SB path too
        first = evaluate_run(spec)
        second = evaluate_run(spec)
        self.assertEqual(first["status"], "ok", first.get("failure"))
        self.assertEqual(_stable(first), _stable(second))
        self.assertTrue(first["artifacts"]["path_sha256"])
        self.assertEqual(first["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(first["run_id"], spec.run_id)
        json.dumps(first)  # fully JSON-serialisable

    def test_sb_run_reports_existing_diagnostics(self) -> None:
        res = evaluate_run(_spec(condition="sb", path_mode="map"))
        self.assertEqual(res["status"], "ok", res.get("failure"))
        graph = res["metrics"]["graph"]
        self.assertGreater(graph["proposed_count"], graph["unique_count"])
        self.assertGreaterEqual(graph["legal_count"], graph["unique_count"])
        self.assertIn("prune_reason_counts", graph)
        self.assertTrue(res["metrics"]["sb"]["converged"])
        self.assertTrue(res["metrics"]["path"]["map_path_score_finite"])
        self.assertTrue(res["metrics"]["path"]["reaches_end"])
        self.assertGreater(res["runtime"]["peak_rss_mb"], 0.0)
        self.assertEqual(res["prior"]["status"], "off_intentional")

    def test_unavailable_prior_is_a_structured_failure_not_a_substitution(self) -> None:
        res = evaluate_run(_spec(condition="trained", prior_mode="trained_neural"))
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["failure"]["kind"], "prior_unavailable")
        self.assertEqual(res["prior"]["status"], "requested_unavailable")
        self.assertIsNone(res["artifacts"]["path_sha256"])

    def test_generation_errors_become_results(self) -> None:
        res = evaluate_run(_spec(condition="bad", edo=0))
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["failure"]["stage"], "generate")
        self.assertIn("runtime", res)


class TestMatrixRunner(unittest.TestCase):
    def test_run_resume_refuse_foreign_config_and_report(self) -> None:
        cfg = parse_config(_raw_config(seeds=[0, 1]))
        specs = expand_matrix(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench"
            messages: list[str] = []
            results = run_matrix(cfg, specs, specs, out, workers=2, progress=messages.append)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r["status"] == "ok" for r in results))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(manifest["expected_runs"]), 2)
            self.assertEqual(manifest["config_hash"], cfg.config_hash)

            # Resume: nothing left to run, nothing overwritten.
            stamp = {p.name: p.stat().st_mtime_ns for p in (out / "runs").glob("*.json")}
            messages.clear()
            run_matrix(cfg, specs, specs, out, workers=1, progress=messages.append)
            self.assertIn("0 to run", messages[0])
            self.assertEqual(stamp, {p.name: p.stat().st_mtime_ns for p in (out / "runs").glob("*.json")})

            # A different config must not write into this directory.
            other = parse_config(_raw_config(name="other", seeds=[5]))
            with self.assertRaises(FileExistsError):
                run_matrix(other, expand_matrix(other), expand_matrix(other), out)

            agg = write_outputs(out)
            self.assertEqual(agg["completed_runs"], 2)
            self.assertEqual(agg["missing_run_ids"], [])
            self.assertTrue((out / "report.md").read_text().startswith("# MUS-28 benchmark report"))
            self.assertEqual(len((out / "results.jsonl").read_text().strip().splitlines()), 2)
            self.assertEqual(len(load_results(out)), 2)

    def test_relative_output_directory_works(self) -> None:
        cfg = parse_config(_raw_config(seeds=[0]))
        specs = expand_matrix(cfg)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                results = run_matrix(cfg, specs, specs, Path("rel_out"), progress=lambda _m: None)
            finally:
                os.chdir(previous)
            self.assertEqual([r["status"] for r in results], ["ok"])
            self.assertTrue((Path(tmp) / "rel_out" / "manifest.json").exists())

    def test_timeout_is_recorded_and_does_not_abort_the_matrix(self) -> None:
        cfg = parse_config(_raw_config(seeds=[0], timeout_s=0.2))
        specs = expand_matrix(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench"
            results = run_matrix(cfg, specs, specs, out, progress=lambda _m: None)
            self.assertEqual(results[0]["status"], "timeout")
            self.assertEqual(results[0]["failure"]["kind"], "timeout")
            agg = write_outputs(out)
            self.assertEqual(agg["completed_runs"], 1)
            cell = next(iter(agg["cells"].values()))
            self.assertEqual(cell["failure_rate"], 1.0)


class TestSeparationFromGeneration(unittest.TestCase):
    def test_generation_code_never_imports_the_evaluation_layer(self) -> None:
        offenders = []
        for path in (REPO_ROOT / "aimusic").rglob("*.py"):
            if "evaluation" in path.relative_to(REPO_ROOT / "aimusic").parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == "aimusic.evaluation" or n.startswith("aimusic.evaluation.") for n in names):
                    offenders.append(str(path))
        self.assertEqual(offenders, [])

    def test_evaluation_does_not_import_audio_quarantine(self) -> None:
        for path in (REPO_ROOT / "aimusic" / "evaluation").rglob("*.py"):
            self.assertNotIn("aimusic.audio", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
