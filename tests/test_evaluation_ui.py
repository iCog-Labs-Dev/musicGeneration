"""Tests for the MUS-28 evaluation dashboard (aimusic.evaluation.ui).

Data loading, summaries, chart preparation and the benchmark job runner are tested without Gradio.
Figure construction needs Plotly and is skipped when it is not installed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from aimusic.evaluation import ui
from aimusic.evaluation.config import expand_matrix, parse_config
from aimusic.evaluation.matrix import write_outputs
from aimusic.evaluation.schema import RunSpec, new_result
from aimusic.evaluation.stats import aggregate

try:  # pragma: no cover - environment dependent
    import plotly  # noqa: F401

    HAVE_PLOTLY = True
except ImportError:  # pragma: no cover
    HAVE_PLOTLY = False

REPO_ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    # condition: (prior_mode, lambda_gttm, bridge_mode)
    "full": ("placeholder_structured", 1.0, "sb"),
    "no_gttm": ("placeholder_structured", 0.0, "sb"),
    "no_prior": ("null", 1.0, "sb"),
    "no_sb": ("placeholder_structured", 1.0, "uniform"),
}


def _pt(point: Optional[ui.Point]) -> ui.Point:
    assert point is not None
    return point


def _spec(condition: str, horizon: int, edo: int, seed: int) -> RunSpec:
    prior, lam, bridge = CONDITIONS[condition]
    return RunSpec(
        condition=condition, horizon=horizon, seed=seed, edo=edo, path_mode="sample", prior_mode=prior,
        lambda_data=1.0, lambda_gttm=lam, bridge_mode=bridge, k_max=64, d_max=8, proposal_budget=256,
        prior_guided_proposals=False, max_iterations=200, tolerance=1e-6, temperature=1.0,
        config_name="synthetic", config_hash="h" * 64,
    )


def _ok_result(spec: RunSpec, curves: bool = False) -> Dict[str, Any]:
    res = new_result(spec)
    res["status"] = "ok"
    gttm_off = spec.condition == "no_gttm"
    wall = 10.0 * spec.horizon / 8 + spec.seed * 0.5 + (5 if spec.edo == 19 else 0)
    res["runtime"] = {"wall_s": wall, "cpu_s": wall, "generate_wall_s": wall, "peak_rss_mb": 100.0 + spec.horizon, "baseline_rss_mb": 30.0}
    res["metrics"] = {
        "graph": {"layer_width_mean": 60.0 + spec.seed},
        "sb": {"converged": True, "iterations": 1},
        "path": {"distinct_state_fraction": 0.9, "chord_change_rate": 0.1 + 0.01 * spec.seed, "length_ok": True},
        "tension": {
            "mean_absolute_error": (0.35 if gttm_off else 0.2) + 0.01 * spec.seed,
            "shape_correlation": (-0.2 if gttm_off else 0.2) + 0.02 * spec.seed,
            "realized_mean": 0.4, "realized_min": 0.0, "realized_max": 0.8,
        },
        "gttm": {"mean_gttm_score": 8.0},
    }
    if curves:
        n = spec.horizon
        res["metrics"]["tension"]["target_curve"] = [i / n for i in range(n)]
        res["metrics"]["tension"]["realized_curve"] = [min(1.0, i / n + 0.05 * spec.seed) for i in range(n)]
    return res


def _failed_result(spec: RunSpec, status: str = "failed", kind: str = "boom") -> Dict[str, Any]:
    res = new_result(spec)
    res["status"] = status
    res["failure"] = {"kind": kind, "stage": "generate", "message": "it broke", "traceback_tail": "Traceback...\nValueError: x"}
    res["runtime"] = {"wall_s": 1.5}
    return res


def _write_output_dir(
    out: Path, results: List[Dict[str, Any]], expected: Optional[List[RunSpec]] = None,
    aggregate_files: bool = True, runs_files: bool = True, results_jsonl: bool = False,
) -> None:
    specs = expected if expected is not None else [RunSpec.from_dict(r["spec"]) for r in results]
    manifest = {
        "schema_version": "mus28-manifest/1", "config_name": "synthetic", "config_version": "0", "config_hash": "h" * 64,
        "config": {}, "created_with": {}, "expected_runs": [{"run_id": s.run_id, "spec": s.to_dict()} for s in specs],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest))
    if runs_files:
        (out / "runs").mkdir(exist_ok=True)
        for r in results:
            (out / "runs" / f"{r['run_id']}.json").write_text(json.dumps(r))
    if results_jsonl:
        (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    if aggregate_files:
        write_outputs(out)


def _full_matrix(curves: bool = False, edos=(12, 19), horizons=(8, 16), seeds=(0, 1, 2)) -> List[Dict[str, Any]]:
    return [
        _ok_result(_spec(c, h, e, s), curves)
        for c in CONDITIONS for h in horizons for e in edos for s in seeds
        if not (c == "no_sb" and False)
    ]


class LoadingTests(unittest.TestCase):
    def test_load_complete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_output_dir(out, _full_matrix())
            ds = ui.load_benchmark_output(out)
            self.assertEqual(len(ds.results), 4 * 2 * 2 * 3)
            self.assertEqual(ds.aggregate_source, "aggregate.json")
            self.assertTrue(ds.report_text and ds.report_text.startswith("# MUS-28"))
            self.assertEqual(ds.warnings, [])

    def test_missing_manifest_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                ui.load_benchmark_output(Path(tmp))
            self.assertIn("manifest.json", str(ctx.exception))
        with self.assertRaises(FileNotFoundError):
            ui.load_benchmark_output(Path(tmp) / "does-not-exist")

    def test_empty_manifest_only_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            specs = [_spec("full", 8, 12, s) for s in range(3)]
            _write_output_dir(out, [], expected=specs, aggregate_files=False)
            ds = ui.load_benchmark_output(out)
            summary = ui.summarize_results(ds)
            self.assertEqual(summary.state, "empty")
            self.assertEqual((summary.expected, summary.completed, summary.not_run), (3, 0, 3))
            self.assertIsNone(summary.mean_wall_s)
            self.assertTrue(ui.runtime_chart_data(ds).empty)
            self.assertEqual(ui.run_choices(ds), [])
            self.assertTrue(ui.ablation_delta_data(ds).empty)
            self.assertIn("No benchmark loaded", ui.render_overview_cards(None))
            self.assertIn("0/3", ui.render_overview_cards(summary))

    def test_partial_output_without_aggregate_is_computed_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            all_specs = [_spec(c, 8, 12, s) for c in ("full", "no_gttm") for s in range(3)]
            done = [_ok_result(s) for s in all_specs[:4]]
            _write_output_dir(out, done, expected=all_specs, aggregate_files=False)
            ds = ui.load_benchmark_output(out)
            self.assertEqual(ds.aggregate_source, "recomputed")
            summary = ui.summarize_results(ds)
            self.assertEqual((summary.state, summary.completed, summary.expected, summary.not_run), ("partial", 4, 6, 2))
            self.assertTrue(any("not found" in w for w in ds.warnings))

    def test_stale_aggregate_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            specs = [_spec("full", 8, 12, s) for s in range(3)]
            _write_output_dir(out, [_ok_result(specs[0])], expected=specs)  # aggregate says 1 completed
            extra = _ok_result(specs[1])
            (out / "runs" / f"{extra['run_id']}.json").write_text(json.dumps(extra))  # a run finishes later
            ds = ui.load_benchmark_output(out)
            self.assertEqual(ds.aggregate_source, "recomputed")
            self.assertEqual(ds.cells[next(iter(ds.cells))]["n_runs"], 2)
            self.assertTrue(any("out of date" in w for w in ds.warnings))

    def test_falls_back_to_results_jsonl_and_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results = [_ok_result(_spec("full", 8, 12, s)) for s in range(2)]
            _write_output_dir(out, results, aggregate_files=False, runs_files=False, results_jsonl=True)
            with open(out / "results.jsonl", "a") as handle:
                handle.write("{not json\n")
                handle.write(json.dumps({"run_id": "x"}) + "\n")
            ds = ui.load_benchmark_output(out)
            self.assertEqual(len(ds.results), 2)
            self.assertEqual(sum("Skipped" in w for w in ds.warnings), 2)

    def test_unreadable_run_file_is_skipped_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results = [_ok_result(_spec("full", 8, 12, s)) for s in range(2)]
            _write_output_dir(out, results, aggregate_files=False)
            (out / "runs" / f"{results[0]['run_id']}.json").write_text("{truncated")
            ds = ui.load_benchmark_output(out)
            self.assertEqual(len(ds.results), 1)
            self.assertTrue(any("unreadable" in w for w in ds.warnings))

    def test_discover_configs_lists_repo_benchmarks(self) -> None:
        names = ui.discover_benchmark_configs()
        self.assertIn("m1_smoke.json", names)
        self.assertIn("m1_64_128.json", names)
        self.assertEqual(ui.discover_benchmark_configs(Path("/nonexistent")), [])


class SummaryTests(unittest.TestCase):
    def test_status_counts_use_raw_statuses_and_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            specs = [_spec("full", 8, 12, s) for s in range(6)]
            results = [
                _ok_result(specs[0]), _ok_result(specs[1]),
                _failed_result(specs[2], "failed"), _failed_result(specs[3], "timeout", "timeout"),
                _failed_result(specs[4], "crashed", "worker_crashed"),
            ]
            _write_output_dir(out, results, expected=specs)
            summary = ui.summarize_results(ui.load_benchmark_output(out))
            self.assertEqual(summary.status_counts, {"ok": 2, "failed": 1, "timeout": 1, "crashed": 1, "not_run": 1})
            self.assertEqual((summary.ok, summary.failed, summary.timeouts, summary.crashed, summary.not_run), (2, 1, 1, 1, 1))
            self.assertEqual(summary.state, "partial")
            html = ui.render_overview_cards(summary)
            for text in ("5/6", "failed", "timeouts", "crashed", "not run yet"):
                self.assertIn(text, html)

    def test_overview_values_for_full_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_output_dir(out, _full_matrix())
            s = ui.summarize_results(ui.load_benchmark_output(out))
            self.assertEqual(s.state, "complete")
            self.assertEqual(s.longest_horizon, 16)
            self.assertEqual(s.edos, [12, 19])
            self.assertEqual(s.conditions, ["full", "no_gttm", "no_prior", "no_sb"])
            self.assertTrue(s.ci_enabled)
            self.assertIsNotNone(s.mean_wall_s)
            self.assertIn("12-EDO, 19-EDO", ui.render_overview_cards(s))

    def test_card_html_escapes_config_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_output_dir(out, _full_matrix(horizons=(8,), edos=(12,)))
            ds = ui.load_benchmark_output(out)
            ds.manifest["config_name"] = "<script>alert(1)</script>"
            html = ui.render_overview_cards(ui.summarize_results(ds))
            self.assertNotIn("<script>", html)


class ChartDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        _write_output_dir(self.out, _full_matrix())
        self.ds = ui.load_benchmark_output(self.out)

    def test_runtime_chart_keeps_edos_and_horizons_separate_with_ci(self) -> None:
        data = ui.runtime_chart_data(self.ds)
        self.assertEqual(data.categories, ["8b / 12-EDO", "16b / 12-EDO", "8b / 19-EDO", "16b / 19-EDO"])
        self.assertEqual(list(data.series), ["full", "no_gttm", "no_prior", "no_sb"])
        pt = _pt(data.series["full"][0])
        self.assertEqual(pt.n, 3)
        assert pt.low is not None and pt.high is not None
        self.assertLess(pt.low, pt.mean)
        self.assertGreater(pt.high, pt.mean)
        # 19-EDO runs are 5 s slower in the fixture; they must not be pooled into the 12-EDO bar.
        self.assertAlmostEqual(_pt(data.series["full"][2]).mean - _pt(data.series["full"][0]).mean, 5.0)

    def test_runtime_chart_values_come_from_aggregate_not_recomputed(self) -> None:
        agg = json.loads((self.out / "aggregate.json").read_text())
        cell = next(c for c in agg["cells"].values() if c["condition"] == "full" and c["horizon"] == 16 and c["edo"] == 19)
        pt = _pt(ui.runtime_chart_data(self.ds).series["full"][3])
        self.assertEqual(pt.mean, cell["metrics"]["runtime.wall_s"]["mean"])
        self.assertEqual(pt.low, cell["metrics"]["runtime.wall_s"]["ci95_low"])

    def test_single_seed_cell_has_no_ci(self) -> None:
        out = self.out / "single"
        _write_output_dir(out, _full_matrix(horizons=(8,), edos=(12,), seeds=(0,)))
        data = ui.runtime_chart_data(ui.load_benchmark_output(out))
        self.assertIsNone(_pt(data.series["full"][0]).low)
        self.assertTrue(any("single successful seed" in n for n in data.notes))

    def test_failed_cell_has_missing_point_not_zero(self) -> None:
        out = self.out / "withfail"
        results = [_ok_result(_spec("full", 8, 12, s)) for s in range(3)]
        results += [_failed_result(_spec("no_gttm", 8, 12, s)) for s in range(3)]
        _write_output_dir(out, results)
        data = ui.runtime_chart_data(ui.load_benchmark_output(out))
        self.assertIsNone(data.series["no_gttm"][0])
        self.assertIsNotNone(data.series["full"][0])

    def test_ablation_deltas_are_paired_vs_full_with_direction(self) -> None:
        data = ui.ablation_delta_data(self.ds, 8, 12)
        self.assertEqual((data.horizon, data.edo), (8, 12))
        mae = {b.condition: b for b in data.metrics["tension.mean_absolute_error"]}
        self.assertEqual(set(mae), {"no_gttm", "no_prior", "no_sb"})
        self.assertAlmostEqual(mae["no_gttm"].mean, 0.15)
        self.assertEqual(mae["no_gttm"].tone, "worse")  # higher MAE is worse
        corr = {b.condition: b for b in data.metrics["tension.shape_correlation"]}
        self.assertEqual(corr["no_gttm"].tone, "worse")  # lower correlation is worse
        self.assertEqual(corr["no_prior"].tone, "neutral")  # identical to full => zero delta
        distinct = data.metrics["path.distinct_state_fraction"]
        self.assertTrue(all(b.tone == "neutral" for b in distinct))  # no clear direction => neutral colour
        self.assertEqual([b.condition for b in distinct], ["no_gttm", "no_prior", "no_sb"])

    def test_ablation_unknown_cell_falls_back_and_choices_listed(self) -> None:
        self.assertEqual([(h, e) for h, e, _ in ui.comparison_keys(self.ds)], [(8, 12), (16, 12), (8, 19), (16, 19)])
        data = ui.ablation_delta_data(self.ds, 999, 7)
        self.assertEqual((data.horizon, data.edo), (8, 12))

    def test_ablation_needs_a_full_control(self) -> None:
        out = self.out / "nofull"
        _write_output_dir(out, [_ok_result(_spec("no_gttm", 8, 12, s)) for s in range(3)])
        data = ui.ablation_delta_data(ui.load_benchmark_output(out))
        self.assertTrue(data.empty)
        self.assertTrue(data.notes)

    def test_tension_without_curves_falls_back_to_summaries(self) -> None:
        data = ui.tension_chart_data(self.ds, 8, 12)
        self.assertFalse(data.has_curves)
        self.assertFalse(data.summary_mae.empty)
        self.assertTrue(any("not stored" in n for n in data.notes))

    def test_tension_curves_target_and_realized_per_condition(self) -> None:
        out = self.out / "curves"
        _write_output_dir(out, _full_matrix(curves=True))
        data = ui.tension_chart_data(ui.load_benchmark_output(out), 16, 19)
        self.assertTrue(data.has_curves)
        self.assertEqual(len(data.target or []), 16)
        self.assertEqual([c.condition for c in data.curves], ["full", "no_gttm", "no_prior", "no_sb"])
        c = data.curves[0]
        self.assertEqual((c.n, len(c.mean)), (3, 16))
        self.assertTrue(all(lo <= m <= hi for lo, m, hi in zip(c.low, c.mean, c.high)))

    def test_curves_do_not_change_aggregates(self) -> None:
        with_curves = aggregate(_full_matrix(curves=True))
        without = aggregate(_full_matrix(curves=False))
        self.assertEqual(with_curves["cells"].keys(), without["cells"].keys())
        for key, cell in with_curves["cells"].items():
            self.assertEqual(sorted(cell["metrics"]), sorted(without["cells"][key]["metrics"]))

    def test_scaling_data_is_per_edo_and_sorted_by_horizon(self) -> None:
        data = ui.scaling_chart_data(self.ds)
        self.assertEqual(data.edos, [12, 19])
        series = data.metrics["runtime.wall_s"][19]["full"]
        self.assertEqual([h for h, _ in series], [8, 16])
        self.assertLess(series[0][1].mean, series[1][1].mean)
        self.assertEqual(sorted(data.metrics["runtime.rss_growth_mb"]), [12, 19])
        self.assertFalse(data.notes)

    def test_scaling_single_horizon_notes_no_trend(self) -> None:
        out = self.out / "oneh"
        _write_output_dir(out, _full_matrix(horizons=(8,)))
        self.assertTrue(ui.scaling_chart_data(ui.load_benchmark_output(out)).notes)

    def test_edo_comparison_never_pools_edos(self) -> None:
        data = ui.edo_comparison_data(self.ds, 8)
        self.assertEqual(data.edos, [12, 19])
        rt = data.metrics["runtime.wall_s"]
        self.assertAlmostEqual(_pt(rt[19]["full"]).mean - _pt(rt[12]["full"]).mean, 5.0)
        self.assertEqual(data.horizon, 8)
        self.assertEqual(ui.edo_comparison_data(self.ds, 12345).horizon, 16)  # unknown -> longest

    def test_status_chart_counts_raw_statuses(self) -> None:
        out = self.out / "status"
        specs = [_spec("full", 8, 12, s) for s in range(4)] + [_spec("no_gttm", 8, 12, 0)]
        results = [_ok_result(specs[0]), _failed_result(specs[1], "timeout", "timeout"), _failed_result(specs[2], "crashed", "x")]
        _write_output_dir(out, results, expected=specs)
        data = ui.status_chart_data(ui.load_benchmark_output(out))
        self.assertEqual(data.totals, {"ok": 1, "failed": 0, "timeout": 1, "crashed": 1, "not_run": 2})
        self.assertEqual(data.counts["not_run"], {"full": 1, "no_gttm": 1})

    @unittest.skipUnless(HAVE_PLOTLY, "plotly not installed")
    def test_all_figures_build_for_complete_partial_and_empty_data(self) -> None:
        empty_dir = self.out / "empty"
        _write_output_dir(empty_dir, [], expected=[_spec("full", 8, 12, 0)], aggregate_files=False)
        curves_dir = self.out / "curves2"
        _write_output_dir(curves_dir, _full_matrix(curves=True))
        for ds in (self.ds, ui.load_benchmark_output(empty_dir), ui.load_benchmark_output(curves_dir)):
            for fig in (
                ui.make_runtime_chart(ds), ui.make_ablation_chart(ds), ui.make_tension_chart(ds),
                ui.make_scaling_chart(ds), ui.make_edo_chart(ds), ui.make_status_chart(ds),
            ):
                self.assertTrue(fig.to_json())

    @unittest.skipUnless(HAVE_PLOTLY, "plotly not installed")
    def test_runtime_figure_has_error_bars_and_one_trace_per_condition(self) -> None:
        fig = ui.make_runtime_chart(self.ds)
        self.assertEqual([t.name for t in fig.data], ["full", "no_gttm", "no_prior", "no_sb"])
        err = fig.data[0].error_y
        self.assertEqual(len(err.array), 4)
        self.assertTrue(all(v > 0 for v in err.array))

    @unittest.skipUnless(HAVE_PLOTLY, "plotly not installed")
    def test_tension_figure_has_target_trace_when_curves_exist(self) -> None:
        out = self.out / "curves3"
        _write_output_dir(out, _full_matrix(curves=True))
        fig = ui.make_tension_chart(ui.load_benchmark_output(out), 16, 12)
        self.assertIn("target", [t.name for t in fig.data])


class RunDetailTests(unittest.TestCase):
    def test_choices_flag_failures_and_details_render_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ok = _ok_result(_spec("full", 8, 12, 0), curves=True)
            bad = _failed_result(_spec("no_gttm", 8, 12, 1), "timeout", "timeout")
            _write_output_dir(out, [ok, bad])
            ds = ui.load_benchmark_output(out)
            labels = dict(ui.run_choices(ds))
            self.assertEqual(len(labels), 2)
            self.assertTrue(any("[TIMEOUT]" in label for label in labels))
            text = ui.render_run_details(ui.find_run(ds, bad["run_id"]))
            for needle in ("Failure", "timeout", "it broke", "Traceback tail"):
                self.assertIn(needle, text)
            good = ui.render_run_details(ui.find_run(ds, ok["run_id"]))
            for needle in ("Graph metrics", "SB convergence", "Tension metrics", "GTTM score", "Runtime and memory", "Generated artifacts"):
                self.assertIn(needle, good)
            self.assertNotIn("#### Failure", good)
            self.assertIsNone(ui.find_run(ds, "nope"))
            self.assertIn("Select a run", ui.render_run_details(None))

    def test_long_curves_are_not_dumped_into_the_detail_table(self) -> None:
        res = _ok_result(_spec("full", 64, 12, 0), curves=True)
        self.assertNotIn("0.015625, 0.03125", ui.render_run_details(res))

    @unittest.skipUnless(HAVE_PLOTLY, "plotly not installed")
    def test_run_tension_figure_variants(self) -> None:
        with_curve = ui.run_tension_figure(_ok_result(_spec("full", 8, 12, 0), curves=True))
        self.assertEqual([t.name for t in with_curve.data], ["target", "realized"])
        summary_only = ui.run_tension_figure(_ok_result(_spec("full", 8, 12, 0)))
        self.assertEqual(len(summary_only.data), 1)
        self.assertTrue(ui.run_tension_figure(_failed_result(_spec("full", 8, 12, 0))).layout.annotations)
        self.assertTrue(ui.run_tension_figure(None).layout.annotations)


class CommandAndPreflightTests(unittest.TestCase):
    def test_commands_use_the_existing_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = ui.BenchmarkRequest(
                "m1_smoke.json", tmp, workers=3, conditions=("full", "no_sb"), horizons=(8,), edos=(12,), seeds=(0, 1), retry_failed=True,
            )
            cmd = ui.build_cli_command("run", req, python="py")
            self.assertEqual(cmd[:4], ["py", "-m", "aimusic.evaluation", "run"])
            self.assertEqual(cmd[cmd.index("--config") + 1], str((REPO_ROOT / "config/benchmarks/m1_smoke.json").resolve()))
            self.assertEqual(cmd[cmd.index("--out") + 1], str(Path(tmp).resolve()))
            self.assertEqual(cmd[cmd.index("--workers") + 1], "3")
            self.assertEqual(cmd[cmd.index("--conditions") + 1:cmd.index("--conditions") + 3], ["full", "no_sb"])
            self.assertIn("--retry-failed", cmd)
            self.assertEqual(cmd[cmd.index("--seeds") + 1:], ["0", "1"])
            self.assertEqual(ui.build_cli_command("resume", req, python="py")[3], "run")
            self.assertIn("--dry-run", ui.build_cli_command("list", req, python="py"))
            self.assertEqual(ui.build_cli_command("report", req, python="py")[3:], ["report", "--out", str(Path(tmp).resolve())])
            with self.assertRaises(ValueError):
                ui.build_cli_command("explode", req)

    def test_relative_out_dir_resolves_from_repo_root(self) -> None:
        self.assertEqual(ui.resolve_out_dir("outputs/x"), (REPO_ROOT / "outputs" / "x").resolve())
        with self.assertRaises(ui.PreflightError):
            ui.resolve_out_dir("  ")

    def test_preflight_refuses_directory_from_a_different_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_output_dir(out, [_ok_result(_spec("full", 8, 12, 0))], aggregate_files=False)  # hash "hhh..."
            with self.assertRaises(ui.PreflightError) as ctx:
                ui.preflight("run", ui.BenchmarkRequest("m1_smoke.json", tmp))
            self.assertIn("different config", str(ctx.exception))
            with self.assertRaises(ui.PreflightError):
                ui.preflight("resume", ui.BenchmarkRequest("m1_smoke.json", tmp))
            ui.preflight("report", ui.BenchmarkRequest("m1_smoke.json", tmp))  # reporting needs only the manifest

    def test_preflight_other_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ui.PreflightError):
                ui.preflight("resume", ui.BenchmarkRequest("m1_smoke.json", tmp))  # nothing to resume
            with self.assertRaises(ui.PreflightError):
                ui.preflight("report", ui.BenchmarkRequest("m1_smoke.json", tmp))  # no manifest
            with self.assertRaises(ui.PreflightError):
                ui.preflight("run", ui.BenchmarkRequest("missing.json", tmp))
            with self.assertRaises(ui.PreflightError):
                ui.preflight("run", ui.BenchmarkRequest("m1_smoke.json", tmp, horizons=(9999,)))  # filter selects nothing
            ui.preflight("run", ui.BenchmarkRequest("m1_smoke.json", tmp, horizons=(8,)))

    def test_preflight_accepts_resume_with_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = parse_config(json.loads((REPO_ROOT / "config/benchmarks/m1_smoke.json").read_text()))
            specs = expand_matrix(config)
            out = Path(tmp)
            (out).mkdir(exist_ok=True)
            (out / "manifest.json").write_text(json.dumps({"config_hash": config.config_hash, "config_name": config.name, "expected_runs": []}))
            ui.preflight("resume", ui.BenchmarkRequest("m1_smoke.json", tmp))
            self.assertTrue(specs)

    def test_parse_int_list(self) -> None:
        self.assertEqual(ui._parse_int_list("1, 2  3", "x"), (1, 2, 3))
        self.assertEqual(ui._parse_int_list("", "x"), ())
        with self.assertRaises(ui.PreflightError):
            ui._parse_int_list("a", "x")


class JobTests(unittest.TestCase):
    """The job runner is non-blocking, captures output, and can be stopped."""

    def _job(self, out: Path, code: str) -> ui.BenchmarkJob:
        job = ui.BenchmarkJob([sys.executable, "-u", "-c", code], out, "run")
        job.start()
        self.addCleanup(job.stop)
        return job

    def test_start_returns_immediately_and_log_is_captured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "o"
            begin = time.monotonic()
            job = self._job(out, "import time,sys; print('hello'); sys.stderr.write('oops\\n'); time.sleep(3); print('bye')")
            self.assertLess(time.monotonic() - begin, 2.0)  # did not wait for the 3 s sleep
            self.assertTrue(job.running)
            self.assertIn("Running", job.status_text())
            deadline = time.monotonic() + 5
            while "hello" not in job.log_tail() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIn("hello", job.log_tail())  # partial output visible while running
            self.assertEqual(job.wait(10), 0)
            tail = job.log_tail()
            self.assertIn("oops", tail)  # stderr preserved
            self.assertIn("bye", tail)
            self.assertIn("Finished", job.status_text())

    def test_failure_exit_code_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(Path(tmp) / "o", "import sys; print('bad'); sys.exit(3)")
            self.assertEqual(job.wait(10), 3)
            self.assertIn("code 3", job.status_text())
            self.assertIn("bad", job.log_tail())

    def test_stop_terminates_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(Path(tmp) / "o", "import time; time.sleep(60)")
            self.assertTrue(job.running)
            job.stop()
            self.assertFalse(job.running)
            self.assertIn("Stopped", job.status_text())

    def test_registry_refuses_second_job_in_same_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ui.JobRegistry()
            out = Path(tmp) / "reg"
            req = ui.BenchmarkRequest("m1_smoke.json", str(out), horizons=(8,), seeds=(0,))
            calls: List[Any] = []
            real_preflight, real_build = ui.preflight, ui.build_cli_command
            ui.preflight = lambda *a, **k: None  # type: ignore[assignment]
            ui.build_cli_command = lambda *a, **k: [sys.executable, "-c", "import time; time.sleep(30)"]  # type: ignore[assignment]
            try:
                first = registry.start("run", req)
                calls.append(first)
                self.addCleanup(first.stop)
                with self.assertRaises(ui.PreflightError):
                    registry.start("run", req)
                self.assertIs(registry.get(out.resolve()), first)
                self.assertEqual(registry.running_jobs(), [first])
            finally:
                ui.preflight, ui.build_cli_command = real_preflight, real_build  # type: ignore[assignment]

    def test_real_cli_run_resume_and_foreign_config_refusal(self) -> None:
        """End to end through the same CLI the UI uses: run, resume (no-op), load, refuse a foreign config."""
        raw = {
            "schema_version": "mus28-config/1", "name": "ui-e2e", "config_version": "0.0.1", "horizons": [4], "seeds": [0],
            "edos": [12], "path_modes": ["sample"], "conditions": ["no_prior"],
            "graph_grid": [{"k_max": 64, "d_max": 8, "proposal_budget": 256}], "prior_guided_proposals": False,
            "sb": {"max_iterations": 200, "tolerance": 1e-6, "temperature": 1.0}, "timeout_s": 600,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "ui_e2e.json"
            cfg.write_text(json.dumps(raw))
            out = Path(tmp) / "out"
            req = ui.BenchmarkRequest(str(cfg), str(out), workers=1)
            registry = ui.JobRegistry()
            job = registry.start("run", req)
            self.assertEqual(job.wait(240), 0, job.log_tail())
            ds = ui.load_benchmark_output(out)
            summary = ui.summarize_results(ds)
            self.assertEqual((summary.state, summary.ok), ("complete", 1))
            # new runs record per-beat tension curves for the dashboard
            self.assertTrue(ui.tension_chart_data(ds).has_curves)
            self.assertTrue(ds.report_text)
            # resume of a finished benchmark is a harmless no-op
            again = registry.start("resume", req)
            self.assertEqual(again.wait(120), 0, again.log_tail())
            self.assertIn("already finished", again.log_tail())
            # a different config must not be allowed into this directory
            raw2 = dict(raw, name="other", seeds=[0, 1])
            cfg2 = Path(tmp) / "other.json"
            cfg2.write_text(json.dumps(raw2))
            with self.assertRaises(ui.PreflightError):
                registry.start("run", ui.BenchmarkRequest(str(cfg2), str(out)))
            # report regeneration goes through the CLI as well
            code, text = ui.run_cli_blocking("report", req)
            self.assertEqual(code, 0, text)
            code, text = ui.run_cli_blocking("list", req)
            self.assertEqual(code, 0, text)
            self.assertIn("1 of 1 runs selected", text)


class ImportIsolationTests(unittest.TestCase):
    def test_module_imports_without_gradio_or_plotly(self) -> None:
        """Data/summary/chart-prep code must not need the UI dependencies at import time."""
        import ast

        tree = ast.parse(Path(ui.__file__).read_text())
        top_level = [
            n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        names = {a.name.split(".")[0] for n in top_level for a in getattr(n, "names", [])}
        names |= {n.module.split(".")[0] for n in top_level if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse({"gradio", "plotly"} & names)


if __name__ == "__main__":
    unittest.main()
