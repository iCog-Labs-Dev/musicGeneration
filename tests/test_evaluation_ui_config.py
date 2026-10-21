"""Tests for the MUS-28 Evaluation UI improvements: config metadata, dynamic filters, validation, run preview.

Everything here is pure (no Gradio, no subprocess) except the optional page-construction smoke test.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from aimusic.evaluation import ui
from aimusic.evaluation.config import expand_matrix, load_config, parse_config
from aimusic.evaluation.matrix import prepare_output_dir
from aimusic.evaluation.schema import new_result

try:  # pragma: no cover - environment dependent
    import gradio  # noqa: F401

    HAVE_GRADIO = True
except ImportError:  # pragma: no cover
    HAVE_GRADIO = False

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "benchmarks"
EXPECTED_TYPES = {"m1_smoke.json": "SMOKE", "m1_profile_256.json": "PROFILE", "m1_64_128.json": "BENCHMARK"}
EXPECTED_RUNS = {"m1_smoke.json": 24, "m1_profile_256.json": 2, "m1_64_128.json": 80}


def _raw(name: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)
    return data


def _custom_config(**changes: Any) -> Any:
    raw = copy.deepcopy(_raw("m1_smoke.json"))
    raw.update(changes)
    return parse_config(raw)


def _request(config: str = "m1_smoke.json", out: str = "", **kw: Any) -> ui.BenchmarkRequest:
    return ui.BenchmarkRequest(config, out, **kw)


def _write_results(config_name: str, out: Path, statuses: Sequence[str], wall_s: Optional[float] = None) -> None:
    """Manifest plus one result file per status, for the first len(statuses) specs of the config."""
    config = load_config(CONFIG_DIR / config_name)
    specs = expand_matrix(config)
    prepare_output_dir(config, specs, out)
    for spec, status in zip(specs, statuses):
        res = new_result(spec)
        res["status"] = status
        if wall_s is not None:
            res["runtime"] = {"wall_s": wall_s}
        (out / "runs").mkdir(exist_ok=True)
        (out / "runs" / f"{spec.run_id}.json").write_text(json.dumps(res), encoding="utf-8")


class ConfigMetadataTests(unittest.TestCase):
    def test_every_shipped_config_loads_metadata_from_the_json(self) -> None:
        for name in ui.discover_benchmark_configs():
            with self.subTest(config=name):
                raw, meta = _raw(name), ui.load_config_metadata(name)
                self.assertEqual(meta.file_name, name)
                self.assertEqual(meta.name, raw["name"])
                self.assertEqual(meta.version, raw["config_version"])
                self.assertEqual(meta.description, raw["description"])
                self.assertEqual(meta.horizons, tuple(raw["horizons"]))
                self.assertEqual(meta.edos, tuple(raw["edos"]))
                self.assertEqual(meta.seeds, tuple(raw["seeds"]))
                self.assertEqual(meta.conditions, tuple(raw["conditions"]))
                self.assertEqual(meta.path_modes, tuple(raw["path_modes"]))
                self.assertEqual(meta.k_max, tuple(p["k_max"] for p in raw["graph_grid"]))
                self.assertEqual(meta.d_max, tuple(p["d_max"] for p in raw["graph_grid"]))
                self.assertEqual(meta.proposal_budget, tuple(p["proposal_budget"] for p in raw["graph_grid"]))
                self.assertEqual(meta.sb_max_iterations, raw["sb"]["max_iterations"])
                self.assertEqual(meta.sb_tolerance, raw["sb"]["tolerance"])
                self.assertEqual(meta.sb_temperature, raw["sb"]["temperature"])
                self.assertEqual(meta.timeout_s, raw["timeout_s"])

    def test_expected_runs_come_from_expand_matrix(self) -> None:
        for name, expected in EXPECTED_RUNS.items():
            with self.subTest(config=name):
                meta = ui.load_config_metadata(name)
                self.assertEqual(meta.expected_runs, expected)
                self.assertEqual(meta.expected_runs, len(expand_matrix(load_config(CONFIG_DIR / name))))

    def test_metadata_is_not_hardcoded(self) -> None:
        config = _custom_config(horizons=[4], seeds=[5, 6], conditions=["full", "no_sb"], edos=[19])
        meta = ui.config_metadata(config)
        self.assertEqual((meta.horizons, meta.seeds, meta.conditions, meta.edos), ((4,), (5, 6), ("full", "no_sb"), (19,)))
        self.assertEqual(meta.expected_runs, 4)

    def test_unknown_or_missing_config_raises_preflight_error(self) -> None:
        with self.assertRaises(ui.PreflightError):
            ui.load_config_metadata("does_not_exist.json")
        with self.assertRaises(ui.PreflightError):
            ui.load_config_metadata("")

    def test_available_filter_choices_match_each_config(self) -> None:
        for name in EXPECTED_TYPES:
            with self.subTest(config=name):
                raw, config = _raw(name), load_config(CONFIG_DIR / name)
                choices = ui.available_filter_choices(config)
                self.assertEqual(choices.horizons, tuple(raw["horizons"]))
                self.assertEqual(choices.edos, tuple(raw["edos"]))
                self.assertEqual(choices.seeds, tuple(raw["seeds"]))
                self.assertEqual(choices.conditions, tuple(raw["conditions"]))

    def test_smoke_metadata_matches_the_spec_example(self) -> None:
        meta = ui.load_config_metadata("m1_smoke.json")
        self.assertEqual((meta.horizons, meta.edos, meta.seeds), ((8, 12), (12,), (0, 1, 2)))
        self.assertEqual(meta.conditions, ("full", "no_gttm", "no_prior", "no_sb"))
        self.assertEqual(meta.expected_runs, 24)


class BenchmarkTypeTests(unittest.TestCase):
    def test_shipped_config_labels(self) -> None:
        for name, label in EXPECTED_TYPES.items():
            with self.subTest(config=name):
                self.assertEqual(ui.benchmark_type(load_config(CONFIG_DIR / name)), label)
                self.assertEqual(ui.load_config_metadata(name).benchmark_type, label)

    def test_dropdown_labels_carry_the_type(self) -> None:
        labels = dict((value, label) for label, value in ui.config_dropdown_choices())
        self.assertEqual(labels["m1_smoke.json"], "[SMOKE] m1_smoke.json")
        self.assertEqual(labels["m1_profile_256.json"], "[PROFILE] m1_profile_256.json")
        self.assertEqual(labels["m1_64_128.json"], "[BENCHMARK] m1_64_128.json")

    def test_explicit_metadata_wins_and_description_is_a_fallback(self) -> None:
        self.assertEqual(ui.benchmark_type(_custom_config(name="m1_64_128", benchmark_type="profile")), "PROFILE")
        self.assertEqual(ui.benchmark_type(_custom_config(name="nightly", description="A smoke check.")), "SMOKE")
        self.assertEqual(ui.benchmark_type(_custom_config(name="nightly", description="Scaling probe.")), "PROFILE")
        self.assertEqual(ui.benchmark_type(_custom_config(name="nightly", description="Ablations.")), "BENCHMARK")
        # an unrecognised explicit value is ignored rather than trusted
        self.assertEqual(ui.benchmark_type(_custom_config(name="nightly", description="x", benchmark_type="huge")), "BENCHMARK")

    def test_shipped_configs_are_untouched_by_labelling(self) -> None:
        for name in EXPECTED_TYPES:
            self.assertNotIn("benchmark_type", _raw(name))  # adding it would change config_hash


class ConditionDescriptionTests(unittest.TestCase):
    def test_spec_texts(self) -> None:
        self.assertEqual(ui.condition_description("full"), "Normal configuration using GTTM, the configured prior, and SB.")
        self.assertEqual(ui.condition_description("no_gttm"), "Disables GTTM scoring while preserving legality constraints.")
        self.assertEqual(ui.condition_description("no_prior"), "Uses NullPrior to measure the contribution of the prior.")
        self.assertEqual(
            ui.condition_description("no_sb"), "Uses the uniform reachable-walk baseline instead of SB reweighting."
        )

    def test_every_known_condition_has_a_description(self) -> None:
        from aimusic.evaluation.config import CONDITIONS

        for cond in CONDITIONS:
            self.assertNotEqual(ui.condition_description(cond), "No description is defined for this condition.", cond)

    def test_config_metadata_describes_exactly_its_conditions(self) -> None:
        meta = ui.load_config_metadata("m1_profile_256.json")
        self.assertEqual(tuple(meta.condition_descriptions), ("full", "no_prior"))
        self.assertEqual(meta.condition_descriptions["no_prior"], ui.condition_description("no_prior"))

    def test_override_via_config_and_unknown_fallback(self) -> None:
        config = _custom_config(condition_descriptions={"full": "Custom text."})
        meta = ui.config_metadata(config)
        self.assertEqual(meta.condition_descriptions["full"], "Custom text.")
        self.assertEqual(meta.condition_descriptions["no_gttm"], ui.condition_description("no_gttm"))
        self.assertEqual(ui.condition_description("brand_new"), "No description is defined for this condition.")
        self.assertEqual(ui.condition_description("brand_new", {"brand_new": "Mine."}), "Mine.")


class TerminologyTests(unittest.TestCase):
    def test_required_terms_are_explained(self) -> None:
        for term in ("Horizon", "EDO", "Seed", "Condition", "Workers", "Proposal budget", "K_max", "D_max", "SB"):
            self.assertIn(term, ui.TERM_HELP)
            self.assertTrue(ui.TERM_HELP[term].strip())
        self.assertEqual(ui.TERM_HELP["D_max"], "Maximum number of outgoing edges retained per state.")
        self.assertIn("19-EDO divides the octave into 19 pitch steps", ui.TERM_HELP["EDO"])
        self.assertIn("Schrödinger Bridge", ui.TERM_HELP["SB"])

    def test_help_is_visible_in_panel_and_terminology_text(self) -> None:
        panel = ui.render_config_panel(ui.load_config_metadata("m1_smoke.json"))
        self.assertIn("Benchmark Configuration", panel)
        self.assertIn('title="Maximum number of outgoing edges retained per state."', panel)
        text = ui.render_terminology_markdown()
        for term in ui.TERM_HELP:
            self.assertIn(f"**{term}**", text)


class PanelRenderingTests(unittest.TestCase):
    def test_panel_shows_config_values_and_escapes_html(self) -> None:
        panel = ui.render_config_panel(ui.load_config_metadata("m1_smoke.json"))
        for fragment in ("SMOKE", "m1_smoke.json", "8, 12 beats", "Expected total runs", "<b>24</b>", "600 s per run",
                         "max iterations 200", "NullPrior"):
            self.assertIn(fragment, panel)
        meta = ui.config_metadata(_custom_config(description="<script>alert(1)</script>"))
        self.assertNotIn("<script>", ui.render_config_panel(meta))

    def test_panel_without_config_shows_the_error(self) -> None:
        self.assertIn("Cannot load", ui.render_config_panel(None, "Cannot load benchmark config x.json"))


class FilterChoiceTests(unittest.TestCase):
    def test_switching_configs_updates_choices(self) -> None:
        smoke = ui.switch_config("m1_smoke.json")
        main = ui.switch_config("m1_64_128.json")
        self.assertEqual((smoke.choices.horizons, smoke.choices.edos), ((8, 12), (12,)))
        self.assertEqual((main.choices.horizons, main.choices.edos), ((64, 128), (12, 19)))
        self.assertEqual(main.choices.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(ui.switch_config("m1_profile_256.json").choices.conditions, ("full", "no_prior"))

    def test_switching_clears_selections_that_are_no_longer_valid(self) -> None:
        current = ui.FilterSelection(conditions=("full", "no_sb"), horizons=(8,), edos=(12,), seeds=(2,))
        sw = ui.switch_config("m1_profile_256.json", current)
        self.assertEqual(sw.selection, ui.FilterSelection(conditions=("full",), horizons=(), edos=(12,), seeds=()))
        # still-valid values survive a switch between configs that share them
        keep = ui.switch_config("m1_64_128.json", ui.FilterSelection(conditions=("no_gttm",), edos=(12,), seeds=(0, 4)))
        self.assertEqual(keep.selection, ui.FilterSelection(conditions=("no_gttm",), horizons=(), edos=(12,), seeds=(0, 4)))

    def test_switch_to_broken_config_raises_preflight_error(self) -> None:
        with self.assertRaises(ui.PreflightError):
            ui.switch_config("nope.json")


class RunCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.smoke = load_config(CONFIG_DIR / "m1_smoke.json")

    def test_empty_filters_select_everything(self) -> None:
        for name, expected in EXPECTED_RUNS.items():
            config = load_config(CONFIG_DIR / name)
            self.assertEqual(ui.selected_run_count(config, (), (), (), ()), expected, name)
            self.assertEqual(ui.validate_filter_selection(config, (), (), (), ()), [])

    def test_count_matches_the_cli_selection(self) -> None:
        from aimusic.evaluation.config import select_specs

        cfg = load_config(CONFIG_DIR / "m1_64_128.json")
        combos = [
            (("full",), (64,), (12,), (0, 1)),
            (("full", "no_sb"), (), (19,), ()),
            ((), (128,), (), (4,)),
        ]
        for conds, hors, edos, seeds in combos:
            expected = len(select_specs(expand_matrix(cfg), conditions=conds or None, horizons=hors or None,
                                        edos=edos or None, seeds=seeds or None))
            self.assertEqual(ui.selected_run_count(cfg, conds, hors, edos, seeds), expected)

    def test_spec_example_matrix_text(self) -> None:
        sel = ui.FilterSelection(conditions=("full", "no_gttm"), horizons=(8,), edos=(12,), seeds=(0, 1))
        n = ui.selected_run_count(self.smoke, sel.conditions, sel.horizons, sel.edos, sel.seeds)
        self.assertEqual(n, 4)
        self.assertEqual(ui.matrix_text(self.smoke, sel, n), "2 conditions × 1 horizon × 2 seeds × 1 EDO = 4 runs")

    def test_matrix_text_for_all_values_and_singular(self) -> None:
        self.assertEqual(
            ui.matrix_text(self.smoke, ui.FilterSelection(), 24), "4 conditions × 2 horizons × 3 seeds × 1 EDO = 24 runs"
        )
        sel = ui.FilterSelection(conditions=("full",), horizons=(8,), seeds=(0,))
        self.assertEqual(ui.matrix_text(self.smoke, sel, 1), "1 condition × 1 horizon × 1 seed × 1 EDO = 1 run")

    def test_matrix_text_mentions_extra_axes_and_skipped_cells(self) -> None:
        config = _custom_config(path_modes=["map", "sample"], conditions=["full", "no_sb"])
        n = ui.selected_run_count(config, (), (), (), ())
        # full: 2 path modes x 2 horizons x 3 seeds = 12; no_sb has no MAP cell: 1 x 2 x 3 = 6
        self.assertEqual(n, 12 + 6)
        text = ui.matrix_text(config, ui.FilterSelection(), n)
        self.assertIn("2 path modes", text)
        self.assertIn("not define", text)

    def test_count_from_the_ui_equals_the_manifest_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_results("m1_smoke.json", Path(tmp), [])
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["expected_runs"]), ui.selected_run_count(self.smoke, (), (), (), ()))


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.smoke = load_config(CONFIG_DIR / "m1_smoke.json")

    def test_values_outside_the_config_are_rejected(self) -> None:
        problems = ui.validate_filter_selection(self.smoke, ("full", "bogus"), (9999,), (19,), (7,))
        self.assertEqual(len(problems), 4)
        joined = " ".join(problems)
        for value in ("bogus", "9999", "19", "7"):
            self.assertIn(value, joined)

    def test_error_message_names_the_config_and_lists_valid_values(self) -> None:
        problems = ui.validate_filter_selection(self.smoke, (), (9999,), (), ())
        msg = ui.format_validation_error(self.smoke, "m1_smoke.json", problems)
        self.assertIn("m1_smoke.json", msg)
        self.assertIn("Horizons: 8, 12", msg)
        self.assertIn("Seeds: 0, 1, 2", msg)
        self.assertIn("EDO: 12", msg)
        self.assertIn("full, no_gttm, no_prior, no_sb", msg)
        self.assertIn("clear the filters to use all values", msg)
        self.assertNotIn("Refused: filters select no runs", msg)

    def test_zero_runs_from_an_undefined_combination_is_rejected(self) -> None:
        config = _custom_config(path_modes=["map"], conditions=["no_sb"])  # uniform bridge has no MAP cell
        self.assertEqual(ui.selected_run_count(config, (), (), (), ()), 0)
        problems = ui.validate_filter_selection(config, (), (), (), ())
        self.assertEqual(problems, [ui.NO_RUNS_MESSAGE])
        self.assertTrue(ui.format_validation_error(config, "x.json", problems).startswith("No runs selected."))

    def test_valid_partial_selection_has_no_problems(self) -> None:
        self.assertEqual(ui.validate_filter_selection(self.smoke, ("full",), (8,), (12,), (0,)), [])


class PreflightIntegrationTests(unittest.TestCase):
    def test_preflight_refuses_invalid_filters_with_a_specific_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ui.PreflightError) as ctx:
                ui.preflight("run", _request(out=tmp, horizons=(9999,), seeds=(0,)))
            msg = str(ctx.exception)
            self.assertIn("m1_smoke.json", msg)
            self.assertIn("Horizons: 8, 12", msg)
            ui.preflight("run", _request(out=tmp, horizons=(8,), seeds=(0,)))

    def test_no_subprocess_or_directory_when_validation_fails(self) -> None:
        launched: list[Any] = []
        real_popen = ui.subprocess.Popen

        def spy(*args: Any, **kwargs: Any) -> Any:
            launched.append(args)
            return real_popen(*args, **kwargs)

        ui.subprocess.Popen = spy  # type: ignore[assignment,misc]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "never_created"
                for action in ("run", "resume"):
                    with self.assertRaises(ui.PreflightError):
                        ui.JOBS.start(action, _request(out=str(out), seeds=(99,)))
                self.assertFalse(out.exists())
        finally:
            ui.subprocess.Popen = real_popen  # type: ignore[misc]
        self.assertEqual(launched, [])

    def test_existing_directory_and_config_safety_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_results("m1_profile_256.json", Path(tmp), [])
            with self.assertRaises(ui.PreflightError):
                ui.preflight("run", _request("m1_smoke.json", tmp))
            with self.assertRaises(ui.PreflightError):
                ui.preflight("resume", _request("m1_smoke.json", tmp))
            ui.preflight("run", _request("m1_profile_256.json", tmp))
            plan = ui.plan_run(_request("m1_smoke.json", tmp))
            self.assertFalse(plan.ok)
            self.assertIn("different config", plan.errors[0])


class RunPlanTests(unittest.TestCase):
    def test_new_directory_starts_new_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = ui.plan_run(_request(out=str(Path(tmp) / "fresh"), conditions=("full", "no_gttm"), horizons=(8,),
                                        seeds=(0, 1)))
            self.assertTrue(plan.ok)
            self.assertEqual(plan.selected_runs, 4)
            self.assertEqual(plan.matrix_text, "2 conditions × 1 horizon × 2 seeds × 1 EDO = 4 runs")
            self.assertIn("Start new runs", plan.action_text)
            self.assertEqual(plan.output_state.finished, 0)  # type: ignore[union-attr]
            self.assertIsNone(plan.estimated_runtime_s)
            text = ui.render_run_preview(plan)
            self.assertIn("Selected runs:", text)
            self.assertIn(str(Path(tmp) / "fresh"), text)

    def test_partial_directory_resumes_and_skips_finished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_results("m1_smoke.json", Path(tmp), ["ok"] * 10)
            plan = ui.plan_run(_request(out=tmp))
            self.assertEqual((plan.output_state.finished, plan.output_state.pending), (10, 14))  # type: ignore[union-attr]
            self.assertIn("Resume: 10 already finished will be skipped, 14 runs will be executed", plan.action_text)

    def test_complete_directory_has_nothing_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_results("m1_smoke.json", Path(tmp), ["ok"] * 24)
            plan = ui.plan_run(_request(out=tmp))
            self.assertIn("Nothing to run", plan.action_text)
            self.assertIn("none (nothing to execute)", ui.render_run_preview(plan))

    def test_failed_runs_are_skipped_unless_retry_is_ticked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_results("m1_smoke.json", Path(tmp), ["ok"] * 5 + ["failed"] * 2)
            skip = ui.plan_run(_request(out=tmp))
            self.assertEqual(skip.output_state.finished, 7)  # type: ignore[union-attr]
            self.assertIn("Retry failed runs", skip.action_text)
            retry = ui.plan_run(_request(out=tmp, retry_failed=True))
            self.assertEqual((retry.output_state.finished, retry.output_state.pending), (5, 19))  # type: ignore[union-attr]

    def test_plan_agrees_with_the_runner_skip_rule(self) -> None:
        from aimusic.evaluation.matrix import _load_json, _run_path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_results("m1_smoke.json", out, ["ok", "failed", "timeout", "crashed", "ok"])
            specs = expand_matrix(load_config(CONFIG_DIR / "m1_smoke.json"))
            for retry in (False, True):
                runner_pending = 0
                for spec in specs:
                    prior = _load_json(_run_path(out, spec))
                    if not (prior is not None and (prior.get("status") == "ok" or not retry)):
                        runner_pending += 1
                plan = ui.plan_run(_request(out=tmp, retry_failed=retry))
                self.assertEqual(plan.output_state.pending, runner_pending)  # type: ignore[union-attr]

    def test_runtime_estimate_uses_same_horizon_and_edo_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # specs are ordered horizon-major: the first 12 are h8 (4 conditions x 3 seeds)
            _write_results("m1_smoke.json", Path(tmp), ["ok"] * 12, wall_s=10.0)
            h8_done = ui.plan_run(_request(out=tmp, horizons=(8,)))
            self.assertEqual(h8_done.estimated_runtime_s, 0.0)
            # nothing pending at h8; h12 has no finished reference -> no guess
            self.assertIsNone(ui.plan_run(_request(out=tmp)).estimated_runtime_s)
            # resume only some h8 runs with a reference: 12 pending h8 runs x 10 s / 2 workers
            with tempfile.TemporaryDirectory() as tmp2:
                _write_results("m1_smoke.json", Path(tmp2), ["ok"] * 6, wall_s=10.0)
                plan = ui.plan_run(_request(out=tmp2, horizons=(8,), workers=2))
                self.assertEqual(plan.estimated_runtime_s, 6 * 10.0 / 2)

    def test_problems_are_reported_not_raised(self) -> None:
        bad_filter = ui.plan_run(_request(out="x", horizons=(9999,)))
        self.assertFalse(bad_filter.ok)
        self.assertIn("Horizons: 8, 12", bad_filter.errors[0])
        self.assertIn("Cannot start", ui.render_run_preview(bad_filter))
        self.assertFalse(ui.plan_run(_request("missing.json", "x")).ok)
        self.assertFalse(ui.plan_run(_request("", "x")).ok)
        no_dir = ui.plan_run(_request(out=""))
        self.assertFalse(no_dir.ok)
        self.assertEqual(no_dir.selected_runs, 24)  # the matrix is still previewed when only the directory is missing


@unittest.skipUnless(HAVE_GRADIO, "gradio not installed")
class PageConstructionTests(unittest.TestCase):
    def test_page_builds_with_dynamic_controls(self) -> None:
        import gradio as gr

        with gr.Blocks() as demo:
            ui.build_evaluation_page(demo)
        labels = {getattr(b, "label", None) for b in demo.blocks.values()}
        for label in ("Benchmark config", "Workers", "Horizons (none = all)", "EDO (none = all)", "Seeds (none = all)"):
            self.assertIn(label, labels)
        # free-text filters are gone: the filter controls only offer values from the config
        groups = {b.label: b for b in demo.blocks.values() if isinstance(b, gr.CheckboxGroup)}
        self.assertEqual([v for _, v in groups["Horizons (none = all)"].choices], [8, 12])
        self.assertEqual([v for _, v in groups["Seeds (none = all)"].choices], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
