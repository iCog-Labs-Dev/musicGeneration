"""Concise Markdown report for a benchmark output directory."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from aimusic.evaluation.stats import MIN_PAIRS_FOR_CI

MIN_PAIRS_FOR_CI_TEXT = str(MIN_PAIRS_FOR_CI)

CONDITION_ORDER = ("full", "no_gttm", "no_prior", "no_sb", "trained_prior")

MUS15_SINGLETON_ONLY = (
    "Distribution-to-distribution Method A (MUS-15) was **not evaluated** in this output: every cell is "
    "singleton-endpoint, so endpoint-marginal error, residuals, entropy and calibration are not measured. "
    "Use `config/benchmarks/m2_distribution_method_a.json`."
)
MUS15_DISTRIBUTION = (
    "Endpoint-distribution evaluation is a correctness/fidelity check of Method A at small horizons "
    "(8/16 beats here), not a long-form quality benchmark; calibration compares repeated samples from "
    "one solved bridge and says nothing about MAP paths."
)

DEFERRED = (
    "Method B start-middle-return planning (MUS-18): not implemented in `plans.py`; no A/B comparison.",
    "Factorized corpus NGramPrior (MUS-26) and a trained neural prior (MUS-27): `trained_prior` "
    "reports an explicit `prior_unavailable` failure unless a bundle is configured.",
    "Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`), "
    "decoder-density ablation, ordinary Markov/Viterbi baselines, and the musical-structure metrics "
    "(cadence, motif recurrence, voice leading, groove continuity) are Milestone 2.",
    "Listening-test package: Milestone 3.",
)


MUS16_NOT_RUN = (
    "Section-wise SB with soft stitching (MUS-16) was **not evaluated** in this output: every cell is "
    "single-pass, so there is no monolithic vs section-wise comparison. "
    "Use `config/benchmarks/m2_section_wise_sb.json`."
)
MUS16_RUN = (
    "Section-wise SB with soft stitching (MUS-16) is evaluated on the cells above only; see "
    "`## Section-wise SB evaluation` for what that does and does not show."
)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _ci(summary: Mapping[str, Any], digits: int = 3) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "-"
    if summary.get("ci95_low") is None:
        return f"{_fmt(summary['mean'], digits)} (n=1)"
    return f"{_fmt(summary['mean'], digits)} [{_fmt(summary['ci95_low'], digits)}, {_fmt(summary['ci95_high'], digits)}]"


def _sort_key(item: Any) -> Any:
    _, cell = item
    cond = cell["condition"]
    rank = CONDITION_ORDER.index(cond) if cond in CONDITION_ORDER else len(CONDITION_ORDER)
    return (_mode(cell), cell["edo"], cell["horizon"], cell["path_mode"], cell["k_max"], cell["d_max"], cell["proposal_budget"], rank, _sectioning(cell))


def _mode(cell: Mapping[str, Any]) -> str:
    return str(cell.get("endpoint_mode", "singleton"))


def _sectioning(cell: Mapping[str, Any]) -> str:
    return str(cell.get("sectioning", "single_pass"))


def _label(cell: Mapping[str, Any]) -> str:
    base = f"{cell['condition']} / {cell['horizon']}b / {cell['edo']}-EDO / {cell['path_mode']}"
    # Cells of different endpoint modes are never merged; the mode is shown whenever it is not the M1 default.
    label = base if _mode(cell) == "singleton" else f"{base} / {_mode(cell)}"
    # Likewise single-pass and section-wise cells: the sectioning mode is shown whenever it is not single-pass.
    return label if _sectioning(cell) == "single_pass" else f"{label} / {_sectioning(cell)}"


def _sci(summary: Mapping[str, Any]) -> str:
    """Scientific notation: marginal errors are ~1e-9, which fixed-point formatting would print as 0.000."""
    if not summary or summary.get("n", 0) == 0:
        return "-"
    return f"{summary['mean']:.2e}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out + [""]


def _metric(cell: Mapping[str, Any], name: str, digits: int = 3) -> str:
    return _ci(cell["metrics"].get(name, {}), digits)


def _cell_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    from aimusic.evaluation.schema import RunSpec

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for r in results:
        grouped.setdefault(RunSpec.from_dict(dict(r["spec"])).group_key, []).append(r)
    return grouped


def _endpoint_section(cells: Sequence[Any], results: Sequence[Mapping[str, Any]]) -> List[str]:
    """``## Endpoint-distribution evaluation``: distribution cells only; singleton cells are listed as unavailable."""
    lines = ["## Endpoint-distribution evaluation", ""]
    dist = [(k, c) for k, c in cells if _mode(c) == "distribution"]
    singleton = [c for _, c in cells if _mode(c) == "singleton"]
    if singleton:
        lines += [
            f"Distribution metrics unavailable: singleton endpoint mode ({len(singleton)} cell(s) not shown; "
            "singleton and distribution cells are never combined).",
            "",
        ]
    if not dist:
        return lines
    by_cell = _cell_results(results)

    lines += ["### Endpoint handling outcome", "",
              "How each run's requested endpoints were treated by the MUS-15 policy "
              "(`no_modification` = all support feasible, nothing removed; `removed_and_renormalized` = "
              "unreachable states dropped and survivors renormalized; `strict_failure` = the policy forbade "
              "changing the request, so the run failed before solving).", ""]
    rows = []
    for key, c in dist:
        counts: Dict[str, int] = {}
        for r in by_cell.get(key, []):
            ep = (r.get("metrics") or {}).get("endpoint") or {}
            status = str(ep.get("accounting_status", "no_diagnostics"))
            counts[status] = counts.get(status, 0) + 1
        policy = next(
            (r["spec"].get("endpoint_policy") for r in by_cell.get(key, [])), "-"
        )
        rows.append([_label(c), f"`{policy}`", ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "-"])
    lines += _table(["cell", "policy", "outcome (runs)"], rows)

    for side, title in (("start", "Start"), ("terminal", "Terminal")):
        lines += [f"### {title} boundary: mass accounting and support", ""]
        rows = []
        for _, c in dist:
            def m(name: str, digits: int = 3, side: str = side, c: Mapping[str, Any] = c) -> str:
                return _metric(c, f"endpoint.{side}.{name}", digits)

            rows.append([
                _label(c), m("support_size_requested", 1), m("support_size_effective", 1),
                m("requested_mass"), m("feasible_mass"), m("removed_mass"), m("trimmed_mass"),
                m("effective_mass"), m("realized_mass"),
            ])
        lines += _table(
            ["cell", "support requested", "support effective", "requested mass", "feasible mass",
             "removed mass", "trimmed mass", "effective mass", "realized mass"],
            rows,
        )
    lines += ["Masses are probability mass; `effective` is what the solver was asked to match (1 after "
              "renormalization) and `realized` is the bridge's measured marginal.", ""]

    lines += ["### Marginal error and residuals", ""]
    rows = []
    for _, c in dist:
        for side in ("start", "terminal"):
            def ms(name: str, side: str = side, c: Mapping[str, Any] = c) -> Mapping[str, Any]:
                return c["metrics"].get(f"endpoint.{side}.{name}", {})

            rows.append([
                _label(c), side,
                _sci(ms("marginal_l1_vs_requested")), _sci(ms("marginal_max_abs_vs_requested")),
                _sci(ms("marginal_l1_vs_effective")), _sci(ms("marginal_max_abs_vs_effective")),
                _sci(ms("residual_max_abs")), _fmt(ms("residual_within_tolerance").get("mean"), 2),
                _fmt(ms("requested_equals_realized_within_tolerance").get("mean"), 2),
            ])
    lines += _table(
        ["cell", "side", "L1 vs requested", "max abs vs requested", "L1 vs effective", "max abs vs effective",
         "max per-state residual", "within tolerance (rate)", "realized = requested (rate)"],
        rows,
    )
    lines += ["`vs requested` includes any mass removed or trimmed by the policy; `vs effective` is the solver's "
              "own marginal error (residual = realized - effective per state, checked against "
              "`endpoint_marginal_tolerance`). Per-state rows are in `metrics.endpoint.<side>.residuals`.", ""]

    lines += ["### Entropy (nats)", ""]
    rows = []
    for _, c in dist:
        for side in ("start", "terminal"):
            rows.append([
                _label(c), side, _metric(c, f"endpoint.{side}.entropy_requested"),
                _metric(c, f"endpoint.{side}.entropy_effective"),
                _metric(c, f"endpoint.{side}.entropy_empirical"),
            ])
    lines += _table(["cell", "side", "requested", "effective", "empirical (sampled)"], rows)

    lines += ["### Calibration (repeated samples from the solved bridge)", ""]
    rows = []
    for key, c in dist:
        runs = by_cell.get(key, [])
        statuses: Dict[str, int] = {}
        samples = []
        for r in runs:
            ep = (r.get("metrics") or {}).get("endpoint") or {}
            if r["status"] != "ok":
                continue
            st = str(ep.get("calibration_status", "unavailable_no_diagnostics"))
            statuses[st] = statuses.get(st, 0) + 1
            samples.append(int(ep.get("calibration_n_samples", 0)))
        for side in ("start", "terminal"):
            rows.append([
                _label(c), side,
                ", ".join(f"{k} x{v}" for k, v in sorted(statuses.items())) or "-",
                str(max(samples)) if samples else "-",
                _metric(c, f"endpoint.{side}.calibration_l1_vs_effective"),
                _metric(c, f"endpoint.{side}.calibration_expected_l1_sampling_noise"),
                _metric(c, f"endpoint.{side}.calibration_max_abs_z_vs_effective", 2),
            ])
    lines += _table(
        ["cell", "side", "status (runs)", "samples/run", "L1 vs effective", "expected L1 noise",
         "max abs z"],
        rows,
    )
    lines += ["Calibration needs at least 30 repeated samples; otherwise it is reported as unavailable (shown as `-`), "
              "never as zero. A single MAP or sampled path is not a calibration. `expected L1 noise` is the L1 a "
              "perfectly calibrated sampler would show from finite sampling alone.", ""]
    return lines


def _paired_cell(d: Optional[Mapping[str, Any]], level: str, digits: int = 3) -> str:
    """Paired difference: a t-interval only with enough matched seeds, otherwise mean and range."""
    if not d or d.get("n", 0) == 0:
        return "-"
    if level == "paired_ci" and d.get("ci95_low") is not None:
        return _ci(d, digits)
    if d["n"] == 1:
        return f"{_fmt(d['mean'], digits)} (n=1)"
    return f"{_fmt(d['mean'], digits)} (range {_fmt(d['min'], digits)} to {_fmt(d['max'], digits)}, n={d['n']})"


def _section_names_of(cell: Mapping[str, Any]) -> List[str]:
    return [str(n) for n in cell.get("section_names") or []]


def _failure_rows(results: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    from aimusic.evaluation.schema import RunSpec

    counts: Dict[Any, Dict[str, Any]] = {}
    for r in results:
        spec = r["spec"]
        if r["status"] == "ok" or spec.get("sectioning") != "section_wise":
            continue
        f = r.get("failure") or {}
        key = (
            RunSpec.from_dict(dict(spec)).group_key, f.get("failure_class") or f.get("kind") or r["status"],
            f.get("section_name") or "-", f.get("planner_stage") or f.get("stage") or "-",
        )
        entry = counts.setdefault(key, {"n": 0, "seeds": [], "cause": f.get("cause_kind") or "-", "msg": f.get("message", "")})
        entry["n"] += 1
        entry["seeds"].append(int(spec["seed"]))
    rows = []
    for (cell, klass, section, stage), e in sorted(counts.items(), key=lambda kv: str(kv[0])):
        msg = str(e["msg"]).replace("|", "/").replace("\n", " ")
        rows.append([f"`{cell}`", klass, section, stage, str(e["cause"]), f"x{e['n']} (seeds {sorted(e['seeds'])})", msg[:140]])
    return rows


def _section_wise_section(
    cells: Sequence[Any], results: Sequence[Mapping[str, Any]], agg: Mapping[str, Any]
) -> List[str]:
    """``## Section-wise SB evaluation``: monolithic vs section-wise, per-section and join behavior."""
    lines = ["## Section-wise SB evaluation", ""]
    sw_cells = [(k, c) for k, c in cells if _sectioning(c) == "section_wise"]
    if not sw_cells:
        lines += ["Section-wise metrics unavailable: every cell in this output used single-pass planning "
                  "(`sectioning = single_pass`). Nothing is shown for section-wise cells, and no values are imputed.", ""]
        return lines
    comparison: Mapping[str, Any] = agg.get("sectioning_comparison") or {}
    cell_map = dict(cells)
    names = _section_names_of(sw_cells[0][1])
    n_pairs = [e["n_pairs"] for e in comparison.values()]
    paired_ok = bool(comparison) and all(e["ci_level"] == "paired_ci" for e in comparison.values())
    lines += [
        "Monolithic Method A (`single_pass`) against section-wise Method A with soft stitching "
        f"(`section_wise`, sections {', '.join(f'`{n}`' for n in names)}). Pairs are matched on seed, horizon, EDO, "
        "endpoint mode, graph limits, prior, path mode, style and the authored section plan; only the sectioning mode "
        "differs. `single_pass` and `section_wise` cells are never pooled.",
        "",
    ]
    if not comparison:
        lines += ["> **No matched pairs:** the output holds no `single_pass` cell that matches a `section_wise` cell, so "
                  "nothing below is a comparison.", ""]
    elif paired_ok:
        lines += [f"> **Comparison type: paired confidence intervals** (at least {MIN_PAIRS_FOR_CI_TEXT} matched seeds in "
                  "every pair). Differences are `section_wise - single_pass` with a 95% t-interval over seeds."]
        lines += [""]
    else:
        lines += [f"> **Comparison type: descriptive.** Matched seeds per pair: {', '.join(str(n) for n in n_pairs)}; "
                  f"a paired confidence interval needs at least {MIN_PAIRS_FOR_CI_TEXT}. Differences below are "
                  "`section_wise - single_pass` shown as mean and range over matched seeds, not as significance tests.", ""]

    # --- success levels -------------------------------------------------------------------------------
    lines += ["### Success and failure", "",
              "Three levels are reported separately: every **section solved**, every **join valid** (within the "
              "stitching tolerance) and the **whole-song path assembled**. A song can pass one and fail the next.", ""]
    by_cell = _cell_results(results)
    rows = []
    for key, c in sw_cells:
        runs = by_cell.get(key, [])
        status = [r.get("section_wise_status") or {} for r in runs]
        solved = sum(1 for st in status if st.get("sections_solved"))
        joins = sum(1 for st in status if st.get("joins_valid"))
        built = sum(1 for st in status if st.get("path_assembled"))
        lo, hi = c["failure_rate_ci95"]
        rows.append([_label(c), f"{solved}/{len(runs)}", f"{joins}/{len(runs)}", f"{built}/{len(runs)}",
                     f"{c['failure_rate']:.2f} [{lo:.2f}, {hi:.2f}]"])
    lines += _table(["cell", "all sections solved", "all joins valid", "whole-song path assembled",
                     "failure rate [Wilson 95%]"], rows)
    rows = []
    for entry in comparison.values():
        sp, sw = cell_map.get(entry["single_pass_cell"]), cell_map.get(entry["section_wise_cell"])
        if sp is None or sw is None:
            continue
        p = entry["status_pairs"]
        rows.append([_label(sw).replace(" / section_wise", ""), f"{sp['n_ok']}/{sp['n_runs']}", f"{sw['n_ok']}/{sw['n_runs']}",
                     str(p["both_ok"]), str(p["only_single_pass_ok"]), str(p["only_section_wise_ok"]), str(p["both_failed"])])
    if rows:
        lines += _table(["matched cell", "single_pass ok", "section_wise ok", "both ok", "only single_pass ok",
                         "only section_wise ok", "both failed"], rows)

    # --- runtime / memory -----------------------------------------------------------------------------
    lines += ["### Runtime and memory", ""]
    rows = []
    for entry in comparison.values():
        sp, sw = cell_map.get(entry["single_pass_cell"]), cell_map.get(entry["section_wise_cell"])
        if sp is None or sw is None:
            continue
        m = entry["metrics"]
        rows.append([
            _label(sw).replace(" / section_wise", ""),
            _metric(sp, "runtime.wall_s", 1), _metric(sw, "runtime.wall_s", 1),
            _paired_cell(m.get("runtime.wall_s"), entry["ci_level"], 1),
            _metric(sp, "runtime.peak_rss_mb", 0), _metric(sw, "runtime.peak_rss_mb", 0),
            _paired_cell(m.get("runtime.peak_rss_mb"), entry["ci_level"], 1),
        ])
    if rows:
        lines += _table(["matched cell", "single_pass wall s", "section_wise wall s", "delta wall s",
                         "single_pass peak RSS MB", "section_wise peak RSS MB", "delta peak RSS MB"], rows)
    lines += ["Wall time includes graph construction and bridge solving for every section. Peak RSS is the process "
              "high-water mark of one run (each run is its own process), so it is a whole-run figure only: per-section "
              "peak memory is not measurable.", ""]

    # --- tension / path / endpoint -------------------------------------------------------------------
    lines += ["### Tension, path and endpoints", "",
              "Both modes are scored against the **same authored section-plan targets** (`tension_section_plan`), because "
              "the monolithic planner itself targets a single whole-song arc. Whole-song `tension` against each planner's own "
              "targets is in `## Path, tension and GTTM` above.", ""]
    shown = (
        ("tension_section_plan.mean_absolute_error", "tension MAE (section plan)"),
        ("tension_section_plan.shape_correlation", "tension shape corr (section plan)"),
        ("path.distinct_state_fraction", "distinct states"),
        ("path.chord_change_rate", "chord change rate"),
        ("path.role_change_rate", "role change rate"),
        ("path.groove_change_rate", "groove change rate"),
        ("gttm.mean_gttm_score", "mean GTTM score/edge"),
        ("sb.effective_entropy", "SB effective entropy"),
    )
    rows = []
    for entry in comparison.values():
        sp, sw = cell_map.get(entry["single_pass_cell"]), cell_map.get(entry["section_wise_cell"])
        if sp is None or sw is None:
            continue
        label = _label(sw).replace(" / section_wise", "")
        for name, title in shown:
            rows.append([label, title, _metric(sp, name), _metric(sw, name), _paired_cell(entry["metrics"].get(name), entry["ci_level"])])
    if rows:
        lines += _table(["matched cell", "metric", "single_pass", "section_wise", "delta (sw - sp)"], rows)
    rows = []
    for _, c in sw_cells:
        for side in ("start", "terminal"):
            rows.append([_label(c), f"whole-song {side}", _metric(c, f"endpoint.{side}.requested_mass"),
                         _metric(c, f"endpoint.{side}.effective_mass"), _metric(c, f"endpoint.{side}.realized_mass"),
                         _sci(c["metrics"].get(f"endpoint.{side}.marginal_l1_vs_effective", {}))])
    lines += _table(["section-wise cell", "outer endpoint", "requested mass", "effective mass", "realized mass",
                     "L1 vs effective"], rows)
    lines += ["Outer endpoints only; per-section requested/effective/realized mass (including the shared interior "
              "boundaries) is in the per-section table and in `per_section_endpoint_metrics`. The SB entropy of a "
              "section-wise run summarizes independent section solves and is not the entropy of a joint whole-song "
              "bridge. Lower tension error or fewer changes are not, by themselves, musical quality claims.", ""]

    # --- per-section ----------------------------------------------------------------------------------
    lines += ["### Per-section summary", "",
              "One row per section of each section-wise cell, aggregated over seeds whose run succeeded. Tension is "
              "measured on the whole-song realized curve sliced to the section; `mono MAE` is the monolithic run's "
              "error on the same slice and the same target.", ""]
    rows = []
    for key, c in sw_cells:
        sp_cell = None
        for entry in comparison.values():
            if entry["section_wise_cell"] == key:
                sp_cell = cell_map.get(entry["single_pass_cell"])
        runs = by_cell.get(key, [])
        for name in _section_names_of(c):
            pre = f"section.{name}"
            statuses: Dict[str, int] = {}
            for r in runs:
                names_r = r.get("section_names") or []
                if name in names_r and r.get("per_section_status"):
                    st = r["per_section_status"][names_r.index(name)]
                    statuses[st] = statuses.get(st, 0) + 1
            mono = _metric(sp_cell, f"tension_section_plan.per_section.{name}.mean_absolute_error") if sp_cell else "-"
            rows.append([
                _label(c).replace(" / section_wise", ""), name, _metric(c, f"{pre}.n_beats", 0),
                ", ".join(f"{k} x{v}" for k, v in sorted(statuses.items())) or "-",
                _metric(c, f"{pre}.runtime.runtime_s", 2), _metric(c, f"{pre}.graph.layer_width_mean", 1),
                _metric(c, f"{pre}.graph.retained_count", 0),
                _metric(c, f"{pre}.sb.iterations", 1), _metric(c, f"{pre}.sb.converged", 2),
                _metric(c, f"{pre}.sb.effective_entropy"),
                _metric(c, f"{pre}.tension.mean_absolute_error"), mono,
                _metric(c, f"{pre}.tension.shape_correlation"),
                _metric(c, f"{pre}.path.chord_change_rate"),
            ])
    lines += _table(["cell", "section", "beats", "status (runs)", "runtime s", "layer width", "retained states",
                     "SB iters", "SB converged", "SB entropy", "tension MAE", "mono MAE", "tension corr",
                     "chord change rate"], rows)
    rows = []
    for key, c in sw_cells:
        for name in _section_names_of(c):
            pre = f"section.{name}"
            for side in ("start", "terminal"):
                rows.append([
                    _label(c).replace(" / section_wise", ""), name, side,
                    _metric(c, f"{pre}.endpoint.{side}.requested_mass"), _metric(c, f"{pre}.endpoint.{side}.effective_mass"),
                    _metric(c, f"{pre}.endpoint.{side}.realized_mass"),
                    _sci(c["metrics"].get(f"{pre}.endpoint.{side}.marginal_l1_vs_effective", {})),
                ])
    lines += ["#### Per-section endpoint mass", ""]
    lines += _table(["cell", "section", "side", "requested mass", "effective mass", "realized mass", "L1 vs effective"], rows)
    lines += ["Each interior boundary is one authored distribution shared by both neighbours; a section may not "
              "trim or drop it, so requested = effective = realized there unless that section failed.", ""]

    # --- stitching ------------------------------------------------------------------------------------
    lines += ["### Stitching and joins", "",
              "Costs are the planner's `StitchCost` components (mismatch in [0, 1]); the **worse** of the incoming edge "
              "(last transition of the left section) and the outgoing edge (first transition of the right section) is "
              "reported. Continuity is `1 - mismatch`. `within tolerance` compares the stitching cost with the configured "
              "`max_cost`.", ""]
    rows = []
    for _, c in sw_cells:
        j = lambda n, c=c: _metric(c, f"join.{n}")  # noqa: E731
        rows.append([
            _label(c), j("join_count"), j("key_distance_jump_mean"), j("chord_distance_jump_mean"),
            j("meter_continuity_mean"), j("groove_continuity_mean"), j("role_continuity_mean"), j("head_continuity_mean"),
            j("boundary_continuity_mean"), j("stitching_cost_mean"), j("stitching_cost_worst"), j("within_tolerance_rate"),
        ])
    lines += _table(["cell", "joins", "key jump", "chord jump", "meter cont.", "groove cont.", "role cont.",
                     "head cont.", "boundary cont.", "cost (mean)", "cost (worst)", "within tolerance (rate)"], rows)
    rows = []
    for key, c in sw_cells:
        names_c = _section_names_of(c)
        for i in range(max(len(names_c) - 1, 0)):
            pre = f"join.{i}"
            rows.append([
                _label(c), f"{names_c[i]} -> {names_c[i + 1]}", _metric(c, f"{pre}.time_index", 0),
                _metric(c, f"{pre}.boundary_support_size", 1), _metric(c, f"{pre}.key_distance_jump"),
                _metric(c, f"{pre}.chord_distance_jump"), _metric(c, f"{pre}.groove_continuity", 2),
                _metric(c, f"{pre}.role_continuity", 2), _metric(c, f"{pre}.head_continuity", 2),
                _metric(c, f"{pre}.boundary_continuity", 2), _metric(c, f"{pre}.stitching_cost"),
                _metric(c, f"{pre}.tolerance_margin"), _metric(c, f"{pre}.within_tolerance", 2),
            ])
    lines += ["#### Join detail", ""]
    lines += _table(["cell", "join", "beat", "boundary support", "key jump", "chord jump", "groove cont.", "role cont.",
                     "head cont.", "boundary cont.", "cost", "tolerance margin", "within tolerance"], rows)
    lines += ["The planner enforces `max_cost` while building the section graphs, so a run that completes always has "
              "every join within tolerance; a join that cannot satisfy it is a section/join failure, listed below, "
              "not a silent pass.", ""]

    # --- local failures -------------------------------------------------------------------------------
    lines += ["### Section-local failures", ""]
    rows = _failure_rows(results)
    if rows:
        lines += _table(["cell", "class", "section", "planner stage", "cause", "runs", "message"], rows)
        lines += ["`run_section_wise` raises on the first failing section or join and returns no partial result, so only the "
                  "failing section and stage are known for a failed run; sections after it are `not_reached` and nothing "
                  "is imputed for them.", ""]
    else:
        lines += ["No section-wise run failed.", ""]

    # --- limitations ----------------------------------------------------------------------------------
    lines += ["### Limitations of this comparison", "",
              "- **Not a quality ranking.** Section-wise generation succeeding does not make it better; the tables above "
              "are trade-offs (runtime, memory, success rate, tension and path statistics, join continuity), each with its "
              "own direction.",
              "- Section-wise Method A solves each section independently against shared boundary marginals, so it "
              "approximates, and can differ from, the joint monolithic bridge; entropy and MAP scores are not comparable "
              "one-to-one.",
              "- Tension error is a proxy against hand-authored arcs, not a listening result; the monolithic planner's own "
              "target is a single whole-song arc, which is why both are also scored on the shared section plan.",
              "- Per-section runtime comes from timing the public graph-build and bridge-solve calls; boundary planning, joint "
              "path extraction and join validation are reported once as `section_wise_overhead_s`. Per-section peak memory "
              "is not available.",
              "- Cells are small (few seeds, short horizons, one EDO, one meter) and the placeholder prior is not trained. "
              "A restricted 4/4 style is used because short sections cannot preserve the shared boundary under the default "
              "mixed 4/4-5/4-7/4 style; that is a MUS-16 limitation, recorded in the docs, not hidden here.",
              ""]
    return lines


def render_report(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]], agg: Mapping[str, Any]) -> str:
    cells = sorted(agg["cells"].items(), key=_sort_key)
    repro = (results[0].get("reproducibility") if results else None) or manifest.get("created_with", {})
    n_ok = sum(1 for r in results if r["status"] == "ok")
    methods = ", ".join(
        f"`{m}`" for m in sorted({c.get("generation_method", "method_a_singleton_endpoints") for _, c in cells})
    ) or "`method_a_singleton_endpoints`"
    has_distribution = any(_mode(c) == "distribution" for _, c in cells)
    has_section_wise = any(_sectioning(c) == "section_wise" for _, c in cells)
    sectioning_text = (
        "single pass and section-wise (MUS-16, reported as separate cells)"
        if has_section_wise and any(_sectioning(c) == "single_pass" for _, c in cells)
        else "section-wise (MUS-16)" if has_section_wise else "single pass"
    )
    lines: List[str] = [
        f"# MUS-28 benchmark report: {manifest['config_name']}",
        "",
        f"- Config: `{manifest['config_name']}` version `{manifest['config_version']}`, "
        f"hash `{manifest['config_hash'][:12]}`",
        f"- Code: commit `{(repro.get('git_commit') or 'unknown')[:12]}` on `{repro.get('git_branch')}`"
        f"{' (dirty working tree)' if repro.get('git_dirty') else ''}; Python {repro.get('python')}, "
        f"numpy {repro.get('numpy')}",
        f"- Runs: {agg['completed_runs']}/{agg['expected_runs']} completed, {n_ok} ok, "
        f"{len(results) - n_ok} failed/timeout/crashed",
        f"- Generation: {methods}, {sectioning_text}. `full` uses the **placeholder** structured "
        "prior (a deterministic heuristic, not a trained or corpus-backed model).",
        "- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, path modes, "
        "endpoint modes or sectioning modes. "
        "No pass/fail thresholds are defined yet, so nothing here is a quality claim.",
        "",
    ]
    if agg.get("missing_run_ids"):
        lines += [f"> **Incomplete:** {len(agg['missing_run_ids'])} expected runs have no result yet.", ""]

    lines += ["## Success and failure", ""]
    rows = []
    for _, c in cells:
        lo, hi = c["failure_rate_ci95"]
        kinds = ", ".join(f"{k} x{v}" for k, v in sorted(c["failure_kinds"].items())) or "-"
        rows.append([_label(c), f"{c['n_ok']}/{c['n_runs']}", f"{c['failure_rate']:.2f} [{lo:.2f}, {hi:.2f}]", kinds])
    lines += _table(["cell", "ok/runs", "failure rate [Wilson 95%]", "failure kinds"], rows)

    lines += ["## Runtime and memory", ""]
    rows = []
    for _, c in cells:
        m = c["metrics"]
        wall = m.get("runtime.wall_s", {})
        rows.append([
            _label(c),
            _ci(wall, 1),
            _fmt(wall.get("median"), 1) if wall.get("n") else "-",
            _fmt(wall.get("p95"), 1) if wall.get("n") else "-",
            _metric(c, "runtime.peak_rss_mb", 0),
            _metric(c, "runtime.rss_growth_mb", 0),
        ])
    lines += _table(["cell", "wall s", "median s", "p95 s", "peak RSS MB (process)", "RSS growth MB (run only)"], rows)

    lines += ["## Graph support", ""]
    rows = []
    for _, c in cells:
        uniq = c["metrics"].get("graph.unique_count", {})
        sd = _fmt(uniq.get("sd"), 1) if uniq.get("sd") is not None else "-"
        rows.append([
            _label(c),
            _metric(c, "graph.layer_width_mean", 1),
            _metric(c, "graph.legal_fraction"),
            _metric(c, "graph.unique_fraction"),
            _metric(c, "graph.retention_fraction"),
            _fmt(uniq.get("mean"), 0) if uniq.get("n") else "-",
            sd,
            _metric(c, "graph.inactive_feature_count", 1),
        ])
    lines += _table(
        ["cell", "layer width", "legal/proposed", "unique/legal", "retained/raw edges",
         "unique cands", "support sd (seeds)", "inactive GTTM features"],
        rows,
    )
    inactive: Dict[str, int] = {}
    for r in results:
        if r["status"] == "ok":
            for name in r["metrics"]["graph"].get("inactive_features", []):
                inactive[name] = inactive.get(name, 0) + 1
    inactive_text = ", ".join(f"`{k}` ({v} runs)" for k, v in sorted(inactive.items())) or "none"
    lines += [
        "`scored/legal` is 0 unless `prior_guided_proposals` is enabled. Prune/reject reason counts are in "
        "the raw `metrics.graph.prune_reason_counts` fields. Registered GTTM features with no non-zero "
        f"activation on any retained edge: {inactive_text}. Intervals on fractions can overshoot [0, 1] "
        "(t-interval).",
        "",
    ]

    lines += ["## Schrodinger bridge", ""]
    rows = []
    for _, c in cells:
        if c["condition"] == "no_sb":
            continue
        rows.append([
            _label(c),
            _metric(c, "sb.converged", 2),
            _metric(c, "sb.iterations", 1),
            _metric(c, "sb.effective_entropy"),
        ])
    lines += _table(["cell", "converged rate", "iterations", "effective entropy"], rows)
    if any(_mode(c) == "singleton" for _, c in cells):
        lines += ["With singleton endpoints the bridge is degenerate (about one iteration); see Deferred.", ""]

    lines += ["## Path, tension and GTTM", ""]
    rows = []
    for _, c in cells:
        rows.append([
            _label(c),
            _metric(c, "path.length_ok", 2),
            _metric(c, "path.distinct_state_fraction"),
            _metric(c, "path.chord_change_rate"),
            _metric(c, "tension.mean_absolute_error"),
            _metric(c, "tension.shape_correlation"),
            _metric(c, "gttm.mean_gttm_score"),
        ])
    lines += _table(
        ["cell", "length ok", "distinct states", "chord change rate", "tension MAE",
         "tension shape corr", "mean GTTM score/edge"],
        rows,
    )
    lines += ["`mean GTTM score/edge` uses the per-family feature weights, not `lambda_gttm`, so it is "
              "comparable between `full` and `no_gttm`.", ""]

    lines += _endpoint_section(cells, results)
    lines += _section_wise_section(cells, results, agg)

    lines += ["## Ablations: paired differences vs `full` (same seeds)", ""]
    shown = (
        "runtime.wall_s", "tension.mean_absolute_error", "tension.shape_correlation",
        "gttm.mean_gttm_score", "path.distinct_state_fraction", "path.chord_change_rate",
    )
    rows = []
    for key, entry in sorted(agg["paired_vs_control"].items()):
        cell = agg["cells"][key]
        for name in shown:
            d = entry["metrics"].get(name)
            if not d or d.get("n", 0) == 0:
                continue
            excl = d.get("ci_excludes_zero")
            rows.append([
                _label(cell), name, str(d["n"]), _ci(d, 3),
                "-" if excl is None else ("yes" if excl else "no"),
            ])
    if rows:
        lines += _table(["cell", "metric", "paired n", "delta vs full [95% CI]", "CI excludes 0"], rows)
        lines += ["With few seeds these intervals are wide; read them as descriptive, not as significance tests.", ""]
    else:
        lines += ["No ablation cell has a matching `full` cell yet.", ""]

    lines += ["## Not measured / deferred", ""]
    lines += [f"- {MUS15_DISTRIBUTION if has_distribution else MUS15_SINGLETON_ONLY}"]
    lines += [f"- {MUS16_RUN if has_section_wise else MUS16_NOT_RUN}"]
    lines += [f"- {item}" for item in DEFERRED] + [""]
    return "\n".join(lines)
