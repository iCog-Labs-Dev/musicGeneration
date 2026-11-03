"""Concise Markdown report for a benchmark output directory."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    "Section-wise SB with soft stitching (MUS-16): not implemented; no monolithic vs section-wise comparison.",
    "Factorized corpus NGramPrior (MUS-26) and a trained neural prior (MUS-27): `trained_prior` "
    "reports an explicit `prior_unavailable` failure unless a bundle is configured.",
    "Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`), "
    "decoder-density ablation, ordinary Markov/Viterbi baselines, and the musical-structure metrics "
    "(cadence, motif recurrence, voice leading, groove continuity) are Milestone 2.",
    "Listening-test package: Milestone 3.",
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
    return (_mode(cell), cell["edo"], cell["horizon"], cell["path_mode"], cell["k_max"], cell["d_max"], cell["proposal_budget"], rank)


def _mode(cell: Mapping[str, Any]) -> str:
    return str(cell.get("endpoint_mode", "singleton"))


def _label(cell: Mapping[str, Any]) -> str:
    base = f"{cell['condition']} / {cell['horizon']}b / {cell['edo']}-EDO / {cell['path_mode']}"
    # Cells of different endpoint modes are never merged; the mode is shown whenever it is not the M1 default.
    return base if _mode(cell) == "singleton" else f"{base} / {_mode(cell)}"


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


def render_report(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]], agg: Mapping[str, Any]) -> str:
    cells = sorted(agg["cells"].items(), key=_sort_key)
    repro = (results[0].get("reproducibility") if results else None) or manifest.get("created_with", {})
    n_ok = sum(1 for r in results if r["status"] == "ok")
    methods = ", ".join(
        f"`{m}`" for m in sorted({c.get("generation_method", "method_a_singleton_endpoints") for _, c in cells})
    ) or "`method_a_singleton_endpoints`"
    has_distribution = any(_mode(c) == "distribution" for _, c in cells)
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
        f"- Generation: {methods}, single pass. `full` uses the **placeholder** structured "
        "prior (a deterministic heuristic, not a trained or corpus-backed model).",
        "- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, path modes or "
        "endpoint modes. "
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
    lines += [f"- {item}" for item in DEFERRED] + [""]
    return "\n".join(lines)
