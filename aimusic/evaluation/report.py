"""Concise Markdown report for a benchmark output directory."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

CONDITION_ORDER = ("full", "no_gttm", "no_prior", "no_sb", "trained_prior")

DEFERRED = (
    "True distribution-to-distribution Method A (MUS-15): generation still collapses to one start "
    "and one terminal state, so SB convergence is trivial and endpoint-marginal error, residuals "
    "and calibration are **not measured**.",
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
    return (cell["edo"], cell["horizon"], cell["path_mode"], cell["k_max"], cell["d_max"], cell["proposal_budget"], rank)


def _label(cell: Mapping[str, Any]) -> str:
    return f"{cell['condition']} / {cell['horizon']}b / {cell['edo']}-EDO / {cell['path_mode']}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out + [""]


def _metric(cell: Mapping[str, Any], name: str, digits: int = 3) -> str:
    return _ci(cell["metrics"].get(name, {}), digits)


def render_report(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]], agg: Mapping[str, Any]) -> str:
    cells = sorted(agg["cells"].items(), key=_sort_key)
    repro = (results[0].get("reproducibility") if results else None) or manifest.get("created_with", {})
    n_ok = sum(1 for r in results if r["status"] == "ok")
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
        "- Generation: `method_a_singleton_endpoints`, single pass. `full` uses the **placeholder** structured "
        "prior (a deterministic heuristic, not a trained or corpus-backed model).",
        "- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, or path modes. "
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
    lines += [f"- {item}" for item in DEFERRED] + [""]
    return "\n".join(lines)
