"""One-run evaluator for the MUS-28 evaluation suite.

:func:`evaluate_run` executes a single :class:`RunSpec` through the *public*
generation API (``run_method_a``) and turns the existing graph / SB / tension /
GTTM diagnostics into a flat, machine-readable result.  It reimplements none of
the graph, SB, prior or scoring logic.

A failed run is a result, not an exception: every error is captured as a
structured ``failure`` record so one bad cell never aborts the matrix.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import platform
import resource
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from aimusic.evaluation.schema import (
    STATUS_FAILED,
    STATUS_OK,
    RunSpec,
    canonical_hash,
    new_result,
)

PriorBuild = Tuple[Any, Dict[str, Any]]


class PriorUnavailable(RuntimeError):
    """A prior that was explicitly requested could not be provided."""


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB, macOS reports bytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def _git_info() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    # Hash of the code that can affect results, so results from an uncommitted tree are
    # still attributable to an exact source state.
    digest = hashlib.sha256()
    for src in sorted((root / "aimusic").rglob("*.py")):
        digest.update(str(src.relative_to(root)).encode())
        digest.update(src.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def reproducibility_info() -> Dict[str, Any]:
    import numpy

    import aimusic

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "aimusic_version": getattr(aimusic, "__version__", "0.1.0"),
    }
    info.update(_git_info())
    return info


def build_prior(spec: RunSpec) -> PriorBuild:
    """Return ``(prior, info)``; never silently substitutes an unavailable prior."""
    from aimusic.core.config import NeuralPriorConfig, PlaceholderPriorMode
    from aimusic.scoring.priors import NeuralPrior, NullPrior

    if spec.prior_mode == "null":
        return NullPrior(), {"mode": "null", "status": "off_intentional", "trained": False}
    if spec.prior_mode in ("placeholder_neutral", "placeholder_structured"):
        placeholder = (
            PlaceholderPriorMode.NEUTRAL
            if spec.prior_mode == "placeholder_neutral"
            else PlaceholderPriorMode.STRUCTURED
        )
        config = NeuralPriorConfig(placeholder_mode=placeholder)
        status = "neutral_algorithmic" if spec.prior_mode == "placeholder_neutral" else "placeholder_not_trained"
        return NeuralPrior(config), {
            "mode": spec.prior_mode,
            "status": status,
            "trained": False,
            "model_family": config.model_family,
            "model_version": config.model_version,
            "placeholder_mode": placeholder.value,
        }
    # trained_neural
    if not spec.prior_bundle:
        raise PriorUnavailable("trained_neural requested but no prior_bundle is configured.")
    try:
        from aimusic.ml.inference import load_trained_neural_prior
    except ImportError as exc:
        raise PriorUnavailable(f"trained_neural requested but ML dependencies are missing: {exc}") from exc
    try:
        prior = load_trained_neural_prior(spec.prior_bundle)
    except Exception as exc:  # noqa: BLE001 - reported verbatim as a structured failure
        raise PriorUnavailable(f"trained prior bundle failed to load: {exc}") from exc
    return prior, {"mode": "trained_neural", "status": "loaded", "trained": True, "bundle": spec.prior_bundle}


def _ratio(num: float, den: float) -> Optional[float]:
    return float(num) / float(den) if den else None


def graph_metrics(graph: Any) -> Dict[str, Any]:
    """Graph-support metrics computed purely from existing REQ-13 diagnostics."""
    layers = graph.diagnostics.layer_diagnostics
    widths = [int(w) for w in graph.diagnostics.layer_sizes]
    raw = sum(ld.raw_candidate_count for ld in layers)
    legal = sum(ld.legal_candidate_count for ld in layers)
    unique = sum(ld.unique_candidate_count for ld in layers)
    scored = sum(ld.scored_candidate_count for ld in layers)
    kept = sum(ld.kept_candidate_count for ld in layers)
    raw_edges = sum(ld.raw_edge_count for ld in layers)
    kept_edges = sum(ld.kept_edge_count for ld in layers)
    reasons: Counter[str] = Counter()
    for ld in layers:
        reasons.update(f"pruned:{p.reason}" for p in ld.pruned_states)
        reasons.update(f"rejected:{r.reason}" for r in ld.rejected_proposals)
    inactive = graph.inactive_feature_names()
    return {
        "layer_count": len(widths),
        "layer_width_mean": sum(widths) / len(widths),
        "layer_width_min": min(widths),
        "layer_width_max": max(widths),
        "proposed_count": raw,
        "legal_count": legal,
        "unique_count": unique,
        "scored_count": scored,
        "retained_count": kept,
        "raw_edge_count": raw_edges,
        "retained_edge_count": kept_edges,
        "outdegree_pruned_count": sum(ld.outdegree_pruned_count for ld in layers),
        "legal_fraction": _ratio(legal, raw),
        "unique_fraction": _ratio(unique, legal),
        "scored_fraction": _ratio(scored, legal),
        "retention_fraction": _ratio(kept_edges, raw_edges),
        "prune_reason_counts": dict(sorted(reasons.items())),
        "inactive_feature_count": len(inactive),
        "inactive_features": list(inactive),
    }


def sb_metrics(solution: Any, *, used: bool) -> Dict[str, Any]:
    from aimusic.core.diagnostics import SBDiagnostics

    diag = SBDiagnostics.from_solution(solution)
    return {
        "used_for_path": used,
        "iterations": diag.iterations_run,
        "converged": diag.converged,
        "final_max_delta": diag.final_max_delta,
        "effective_entropy": diag.effective_entropy,
        "zero_degree_pruned_nodes": diag.pruned_nodes,
    }


def path_metrics(path: Sequence[Any], plan: Any, horizon: int) -> Dict[str, Any]:
    """Cheap structural sanity statistics; deliberately NOT musical-quality claims."""
    states = list(path)
    body = states[:-1]
    n = max(len(body), 1)
    changes = lambda attr: sum(  # noqa: E731
        1 for a, b in zip(body, body[1:]) if getattr(a, attr) != getattr(b, attr)
    )
    return {
        "path_length": len(states),
        "expected_length": horizon + 1,
        "length_ok": len(states) == horizon + 1,
        "reaches_start": states[0] == plan.endpoints.start_choice.state,
        "reaches_end": states[-1] == plan.endpoints.end_choice.state,
        "distinct_state_fraction": len(set(body)) / n,
        "distinct_chords": len({s.chord_id for s in body}),
        "distinct_keys": len({s.key_id for s in body}),
        "chord_change_rate": changes("chord_id") / n,
        "key_change_rate": changes("key_id") / n,
        "role_change_rate": changes("role_id") / n,
        "groove_change_rate": changes("groove_id") / n,
        "boundary_count": sum(1 for s in body if s.boundary_lvl > 0),
    }


def tension_metrics(path: Sequence[Any], plan: Any, edo: int) -> Dict[str, Any]:
    from aimusic.scoring.tension import (
        compare_tension_curves,
        realized_tension_curve,
        target_tension_curve,
    )

    sections = plan.endpoints.sections
    decoded = list(path)[:-1]  # same convention as the generate CLI
    realized = realized_tension_curve(decoded, plan.vocabularies, edo)
    target = target_tension_curve(sections)
    if not target:
        return {}
    report = compare_tension_curves(target, realized, sections)
    values = [v for _, v in realized]
    return {
        "mean_absolute_error": report.mean_absolute_error,
        "max_absolute_error": report.max_absolute_error,
        "shape_correlation": report.shape_correlation,
        "peak_timing_offset": report.peak_timing_offset,
        "realized_mean": sum(values) / len(values),
        "realized_min": min(values),
        "realized_max": max(values),
        # Per-beat curves (index = beat) so dashboards can plot target vs realized.  Lists are
        # not numeric leaves, so ``stats.flatten_metrics`` ignores them and aggregates are unchanged.
        "target_curve": [round(float(v), 6) for _, v in target],
        "realized_curve": [round(float(v), 6) for v in values],
    }


def gttm_path_metrics(edge_diags: Sequence[Any]) -> Dict[str, Any]:
    """GTTM activity along the chosen path.

    ``gttm_score`` is computed from the per-family feature weights, *not* from
    ``lambda_gttm``, so it stays comparable between ``full`` and ``no_gttm``.
    """
    if not edge_diags:
        return {}
    n = len(edge_diags)
    return {
        "mean_gttm_score": sum(d.gttm_score for d in edge_diags) / n,
        "mean_data_logp": sum(d.data_logp for d in edge_diags) / n,
        "mean_final_log_weight": sum(d.final_log_weight for d in edge_diags) / n,
        "edge_count": n,
    }


def _path_sha256(path: Sequence[Any]) -> str:
    return canonical_hash([dataclasses.astuple(s) for s in path])


def _failure(exc: BaseException, stage: str, kind: Optional[str] = None) -> Dict[str, Any]:
    return {
        "kind": kind or type(exc).__name__,
        "stage": stage,
        "message": str(exc)[:500],
        "traceback_tail": "".join(traceback.format_exception(exc)[-3:])[-1500:],
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def evaluate_run(spec: RunSpec, *, trace_python_memory: bool = False) -> Dict[str, Any]:
    """Evaluate one run.  Never raises for run-level problems; returns a result dict."""
    logging.disable(logging.CRITICAL)  # generation logs are per-edge noise at benchmark scale
    result = new_result(spec)
    result["reproducibility"] = reproducibility_info()
    stage = "setup"
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    rss_before = _peak_rss_mb()
    if trace_python_memory:
        import tracemalloc

        tracemalloc.start()
    try:
        from aimusic.core.config import PriorWeights, SBConfig
        from aimusic.core.rng import RNGKey, derive_named_key
        from aimusic.planning.plans import MethodARunConfig, run_method_a
        from aimusic.evaluation.baselines import reachable_uniform_walk

        stage = "prior"
        prior, prior_info = build_prior(spec)
        result["prior"] = prior_info

        stage = "generate"
        sb_config = SBConfig(
            horizon_t=spec.horizon,
            max_iterations=spec.max_iterations,
            tolerance=spec.tolerance,
            temperature=spec.temperature,
            k_max=spec.k_max,
            d_max=spec.d_max,
            proposal_budget=spec.proposal_budget,
            prior_guided_proposals=spec.prior_guided_proposals,
        )
        run_config = MethodARunConfig(
            total_beats=spec.horizon,
            seed=spec.seed,
            use_sampling=spec.path_mode == "sample",
            prior_weights=PriorWeights(lambda_data=spec.lambda_data, lambda_gttm=spec.lambda_gttm),
            sb_config=sb_config,
            edo=spec.edo,
        )
        plan, _ = run_method_a(run_config, key=RNGKey(seed=spec.seed), prior=prior)
        gen_wall = time.perf_counter() - start_wall

        stage = "path"
        path = plan.path
        edge_diags = plan.path_edge_diagnostics
        if spec.bridge_mode == "uniform":
            # no-SB ablation: identical graph, but a uniform end-reachable random walk
            # replaces the Schrodinger-bridge solution (edge weights are ignored).
            key = derive_named_key(RNGKey(seed=spec.seed), "eval.no_sb_walk")
            path = reachable_uniform_walk(plan.graph, key)
            edge_diags = plan.graph.diagnostics_for_path(path)

        stage = "metrics"
        metrics = result["metrics"]
        metrics["graph"] = graph_metrics(plan.graph)
        metrics["sb"] = sb_metrics(plan.sb_solution, used=spec.bridge_mode == "sb")
        metrics["path"] = path_metrics(path, plan, spec.horizon)
        if spec.path_mode == "map" and spec.bridge_mode == "sb":
            score = plan.path_score
            metrics["path"]["map_path_score"] = score
            metrics["path"]["map_path_score_finite"] = _finite(score)
        metrics["tension"] = tension_metrics(path, plan, spec.edo)
        metrics["gttm"] = gttm_path_metrics(edge_diags)
        result["artifacts"]["path_sha256"] = _path_sha256(path)
        result["runtime"]["generate_wall_s"] = gen_wall
        result["status"] = STATUS_OK
    except PriorUnavailable as exc:
        result["failure"] = _failure(exc, stage, kind="prior_unavailable")
        result["prior"] = {"mode": spec.prior_mode, "status": "requested_unavailable", "trained": False}
        result["status"] = STATUS_FAILED
    except Exception as exc:  # noqa: BLE001 - a failed run is a benchmark result
        result["failure"] = _failure(exc, stage)
        result["status"] = STATUS_FAILED
    finally:
        runtime = result["runtime"]
        runtime["wall_s"] = time.perf_counter() - start_wall
        runtime["cpu_s"] = time.process_time() - start_cpu
        runtime["peak_rss_mb"] = _peak_rss_mb()
        runtime["baseline_rss_mb"] = rss_before
        if trace_python_memory:
            import tracemalloc

            runtime["python_peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
    return result
