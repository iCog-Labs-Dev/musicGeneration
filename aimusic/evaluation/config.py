"""Versioned benchmark configuration for the MUS-28 evaluation suite.

A benchmark config is a JSON file (see ``config/benchmarks/``).  It lists the
axes of the matrix; :func:`expand_matrix` turns it into explicit
:class:`~aimusic.evaluation.schema.RunSpec` objects, one per run.

Ablation conditions change **one factor at a time** relative to ``full``:

=========  ======================================================  ==========
condition  difference from ``full``                                factor
=========  ======================================================  ==========
full       GTTM on, placeholder prior on, SB bridge                -
no_gttm    ``lambda_gttm = 0`` (graph legality unchanged)          GTTM
no_prior   ``NullPrior`` instead of the placeholder prior          prior
no_sb      same graph, uniform edge probabilities instead of SB    SB
=========  ======================================================  ==========

``full`` uses the *placeholder* structured prior, which is a deterministic
heuristic and NOT a trained or corpus-backed model.  Results record this via
``prior.status`` so nobody mistakes it for a trained-prior result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from aimusic.evaluation.schema import (
    CONFIG_SCHEMA_VERSION,
    PATH_MODES,
    RunSpec,
    canonical_hash,
)


@dataclass(frozen=True)
class ConditionDef:
    """How one named condition differs from the full system."""

    prior_mode: str
    lambda_data: float
    lambda_gttm: float
    bridge_mode: str
    factor: str


CONDITIONS: Dict[str, ConditionDef] = {
    "full": ConditionDef("placeholder_structured", 1.0, 1.0, "sb", "none"),
    "no_gttm": ConditionDef("placeholder_structured", 1.0, 0.0, "sb", "gttm"),
    "no_prior": ConditionDef("null", 1.0, 1.0, "sb", "prior"),
    "no_sb": ConditionDef("placeholder_structured", 1.0, 1.0, "uniform", "sb"),
    # Reserved: fails with an explicit "prior_unavailable" result unless a
    # trained bundle is configured and loadable.  Never silently substituted.
    "trained_prior": ConditionDef("trained_neural", 1.0, 1.0, "sb", "prior"),
}


@dataclass(frozen=True)
class GraphPoint:
    k_max: int
    d_max: int
    proposal_budget: int


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    config_version: str
    description: str
    horizons: Tuple[int, ...]
    seeds: Tuple[int, ...]
    edos: Tuple[int, ...]
    path_modes: Tuple[str, ...]
    conditions: Tuple[str, ...]
    graph_grid: Tuple[GraphPoint, ...]
    prior_guided_proposals: bool
    max_iterations: int
    tolerance: float
    temperature: float
    timeout_s: float
    prior_bundle: Optional[str]
    raw: Mapping[str, Any]

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.raw)


def _require(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"benchmark config is missing required key '{key}'.")
    return raw[key]


def parse_config(raw: Mapping[str, Any]) -> BenchmarkConfig:
    """Validate a raw config mapping.  Every knob must be explicit (no hidden defaults)."""
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be '{CONFIG_SCHEMA_VERSION}'.")
    horizons = tuple(int(h) for h in _require(raw, "horizons"))
    seeds = tuple(int(s) for s in _require(raw, "seeds"))
    edos = tuple(int(e) for e in _require(raw, "edos"))
    path_modes = tuple(str(m) for m in _require(raw, "path_modes"))
    conditions = tuple(str(c) for c in _require(raw, "conditions"))
    if not horizons or not seeds or not edos or not path_modes or not conditions:
        raise ValueError("horizons, seeds, edos, path_modes and conditions must be non-empty.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique.")
    for mode in path_modes:
        if mode not in PATH_MODES:
            raise ValueError(f"unknown path_mode '{mode}'.")
    for cond in conditions:
        if cond not in CONDITIONS:
            raise ValueError(f"unknown condition '{cond}'; known: {sorted(CONDITIONS)}.")
    grid = tuple(
        GraphPoint(int(p["k_max"]), int(p["d_max"]), int(p["proposal_budget"]))
        for p in _require(raw, "graph_grid")
    )
    if not grid:
        raise ValueError("graph_grid must be non-empty.")
    sb = _require(raw, "sb")
    return BenchmarkConfig(
        name=str(_require(raw, "name")),
        config_version=str(_require(raw, "config_version")),
        description=str(raw.get("description", "")),
        horizons=horizons,
        seeds=seeds,
        edos=edos,
        path_modes=path_modes,
        conditions=conditions,
        graph_grid=grid,
        prior_guided_proposals=bool(_require(raw, "prior_guided_proposals")),
        max_iterations=int(sb["max_iterations"]),
        tolerance=float(sb["tolerance"]),
        temperature=float(sb["temperature"]),
        timeout_s=float(_require(raw, "timeout_s")),
        prior_bundle=raw.get("prior_bundle"),
        raw=dict(raw),
    )


def load_config(path: Path | str) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_config(json.load(handle))


def expand_matrix(config: BenchmarkConfig) -> List[RunSpec]:
    """Expand a config into explicit run specs in a deterministic order.

    ``(no_sb, map)`` cells are skipped: a uniform bridge has no meaningful MAP
    path, so the no-SB ablation is only defined for sampled paths.
    """
    specs: List[RunSpec] = []
    config_hash = config.config_hash
    for point in config.graph_grid:
        for horizon in config.horizons:
            for edo in config.edos:
                for path_mode in config.path_modes:
                    for condition in config.conditions:
                        cdef = CONDITIONS[condition]
                        if cdef.bridge_mode == "uniform" and path_mode != "sample":
                            continue
                        for seed in config.seeds:
                            specs.append(
                                RunSpec(
                                    condition=condition,
                                    horizon=horizon,
                                    seed=seed,
                                    edo=edo,
                                    path_mode=path_mode,
                                    prior_mode=cdef.prior_mode,
                                    lambda_data=cdef.lambda_data,
                                    lambda_gttm=cdef.lambda_gttm,
                                    bridge_mode=cdef.bridge_mode,
                                    k_max=point.k_max,
                                    d_max=point.d_max,
                                    proposal_budget=point.proposal_budget,
                                    prior_guided_proposals=config.prior_guided_proposals,
                                    max_iterations=config.max_iterations,
                                    tolerance=config.tolerance,
                                    temperature=config.temperature,
                                    prior_bundle=config.prior_bundle,
                                    config_name=config.name,
                                    config_hash=config_hash,
                                )
                            )
    ids = [s.run_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark config expands to duplicate run ids.")
    return specs


def select_specs(
    specs: Sequence[RunSpec],
    *,
    conditions: Optional[Sequence[str]] = None,
    horizons: Optional[Sequence[int]] = None,
    edos: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[RunSpec]:
    """Subset of the matrix; the manifest still lists every expected run."""
    out = []
    for s in specs:
        if conditions and s.condition not in conditions:
            continue
        if horizons and s.horizon not in horizons:
            continue
        if edos and s.edo not in edos:
            continue
        if seeds and s.seed not in seeds:
            continue
        out.append(s)
    return out
