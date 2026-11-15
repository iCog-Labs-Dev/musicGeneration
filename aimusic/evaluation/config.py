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
    DEFAULT_ENDPOINT_MARGINAL_TOLERANCE,
    DEFAULT_ENDPOINT_POLICY,
    ENDPOINT_MODES,
    ENDPOINT_POLICIES,
    PATH_MODES,
    SECTIONING_MODES,
    SECTIONING_SECTION_WISE,
    SECTIONING_SINGLE_PASS,
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
    # MUS-15 endpoint evaluation.  Optional in the file (absent => singleton, the M1 behaviour),
    # so shipped M1 configs keep their config_hash.
    endpoint_mode: str = "singleton"
    endpoint_policy: str = DEFAULT_ENDPOINT_POLICY
    endpoint_marginal_tolerance: float = DEFAULT_ENDPOINT_MARGINAL_TOLERANCE
    calibration_samples: int = 0
    # MUS-16 sectioning axis.  Optional in the file (absent => ("single_pass",), the M1 behaviour), so
    # shipped M1 configs keep their config_hash.  The raw file is what is hashed.
    sectioning: Tuple[str, ...] = (SECTIONING_SINGLE_PASS,)
    section_names: Tuple[str, ...] = ()
    stitch_max_cost: Optional[float] = None
    allowed_meters: Tuple[str, ...] = ()
    groove_families: Tuple[str, ...] = ()

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
    endpoint_mode = str(raw.get("endpoint_mode", "singleton"))
    if endpoint_mode not in ENDPOINT_MODES:
        raise ValueError(f"endpoint_mode must be one of {ENDPOINT_MODES}, got '{endpoint_mode}'.")
    endpoint_policy = str(raw.get("endpoint_policy", DEFAULT_ENDPOINT_POLICY))
    if endpoint_policy not in ENDPOINT_POLICIES:
        raise ValueError(f"endpoint_policy must be one of {ENDPOINT_POLICIES}, got '{endpoint_policy}'.")
    marginal_tol = float(raw.get("endpoint_marginal_tolerance", DEFAULT_ENDPOINT_MARGINAL_TOLERANCE))
    if not marginal_tol > 0:
        raise ValueError("endpoint_marginal_tolerance must be > 0.")
    calibration_samples = int(raw.get("calibration_samples", 0))
    if calibration_samples < 0:
        raise ValueError("calibration_samples must be >= 0.")
    if endpoint_mode == "singleton":
        for key in ("endpoint_policy", "endpoint_marginal_tolerance", "calibration_samples"):
            if key in raw:
                raise ValueError(f"'{key}' only applies to endpoint_mode='distribution'.")
    if endpoint_mode == "distribution" and "no_sb" in conditions:
        raise ValueError(
            "condition 'no_sb' cannot be evaluated with endpoint_mode='distribution': the "
            "uniform baseline has no endpoint-marginal bridge.  Use a separate singleton config."
        )
    sectioning, section_names, stitch_max_cost = _parse_sectioning(raw, horizons, conditions, calibration_samples)
    allowed_meters, groove_families = _parse_style(raw.get("style"))
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
        endpoint_mode=endpoint_mode,
        endpoint_policy=endpoint_policy,
        endpoint_marginal_tolerance=marginal_tol,
        calibration_samples=calibration_samples,
        sectioning=sectioning,
        section_names=section_names,
        stitch_max_cost=stitch_max_cost,
        allowed_meters=allowed_meters,
        groove_families=groove_families,
    )


def _parse_style(style: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Optional ``style`` block restricting meters / groove families (validated by ``StyleConfig``)."""
    if style is None:
        return (), ()
    if not isinstance(style, Mapping) or set(style) - {"allowed_meters", "groove_families"}:
        raise ValueError("style only supports the keys 'allowed_meters' and 'groove_families'.")
    out: List[Tuple[str, ...]] = []
    for key in ("allowed_meters", "groove_families"):
        value = style.get(key, ())
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(f"style.{key} must be a list of non-empty strings.")
        if not all(isinstance(v, str) and v.strip() for v in value):
            raise ValueError(f"style.{key} entries must be non-empty strings.")
        out.append(tuple(str(v) for v in value))
    from aimusic.core.config import StyleConfig

    # Raises ValueError for an unusable style, at parse time rather than inside a worker.
    StyleConfig(
        **({"allowed_meters": out[0]} if out[0] else {}),  # type: ignore[arg-type]
        **({"groove_families": out[1]} if out[1] else {}),  # type: ignore[arg-type]
    )
    return out[0], out[1]


def _parse_sectioning(
    raw: Mapping[str, Any],
    horizons: Sequence[int],
    conditions: Sequence[str],
    calibration_samples: int,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Optional[float]]:
    """Validate the MUS-16 sectioning axis, section plan and stitching tolerance."""
    value = raw.get("sectioning", SECTIONING_SINGLE_PASS)
    modes = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, (list, tuple)) else None
    if not modes or not all(isinstance(m, str) for m in modes):
        raise ValueError(
            f"sectioning must be one of {SECTIONING_MODES} or a non-empty list of them, got {value!r}."
        )
    for mode in modes:
        if mode not in SECTIONING_MODES:
            raise ValueError(f"unknown sectioning '{mode}'; known: {list(SECTIONING_MODES)}.")
    if len(set(modes)) != len(modes):
        raise ValueError("sectioning must not repeat a mode.")

    names_raw = raw.get("section_names", ())
    if isinstance(names_raw, str) or not isinstance(names_raw, (list, tuple)):
        raise ValueError("section_names must be a list of non-empty strings.")
    if not all(isinstance(n, str) and n.strip() for n in names_raw):
        raise ValueError("section_names entries must be non-empty strings.")
    names = tuple(str(n) for n in names_raw)
    if len(set(names)) != len(names):
        raise ValueError(f"section_names must be unique, got {list(names)}.")

    stitching = raw.get("stitching")
    max_cost: Optional[float] = None
    if stitching is not None:
        if not isinstance(stitching, Mapping) or set(stitching) - {"max_cost"}:
            raise ValueError("stitching only supports the key 'max_cost'.")
        if "max_cost" in stitching:
            max_cost = float(stitching["max_cost"])
            if not max_cost >= 0:
                raise ValueError("stitching.max_cost must be >= 0.")
        if SECTIONING_SECTION_WISE not in modes:
            raise ValueError("'stitching' only applies when sectioning includes 'section_wise'.")

    if SECTIONING_SECTION_WISE in modes:
        if len(names) < 2:
            raise ValueError("sectioning 'section_wise' requires section_names with at least two sections.")
        too_short = [h for h in horizons if h < len(names)]
        if too_short:
            raise ValueError(
                f"horizons {too_short} are shorter than the {len(names)} section_names {list(names)}; "
                "every section needs at least one beat."
            )
        if "no_sb" in conditions:
            raise ValueError(
                "condition 'no_sb' cannot be evaluated with sectioning 'section_wise': the uniform "
                "baseline has no per-section bridge.  Use a separate single_pass config."
            )
        if calibration_samples:
            raise ValueError("calibration_samples is not supported with sectioning 'section_wise'.")
    return modes, names, max_cost


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
                        for sectioning in config.sectioning:
                            section_wise = sectioning == SECTIONING_SECTION_WISE
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
                                        endpoint_mode=config.endpoint_mode,
                                        endpoint_policy=config.endpoint_policy,
                                        endpoint_marginal_tolerance=config.endpoint_marginal_tolerance,
                                        calibration_samples=config.calibration_samples,
                                        sectioning=sectioning,
                                        # Matched pairs: both modes carry the same authored section plan.
                                        section_names=config.section_names,
                                        stitch_max_cost=config.stitch_max_cost if section_wise else None,
                                        allowed_meters=config.allowed_meters,
                                        groove_families=config.groove_families,
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
    sectioning: Optional[Sequence[str]] = None,
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
        if sectioning and s.sectioning not in sectioning:
            continue
        out.append(s)
    return out
