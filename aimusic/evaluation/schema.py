"""Versioned, machine-readable schemas for the MUS-28 evaluation suite.

Everything an evaluation run needs to be reproduced is carried explicitly in
:class:`RunSpec`; nothing is read from an undocumented default.  A finished run
is a plain JSON-able ``dict`` (see :func:`new_result`) so result files can be
consumed without importing this package.

The evaluation layer is strictly *downstream* of generation: it only calls the
public planning/scoring APIs and never changes generation behavior.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

CONFIG_SCHEMA_VERSION = "mus28-config/1"
RESULT_SCHEMA_VERSION = "mus28-result/1"
AGGREGATE_SCHEMA_VERSION = "mus28-aggregate/1"
MANIFEST_SCHEMA_VERSION = "mus28-manifest/1"

# What the generator actually does.  Recorded in every result so that a singleton
# run is never mistaken for a true distribution-to-distribution Method A (MUS-15)
# run, nor either for Method B (MUS-18) or section-wise (MUS-16).  The generation
# method is derived from ``endpoint_mode``; the two can never disagree.
ENDPOINT_MODES = ("singleton", "distribution")
GENERATION_METHODS = {
    "singleton": "method_a_singleton_endpoints",
    "distribution": "method_a_distribution_endpoints",
}
GENERATION_METHOD = GENERATION_METHODS["singleton"]  # M1 default, kept for compatibility
SECTIONING = "single_pass"

# Mirrors ``EndpointFallbackPolicy`` values (MUS-15); validated by the runner against the enum.
ENDPOINT_POLICIES = (
    "strict",
    "drop_unreachable_only",
    "drop_and_renormalize",
    "partial_trim_and_renormalize",
    "conditional_drop_then_trim",
)
DEFAULT_ENDPOINT_POLICY = "drop_unreachable_only"  # MethodARunConfig default
DEFAULT_ENDPOINT_MARGINAL_TOLERANCE = 1e-6
BENCHMARK_TYPES = ("evaluation", "long_form_benchmark")

PATH_MODES = ("map", "sample")
BRIDGE_MODES = ("sb", "uniform")
PRIOR_MODES = (
    "null",                   # NullPrior: prior intentionally off
    "placeholder_neutral",    # NeuralPrior placeholder, constant score
    "placeholder_structured", # NeuralPrior placeholder, heuristic score (NOT trained)
    "trained_neural",         # real trained prior; reports failure if it cannot load
)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_CRASHED = "crashed"
TERMINAL_STATUSES = (STATUS_OK, STATUS_FAILED, STATUS_TIMEOUT, STATUS_CRASHED)


@dataclass(frozen=True)
class RunSpec:
    """Complete, explicit description of one evaluation run."""

    condition: str
    horizon: int
    seed: int
    edo: int
    path_mode: str
    prior_mode: str
    lambda_data: float
    lambda_gttm: float
    bridge_mode: str
    k_max: int
    d_max: int
    proposal_budget: int
    prior_guided_proposals: bool
    max_iterations: int
    tolerance: float
    temperature: float
    prior_bundle: Optional[str] = None
    # Derived from ``endpoint_mode`` when left empty; an explicit value must agree with it.
    generation_method: str = ""
    sectioning: str = SECTIONING
    config_name: str = ""
    config_hash: str = ""
    # MUS-15 endpoint evaluation.  Singleton is the M1 behaviour and the default.
    endpoint_mode: str = "singleton"
    endpoint_policy: str = DEFAULT_ENDPOINT_POLICY
    endpoint_marginal_tolerance: float = DEFAULT_ENDPOINT_MARGINAL_TOLERANCE
    # Extra sampled paths drawn from the one solved bridge to measure calibration (0 = none).
    calibration_samples: int = 0

    def __post_init__(self) -> None:
        if self.endpoint_mode not in ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {ENDPOINT_MODES}.")
        if self.endpoint_policy not in ENDPOINT_POLICIES:
            raise ValueError(f"endpoint_policy must be one of {ENDPOINT_POLICIES}.")
        if not self.endpoint_marginal_tolerance > 0:
            raise ValueError("endpoint_marginal_tolerance must be > 0.")
        if self.calibration_samples < 0:
            raise ValueError("calibration_samples must be >= 0.")
        expected = GENERATION_METHODS[self.endpoint_mode]
        if not self.generation_method:
            object.__setattr__(self, "generation_method", expected)
        elif self.generation_method != expected:
            raise ValueError(
                f"generation_method '{self.generation_method}' contradicts endpoint_mode "
                f"'{self.endpoint_mode}' (expected '{expected}')."
            )
        if self.bridge_mode == "uniform" and self.endpoint_mode == "distribution":
            raise ValueError(
                "The no-SB baseline has no endpoint-marginal bridge, so it cannot be evaluated "
                "in distribution endpoint mode."
            )
        if self.calibration_samples and self.endpoint_mode != "distribution":
            raise ValueError("calibration_samples requires endpoint_mode='distribution'.")
        if self.path_mode not in PATH_MODES:
            raise ValueError(f"path_mode must be one of {PATH_MODES}.")
        if self.bridge_mode not in BRIDGE_MODES:
            raise ValueError(f"bridge_mode must be one of {BRIDGE_MODES}.")
        if self.prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}.")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self.bridge_mode == "uniform" and self.path_mode != "sample":
            raise ValueError("The uniform (no-SB) bridge has no meaningful MAP path; use path_mode='sample'.")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def run_id(self) -> str:
        """Stable id: readable prefix plus a hash of every knob that affects the run."""
        knobs = self.to_dict()
        # The config file name/hash do not change what a run computes, so the same
        # run is recognised (and resumable) across configs that share it.
        knobs.pop("config_name")
        knobs.pop("config_hash")
        if self.endpoint_mode == "singleton":
            # Legacy (M1) runs never had the endpoint knobs; dropping them keeps old run ids valid.
            for key in ("endpoint_mode", "endpoint_policy", "endpoint_marginal_tolerance", "calibration_samples"):
                knobs.pop(key)
        digest = canonical_hash(knobs)[:8]
        return (
            f"{self.condition}-h{self.horizon}-e{self.edo}-s{self.seed}-{self.path_mode}"
            f"-{digest}"
        )

    @property
    def group_key(self) -> str:
        """Key identifying a condition cell; seeds are replicates *within* a cell."""
        return (
            f"{self.condition}|h={self.horizon}|edo={self.edo}|{self.path_mode}"
            f"|k={self.k_max}|d={self.d_max}|pb={self.proposal_budget}"
            f"|ep={self.endpoint_mode}"
        )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def new_result(spec: RunSpec) -> Dict[str, Any]:
    """Skeleton result; the evaluator fills in the sections it can."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": spec.run_id,
        "status": STATUS_FAILED,
        "spec": spec.to_dict(),
        "reproducibility": {},
        "prior": {},
        "runtime": {},
        "failure": None,
        "metrics": {
            "graph": {},
            "sb": {},
            "path": {},
            "tension": {},
            "gttm": {},
            # MUS-15 endpoint metrics; filled only in distribution mode.  A singleton run
            # records an explicit "not_applicable" marker instead of zeros.
            "endpoint": {},
        },
        "artifacts": {"path_sha256": None},
    }
