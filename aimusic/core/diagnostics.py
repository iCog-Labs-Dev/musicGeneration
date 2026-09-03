from __future__ import annotations

import dataclasses
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0.0"
APPLICATION_VERSION = "0.1.0"


class ManifestValidationError(ValueError):
    """Raised when manifest data does not match its declared schema."""


def _error(path: str, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _error(path, f"must be an object, got {type(value).__name__}")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, f"must be an array, got {type(value).__name__}")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _error(path, f"must be a string, got {type(value).__name__}")
    if not value.strip():
        raise _error(path, "must not be empty")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, f"must be a boolean, got {type(value).__name__}")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(path, f"must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise _error(path, f"must be >= {minimum}")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, f"must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "must be finite")
    if minimum is not None and result < minimum:
        raise _error(path, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise _error(path, f"must be <= {maximum}")
    return result


def _optional_number(value: object, path: str) -> float | None:
    return None if value is None else _number(value, path)


def _required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise _error(path, f"missing field '{key}' (required)")
    return data[key]


def _keys(
    data: Mapping[str, Any],
    path: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    missing = [key for key in required if key not in data]
    if missing:
        raise _error(path, f"missing field(s): {', '.join(missing)} (required)")
    allowed = set(required).union(optional)
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise _error(path, f"unknown field(s): {', '.join(unknown)}")


def _string_count_map(value: object, path: str) -> Dict[str, int]:
    data = _mapping(value, path)
    result: Dict[str, int] = {}
    for key, count in data.items():
        label = _string(key, f"{path}.<key>")
        result[label] = _integer(count, f"{path}.{label}", minimum=0)
    return result


@dataclass(frozen=True)
class TimelineEvent:
    start_time: float
    end_time: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, path: str = "timeline_event") -> "TimelineEvent":
        data = _mapping(value, path)
        _keys(data, path, required=("start_time", "end_time", "label"))
        start = _number(data["start_time"], f"{path}.start_time", minimum=0.0)
        end = _number(data["end_time"], f"{path}.end_time", minimum=0.0)
        if end < start:
            raise _error(f"{path}.end_time", "must be >= start_time")
        return cls(start, end, _string(data["label"], f"{path}.label"))


@dataclass
class StructuralDiagnostics:
    key_timeline: List[TimelineEvent] = field(default_factory=list)
    chord_timeline: List[TimelineEvent] = field(default_factory=list)
    role_timeline: List[TimelineEvent] = field(default_factory=list)
    groove_timeline: List[TimelineEvent] = field(default_factory=list)
    boundaries: List[float] = field(default_factory=list)
    tension_curve: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_timeline": [event.to_dict() for event in self.key_timeline],
            "chord_timeline": [event.to_dict() for event in self.chord_timeline],
            "role_timeline": [event.to_dict() for event in self.role_timeline],
            "groove_timeline": [event.to_dict() for event in self.groove_timeline],
            "boundaries": list(self.boundaries),
            "tension_curve": [[time_value, tension] for time_value, tension in self.tension_curve],
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "structure") -> "StructuralDiagnostics":
        data = _mapping(value, path)
        names = ("key_timeline", "chord_timeline", "role_timeline", "groove_timeline")
        _keys(data, path, required=(*names, "boundaries", "tension_curve"))

        def timeline(name: str) -> List[TimelineEvent]:
            return [
                TimelineEvent.from_dict(item, path=f"{path}.{name}[{index}]")
                for index, item in enumerate(_list(data[name], f"{path}.{name}"))
            ]

        boundaries = [
            _number(item, f"{path}.boundaries[{index}]", minimum=0.0)
            for index, item in enumerate(_list(data["boundaries"], f"{path}.boundaries"))
        ]
        tension_curve: List[Tuple[float, float]] = []
        for index, item in enumerate(_list(data["tension_curve"], f"{path}.tension_curve")):
            pair = _list(item, f"{path}.tension_curve[{index}]")
            if len(pair) != 2:
                raise _error(f"{path}.tension_curve[{index}]", "must contain [time, tension]")
            tension_curve.append(
                (
                    _number(pair[0], f"{path}.tension_curve[{index}][0]", minimum=0.0),
                    _number(pair[1], f"{path}.tension_curve[{index}][1]", minimum=0.0, maximum=1.0),
                )
            )
        return cls(
            key_timeline=timeline("key_timeline"),
            chord_timeline=timeline("chord_timeline"),
            role_timeline=timeline("role_timeline"),
            groove_timeline=timeline("groove_timeline"),
            boundaries=boundaries,
            tension_curve=tension_curve,
        )


@dataclass(frozen=True)
class LayerGraphStats:
    time_index: int
    source_states: int
    proposed: int
    legal: int
    scored: int
    duplicate_proposals: int
    candidate_pruned: int
    scored_unique_states: int
    retained: int
    pruned: int
    scored_edges: int
    retained_edges: int
    pruned_edges: int

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, path: str) -> "LayerGraphStats":
        data = _mapping(value, path)
        fields = (
            "time_index", "source_states", "proposed", "legal", "scored",
            "duplicate_proposals", "candidate_pruned", "scored_unique_states",
            "retained", "pruned", "scored_edges", "retained_edges", "pruned_edges",
        )
        _keys(data, path, required=fields)
        result = cls(**{
            name: _integer(data[name], f"{path}.{name}", minimum=0)
            for name in fields
        })
        if result.legal > result.proposed:
            raise _error(f"{path}.legal", "must be <= proposed")
        removed = result.duplicate_proposals + result.candidate_pruned
        if result.legal - result.scored != removed:
            raise _error(
                path,
                "legal - scored must equal duplicate_proposals + candidate_pruned",
            )
        if result.scored_unique_states > result.scored:
            raise _error(f"{path}.scored_unique_states", "must be <= scored")
        if result.scored_edges != result.scored:
            raise _error(f"{path}.scored_edges", "must equal scored")
        if result.retained_edges + result.pruned_edges != result.scored_edges:
            raise _error(path, "retained_edges + pruned_edges must equal scored_edges")
        return result


@dataclass
class GraphDiagnosticsData:
    layer_sizes: List[int] = field(default_factory=list)
    per_layer_stats: List[LayerGraphStats] = field(default_factory=list)
    rejection_summary: Dict[str, int] = field(default_factory=dict)
    pruning_summary: Dict[str, int] = field(default_factory=dict)
    total_proposed: int = 0
    total_legal: int = 0
    total_scored: int = 0
    total_duplicate_proposals: int = 0
    total_candidate_pruned: int = 0
    total_scored_unique_states: int = 0
    total_retained: int = 0
    total_pruned: int = 0
    total_scored_edges: int = 0
    total_retained_edges: int = 0
    total_pruned_edges: int = 0
    total_prune_operations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_sizes": list(self.layer_sizes),
            "per_layer_stats": [stats.to_dict() for stats in self.per_layer_stats],
            "rejection_summary": dict(self.rejection_summary),
            "pruning_summary": dict(self.pruning_summary),
            "total_proposed": self.total_proposed,
            "total_legal": self.total_legal,
            "total_scored": self.total_scored,
            "total_duplicate_proposals": self.total_duplicate_proposals,
            "total_candidate_pruned": self.total_candidate_pruned,
            "total_scored_unique_states": self.total_scored_unique_states,
            "total_retained": self.total_retained,
            "total_pruned": self.total_pruned,
            "total_scored_edges": self.total_scored_edges,
            "total_retained_edges": self.total_retained_edges,
            "total_pruned_edges": self.total_pruned_edges,
            "total_prune_operations": self.total_prune_operations,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "graph_stats") -> "GraphDiagnosticsData":
        data = _mapping(value, path)
        totals = (
            "total_proposed", "total_legal", "total_scored",
            "total_duplicate_proposals", "total_candidate_pruned",
            "total_scored_unique_states", "total_retained", "total_pruned",
            "total_scored_edges", "total_retained_edges", "total_pruned_edges",
            "total_prune_operations",
        )
        _keys(data, path, required=(
            "layer_sizes", "per_layer_stats", "rejection_summary", "pruning_summary", *totals,
        ))
        layer_sizes = [
            _integer(item, f"{path}.layer_sizes[{index}]", minimum=0)
            for index, item in enumerate(_list(data["layer_sizes"], f"{path}.layer_sizes"))
        ]
        per_layer = [
            LayerGraphStats.from_dict(item, path=f"{path}.per_layer_stats[{index}]")
            for index, item in enumerate(_list(data["per_layer_stats"], f"{path}.per_layer_stats"))
        ]
        result = cls(
            layer_sizes=layer_sizes,
            per_layer_stats=per_layer,
            rejection_summary=_string_count_map(data["rejection_summary"], f"{path}.rejection_summary"),
            pruning_summary=_string_count_map(data["pruning_summary"], f"{path}.pruning_summary"),
            **{name: _integer(data[name], f"{path}.{name}", minimum=0) for name in totals},
        )
        pairs = (
            ("proposed", "total_proposed"), ("legal", "total_legal"),
            ("scored", "total_scored"),
            ("duplicate_proposals", "total_duplicate_proposals"),
            ("candidate_pruned", "total_candidate_pruned"),
            ("scored_unique_states", "total_scored_unique_states"),
            ("retained", "total_retained"), ("pruned", "total_pruned"),
            ("scored_edges", "total_scored_edges"),
            ("retained_edges", "total_retained_edges"),
            ("pruned_edges", "total_pruned_edges"),
        )
        for layer_field, total_field in pairs:
            expected = sum(getattr(item, layer_field) for item in per_layer)
            if getattr(result, total_field) != expected:
                raise _error(f"{path}.{total_field}", f"must equal per-layer sum {expected}")
        expected_rejections = result.total_proposed - result.total_scored
        if sum(result.rejection_summary.values()) != expected_rejections:
            raise _error(f"{path}.rejection_summary", f"counts must sum to {expected_rejections}")
        expected_prunes = (
            result.total_candidate_pruned
            + result.total_pruned
            + result.total_pruned_edges
        )
        if result.total_prune_operations != expected_prunes:
            raise _error(f"{path}.total_prune_operations", f"must equal {expected_prunes}")
        if sum(result.pruning_summary.values()) != expected_prunes:
            raise _error(f"{path}.pruning_summary", f"counts must sum to {expected_prunes}")
        return result


@dataclass(frozen=True)
class EndpointDiagnosticsData:
    original_pi0_support_size: int = 0
    original_piT_support_size: int = 0
    solver_pi0_support_size: int = 0
    solver_piT_support_size: int = 0
    unreachable_pi0_mass: float = 0.0
    unreachable_piT_mass: float = 0.0
    selected_start_probability: float = 0.0
    selected_end_probability: float = 0.0

    @property
    def unreachable_probability_mass(self) -> float:
        return self.unreachable_pi0_mass + self.unreachable_piT_mass

    def to_dict(self) -> Dict[str, Any]:
        return {**dataclasses.asdict(self), "unreachable_probability_mass": self.unreachable_probability_mass}

    @classmethod
    def from_dict(cls, value: object, *, path: str = "endpoint_stats") -> "EndpointDiagnosticsData":
        data = _mapping(value, path)
        supports = (
            "original_pi0_support_size", "original_piT_support_size",
            "solver_pi0_support_size", "solver_piT_support_size",
        )
        probabilities = (
            "unreachable_pi0_mass", "unreachable_piT_mass",
            "selected_start_probability", "selected_end_probability",
        )
        _keys(data, path, required=(*supports, *probabilities, "unreachable_probability_mass"))
        result = cls(
            original_pi0_support_size=_integer(
                data["original_pi0_support_size"],
                f"{path}.original_pi0_support_size",
                minimum=0,
            ),
            original_piT_support_size=_integer(
                data["original_piT_support_size"],
                f"{path}.original_piT_support_size",
                minimum=0,
            ),
            solver_pi0_support_size=_integer(
                data["solver_pi0_support_size"],
                f"{path}.solver_pi0_support_size",
                minimum=0,
            ),
            solver_piT_support_size=_integer(
                data["solver_piT_support_size"],
                f"{path}.solver_piT_support_size",
                minimum=0,
            ),
            unreachable_pi0_mass=_number(
                data["unreachable_pi0_mass"],
                f"{path}.unreachable_pi0_mass",
                minimum=0.0,
                maximum=1.0,
            ),
            unreachable_piT_mass=_number(
                data["unreachable_piT_mass"],
                f"{path}.unreachable_piT_mass",
                minimum=0.0,
                maximum=1.0,
            ),
            selected_start_probability=_number(
                data["selected_start_probability"],
                f"{path}.selected_start_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            selected_end_probability=_number(
                data["selected_end_probability"],
                f"{path}.selected_end_probability",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        total = _number(data["unreachable_probability_mass"], f"{path}.unreachable_probability_mass", minimum=0.0, maximum=2.0)
        if not math.isclose(total, result.unreachable_probability_mass, abs_tol=1e-12):
            raise _error(f"{path}.unreachable_probability_mass", "must equal both unreachable masses")
        return result


@dataclass(frozen=True)
class PathDiagnosticsData:
    path_mode: str = "map"
    path_score: Optional[float] = None
    path_state_count: int = 0
    path_transition_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, path: str = "path_stats") -> "PathDiagnosticsData":
        data = _mapping(value, path)
        _keys(data, path, required=("path_mode", "path_score", "path_state_count", "path_transition_count"))
        mode = _string(data["path_mode"], f"{path}.path_mode")
        if mode not in {"map", "sample", "unknown"}:
            raise _error(f"{path}.path_mode", "must be 'map', 'sample', or 'unknown'")
        states = _integer(data["path_state_count"], f"{path}.path_state_count", minimum=0)
        transitions = _integer(data["path_transition_count"], f"{path}.path_transition_count", minimum=0)
        if states and transitions != states - 1:
            raise _error(f"{path}.path_transition_count", "must equal path_state_count - 1")
        return cls(mode, _optional_number(data["path_score"], f"{path}.path_score"), states, transitions)


@dataclass
class SBDiagnostics:
    iterations_run: int = 0
    converged: bool = False
    final_max_delta: float = 0.0
    layer_sizes: List[int] = field(default_factory=list)
    disconnected_nodes: int = 0
    effective_entropy: float = 0.0
    residual_history: List[float] = field(default_factory=list)
    layer_entropies: List[float] = field(default_factory=list)

    @property
    def pruned_nodes(self) -> int:
        """Deprecated legacy alias; these are connectivity defects, not prunes."""
        return self.disconnected_nodes

    @classmethod
    def from_solution(cls, solution: Any) -> "SBDiagnostics":
        trace = solution.trace
        problem = solution.problem.diagnostics
        entropies: List[float] = []
        if solution.marginals is not None:
            for probabilities in solution.marginals.node_marginals_by_layer:
                entropies.append(float(-sum(
                    float(probability) * math.log(float(probability))
                    for probability in probabilities if probability > 0.0
                )))
        return cls(
            iterations_run=trace.iterations,
            converged=trace.converged,
            final_max_delta=trace.final_max_delta,
            layer_sizes=list(problem.layer_sizes),
            disconnected_nodes=problem.zero_outdegree_count + problem.zero_indegree_count,
            effective_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
            residual_history=list(trace.residual_history),
            layer_entropies=entropies,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, path: str = "sb_stats") -> "SBDiagnostics":
        data = _mapping(value, path)
        fields = (
            "iterations_run", "converged", "final_max_delta", "layer_sizes",
            "disconnected_nodes", "effective_entropy", "residual_history", "layer_entropies",
        )
        _keys(data, path, required=fields)
        result = cls(
            iterations_run=_integer(data["iterations_run"], f"{path}.iterations_run", minimum=0),
            converged=_boolean(data["converged"], f"{path}.converged"),
            final_max_delta=_number(data["final_max_delta"], f"{path}.final_max_delta", minimum=0.0),
            layer_sizes=[
                _integer(item, f"{path}.layer_sizes[{index}]", minimum=0)
                for index, item in enumerate(_list(data["layer_sizes"], f"{path}.layer_sizes"))
            ],
            disconnected_nodes=_integer(data["disconnected_nodes"], f"{path}.disconnected_nodes", minimum=0),
            effective_entropy=_number(data["effective_entropy"], f"{path}.effective_entropy", minimum=0.0),
            residual_history=[
                _number(item, f"{path}.residual_history[{index}]", minimum=0.0)
                for index, item in enumerate(_list(data["residual_history"], f"{path}.residual_history"))
            ],
            layer_entropies=[
                _number(item, f"{path}.layer_entropies[{index}]", minimum=0.0)
                for index, item in enumerate(_list(data["layer_entropies"], f"{path}.layer_entropies"))
            ],
        )
        if result.layer_entropies and len(result.layer_entropies) != len(result.layer_sizes):
            raise _error(f"{path}.layer_entropies", "must align with layer_sizes")
        if result.residual_history and len(result.residual_history) != result.iterations_run:
            raise _error(f"{path}.residual_history", "must contain one value per iteration")
        return result


@dataclass(frozen=True)
class RunManifest:
    seed: int
    config_dump: Dict[str, Any]
    structural_stats: StructuralDiagnostics = field(default_factory=StructuralDiagnostics)
    sb_stats: SBDiagnostics = field(default_factory=SBDiagnostics)
    graph_stats: GraphDiagnosticsData = field(default_factory=GraphDiagnosticsData)
    endpoint_stats: EndpointDiagnosticsData = field(default_factory=EndpointDiagnosticsData)
    path_stats: PathDiagnosticsData = field(default_factory=PathDiagnosticsData)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: str = APPLICATION_VERSION
    schema_version: str = SCHEMA_VERSION
    migration_source_version: Optional[str] = None
    migration_warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "version": self.version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "config": self.config_dump,
            "structure": self.structural_stats.to_dict(),
            "graph_stats": self.graph_stats.to_dict(),
            "endpoint_stats": self.endpoint_stats.to_dict(),
            "sb_stats": self.sb_stats.to_dict(),
            "path_stats": self.path_stats.to_dict(),
        }
        if self.migration_source_version is not None or self.migration_warnings:
            data["migration"] = {
                "source_schema_version": self.migration_source_version or "legacy-unversioned",
                "warnings": list(self.migration_warnings),
            }
        _manifest_from_v1(data)
        return data

    @classmethod
    def from_dict(cls, value: object) -> "RunManifest":
        data = _mapping(value, "manifest")
        return _manifest_from_v1(migrate_legacy_manifest(data) if "schema_version" not in data else data)


def _manifest_from_v1(value: object) -> RunManifest:
    data = _mapping(value, "manifest")
    required = (
        "schema_version", "version", "run_id", "timestamp", "seed", "config",
        "structure", "graph_stats", "endpoint_stats", "sb_stats", "path_stats",
    )
    _keys(data, "manifest", required=required, optional=("migration",))
    schema = _string(data["schema_version"], "manifest.schema_version")
    if schema != SCHEMA_VERSION:
        raise _error("manifest.schema_version", f"unsupported version '{schema}'; supported version is '{SCHEMA_VERSION}'")
    migration_source: str | None = None
    migration_warnings: Tuple[str, ...] = ()
    if "migration" in data:
        migration = _mapping(data["migration"], "manifest.migration")
        _keys(migration, "manifest.migration", required=("source_schema_version", "warnings"))
        migration_source = _string(migration["source_schema_version"], "manifest.migration.source_schema_version")
        migration_warnings = tuple(
            _string(item, f"manifest.migration.warnings[{index}]")
            for index, item in enumerate(_list(migration["warnings"], "manifest.migration.warnings"))
        )
    result = RunManifest(
        seed=_integer(data["seed"], "manifest.seed"),
        config_dump=dict(_mapping(data["config"], "manifest.config")),
        structural_stats=StructuralDiagnostics.from_dict(data["structure"]),
        graph_stats=GraphDiagnosticsData.from_dict(data["graph_stats"]),
        endpoint_stats=EndpointDiagnosticsData.from_dict(data["endpoint_stats"]),
        sb_stats=SBDiagnostics.from_dict(data["sb_stats"]),
        path_stats=PathDiagnosticsData.from_dict(data["path_stats"]),
        run_id=_string(data["run_id"], "manifest.run_id"),
        timestamp=_number(data["timestamp"], "manifest.timestamp", minimum=0.0),
        version=_string(data["version"], "manifest.version"),
        schema_version=schema,
        migration_source_version=migration_source,
        migration_warnings=migration_warnings,
    )
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _error("manifest", f"must be JSON serializable ({exc})") from exc
    return result


def migrate_legacy_manifest(value: object) -> Dict[str, Any]:
    """Migrate the unversioned manifest without inventing unavailable diagnostics."""
    data = _mapping(value, "legacy_manifest")
    seed = _integer(_required(data, "seed", "legacy_manifest"), "legacy_manifest.seed")
    config = dict(_mapping(_required(data, "config", "legacy_manifest"), "legacy_manifest.config"))
    legacy_sb = _mapping(data.get("sb_stats", {}), "legacy_manifest.sb_stats")
    disconnected = legacy_sb.get("disconnected_nodes", legacy_sb.get("pruned_nodes", 0))
    warnings = [
        "Graph proposal, scoring, rejection, and pruning diagnostics were unavailable.",
        "Original endpoint support and unreachable mass were unavailable.",
        "Path score was unavailable.",
    ]
    if "pruned_nodes" in legacy_sb and "disconnected_nodes" not in legacy_sb:
        warnings.append("Legacy pruned_nodes was renamed to disconnected_nodes because it counted connectivity defects.")
    return {
        "schema_version": SCHEMA_VERSION,
        "version": data.get("version", APPLICATION_VERSION),
        "run_id": data.get("run_id", str(uuid.uuid4())),
        "timestamp": data.get("timestamp", time.time()),
        "seed": seed,
        "config": config,
        "structure": data.get("structure", StructuralDiagnostics().to_dict()),
        "graph_stats": GraphDiagnosticsData().to_dict(),
        "endpoint_stats": EndpointDiagnosticsData().to_dict(),
        "sb_stats": {
            "iterations_run": legacy_sb.get("iterations_run", 0),
            "converged": legacy_sb.get("converged", False),
            "final_max_delta": legacy_sb.get("final_max_delta", 0.0),
            "layer_sizes": legacy_sb.get("layer_sizes", []),
            "disconnected_nodes": disconnected,
            "effective_entropy": legacy_sb.get("effective_entropy", 0.0),
            "residual_history": [],
            "layer_entropies": [],
        },
        "path_stats": PathDiagnosticsData(path_mode="unknown").to_dict(),
        "migration": {"source_schema_version": "legacy-unversioned", "warnings": warnings},
    }


ROLE_TENSION = {"hold": 0.20, "prep": 0.45, "change": 0.65, "cad": 0.90}


def compute_tension_curve(role_timeline: List[TimelineEvent]) -> List[Tuple[float, float]]:
    tension_map = {"Tonic": 0.1, "Subdominant": 0.5, "Dominant": 0.9, "Transition": 0.6}
    return [(event.start_time, tension_map.get(event.label, 0.5)) for event in role_timeline]


def _segment_timeline(values: Iterable[str]) -> List[TimelineEvent]:
    items = tuple(values)
    if not items:
        return []
    events: List[TimelineEvent] = []
    start = 0
    current = items[0]
    for index, label in enumerate(items[1:], start=1):
        if label != current:
            events.append(TimelineEvent(float(start), float(index), current))
            start, current = index, label
    events.append(TimelineEvent(float(start), float(len(items)), current))
    return events


def build_structural_diagnostics(path: tuple[Any, ...], vocabularies: Any) -> StructuralDiagnostics:
    states = path[:-1] if len(path) > 1 else path
    roles = [vocabularies.roles.token_for_id(state.role_id).label for state in states]
    return StructuralDiagnostics(
        key_timeline=_segment_timeline(vocabularies.keys.token_for_id(state.key_id).label for state in states),
        chord_timeline=_segment_timeline(vocabularies.chords.token_for_id(state.chord_id).label for state in states),
        role_timeline=_segment_timeline(roles),
        groove_timeline=_segment_timeline(vocabularies.grooves.token_for_id(state.groove_id).label for state in states),
        boundaries=[float(index) for index, state in enumerate(states) if state.boundary_lvl > 0],
        tension_curve=[
            (float(index), min(1.0, ROLE_TENSION.get(role, 0.5) + 0.05 * state.boundary_lvl))
            for index, (state, role) in enumerate(zip(states, roles))
        ],
    )


def extract_graph_diagnostics_data(graph_diagnostics: Any) -> GraphDiagnosticsData:
    layers: List[LayerGraphStats] = []
    rejections: Dict[str, int] = {}
    prunes: Dict[str, int] = {}
    for layer in graph_diagnostics.layer_diagnostics:
        layers.append(LayerGraphStats(
            time_index=layer.time_index,
            source_states=layer.source_state_count,
            proposed=layer.raw_candidate_count,
            legal=layer.legal_candidate_count,
            scored=layer.scored_candidate_count,
            duplicate_proposals=layer.duplicate_candidate_count,
            candidate_pruned=layer.d_max_pruned_candidate_count,
            scored_unique_states=layer.scored_unique_state_count,
            retained=layer.kept_candidate_count,
            pruned=layer.pruned_candidate_count,
            scored_edges=layer.raw_edge_count,
            retained_edges=layer.kept_edge_count,
            pruned_edges=layer.pruned_edge_count,
        ))
        for rejection in layer.rejected_proposals:
            rejections[rejection.reason] = rejections.get(rejection.reason, 0) + 1
        for pruned_state in layer.pruned_states:
            prunes[pruned_state.reason] = prunes.get(pruned_state.reason, 0) + 1
        if layer.d_max_pruned_candidate_count:
            prunes["candidate_outdegree_prune"] = (
                prunes.get("candidate_outdegree_prune", 0)
                + layer.d_max_pruned_candidate_count
            )
        if layer.outdegree_pruned_count:
            prunes["d_max_outdegree_prune"] = prunes.get("d_max_outdegree_prune", 0) + layer.outdegree_pruned_count
        if layer.state_pruned_edge_count:
            prunes["target_state_prune_edge"] = prunes.get("target_state_prune_edge", 0) + layer.state_pruned_edge_count

    def total(name: str) -> int:
        return sum(getattr(item, name) for item in layers)

    state_prunes = total("pruned")
    edge_prunes = total("pruned_edges")
    candidate_prunes = total("candidate_pruned")
    return GraphDiagnosticsData(
        layer_sizes=list(graph_diagnostics.layer_sizes),
        per_layer_stats=layers,
        rejection_summary=rejections,
        pruning_summary=prunes,
        total_proposed=total("proposed"),
        total_legal=total("legal"),
        total_scored=total("scored"),
        total_duplicate_proposals=total("duplicate_proposals"),
        total_candidate_pruned=candidate_prunes,
        total_scored_unique_states=total("scored_unique_states"),
        total_retained=total("retained"),
        total_pruned=state_prunes,
        total_scored_edges=total("scored_edges"),
        total_retained_edges=total("retained_edges"),
        total_pruned_edges=edge_prunes,
        total_prune_operations=candidate_prunes + state_prunes + edge_prunes,
    )


def extract_endpoint_diagnostics_data(plan_result: Any) -> EndpointDiagnosticsData:
    endpoints = plan_result.endpoints
    graph = plan_result.graph
    solver_pi0 = endpoints.solver_pi0 or plan_result.sb_problem.pi0
    solver_piT = endpoints.solver_piT or plan_result.sb_problem.piT
    start_states = set(graph.layers[0].states)
    end_states = set(graph.layers[-1].states)
    return EndpointDiagnosticsData(
        original_pi0_support_size=len(endpoints.pi0.layer.states),
        original_piT_support_size=len(endpoints.piT.layer.states),
        solver_pi0_support_size=len(solver_pi0.layer.states),
        solver_piT_support_size=len(solver_piT.layer.states),
        unreachable_pi0_mass=float(sum(
            probability for state, probability in zip(endpoints.pi0.layer.states, endpoints.pi0.probabilities)
            if state not in start_states
        )),
        unreachable_piT_mass=float(sum(
            probability for state, probability in zip(endpoints.piT.layer.states, endpoints.piT.probabilities)
            if state not in end_states
        )),
        selected_start_probability=float(endpoints.start_choice.selected_probability),
        selected_end_probability=float(endpoints.end_choice.selected_probability),
    )


def extract_path_diagnostics_data(plan_result: Any) -> PathDiagnosticsData:
    return PathDiagnosticsData(
        path_mode=plan_result.diagnostics.path_mode,
        path_score=float(plan_result.path_score) if plan_result.path_score is not None else None,
        path_state_count=len(plan_result.path),
        path_transition_count=max(0, len(plan_result.path) - 1),
    )


def build_run_manifest(plan_result: Any, *, seed: int, config_dump: Dict[str, Any]) -> RunManifest:
    manifest = RunManifest(
        seed=seed,
        config_dump=config_dump,
        structural_stats=build_structural_diagnostics(plan_result.path, plan_result.vocabularies),
        graph_stats=extract_graph_diagnostics_data(plan_result.graph.diagnostics),
        endpoint_stats=extract_endpoint_diagnostics_data(plan_result),
        sb_stats=SBDiagnostics.from_solution(plan_result.sb_solution),
        path_stats=extract_path_diagnostics_data(plan_result),
    )
    manifest.to_dict()
    return manifest
