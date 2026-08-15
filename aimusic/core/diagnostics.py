import math
import time
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"

@dataclass(frozen=True)
class TimelineEvent:
    start_time: float
    end_time: float
    label: str

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
            "key_timeline": [dataclasses.asdict(e) for e in self.key_timeline],
            "chord_timeline": [dataclasses.asdict(e) for e in self.chord_timeline],
            "role_timeline": [dataclasses.asdict(e) for e in self.role_timeline],
            "groove_timeline": [dataclasses.asdict(e) for e in self.groove_timeline],
            "boundaries": self.boundaries,
            "tension_curve": self.tension_curve,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuralDiagnostics":
        if not isinstance(data, dict):
            raise TypeError("StructuralDiagnostics data must be a dict.")
        return cls(
            key_timeline=[TimelineEvent(**e) for e in data.get("key_timeline", [])],
            chord_timeline=[TimelineEvent(**e) for e in data.get("chord_timeline", [])],
            role_timeline=[TimelineEvent(**e) for e in data.get("role_timeline", [])],
            groove_timeline=[TimelineEvent(**e) for e in data.get("groove_timeline", [])],
            boundaries=[float(b) for b in data.get("boundaries", [])],
            tension_curve=[(float(t[0]), float(t[1])) for t in data.get("tension_curve", [])],
        )

@dataclass
class LayerGraphStats:
    time_index: int
    proposed: int = 0
    legal: int = 0
    scored: int = 0
    retained: int = 0
    pruned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerGraphStats":
        if not isinstance(data, dict):
            raise TypeError("LayerGraphStats data must be a dict.")
        return cls(
            time_index=int(data.get("time_index", 0)),
            proposed=int(data.get("proposed", 0)),
            legal=int(data.get("legal", 0)),
            scored=int(data.get("scored", 0)),
            retained=int(data.get("retained", 0)),
            pruned=int(data.get("pruned", 0)),
        )

@dataclass
class GraphDiagnosticsData:
    layer_sizes: List[int] = field(default_factory=list)
    per_layer_stats: List[LayerGraphStats] = field(default_factory=list)
    rejection_summary: Dict[str, int] = field(default_factory=dict)
    pruning_summary: Dict[str, int] = field(default_factory=dict)
    total_proposed: int = 0
    total_legal: int = 0
    total_scored: int = 0
    total_retained: int = 0
    total_pruned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_sizes": list(self.layer_sizes),
            "per_layer_stats": [s.to_dict() for s in self.per_layer_stats],
            "rejection_summary": dict(self.rejection_summary),
            "pruning_summary": dict(self.pruning_summary),
            "total_proposed": self.total_proposed,
            "total_legal": self.total_legal,
            "total_scored": self.total_scored,
            "total_retained": self.total_retained,
            "total_pruned": self.total_pruned,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphDiagnosticsData":
        if not isinstance(data, dict):
            raise TypeError("GraphDiagnosticsData data must be a dict.")
        per_layer = [LayerGraphStats.from_dict(item) for item in data.get("per_layer_stats", [])]
        return cls(
            layer_sizes=[int(x) for x in data.get("layer_sizes", [])],
            per_layer_stats=per_layer,
            rejection_summary={str(k): int(v) for k, v in data.get("rejection_summary", {}).items()},
            pruning_summary={str(k): int(v) for k, v in data.get("pruning_summary", {}).items()},
            total_proposed=int(data.get("total_proposed", 0)),
            total_legal=int(data.get("total_legal", 0)),
            total_scored=int(data.get("total_scored", 0)),
            total_retained=int(data.get("total_retained", 0)),
            total_pruned=int(data.get("total_pruned", 0)),
        )

@dataclass
class EndpointDiagnosticsData:
    pi0_support_size: int = 0
    piT_support_size: int = 0
    unreachable_probability_mass: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndpointDiagnosticsData":
        if not isinstance(data, dict):
            raise TypeError("EndpointDiagnosticsData data must be a dict.")
        return cls(
            pi0_support_size=int(data.get("pi0_support_size", 0)),
            piT_support_size=int(data.get("piT_support_size", 0)),
            unreachable_probability_mass=float(data.get("unreachable_probability_mass", 0.0)),
        )

@dataclass
class PathDiagnosticsData:
    path_mode: str = "map"
    path_score: Optional[float] = None
    path_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathDiagnosticsData":
        if not isinstance(data, dict):
            raise TypeError("PathDiagnosticsData data must be a dict.")
        score = data.get("path_score")
        return cls(
            path_mode=str(data.get("path_mode", "map")),
            path_score=float(score) if score is not None else None,
            path_length=int(data.get("path_length", 0)),
        )

@dataclass
class SBDiagnostics:
    """Logs the mathematical health and convergence of the Schrödinger Bridge solver."""
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
        return self.disconnected_nodes

    @classmethod
    def from_solution(cls, solution: Any) -> "SBDiagnostics":
        """Safely extracts stats from an aimusic.planning.sb.SBSolution object."""
        trace = solution.trace
        problem_diags = solution.problem.diagnostics
        
        disconnected = problem_diags.zero_outdegree_count + problem_diags.zero_indegree_count
        
        # Calculate Shannon Entropy
        entropy = 0.0
        layer_entropies: List[float] = []
        if solution.marginals and solution.marginals.node_marginals_by_layer:
            for layer_probs in solution.marginals.node_marginals_by_layer:
                h = 0.0
                for p in layer_probs:
                    if p > 0.0:
                        h -= p * math.log(p)
                layer_entropies.append(h)
            if layer_entropies:
                entropy = sum(layer_entropies) / len(layer_entropies)

        return cls(
            iterations_run=trace.iterations,
            converged=trace.converged,
            final_max_delta=trace.final_max_delta,
            layer_sizes=list(problem_diags.layer_sizes),
            disconnected_nodes=disconnected,
            effective_entropy=entropy,
            residual_history=list(trace.residual_history),
            layer_entropies=layer_entropies,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SBDiagnostics":
        if not isinstance(data, dict):
            raise TypeError("SBDiagnostics data must be a dict.")
        disc_nodes = data.get("disconnected_nodes", data.get("pruned_nodes", 0))
        return cls(
            iterations_run=int(data.get("iterations_run", 0)),
            converged=bool(data.get("converged", False)),
            final_max_delta=float(data.get("final_max_delta", 0.0)),
            layer_sizes=[int(x) for x in data.get("layer_sizes", [])],
            disconnected_nodes=int(disc_nodes),
            effective_entropy=float(data.get("effective_entropy", 0.0)),
            residual_history=[float(x) for x in data.get("residual_history", [])],
            layer_entropies=[float(x) for x in data.get("layer_entropies", [])],
        )

@dataclass
class RunManifest:
    """Captures all parameters required to perfectly reproduce a generation run."""
    seed: int
    config_dump: Dict[str, Any]
    structural_stats: StructuralDiagnostics = field(default_factory=StructuralDiagnostics)
    sb_stats: SBDiagnostics = field(default_factory=SBDiagnostics)
    graph_stats: GraphDiagnosticsData = field(default_factory=GraphDiagnosticsData)
    endpoint_stats: EndpointDiagnosticsData = field(default_factory=EndpointDiagnosticsData)
    path_stats: PathDiagnosticsData = field(default_factory=PathDiagnosticsData)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: str = "0.1.0"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Converts the manifest to a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "config": self.config_dump,
            "structure": self.structural_stats.to_dict(),
            "sb_stats": self.sb_stats.to_dict(),
            "graph_stats": self.graph_stats.to_dict(),
            "endpoint_stats": self.endpoint_stats.to_dict(),
            "path_stats": self.path_stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        """Deserializes and validates a dictionary into a RunManifest instance."""
        if not isinstance(data, dict):
            raise TypeError(f"Manifest data must be a dict, got {type(data).__name__}.")

        if "seed" not in data:
            raise ValueError("Manifest missing required field 'seed'.")
        if "config" not in data and "config_dump" not in data:
            raise ValueError("Manifest missing required field 'config'.")

        schema_ver = data.get("schema_version")
        if schema_ver is not None:
            if not isinstance(schema_ver, str):
                raise ValueError("schema_version must be a string.")
            try:
                major = int(schema_ver.split(".")[0])
            except (ValueError, IndexError):
                raise ValueError(f"Unsupported schema version: {schema_ver}")
            if major != 1:
                raise ValueError(f"Unsupported schema version: {schema_ver}")

        config_data = data.get("config", data.get("config_dump", {}))
        struct_data = data.get("structure", {})
        sb_data = data.get("sb_stats", {})
        graph_data = data.get("graph_stats", {})
        endpoint_data = data.get("endpoint_stats", {})
        path_data = data.get("path_stats", {})

        return cls(
            seed=int(data["seed"]),
            config_dump=config_data,
            structural_stats=StructuralDiagnostics.from_dict(struct_data),
            sb_stats=SBDiagnostics.from_dict(sb_data),
            graph_stats=GraphDiagnosticsData.from_dict(graph_data),
            endpoint_stats=EndpointDiagnosticsData.from_dict(endpoint_data),
            path_stats=PathDiagnosticsData.from_dict(path_data),
            run_id=str(data.get("run_id", str(uuid.uuid4()))),
            timestamp=float(data.get("timestamp", time.time())),
            version=str(data.get("version", "0.1.0")),
            schema_version=schema_ver if schema_ver is not None else SCHEMA_VERSION,
        )

ROLE_TENSION = {"hold": 0.20, "prep": 0.45, "change": 0.65, "cad": 0.90}

def compute_tension_curve(role_timeline: List[TimelineEvent]) -> List[Tuple[float, float]]:
    tension_map = {"Tonic": 0.1, "Subdominant": 0.5, "Dominant": 0.9, "Transition": 0.6}
    return [(e.start_time, tension_map.get(e.label, 0.5)) for e in role_timeline]

def _segment_timeline(values: Iterable[str]) -> List[TimelineEvent]:
    items = tuple(values)
    if not items:
        return []
    events: List[TimelineEvent] = []
    start = 0
    current = items[0]
    for index, label in enumerate(items[1:], start=1):
        if label == current:
            continue
        events.append(TimelineEvent(float(start), float(index), current))
        start = index
        current = label
    events.append(TimelineEvent(float(start), float(len(items)), current))
    return events

def build_structural_diagnostics(path: tuple[Any, ...], vocabularies: Any) -> StructuralDiagnostics:
    decoded_states = path[:-1] if len(path) > 1 else path
    key_labels = [vocabularies.keys.token_for_id(state.key_id).label for state in decoded_states]
    chord_labels = [vocabularies.chords.token_for_id(state.chord_id).label for state in decoded_states]
    role_labels = [vocabularies.roles.token_for_id(state.role_id).label for state in decoded_states]
    groove_labels = [vocabularies.grooves.token_for_id(state.groove_id).label for state in decoded_states]
    boundaries = [float(index) for index, state in enumerate(decoded_states) if state.boundary_lvl > 0]
    tension_curve = [
        (
            float(index),
            min(
                1.0,
                ROLE_TENSION.get(vocabularies.roles.token_for_id(state.role_id).label, 0.5)
                + (0.05 * state.boundary_lvl),
            ),
        )
        for index, state in enumerate(decoded_states)
    ]
    return StructuralDiagnostics(
        key_timeline=_segment_timeline(key_labels),
        chord_timeline=_segment_timeline(chord_labels),
        role_timeline=_segment_timeline(role_labels),
        groove_timeline=_segment_timeline(groove_labels),
        boundaries=boundaries,
        tension_curve=tension_curve,
    )

def extract_graph_diagnostics_data(graph_diags: Any) -> GraphDiagnosticsData:
    if graph_diags is None or not hasattr(graph_diags, "layer_diagnostics"):
        return GraphDiagnosticsData()
    per_layer_stats: List[LayerGraphStats] = []
    rejection_summary: Dict[str, int] = {}
    pruning_summary: Dict[str, int] = {}

    total_proposed = 0
    total_legal = 0
    total_scored = 0
    total_retained = 0
    total_pruned = 0

    for layer_diag in graph_diags.layer_diagnostics:
        proposed = layer_diag.raw_candidate_count
        legal = layer_diag.raw_edge_count
        scored = layer_diag.raw_edge_count
        retained = layer_diag.kept_candidate_count
        pruned = len(layer_diag.pruned_states)

        per_layer_stats.append(
            LayerGraphStats(
                time_index=layer_diag.time_index,
                proposed=proposed,
                legal=legal,
                scored=scored,
                retained=retained,
                pruned=pruned,
            )
        )

        total_proposed += proposed
        total_legal += legal
        total_scored += scored
        total_retained += retained
        total_pruned += pruned

        for rej in layer_diag.rejected_proposals:
            reason = rej.reason
            rejection_summary[reason] = rejection_summary.get(reason, 0) + 1

        for pruned_st in layer_diag.pruned_states:
            reason = pruned_st.reason
            pruning_summary[reason] = pruning_summary.get(reason, 0) + 1

    return GraphDiagnosticsData(
        layer_sizes=list(graph_diags.layer_sizes),
        per_layer_stats=per_layer_stats,
        rejection_summary=rejection_summary,
        pruning_summary=pruning_summary,
        total_proposed=total_proposed,
        total_legal=total_legal,
        total_scored=total_scored,
        total_retained=total_retained,
        total_pruned=total_pruned,
    )

def extract_endpoint_diagnostics_data(endpoints: Any, graph: Any) -> EndpointDiagnosticsData:
    if endpoints is None or graph is None:
        return EndpointDiagnosticsData()

    pi0_support_size = len(endpoints.pi0.layer.states) if hasattr(endpoints, "pi0") and hasattr(endpoints.pi0, "layer") else 0
    piT_support_size = len(endpoints.piT.layer.states) if hasattr(endpoints, "piT") and hasattr(endpoints.piT, "layer") else 0

    graph_start_states = set(graph.layers[0].states) if hasattr(graph, "layers") and graph.layers else set()
    graph_end_states = set(graph.layers[-1].states) if hasattr(graph, "layers") and graph.layers else set()

    unreachable_mass = 0.0

    if hasattr(endpoints, "pi0") and hasattr(endpoints.pi0, "layer"):
        for state, prob in zip(endpoints.pi0.layer.states, endpoints.pi0.probabilities):
            if state not in graph_start_states:
                unreachable_mass += float(prob)

    if hasattr(endpoints, "piT") and hasattr(endpoints.piT, "layer"):
        for state, prob in zip(endpoints.piT.layer.states, endpoints.piT.probabilities):
            if state not in graph_end_states:
                unreachable_mass += float(prob)

    if hasattr(graph, "diagnostics") and hasattr(graph.diagnostics, "layer_diagnostics") and graph.diagnostics.layer_diagnostics:
        last_layer_diag = graph.diagnostics.layer_diagnostics[-1]
        for pruned_st in last_layer_diag.pruned_states:
            if pruned_st.reason == "unreachable_endpoint":
                if hasattr(endpoints, "piT") and hasattr(endpoints.piT, "layer") and pruned_st.state in endpoints.piT.layer.states:
                    idx = endpoints.piT.layer.states.index(pruned_st.state)
                    if pruned_st.state in graph_end_states:
                        unreachable_mass += float(endpoints.piT.probabilities[idx])

    return EndpointDiagnosticsData(
        pi0_support_size=pi0_support_size,
        piT_support_size=piT_support_size,
        unreachable_probability_mass=float(unreachable_mass),
    )

def extract_path_diagnostics_data(plan_result: Any) -> PathDiagnosticsData:
    path_mode = getattr(plan_result.diagnostics, "path_mode", "map") if hasattr(plan_result, "diagnostics") else "map"
    path_score = getattr(plan_result, "path_score", None)
    path_length = len(plan_result.path) if hasattr(plan_result, "path") else 0
    return PathDiagnosticsData(
        path_mode=str(path_mode),
        path_score=float(path_score) if path_score is not None else None,
        path_length=path_length,
    )

def build_run_manifest(
    plan_result: Any,
    *,
    seed: int,
    config_dump: Dict[str, Any],
) -> RunManifest:
    structural_stats = build_structural_diagnostics(plan_result.path, plan_result.vocabularies)
    sb_stats = SBDiagnostics.from_solution(plan_result.sb_solution)
    graph_stats = extract_graph_diagnostics_data(plan_result.graph.diagnostics)
    endpoint_stats = extract_endpoint_diagnostics_data(plan_result.endpoints, plan_result.graph)
    path_stats = extract_path_diagnostics_data(plan_result)

    return RunManifest(
        seed=seed,
        config_dump=config_dump,
        structural_stats=structural_stats,
        sb_stats=sb_stats,
        graph_stats=graph_stats,
        endpoint_stats=endpoint_stats,
        path_stats=path_stats,
    )