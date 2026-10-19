import math
import time
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

@dataclass(frozen=True)
class TimelineEvent:
    start_time: float
    end_time: float
    label: str

@dataclass(frozen=True)
class StructuralDiagnostics:
    key_timeline: Tuple[TimelineEvent, ...] = ()
    chord_timeline: Tuple[TimelineEvent, ...] = ()
    role_timeline: Tuple[TimelineEvent, ...] = ()
    groove_timeline: Tuple[TimelineEvent, ...] = ()
    boundaries: Tuple[float, ...] = ()
    tension_curve: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        for name in ("key_timeline", "chord_timeline", "role_timeline", "groove_timeline"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "boundaries", tuple(float(x) for x in self.boundaries))
        object.__setattr__(
            self,
            "tension_curve",
            tuple((float(t), float(v)) for t, v in self.tension_curve),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_timeline": [dataclasses.asdict(e) for e in self.key_timeline],
            "chord_timeline": [dataclasses.asdict(e) for e in self.chord_timeline],
            "role_timeline": [dataclasses.asdict(e) for e in self.role_timeline],
            "groove_timeline": [dataclasses.asdict(e) for e in self.groove_timeline],
            "boundaries": list(self.boundaries),
            "tension_curve": [list(item) for item in self.tension_curve]
        }

@dataclass(frozen=True)
class SBDiagnostics:
    """Logs the mathematical health and convergence of the Schrödinger Bridge solver."""
    iterations_run: int = 0
    converged: bool = False
    final_max_delta: float = 0.0
    layer_sizes: Tuple[int, ...] = ()
    pruned_nodes: int = 0
    effective_entropy: float = 0.0

    @classmethod
    def from_solution(cls, solution: Any) -> "SBDiagnostics":
        """Safely extracts stats from an aimusic.planning.sb.SBSolution object."""
        trace = solution.trace
        problem_diags = solution.problem.diagnostics
        
        pruned = problem_diags.zero_outdegree_count + problem_diags.zero_indegree_count
        
        # Calculate Average Effective Entropy (Shannon Entropy)
        entropy = 0.0
        if solution.marginals and solution.marginals.node_marginals_by_layer:
            layer_entropies = []
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
            layer_sizes=tuple(problem_diags.layer_sizes),
            pruned_nodes=pruned,
            effective_entropy=entropy
        )

    @classmethod
    def from_solutions(cls, solutions: Sequence[Any]) -> "SBDiagnostics":
        """Aggregate diagnostics from the two (or more) bridges in one plan."""
        items = [cls.from_solution(solution) for solution in solutions]
        if not items:
            return cls()
        total_weight = sum(max(1, len(item.layer_sizes)) for item in items)
        return cls(
            iterations_run=sum(item.iterations_run for item in items),
            converged=all(item.converged for item in items),
            final_max_delta=max(item.final_max_delta for item in items),
            layer_sizes=tuple(
                size
                for index, item in enumerate(items)
                for size in (item.layer_sizes if index == 0 else item.layer_sizes[1:])
            ),
            pruned_nodes=sum(item.pruned_nodes for item in items),
            effective_entropy=sum(
                item.effective_entropy * max(1, len(item.layer_sizes)) for item in items
            ) / total_weight,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["layer_sizes"] = list(self.layer_sizes)
        return data

@dataclass(frozen=True)
class RunManifest:
    """Captures all parameters required to perfectly reproduce a generation run."""
    seed: int
    config_dump: Dict[str, Any]
    structural_stats: StructuralDiagnostics = field(default_factory=StructuralDiagnostics)
    sb_stats: SBDiagnostics = field(default_factory=SBDiagnostics)
    sb_legs: Dict[str, SBDiagnostics] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the manifest to a JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "seed": self.seed,
            "config": self.config_dump,
            "structure": self.structural_stats.to_dict(),
            "sb_stats": self.sb_stats.to_dict(),
            "sb_legs": {name: stats.to_dict() for name, stats in self.sb_legs.items()},
        }

def compute_tension_curve(role_timeline: List[TimelineEvent]) -> List[Tuple[float, float]]:
    tension_map = {"Tonic": 0.1, "Subdominant": 0.5, "Dominant": 0.9, "Transition": 0.6}
    return [(e.start_time, tension_map.get(e.label, 0.5)) for e in role_timeline]
