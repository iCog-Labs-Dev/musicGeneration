import math
import time
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from aimusic.scoring.tension import TENSION_MODEL_VERSION

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
    target_tension_curve: List[Tuple[float, float]] = field(default_factory=list)
    tension_deviation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_timeline": [dataclasses.asdict(e) for e in self.key_timeline],
            "chord_timeline": [dataclasses.asdict(e) for e in self.chord_timeline],
            "role_timeline": [dataclasses.asdict(e) for e in self.role_timeline],
            "groove_timeline": [dataclasses.asdict(e) for e in self.groove_timeline],
            "boundaries": self.boundaries,
            "tension_curve": self.tension_curve,
            "target_tension_curve": self.target_tension_curve,
            "tension_deviation": self.tension_deviation,
        }

@dataclass
class SBDiagnostics:
    """Logs the mathematical health and convergence of the Schrödinger Bridge solver."""
    iterations_run: int = 0
    converged: bool = False
    final_max_delta: float = 0.0
    layer_sizes: List[int] = field(default_factory=list)
    pruned_nodes: int = 0
    effective_entropy: float = 0.0
    sections: List[Dict[str, Any]] = field(default_factory=list)
    joins: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_solutions(cls, solutions: Sequence[Any]) -> "SBDiagnostics":
        """Aggregate independent solves, counting shared boundary layers once."""
        if not solutions:
            raise ValueError("At least one solution is required.")
        parts = [cls.from_solution(solution) for solution in solutions]
        layers = parts[0].layer_sizes + [size for part in parts[1:] for size in part.layer_sizes[1:]]
        # Shared anchors have zero entropy, so removing duplicates changes only the divisor.
        return cls(
            iterations_run=sum(part.iterations_run for part in parts),
            converged=all(part.converged for part in parts),
            final_max_delta=max(part.final_max_delta for part in parts),
            layer_sizes=layers, pruned_nodes=sum(part.pruned_nodes for part in parts),
            effective_entropy=sum(part.effective_entropy * len(part.layer_sizes) for part in parts) / len(layers),
        )

    @classmethod
    def from_plan(cls, plan: Any) -> "SBDiagnostics":
        result = cls.from_solutions(plan.solutions)
        for section in getattr(plan, "section_results", ()):
            item = cls.from_solution(section.sb_solution).to_dict()
            item.update(dataclasses.asdict(section.section))
            item["rng_stream_id"] = section.rng_stream_id
            result.sections.append(item)
        result.joins = [dataclasses.asdict(join) for join in getattr(plan, "joins", ())]
        return result

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
            layer_sizes=list(problem_diags.layer_sizes),
            pruned_nodes=pruned,
            effective_entropy=entropy
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclass(frozen=True)
class RunManifest:
    """Captures all parameters required to perfectly reproduce a generation run."""
    seed: int
    config_dump: Dict[str, Any]
    structural_stats: StructuralDiagnostics = field(default_factory=StructuralDiagnostics)
    sb_stats: SBDiagnostics = field(default_factory=SBDiagnostics)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: str = "0.1.0"
    tension_model_version: str = TENSION_MODEL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Converts the manifest to a JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "tension_model_version": self.tension_model_version,
            "seed": self.seed,
            "config": self.config_dump,
            "structure": self.structural_stats.to_dict(),
            "sb_stats": self.sb_stats.to_dict()
        }
