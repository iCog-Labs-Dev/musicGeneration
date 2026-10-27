""" Endpoint reachability, global feasibility, fallbacks, validated SB solve.

Pipeline (distribution mode, and singleton mode as the special case):

    requested pi0, piT (strictly positive support)
      -> Level 1  individual reachability   (per state; retained-graph connectivity)
      -> Level 2  global feasibility        (max-flow on the exact reachability
                                             relation; respects the masses)
      -> fallback chosen by policy AND by problem level
      -> restricted graph (zero-mass boundary states and dead paths pruned)
      -> effective pi0, piT aligned to the restricted boundary layers
      -> build_sb_problem / solve_sb
      -> solution validation (convergence + both marginal errors)

Level 1 only proves every endpoint state can join *some* path. Level 2 proves the
marginals can be met *simultaneously*: a coupling F >= 0 supported on the
reachability relation with row sums pi0 and column sums piT. By Gale/Hall this holds
iff pi0(X) <= piT(N(X)) for every start set X; max-flow decides it.

Fallback policies (STRICT is the library default and never changes the request):
  DROP_UNREACHABLE_ONLY        Level 1 whole-state removal of individually unreachable
                               states; Level 2 conflicts raise. Default of the Method A
                               pipeline, whose endpoint generator is graph-blind.
  DROP_AND_RENORMALIZE         whole-state removal at both levels (Level 2 removal is a
                               greedy heuristic: choosing the best state subset is a
                               subset-selection problem; no optimality is claimed).
  PARTIAL_TRIM_AND_RENORMALIZE Level 2 only: max-flow routable-mass trim. Level 1
                               failures raise (it never removes unreachable states).
  CONDITIONAL_DROP_THEN_TRIM   Level 1 -> whole-state removal; Level 2 -> routable-mass
                               trim. Never uses whole-state removal for Level 2.

Partial trim is OPTIMAL for its objective: it retains the maximum total mass F*
(= minimum total trimmed mass 1 - F* on each side, by max-flow = min-cut). It does not
minimize distribution distance, and the per-state split among equal-F* flows is the
deterministic Dinic split, not a uniquely optimal one.

Mass-accounting vocabulary (kept strictly separate):
  requested : the distributions asked for.
  removed   : requested probability of states removed *whole* (any stage; includes states
              a trim reduced to exactly zero).
  trimmed   : probability reduced from states that are *retained*.
  effective : normalized distributions actually passed to the SB solver.
  realized  : boundary marginals of the numerical SB solution.

Numerical semantics (see ``EndpointTolerances``; the relations are enforced):
  * Positivity is strict: a requested mass is positive iff ``mass > 0.0``. No epsilon
    decides whether input probability exists.
  * capacity_epsilon: the float noise floor of max-flow arithmetic. A residual capacity
    at or below it counts as saturated and a *computed* routable mass at or below it is
    snapped to 0. It is not applied to requested masses.
  * flow_tolerance: the pair is feasible iff the deficit (1 - max-flow) <= this. Must be
    >= 1000 * capacity_epsilon so accumulated flow noise cannot flip the verdict.
  * marginal_tolerance: post-solve |realized - effective|. Must be >= flow_tolerance:
    a pair accepted with deficit delta can only be matched to ~delta, so a tighter
    marginal tolerance could reject pairs the feasibility check accepted.
  * SBConfig.tolerance bounds the solver's potential delta and is independent: a
    "converged" solve that misses marginal_tolerance still fails validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from aimusic.core.config import SBConfig
from aimusic.core.core_types import BeatState, EndpointDistribution, Layer
from aimusic.planning.sb import (
    SBProblem,
    SBSolution,
    SolvedBridge,
    build_sb_problem,
    solve_sb,
)

if TYPE_CHECKING:
    from aimusic.core.vocab import Vocabularies
    from aimusic.planning.graph import EdgeScoreDiagnostics, SparseGraph

_logger = logging.getLogger(__name__)

_ROUND_MASS = 12  # decimals used when comparing masses for deterministic ties
_INF = float("inf")

SIDE_START = "start"
SIDE_TERMINAL = "terminal"
_SIDE_RANK = {SIDE_START: 0, SIDE_TERMINAL: 1}


# --------------------------------------------------------------------------- #
# Enums and exceptions
# --------------------------------------------------------------------------- #
class EndpointMode(Enum):
    """How Method A treats the configured endpoint distributions."""

    DISTRIBUTION = "distribution"  # true distribution-to-distribution bridge (default)
    SINGLETON = "singleton"  # explicit compatibility mode: one state per side


class EndpointFallbackPolicy(Enum):
    """What to do when requested endpoint distributions are infeasible.

    STRICT (library default): never change the request; raise before the SB solver runs.
    DROP_UNREACHABLE_ONLY (pipeline default, ``MethodARunConfig``): Level 1 only. The
        endpoint generator proposes states without seeing the graph, so some are
        individually unreachable; those are removed whole (reported, survivors
        renormalized). A Level 2 mass conflict is NOT repaired: it raises, exactly like
        STRICT. No trimming, no whole-state removal for mass reasons, no interior blend.
    DROP_AND_RENORMALIZE: whole-state removal only (Level 1 and Level 2); survivors keep
        their relative probabilities and each side is renormalized independently.
    PARTIAL_TRIM_AND_RENORMALIZE: opt-in; Level 2 only. Reduces masses of retained
        states by the max-flow routable-mass rule; Level 1 failures raise.
    CONDITIONAL_DROP_THEN_TRIM: Level 1 problems -> whole-state removal, then Level 2
        problems -> routable-mass trim. Opt-in; changes the requested distribution.
    """

    STRICT = "strict"
    DROP_UNREACHABLE_ONLY = "drop_unreachable_only"
    DROP_AND_RENORMALIZE = "drop_and_renormalize"
    PARTIAL_TRIM_AND_RENORMALIZE = "partial_trim_and_renormalize"
    CONDITIONAL_DROP_THEN_TRIM = "conditional_drop_then_trim"


UnreachablePolicy = EndpointFallbackPolicy  # backwards-compatible name


class EndpointFeasibilityError(ValueError):
    """Endpoint support/mass is infeasible (raised before the SB solver runs)."""

    def __init__(
        self,
        message: str,
        *,
        flow_check: Optional["FlowCheck"] = None,
        states: Sequence[BeatState] = (),
    ) -> None:
        super().__init__(message)
        self.flow_check = flow_check
        self.states = tuple(states)


class EndpointReachabilityError(EndpointFeasibilityError):
    """Level 1 failure: some endpoint states are individually unreachable."""


class EndpointMassInfeasibleError(EndpointFeasibilityError):
    """Level 2 failure: marginals cannot be met jointly on the retained graph."""


class EndpointValidationError(RuntimeError):
    """Post-solve validation failure (never a successful result)."""

    def __init__(
        self,
        message: str,
        *,
        start_error: Optional[float] = None,
        terminal_error: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.start_error = start_error
        self.terminal_error = terminal_error


class EndpointMarginalError(EndpointValidationError):
    """Realized marginal deviates from the effective target beyond tolerance."""


class EndpointSolverNonConvergenceError(EndpointValidationError):
    """The SB solver did not converge."""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def endpoint_state_sort_key(state: BeatState) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        state.meter_id,
        state.beat_in_bar,
        state.boundary_lvl,
        state.key_id,
        state.chord_id,
        state.role_id,
        state.head_id,
        state.groove_id,
    )


@dataclass(frozen=True)
class EndpointTolerances:
    """Numerical tolerances with enforced relations (see module docstring)."""

    flow_tolerance: float = 1e-9
    marginal_tolerance: float = 1e-6
    capacity_epsilon: float = 1e-14
    normalization_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in (
            "flow_tolerance", "marginal_tolerance", "capacity_epsilon", "normalization_tolerance"
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0.0:
                raise ValueError(f"{name} must be a positive number (got {value!r}).")
        if self.capacity_epsilon * 1000.0 > self.flow_tolerance:
            raise ValueError(
                "capacity_epsilon must be <= flow_tolerance / 1000 so accumulated max-flow "
                f"noise cannot flip feasibility (got {self.capacity_epsilon} vs "
                f"{self.flow_tolerance})."
            )
        if self.flow_tolerance > self.marginal_tolerance:
            raise ValueError(
                "flow_tolerance must be <= marginal_tolerance: a pair accepted with deficit "
                "delta can only be matched to ~delta, so a tighter marginal tolerance would "
                f"reject accepted pairs (got {self.flow_tolerance} > {self.marginal_tolerance})."
            )


@dataclass(frozen=True)
class FlowCheck:
    """Result of one global feasibility (max-flow) check.

    When infeasible, the min-cut is reported as a violated Hall set:
    ``violated_start_states`` = X, ``neighbor_terminal_states`` = N(X), with
    ``excess = pi0(X) - piT(N(X)) == deficit`` (up to floating point).
    """

    flow_value: float
    deficit: float
    feasible: bool
    tolerance: float
    total_start_mass: float
    total_terminal_mass: float
    violated_start_states: Tuple[BeatState, ...] = ()
    neighbor_terminal_states: Tuple[BeatState, ...] = ()
    start_mass_in_cut: float = 0.0
    terminal_mass_in_neighborhood: float = 0.0

    @property
    def excess(self) -> float:
        return self.start_mass_in_cut - self.terminal_mass_in_neighborhood

    def to_dict(self, vocabularies: Optional["Vocabularies"] = None) -> dict:
        return {
            "flow_value": self.flow_value,
            "deficit": self.deficit,
            "feasible": self.feasible,
            "tolerance": self.tolerance,
            "violated_start_states": [s.to_dict(vocabularies) for s in self.violated_start_states],
            "neighbor_terminal_states": [
                s.to_dict(vocabularies) for s in self.neighbor_terminal_states
            ],
            "start_mass_in_cut": self.start_mass_in_cut,
            "terminal_mass_in_neighborhood": self.terminal_mass_in_neighborhood,
        }


@dataclass(frozen=True)
class RemovedEndpointState:
    """A state removed *whole*. ``requested_mass`` is its originally requested probability."""

    side: str
    state: BeatState
    requested_mass: float
    reason: str
    stage: str  # individual_reachability | global_feasibility | fallback_cascade
    iteration: int = 0


@dataclass(frozen=True)
class TrimOperation:
    """One retained state whose mass was reduced by the routable-mass trim.

    Masses are in the normalized units current when the trim ran (both sides sum to 1).
    States trimmed to exactly zero are NOT trims: they are recorded as removed states
    (stage ``partial_trim_zeroed``).
    """

    iteration: int
    side: str
    state: BeatState
    original_mass: float  # state's normalized mass before this trim
    trimmed_mass: float  # tau, subtracted from the state's mass
    resulting_mass: float  # original_mass - tau, before renormalization
    effective_mass: float  # after renormalizing the trimmed side
    reason: str
    flow_before: float
    deficit_before: float
    flow_after: Optional[float] = None  # filled from the next feasibility check
    feasible_after: Optional[bool] = None


@dataclass(frozen=True)
class ProbabilityChange:
    state: BeatState
    requested: float
    effective: float

    @property
    def delta(self) -> float:
        return self.effective - self.requested


@dataclass(frozen=True)
class EndpointMassReport:
    """Requested / removed / trimmed / effective / realized accounting for one side."""

    side: str
    requested: EndpointDistribution
    effective: EndpointDistribution
    requested_mass: float
    removed: Tuple[RemovedEndpointState, ...] = ()
    trims: Tuple[TrimOperation, ...] = ()
    probability_changes: Tuple[ProbabilityChange, ...] = ()
    realized_probabilities: Tuple[float, ...] = ()
    max_abs_realized_error: Optional[float] = None
    interior_margin: float = 0.0  # lambda of the explicit interior blend (0 = none)
    retained_mass: Optional[float] = None
    trimmed_requested_mass: float = 0.0

    @property
    def target(self) -> EndpointDistribution:  # compatibility alias
        return self.effective

    @property
    def removed_state_mass(self) -> float:
        """Requested probability of states removed whole (any stage)."""
        return float(sum(item.requested_mass for item in self.removed))

    @property
    def removed_mass(self) -> float:  # compatibility alias
        return self.removed_state_mass

    @property
    def individually_unreachable_mass(self) -> float:
        return float(
            sum(i.requested_mass for i in self.removed if i.stage == "individual_reachability")
        )

    @property
    def feasible_mass(self) -> float:
        if self.retained_mass is not None:
            return self.retained_mass
        return max(
            0.0,
            self.requested_mass
            - self.removed_state_mass
            - max(0.0, self.trimmed_requested_mass),
        )

    @property
    def modified(self) -> bool:
        """True if this side's effective distribution differs from the request by design
        (removed states, trimmed states or an interior blend)."""
        return bool(self.removed or self.trims or self.interior_margin > 0.0)

    @property
    def trimmed_mass(self) -> float:
        """Sum of raw per-iteration trims. NOT the full distribution change:
        renormalization shifts every surviving state, see ``total_variation_shift``."""
        return float(sum(t.trimmed_mass for t in self.trims))

    @property
    def total_variation_shift(self) -> float:
        """0.5 * sum |effective - requested| over all requested states."""
        return 0.5 * float(sum(abs(c.delta) for c in self.probability_changes))

    def with_realized(self, realized: Sequence[float]) -> "EndpointMassReport":
        realized_t = tuple(float(p) for p in realized)
        if len(realized_t) != len(self.effective.probabilities):
            raise ValueError("realized marginal must align with the effective layer.")
        error = max(
            (abs(r - t) for r, t in zip(realized_t, self.effective.probabilities)),
            default=0.0,
        )
        return replace(self, realized_probabilities=realized_t, max_abs_realized_error=error)

    def to_dict(self, vocabularies: Optional["Vocabularies"] = None) -> dict:
        return {
            "side": self.side,
            "requested": self.requested.to_dict(vocabularies),
            "effective": self.effective.to_dict(vocabularies),
            "requested_mass": self.requested_mass,
            "removed_state_mass": self.removed_state_mass,
            "individually_unreachable_mass": self.individually_unreachable_mass,
            "feasible_mass": self.feasible_mass,
            "retained_mass": self.retained_mass,
            "trimmed_requested_mass": self.trimmed_requested_mass,
            "trimmed_mass": self.trimmed_mass,
            "modified": self.modified,
            "total_variation_shift": self.total_variation_shift,
            "interior_margin": self.interior_margin,
            "removed": [
                {
                    "side": r.side,
                    "state": r.state.to_dict(vocabularies),
                    "requested_mass": r.requested_mass,
                    "reason": r.reason,
                    "stage": r.stage,
                    "iteration": r.iteration,
                }
                for r in self.removed
            ],
            "trims": [
                {
                    "iteration": t.iteration,
                    "state": t.state.to_dict(vocabularies),
                    "original_mass": t.original_mass,
                    "trimmed_mass": t.trimmed_mass,
                    "resulting_mass": t.resulting_mass,
                    "effective_mass": t.effective_mass,
                    "reason": t.reason,
                    "flow_before": t.flow_before,
                    "deficit_before": t.deficit_before,
                    "flow_after": t.flow_after,
                    "feasible_after": t.feasible_after,
                }
                for t in self.trims
            ],
            "probability_changes": [
                {
                    "state": c.state.to_dict(vocabularies),
                    "requested": c.requested,
                    "effective": c.effective,
                    "delta": c.delta,
                }
                for c in self.probability_changes
            ],
            "realized_probabilities": list(self.realized_probabilities),
            "max_abs_realized_error": self.max_abs_realized_error,
        }


@dataclass(frozen=True)
class EndpointResolution:
    """Everything decided before the SB solver runs."""

    policy: EndpointFallbackPolicy
    start: EndpointMassReport
    end: EndpointMassReport
    initial_flow_check: FlowCheck  # first global check (after Level 1 handling)
    final_flow_check: FlowCheck
    feasibility_history: Tuple[FlowCheck, ...]
    iterations: int
    termination_reason: str
    decisions: Tuple[str, ...] = ()
    graph: Optional["SparseGraph"] = None  # restricted graph actually handed to the solver
    layer_sizes_before: Tuple[int, ...] = ()
    layer_sizes_after: Tuple[int, ...] = ()
    tightness: Optional[Tightness] = None  # of the final pair, BEFORE any interior blend
    request_modified: bool = False
    modifications: Tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        """True iff the effective distributions equal the requested ones (no removal,
        trim or blend)."""
        return not self.request_modified

    def to_dict(self, vocabularies: Optional["Vocabularies"] = None) -> dict:
        return {
            "policy": self.policy.value,
            "termination_reason": self.termination_reason,
            "request_modified": self.request_modified,
            "modifications": list(self.modifications),
            "iterations": self.iterations,
            "decisions": list(self.decisions),
            "initial_flow_check": self.initial_flow_check.to_dict(vocabularies),
            "final_flow_check": self.final_flow_check.to_dict(vocabularies),
            "feasibility_history": [c.to_dict(vocabularies) for c in self.feasibility_history],
            "tight": None if self.tightness is None else self.tightness.tight,
            "forced_zero_pair_count": (
                None if self.tightness is None else len(self.tightness.forced_zero_pairs)
            ),
            "layer_sizes_before": list(self.layer_sizes_before),
            "layer_sizes_after": list(self.layer_sizes_after),
            "start": self.start.to_dict(vocabularies),
            "end": self.end.to_dict(vocabularies),
        }


@dataclass(frozen=True)
class EndpointSolveResult:
    graph: "SparseGraph"  # restricted graph the bridge was solved on
    resolution: EndpointResolution
    problem: SBProblem
    solution: SBSolution
    bridge: SolvedBridge
    start_mass: EndpointMassReport  # with realized marginals
    end_mass: EndpointMassReport


# --------------------------------------------------------------------------- #
# Reachability (Level 1 data) and max-flow (Level 2)
# --------------------------------------------------------------------------- #
def positive_mass_distribution(endpoint: EndpointDistribution) -> EndpointDistribution:
    """Restrict a distribution to its strictly positive support (order preserved)."""
    pairs = [
        (s, p) for s, p in zip(endpoint.layer.states, endpoint.probabilities) if p > 0.0
    ]
    if len(pairs) == len(endpoint.layer.states):
        return endpoint
    total = sum(p for _, p in pairs)
    return EndpointDistribution(
        layer=Layer(time_index=endpoint.layer.time_index, states=tuple(s for s, _ in pairs)),
        probabilities=tuple(p / total for _, p in pairs),
    )


def compute_endpoint_reachability(
    graph: "SparseGraph",
    starts: Sequence[BeatState],
    terminals: Sequence[BeatState],
) -> Dict[BeatState, Tuple[BeatState, ...]]:
    """Exact start -> terminals relation from the *retained* graph edges.

    ``relation[s]`` is the sorted tuple of terminals joined to ``s`` by at least one
    retained path. Nothing is assumed about unlisted pairs.
    """
    adjacency: List[Dict[BeatState, set]] = []
    for edges in graph.edges_by_time:
        adj: Dict[BeatState, set] = {}
        for edge in edges:
            adj.setdefault(edge.source, set()).add(edge.target)
        adjacency.append(adj)
    terminal_set = set(terminals)
    relation: Dict[BeatState, Tuple[BeatState, ...]] = {}
    for start in sorted(starts, key=endpoint_state_sort_key):
        frontier = {start}
        for adj in adjacency:
            nxt: set = set()
            for node in frontier:
                nxt.update(adj.get(node, ()))
            frontier = nxt
            if not frontier:
                break
        relation[start] = tuple(sorted(frontier & terminal_set, key=endpoint_state_sort_key))
    return relation


class _Dinic:
    """Small deterministic float Dinic max-flow (edges are added in a fixed order)."""

    def __init__(self, n: int, epsilon: float) -> None:
        self.n = n
        self.eps = epsilon
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.to: List[int] = []
        self.cap: List[float] = []
        self.level: List[int] = []
        self.it: List[int] = []

    def add_edge(self, u: int, v: int, capacity: float) -> int:
        index = len(self.to)
        self.adj[u].append(index)
        self.to.append(v)
        self.cap.append(capacity)
        self.adj[v].append(index + 1)
        self.to.append(u)
        self.cap.append(0.0)
        return index

    def _bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for e in self.adj[u]:
                v = self.to[e]
                if self.cap[e] > self.eps and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        return self.level[t] >= 0

    def _dfs(self, u: int, t: int, pushed: float) -> float:
        if u == t:
            return pushed
        while self.it[u] < len(self.adj[u]):
            e = self.adj[u][self.it[u]]
            v = self.to[e]
            if self.cap[e] > self.eps and self.level[v] == self.level[u] + 1:
                d = self._dfs(v, t, min(pushed, self.cap[e]))
                if d > self.eps:
                    self.cap[e] -= d
                    self.cap[e ^ 1] += d
                    return d
            self.it[u] += 1
        return 0.0

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        while self._bfs(s, t):
            self.it = [0] * self.n
            while True:
                pushed = self._dfs(s, t, _INF)
                if pushed <= self.eps:
                    break
                flow += pushed
        return flow

    def residual_reachable(self, s: int) -> List[bool]:
        seen = [False] * self.n
        seen[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for e in self.adj[u]:
                v = self.to[e]
                if self.cap[e] > self.eps and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        return seen


@dataclass
class _Flow:
    net: _Dinic
    starts: List[BeatState]
    terms: List[BeatState]
    s_idx: Dict[BeatState, int]
    t_idx: Dict[BeatState, int]
    source_edge: Dict[BeatState, int]
    sink_edge: Dict[BeatState, int]
    pair_edge: Dict[Tuple[BeatState, BeatState], int]
    total_start: float
    total_terminal: float
    flow: float


def _solve_flow(
    start_masses: Mapping[BeatState, float],
    terminal_masses: Mapping[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    *,
    capacity_epsilon: float,
    normalization_tolerance: float,
) -> _Flow:
    starts = sorted((s for s, m in start_masses.items() if m > 0.0), key=endpoint_state_sort_key)
    terms = sorted(
        (t for t, m in terminal_masses.items() if m > 0.0), key=endpoint_state_sort_key
    )
    total_s = float(sum(start_masses[s] for s in starts))
    total_t = float(sum(terminal_masses[t] for t in terms))
    if abs(total_s - total_t) > normalization_tolerance:
        raise ValueError(
            f"Endpoint masses must have equal totals (start={total_s}, terminal={total_t})."
        )
    source, sink = 0, 1
    s_idx = {s: 2 + i for i, s in enumerate(starts)}
    t_idx = {t: 2 + len(starts) + j for j, t in enumerate(terms)}
    net = _Dinic(2 + len(starts) + len(terms), capacity_epsilon)
    source_edge = {s: net.add_edge(source, s_idx[s], float(start_masses[s])) for s in starts}
    pair_edge: Dict[Tuple[BeatState, BeatState], int] = {}
    for s in starts:
        for t in relation.get(s, ()):
            if t in t_idx:
                pair_edge[(s, t)] = net.add_edge(s_idx[s], t_idx[t], _INF)
    sink_edge = {t: net.add_edge(t_idx[t], sink, float(terminal_masses[t])) for t in terms}
    flow = net.max_flow(source, sink)
    return _Flow(
        net, starts, terms, s_idx, t_idx, source_edge, sink_edge, pair_edge, total_s, total_t, flow
    )


def _tolerances_for_flow(tolerance: float, capacity_epsilon: float) -> EndpointTolerances:
    return EndpointTolerances(
        flow_tolerance=tolerance,
        capacity_epsilon=capacity_epsilon,
        marginal_tolerance=max(1e-6, tolerance),
    )


def global_flow_check(
    start_masses: Mapping[BeatState, float],
    terminal_masses: Mapping[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    *,
    tolerance: float = 1e-9,
    capacity_epsilon: float = 1e-14,
    normalization_tolerance: float = 1e-9,
) -> FlowCheck:
    """Level 2: can both marginals be met jointly through ``relation``?

    Network: source -> start (cap pi0[s]); start -> terminal (cap inf, only where
    ``relation`` joins them); terminal -> sink (cap piT[t]). For normalized marginals the
    pair is feasible iff max-flow == 1 within ``tolerance``.
    """
    _tolerances_for_flow(tolerance, capacity_epsilon)  # enforce the tolerance relations
    f = _solve_flow(
        start_masses, terminal_masses, relation,
        capacity_epsilon=capacity_epsilon, normalization_tolerance=normalization_tolerance,
    )
    deficit = max(0.0, min(f.total_start, f.total_terminal) - f.flow)
    if deficit <= tolerance:
        return FlowCheck(f.flow, deficit, True, tolerance, f.total_start, f.total_terminal)
    reach = f.net.residual_reachable(0)
    cut_starts = tuple(s for s in f.starts if reach[f.s_idx[s]])
    cut_terms = tuple(t for t in f.terms if reach[f.t_idx[t]])  # == N(cut_starts)
    return FlowCheck(
        flow_value=f.flow,
        deficit=deficit,
        feasible=False,
        tolerance=tolerance,
        total_start_mass=f.total_start,
        total_terminal_mass=f.total_terminal,
        violated_start_states=cut_starts,
        neighbor_terminal_states=cut_terms,
        start_mass_in_cut=float(sum(start_masses[s] for s in cut_starts)),
        terminal_mass_in_neighborhood=float(sum(terminal_masses[t] for t in cut_terms)),
    )


@dataclass(frozen=True)
class RoutableMasses:
    """Per-state mass carried by one maximum flow (<= requested, summing to ``flow_value``)."""

    flow_value: float
    start: Dict[BeatState, float]
    terminal: Dict[BeatState, float]


def routable_masses(
    start_masses: Mapping[BeatState, float],
    terminal_masses: Mapping[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    *,
    capacity_epsilon: float = 1e-14,
    normalization_tolerance: float = 1e-9,
) -> RoutableMasses:
    """Max-flow routable masses a_i <= pi0_i and b_j <= piT_j (each side sums to F*).

    F* is the maximum total mass any coupling on ``relation`` can carry, so
    ``1 - F*`` is the minimum total mass that must be trimmed from each side. The split
    among states is the deterministic Dinic split. Computed values at or below
    ``capacity_epsilon`` are float noise and are snapped to 0.
    """
    f = _solve_flow(
        start_masses, terminal_masses, relation,
        capacity_epsilon=capacity_epsilon, normalization_tolerance=normalization_tolerance,
    )

    def carried(masses, edges, states):
        out: Dict[BeatState, float] = {}
        for state in states:
            value = min(max(float(masses[state]) - f.net.cap[edges[state]], 0.0), masses[state])
            out[state] = value if value > capacity_epsilon else 0.0
        return out

    return RoutableMasses(
        flow_value=f.flow,
        start=carried(start_masses, f.source_edge, f.starts),
        terminal=carried(terminal_masses, f.sink_edge, f.terms),
    )


@dataclass(frozen=True)
class Tightness:
    """Whether a FEASIBLE endpoint pair lies on a Hall cut (is not strictly interior).

    ``forced_zero_pairs`` are the (start, terminal) pairs that carry zero mass in EVERY
    feasible coupling. A pair is strictly interior iff there are none.
    """

    tight: bool
    forced_zero_pairs: Tuple[Tuple[BeatState, BeatState], ...] = ()


def _strongly_connected_components(n: int, arcs: Sequence[Tuple[int, int]]) -> List[int]:
    """Deterministic iterative Kosaraju; returns a component id per node."""
    out: List[List[int]] = [[] for _ in range(n)]
    back: List[List[int]] = [[] for _ in range(n)]
    for u, v in arcs:
        out[u].append(v)
        back[v].append(u)
    order: List[int] = []
    seen = [False] * n
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        stack = [(root, 0)]
        while stack:
            node, i = stack.pop()
            if i < len(out[node]):
                stack.append((node, i + 1))
                nxt = out[node][i]
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append((nxt, 0))
            else:
                order.append(node)
    comp = [-1] * n
    cid = 0
    for root in reversed(order):
        if comp[root] >= 0:
            continue
        comp[root] = cid
        stack2 = [root]
        while stack2:
            node = stack2.pop()
            for nxt in back[node]:
                if comp[nxt] < 0:
                    comp[nxt] = cid
                    stack2.append(nxt)
        cid += 1
    return comp


def detect_tight_pairs(
    start_masses: Mapping[BeatState, float],
    terminal_masses: Mapping[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    *,
    capacity_epsilon: float = 1e-14,
    normalization_tolerance: float = 1e-9,
) -> Tightness:
    """Find (start, terminal) pairs forced to zero mass in every feasible coupling.

    Take a maximum flow f (value 1, so source and sink arcs are saturated and no residual
    cycle can pass through them). The pair arc i -> j can carry positive mass in some
    feasible coupling iff j can reach i in the residual graph, i.e. iff i and j lie in
    the same strongly connected component of {i -> j for every pair} + {j -> i where
    f_ij > 0}. Different components means the pair is forced to zero. Only meaningful for
    a feasible pair.
    """
    f = _solve_flow(
        start_masses, terminal_masses, relation,
        capacity_epsilon=capacity_epsilon, normalization_tolerance=normalization_tolerance,
    )
    s_node = {s: i for i, s in enumerate(f.starts)}
    t_node = {t: len(f.starts) + j for j, t in enumerate(f.terms)}
    arcs: List[Tuple[int, int]] = []
    for (s, t), edge in f.pair_edge.items():
        arcs.append((s_node[s], t_node[t]))
        if f.net.cap[edge ^ 1] > capacity_epsilon:  # flow on the pair
            arcs.append((t_node[t], s_node[s]))
    comp = _strongly_connected_components(len(f.starts) + len(f.terms), arcs)
    forced = tuple(
        pair
        for pair in sorted(
            f.pair_edge, key=lambda st: (endpoint_state_sort_key(st[0]), endpoint_state_sort_key(st[1]))
        )
        if comp[s_node[pair[0]]] != comp[t_node[pair[1]]]
    )
    return Tightness(tight=bool(forced), forced_zero_pairs=forced)


def restrict_graph_to_endpoint_support(
    graph: "SparseGraph",
    start_states: Sequence[BeatState],
    terminal_states: Sequence[BeatState],
) -> "SparseGraph":
    """Physically prune the graph to what lies on a retained start -> terminal path.

    Boundary layers keep only the given (positive-mass) states; every layer keeps only
    states that are both forward-reachable from a kept start and backward-reachable from
    a kept terminal; edges and edge diagnostics are filtered consistently.
    """
    layers = graph.layers
    n_edges = len(graph.edges_by_time)
    forward: list[set[BeatState]] = [set() for _ in layers]
    forward[0] = set(start_states) & set(layers[0].states)
    for t, edges in enumerate(graph.edges_by_time):
        forward[t + 1] = {e.target for e in edges if e.source in forward[t]}
    backward: list[set[BeatState]] = [set() for _ in layers]
    backward[-1] = set(terminal_states) & set(layers[-1].states)
    for t in reversed(range(n_edges)):
        backward[t] = {e.source for e in graph.edges_by_time[t] if e.target in backward[t + 1]}
    alive = [f & b for f, b in zip(forward, backward)]
    new_layers = tuple(
        Layer(layer.time_index, tuple(s for s in layer.states if s in alive[i]))
        for i, layer in enumerate(layers)
    )
    masks = [
        [e.source in alive[t] and e.target in alive[t + 1] for e in edges]
        for t, edges in enumerate(graph.edges_by_time)
    ]
    new_edges = tuple(
        tuple(e for e, keep in zip(edges, mask) if keep)
        for edges, mask in zip(graph.edges_by_time, masks)
    )
    new_edge_diag: tuple[tuple[EdgeScoreDiagnostics, ...], ...] = ()
    if graph.edge_diagnostics_by_time:
        new_edge_diag = tuple(
            tuple(d for d, keep in zip(diags, mask) if keep)
            for diags, mask in zip(graph.edge_diagnostics_by_time, masks)
        )
    diagnostics = replace(
        graph.diagnostics, layer_sizes=tuple(len(layer.states) for layer in new_layers)
    )
    return replace(
        graph,
        layers=new_layers,
        edges_by_time=new_edges,
        diagnostics=diagnostics,
        edge_diagnostics_by_time=new_edge_diag,
    )


# --------------------------------------------------------------------------- #
# Level 1 and shared helpers
# --------------------------------------------------------------------------- #
def _positive(masses: Mapping[BeatState, float]) -> List[BeatState]:
    """Strictly positive states (``mass > 0.0``) in stable order. No epsilon."""
    return sorted((s for s, m in masses.items() if m > 0.0), key=endpoint_state_sort_key)


def _find_unreachable(
    p: Mapping[BeatState, float],
    q: Mapping[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    start_layer: frozenset,
    end_layer: frozenset,
) -> Tuple[List[Tuple[BeatState, str]], List[Tuple[BeatState, str]]]:
    pos_s = _positive(p)
    pos_t = _positive(q)
    pos_t_set = set(pos_t)
    reached: set[BeatState] = set()
    for s in pos_s:
        if s in start_layer:
            reached.update(relation.get(s, ()))
    bad_s: List[Tuple[BeatState, str]] = []
    for s in pos_s:
        if s not in start_layer:
            bad_s.append((s, "not_in_graph_start_layer"))
        elif not any(t in pos_t_set for t in relation.get(s, ())):
            bad_s.append((s, "no_retained_path_to_positive_mass_terminal"))
    bad_t: List[Tuple[BeatState, str]] = []
    for t in pos_t:
        if t not in end_layer:
            bad_t.append(
                (
                    t,
                    "not_retained_in_graph_terminal_layer (no retained edge reaches it; "
                    "check d_max, k_max and transition legality)",
                )
            )
        elif t not in reached:
            bad_t.append((t, "no_retained_path_from_positive_mass_start"))
    return bad_s, bad_t


def _normalize(masses: Dict[BeatState, float]) -> Dict[BeatState, float]:
    total = float(sum(masses.values()))
    if not total > 0.0:
        raise EndpointFeasibilityError("No positive endpoint mass remains to normalize.")
    return {s: m / total for s, m in masses.items()}


def _describe(states: Sequence[Tuple[BeatState, float, str]], limit: int = 5) -> str:
    parts = [f"{s.pretty()} mass={m:.6f} ({r})" for s, m, r in states[:limit]]
    if len(states) > limit:
        parts.append(f"... +{len(states) - limit} more")
    return "; ".join(parts)


def _drop_unreachable(
    p: Dict[BeatState, float],
    q: Dict[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    start_layer: frozenset,
    end_layer: frozenset,
    req_p: Mapping[BeatState, float],
    req_q: Mapping[BeatState, float],
    removed: List[RemovedEndpointState],
    *,
    stage: str,
    iteration: int,
) -> None:
    """Remove individually unreachable positive-mass states until stable."""
    while True:
        bad_s, bad_t = _find_unreachable(p, q, relation, start_layer, end_layer)
        if not bad_s and not bad_t:
            return
        for state, reason in bad_s:
            removed.append(
                RemovedEndpointState(SIDE_START, state, req_p[state], reason, stage, iteration)
            )
            del p[state]
        for state, reason in bad_t:
            removed.append(
                RemovedEndpointState(SIDE_TERMINAL, state, req_q[state], reason, stage, iteration)
            )
            del q[state]
        if not _positive(p) or not _positive(q):
            return


# --------------------------------------------------------------------------- #
# Level 2 fallbacks
# --------------------------------------------------------------------------- #
def _cut_candidates(
    check: FlowCheck,
    p: Mapping[BeatState, float],
    q: Mapping[BeatState, float],
) -> List[Tuple[str, BeatState, float]]:
    """States contributing to the violated Hall set: starts in X (over-supplied) and
    positive terminals outside N(X) (they absorb mass X cannot reach)."""
    cands: List[Tuple[str, BeatState, float]] = []
    for s in check.violated_start_states:
        if p.get(s, 0.0) > 0.0:
            cands.append((SIDE_START, s, p[s]))
    neighbors = set(check.neighbor_terminal_states)
    for t in _positive(q):
        if t not in neighbors:
            cands.append((SIDE_TERMINAL, t, q[t]))
    return cands


def _select_removal(
    cands: Sequence[Tuple[str, BeatState, float]],
) -> Tuple[str, BeatState, float]:
    """Whole-state rule (heuristic, no optimality claim): remove the candidate with the
    least requested-mass loss; ties -> start before terminal -> ascending state key."""
    return min(
        cands,
        key=lambda c: (round(c[2], _ROUND_MASS), _SIDE_RANK[c[0]], endpoint_state_sort_key(c[1])),
    )


def _trim_to_routable(
    p: Dict[BeatState, float],
    q: Dict[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    check: FlowCheck,
    req_p: Mapping[BeatState, float],
    req_q: Mapping[BeatState, float],
    iteration: int,
    removed: List[RemovedEndpointState],
    trims: List[TrimOperation],
    tol: EndpointTolerances,
) -> Tuple[
    Dict[BeatState, float], Dict[BeatState, float], Dict[BeatState, float], Dict[BeatState, float]
]:
    """Optimal-for-retained-mass trim: keep the max-flow routable masses, renormalize.

    Each side keeps a_i <= pi0_i (b_j <= piT_j) summing to F*; both sides are divided by
    F*, which makes the pair exactly feasible (the flow itself is a coupling). A state
    whose routable mass is zero is recorded as REMOVED (not as a trim to zero).
    """
    rm = routable_masses(
        p, q, relation,
        capacity_epsilon=tol.capacity_epsilon,
        normalization_tolerance=tol.normalization_tolerance,
    )
    if not rm.flow_value > tol.capacity_epsilon:
        raise EndpointFeasibilityError(
            "Partial trimming cannot make progress: max-flow routes no mass.", flow_check=check
        )
    kept: Dict[str, Dict[BeatState, float]] = {SIDE_START: {}, SIDE_TERMINAL: {}}
    pending: List[Tuple[str, BeatState, float, float]] = []
    for side, masses, routed, req in (
        (SIDE_START, p, rm.start, req_p),
        (SIDE_TERMINAL, q, rm.terminal, req_q),
    ):
        for state in _positive(masses):
            mass, carried = masses[state], routed.get(state, 0.0)
            if not carried > 0.0:
                removed.append(
                    RemovedEndpointState(
                        side, state, req[state],
                        f"partial_trim_zeroed: routable mass is 0 of {mass:.6f} "
                        f"(max-flow={rm.flow_value:.6f})",
                        "partial_trim_zeroed", iteration,
                    )
                )
                continue
            kept[side][state] = carried
            if mass - carried > tol.capacity_epsilon:
                pending.append((side, state, mass, carried))
    if not kept[SIDE_START] or not kept[SIDE_TERMINAL]:
        raise EndpointFeasibilityError("Partial trimming removed all mass on a side.", flow_check=check)
    new_p, new_q = _normalize(kept[SIDE_START]), _normalize(kept[SIDE_TERMINAL])
    for side, state, mass, carried in pending:
        trims.append(
            TrimOperation(
                iteration=iteration,
                side=side,
                state=state,
                original_mass=mass,
                trimmed_mass=mass - carried,
                resulting_mass=carried,
                effective_mass=(new_p if side == SIDE_START else new_q)[state],
                reason=(
                    f"max-flow routable-mass trim (F*={rm.flow_value:.6f}, "
                    f"deficit={check.deficit:.6f})"
                ),
                flow_before=check.flow_value,
                deficit_before=check.deficit,
            )
        )
    return new_p, new_q, dict(kept[SIDE_START]), dict(kept[SIDE_TERMINAL])


def _blend_interior(
    p: Dict[BeatState, float],
    q: Dict[BeatState, float],
    relation: Mapping[BeatState, Sequence[BeatState]],
    margin: float,
) -> Tuple[Dict[BeatState, float], Dict[BeatState, float], bool]:
    """Blend toward the marginals of the uniform coupling over all relation pairs.

    A trimmed pair sits exactly on a Hall cut: some pairs are forced to zero mass and the
    SB scaling then converges only sub-linearly. The uniform-over-pairs coupling has every
    pair strictly positive, so (1 - m) * trimmed + m * uniform is feasible (convexity) and
    strictly interior, restoring linear convergence at a cost of at most ``m`` in total
    variation per side. Only strictly positive states take part, so no zero-mass state
    gains mass.
    """
    ps, qs = _positive(p), _positive(q)
    qset = set(qs)
    pairs = [(s, t) for s in ps for t in relation.get(s, ()) if t in qset]
    if not pairs:
        return p, q, False
    ip = {s: 0.0 for s in ps}
    iq = {t: 0.0 for t in qs}
    for s, t in pairs:
        ip[s] += 1.0 / len(pairs)
        iq[t] += 1.0 / len(pairs)
    if any(not v > 0.0 for v in ip.values()) or any(not v > 0.0 for v in iq.values()):
        return p, q, False
    return (
        {s: (1.0 - margin) * p[s] + margin * ip[s] for s in ps},
        {t: (1.0 - margin) * q[t] + margin * iq[t] for t in qs},
        True,
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _aligned(layer: Layer, masses: Mapping[BeatState, float]) -> EndpointDistribution:
    raw = [max(0.0, float(masses.get(s, 0.0))) for s in layer.states]
    total = sum(raw)
    return EndpointDistribution(layer=layer, probabilities=tuple(m / total for m in raw))


def resolve_endpoint_distributions(
    graph: "SparseGraph",
    start_requested: EndpointDistribution,
    end_requested: EndpointDistribution,
    *,
    policy: EndpointFallbackPolicy = EndpointFallbackPolicy.STRICT,
    tolerances: Optional[EndpointTolerances] = None,
    flow_tolerance: Optional[float] = None,
    max_iterations: int = 100,
    interior_margin: float = 0.0,
) -> EndpointResolution:
    """Level 1 + Level 2 feasibility, policy-dependent fallback, graph restriction.

    Runs entirely before ``build_sb_problem`` / ``solve_sb``. See the module docstring for
    the policy semantics. ``interior_margin`` (0 disables) is applied, and reported, to any
    final pair that is *tight* (some pair forced to zero mass in every feasible coupling,
    e.g. after a trim or a removal, or already as requested) under every policy except
    STRICT, which never changes the request. It restores linear SB convergence.
    """
    if not isinstance(policy, EndpointFallbackPolicy):
        raise TypeError("policy must be an EndpointFallbackPolicy.")
    if not 0.0 <= interior_margin < 1.0:
        raise ValueError("interior_margin must be in [0, 1).")
    if tolerances is None:
        tolerances = EndpointTolerances(
            flow_tolerance=1e-9 if flow_tolerance is None else flow_tolerance
        )
    tol = tolerances
    drops_level1 = policy in (
        EndpointFallbackPolicy.DROP_UNREACHABLE_ONLY,
        EndpointFallbackPolicy.DROP_AND_RENORMALIZE,
        EndpointFallbackPolicy.CONDITIONAL_DROP_THEN_TRIM,
    )
    # Policies that never alter masses of retained states for Level 2 / tightness reasons.
    mass_preserving = policy in (
        EndpointFallbackPolicy.STRICT,
        EndpointFallbackPolicy.DROP_UNREACHABLE_ONLY,
    )
    start_layer, end_layer = graph.layers[0], graph.layers[-1]
    start_set, end_set = frozenset(start_layer.states), frozenset(end_layer.states)
    req_p = {
        s: float(m) for s, m in zip(start_requested.layer.states, start_requested.probabilities)
        if m > 0.0
    }
    req_q = {
        s: float(m) for s, m in zip(end_requested.layer.states, end_requested.probabilities)
        if m > 0.0
    }
    relation = compute_endpoint_reachability(
        graph,
        [s for s in req_p if s in start_set],
        [t for t in req_q if t in end_set],
    )
    p, q = dict(req_p), dict(req_q)
    removed: List[RemovedEndpointState] = []
    trims: List[TrimOperation] = []
    decisions: List[str] = []
    history: List[FlowCheck] = []
    margin_applied = False

    # ---- Level 1: individual reachability -------------------------------- #
    bad_s, bad_t = _find_unreachable(p, q, relation, start_set, end_set)
    if bad_s or bad_t:
        if not drops_level1:
            items = [(s, req_p[s], r) for s, r in bad_s] + [(t, req_q[t], r) for t, r in bad_t]
            hint = (
                "Use DROP_UNREACHABLE_ONLY, DROP_AND_RENORMALIZE or CONDITIONAL_DROP_THEN_TRIM "
                "(PARTIAL_TRIM only trims mass and never removes unreachable states), or "
                "widen d_max/k_max."
                if policy is EndpointFallbackPolicy.PARTIAL_TRIM_AND_RENORMALIZE
                else "Use DROP_UNREACHABLE_ONLY / DROP_AND_RENORMALIZE / "
                "CONDITIONAL_DROP_THEN_TRIM or widen d_max/k_max."
            )
            raise EndpointReachabilityError(
                "Level 1 (individual reachability) failed: "
                f"{len(bad_s)} start and {len(bad_t)} terminal states are unreachable "
                f"(start mass {sum(req_p[s] for s, _ in bad_s):.6f}, terminal mass "
                f"{sum(req_q[t] for t, _ in bad_t):.6f}): {_describe(items)}. {hint}",
                states=[s for s, _ in bad_s] + [t for t, _ in bad_t],
            )
        _drop_unreachable(
            p, q, relation, start_set, end_set, req_p, req_q, removed,
            stage="individual_reachability", iteration=0,
        )
        decisions.append(
            f"L1: removed {len(removed)} individually unreachable states "
            f"({sum(r.requested_mass for r in removed if r.side == SIDE_START):.6f} start mass, "
            f"{sum(r.requested_mass for r in removed if r.side == SIDE_TERMINAL):.6f} terminal mass)"
        )
    if not _positive(p) or not _positive(q):
        raise EndpointFeasibilityError(
            "No feasible endpoint support remains after removing individually unreachable states."
        )
    ret_p, ret_q = dict(p), dict(q)
    p, q = _normalize(ret_p), _normalize(ret_q)

    # ---- Level 2: global feasibility (+ fallback) ------------------------ #
    iteration = 0
    while True:
        check = global_flow_check(
            p, q, relation,
            tolerance=tol.flow_tolerance,
            capacity_epsilon=tol.capacity_epsilon,
            normalization_tolerance=tol.normalization_tolerance,
        )
        history.append(check)
        if trims and trims[-1].flow_after is None:
            trims[:] = [
                replace(t, flow_after=check.flow_value, feasible_after=check.feasible)
                if t.iteration == iteration and t.flow_after is None else t
                for t in trims
            ]
        if check.feasible:
            break
        if mass_preserving:
            affected = [(s, p[s], "in violated start set") for s in check.violated_start_states]
            raise EndpointMassInfeasibleError(
                "Level 2 (global distribution feasibility) failed: max-flow="
                f"{check.flow_value:.6f}, feasibility deficit={check.deficit:.6f}. "
                f"Violated start set X has pi0(X)={check.start_mass_in_cut:.6f} but its reachable "
                f"terminals N(X) carry only piT(N(X))={check.terminal_mass_in_neighborhood:.6f}. "
                f"Starts in X: {_describe(affected)}; terminals in N(X): "
                f"{_describe([(t, q[t], 'reachable from X') for t in check.neighbor_terminal_states])}. "
                f"Policy {policy.value}: masses of retained states are unchanged"
                + (
                    " (individually unreachable states, if any, were already removed whole). "
                    if policy is EndpointFallbackPolicy.DROP_UNREACHABLE_ONLY
                    else ". "
                )
                + "Choose DROP_AND_RENORMALIZE, PARTIAL_TRIM_AND_RENORMALIZE or "
                "CONDITIONAL_DROP_THEN_TRIM explicitly to allow a modified distribution.",
                flow_check=check,
                states=list(check.violated_start_states) + list(check.neighbor_terminal_states),
            )
        if iteration >= max_iterations:
            raise EndpointFeasibilityError(
                f"Fallback reached max_iterations={max_iterations} without a feasible "
                f"endpoint pair (last max-flow={check.flow_value:.6f}, "
                f"deficit={check.deficit:.6f}).",
                flow_check=check,
            )
        iteration += 1

        if policy is EndpointFallbackPolicy.DROP_AND_RENORMALIZE:
            side, state, mass = _select_removal(_cut_candidates(check, p, q))
            removed.append(
                RemovedEndpointState(
                    side, state,
                    (req_p if side == SIDE_START else req_q)[state],
                    f"min-cut candidate (deficit={check.deficit:.6f}); least mass in cut",
                    "global_feasibility", iteration,
                )
            )
            del (p if side == SIDE_START else q)[state]
            decisions.append(
                f"iter {iteration}: removed {side} {state.pretty()} (mass {mass:.6f}); "
                f"flow was {check.flow_value:.6f}"
            )
            _drop_unreachable(
                p, q, relation, start_set, end_set, req_p, req_q, removed,
                stage="fallback_cascade", iteration=iteration,
            )
            if not _positive(p) or not _positive(q):
                raise EndpointFeasibilityError(
                    "Whole-state removal exhausted the endpoint support: no feasible retained "
                    "support exists under DROP_AND_RENORMALIZE (greedy removal; this does not "
                    "prove that no other subset is feasible).",
                    flow_check=check,
                )
            ret_p, ret_q = {s: ret_p[s] for s in p}, {t: ret_q[t] for t in q}
            p, q = _normalize(ret_p), _normalize(ret_q)
        else:  # PARTIAL_TRIM_AND_RENORMALIZE / CONDITIONAL_DROP_THEN_TRIM: Level 2 -> trim
            before = len(removed)
            scale_p, scale_q = sum(ret_p.values()), sum(ret_q.values())
            p, q, kept_p, kept_q = _trim_to_routable(
                p, q, relation, check, req_p, req_q, iteration, removed, trims, tol
            )
            # routed masses are in the current normalized units; convert back to requested
            ret_p = {s: a * scale_p for s, a in kept_p.items()}
            ret_q = {t: b * scale_q for t, b in kept_q.items()}
            decisions.append(
                f"iter {iteration}: routable-mass trim (max-flow {check.flow_value:.6f}, "
                f"deficit {check.deficit:.6f}); {len(removed) - before} states zeroed, "
                f"{sum(1 for t in trims if t.iteration == iteration)} states trimmed"
            )

    # ---- Tightness: a feasible pair on a Hall cut converges only like 1/k ---------- #
    tightness = detect_tight_pairs(
        p, q, relation,
        capacity_epsilon=tol.capacity_epsilon,
        normalization_tolerance=tol.normalization_tolerance,
    )
    if tightness.tight and interior_margin > 0.0:
        if mass_preserving:
            # STRICT / DROP_UNREACHABLE_ONLY never change retained masses; the solve may
            # report non-convergence.
            decisions.append(
                f"pair is tight ({len(tightness.forced_zero_pairs)} forced-zero pairs); "
                "STRICT leaves it unblended"
            )
        else:
            p, q, margin_applied = _blend_interior(p, q, relation, interior_margin)
            decisions.append(
                f"tight pair ({len(tightness.forced_zero_pairs)} forced-zero pairs): interior "
                f"blend margin={interior_margin:g} "
                f"{'applied' if margin_applied else 'skipped (no certified interior)'}"
            )
            if margin_applied:
                recheck = global_flow_check(
                    p, q, relation,
                    tolerance=tol.flow_tolerance,
                    capacity_epsilon=tol.capacity_epsilon,
                    normalization_tolerance=tol.normalization_tolerance,
                )
                history.append(recheck)
                if not recheck.feasible:  # convexity guarantees feasibility; defensive
                    raise EndpointFeasibilityError(
                        "Internal error: interior blend produced an infeasible pair "
                        f"(max-flow={recheck.flow_value:.6f}).",
                        flow_check=recheck,
                    )

    # ---- Restricted graph and reports ------------------------------------ #
    restricted = restrict_graph_to_endpoint_support(graph, _positive(p), _positive(q))
    for side, layer, keep in (
        (SIDE_START, restricted.layers[0], _positive(p)),
        (SIDE_TERMINAL, restricted.layers[-1], _positive(q)),
    ):
        if len(layer.states) != len(keep):
            raise EndpointFeasibilityError(
                f"Internal invariant violated: restricted {side} layer has "
                f"{len(layer.states)} states for {len(keep)} positive-mass endpoints."
            )
    modifications: List[str] = []
    if any(r.stage == "individual_reachability" for r in removed):
        modifications.append("individual_removal")
    if any(r.stage == "global_feasibility" for r in removed):
        modifications.append("whole_state_removal")
    if trims or any(r.stage == "partial_trim_zeroed" for r in removed):
        modifications.append("partial_trim")
    if margin_applied:
        modifications.append("interior_blend")
    if "whole_state_removal" in modifications:
        reason = "feasible_after_whole_state_removal"
    elif "partial_trim" in modifications:
        reason = (
            "feasible_after_individual_removal_and_partial_trim"
            if "individual_removal" in modifications
            else "feasible_after_partial_trim"
        )
    elif "individual_removal" in modifications:
        reason = "feasible_after_individual_removal"
    else:
        reason = "feasible_as_requested"
    if margin_applied:  # never report a blended pair as exact
        reason = (
            "feasible_after_interior_blend"
            if reason == "feasible_as_requested"
            else f"{reason}_with_interior_blend"
        )

    def _report(side, requested, layer, req, eff, ret) -> EndpointMassReport:
        changes = tuple(
            ProbabilityChange(s, req[s], eff.get(s, 0.0))
            for s in sorted(req, key=endpoint_state_sort_key)
        )
        return EndpointMassReport(
            side=side,
            requested=requested,
            effective=_aligned(layer, eff),
            requested_mass=float(sum(req.values())),
            removed=tuple(r for r in removed if r.side == side),
            trims=tuple(t for t in trims if t.side == side),
            probability_changes=changes,
            interior_margin=interior_margin if margin_applied else 0.0,
            retained_mass=float(sum(ret.values())),
            trimmed_requested_mass=float(sum(max(0.0, req[s] - ret[s]) for s in ret)),
        )

    return EndpointResolution(
        policy=policy,
        start=_report(SIDE_START, start_requested, restricted.layers[0], req_p, p, ret_p),
        end=_report(SIDE_TERMINAL, end_requested, restricted.layers[-1], req_q, q, ret_q),
        initial_flow_check=history[0],
        final_flow_check=history[-1],
        feasibility_history=tuple(history),
        iterations=iteration,
        termination_reason=reason,
        decisions=tuple(decisions),
        graph=restricted,
        layer_sizes_before=tuple(len(layer.states) for layer in graph.layers),
        layer_sizes_after=tuple(len(layer.states) for layer in restricted.layers),
        tightness=tightness,
        request_modified=bool(modifications),
        modifications=tuple(modifications),
    )


def resolve_endpoint_support(
    graph: "SparseGraph",
    start_requested: EndpointDistribution,
    end_requested: EndpointDistribution,
    *,
    policy: EndpointFallbackPolicy = EndpointFallbackPolicy.STRICT,
    flow_tolerance: float = 1e-9,
    max_iterations: int = 100,
    interior_margin: float = 0.0,
) -> tuple[EndpointMassReport, EndpointMassReport]:
    """Compatibility wrapper returning only the two mass reports."""
    resolution = resolve_endpoint_distributions(
        graph, start_requested, end_requested,
        policy=policy, flow_tolerance=flow_tolerance,
        max_iterations=max_iterations, interior_margin=interior_margin,
    )
    return resolution.start, resolution.end


# --------------------------------------------------------------------------- #
# Solution validation and the validated solve
# --------------------------------------------------------------------------- #
def validate_realized_endpoint_marginals(
    report: EndpointMassReport,
    *,
    tolerance: float,
    trace: Optional[object] = None,
) -> None:
    """Single-side check kept for compatibility (see ``validate_solution_marginals``)."""
    error = report.max_abs_realized_error
    if error is None:
        raise EndpointMarginalError(
            f"{report.side} marginal was not measured; solution has no marginals."
        )
    if error > tolerance:
        raise EndpointMarginalError(
            f"Realized {report.side} marginal deviates from target by {error:.3e} "
            f"(> endpoint_marginal_tolerance={tolerance:.3e}).",
            start_error=error if report.side == SIDE_START else None,
            terminal_error=error if report.side == SIDE_TERMINAL else None,
        )


def validate_solution_marginals(
    solution: SBSolution,
    resolution: EndpointResolution,
    *,
    tolerance: float,
) -> Tuple[EndpointMassReport, EndpointMassReport]:
    """Check convergence and BOTH realized marginals against the effective targets.

    Returns the reports with realized marginals attached, or raises. Global feasibility
    proves a coupling *exists*; this proves the solver found it. A returned ``SBSolution``
    with ``trace.converged == False`` is rejected here, never reported as success.
    """
    if solution.marginals is None:
        raise EndpointValidationError("SB solution carries no marginals to validate.")
    start = resolution.start.with_realized(solution.marginals.node_marginals_by_layer[0])
    end = resolution.end.with_realized(solution.marginals.node_marginals_by_layer[-1])
    e0, eT = start.max_abs_realized_error, end.max_abs_realized_error
    if e0 is None or eT is None:
        raise EndpointValidationError("Realized endpoint marginal errors were not measured.")
    trace = solution.trace
    summary = (
        f"start_error={e0:.3e}, terminal_error={eT:.3e}, "
        f"endpoint_marginal_tolerance={tolerance:.3e}; solver iterations={trace.iterations}, "
        f"converged={trace.converged}, final_max_delta={trace.final_max_delta:.3e}"
    )
    if not trace.converged:
        raise EndpointSolverNonConvergenceError(
            f"SB solver did not converge ({summary}). Increase SBConfig.max_iterations or "
            "relax SBConfig.tolerance. Pairs sitting exactly on a Hall cut converge slowly: "
            "use DROP_AND_RENORMALIZE / PARTIAL_TRIM / CONDITIONAL with interior_margin > 0 "
            "(STRICT and DROP_UNREACHABLE_ONLY never blend).",
            start_error=e0,
            terminal_error=eT,
        )
    failed = [name for name, err in (("start", e0), ("terminal", eT)) if err > tolerance]
    if failed:
        raise EndpointMarginalError(
            f"Realized {' and '.join(failed)} marginal(s) deviate from the effective "
            f"target ({summary}).",
            start_error=e0,
            terminal_error=eT,
        )
    return start, end


def solve_endpoint_bridge(
    graph: "SparseGraph",
    start_requested: EndpointDistribution,
    end_requested: EndpointDistribution,
    *,
    sb_config: Optional[SBConfig] = None,
    policy: EndpointFallbackPolicy = EndpointFallbackPolicy.STRICT,
    flow_tolerance: float = 1e-9,
    max_fallback_iterations: int = 100,
    marginal_tolerance: float = 1e-6,
    capacity_epsilon: float = 1e-14,
    interior_margin: float = 0.0,
) -> EndpointSolveResult:
    """feasibility -> restricted graph -> build_sb_problem -> solve_sb -> validation.

    Path extraction is the caller's job and must only happen after this returns: every
    failure (infeasibility, non-convergence, marginal error) raises.
    """
    tolerances = EndpointTolerances(
        flow_tolerance=flow_tolerance,
        marginal_tolerance=marginal_tolerance,
        capacity_epsilon=capacity_epsilon,
    )
    resolution = resolve_endpoint_distributions(
        graph, start_requested, end_requested,
        policy=policy, tolerances=tolerances,
        max_iterations=max_fallback_iterations,
        interior_margin=interior_margin,
    )
    restricted_graph = resolution.graph
    if restricted_graph is None:
        raise EndpointValidationError("Endpoint resolution produced no restricted graph.")
    problem = build_sb_problem(
        restricted_graph, resolution.start.effective, resolution.end.effective,
        sb_config=sb_config,
    )
    solution = solve_sb(problem)
    start_mass, end_mass = validate_solution_marginals(
        solution, resolution, tolerance=tolerances.marginal_tolerance
    )
    return EndpointSolveResult(
        graph=restricted_graph,
        resolution=resolution,
        problem=problem,
        solution=solution,
        bridge=solution.to_bridge(),
        start_mass=start_mass,
        end_mass=end_mass,
    )
