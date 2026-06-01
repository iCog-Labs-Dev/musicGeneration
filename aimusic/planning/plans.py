from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from aimusic.core.config import (
    DecodeConfig,
    NeuralPriorConfig,
    PlanConfig,
    PlanMethod,
    PriorWeights,
    SBConfig,
    SectioningStrategy,
    StyleConfig,
)
from aimusic.core.core_types import BeatState, EndpointDistribution, Layer
from aimusic.core.rng import RNGKey, random_unit
from aimusic.core.vocab import DEFAULT_VOCABULARIES, Vocabularies, build_default_vocabularies
from aimusic.planning.graph import SparseGraph, StitchAnchor, build_sparse_graph
from aimusic.planning.sb import (
    SBProblem,
    SBSolution,
    SampledBridgePath,
    SolvedBridge,
    build_sb_problem,
    map_bridge_path,
    sample_bridge_path,
    solve_sb,
)
from aimusic.scoring.priors import NullPrior, Prior
from aimusic.theory.tonal import get_fifth_steps


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float = 0.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _state_sort_key(state: BeatState) -> tuple[int, int, int, int, int, int, int, int]:
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
class PlanningSection:
    """Single section descriptor for structural planning diagnostics."""

    name: str
    start_time: int
    end_time: int
    boundary_level: int
    target_tension_arc: Tuple[float, ...] = (0.2, 0.85, 0.25)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        _require_int("start_time", self.start_time, minimum=0)
        _require_int("end_time", self.end_time, minimum=1)
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be > start_time.")
        _require_int("boundary_level", self.boundary_level, minimum=0)
        arc = tuple(float(item) for item in self.target_tension_arc)
        if len(arc) < 2:
            raise ValueError("target_tension_arc must contain at least two values.")
        for idx, value in enumerate(arc):
            _require_real(f"target_tension_arc[{idx}]", value)
        object.__setattr__(self, "target_tension_arc", arc)


@dataclass(frozen=True)
class MethodARunConfig:
    """Pure run configuration bundle for EPIC 6 Method A orchestration."""

    total_beats: int
    seed: int = 0
    use_sampling: bool = False
    style_config: StyleConfig = field(default_factory=StyleConfig)
    prior_weights: PriorWeights = field(default_factory=PriorWeights)
    sb_config: Optional[SBConfig] = None
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)
    plan_config: PlanConfig = field(default_factory=PlanConfig)
    neural_prior_config: NeuralPriorConfig = field(default_factory=NeuralPriorConfig)
    edo: int = 12

    def __post_init__(self) -> None:
        _require_int("total_beats", self.total_beats, minimum=1)
        _require_int("seed", self.seed, minimum=0)
        if not isinstance(self.use_sampling, bool):
            raise TypeError("use_sampling must be a bool.")
        if self.plan_config.method is not PlanMethod.METHOD_A:
            raise ValueError("MethodARunConfig requires plan_config.method == METHOD_A.")
        _require_int("edo", self.edo, minimum=1)
        if self.sb_config is not None and self.sb_config.horizon_t != self.total_beats:
            raise ValueError("sb_config.horizon_t must equal total_beats for Method A runs.")
        if (
            self.plan_config.sectioning_strategy is SectioningStrategy.SECTION_WISE
            and len(self.plan_config.section_names) > self.total_beats
        ):
            raise ValueError(
                "SECTION_WISE planning requires total_beats >= len(section_names)."
            )


@dataclass(frozen=True)
class EndpointChoice:
    """Explicit chosen endpoint state plus provenance within a candidate distribution."""

    state: BeatState
    source_distribution: EndpointDistribution
    selected_index: int
    selected_probability: float
    selection_mode: str

    def __post_init__(self) -> None:
        if self.state not in self.source_distribution.layer.states:
            raise ValueError("state must belong to source_distribution.layer.")
        _require_int("selected_index", self.selected_index, minimum=0)
        if self.selected_index >= len(self.source_distribution.layer.states):
            raise ValueError("selected_index must be within source_distribution support.")
        _require_real("selected_probability", self.selected_probability)
        if not isinstance(self.selection_mode, str) or not self.selection_mode.strip():
            raise ValueError("selection_mode must be a non-empty string.")


@dataclass(frozen=True)
class MethodAEndpoints:
    """Endpoint distributions and section metadata for a Method A run."""

    pi0: EndpointDistribution
    piT: EndpointDistribution
    start_choice: EndpointChoice
    end_choice: EndpointChoice
    sections: Tuple[PlanningSection, ...]


@dataclass(frozen=True)
class MethodAPlanDiagnostics:
    """Inspectable diagnostics emitted by Method A orchestration."""

    section_tags: Tuple[str, ...]
    target_tension_arcs: Tuple[Tuple[float, ...], ...]
    chosen_start_state: BeatState
    chosen_end_state: BeatState
    endpoint_selection_mode: str
    chosen_start_probability: float
    chosen_end_probability: float
    path_mode: str
    graph_layer_sizes: Tuple[int, ...]
    bridge_iterations: int
    bridge_converged: bool


@dataclass(frozen=True)
class MethodAPlanResult:
    """Full output of a Method A planning pass."""

    run_config: MethodARunConfig
    vocabularies: Vocabularies
    endpoints: MethodAEndpoints
    graph: SparseGraph
    sb_problem: SBProblem
    sb_solution: SBSolution
    bridge: SolvedBridge
    path: Tuple[BeatState, ...]
    path_score: Optional[float]
    sampled_path: Optional[SampledBridgePath]
    diagnostics: MethodAPlanDiagnostics


def _resolved_vocabs(
    vocabularies: Optional[Vocabularies],
    style_config: StyleConfig,
) -> Vocabularies:
    if vocabularies is not None:
        return vocabularies
    if style_config == StyleConfig():
        return DEFAULT_VOCABULARIES
    return build_default_vocabularies(style_config)


def _resolved_sb_config(run_config: MethodARunConfig) -> SBConfig:
    if run_config.sb_config is not None:
        return run_config.sb_config
    return SBConfig(horizon_t=run_config.total_beats)


def _numpy_generator_from_key(key: RNGKey) -> np.random.Generator:
    seed = key.generator().randrange(0, 2**63)
    return np.random.default_rng(seed)


def _softmax(scores: Sequence[float], temperature: float) -> Tuple[float, ...]:
    logits = np.asarray(tuple(float(score) for score in scores), dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence.")
    scaled = logits / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    normalized = weights / np.sum(weights)
    return tuple(float(value) for value in normalized)


def _align_endpoint_distribution(
    endpoint: EndpointDistribution,
    layer: Layer,
) -> EndpointDistribution:
    masses = [endpoint.probability_of(state) for state in layer.states]
    total = float(sum(masses))
    if total <= 0.0:
        raise ValueError("Endpoint support vanished after graph construction.")
    return EndpointDistribution(
        layer=layer,
        probabilities=tuple(mass / total for mass in masses),
    )


def _singleton_endpoint_distribution(state: BeatState, *, time_index: int) -> EndpointDistribution:
    return EndpointDistribution(
        layer=Layer(time_index=time_index, states=(state,)),
        probabilities=(1.0,),
    )


def _sample_index_from_distribution(
    endpoint: EndpointDistribution,
    key: RNGKey,
) -> tuple[int, RNGKey]:
    threshold, next_key = random_unit(key)
    running = 0.0
    for idx, probability in enumerate(endpoint.probabilities):
        running += probability
        if threshold <= running:
            return idx, next_key
    return len(endpoint.probabilities) - 1, next_key


def _choose_endpoint_state(
    endpoint: EndpointDistribution,
    *,
    key: RNGKey,
    sample: bool,
) -> tuple[EndpointChoice, RNGKey]:
    if sample:
        selected_index, next_key = _sample_index_from_distribution(endpoint, key)
        selection_mode = "sample"
    else:
        selected_index = max(
            range(len(endpoint.probabilities)),
            key=lambda idx: (endpoint.probabilities[idx], -idx),
        )
        next_key = key
        selection_mode = "argmax"
    return (
        EndpointChoice(
            state=endpoint.layer.states[selected_index],
            source_distribution=endpoint,
            selected_index=selected_index,
            selected_probability=endpoint.probabilities[selected_index],
            selection_mode=selection_mode,
        ),
        next_key,
    )


def build_section_plan(run_config: MethodARunConfig) -> Tuple[PlanningSection, ...]:
    plan_config = run_config.plan_config
    if plan_config.sectioning_strategy is SectioningStrategy.SINGLE_PASS:
        name = (
            plan_config.section_names[0]
            if plan_config.section_names
            else "method_a_single_pass"
        )
        return (
            PlanningSection(
                name=name,
                start_time=0,
                end_time=run_config.total_beats,
                boundary_level=3,
            ),
        )

    section_names = plan_config.section_names
    section_count = len(section_names)
    if section_count > run_config.total_beats:
        raise ValueError(
            "SECTION_WISE planning requires total_beats >= len(section_names)."
        )
    chunk = run_config.total_beats // section_count
    remainder = run_config.total_beats % section_count
    sections = []
    cursor = 0
    for idx, name in enumerate(section_names):
        length = chunk + (1 if idx < remainder else 0)
        next_cursor = cursor + max(1, length)
        sections.append(
            PlanningSection(
                name=name,
                start_time=cursor,
                end_time=next_cursor,
                boundary_level=3 if idx == section_count - 1 else 2,
                target_tension_arc=(0.2 + (0.1 * idx), 0.8, 0.25),
            )
        )
        cursor = next_cursor
    last = sections[-1]
    if last.end_time != run_config.total_beats:
        sections[-1] = PlanningSection(
            name=last.name,
            start_time=last.start_time,
            end_time=run_config.total_beats,
            boundary_level=last.boundary_level,
            target_tension_arc=last.target_tension_arc,
        )
    return tuple(sections)


def _meter_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    ids = []
    for signature in style_config.allowed_meters:
        if signature in vocabularies.meters.label_map:
            ids.append(vocabularies.meters.token_for_label(signature).id)
    if not ids:
        ids.append(vocabularies.meters.token_for_id(0).id)
    return tuple(dict.fromkeys(ids))


def _key_anchor_ids(run_config: MethodARunConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    fifth = get_fifth_steps(run_config.edo) % len(vocabularies.keys)
    anchors = (0, fifth, len(vocabularies.keys) // 2)
    return tuple(dict.fromkeys(anchor % len(vocabularies.keys) for anchor in anchors))


def _chord_id_for(key_id: int, quality: str, vocabularies: Vocabularies) -> int:
    for chord in vocabularies.chords:
        if chord.root_pc == key_id and chord.quality == quality:
            return chord.id
    return vocabularies.chords.token_for_id(0).id


def _groove_anchor_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    ids = []
    for groove in vocabularies.grooves:
        if groove.family in style_config.groove_families:
            ids.append(groove.id)
    return tuple(dict.fromkeys(ids[: max(1, min(4, len(ids)))]))


def _endpoint_boundary_level(*, is_start: bool, beat_in_bar: int) -> int:
    if beat_in_bar != 0:
        return 0 if is_start else 1
    return 3 if is_start else 2


def _candidate_score(
    state: BeatState,
    *,
    is_start: bool,
    boundary_level: int,
    primary_key_id: int,
) -> float:
    score = 0.0
    score += 2.0 if state.beat_in_bar == 0 else 0.4
    score += 1.5 if state.boundary_lvl == boundary_level else 0.0
    score += 1.2 if state.key_id == primary_key_id else 0.5
    if is_start:
        score += 1.1 if state.role_id == 0 else 0.0
        score += 0.8 if state.head_id == 1 else 0.2
    else:
        score += 1.1 if state.role_id == 3 else 0.4
        score += 0.8 if state.head_id == 1 else 0.3
    return score


def _build_endpoint_distribution(
    *,
    time_index: int,
    beat_in_bar_by_meter: dict[int, int],
    is_start: bool,
    run_config: MethodARunConfig,
    vocabularies: Vocabularies,
) -> EndpointDistribution:
    plan_config = run_config.plan_config
    groove_ids = _groove_anchor_ids(run_config.style_config, vocabularies)
    key_ids = _key_anchor_ids(run_config, vocabularies)
    role_ids = (0, 1) if is_start else (3, 2)
    head_ids = (1, 2)
    chord_qualities = ("maj", "min")

    scored_candidates: list[tuple[float, BeatState]] = []
    for meter_id in _meter_ids(run_config.style_config, vocabularies):
        beat_in_bar = beat_in_bar_by_meter[meter_id]
        boundary_level = _endpoint_boundary_level(is_start=is_start, beat_in_bar=beat_in_bar)
        for key_id in key_ids:
            for quality in chord_qualities:
                chord_id = _chord_id_for(key_id, quality, vocabularies)
                for role_id in role_ids:
                    for head_id in head_ids:
                        for groove_id in groove_ids:
                            state = BeatState(
                                meter_id=meter_id,
                                beat_in_bar=beat_in_bar,
                                boundary_lvl=boundary_level,
                                key_id=key_id,
                                chord_id=chord_id,
                                role_id=role_id,
                                head_id=head_id,
                                groove_id=groove_id,
                            )
                            score = _candidate_score(
                                state,
                                is_start=is_start,
                                boundary_level=boundary_level,
                                primary_key_id=key_ids[0],
                            )
                            score += (
                                run_config.plan_config.start_anchor_weight
                                if is_start
                                else run_config.plan_config.end_anchor_weight
                            )
                            scored_candidates.append((score, state))

    scored_candidates.sort(key=lambda item: (-item[0], _state_sort_key(item[1])))
    unique_states: list[BeatState] = []
    unique_scores: list[float] = []
    seen = set()
    for score, state in scored_candidates:
        if state in seen:
            continue
        seen.add(state)
        unique_states.append(state)
        unique_scores.append(score)
        if len(unique_states) >= plan_config.endpoint_top_k:
            break

    layer = Layer(time_index=time_index, states=tuple(unique_states))
    return EndpointDistribution(
        layer=layer,
        probabilities=_softmax(unique_scores, plan_config.endpoint_temperature),
    )


def generate_start_endpoint_distribution(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> EndpointDistribution:
    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)
    beat_positions = {meter_id: 0 for meter_id in _meter_ids(run_config.style_config, resolved_vocabs)}
    return _build_endpoint_distribution(
        time_index=0,
        beat_in_bar_by_meter=beat_positions,
        is_start=True,
        run_config=run_config,
        vocabularies=resolved_vocabs,
    )


def generate_end_endpoint_distribution(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> EndpointDistribution:
    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)
    beat_positions = {}
    for meter_id in _meter_ids(run_config.style_config, resolved_vocabs):
        beats_per_bar = resolved_vocabs.meters.token_for_id(meter_id).beats_per_bar
        beat_positions[meter_id] = run_config.total_beats % beats_per_bar
    return _build_endpoint_distribution(
        time_index=run_config.total_beats,
        beat_in_bar_by_meter=beat_positions,
        is_start=False,
        run_config=run_config,
        vocabularies=resolved_vocabs,
    )


def generate_method_a_endpoints(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    selection_key: Optional[RNGKey] = None,
    sample_endpoints: bool = False,
) -> MethodAEndpoints:
    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)
    pi0 = generate_start_endpoint_distribution(run_config, vocabularies=resolved_vocabs)
    piT = generate_end_endpoint_distribution(run_config, vocabularies=resolved_vocabs)
    root_key = RNGKey(seed=run_config.seed) if selection_key is None else selection_key
    start_choice, next_key = _choose_endpoint_state(
        pi0,
        key=root_key,
        sample=sample_endpoints,
    )
    end_choice, _ = _choose_endpoint_state(
        piT,
        key=next_key,
        sample=sample_endpoints,
    )
    return MethodAEndpoints(
        pi0=pi0,
        piT=piT,
        start_choice=start_choice,
        end_choice=end_choice,
        sections=build_section_plan(run_config),
    )


def run_method_a(
    run_config: MethodARunConfig,
    *,
    prior: Optional[Prior] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> MethodAPlanResult:
    """Run Method A from endpoint planning through SB path extraction."""
    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)
    resolved_sb = _resolved_sb_config(run_config)
    root_key = RNGKey(seed=run_config.seed)
    endpoint_key, graph_key, bridge_key = root_key.split(3)
    endpoints = generate_method_a_endpoints(
        run_config,
        vocabularies=resolved_vocabs,
        selection_key=endpoint_key,
        sample_endpoints=run_config.use_sampling,
    )
    start_endpoint = _singleton_endpoint_distribution(
        endpoints.start_choice.state,
        time_index=0,
    )
    end_endpoint = _singleton_endpoint_distribution(
        endpoints.end_choice.state,
        time_index=run_config.total_beats,
    )
    graph = build_sparse_graph(
        start_layer=start_endpoint.layer,
        end_layer=end_endpoint.layer,
        total_beats=run_config.total_beats,
        sb_config=resolved_sb,
        style_config=run_config.style_config,
        vocabularies=resolved_vocabs,
        prior=NullPrior() if prior is None else prior,
        weights=run_config.prior_weights,
        edo=run_config.edo,
        rng=_numpy_generator_from_key(graph_key),
        d_max=resolved_sb.d_max,
    )
    aligned_endpoints = MethodAEndpoints(
        pi0=_align_endpoint_distribution(endpoints.pi0, graph.layers[0]),
        piT=_align_endpoint_distribution(endpoints.piT, graph.layers[-1]),
        start_choice=endpoints.start_choice,
        end_choice=endpoints.end_choice,
        sections=endpoints.sections,
    )
    problem = build_sb_problem(graph, start_endpoint, end_endpoint, sb_config=resolved_sb)
    solution = solve_sb(problem)
    bridge = solution.to_bridge()

    if run_config.use_sampling:
        sampled_path, _ = sample_bridge_path(bridge, bridge_key, include_edges=True, include_debug=True)
        path = sampled_path.path
        path_score = None
    else:
        path, path_score = map_bridge_path(bridge)
        sampled_path = None

    diagnostics = MethodAPlanDiagnostics(
        section_tags=tuple(section.name for section in endpoints.sections),
        target_tension_arcs=tuple(section.target_tension_arc for section in endpoints.sections),
        chosen_start_state=endpoints.start_choice.state,
        chosen_end_state=endpoints.end_choice.state,
        endpoint_selection_mode=endpoints.start_choice.selection_mode,
        chosen_start_probability=endpoints.start_choice.selected_probability,
        chosen_end_probability=endpoints.end_choice.selected_probability,
        path_mode="sample" if run_config.use_sampling else "map",
        graph_layer_sizes=graph.diagnostics.layer_sizes,
        bridge_iterations=solution.trace.iterations,
        bridge_converged=solution.trace.converged,
    )
    return MethodAPlanResult(
        run_config=run_config,
        vocabularies=resolved_vocabs,
        endpoints=aligned_endpoints,
        graph=graph,
        sb_problem=problem,
        sb_solution=solution,
        bridge=bridge,
        path=path,
        path_score=path_score,
        sampled_path=sampled_path,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# EPIC 10 — 1: Method B  (start → midpoint → return, two SB passes)

@dataclass(frozen=True)
class MethodBRunConfig:
    """
    Run configuration for Method B: two-leg SB with an explicit midpoint.
    """

    total_beats: int
    seed: int = 0
    use_sampling: bool = False
    style_config: StyleConfig = field(default_factory=StyleConfig)
    prior_weights: PriorWeights = field(default_factory=PriorWeights)
    sb_config: Optional[SBConfig] = None
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)
    plan_config: PlanConfig = field(default_factory=lambda: PlanConfig(
        method=PlanMethod.METHOD_B,
        loop_midpoint=1,
    ))
    neural_prior_config: NeuralPriorConfig = field(default_factory=NeuralPriorConfig)
    edo: int = 12

    def __post_init__(self) -> None:
        _require_int("total_beats", self.total_beats, minimum=2)
        _require_int("seed", self.seed, minimum=0)
        if not isinstance(self.use_sampling, bool):
            raise TypeError("use_sampling must be a bool.")
        if self.plan_config.method is not PlanMethod.METHOD_B:
            raise ValueError("MethodBRunConfig requires plan_config.method == METHOD_B.")
        midpoint = self.plan_config.loop_midpoint
        if midpoint is None:
            raise ValueError("plan_config.loop_midpoint must be set for Method B.")
        if not (1 <= midpoint < self.total_beats):
            raise ValueError(
                f"loop_midpoint must satisfy 1 <= loop_midpoint < total_beats, "
                f"got loop_midpoint={midpoint}, total_beats={self.total_beats}."
            )
        _require_int("edo", self.edo, minimum=1)

    @property
    def loop_midpoint(self) -> int:
        """Convenience accessor for the validated midpoint beat index."""
        return self.plan_config.loop_midpoint  # type: ignore[return-value]

    @property
    def leg1_beats(self) -> int:
        """Number of beats in the first leg (start → midpoint)."""
        return self.loop_midpoint

    @property
    def leg2_beats(self) -> int:
        """Number of beats in the second leg (midpoint → return)."""
        return self.total_beats - self.loop_midpoint


@dataclass(frozen=True)
class MethodBEndpoints:
    """Endpoint distributions and choices for both Method B legs."""

    pi0: EndpointDistribution      # start  (t=0)
    piMid: EndpointDistribution    # midpoint (t=loop_midpoint), leg-1 terminal
    piT: EndpointDistribution      # return (t=total_beats)
    start_choice: EndpointChoice
    mid_choice: EndpointChoice     # drawn from piMid; pinned as leg-2 start
    end_choice: EndpointChoice
    sections: Tuple[PlanningSection, ...]


@dataclass(frozen=True)
class MethodBLegDiagnostics:
    """Per-leg solver summary for Method B diagnostics."""

    label: str                          # "leg1" or "leg2"
    start_time: int
    end_time: int
    graph_layer_sizes: Tuple[int, ...]
    bridge_iterations: int
    bridge_converged: bool
    path_mode: str

@dataclass(frozen=True)
class MethodBPlanDiagnostics:
    """Inspectable diagnostics for a full Method B planning run."""

    chosen_start_state: BeatState
    chosen_mid_state: BeatState
    chosen_end_state: BeatState
    endpoint_selection_mode: str
    chosen_start_probability: float
    chosen_mid_probability: float
    chosen_end_probability: float
    section_tags: Tuple[str, ...]
    target_tension_arcs: Tuple[Tuple[float, ...], ...]
    leg1: MethodBLegDiagnostics
    leg2: MethodBLegDiagnostics


@dataclass(frozen=True)
class MethodBLegResult:
    """Outputs of one SB leg within a Method B run."""
    graph: SparseGraph
    sb_problem: SBProblem
    sb_solution: SBSolution
    bridge: SolvedBridge
    path: Tuple[BeatState, ...]
    path_score: Optional[float]
    sampled_path: Optional[SampledBridgePath]
    diagnostics: MethodBLegDiagnostics


@dataclass(frozen=True)
class MethodBPlanResult:
    """Full output of a Method B planning run."""
    run_config: MethodBRunConfig
    vocabularies: Vocabularies
    endpoints: MethodBEndpoints
    leg1: MethodBLegResult
    leg2: MethodBLegResult
    path: Tuple[BeatState, ...]        # len == total_beats + 1
    diagnostics: MethodBPlanDiagnostics


# ---------------------------------------------------------------------------
# Internal helpers for Method B

def _sb_config_for_leg(run_config: MethodBRunConfig, leg_beats: int) -> SBConfig:
    """Return an SBConfig whose horizon_t matches the leg length."""
    if run_config.sb_config is not None:
        base = run_config.sb_config
        # Re-stamp horizon_t to the leg length; all other knobs are preserved.
        return SBConfig(
            horizon_t=leg_beats,
            max_iterations=base.max_iterations,
            tolerance=base.tolerance,
            temperature=base.temperature,
            k_max=base.k_max,
            d_max=base.d_max,
            log_underflow_floor=base.log_underflow_floor,
            raise_on_non_convergence=base.raise_on_non_convergence,
            backend_selection=base.backend_selection,
        )
    return SBConfig(horizon_t=leg_beats)


def _run_single_leg(
    *,
    start_state: BeatState,
    end_state: BeatState,
    start_time: int,
    leg_beats: int,
    sb_config: SBConfig,
    run_config: MethodBRunConfig,
    resolved_vocabs: Vocabularies,
    prior: Prior,
    rng: np.random.Generator,
    bridge_key: RNGKey,
    label: str,
) -> MethodBLegResult:
    """Build graph, solve SB, and extract path for one Method B leg."""
    start_layer = Layer(time_index=start_time, states=(start_state,))
    end_layer = Layer(time_index=start_time + leg_beats, states=(end_state,))

    graph = build_sparse_graph(
        start_layer=start_layer,
        end_layer=end_layer,
        total_beats=leg_beats,
        sb_config=sb_config,
        style_config=run_config.style_config,
        vocabularies=resolved_vocabs,
        prior=prior,
        weights=run_config.prior_weights,
        edo=run_config.edo,
        rng=rng,
        d_max=sb_config.d_max,
    )

    pi0_leg = _singleton_endpoint_distribution(start_state, time_index=start_time)
    piT_leg = _singleton_endpoint_distribution(end_state, time_index=start_time + leg_beats)
    problem = build_sb_problem(graph, pi0_leg, piT_leg, sb_config=sb_config)
    solution = solve_sb(problem)
    bridge = solution.to_bridge()

    if run_config.use_sampling:
        sampled_path, _ = sample_bridge_path(
            bridge, bridge_key, include_edges=True, include_debug=True
        )
        path = sampled_path.path
        path_score = None
    else:
        path, path_score = map_bridge_path(bridge)
        sampled_path = None

    leg_diag = MethodBLegDiagnostics(
        label=label,
        start_time=start_time,
        end_time=start_time + leg_beats,
        graph_layer_sizes=graph.diagnostics.layer_sizes,
        bridge_iterations=solution.trace.iterations,
        bridge_converged=solution.trace.converged,
        path_mode="sample" if run_config.use_sampling else "map",
    )

    return MethodBLegResult(
        graph=graph,
        sb_problem=problem,
        sb_solution=solution,
        bridge=bridge,
        path=path,
        path_score=path_score,
        sampled_path=sampled_path,
        diagnostics=leg_diag,
    )


def _build_midpoint_distribution(
    *,
    run_config: MethodBRunConfig,
    vocabularies: Vocabularies,
) -> EndpointDistribution:
    """Build a midpoint endpoint distribution at t=loop_midpoint."""
    mid_time = run_config.loop_midpoint
    plan_config = run_config.plan_config

    # Build candidate scores treating the midpoint as an end-of-leg-1
    end_scores: dict[BeatState, float] = {}
    for meter_id in _meter_ids(run_config.style_config, vocabularies):
        beats_per_bar = vocabularies.meters.token_for_id(meter_id).beats_per_bar
        beat_in_bar = mid_time % beats_per_bar
        boundary_level = _endpoint_boundary_level(is_start=False, beat_in_bar=beat_in_bar)
        key_ids = _key_anchor_ids(run_config, vocabularies)
        groove_ids = _groove_anchor_ids(run_config.style_config, vocabularies)
        for key_id in key_ids:
            for quality in ("maj", "min"):
                chord_id = _chord_id_for(key_id, quality, vocabularies)
                for role_id in (3, 2):      # cad / change — end-of-leg feel
                    for head_id in (1, 2):
                        for groove_id in groove_ids:
                            state = BeatState(
                                meter_id=meter_id,
                                beat_in_bar=beat_in_bar,
                                boundary_lvl=boundary_level,
                                key_id=key_id,
                                chord_id=chord_id,
                                role_id=role_id,
                                head_id=head_id,
                                groove_id=groove_id,
                            )
                            end_scores[state] = _candidate_score(
                                state,
                                is_start=False,
                                boundary_level=boundary_level,
                                primary_key_id=key_ids[0],
                            )

    # Build candidate scores treating the midpoint as a start-of-leg-2
    start_scores: dict[BeatState, float] = {}
    for meter_id in _meter_ids(run_config.style_config, vocabularies):
        beats_per_bar = vocabularies.meters.token_for_id(meter_id).beats_per_bar
        beat_in_bar = mid_time % beats_per_bar
        boundary_level = _endpoint_boundary_level(is_start=True, beat_in_bar=beat_in_bar)
        key_ids = _key_anchor_ids(run_config, vocabularies)
        groove_ids = _groove_anchor_ids(run_config.style_config, vocabularies)
        for key_id in key_ids:
            for quality in ("maj", "min"):
                chord_id = _chord_id_for(key_id, quality, vocabularies)
                for role_id in (0, 1):      # hold / prep — start-of-leg feel
                    for head_id in (1, 2):
                        for groove_id in groove_ids:
                            state = BeatState(
                                meter_id=meter_id,
                                beat_in_bar=beat_in_bar,
                                boundary_lvl=boundary_level,
                                key_id=key_id,
                                chord_id=chord_id,
                                role_id=role_id,
                                head_id=head_id,
                                groove_id=groove_id,
                            )
                            start_scores[state] = _candidate_score(
                                state,
                                is_start=True,
                                boundary_level=boundary_level,
                                primary_key_id=key_ids[0],
                            )

    # Average scores for states that appear in both pools; keep union
    all_states = set(end_scores) | set(start_scores)
    combined: list[tuple[float, BeatState]] = []
    for state in all_states:
        score = (end_scores.get(state, 0.0) + start_scores.get(state, 0.0)) / 2.0
        combined.append((score, state))

    combined.sort(key=lambda item: (-item[0], _state_sort_key(item[1])))
    top_k = plan_config.endpoint_top_k
    combined = combined[:top_k]

    states = tuple(state for _, state in combined)
    scores = [score for score, _ in combined]
    layer = Layer(time_index=mid_time, states=states)
    return EndpointDistribution(
        layer=layer,
        probabilities=_softmax(scores, plan_config.endpoint_temperature),
    )


def generate_method_b_endpoints(
    run_config: MethodBRunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    selection_key: Optional[RNGKey] = None,
    sample_endpoints: bool = False,
) -> MethodBEndpoints:
    """Build and select endpoint distributions for all three Method B anchors."""
    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)

    # Re-use Method A helpers for the outer endpoints.
    # We construct a temporary MethodARunConfig purely to delegate to the
    # existing distribution builders — no SB is run here.
    _a_config = MethodARunConfig(
        total_beats=run_config.total_beats,
        seed=run_config.seed,
        use_sampling=run_config.use_sampling,
        style_config=run_config.style_config,
        prior_weights=run_config.prior_weights,
        decode_config=run_config.decode_config,
        plan_config=PlanConfig(
            method=PlanMethod.METHOD_A,
            sectioning_strategy=SectioningStrategy.SINGLE_PASS,
            endpoint_top_k=run_config.plan_config.endpoint_top_k,
            endpoint_temperature=run_config.plan_config.endpoint_temperature,
            start_anchor_weight=run_config.plan_config.start_anchor_weight,
            end_anchor_weight=run_config.plan_config.end_anchor_weight,
        ),
        neural_prior_config=run_config.neural_prior_config,
        edo=run_config.edo,
    )

    pi0 = generate_start_endpoint_distribution(_a_config, vocabularies=resolved_vocabs)
    piT = generate_end_endpoint_distribution(_a_config, vocabularies=resolved_vocabs)
    piMid = _build_midpoint_distribution(run_config=run_config, vocabularies=resolved_vocabs)

    root_key = RNGKey(seed=run_config.seed) if selection_key is None else selection_key
    start_key, mid_key, end_key = root_key.split(3)

    start_choice, _ = _choose_endpoint_state(pi0, key=start_key, sample=sample_endpoints)
    mid_choice, _ = _choose_endpoint_state(piMid, key=mid_key, sample=sample_endpoints)
    end_choice, _ = _choose_endpoint_state(piT, key=end_key, sample=sample_endpoints)

    # Build a two-section plan: leg1 and leg2.
    sections = (
        PlanningSection(
            name="leg1",
            start_time=0,
            end_time=run_config.loop_midpoint,
            boundary_level=2,
            target_tension_arc=(0.2, 0.85, 0.5),
        ),
        PlanningSection(
            name="leg2",
            start_time=run_config.loop_midpoint,
            end_time=run_config.total_beats,
            boundary_level=3,
            target_tension_arc=(0.5, 0.85, 0.2),
        ),
    )

    return MethodBEndpoints(
        pi0=pi0,
        piMid=piMid,
        piT=piT,
        start_choice=start_choice,
        mid_choice=mid_choice,
        end_choice=end_choice,
        sections=sections,
    )


def run_method_b(
    run_config: MethodBRunConfig,
    *,
    prior: Optional[Prior] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> MethodBPlanResult:
    """Run Method B: two sequential SB passes joined at the midpoint."""
    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)
    resolved_prior: Prior = NullPrior() if prior is None else prior

    root_key = RNGKey(seed=run_config.seed)
    endpoint_key, leg1_graph_key, leg1_bridge_key, leg2_graph_key, leg2_bridge_key = (
        root_key.split(5)
    )

    endpoints = generate_method_b_endpoints(
        run_config,
        vocabularies=resolved_vocabs,
        selection_key=endpoint_key,
        sample_endpoints=run_config.use_sampling,
    )

    start_state = endpoints.start_choice.state
    mid_state = endpoints.mid_choice.state
    end_state = endpoints.end_choice.state

    sb1 = _sb_config_for_leg(run_config, run_config.leg1_beats)
    sb2 = _sb_config_for_leg(run_config, run_config.leg2_beats)

    leg1 = _run_single_leg(
        start_state=start_state,
        end_state=mid_state,
        start_time=0,
        leg_beats=run_config.leg1_beats,
        sb_config=sb1,
        run_config=run_config,
        resolved_vocabs=resolved_vocabs,
        prior=resolved_prior,
        rng=_numpy_generator_from_key(leg1_graph_key),
        bridge_key=leg1_bridge_key,
        label="leg1",
    )

    leg2 = _run_single_leg(
        start_state=mid_state,
        end_state=end_state,
        start_time=run_config.loop_midpoint,
        leg_beats=run_config.leg2_beats,
        sb_config=sb2,
        run_config=run_config,
        resolved_vocabs=resolved_vocabs,
        prior=resolved_prior,
        rng=_numpy_generator_from_key(leg2_graph_key),
        bridge_key=leg2_bridge_key,
        label="leg2",
    )

    # Concatenate: leg1 gives beats 0..midpoint (inclusive),
    # leg2 gives beats midpoint..total (inclusive).
    # Drop leg2[0] because it duplicates leg1[-1] (both are mid_state).
    path: Tuple[BeatState, ...] = leg1.path + leg2.path[1:]

    diagnostics = MethodBPlanDiagnostics(
        chosen_start_state=start_state,
        chosen_mid_state=mid_state,
        chosen_end_state=end_state,
        endpoint_selection_mode=endpoints.start_choice.selection_mode,
        chosen_start_probability=endpoints.start_choice.selected_probability,
        chosen_mid_probability=endpoints.mid_choice.selected_probability,
        chosen_end_probability=endpoints.end_choice.selected_probability,
        section_tags=tuple(s.name for s in endpoints.sections),
        target_tension_arcs=tuple(s.target_tension_arc for s in endpoints.sections),
        leg1=leg1.diagnostics,
        leg2=leg2.diagnostics,
    )

    return MethodBPlanResult(
        run_config=run_config,
        vocabularies=resolved_vocabs,
        endpoints=endpoints,
        leg1=leg1,
        leg2=leg2,
        path=path,
        diagnostics=diagnostics,
    )

# ---------------------------------------------------------------------------
# EPIC 10 — 2: Section-wise SB
# (intro → theme → solo → bridge → return, one SB pass per section)

@dataclass(frozen=True)
class SectionResult:
    """Outputs of one SB pass within a section-wise run."""

    section: PlanningSection
    graph: SparseGraph
    sb_problem: SBProblem
    sb_solution: SBSolution
    bridge: SolvedBridge
    path: Tuple[BeatState, ...]
    path_score: Optional[float]
    sampled_path: Optional[SampledBridgePath]
    diagnostics: MethodBLegDiagnostics   # reused — label = section.name


@dataclass(frozen=True)
class SectionWisePlanDiagnostics:
    """Diagnostics for a full section-wise planning run."""

    section_count: int
    section_tags: Tuple[str, ...]
    target_tension_arcs: Tuple[Tuple[float, ...], ...]
    endpoint_selection_mode: str
    chosen_start_state: BeatState
    chosen_end_state: BeatState
    chosen_start_probability: float
    chosen_end_probability: float
    per_section: Tuple[MethodBLegDiagnostics, ...]
    total_bridge_iterations: int
    all_sections_converged: bool


@dataclass(frozen=True)
class SectionWisePlanResult:
    """Full output of a section-wise Method A planning run."""

    run_config: MethodARunConfig
    vocabularies: Vocabularies
    section_results: Tuple[SectionResult, ...]
    path: Tuple[BeatState, ...]      # length == total_beats + 1
    diagnostics: SectionWisePlanDiagnostics

# ---------------------------------------------------------------------------
# Internal helpers for section-wise planning
# ---------------------------------------------------------------------------
def _key_anchor_ids_from_config(
    edo: int,
    style_config: StyleConfig,
    vocabularies: Vocabularies,
) -> Tuple[int, ...]:
    """Variant of _key_anchor_ids that takes explicit edo + style instead of run_config."""
    fifth = get_fifth_steps(edo) % len(vocabularies.keys)
    anchors = (0, fifth, len(vocabularies.keys) // 2)
    return tuple(dict.fromkeys(anchor % len(vocabularies.keys) for anchor in anchors))


def _build_section_endpoint_distribution(
    *,
    section: PlanningSection,
    is_start: bool,
    run_config: MethodARunConfig,
    vocabularies: Vocabularies,
) -> EndpointDistribution:
    """Build an endpoint distribution for one boundary of a section."""
    time_index = section.start_time if is_start else section.end_time
    beat_positions: dict[int, int] = {}
    for meter_id in _meter_ids(run_config.style_config, vocabularies):
        beats_per_bar = vocabularies.meters.token_for_id(meter_id).beats_per_bar
        beat_positions[meter_id] = time_index % beats_per_bar

    plan_config = run_config.plan_config
    groove_ids = _groove_anchor_ids(run_config.style_config, vocabularies)
    key_ids = _key_anchor_ids(run_config, vocabularies)
    role_ids = (0, 1) if is_start else (3, 2)
    head_ids = (1, 2)
    anchor_weight = (
        plan_config.start_anchor_weight if is_start else plan_config.end_anchor_weight
    )

    scored_candidates: list[tuple[float, BeatState]] = []
    for meter_id in _meter_ids(run_config.style_config, vocabularies):
        beat_in_bar = beat_positions[meter_id]
        boundary_level = _endpoint_boundary_level(is_start=is_start, beat_in_bar=beat_in_bar)
        for key_id in key_ids:
            for quality in ("maj", "min"):
                chord_id = _chord_id_for(key_id, quality, vocabularies)
                for role_id in role_ids:
                    for head_id in head_ids:
                        for groove_id in groove_ids:
                            state = BeatState(
                                meter_id=meter_id,
                                beat_in_bar=beat_in_bar,
                                boundary_lvl=boundary_level,
                                key_id=key_id,
                                chord_id=chord_id,
                                role_id=role_id,
                                head_id=head_id,
                                groove_id=groove_id,
                            )
                            score = _candidate_score(
                                state,
                                is_start=is_start,
                                boundary_level=boundary_level,
                                primary_key_id=key_ids[0],
                            ) + anchor_weight
                            scored_candidates.append((score, state))

    scored_candidates.sort(key=lambda item: (-item[0], _state_sort_key(item[1])))
    unique_states: list[BeatState] = []
    unique_scores: list[float] = []
    seen: set[BeatState] = set()
    for score, state in scored_candidates:
        if state in seen:
            continue
        seen.add(state)
        unique_states.append(state)
        unique_scores.append(score)
        if len(unique_states) >= plan_config.endpoint_top_k:
            break

    layer = Layer(time_index=time_index, states=tuple(unique_states))
    return EndpointDistribution(
        layer=layer,
        probabilities=_softmax(unique_scores, plan_config.endpoint_temperature),
    )


def _run_section(
    *,
    section: PlanningSection,
    start_state: BeatState,
    end_state: BeatState,
    run_config: MethodARunConfig,
    resolved_vocabs: Vocabularies,
    prior: Prior,
    rng: np.random.Generator,
    bridge_key: RNGKey,
    stitch_anchor: Optional[StitchAnchor] = None,
) -> SectionResult:
    """Build graph, solve SB, and extract path for one named section."""
    section_beats = section.end_time - section.start_time
    sb_config = SBConfig(
        horizon_t=section_beats,
        **(
            {
                "max_iterations": run_config.sb_config.max_iterations,
                "tolerance": run_config.sb_config.tolerance,
                "temperature": run_config.sb_config.temperature,
                "k_max": run_config.sb_config.k_max,
                "d_max": run_config.sb_config.d_max,
                "log_underflow_floor": run_config.sb_config.log_underflow_floor,
                "raise_on_non_convergence": run_config.sb_config.raise_on_non_convergence,
                "backend_selection": run_config.sb_config.backend_selection,
            }
            if run_config.sb_config is not None
            else {}
        ),
    )

    start_layer = Layer(time_index=section.start_time, states=(start_state,))
    end_layer = Layer(time_index=section.end_time, states=(end_state,))

    graph = build_sparse_graph(
        start_layer=start_layer,
        end_layer=end_layer,
        total_beats=section_beats,
        sb_config=sb_config,
        style_config=run_config.style_config,
        vocabularies=resolved_vocabs,
        prior=prior,
        weights=run_config.prior_weights,
        edo=run_config.edo,
        rng=rng,
        d_max=sb_config.d_max,
        stitch_anchor=stitch_anchor,
    )

    pi0_sec = _singleton_endpoint_distribution(start_state, time_index=section.start_time)
    piT_sec = _singleton_endpoint_distribution(end_state, time_index=section.end_time)
    problem = build_sb_problem(graph, pi0_sec, piT_sec, sb_config=sb_config)
    solution = solve_sb(problem)
    bridge = solution.to_bridge()

    if run_config.use_sampling:
        sampled_path, _ = sample_bridge_path(
            bridge, bridge_key, include_edges=True, include_debug=True
        )
        path = sampled_path.path
        path_score = None
    else:
        path, path_score = map_bridge_path(bridge)
        sampled_path = None

    leg_diag = MethodBLegDiagnostics(
        label=section.name,
        start_time=section.start_time,
        end_time=section.end_time,
        graph_layer_sizes=graph.diagnostics.layer_sizes,
        bridge_iterations=solution.trace.iterations,
        bridge_converged=solution.trace.converged,
        path_mode="sample" if run_config.use_sampling else "map",
    )

    return SectionResult(
        section=section,
        graph=graph,
        sb_problem=problem,
        sb_solution=solution,
        bridge=bridge,
        path=path,
        path_score=path_score,
        sampled_path=sampled_path,
        diagnostics=leg_diag,
    )


def run_method_a_sectioned(
    run_config: MethodARunConfig,
    *,
    prior: Optional[Prior] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> SectionWisePlanResult:
    """Run section-wise Method A: one SB pass per named section, chained."""
    if run_config.plan_config.sectioning_strategy is not SectioningStrategy.SECTION_WISE:
        raise ValueError(
            "run_method_a_sectioned requires plan_config.sectioning_strategy == SECTION_WISE. "
            "Use run_method_a for single-pass planning."
        )
    if len(run_config.plan_config.section_names) < 2:
        raise ValueError(
            "Section-wise planning requires at least two section_names."
        )

    resolved_vocabs = _resolved_vocabs(vocabularies, run_config.style_config)
    resolved_prior: Prior = NullPrior() if prior is None else prior

    sections = build_section_plan(run_config)
    boundary_count = len(sections) + 1   # one state per boundary

    # Derive independent RNG keys: one selection key, then pairs per section.
    root_key = RNGKey(seed=run_config.seed)
    all_keys = root_key.split(1 + 2 * len(sections))
    selection_key = all_keys[0]
    section_keys = [
        (all_keys[1 + 2 * idx], all_keys[1 + 2 * idx + 1])
        for idx in range(len(sections))
    ]

    # --- Build and choose one state per boundary -------------------------
    # Boundaries: t_0, t_1, t_2, ..., t_N  (N+1 values for N sections)
    # t_0  = sections[0].start_time  (global start)
    # t_k  = sections[k-1].end_time == sections[k].start_time
    # t_N  = sections[-1].end_time   (global end)

    boundary_distributions: list[EndpointDistribution] = []

    # First boundary (global start) — always is_start=True
    boundary_distributions.append(
        _build_section_endpoint_distribution(
            section=sections[0],
            is_start=True,
            run_config=run_config,
            vocabularies=resolved_vocabs,
        )
    )

    # Interior boundaries: end of section[k-1] == start of section[k]
    # We average end-of-section[k-1] and start-of-section[k] scores to
    # find states that are good for both roles.
    for idx in range(len(sections) - 1):
        end_dist = _build_section_endpoint_distribution(
            section=sections[idx],
            is_start=False,
            run_config=run_config,
            vocabularies=resolved_vocabs,
        )
        start_dist = _build_section_endpoint_distribution(
            section=sections[idx + 1],
            is_start=True,
            run_config=run_config,
            vocabularies=resolved_vocabs,
        )
        # Merge: union of states, averaged probabilities (re-normalised)
        all_states = list(
            dict.fromkeys(end_dist.layer.states + start_dist.layer.states)
        )
        merged_scores = []
        for state in all_states:
            p_end = end_dist.probability_of(state)
            p_start = start_dist.probability_of(state)
            merged_scores.append((p_end + p_start) / 2.0)
        total = sum(merged_scores) or 1.0
        merged_probs = tuple(s / total for s in merged_scores)
        time_index = sections[idx].end_time
        boundary_distributions.append(
            EndpointDistribution(
                layer=Layer(time_index=time_index, states=tuple(all_states)),
                probabilities=merged_probs,
            )
        )

    # Last boundary (global end) — always is_start=False
    boundary_distributions.append(
        _build_section_endpoint_distribution(
            section=sections[-1],
            is_start=False,
            run_config=run_config,
            vocabularies=resolved_vocabs,
        )
    )

    assert len(boundary_distributions) == boundary_count

    # Choose one state at each boundary
    chosen_states: list[BeatState] = []
    current_key = selection_key
    for dist in boundary_distributions:
        choice, current_key = _choose_endpoint_state(
            dist, key=current_key, sample=run_config.use_sampling
        )
        chosen_states.append(choice.state)

    # --- Run one SB per section ------------------------------------------
    section_results: list[SectionResult] = []
    for idx, section in enumerate(sections):
        graph_rng_key, bridge_key = section_keys[idx]
        # Build a soft stitch anchor from the previous section's terminal path state 
        if idx == 0 or not section_results:
            anchor: Optional[StitchAnchor] = None
        else:
            prev_path = section_results[-1].path
            prev_terminal = prev_path[-1] if prev_path else None
            anchor = (
                StitchAnchor(prev_terminal=prev_terminal)
                if prev_terminal is not None
                else None
            )
        result = _run_section(
            section=section,
            start_state=chosen_states[idx],
            end_state=chosen_states[idx + 1],
            run_config=run_config,
            resolved_vocabs=resolved_vocabs,
            prior=resolved_prior,
            rng=_numpy_generator_from_key(graph_rng_key),
            bridge_key=bridge_key,
            stitch_anchor=anchor,
        )
        section_results.append(result)

    # --- Concatenate paths -----------------------------------------------
    # Each section path spans [t_start .. t_end] inclusive.
    # Drop path[0] of every section except the first to avoid duplicates.
    joined: list[BeatState] = list(section_results[0].path)
    for result in section_results[1:]:
        joined.extend(result.path[1:])
    path: Tuple[BeatState, ...] = tuple(joined)

    # --- Diagnostics -----------------------------------------------
    per_section_diag = tuple(r.diagnostics for r in section_results)
    total_iterations = sum(d.bridge_iterations for d in per_section_diag)
    all_converged = all(d.bridge_converged for d in per_section_diag)

    start_choice_prob = boundary_distributions[0].probabilities[
        boundary_distributions[0].layer.states.index(chosen_states[0])
    ]
    end_choice_prob = boundary_distributions[-1].probabilities[
        boundary_distributions[-1].layer.states.index(chosen_states[-1])
    ]

    diagnostics = SectionWisePlanDiagnostics(
        section_count=len(sections),
        section_tags=tuple(s.name for s in sections),
        target_tension_arcs=tuple(s.target_tension_arc for s in sections),
        endpoint_selection_mode="sample" if run_config.use_sampling else "argmax",
        chosen_start_state=chosen_states[0],
        chosen_end_state=chosen_states[-1],
        chosen_start_probability=start_choice_prob,
        chosen_end_probability=end_choice_prob,
        per_section=per_section_diag,
        total_bridge_iterations=total_iterations,
        all_sections_converged=all_converged,
    )

    return SectionWisePlanResult(
        run_config=run_config,
        vocabularies=resolved_vocabs,
        section_results=tuple(section_results),
        path=path,
        diagnostics=diagnostics,
    )
    
# ---------------------------------------------------------------------------
# EPIC 10 — 6 & 7: Evaluation harness + section-level diagnostics

# 7: Unified section-level summary that all three run paths emit \

@dataclass(frozen=True)
class SectionSummary:
    """Normalised per-section diagnostic record."""

    label: str
    start_time: int
    end_time: int
    beat_count: int
    graph_layer_sizes: Tuple[int, ...]
    mean_layer_size: float
    min_layer_size: int
    max_layer_size: int
    bridge_iterations: int
    bridge_converged: bool
    final_max_delta: float
    path_mode: str
    neighbor_cache_hit_rate: float = 0.0

    @classmethod
    def from_leg_result(cls, result: "MethodBLegResult") -> "SectionSummary":
        """Build from a single Method B leg result."""
        diag = result.diagnostics
        sizes = diag.graph_layer_sizes
        trace = result.sb_solution.trace
        graph_diag = result.graph.diagnostics
        return cls(
            label=diag.label,
            start_time=diag.start_time,
            end_time=diag.end_time,
            beat_count=diag.end_time - diag.start_time,
            graph_layer_sizes=sizes,
            mean_layer_size=sum(sizes) / len(sizes) if sizes else 0.0,
            min_layer_size=min(sizes) if sizes else 0,
            max_layer_size=max(sizes) if sizes else 0,
            bridge_iterations=trace.iterations,
            bridge_converged=trace.converged,
            final_max_delta=trace.final_max_delta,
            path_mode=diag.path_mode,
            neighbor_cache_hit_rate=graph_diag.neighbor_cache_hit_rate,
        )

    @classmethod
    def from_section_result(cls, result: "SectionResult") -> "SectionSummary":
        """Build from a section-wise SectionResult."""
        diag = result.diagnostics
        sizes = diag.graph_layer_sizes
        trace = result.sb_solution.trace
        graph_diag = result.graph.diagnostics
        return cls(
            label=diag.label,
            start_time=diag.start_time,
            end_time=diag.end_time,
            beat_count=diag.end_time - diag.start_time,
            graph_layer_sizes=sizes,
            mean_layer_size=sum(sizes) / len(sizes) if sizes else 0.0,
            min_layer_size=min(sizes) if sizes else 0,
            max_layer_size=max(sizes) if sizes else 0,
            bridge_iterations=trace.iterations,
            bridge_converged=trace.converged,
            final_max_delta=trace.final_max_delta,
            path_mode=diag.path_mode,
            neighbor_cache_hit_rate=graph_diag.neighbor_cache_hit_rate,
        )

    @classmethod
    def from_method_a_result(cls, result: "MethodAPlanResult") -> "SectionSummary":
        """Build from a single-pass Method A result."""
        diag = result.diagnostics
        sizes = diag.graph_layer_sizes
        trace = result.sb_solution.trace
        graph_diag = result.graph.diagnostics
        return cls(
            label="full",
            start_time=0,
            end_time=len(result.path) - 1,
            beat_count=len(result.path) - 1,
            graph_layer_sizes=sizes,
            mean_layer_size=sum(sizes) / len(sizes) if sizes else 0.0,
            min_layer_size=min(sizes) if sizes else 0,
            max_layer_size=max(sizes) if sizes else 0,
            bridge_iterations=trace.iterations,
            bridge_converged=trace.converged,
            final_max_delta=trace.final_max_delta,
            path_mode=diag.path_mode,
            neighbor_cache_hit_rate=graph_diag.neighbor_cache_hit_rate,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "beat_count": self.beat_count,
            "mean_layer_size": round(self.mean_layer_size, 2),
            "min_layer_size": self.min_layer_size,
            "max_layer_size": self.max_layer_size,
            "bridge_iterations": self.bridge_iterations,
            "bridge_converged": self.bridge_converged,
            "final_max_delta": self.final_max_delta,
            "path_mode": self.path_mode,
            "neighbor_cache_hit_rate": round(self.neighbor_cache_hit_rate, 4),
        }


@dataclass(frozen=True)
class LongFormDiagnostics:
    """Top-level diagnostics for any long-form planning run."""

    method: str
    total_beats: int
    total_sections: int
    sections: Tuple[SectionSummary, ...]
    all_sections_converged: bool
    total_bridge_iterations: int
    mean_bridge_iterations: float
    total_graph_states: int
    mean_neighbor_cache_hit_rate: float
    path_length: int

    @property
    def converged_section_count(self) -> int:
        return sum(1 for s in self.sections if s.bridge_converged)

    @property
    def slowest_section(self) -> Optional[SectionSummary]:
        """Section with the most bridge iterations."""
        return max(self.sections, key=lambda s: s.bridge_iterations) if self.sections else None

    @property
    def largest_section(self) -> Optional[SectionSummary]:
        """Section with the highest max_layer_size."""
        return max(self.sections, key=lambda s: s.max_layer_size) if self.sections else None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "total_beats": self.total_beats,
            "total_sections": self.total_sections,
            "all_sections_converged": self.all_sections_converged,
            "total_bridge_iterations": self.total_bridge_iterations,
            "mean_bridge_iterations": round(self.mean_bridge_iterations, 1),
            "total_graph_states": self.total_graph_states,
            "mean_neighbor_cache_hit_rate": round(self.mean_neighbor_cache_hit_rate, 4),
            "path_length": self.path_length,
            "sections": [s.to_dict() for s in self.sections],
        }

    def format_summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"Method:             {self.method}",
            f"Total beats:        {self.total_beats}",
            f"Sections:           {self.total_sections}",
            f"Path length:        {self.path_length}",
            f"All converged:      {self.all_sections_converged}",
            f"Total SB iters:     {self.total_bridge_iterations}",
            f"Mean SB iters:      {self.mean_bridge_iterations:.1f}",
            f"Total graph states: {self.total_graph_states}",
            f"Cache hit rate:     {self.mean_neighbor_cache_hit_rate:.1%}",
            "",
            "Section breakdown:",
        ]
        for sec in self.sections:
            conv = "✓" if sec.bridge_converged else "✗"
            lines.append(
                f"  [{conv}] {sec.label:20s}  "
                f"beats={sec.beat_count:4d}  "
                f"iters={sec.bridge_iterations:4d}  "
                f"layers={sec.mean_layer_size:5.1f}avg  "
                f"cache={sec.neighbor_cache_hit_rate:.0%}"
            )
        return "\n".join(lines)


def _long_form_diagnostics_from_summaries(
    method: str,
    total_beats: int,
    path_length: int,
    sections: Tuple[SectionSummary, ...],
) -> LongFormDiagnostics:
    """Shared constructor for all three run paths."""
    all_converged = all(s.bridge_converged for s in sections)
    total_iters = sum(s.bridge_iterations for s in sections)
    mean_iters = total_iters / len(sections) if sections else 0.0
    total_states = sum(sum(s.graph_layer_sizes) for s in sections)
    mean_cache = (
        sum(s.neighbor_cache_hit_rate for s in sections) / len(sections)
        if sections else 0.0
    )
    return LongFormDiagnostics(
        method=method,
        total_beats=total_beats,
        total_sections=len(sections),
        sections=sections,
        all_sections_converged=all_converged,
        total_bridge_iterations=total_iters,
        mean_bridge_iterations=mean_iters,
        total_graph_states=total_states,
        mean_neighbor_cache_hit_rate=mean_cache,
        path_length=path_length,
    )


def extract_long_form_diagnostics(
    result: "MethodAPlanResult | MethodBPlanResult | SectionWisePlanResult",
) -> LongFormDiagnostics:
    """Extract a ``LongFormDiagnostics`` from any planning result type."""
    if isinstance(result, MethodAPlanResult):
        summaries = (SectionSummary.from_method_a_result(result),)
        return _long_form_diagnostics_from_summaries(
            method="method_a",
            total_beats=result.run_config.total_beats,
            path_length=len(result.path),
            sections=summaries,
        )

    if isinstance(result, MethodBPlanResult):
        summaries = (
            SectionSummary.from_leg_result(result.leg1),
            SectionSummary.from_leg_result(result.leg2),
        )
        return _long_form_diagnostics_from_summaries(
            method="method_b",
            total_beats=result.run_config.total_beats,
            path_length=len(result.path),
            sections=summaries,
        )

    if isinstance(result, SectionWisePlanResult):
        summaries = tuple(
            SectionSummary.from_section_result(r) for r in result.section_results
        )
        return _long_form_diagnostics_from_summaries(
            method="section_wise",
            total_beats=result.run_config.total_beats,
            path_length=len(result.path),
            sections=summaries,
        )

    raise TypeError(
        f"Unsupported result type: {type(result).__name__}. "
        "Expected MethodAPlanResult, MethodBPlanResult, or SectionWisePlanResult."
    )


# 6: Comparison harness

@dataclass(frozen=True)
class ComparisonRunSpec:
    """Specification for one arm of a comparison experiment."""

    label: str
    run_config: "MethodARunConfig | MethodBRunConfig"
    prior: Optional[Prior] = None
    vocabularies: Optional[Vocabularies] = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string.")
        if not isinstance(self.run_config, (MethodARunConfig, MethodBRunConfig)):
            raise TypeError(
                "run_config must be a MethodARunConfig or MethodBRunConfig."
            )

@dataclass(frozen=True)
class ComparisonArmResult:
    """Result of one arm within a comparison run."""

    spec: ComparisonRunSpec
    result: "MethodAPlanResult | MethodBPlanResult | SectionWisePlanResult"
    diagnostics: LongFormDiagnostics
    wall_seconds: float
    error: Optional[str] = None       # set if the arm raised; result may be None

    @property
    def succeeded(self) -> bool:
        return self.error is None

@dataclass(frozen=True)
class ComparisonReport:
    """Aggregated result of a multi-arm comparison experiment."""

    arms: Tuple[ComparisonArmResult, ...]
    winner_label: Optional[str]
    total_wall_seconds: float

    @property
    def successful_arms(self) -> Tuple[ComparisonArmResult, ...]:
        return tuple(arm for arm in self.arms if arm.succeeded)

    @property
    def failed_arms(self) -> Tuple[ComparisonArmResult, ...]:
        return tuple(arm for arm in self.arms if not arm.succeeded)

    def format_summary(self) -> str:
        """Return a human-readable comparison table."""
        lines = [
            f"Comparison: {len(self.arms)} arms  "
            f"({len(self.successful_arms)} succeeded, "
            f"{len(self.failed_arms)} failed)",
            f"Total wall time: {self.total_wall_seconds:.2f}s",
            f"Winner: {self.winner_label or 'none (no convergence)'}",
            "",
        ]
        for arm in self.arms:
            status = "OK " if arm.succeeded else "ERR"
            if arm.succeeded:
                diag = arm.diagnostics
                lines.append(
                    f"  [{status}] {arm.spec.label:30s}  "
                    f"iters={diag.total_bridge_iterations:5d}  "
                    f"converged={diag.all_sections_converged!s:5}  "
                    f"states={diag.total_graph_states:6d}  "
                    f"wall={arm.wall_seconds:.2f}s"
                )
            else:
                lines.append(
                    f"  [{status}] {arm.spec.label:30s}  "
                    f"error={arm.error}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "winner_label": self.winner_label,
            "total_wall_seconds": round(self.total_wall_seconds, 3),
            "arms": [
                {
                    "label": arm.spec.label,
                    "succeeded": arm.succeeded,
                    "wall_seconds": round(arm.wall_seconds, 3),
                    "error": arm.error,
                    "diagnostics": arm.diagnostics.to_dict() if arm.succeeded else None,
                }
                for arm in self.arms
            ],
        }


def _run_one_arm(spec: ComparisonRunSpec) -> tuple[
    "MethodAPlanResult | MethodBPlanResult | SectionWisePlanResult",
    LongFormDiagnostics,
]:
    """Dispatch to the correct runner for one comparison arm."""
    cfg = spec.run_config
    prior = spec.prior
    vocabs = spec.vocabularies

    if isinstance(cfg, MethodBRunConfig):
        result = run_method_b(cfg, prior=prior, vocabularies=vocabs)
    elif isinstance(cfg, MethodARunConfig):
        if cfg.plan_config.sectioning_strategy is SectioningStrategy.SECTION_WISE:
            result = run_method_a_sectioned(cfg, prior=prior, vocabularies=vocabs)
        else:
            result = run_method_a(cfg, prior=prior, vocabularies=vocabs)
    else:
        raise TypeError(f"Unsupported run_config type: {type(cfg).__name__}")

    diag = extract_long_form_diagnostics(result)
    return result, diag


def run_comparison(
    specs: Sequence[ComparisonRunSpec],
    *,
    raise_on_arm_error: bool = False,
) -> ComparisonReport:
    """Run multiple planning configurations and return a comparison report."""
    import time

    spec_list = list(specs)
    if not spec_list:
        raise ValueError("specs must contain at least one ComparisonRunSpec.")
    if any(not isinstance(spec, ComparisonRunSpec) for spec in spec_list):
        raise TypeError("specs must contain only ComparisonRunSpec instances.")

    arm_results: list[ComparisonArmResult] = []
    total_start = time.perf_counter()

    for spec in spec_list:
        arm_start = time.perf_counter()
        try:
            result, diag = _run_one_arm(spec)
            wall = time.perf_counter() - arm_start
            arm_results.append(
                ComparisonArmResult(
                    spec=spec,
                    result=result,
                    diagnostics=diag,
                    wall_seconds=wall,
                )
            )
        except Exception as exc:
            if raise_on_arm_error:
                raise
            wall = time.perf_counter() - arm_start
            # Build a stub diagnostics so the arm slot is never empty
            stub_summary = SectionSummary(
                label="error",
                start_time=0,
                end_time=0,
                beat_count=0,
                graph_layer_sizes=(),
                mean_layer_size=0.0,
                min_layer_size=0,
                max_layer_size=0,
                bridge_iterations=0,
                bridge_converged=False,
                final_max_delta=float("inf"),
                path_mode="none",
            )
            stub_diag = LongFormDiagnostics(
                method="error",
                total_beats=0,
                total_sections=0,
                sections=(stub_summary,),
                all_sections_converged=False,
                total_bridge_iterations=0,
                mean_bridge_iterations=0.0,
                total_graph_states=0,
                mean_neighbor_cache_hit_rate=0.0,
                path_length=0,
            )
            arm_results.append(
                ComparisonArmResult(
                    spec=spec,
                    result=None,         # type: ignore[arg-type]
                    diagnostics=stub_diag,
                    wall_seconds=wall,
                    error=str(exc),
                )
            )

    total_wall = time.perf_counter() - total_start

    # Winner: fewest total bridge iterations among converging arms
    converging = [
        arm for arm in arm_results
        if arm.succeeded and arm.diagnostics.all_sections_converged
    ]
    winner_label: Optional[str] = None
    if converging:
        winner = min(converging, key=lambda arm: arm.diagnostics.total_bridge_iterations)
        winner_label = winner.spec.label

    return ComparisonReport(
        arms=tuple(arm_results),
        winner_label=winner_label,
        total_wall_seconds=total_wall,
    )