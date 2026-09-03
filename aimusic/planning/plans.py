from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    NeuralPriorConfig,
    PlanConfig,
    PlanMethod,
    PriorWeights,
    SBConfig,
    SectioningStrategy,
    StyleConfig,
)
from aimusic.core.core_types import BeatState, EndpointDistribution, Layer, Score
from aimusic.core.rng import RNGKey, random_unit
from aimusic.core.vocab import TonalContext, Vocabularies, build_tonal_context
from aimusic.decode import decode_path_to_score
from aimusic.planning.graph import SparseGraph, build_sparse_graph
from aimusic.render import render_midi
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
from aimusic.theory.edo import EDO
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
    solver_pi0: Optional[EndpointDistribution] = None
    solver_piT: Optional[EndpointDistribution] = None


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
    tonal_context: TonalContext
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


@dataclass(frozen=True)
class ExactBridgeDemoResult:
    """Compatibility wrapper for legacy bridge-demo scripts."""

    plan_result: MethodAPlanResult
    score: Score
    output_path: str


def _resolved_vocabs(
    vocabularies: Optional[Vocabularies],
    style_config: StyleConfig,
    edo: int,
) -> Vocabularies:
    return build_tonal_context(
        edo,
        style_config,
        vocabularies=vocabularies,
    ).vocabularies


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


def _endpoint_boundary_level(*, is_start: bool, beat_in_bar: int, strong_beats: Tuple[int, ...]) -> int:
    if beat_in_bar not in strong_beats:
        return 0
    if is_start:
        return 3
    return 2 if beat_in_bar == 0 else 1


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
    head_ids = (1, 2)
    chord_qualities = ("maj", "min")

    scored_candidates: list[tuple[float, BeatState]] = []
    for meter_id in _meter_ids(run_config.style_config, vocabularies):
        beat_in_bar = beat_in_bar_by_meter[meter_id]
        strong_beats = vocabularies.meters.token_for_id(meter_id).strong_beats
        boundary_level = _endpoint_boundary_level(is_start=is_start, beat_in_bar=beat_in_bar, strong_beats=strong_beats)
        # Cadence and change roles require a boundary on strong beats
        if is_start or boundary_level > 0:
            role_ids = (0, 1) if is_start else (3, 2)
        else:
            role_ids = (0, 1)
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
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, run_config.edo
    )
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
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, run_config.edo
    )
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
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, run_config.edo
    )
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
    _logger.info(f"Method A: {run_config.total_beats} beats, seed={run_config.seed}")
    tonal_context = build_tonal_context(
        run_config.edo,
        run_config.style_config,
        vocabularies=vocabularies,
    )
    resolved_vocabs = tonal_context.vocabularies
    resolved_sb = _resolved_sb_config(run_config)
    root_key = RNGKey(seed=run_config.seed)
    endpoint_key, graph_key, bridge_key = root_key.split(3)
    endpoints = generate_method_a_endpoints(
        run_config,
        vocabularies=resolved_vocabs,
        selection_key=endpoint_key,
        sample_endpoints=run_config.use_sampling,
    )
    _logger.info(f"Endpoints: start={endpoints.start_choice.state} end={endpoints.end_choice.state}")
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
    _logger.info(f"Graph built: {len(graph.layers)} layers, {sum(len(l.states) for l in graph.layers)} states")
    aligned_endpoints = MethodAEndpoints(
        pi0=endpoints.pi0,
        piT=endpoints.piT,
        start_choice=endpoints.start_choice,
        end_choice=endpoints.end_choice,
        sections=endpoints.sections,
        solver_pi0=_align_endpoint_distribution(start_endpoint, graph.layers[0]),
        solver_piT=_align_endpoint_distribution(end_endpoint, graph.layers[-1]),
    )
    problem = build_sb_problem(graph, start_endpoint, end_endpoint, sb_config=resolved_sb)
    solution = solve_sb(problem)
    bridge = solution.to_bridge()
    _logger.info(f"SB solved: converged={solution.trace.converged}, iterations={solution.trace.iterations}")

    if run_config.use_sampling:
        sampled_path, _ = sample_bridge_path(bridge, bridge_key, include_edges=True, include_debug=True)
        path = sampled_path.path
        path_score = sampled_path.log_probability
        _logger.info(f"Sampled path: {len(path) - 1} beats")
    else:
        path, path_score = map_bridge_path(bridge)
        sampled_path = None
        _logger.info(f"MAP path: {len(path) - 1} beats")

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
        tonal_context=tonal_context,
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


def render_exact_bridge_demo(
    *,
    start_chord: str,
    end_chord: str,
    output_path: str,
    total_beats: int,
    seed: int = 0,
    meter: str = "4/4",
    groove: str = "straight_8ths",
    style_config: Optional[StyleConfig] = None,
    decode_config: Optional[DecodeConfig] = None,
    tempo_bpm: float = 120.0,
    edo: int = 12,
) -> ExactBridgeDemoResult:
    """Render a short bridge example using the current Method A pipeline.

    `start_chord`, `end_chord`, `meter`, and `groove` are accepted for compatibility with
    legacy scripts. The current implementation delegates endpoint selection to Method A.
    """
    del start_chord, end_chord, meter, groove

    resolved_style = StyleConfig() if style_config is None else style_config
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    run_config = MethodARunConfig(
        total_beats=total_beats,
        seed=seed,
        style_config=resolved_style,
        decode_config=resolved_decode,
        edo=edo,
    )
    plan_result = run_method_a(run_config)
    score = decode_path_to_score(
        plan_result.path,
        decode_config=resolved_decode,
        vocabularies=plan_result.vocabularies,
        edo=plan_result.tonal_context.n,
        tempo_bpm=tempo_bpm,
    )
    render_midi(
        score,
        EDO(EDOConfig(n=plan_result.tonal_context.n, base_tuning=0)),
        output_path,
    )
    return ExactBridgeDemoResult(
        plan_result=plan_result,
        score=score,
        output_path=output_path,
    )
