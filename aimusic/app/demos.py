"""Artifact demo composed through the shared application component factory."""

from __future__ import annotations

from typing import Optional
from aimusic.app.components import ComponentConfig, create_components
from aimusic.core.component_types import RenderRequest
from aimusic.core.config import DecodeConfig, EDOConfig, StyleConfig
from aimusic.core.protocols import ScoreDecoder, ScoreRenderer, TonalMetric
from aimusic.core.rng import RNGKey
from aimusic.planning.plans import ExactBridgeDemoResult, MethodARunConfig, run_method_a
from aimusic.planning.protocols import CandidateGenerator
from aimusic.render.midi_adapter import MidiRenderSettings


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
    tonal_metric: TonalMetric | None = None,
    candidate_generator: CandidateGenerator | None = None,
    decoder: ScoreDecoder | None = None,
    renderer: ScoreRenderer | None = None,
    component_config: ComponentConfig | None = None,
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
    components = create_components(
        component_config, midi_settings=MidiRenderSettings(EDOConfig(n=edo, base_tuning=0)),
        tonal_metric=tonal_metric, candidate_generator=candidate_generator, decoder=decoder, renderer=renderer,
    )
    plan_result, next_key = run_method_a(
        run_config, key=RNGKey(seed=seed),
        tonal_metric=components.tonal_metric, candidate_generator=components.candidate_generator,
    )
    score, _ = components.decoder(
        plan_result.path,
        decode_config=resolved_decode,
        vocabularies=plan_result.vocabularies,
        edo=plan_result.tonal_context.n,
        tempo_bpm=tempo_bpm,
        key=next_key,
    )
    render_result = components.renderer(score, RenderRequest(output_path, "midi"))
    return ExactBridgeDemoResult(
        plan_result=plan_result,
        score=score,
        output_path=output_path,
        render_result=render_result,
    )
