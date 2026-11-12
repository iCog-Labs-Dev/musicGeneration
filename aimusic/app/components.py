"""Select trusted component factories once at the application boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
from typing import Callable, Mapping

from aimusic.core.protocols import ScoreDecoder, ScoreRenderer, TonalMetric
from aimusic.decode import decode_path_to_score
from aimusic.planning.candidates import MetricBoundCandidates
from aimusic.planning.protocols import CandidateGenerator
from aimusic.render.adapters import RenderResultAdapter
from aimusic.render.midi_adapter import MidiRenderer, MidiRenderSettings
from aimusic.theory.tonal import ExistingTonalMetric


def _json_value(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise TypeError("component options must contain JSON values with string keys.")


@dataclass(frozen=True)
class ComponentSpec:
    factory: str
    options: Mapping[str, object] = field(default_factory=dict)
    public_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.factory, str):
            raise TypeError("factory must be a module:export string.")
        parts = self.factory.split(":")
        if len(parts) != 2 or not parts[0] or not all(part.isidentifier() for part in parts[0].split(".")) or not parts[1].isidentifier():
            raise ValueError("factory must name an installed module:export.")
        if not isinstance(self.options, Mapping):
            raise TypeError("component options must be a mapping.")
        options = dict(self.options)
        encoded = json.dumps(options, allow_nan=False)
        _json_value(options)
        if not isinstance(self.public_options, tuple) or any(not isinstance(name, str) for name in self.public_options):
            raise TypeError("public_options must be a tuple of option names.")
        if len(set(self.public_options)) != len(self.public_options) or any(name not in options for name in self.public_options):
            raise ValueError("public_options must name distinct configured options.")
        object.__setattr__(self, "options", json.loads(encoded))


@dataclass(frozen=True)
class ComponentConfig:
    tonal_metric: ComponentSpec | None = None
    candidate_generator: ComponentSpec | None = None
    decoder: ComponentSpec | None = None
    renderer: ComponentSpec | None = None

    def __post_init__(self) -> None:
        for role in self.__dataclass_fields__:
            value = getattr(self, role)
            if value is not None and not isinstance(value, ComponentSpec):
                raise TypeError(f"{role} must be a ComponentSpec or None.")

    @classmethod
    def from_dict(cls, value: object) -> ComponentConfig:
        if not isinstance(value, dict):
            raise TypeError("component configuration must be a JSON object.")
        selections: dict[str, ComponentSpec] = {}
        for role, selection in value.items():
            if role not in cls.__dataclass_fields__:
                raise ValueError(f"unknown component role: {role!r}.")
            if not isinstance(selection, dict) or set(selection) - {"factory", "options", "public_options"}:
                raise ValueError(f"{role} must specify factory, options and public_options only.")
            factory = selection.get("factory")
            options = selection.get("options", {})
            public_options = selection.get("public_options", [])
            if not isinstance(factory, str) or not isinstance(options, dict):
                raise TypeError(f"{role} requires a factory string and options object.")
            if not isinstance(public_options, list) or any(not isinstance(name, str) for name in public_options):
                raise TypeError(f"{role} public_options must be an array of option names.")
            selections[role] = ComponentSpec(factory, options, tuple(public_options))
        return cls(**selections)


def load_component_config(path: str | Path | None) -> ComponentConfig:
    if path is None:
        return ComponentConfig()
    with Path(path).open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    return ComponentConfig.from_dict(value)


def _identity(value: object) -> str:
    subject = value if isinstance(value, type) or inspect.isroutine(value) else type(value)
    return f"{subject.__module__}:{subject.__qualname__}"


def default_decoder() -> ScoreDecoder:
    return decode_path_to_score


def _construct(
    role: str, spec: ComponentSpec | None, default: Callable[..., object],
    override: object | None, **dependencies: object,
) -> tuple[object, dict[str, object]]:
    if override is not None:
        if spec is not None:
            raise ValueError(f"{role} cannot select both a configured factory and a runtime object.")
        return override, {"source": "runtime_override", "factory": None, "implementation": _identity(override), "options": None, "unrecorded_option_count": None}
    factory = default
    options = {} if spec is None else deepcopy(dict(spec.options))
    public_names = () if spec is None else spec.public_options
    recorded_options = deepcopy({name: options[name] for name in public_names})
    if options.keys() & dependencies.keys():
        raise ValueError(f"{role} options cannot override injected dependencies: {sorted(options.keys() & dependencies.keys())}.")
    if spec is not None:
        module, export = spec.factory.split(":")
        try:
            factory = getattr(importlib.import_module(module), export)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot load {role} factory {spec.factory!r}: {exc}") from exc
        if not callable(factory):
            raise TypeError(f"{role} factory must be callable.")
    try:
        component: object = factory(**options, **dependencies)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot construct {role}: {exc}") from exc
    return component, {
        "source": "default" if spec is None else "configured",
        "factory": _identity(default) if spec is None else spec.factory,
        "implementation": _identity(component), "options": recorded_options,
        "unrecorded_option_count": len(options) - len(public_names),
    }


def _check_shape(role: str, component: object, prototype: Callable[..., object]) -> None:
    if not callable(component):
        raise TypeError(f"{role} must be callable.")
    positional: list[object] = []
    keywords: dict[str, object] = {}
    named: dict[str, object] = {}
    for parameter in inspect.signature(prototype).parameters.values():
        if parameter.name == "self":
            continue
        named[parameter.name] = None
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional.append(None)
        else:
            keywords[parameter.name] = None
    try:
        signature = inspect.signature(component)
        signature.bind(*positional, **keywords)
        signature.bind(**named)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} does not accept the required protocol call shape: {exc}") from exc


@dataclass(frozen=True)
class Components:
    tonal_metric: TonalMetric
    candidate_generator: CandidateGenerator
    decoder: ScoreDecoder
    renderer: ScoreRenderer
    provenance: Mapping[str, object]


def create_components(
    config: ComponentConfig | None = None, *, midi_settings: MidiRenderSettings,
    tonal_metric: TonalMetric | None = None, candidate_generator: CandidateGenerator | None = None,
    decoder: ScoreDecoder | None = None, renderer: ScoreRenderer | None = None,
) -> Components:
    config = ComponentConfig() if config is None else config
    if not isinstance(config, ComponentConfig):
        raise TypeError("config must be a ComponentConfig.")
    if not isinstance(midi_settings, MidiRenderSettings):
        raise TypeError("midi_settings must be MidiRenderSettings.")
    metric, metric_info = _construct("tonal_metric", config.tonal_metric, ExistingTonalMetric, tonal_metric)
    if not isinstance(metric, TonalMetric):
        raise TypeError("tonal_metric must provide both tonal distance operations.")
    _check_shape("tonal_metric.tonal_distance", metric.tonal_distance, TonalMetric.tonal_distance)
    _check_shape("tonal_metric.basic_space_distance", metric.basic_space_distance, TonalMetric.basic_space_distance)
    candidates, candidate_info = _construct(
        "candidate_generator", config.candidate_generator, MetricBoundCandidates,
        candidate_generator, tonal_metric=metric,
    )
    if not isinstance(candidates, CandidateGenerator):
        raise TypeError("candidate_generator must be callable.")
    _check_shape("candidate_generator", candidates, CandidateGenerator.__call__)
    selected_decoder, decoder_info = _construct("decoder", config.decoder, default_decoder, decoder)
    if not isinstance(selected_decoder, ScoreDecoder):
        raise TypeError("decoder must be callable.")
    _check_shape("decoder", selected_decoder, ScoreDecoder.__call__)
    selected_renderer, renderer_info = _construct(
        "renderer", config.renderer, MidiRenderer, renderer, midi_settings=midi_settings,
    )
    if not isinstance(selected_renderer, ScoreRenderer):
        raise TypeError("renderer must be callable.")
    _check_shape("renderer", selected_renderer, ScoreRenderer.__call__)
    backend = selected_renderer
    while isinstance(backend, RenderResultAdapter):
        backend = backend.renderer
    if isinstance(backend, MidiRenderer):
        if backend.midi_settings != midi_settings:
            raise ValueError("renderer MIDI settings must match the run's effective MIDI settings.")
        renderer_info["construction_settings"] = backend.midi_settings.to_dict()
    elif renderer is None:
        renderer_info["construction_settings"] = midi_settings.to_dict()
    else:
        renderer_info["construction_settings"] = None
    renderer_info["wrapper"] = _identity(RenderResultAdapter)
    return Components(metric, candidates, selected_decoder, RenderResultAdapter(selected_renderer), {
        "tonal_metric": metric_info, "candidate_generator": candidate_info,
        "decoder": decoder_info, "renderer": renderer_info,
    })
