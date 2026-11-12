from dataclasses import dataclass, replace
import json
from operator import setitem
from pathlib import Path
import tempfile
import unittest
from typing import Callable

from aimusic.app.components import ComponentConfig, ComponentSpec, create_components, load_component_config
from aimusic.core.component_types import TrackInstrumentConfig
from aimusic.core.config import EDOConfig, MicrotonalRendering
from aimusic.core.protocols import ScoreRenderer, TonalMetric
from aimusic.render.adapters import RenderResultAdapter
from aimusic.render.midi_adapter import MidiRenderer, MidiRenderSettings

untyped_setitem: Callable[..., None] = setitem


@dataclass(frozen=True)
class ConfiguredMetric:
    distance: float

    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
        return self.distance

    def basic_space_distance(self, a_root: int, a_quality: str, b_root: int, b_quality: str, edo: int = 12) -> float:
        return self.distance


def make_metric(*, distance: float) -> TonalMetric:
    return ConfiguredMetric(distance)


def make_private_metric(*, distance: float, api_key: str, endpoint: str, credentials: dict[str, object] | None = None) -> TonalMetric:
    if not api_key or not endpoint:
        raise ValueError("private provider requires credentials and endpoint")
    return ConfiguredMetric(distance)


def make_wrong_midi_renderer(*, midi_settings: MidiRenderSettings) -> ScoreRenderer:
    return MidiRenderer(replace(midi_settings, edo_config=replace(midi_settings.edo_config, n=19 if midi_settings.edo_config.n == 12 else 12)))


def make_mutating_midi_renderer(*, midi_settings: MidiRenderSettings) -> ScoreRenderer:
    assert midi_settings.track_instruments is not None
    untyped_setitem(midi_settings.track_instruments, "lead", TrackInstrumentConfig(5))
    return MidiRenderer(midi_settings)


def make_mutating_metric(*, weights: list[float]) -> TonalMetric:
    return ConfiguredMetric(weights.pop(0))


def make_unusable_renderer(*, midi_settings: MidiRenderSettings) -> object:
    return lambda: None


def make_unusable_decoder() -> object:
    return lambda: None


def make_unusable_candidates(*, tonal_metric: TonalMetric) -> object:
    return lambda: None


class TestComponentFactory(unittest.TestCase):
    def test_renderer_factory_cannot_change_the_run_instrument_settings(self) -> None:
        settings = MidiRenderSettings(EDOConfig(n=12), {"lead": TrackInstrumentConfig(88)})
        with self.assertRaisesRegex(ValueError, "cannot construct renderer"):
            create_components(ComponentConfig(renderer=ComponentSpec("tests.test_component_factory:make_mutating_midi_renderer")), midi_settings=settings)
        assert settings.track_instruments is not None
        self.assertEqual(settings.track_instruments["lead"].program, 88)

    def test_only_explicit_public_options_enter_manifest_provenance(self) -> None:
        options = {"distance": 2.5, "api_key": "private-fixture-key", "endpoint": "https://example.invalid/music?signature=private-signature",
                   "credentials": {"token": "private-nested-token"}}
        config = ComponentConfig(tonal_metric=ComponentSpec(
            "tests.test_component_factory:make_private_metric", options, public_options=("distance",),
        ))
        components = create_components(config, midi_settings=MidiRenderSettings(EDOConfig(n=12)))
        self.assertEqual(components.tonal_metric.tonal_distance(0, 1), 2.5)
        record = components.provenance["tonal_metric"]
        assert isinstance(record, dict)
        self.assertEqual(record["options"], {"distance": 2.5})
        self.assertEqual(record["unrecorded_option_count"], 3)
        serialized = json.dumps(components.provenance)
        for secret in ("private-fixture-key", "private-signature", "private-nested-token"):
            self.assertNotIn(secret, serialized)
        private_only = create_components(ComponentConfig(tonal_metric=ComponentSpec(
            "tests.test_component_factory:make_metric", {"distance": 3.0},
        )), midi_settings=MidiRenderSettings(EDOConfig(n=12)))
        private_record = private_only.provenance["tonal_metric"]
        assert isinstance(private_record, dict)
        self.assertEqual(private_record["options"], {})
        self.assertEqual(private_record["unrecorded_option_count"], 1)

    def test_known_midi_settings_must_match_the_run_even_through_wrappers(self) -> None:
        settings = MidiRenderSettings(EDOConfig(n=12), {"lead": TrackInstrumentConfig(88)})
        alternatives = (
            replace(settings, edo_config=replace(settings.edo_config, n=19)),
            replace(settings, edo_config=replace(settings.edo_config, base_tuning=61)),
            replace(settings, edo_config=replace(settings.edo_config, pitch_bend_range=2)),
            replace(settings, edo_config=replace(settings.edo_config, microtonal_rendering_method=MicrotonalRendering.MTS)),
            replace(settings, track_instruments={"lead": TrackInstrumentConfig(5)}),
        )
        for other in alternatives:
            renderers: tuple[ScoreRenderer, ...] = (MidiRenderer(other), RenderResultAdapter(MidiRenderer(other)))
            for renderer in renderers:
                with self.subTest(settings=other, renderer=type(renderer)), self.assertRaisesRegex(ValueError, "MIDI settings must match"):
                    create_components(midi_settings=settings, renderer=renderer)
        with self.assertRaisesRegex(ValueError, "MIDI settings must match"):
            create_components(ComponentConfig(renderer=ComponentSpec("tests.test_component_factory:make_wrong_midi_renderer")), midi_settings=settings)
        matching = create_components(midi_settings=settings, renderer=MidiRenderer(settings))
        record = matching.provenance["renderer"]
        assert isinstance(record, dict)
        self.assertEqual(record["construction_settings"], settings.to_dict())
        no_instruments = MidiRenderSettings(EDOConfig(n=12))
        empty_instruments = MidiRenderSettings(EDOConfig(n=12), {})
        for effective, bound in ((no_instruments, empty_instruments), (empty_instruments, no_instruments)):
            with self.subTest(empty_or_absent=effective):
                create_components(midi_settings=effective, renderer=MidiRenderer(bound))

    def test_config_rejects_unknown_roles_fields_and_non_json_values(self) -> None:
        cases: tuple[object, ...] = (
            [], {"unknown": {}}, {"renderer": None},
            {"decoder": {"factory": "not-an-import"}},
            {"decoder": {"factory": "tests.test_component_factory:make_unusable_decoder", "unexpected": 1}},
            {"decoder": {"factory": "tests.test_component_factory:make_unusable_decoder", "options": []}},
            {"tonal_metric": {"factory": "tests.test_component_factory:make_metric", "options": {"distance": float("nan")}}},
            {"tonal_metric": {"factory": "tests.test_component_factory:make_metric", "public_options": "distance"}},
            {"tonal_metric": {"factory": "tests.test_component_factory:make_metric", "public_options": ["missing"]}},
            {"tonal_metric": {"factory": "tests.test_component_factory:make_metric", "options": {"distance": 2}, "public_options": ["distance", "distance"]}},
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                ComponentConfig.from_dict(value)

    def test_constructor_errors_and_reserved_dependencies_are_reported(self) -> None:
        for config in (
            ComponentConfig(tonal_metric=ComponentSpec("tests.test_component_factory:missing")),
            ComponentConfig(tonal_metric=ComponentSpec("tests.test_component_factory:make_metric", {"wrong_option": 2})),
            ComponentConfig(candidate_generator=ComponentSpec("tests.test_component_factory:make_unusable_candidates", {"tonal_metric": 2})),
            ComponentConfig(renderer=ComponentSpec("tests.test_component_factory:make_unusable_renderer", {"midi_settings": {}})),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_components(config, midi_settings=MidiRenderSettings(EDOConfig(n=12)))

    def test_every_callable_role_is_checked_before_planning(self) -> None:
        for role, export in (("decoder", "make_unusable_decoder"), ("candidate_generator", "make_unusable_candidates")):
            config = ComponentConfig.from_dict({role: {"factory": f"tests.test_component_factory:{export}"}})
            with self.subTest(role=role), self.assertRaisesRegex(TypeError, role):
                create_components(config, midi_settings=MidiRenderSettings(EDOConfig(n=12)))

    def test_runtime_override_provenance_does_not_claim_default_construction(self) -> None:
        supplied = ConfiguredMetric(9.0)
        components = create_components(tonal_metric=supplied, midi_settings=MidiRenderSettings(EDOConfig(n=19)))
        self.assertIs(components.tonal_metric, supplied)
        self.assertEqual(components.provenance["tonal_metric"], {
            "source": "runtime_override", "factory": None,
            "implementation": f"{ConfiguredMetric.__module__}:ConfiguredMetric", "options": None,
            "unrecorded_option_count": None,
        })

    def test_factory_cannot_mutate_saved_config_or_recorded_constructor_options(self) -> None:
        config = ComponentConfig(tonal_metric=ComponentSpec(
            "tests.test_component_factory:make_mutating_metric", {"weights": [2.5, 7.0]}, public_options=("weights",),
        ))
        for run in range(2):
            with self.subTest(run=run):
                components = create_components(config, midi_settings=MidiRenderSettings(EDOConfig(n=12)))
                self.assertEqual(components.tonal_metric.tonal_distance(0, 1), 2.5)
                record = components.provenance["tonal_metric"]
                assert isinstance(record, dict)
                self.assertEqual(record["options"], {"weights": [2.5, 7.0]})

    def test_configured_component_must_accept_the_protocol_call_shape(self) -> None:
        config = ComponentConfig(renderer=ComponentSpec("tests.test_component_factory:make_unusable_renderer"))
        with self.assertRaisesRegex(TypeError, "renderer"):
            create_components(config, midi_settings=MidiRenderSettings(EDOConfig(n=12)))

    def test_json_selects_an_installed_alternative_and_records_effective_options(self) -> None:
        selection = {"tonal_metric": {"factory": "tests.test_component_factory:make_metric", "options": {"distance": 2.5}, "public_options": ["distance"]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "components.json"
            path.write_text(json.dumps(selection), encoding="utf-8")
            components = create_components(
                load_component_config(path), midi_settings=MidiRenderSettings(EDOConfig(n=19)),
            )
        self.assertEqual(components.tonal_metric.tonal_distance(0, 1, 19), 2.5)
        self.assertEqual(components.provenance["tonal_metric"], {
            "source": "configured", "factory": "tests.test_component_factory:make_metric",
            "implementation": "tests.test_component_factory:ConfiguredMetric", "options": {"distance": 2.5},
            "unrecorded_option_count": 0,
        })

    def test_runtime_and_config_selection_for_the_same_role_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "tonal_metric"):
            create_components(
                ComponentConfig(tonal_metric=ComponentSpec("tests.test_component_factory:make_metric", {"distance": 2.0})),
                tonal_metric=ConfiguredMetric(7.0), midi_settings=MidiRenderSettings(EDOConfig(n=12)),
            )
