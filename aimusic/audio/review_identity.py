"""Strict, non-coercing validation of the existing native provenance JSON shape.

This is a review contract, not a second config loader: no defaults, normalization,
asset discovery, provider imports, or mutation. Unknown fields require an explicit
schema decision. Keep pair arrays used by RendererBackend.identity byte-compatible.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import re
from typing import Any

Validator = Callable[[Any], None]


def _text(value: Any) -> None:
    if (type(value) is not str or not value.strip() or len(value) > 4096
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('expected nonempty bounded text without controls')


def _bool(value: Any) -> None:
    if type(value) is not bool:
        raise ValueError('expected boolean')


def _number(low: float = -math.inf, high: float = math.inf, *, integer: bool = False,
            positive: bool = False) -> Validator:
    def check(value: Any) -> None:
        types = (int,) if integer else (int, float)
        if type(value) not in types:
            raise ValueError('expected integer' if integer else 'expected number')
        try:
            valid = math.isfinite(value) and low <= value <= high and (not positive or value > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('number outside finite bounds')
    return check


def _enum(*values: str) -> Validator:
    def check(value: Any) -> None:
        _text(value)
        if value not in values:
            raise ValueError('invalid enum value')
    return check


def _nullable(check: Validator) -> Validator:
    def validate(value: Any) -> None:
        if value is not None:
            check(value)
    return validate


def _fields(required: Mapping[str, Validator], optional: Mapping[str, Validator] | None = None) -> Validator:
    allowed = {**required, **(optional or {})}

    def check(value: Any) -> None:
        if not isinstance(value, Mapping) or not required.keys() <= value.keys() or not value.keys() <= allowed.keys():
            raise ValueError('missing or unknown identity fields')
        for key, item in value.items():
            try:
                allowed[key](item)
            except ValueError as exc:
                raise ValueError(f'{key}: {exc}') from exc
    return check


def _map(check: Validator, *, nonempty: bool = False) -> Validator:
    def validate(value: Any) -> None:
        if not isinstance(value, Mapping) or (nonempty and not value):
            raise ValueError('expected identity map')
        for key, item in value.items():
            _text(key)
            check(item)
    return validate


def _array(check: Validator, *, nonempty: bool = False, unique: bool = False) -> Validator:
    def validate(value: Any) -> None:
        if not isinstance(value, (list, tuple)) or (nonempty and not value):
            raise ValueError('expected identity array')
        for item in value:
            check(item)
        if unique and len(value) != len(set(value)):
            raise ValueError('duplicate identity item')
    return validate


def _pairs(check: Validator, *, nonempty: bool = False) -> Validator:
    def validate(value: Any) -> None:
        _array(lambda item: None, nonempty=nonempty)(value)
        keys = set()
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError('expected identity key/value pair')
            key, item = row
            _text(key)
            if key in keys:
                raise ValueError('duplicate identity key')
            keys.add(key)
            check(item)
    return validate


def _sha(value: Any) -> None:
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('expected lowercase SHA-256 digest')


def _setting(value: Any) -> None:
    if type(value) is str:
        _text(value)
    else:
        _number()(value)  # Native settings encode flags as integers, not JSON booleans.


# Groove's optional/null fields mirror the parametric spec, without coercion or
# silently dropping extra fields. No Pydantic/model/native imports in this schema.
_num = _number()
_nonnegative = _number(0)
_count = _number(0, integer=True)
_unit = _number(0, 1)
_microtiming = _fields({}, {
    'by_position_16': _nullable(_array(_num)), 'global_offset': _nullable(_num),
    'jitter_sd': _nullable(_number(0, 4)), 'phrase_rubato': _nullable(_bool),
})
_articulation = _fields({}, {'duration_scale': _nullable(_number(positive=True)), 'strum_ms': _nullable(_num)})
_anticipation = _fields({'prob': _unit, 'push_ms': _array(_num)}, {'at': _text})
_groove_spec = _fields({}, {
    'swing': _nullable(_fields({'ratio': _number(.5, .75)}, {
        'subdivision': _number(1, integer=True), 'applies_to': _array(_text)})),
    'microtiming_ms': _nullable(_fields({}, {role: _nullable(_microtiming) for role in ('drums', 'bass', 'lead', 'comp')})),
    'velocity': _nullable(_fields({}, {
        'accent_map_16': _nullable(_array(_num)),
        'ghost_notes': _nullable(_map(_fields({'prob': _unit, 'vel_scale': _unit}))),
        'phrase_arc': _nullable(_fields({'shape': _text, 'depth': _unit})),
    })),
    'anticipation': _nullable(_fields({}, {role: _nullable(_anticipation) for role in ('comp', 'bass')})),
    'tempo_curve': _nullable(_map(_num)),
    'articulation': _nullable(_fields({}, {role: _nullable(_articulation) for role in ('drums', 'bass', 'lead', 'comp')})),
})
_render_config = _fields({
    'backend': _text, 'sample_rate': _number(1, 768000, integer=True),
    'allow_simple_fallback': _bool, 'routing': _map(_fields({}, {'default': _text, 'microtonal': _text})),
    'release_tail_sec': _nonnegative, 'mix_peak': _number(0, .95, positive=True),
}, {'headroom_dbfs': _num, 'duration_sec': _nonnegative})
_stages = _fields({key: _bool for key in ('analysis', 'expressivization', 'render', 'prompts', 'restyle', 'scoring', 'mixmaster')})
_config = _fields({'render': _render_config, 'stages': _stages, 'paths': _fields({}, {'reference_corpus': _array(_text)})}, {
    'version': _number(1, integer=True), 'mode': _enum('corpus', 'single_ref'), 'code_version': _text,
    'edo': _fields({'tolerance_cents': _nonnegative, 'allow_microtonal_diffusion': _bool}),
    'groove': _fields({}, {
        'spec': _groove_spec, 'author': _text, 'template_from_refs': _bool, 'light_cleanup': _bool,
        'tension_coupling': _fields({'velocity_depth': _num, 'push_scale': _num}),
        'models': _fields({'groovae_drums': _bool, 'midi_gpt': _bool}),
    }),
    'restyle': _fields({'endpoints': _array(_text), 'strength_base': _map(_unit),
                        'microtonal_penalty': _unit, 'tension_gain': _num,
                        'chunk': _fields({'overlap_bars': _count, 'seam_on_boundary_lvl': _count})}),
    'scoring': _fields({'weights': _map(_nonnegative), 'floors': _map(_nonnegative),
                       'role_overrides': _map(_map(_nonnegative))}),
    'search': _fields({'strength_grid': _array(_unit), 'prompt_variants': _count, 'seeds': _count,
                      'successive_halving': _bool}),
    'budget': _fields({'max_endpoint_calls': _count, 'max_usd_estimate': _nonnegative}),
    'mixmaster': _fields({'lufs_targets': _map(_num), 'master_reference': _nullable(_text),
                         'cover_blend': _fields({'enabled': _bool, 'level': _unit, 'highpass_hz': _nonnegative})}),
})
_selection_fields = {'requested_backend': _text, 'effective_backend': _text, 'fallback_reason': _nullable(_text)}
_application = _fields({**_selection_fields, 'requested_backend': _enum('native', 'm2a'),
                       'effective_backend': _enum('native', 'm2a')})
_runtime = _fields({'python': _text, 'platform': _text, 'dependencies': _map(_text, nonempty=True)})
_tuning_mode = _enum('direct_12', 'channel_pitch', 'mts')
_capabilities = _fields({'tuning_modes': _array(_tuning_mode, nonempty=True, unique=True),
                         'revision': _text, 'instrument_fidelity': _bool})
_backend = _fields({'name': _text, 'version': _text, 'code_digest': _sha,
                    'native_versions': _pairs(_text, nonempty=True), 'settings': _pairs(_setting),
                    'capabilities': _capabilities})
_pitch_evidence = _fields({
    'subset': _enum('rpn0-2-semitones-v1'), 'audio_pitch_accuracy': _enum('unmeasured'),
    'notes': _array(_fields({'track': _count, 'channel': _number(0, 15, integer=True), 'tick': _count,
                            'midi_note': _number(0, 127, integer=True), 'wheel': _number(-8192, 8191, integer=True),
                            'hz': _number(positive=True)}), nonempty=True),
})
_renderer = _fields({**_selection_fields, 'required_tuning': _tuning_mode, 'backend': _backend,
                     'soundfont_sha256': _nullable(_sha), 'source_track_index': _count,
                     'role': _text, 'role_source': _enum('inferred', 'structure.json'),
                     'tuning_evidence': _nullable(_pitch_evidence)})


def _selection(value: Mapping[str, Any]) -> None:
    changed = value['requested_backend'] != value['effective_backend']
    if changed != (value['fallback_reason'] is not None):
        raise ValueError('requested/effective backend and fallback reason disagree')


def validate_render_identity(config: Any, application: Any, runtime: Any, renderers: Any) -> None:
    """Validate exact provenance shapes and cross-field consistency; never coerce."""
    for label, check, value in (('effective_config', _config, config), ('application_backend', _application, application),
                                ('runtime', _runtime, runtime), ('renderers', _array(_renderer, nonempty=True), renderers)):
        try:
            check(value)
        except ValueError as exc:
            raise ValueError(f'{label}: {exc}') from exc
    _selection(application)
    if config['stages']['render'] is not True:
        raise ValueError('render identity requires an enabled render stage')
    if config['stages']['expressivization'] and 'spec' not in config.get('groove', {}):
        raise ValueError('expressivization requires effective groove spec')
    tracks = set()
    for renderer in renderers:
        _selection(renderer)
        backend = renderer['backend']
        if backend['name'] != renderer['effective_backend']:
            raise ValueError('backend name differs from effective backend')
        if renderer['required_tuning'] not in backend['capabilities']['tuning_modes']:
            raise ValueError('required tuning not supported by backend capabilities')
        if renderer['fallback_reason'] is not None:
            if not config['render']['allow_simple_fallback'] or renderer['effective_backend'] != 'simple':
                raise ValueError('renderer fallback conflicts with effective config')
        route = config['render']['routing'].get(renderer['role'], {})
        key = 'default' if renderer['required_tuning'] == 'direct_12' else 'microtonal'
        if route.get(key, config['render']['backend']) != renderer['requested_backend']:
            raise ValueError('requested backend differs from effective config routing')
        if backend['name'] == 'fluidsynth' and renderer['soundfont_sha256'] is None:
            raise ValueError('FluidSynth asset digest required')
        expected_pitch = backend['name'] == 'fluidsynth' and renderer['required_tuning'] == 'channel_pitch'
        if expected_pitch != (renderer['tuning_evidence'] is not None):
            raise ValueError('tuning evidence does not match renderer mode')
        track = renderer['source_track_index']
        if track in tracks:
            raise ValueError('duplicate renderer source track')
        tracks.add(track)
