"""Audio-only renderer contracts; resolve capability/resources before cache lookup.

Providers probe availability without allocating a synth. Only an explicitly
classified missing resource permits fallback; render-time errors always escape.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from aimusic.audio.config import normalize_routing
from aimusic.audio.render import simple_r
from aimusic.audio.render.util import limit_peak, render_frames, write_float_wav
from aimusic.audio.render import tuning


# Retain the existing registry API while keeping policy in its own module.
required_tuning = tuning.required_tuning


class MissingRendererResource(RuntimeError):
    """An optional adapter, binding, native library or asset is unavailable."""


class UnsupportedTuning(ValueError):
    """No verified capability can satisfy the request (never flatten pitches)."""


def file_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class RendererCapabilities:
    tuning_modes: tuple[str, ...] = ('direct_12',)
    revision: str = '1'
    instrument_fidelity: bool = False


@dataclass(frozen=True)
class RenderRequest:
    midi_path: Path
    output_path: Path
    tuning: dict[str, Any]
    required_tuning: str
    sample_rate: int
    duration_sec: float
    release_tail_sec: float
    soundfont: Path | None
    config: dict[str, Any]


@dataclass(frozen=True)
class RendererBackend:
    name: str
    version: str
    capabilities: RendererCapabilities
    code_digest: str
    native_versions: tuple[tuple[str, str], ...]
    render: Callable[[RenderRequest], Path]
    settings: tuple[tuple[str, Any], ...] = ()

    def identity(self) -> dict[str, Any]:
        return {'name': self.name, 'version': self.version, 'capabilities': asdict(self.capabilities),
                'code_digest': self.code_digest, 'native_versions': self.native_versions, 'settings': self.settings}


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    output_sha256: str
    requested_backend: str
    effective_backend: str
    fallback_reason: str | None
    backend_version: str
    capabilities: RendererCapabilities
    code_digest: str
    native_versions: tuple[tuple[str, str], ...]
    soundfont_sha256: str | None
    midi_sha256: str
    render_settings: dict[str, Any]
    frames: int
    peak_gain: float
    required_tuning: str
    tuning: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), 'output_path': self.output_path.name}


@dataclass(frozen=True)
class ResolvedRenderer:
    requested_backend: str
    backend: RendererBackend
    required_tuning: str
    fallback_reason: str | None = None
    soundfont: Path | None = None
    soundfont_sha256: str | None = None

    def verify_asset(self) -> None:
        if self.soundfont is not None and file_digest(self.soundfont) != self.soundfont_sha256:
            raise ValueError('SoundFont bytes changed after renderer resolution; retry')

    def identity(self) -> dict[str, Any]:
        return {'requested_backend': self.requested_backend, 'effective_backend': self.backend.name,
                'fallback_reason': self.fallback_reason, 'required_tuning': self.required_tuning,
                'backend': self.backend.identity(), 'soundfont_sha256': self.soundfont_sha256}


def render_request(request: RenderRequest, selected: ResolvedRenderer) -> RenderResult:
    """Execute exactly the resolved backend, verify its result, write provenance."""
    if request.required_tuning != selected.required_tuning:
        raise UnsupportedTuning('tuning changed after renderer resolution')
    if request.soundfont != selected.soundfont:
        raise ValueError('SoundFont request differs from resolved asset')
    frames = render_frames(request.sample_rate, request.duration_sec, request.release_tail_sec)
    selected.verify_asset()
    midi_sha256 = file_digest(request.midi_path)
    evidence = None
    if selected.backend.name == 'fluidsynth' and request.required_tuning == 'channel_pitch':
        from aimusic.audio.midi_events import MidiEvents
        evidence = tuning.channel_pitch_evidence(MidiEvents.load(request.midi_path))
    path = selected.backend.render(request)  # No runtime fallback, even for a missing-resource error.
    selected.verify_asset()
    if file_digest(request.midi_path) != midi_sha256:
        raise ValueError('MIDI bytes changed during renderer execution; retry')
    if path != request.output_path or path.is_symlink() or not path.is_file():
        raise ValueError('renderer did not produce the requested regular output file')
    samples, rate = sf.read(path, always_2d=True)
    if rate != request.sample_rate or not samples.size or not np.isfinite(samples).all():
        raise ValueError('renderer output has invalid sample rate, empty or nonfinite samples')
    if len(samples) > frames:
        raise ValueError('renderer output exceeds the common horizon; refusing to truncate')
    if len(samples) < frames:
        samples = np.pad(samples, ((0, frames - len(samples)), (0, 0)))
    ceiling = float(request.config.get('render', {}).get('mix_peak', 0.95))
    samples, gain = limit_peak(samples, ceiling)
    write_float_wav(path, samples, rate)
    backend = selected.backend
    result = RenderResult(path, file_digest(path), selected.requested_backend, backend.name,
                          selected.fallback_reason, backend.version, backend.capabilities,
                          backend.code_digest, backend.native_versions, selected.soundfont_sha256, midi_sha256,
                          {**dict(backend.settings), 'synth.sample-rate': request.sample_rate,
                           'duration_sec': request.duration_sec, 'release_tail_sec': request.release_tail_sec,
                           'peak_ceiling': ceiling}, len(samples), gain, request.required_tuning,
                          {'metadata': request.tuning, 'event_evidence': evidence})
    Path(str(path) + '.json').write_text(json.dumps(result.to_dict(), sort_keys=True, indent=2), encoding='utf-8')
    return result


def _simple(request: RenderRequest) -> Path:
    return simple_r.render_stem(request.midi_path, output_path=request.output_path,
                                sample_rate=request.sample_rate, config=request.config,
                                duration_sec=request.duration_sec, release_tail_sec=request.release_tail_sec)


def _simple_backend(asset: Path | None) -> RendererBackend:
    return RendererBackend('simple', '3', RendererCapabilities(), file_digest(Path(simple_r.__file__)),
                           (('numpy', np.__version__), ('pretty_midi', version('pretty_midi')),
                            ('soundfile', sf.__version__), ('libsndfile', sf.__libsndfile_version__)), _simple)


def _fluidsynth_backend(asset: Path | None) -> RendererBackend:
    if asset is None or not asset.is_file():
        raise MissingRendererResource('FluidSynth requires a local license-approved SoundFont; '
                                      'set paths.soundfont or AIMUSIC_SOUNDFONT (docs/audio-fluidsynth.md)')
    from aimusic.audio.render import fluidsynth_r
    binding = fluidsynth_r.load_binding()
    native = fluidsynth_r.versions(binding)
    # Promotion is pinned to the runtime measured by the opt-in native lane.
    modes = ('direct_12', 'channel_pitch') if dict(native).get('libfluidsynth') == '2.6.2' else ('direct_12',)
    return RendererBackend('fluidsynth', '2', RendererCapabilities(modes, 'rpn0-2-semitones-v1', True),
                           file_digest(Path(fluidsynth_r.__file__)), native,
                           partial(fluidsynth_r.render, binding=binding),
                           tuple(sorted({**fluidsynth_r.SETTINGS, 'block_frames': fluidsynth_r.BLOCK_FRAMES}.items())))


_BACKENDS: dict[str, Callable[[Path | None], RendererBackend]] = {
    'simple': _simple_backend, 'fluidsynth': _fluidsynth_backend,
}


def _requested(role: str, config: dict[str, Any], tuning: str) -> str:
    render = normalize_routing(config.get('render', {}))
    role = 'comping' if role == 'comp' else role
    route = render['routing'].get(role, {})
    key = 'default' if tuning == 'direct_12' else 'microtonal'
    return str(route.get(key, render.get('backend', 'simple')))


def resolve_renderer(role: str, *, config: dict[str, Any], tuning: str = 'direct_12',
                     soundfont: Path | None = None) -> ResolvedRenderer:
    requested = _requested(role, config, tuning)
    if requested not in _BACKENDS:
        raise KeyError(f'renderer backend not registered: {requested}')
    reason = None
    try:
        backend = _BACKENDS[requested](soundfont)
    except MissingRendererResource as exc:
        if not config.get('render', {}).get('allow_simple_fallback', False):
            raise
        reason = str(exc)
        backend = _BACKENDS['simple'](None)
    if tuning not in backend.capabilities.tuning_modes:
        raise UnsupportedTuning(f'{backend.name} does not support tuning {tuning}; '
                                'select a verified compatible renderer (no pitch flattening)')
    digest = file_digest(soundfont) if soundfont is not None else None
    return ResolvedRenderer(requested, backend, tuning, reason, soundfont, digest)


def get_renderer_name(role: str, *, config: dict[str, Any] | None = None) -> str:
    """Compatibility lookup; actual rendering always uses resolve_renderer."""
    cfg = config or {}
    name = _requested(role, cfg, 'direct_12')
    return resolve_renderer(role, config=cfg).backend.name if name in _BACKENDS else name


def get_renderer(role: str, *, config: dict[str, Any] | None = None) -> Callable[[RenderRequest], Path]:
    return resolve_renderer(role, config=config or {}).backend.render


def list_roles() -> list[str]:
    return ['bass', 'comping', 'drums', 'lead', 'ornament', 'pads', 'piano']
