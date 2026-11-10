"""Logical stems, common render horizon and deterministic rough mixing."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi as pm
import soundfile as sf

from aimusic.audio.analysis.from_midi import _infer_role
from aimusic.audio.midi_events import MidiEvents, MidiEventError
from aimusic.audio.config import normalize_routing, resolve_soundfont
from aimusic.audio.render.registry import RenderRequest, ResolvedRenderer, render_request as render_stem
from aimusic.audio.render import registry, tuning
from aimusic.audio.render.simple_r import _peak_dbfs
from aimusic.audio.publication import pin_output
from aimusic.audio.render.util import peak_normalize, render_frames, limit_peak, write_float_wav
from aimusic.core.core_types import Score


@dataclass(frozen=True)
class SourceStem:
    source_track_index: int
    name: str
    role: str
    role_source: str
    channels: tuple[int, ...]
    midi: MidiEvents


def isolate_source_stems(source: MidiEvents, *, analysis: dict[str, Any] | None = None) -> tuple[SourceStem, ...]:
    """Use source identities, never PrettyMIDI instrument indices, for stems."""
    declarations = {}
    for row in (analysis or {}).get('tracks', []):
        index = row.get('source_track_index')
        if index is None:
            if row.get('role_source') == 'structure.json':
                raise MidiEventError('Authoritative role requires source_track_index; reconcile first')
            continue  # Legacy inferred PrettyMIDI indices are not SMF identities.
        if not isinstance(index, int) or index in declarations:
            raise MidiEventError('Invalid/duplicate source_track_index in stem roles')
        declarations[index] = row
    result = []
    for index, midi in source.isolate_tracks(tuple(declarations)).items():
        name = next((e.message.name for e in source.events
                     if e.track == index and e.message.type == 'track_name'), '')
        channels = tuple(sorted({e.channel for e in source.events if e.track == index and e.channel is not None}))
        program = next((e.message.program for e in source.events
                        if e.channel in channels and e.message.type == 'program_change'), 0)
        inferred = _infer_role(pm.Instrument(program=program, is_drum=channels == (9,), name=name))
        row = declarations.get(index, {})
        role = str(row.get('role') or inferred)
        result.append(SourceStem(index, name, 'comping' if role == 'comp' else role,
                                 str(row.get('role_source', 'inferred')), channels, midi))
    return tuple(result)


def render_click(
    bar_boundaries_sec: list[float],
    *,
    sample_rate: int = 48000,
    output_path: Path | str,
    peak_dbfs: float = -12.0,
    frames: int | None = None,
) -> Path:
    """Simple metronome click at bar starts."""
    if len(bar_boundaries_sec) < 2:
        duration = 1.0
    else:
        duration = float(bar_boundaries_sec[-1])
    n = frames if frames is not None else max(1, int(duration * sample_rate))
    audio = np.zeros(n, dtype=np.float64)
    click_len = int(0.01 * sample_rate)
    t = np.arange(click_len) / sample_rate
    click = np.sin(2 * np.pi * 1000.0 * t) * np.linspace(1.0, 0.0, click_len)
    for b in bar_boundaries_sec[:-1]:
        i = int(b * sample_rate)
        if 0 <= i < n:
            j = min(n, i + click_len)
            audio[i:j] += click[: j - i]
    audio = peak_normalize(audio, peak_dbfs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_float_wav(out, audio.astype(np.float32), sample_rate)
    return out


def rough_mix(
    stem_paths: list[Path | str], *, output_path: Path | str, peak_dbfs: float | None = None,
    peak: float = 0.95
) -> Path:
    """Sum each supplied source once, pad, then attenuate only if over ceiling."""
    waves: list[np.ndarray] = []
    sr = None
    for p in stem_paths:
        data, file_sr = sf.read(str(p), always_2d=False)
        data = np.asarray(data, dtype=np.float64)
        if not np.isfinite(data).all():
            raise ValueError('stem contains nonfinite samples')
        data = data.mean(axis=1) if data.ndim == 2 else data
        if sr is None:
            sr = int(file_sr)
        elif sr != file_sr:
            raise ValueError('stem sample rates differ')
        waves.append(data)
    if not waves or sr is None:
        raise ValueError("no stems to mix")
    max_len = max(w.shape[0] for w in waves)
    mix = np.zeros(max_len, dtype=np.float64)
    for w in waves:
        mix[: w.shape[0]] += w
    ceiling = min(peak, 10 ** (peak_dbfs / 20)) if peak_dbfs is not None else peak
    mix, _ = limit_peak(mix, ceiling)
    out = Path(output_path)
    with pin_output(out.parent) as root:
        root.check_new_files([out.name])
        with tempfile.TemporaryDirectory(prefix='aimusic-mix-') as tmp:
            staging = Path(tmp)
            write_float_wav(staging / out.name, mix.astype(np.float32), sr)
            root.publish(staging, [out.name], replace=False, sources=tuple(Path(p) for p in stem_paths))
    return out


def write_bar_boundary_table(
    bar_boundaries_sec: list[float],
    *,
    output_path: Path | str,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"bar": i, "start_sec": float(t)} for i, t in enumerate(bar_boundaries_sec)]
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out


# Layout revision invalidates old name-keyed stem caches, not renderer policy.
STEM_LAYOUT_VERSION = 3
GENERATED = {'click': 'click.wav', 'rough_mix': 'rough_mix.wav',
             'bar_boundaries': 'bar_boundaries.json', 'role_index': 'role_index.json'}


def _stem_id(stem: SourceStem) -> str:
    label = unicodedata.normalize('NFKD', stem.name or stem.role).encode('ascii', 'ignore').decode()
    slug = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')[:48] or 'stem'
    return f'track_{stem.source_track_index:04d}_{slug}'


def copy_stem_artifacts(source: Path, output: Path, *, verify_cache: bool = False) -> None:
    """Restore a stem directory without deleting or overwriting user files."""
    from aimusic.audio.cache import MANIFEST, restored_entry
    excluded = {'_complete', MANIFEST} if verify_cache else {'_complete'}
    files = sorted(p for p in source.iterdir() if p.name not in excluded)
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('Stem artifacts must be regular files, not symlinks/directories')
    names = [p.name for p in files]
    snapshot = restored_entry(source, source.name, names) if verify_cache else nullcontext(source)
    with pin_output(output) as root:
        root.check_new_files(names + ([MANIFEST] if verify_cache else []))
        with snapshot as verified:
            if verify_cache:
                root.publish(verified, [*names, MANIFEST], replace=False)
            else:
                with tempfile.TemporaryDirectory(prefix='.stems-') as tmp:
                    staging = Path(tmp)
                    for name in names:
                        shutil.copyfile(verified / name, staging / name)
                    root.publish(staging, names, replace=False)


def stem_artifact_names(stems: tuple[SourceStem, ...]) -> list[str]:
    return [_stem_id(stem) + suffix for stem in stems for suffix in ('.mid', '.wav', '.wav.json')] + list(GENERATED.values())


@dataclass(frozen=True)
class StemRenderPlan:
    stems: tuple[SourceStem, ...]
    renderers: tuple[ResolvedRenderer, ...]
    tuning_evidence: tuple[dict[str, Any] | None, ...]
    config: dict[str, Any]
    tuning: dict[str, Any]
    bounds: list[float]
    duration_sec: float
    release_tail_sec: float
    source_sha256: str

    def verify_inputs(self, midi_path: Path | str) -> None:
        if registry.file_digest(Path(midi_path)) != self.source_sha256:
            raise ValueError('MIDI bytes changed after stem planning; retry')
        for selected in self.renderers:
            selected.verify_asset()

    @property
    def names(self) -> list[str]:
        return stem_artifact_names(self.stems)


def prepare_stem_render(midi_path: Path | str, *, config: dict[str, Any] | None = None,
                        analysis: dict[str, Any] | None = None, score: Score | None = None) -> StemRenderPlan:
    """Resolve every role/capability/asset before rendering or querying its cache."""
    payload = Path(midi_path).read_bytes()
    source = MidiEvents.from_bytes(payload)
    logical = isolate_source_stems(source, analysis=analysis)
    if not logical:
        raise ValueError('no source stems to render')
    cfg = deepcopy(config or {})
    cfg['render'] = normalize_routing(cfg.get('render', {}))
    render = cfg['render']
    render.setdefault('release_tail_sec', 0.25)
    render.setdefault('mix_peak', 0.95)
    configured_end = float(render.get('duration_sec', 0))
    tail = float(render['release_tail_sec'])
    render_frames(int(render.get('sample_rate', 48000)), configured_end, tail)
    limit_peak(np.zeros(1), float(render['mix_peak']))
    asset = resolve_soundfont(cfg)
    metadata = analysis or {}
    choices = tuple(registry.resolve_renderer(
        stem.role, config=cfg, soundfont=asset,
        tuning=registry.required_tuning(stem.midi, metadata, score=score, track_name=stem.name,
                                        source_track_index=stem.source_track_index, drums=stem.channels == (9,))) for stem in logical)
    evidence = tuple(tuning.channel_pitch_evidence(stem.midi)
                     if choice.backend.name == 'fluidsynth' and choice.required_tuning == 'channel_pitch' else None
                     for stem, choice in zip(logical, choices))
    end = float(source.tempo_map.seconds(max(e.tick for e in source.events)))
    bounds = metadata.get('bar_boundaries_sec')
    if not bounds:
        tempo = next((value for tick, value in source.tempo_map.changes if tick == 0), 500000)
        bar = 4 * tempo / 1000000
        bounds = [i * bar for i in range(max(1, round(end / bar)) + 1)]
        bounds[-1] = end
    return StemRenderPlan(logical, choices, evidence, cfg, metadata.get('tuning', {}), bounds,
                          max(end, float(bounds[-1]), configured_end), tail,
                          hashlib.sha256(payload).hexdigest())


def render_role_stems(
    midi_path: Path | str,
    *,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    plan: StemRenderPlan | None = None,
) -> dict[str, Path]:
    """Render each logical source once; publish safe IDs plus a role index.

    Preserved event MIDI is the renderer input. Simple still provides only
    integer-pitch sine previews, not instrument/controller/tuning certification.
    """
    plan = plan or prepare_stem_render(midi_path, config=config, analysis=analysis)
    plan.verify_inputs(midi_path)
    logical, cfg = plan.stems, plan.config
    sr = int(cfg.get('render', {}).get('sample_rate', 48000))
    peak_dbfs = _peak_dbfs(cfg)
    frames = render_frames(sr, plan.duration_sec, plan.release_tail_sec)
    rows: list[dict[str, Any]] = []
    roles: dict[str, list[str]] = {}
    for stem in logical:
        key = _stem_id(stem)
        rows.append({'id': key, 'source_track_index': stem.source_track_index,
                     'name': stem.name, 'role': stem.role, 'role_source': stem.role_source,
                     'channels': list(stem.channels), 'note_count': len(stem.midi.notes),
                     'midi': f'{key}.mid', 'wav': f'{key}.wav'})
        roles.setdefault(stem.role, []).append(key)
    names = [row['id'] + suffix for row in rows for suffix in ('.mid', '.wav', '.wav.json')]
    names.extend(GENERATED.values())

    bounds = plan.bounds

    with pin_output(output_dir) as root:
        root.check_new_files(names)
        with tempfile.TemporaryDirectory(prefix='.stems-') as tmp:
            staging = Path(tmp)
            waves: list[Path | str] = []
            for stem, row, selected in zip(logical, rows, plan.renderers):
                stem_path = staging / row['midi']
                stem.midi.save(stem_path)
                wav_path = staging / row['wav']
                request = RenderRequest(stem_path, wav_path, plan.tuning, selected.required_tuning, sr,
                                        plan.duration_sec, plan.release_tail_sec, selected.soundfont, cfg)
                render_stem(request, selected)
                waves.append(wav_path)
            render_click(bounds, sample_rate=sr, output_path=staging / GENERATED['click'],
                         peak_dbfs=peak_dbfs, frames=frames)
            rough_mix(waves, output_path=staging / GENERATED['rough_mix'], peak=float(cfg['render']['mix_peak']))
            write_bar_boundary_table(bounds, output_path=staging / GENERATED['bar_boundaries'])
            (staging / GENERATED['role_index']).write_text(json.dumps(
                {'schema_version': STEM_LAYOUT_VERSION, 'stems': rows, 'roles': roles,
                 'generated': GENERATED}, indent=2, sort_keys=True), encoding='utf-8')
            plan.verify_inputs(midi_path)
            root.publish(staging, names, replace=False)
    return {**{row['id']: Path(output_dir) / row['wav'] for row in rows},
            **{key: Path(output_dir) / name for key, name in GENERATED.items()}}
