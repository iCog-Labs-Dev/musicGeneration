"""Read-only objective admission of existing native-orchestrator artifacts.

No renderer/asset discovery, human inference, network calls or media writes.
Legacy success flags are checked against source/events/audio, not trusted alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import io
import math
import os
from pathlib import Path, PureWindowsPath
import stat
import tempfile
from typing import Any, Literal

import numpy as np
import soundfile as sf

from aimusic.audio.cache import fingerprint
from aimusic.audio.from_midi import from_midi
from aimusic.audio.from_score import from_score, validate_package
from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.publication import validate_names
from aimusic.audio.render.stems import isolate_source_stems, STEM_LAYOUT_VERSION, GENERATED
from aimusic.audio.render.tuning import channel_pitch_evidence, required_tuning
from aimusic.audio.render.util import render_frames
from aimusic.audio.review_bundle import (
    COMMON_GATES, MAX_FILE_BYTES, REQUIRED_METRICS, FileRef, GraphRecord, Metric, RenderNode,
    SourceNode, canonical_bytes, load_json, read_relative, safe_relative,
)
from aimusic.audio.review_identity import validate_render_identity
from aimusic.audio.scoring.tuning_check import ToneWindow, TuningCheckConfig, TuningReport, analyze_tuning


# Admission is for short local review candidates, not arbitrary render archives.
MAX_OUTPUT_FILES = 64
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_READ_BYTES = 512 * 1024 * 1024  # Includes repeated hashing/decoding/rechecks.
MAX_NATIVE_FILES = MAX_OUTPUT_FILES + 32


@dataclass(frozen=True)
class ControlledAdmission:
    report: TuningReport
    admitted: bool
    flattened: bool


def controlled_tuning(samples: Any, sample_rate: int, intent_bytes: bytes, *,
                      expected_sha256: str, flattened: bool = False,
                      config: TuningCheckConfig = TuningCheckConfig(),
                      report: TuningReport | None = None) -> ControlledAdmission:
    """Complete declared mono windows; existing <=5/tone and <=15 mean/P95.

    One f0 observation per declared steady window; its per-tone P95 is that
    observation's absolute error. Never slide, omit, retarget or infer windows.
    A control needs full coverage AND an aggregate P95 breach, not just any error.
    """
    if type(flattened) is not bool:
        raise ValueError('flattened must be boolean')
    if hashlib.sha256(intent_bytes).hexdigest() != expected_sha256:
        raise ValueError('original tuning intent digest mismatch')
    intent = load_json(intent_bytes)
    canonical_bytes(intent)
    if intent.get('edo') != 19 or intent.get('reference_hz') != 440 or intent.get('reference_midi_note') != 69:
        raise ValueError('original 19-EDO targets required')
    if type(sample_rate) is not int or sample_rate != intent.get('sample_rate'):
        raise ValueError('intent sample rate mismatch; automatic retiming is forbidden')
    notes = intent.get('notes', [])
    windows = tuple(ToneWindow(**w) for w in intent.get('windows', []))
    if not windows or len(windows) != len(notes) or len({w.name for w in windows}) != len(windows):
        raise ValueError('complete unique declared windows required')
    previous = -1
    for window, note in zip(windows, notes):
        step = note.get('edo_step_from_a4')
        if type(step) is not int or abs(step) > 190:
            raise ValueError('invalid original target step')
        hz = 440 * 2 ** (step / 19)
        if note.get('expected_hz') != hz or window.expected_hz != hz or window.percussion:
            raise ValueError('window target differs from original 19-EDO intent')
        if window.start_sample < previous:
            raise ValueError('controlled windows must be ordered and non-overlapping')
        previous = window.stop_sample
    measured = analyze_tuning(samples, sample_rate, windows, config=config, controlled=True)
    if report is not None:
        if tuple(r.window for r in report.windows) != windows:
            raise ValueError('report windows differ from complete original intent')
        if report.to_dict() != measured.to_dict():
            raise ValueError('report differs from recomputed audio evidence')
    full = measured.coverage == 1 and len(measured.windows) == len(windows)
    p95 = measured.p95_abs_cents
    admitted = (full and not measured.passed and p95 is not None and p95 > config.max_p95_cents + 1e-9
                if flattened else measured.passed and full)
    return ControlledAdmission(measured, admitted, flattened)


@dataclass(frozen=True)
class CandidateRequest(GraphRecord):
    output: str
    source: str
    source_kind: Literal['imported_midi', 'render_package']
    expected_identity: Mapping[str, Any]
    manifest_sha256: str
    expected_backend: str
    expected_asset_sha256: str | None
    cache_manifest_sha256: str = field(kw_only=True)
    intent: FileRef | None = None
    original_midi: FileRef | None = None
    flattened: bool = False
    max_peak: float = .95
    min_rms: float = 1e-5

    def validate(self) -> None:
        safe_relative(self.output)
        safe_relative(self.source)
        for value in (self.manifest_sha256, self.cache_manifest_sha256, self.expected_asset_sha256):
            if value is not None:
                FileRef('digest', value, 0)
        if not self.expected_identity or not self.expected_backend:
            raise ValueError('expected source and backend identities required')
        if not 0 < self.max_peak <= .95 or not 0 < self.min_rms <= 1:
            raise ValueError('invalid signal admission floors')
        if self.flattened and (self.intent is None or self.original_midi is None):
            raise ValueError('flattened control needs original MIDI and unchanged intent')


@dataclass(frozen=True)
class CandidateEvidence:
    source: SourceNode
    render: RenderNode
    media: tuple[FileRef, ...]


def _equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ValueError(f'{label} evidence mismatch')


def _signature(events: MidiEvents) -> list[Any]:
    return [(e.tick, e.track, e.message.dict()) for e in events.events]


class _Reader:
    def __init__(self, root: Path):
        self.root = root
        self.refs: dict[str, FileRef] = {}
        self.remaining = MAX_READ_BYTES

    def read(self, path: str) -> bytes:
        if self.remaining <= 0:
            raise ValueError('candidate read work budget exceeded')
        data = read_relative(self.root, path, limit=min(MAX_FILE_BYTES, self.remaining))
        self.remaining -= len(data)
        ref = FileRef(path, hashlib.sha256(data).hexdigest(), len(data))
        if path in self.refs and self.refs[path] != ref:
            raise ValueError('candidate changed during admission')
        self.refs[path] = ref
        return data

    def json(self, path: str) -> Any:
        data = load_json(self.read(path))
        canonical_bytes(data)  # finite nested values, including enormous JSON exponents
        return data


def _preflight_outputs(directory: Path, outputs: Any) -> None:
    """Check the whole bounded inventory and declared disk sizes before payload I/O.

    The legacy cache has hashes, not byte sizes. Use no-follow filesystem metadata
    for the aggregate preflight; _Reader also caps actual bytes if a file grows.
    """
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError('missing manifest outputs')
    if len(outputs) > MAX_OUTPUT_FILES:
        raise ValueError('render output count exceeds admission limit')
    names = validate_names([*outputs, 'cache_manifest.json'])
    for name, sha in outputs.items():
        FileRef(name, sha, 0)
    entries = []
    with os.scandir(directory) as children:
        for child in children:
            entries.append(child)
            if len(entries) > len(names):
                raise ValueError('complete render outputs evidence mismatch')
    _equal(sorted(child.name for child in entries), sorted(names), 'complete render outputs')
    total = 0
    for child in entries:
        info = child.stat(follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
                or info.st_size > MAX_FILE_BYTES):
            raise ValueError('nonregular or over-budget output file')
        total += info.st_size
        if total > MAX_OUTPUT_BYTES:
            raise ValueError('total output bytes exceed admission limit')


def _read_candidate(root: Path, request: CandidateRequest) -> CandidateEvidence:
    """Verify a retained run before any public exposure. All references root-relative.

    expected_identity, manifest_sha256 and cache_manifest_sha256 must come from
    the caller's approved run selection, never the untrusted candidate itself.
    The cache pin anchors every output hash independently of render configuration.
    Native lane reports may be retained as additional references; their historical
    human_listening/checks_passed flags are never admission authority.
    """
    request.__post_init__()
    reader = _Reader(Path(root))
    out = request.output
    manifest = reader.json(f'{out}/render.json')
    if reader.refs[f'{out}/render.json'].sha256 != request.manifest_sha256:
        raise ValueError('render manifest digest mismatch')
    validate_render_identity(manifest.get('effective_config'), manifest.get('application_backend'),
                             manifest.get('runtime'), manifest.get('renderers'))
    cache_path = f'{out}/stems/cache_manifest.json'
    cache_bytes = reader.read(cache_path)
    if reader.refs[cache_path].sha256 != request.cache_manifest_sha256:
        raise ValueError('cache manifest digest mismatch')
    cache = load_json(cache_bytes)
    canonical_bytes(cache)
    _equal(cache.get('fingerprint'), manifest, 'render/cache manifest')
    if cache.get('schema_version') != 1 or cache.get('complete') is not True or cache.get('key') != fingerprint(manifest):
        raise ValueError('incomplete or invalid render cache manifest')
    outputs = cache.get('outputs')
    _preflight_outputs(root / out / 'stems', outputs)
    for name, sha in outputs.items():
        reader.read(f'{out}/stems/{name}')
        if reader.refs[f'{out}/stems/{name}'].sha256 != sha:
            raise ValueError(f'render output digest mismatch: {name}')
    _equal(reader.json(f'{out}/stems/render.json'), manifest, 'duplicated render manifest')
    _equal(manifest.get('application_backend'), {'requested_backend': 'native', 'effective_backend': 'native',
                                                'fallback_reason': None}, 'application backend')
    # Package parsers and reconciliation consume only our already bounded snapshot.
    with tempfile.TemporaryDirectory(prefix='review-source-') as tmp:
        private = Path(tmp)
        package = request.source_kind == 'render_package'
        names: tuple[str, ...] = ('score.mid', 'score.json', 'structure.json', 'tuning.json', 'manifest.json') if package else (Path(request.source).name,)
        if package and (root / request.source / 'beatstates.jsonl').exists():
            names += ('beatstates.jsonl',)
        source_refs = []
        for name in names:
            path = f'{request.source}/{name}' if package else request.source
            (private / name).write_bytes(reader.read(path))
            source_refs.append(reader.refs[path])
        context = from_score(private) if package else from_midi(private / names[0])
        source = MidiEvents.from_bytes((private / names[0]).read_bytes())
        identity = {'midi_sha256': hashlib.sha256((private / names[0]).read_bytes()).hexdigest(), **context.verified_identity}
        score = None
        if package:
            from aimusic.audio.from_score import AudioPackageContext
            assert isinstance(context, AudioPackageContext)
            checked = validate_package(context)
            reconciliation = checked.reconciliation.to_dict()
            analysis = context.analysis_document()
            analysis['tracks'] = list(checked.reconciliation.tracks)
            identity.update(structure_sha256=hashlib.sha256((private / 'structure.json').read_bytes()).hexdigest(),
                            tuning_sha256=hashlib.sha256((private / 'tuning.json').read_bytes()).hexdigest())
            score = context.score
        else:
            from aimusic.audio.from_midi import ImportedMidiContext
            assert isinstance(context, ImportedMidiContext)
            reconciliation, analysis = context.reconciliation, context.analysis_document()
            _equal(reader.read(f'{out}/source.mid').hex(), (private / names[0]).read_bytes().hex(), 'source copy')
    _equal(identity, request.expected_identity, 'approved source identity')
    _equal(manifest.get('source'), identity, 'source identity')
    source_node = SourceNode(request.source_kind, tuple(source_refs), identity)
    stem = Path(names[0]).stem
    analysis_path, reconciliation_path = f'{out}/{stem}.analysis.json', f'{out}/{stem}.reconciliation.json'
    _equal(reader.json(analysis_path), analysis, 'source analysis')
    _equal(reader.json(reconciliation_path), reconciliation, 'package reconciliation')
    semantic = {k: v for k, v in analysis.items() if k != 'source_midi'}
    _equal(manifest.get('analysis'), semantic, 'render analysis')
    config = manifest['effective_config']
    expressive = source
    if config.get('stages', {}).get('expressivization', True):
        from aimusic.audio.groove.apply import _apply_to_events, _analysis_roles
        from aimusic.audio.groove.spec import parse_groove_spec
        groove = config.get('groove', {})
        expressive, diff = _apply_to_events(source, parse_groove_spec(groove.get('spec', {})),
                                            light_cleanup=groove.get('light_cleanup', False),
                                            roles=_analysis_roles(source, analysis))
        path = f'{out}/{stem}.expressive.mid'
        actual = MidiEvents.from_bytes(reader.read(path))
        _equal(_signature(actual), _signature(expressive), 'expressive event')
        _equal(reader.json(path + '.diff.json'), diff, 'groove report')
        sidecar = reader.json(path + '.json')
        if sidecar.get('stage') != 'expressivization' or sidecar.get('metadata', {}).get('analysis_hash') != analysis['inputs_hash']:
            raise ValueError('expressivization report source mismatch')
        expressive_sha = reader.refs[path].sha256
    else:
        expressive_sha = identity['midi_sha256']
    _equal(manifest.get('expressive_midi_sha256'), expressive_sha, 'expressive MIDI digest')
    logical = isolate_source_stems(expressive, analysis=analysis)
    index = reader.json(f'{out}/stems/role_index.json')
    rows, choices = index['stems'], manifest['renderers']
    if index.get('schema_version') != STEM_LAYOUT_VERSION or manifest.get('stem_layout_version') != STEM_LAYOUT_VERSION:
        raise ValueError('unsupported stem layout')
    _equal(index.get('generated'), GENERATED, 'generated routing')
    if not logical or len(logical) != len(rows) or len(rows) != len(choices):
        raise ValueError('missing event/routing evidence')
    expected_roles: dict[str, list[str]] = {}
    for row in rows:
        expected_roles.setdefault(row['role'], []).append(row['id'])
    _equal(index.get('roles'), expected_roles, 'role index routing')
    render_cfg = config['render']
    rate = render_cfg['sample_rate']
    frames = render_frames(rate, manifest['duration_sec'], manifest['release_tail_sec'])
    if frames / rate > 120:
        raise ValueError('candidate duration exceeds 120 seconds')
    ceiling = min(request.max_peak, render_cfg['mix_peak'])
    waves, signals, media = [], {}, []

    def audio(path: str, *, active: tuple[int, int]) -> np.ndarray:
        data = reader.read(path)
        from aimusic.audio.listening_review import _validate_wav
        _validate_wav(data)
        pcm, actual_rate = sf.read(io.BytesIO(data), always_2d=True)
        if (len(pcm), actual_rate) != (frames, rate) or not np.isfinite(pcm).all():
            raise ValueError('nonfinite or unaligned audio')
        peak = float(np.max(np.abs(pcm)))
        if peak > ceiling:
            raise ValueError('audio peak exceeds bounded ceiling')
        a, b = active
        if not 0 <= a < b <= frames:
            raise ValueError('missing expected signal window')
        rms = float(np.sqrt(np.mean(pcm[a:b] ** 2)))
        if rms <= request.min_rms:
            raise ValueError('expected signal is silent')
        signals[path] = {'peak': peak, 'rms': rms, 'window': [a, b], 'frames': frames, 'sample_rate': rate}
        media.append(reader.refs[path])
        return pcm.mean(axis=1)

    value: Any
    for logical_stem, row, choice in zip(logical, rows, choices):
        for key, value in (('source_track_index', logical_stem.source_track_index), ('role', logical_stem.role),
                           ('role_source', logical_stem.role_source), ('channels', logical_stem.channels),
                           ('name', logical_stem.name), ('note_count', len(logical_stem.midi.notes))):
            _equal(row.get(key), value, 'stem routing ' + key)
        path = f'{out}/stems/{row["midi"]}'
        actual = MidiEvents.from_bytes(reader.read(path))
        _equal(_signature(actual), _signature(logical_stem.midi), 'isolated event')
        mode = required_tuning(actual, analysis, score=score, track_name=logical_stem.name,
                               source_track_index=logical_stem.source_track_index, drums=logical_stem.channels == (9,))
        for key, value in (('requested_backend', request.expected_backend), ('effective_backend', request.expected_backend),
                           ('fallback_reason', None), ('soundfont_sha256', request.expected_asset_sha256),
                           ('required_tuning', mode), ('role', logical_stem.role),
                           ('source_track_index', logical_stem.source_track_index)):
            _equal(choice.get(key), value, 'renderer ' + key)
        route = render_cfg['routing'].get(logical_stem.role, {})
        requested = route.get('default' if mode == 'direct_12' else 'microtonal', render_cfg['backend'])
        _equal(requested, request.expected_backend, 'effective config routing')
        backend = choice['backend']
        if not backend.get('native_versions') or not backend.get('code_digest'):
            raise ValueError('missing backend runtime/code identity')
        if mode not in backend['capabilities']['tuning_modes']:
            raise ValueError('unsupported renderer tuning capability')
        evidence = channel_pitch_evidence(actual) if request.expected_backend == 'fluidsynth' and mode == 'channel_pitch' else None
        _equal(choice.get('tuning_evidence'), evidence, 'pitch event')
        wav = f'{out}/stems/{row["wav"]}'
        sidecar = reader.json(wav + '.json')
        for key, value in (('output_sha256', reader.refs[wav].sha256), ('midi_sha256', reader.refs[path].sha256),
                           ('frames', frames), ('requested_backend', request.expected_backend),
                           ('effective_backend', request.expected_backend), ('fallback_reason', None),
                           ('soundfont_sha256', request.expected_asset_sha256), ('required_tuning', mode),
                           ('native_versions', backend['native_versions']), ('code_digest', backend['code_digest']),
                           ('capabilities', backend['capabilities']), ('backend_version', backend['version']),
                           ('tuning', {'metadata': analysis['tuning'], 'event_evidence': evidence})):
            _equal(sidecar.get(key), value, 'renderer report ' + key)
        settings = {**dict(backend.get('settings', [])), 'synth.sample-rate': rate,
                    'duration_sec': manifest['duration_sec'], 'release_tail_sec': manifest['release_tail_sec'],
                    'peak_ceiling': render_cfg['mix_peak']}
        _equal(sidecar.get('render_settings'), settings, 'render settings')
        _equal(sidecar.get('output_path'), row['wav'], 'output name')
        if type(sidecar.get('peak_gain')) not in (float, int) or not 0 < sidecar['peak_gain'] <= 1:
            raise ValueError('invalid render peak gain')
        notes = actual.notes
        active = (int(float(actual.tempo_map.seconds(min(n.start for n in notes))) * rate),
                  int(float(actual.tempo_map.seconds(max(n.end for n in notes))) * rate))
        waves.append(audio(wav, active=active))
    total = np.sum(np.stack(waves), axis=0)
    total *= min(1, render_cfg['mix_peak'] / float(np.max(np.abs(total)))) if np.any(total) else 1
    mixed = audio(f'{out}/stems/rough_mix.wav', active=(0, frames))
    if not np.array_equal(mixed, total.astype(np.float32)):
        raise ValueError('rough mix differs from exact source sum (click excluded)')
    audio(f'{out}/stems/click.wav', active=(0, frames))
    evidence_refs = tuple(reader.refs.values())
    states = [Metric(n, 'unmeasured', 'Scorer not implemented and run by this admission reader', None, (), {})
              for n in REQUIRED_METRICS]
    states += [Metric(n, 'passed', 'Recomputed from verified artifacts', 'review-admission/1', evidence_refs,
                      signals if n in ('finite_audio', 'alignment', 'bounded_peak', 'expected_signal') else {}) for n in COMMON_GATES]
    states.append(Metric('reconciliation', 'passed' if package else 'not_applicable',
                         'Recomputed RenderPackage reconciliation' if package else 'Imported MIDI has no planner',
                         'reconcile/1' if package else None, (reader.refs[reconciliation_path],) if package else (), {}))
    tuning = None
    if request.intent is not None:
        intent_bytes = request.intent.read_verified(root)
        reader.read(request.intent.path)
        if len(waves) != 1:
            raise ValueError('controlled tuning requires one monophonic source stem')
        if request.flattened:
            assert request.original_midi is not None
            original_events = MidiEvents.from_bytes(request.original_midi.read_verified(root))
            validate_control_source(original_events, load_json(intent_bytes))
            original = original_events.to_mido()
            reader.read(request.original_midi.path)
            for track in original.tracks:
                for index, message in enumerate(track):
                    if message.type == 'pitchwheel':
                        track[index] = message.copy(pitch=0)
            _equal(_signature(source), _signature(MidiEvents.from_mido(original)), 'flattened original event')
        if not request.flattened:
            validate_control_source(source, load_json(intent_bytes))
        tuning = controlled_tuning(waves[0], rate, intent_bytes, expected_sha256=request.intent.sha256,
                                   flattened=request.flattened)
        if (root / out / 'tuning-check.json').exists():
            _equal(reader.json(f'{out}/tuning-check.json'), tuning.report.to_dict(), 'controlled tuning report')
        if not tuning.admitted:
            raise ValueError('controlled tuning/negative control admission failed')
        tuning_refs = (request.intent, media[0])
        states.append(Metric('controlled_tuning', 'failed' if request.flattened else 'passed',
                             'Measured against unchanged 19-EDO intent', 'controlled-monophonic/1',
                             tuning_refs, tuning.report.to_dict()))
    else:
        states.append(Metric('controlled_tuning', 'not_applicable', 'No controlled mono intent declared', None, (), {}))
    states.append(Metric('negative_control', 'passed' if request.flattened else 'not_applicable',
                         'Flattened calibration control breaches P95 floor' if request.flattened else 'Not a calibration control',
                         'controlled-monophonic/1' if request.flattened else None,
                         (request.intent, media[0]) if request.flattened and request.intent else (), {}))
    # Check every referenced byte again; callers publish only after a later bundle verification too.
    for path in reader.refs:
        reader.read(path)
    node = RenderNode(source_node.sha256, reader.refs[f'{out}/render.json'], tuple(reader.refs.values()),
                      config, manifest['application_backend'], manifest['runtime'], tuple(choices), tuple(states))
    node.require_admitted()
    return CandidateEvidence(source_node, node, tuple(media))


def read_candidate(root: Path, request: CandidateRequest) -> CandidateEvidence:
    """Public fail-closed reader; malformed legacy document shapes are refusal."""
    try:
        return _read_candidate(Path(root), request)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError(f'malformed candidate evidence: {exc}') from exc


def _native_relative(path: Any) -> str:
    """Compatibility only for paths inside an already digest-pinned legacy report.

    Do not use Path normalization: it can erase empty/dot components before they
    are checked. Never relax FileRef or the general candidate path contract.
    """
    if not isinstance(path, str) or not path:
        raise ValueError('expected native relative path')
    windows = PureWindowsPath(path)
    if windows.drive or windows.root or ('\\' in path and '/' in path):
        raise ValueError('absolute/drive/root/mixed native path rejected')
    parts = path.replace('\\', '/').split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError('empty/dot/traversal native path rejected')
    return safe_relative('/'.join(parts))


def _native_hashes(hashes: Any) -> dict[str, str]:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('missing native artifact/source hashes')
    if len(hashes) > MAX_NATIVE_FILES:
        raise ValueError('native artifact count exceeds admission limit')
    paths = [_native_relative(path) for path in hashes]
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError('normalized/casefold-alias native artifact path')
    for expected in hashes.values():
        FileRef('digest', expected, 0)
    return dict(zip(paths, hashes.values()))


def read_native_candidate(root: Path, reference: FileRef, *, workflow: str, run: str,
                          expected_backend: str, expected_asset_sha256: str | None) -> CandidateEvidence:
    """Adapt the retained native-listening-evidence/1 harness report, not its flags.

    The caller pins the *report bytes*, which must pin both render and cache
    manifests. Every listed artifact is checked before returning any evidence.
    Runtime values still come from render sidecars;
    historical human_listening/checks_passed never become metric observations.
    """
    validate_names([workflow])
    validate_names([run])
    try:
        doc = load_json(reference.read_verified(root))
        if doc['schema'] != 'aimusic.native-listening-evidence/1':
            raise ValueError('unsupported native evidence schema')
        prefix = reference.path.split('/')[:-1]
        output = '/'.join([*prefix, workflow, run])
        selected = doc['workflows'][workflow]
        source_path = '/'.join([*prefix, _native_relative(selected['source'])])
        artifact_hashes = _native_hashes(selected['runs'][run]['artifact_sha256'])
        source_hashes = _native_hashes(selected['source_files_sha256'])
        if 'render.json' not in artifact_hashes:
            raise ValueError('native report does not pin render manifest')
        if 'stems/cache_manifest.json' not in artifact_hashes:
            raise ValueError('native report does not pin cache manifest')
        reader = _Reader(root)
        manifest_path = f'{output}/render.json'
        manifest = reader.json(manifest_path)
        if reader.refs[manifest_path].sha256 != artifact_hashes['render.json']:
            raise ValueError('native evidence file digest mismatch')
        package = 'package_hash' in manifest['source']
        intent = None
        if workflow == 'reference-19':
            intent_path = source_path.rsplit('/', 1)[0] + '/expected.json'
            if 'expected.json' not in source_hashes:
                raise ValueError('native reference lacks original intent digest')
            intent = FileRef(intent_path, source_hashes['expected.json'],
                             (root / intent_path).lstat().st_size)
        request = CandidateRequest(output, source_path.rsplit('/', 1)[0] if package else source_path,
                                   'render_package' if package else 'imported_midi', manifest['source'],
                                   artifact_hashes['render.json'], expected_backend, expected_asset_sha256, intent=intent,
                                   cache_manifest_sha256=artifact_hashes['stems/cache_manifest.json'])
        candidate = read_candidate(root, request)
        for base, hashes in ((output, artifact_hashes), (source_path.rsplit('/', 1)[0], source_hashes)):
            for path, expected in hashes.items():
                full = '/'.join([base, path])
                reader.read(full)
                if reader.refs[full].sha256 != expected:
                    raise ValueError('native evidence file digest mismatch')
        # Recheck the pinned record after reading all referenced data.
        reference.read_verified(root)
        for path in reader.refs:
            reader.read(path)
        reports = tuple(dict.fromkeys((*candidate.render.reports, *reader.refs.values(), reference)))
        return replace(candidate, render=replace(candidate.render, reports=reports))
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError(f'malformed native evidence: {exc}') from exc


def validate_control_source(source: MidiEvents, intent: dict[str, Any]) -> None:
    """Bind complete mono intent to original MIDI, never to the flattened output."""
    evidence = channel_pitch_evidence(source)['notes']
    notes, windows = intent['notes'], intent['windows']
    if len(source.notes) != len(notes) or len(notes) != len(windows) or len(evidence) != len(notes):
        raise ValueError('complete original source notes/windows required')
    previous = -1
    for actual, event, target, window in zip(source.notes, evidence, notes, windows):
        _equal((actual.track, actual.channel, actual.start, actual.end, actual.pitch, actual.velocity),
               (target['track'], target['channel'], target['start_tick'], target['end_tick'],
                target['midi_note'], target['velocity']), 'original control event')
        if actual.start < previous:
            raise ValueError('controlled source must be monophonic')
        previous = actual.end
        hz = target['expected_hz']
        if not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
            raise ValueError('invalid original frequency target')
        if abs(1200 * math.log2(event['hz'] / hz)) > 1:
            raise ValueError('original source differs from 19-EDO target')
        start = float(source.tempo_map.seconds(actual.start)) * intent['sample_rate']
        stop = float(source.tempo_map.seconds(actual.end)) * intent['sample_rate']
        if not start <= window['start_sample'] < window['stop_sample'] <= stop:
            raise ValueError('declared window lies outside original note')
