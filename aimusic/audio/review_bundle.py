"""Versioned immutable review graph. This private manifest is not a public page.

SHA-256 detects change relative to a trusted digest, not an author's authenticity.
Assignments/findings are external to avoid both mutating identity and hash cycles.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType, UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

from aimusic.audio.publication import validate_names
from aimusic.audio.review_identity import validate_render_identity

SCHEMA = 'aimusic.review-bundle/1'
MAX_FILE_BYTES = 32 * 1024 * 1024
REQUIRED_METRICS = ('CLAP', 'FAD', 'LUFS', 'rhythm_f_score', 'note_f1', 'polyphonic_f0')
COMMON_GATES = ('integrity', 'source_identity', 'backend_asset', 'finite_audio',
                'alignment', 'bounded_peak', 'expected_signal', 'events', 'routing')
ALL_GATES = (*COMMON_GATES, 'reconciliation', 'controlled_tuning', 'negative_control')
PANEL_SIZES = {'canonical': 4, 'roles': 4, 'parity': 2, 'timing': 2, 'tuning': 2}
State = Literal['passed', 'failed', 'unmeasured', 'not_applicable']
R = TypeVar('R', bound='Record')


def _plain(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('JSON object keys must be strings')
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('expected finite JSON data')


def canonical_bytes(value: Any) -> bytes:
    """ASCII escaped UTF-8, sorted keys, compact separators, no newline or NaN."""
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _sha(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('expected lowercase SHA-256 digest')


def safe_relative(path: str) -> str:
    if not isinstance(path, str) or not path or '%' in path or '\\' in path:
        raise ValueError('expected safe relative path')
    parts = path.split('/')
    for part in parts:
        validate_names([part])
    return path


def _text(value: str) -> None:
    if not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise ValueError('expected nonempty bounded text without controls')


def _unique(values: Any, label: str) -> None:
    items = list(values)
    if len(items) != len(set(items)):
        raise ValueError(f'duplicate {label}')


def _convert(kind: Any, value: Any, *, loading: bool) -> Any:
    origin, args = get_origin(kind), get_args(kind)
    if kind is Any:
        return _freeze(value)
    if origin in (Union, UnionType):
        for option in args:
            try:
                return _convert(option, value, loading=loading)
            except ValueError:
                pass
        raise ValueError('invalid union field')
    if origin is Literal:
        if any(type(value) is type(a) and value == a for a in args):
            return value
        raise ValueError('invalid enum value')
    if origin is tuple:
        if not isinstance(value, tuple if not loading else (tuple, list)):
            raise ValueError('expected immutable tuple')
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, loading=loading) for v in value)
        if len(value) != len(args):
            raise ValueError('invalid tuple length')
        return tuple(_convert(t, v, loading=loading) for t, v in zip(args, value))
    if origin is Mapping:
        if not isinstance(value, Mapping):
            raise ValueError('expected JSON object')
        return _freeze(value)
    if isinstance(kind, type) and issubclass(kind, Record):
        if loading:
            return kind.from_dict(value)
        if not isinstance(value, kind):
            raise ValueError('invalid record type')
        value.__post_init__()
        return value
    if kind is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('expected finite number')
        return float(value)
    if type(value) is not kind:
        raise ValueError('invalid field type')
    return value


def load_json(data: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        _unique((k for k, _ in rows), 'JSON key')
        return dict(rows)

    def bad_constant(value: str) -> None:
        raise ValueError(f'nonfinite JSON: {value}')

    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


@dataclass(frozen=True)
class Record:
    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            try:
                value = _convert(hints[f.name], getattr(self, f.name), loading=False)
                object.__setattr__(self, f.name, value)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f'{type(self).__name__}.{f.name}: {exc}') from exc
        self.validate()

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        """Canonical wire shape, shared by standalone, nested and digest serialization."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    def to_bytes(self) -> bytes:
        return canonical_bytes(self)

    @property
    def digest(self) -> str:
        return digest(self)

    @classmethod
    def from_dict(cls: type[R], data: Any) -> R:
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError(f'{cls.__name__}: missing or unknown fields')
        hints = get_type_hints(cls)
        return cls(**{k: _convert(hints[k], v, loading=True) for k, v in data.items()})

    @classmethod
    def from_bytes(cls: type[R], data: bytes, *, expected_sha256: str | None = None) -> R:
        result = cls.from_dict(load_json(data))
        if expected_sha256 is not None and result.digest != expected_sha256:
            raise ValueError('metadata digest mismatch')
        return result


def read_relative(root: Path, path: str, *, limit: int = MAX_FILE_BYTES) -> bytes:
    """Bounded local reads. Reject links/reparse ancestors, including root parents.

    Trusted local roots only; hostile concurrent ancestor mutation is not promised.
    Publication's retained-handle contract remains separate from this reader.
    """
    safe_relative(path)
    target = Path(root).absolute().joinpath(*path.split('/'))
    for part in (*reversed(target.parents), target):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('link/reparse path rejected')
        if part != target and not stat.S_ISDIR(info.st_mode):
            raise ValueError('non-directory ancestor')
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError('nonregular or over-budget file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(target, flags), 'rb') as stream:
        current = os.fstat(stream.fileno())
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError('file identity changed during read')
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('over-budget file')
    return data


@dataclass(frozen=True)
class FileRef(Record):
    path: str
    sha256: str
    size: int

    def validate(self) -> None:
        safe_relative(self.path)
        _sha(self.sha256)
        if not 0 <= self.size <= MAX_FILE_BYTES:
            raise ValueError('invalid file size')

    def read_verified(self, root: Path) -> bytes:
        data = read_relative(root, self.path)
        if len(data) != self.size or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f'file digest/size mismatch: {self.path}')
        return data


class GraphRecord(Record):
    @property
    def sha256(self) -> str:
        return self.digest


@dataclass(frozen=True)
class Metric(GraphRecord):
    name: str
    state: State
    reason: str
    method: str | None
    evidence: tuple[FileRef, ...]
    observations: Mapping[str, Any]

    def validate(self) -> None:
        _text(self.name)
        _text(self.reason)
        if self.state in ('passed', 'failed'):
            if not self.method or not self.evidence:
                raise ValueError('measured metrics require method and digest-linked evidence')
            if any(word in self.method.casefold() for word in ('human', 'listen', 'subjective', 'reviewer')):
                raise ValueError('human findings cannot supply machine metrics')
        elif self.method is not None or self.evidence or self.observations:
            raise ValueError('unmeasured/not_applicable cannot contain measured results')


@dataclass(frozen=True)
class SourceNode(GraphRecord):
    kind: Literal['imported_midi', 'render_package']
    files: tuple[FileRef, ...]
    identity: Mapping[str, Any]

    def validate(self) -> None:
        if not self.files or not self.identity:
            raise ValueError('source files and identity required')
        for key in ('midi_sha256',) + (('source_hash', 'package_hash') if self.kind == 'render_package' else ()):
            _sha(self.identity.get(key, ''))
        _unique((r.path.casefold() for r in self.files), 'source path')
        midis = [r for r in self.files if r.path.endswith('.mid')]
        if len(midis) != 1 or midis[0].sha256 != self.identity['midi_sha256']:
            raise ValueError('source MIDI identity is not linked to its bytes')
        if self.kind == 'render_package':
            required = {'score.mid', 'score.json', 'structure.json', 'tuning.json', 'manifest.json'}
            if not required <= {r.path.split('/')[-1] for r in self.files}:
                raise ValueError('missing source package files')


@dataclass(frozen=True)
class RenderNode(GraphRecord):
    source_sha256: str
    manifest: FileRef
    reports: tuple[FileRef, ...]
    effective_config: Mapping[str, Any]
    application_backend: Mapping[str, Any]
    runtime: Mapping[str, Any]
    renderers: tuple[Mapping[str, Any], ...]
    metrics: tuple[Metric, ...]

    def validate(self) -> None:
        _sha(self.source_sha256)
        if not self.reports:
            raise ValueError('render reports required')
        validate_render_identity(self.effective_config, self.application_backend, self.runtime, self.renderers)
        _unique((m.name for m in self.metrics), 'metric')
        if not set((*REQUIRED_METRICS, *ALL_GATES)) <= {m.name for m in self.metrics}:
            raise ValueError('missing explicit metric/gate state')

    def require_admitted(self) -> None:
        rows = {m.name: m for m in self.metrics}
        for name in COMMON_GATES:
            if rows[name].state != 'passed':
                raise ValueError(f'candidate rejected: {name}')
        if rows['reconciliation'].state not in ('passed', 'not_applicable'):
            raise ValueError('candidate rejected: reconciliation')
        control = rows['negative_control'].state == 'passed'
        expected = ('failed',) if control else ('passed', 'not_applicable')
        if rows['controlled_tuning'].state not in expected:
            raise ValueError('candidate rejected: controlled_tuning')
        if rows['negative_control'].state not in ('passed', 'not_applicable'):
            raise ValueError('candidate rejected: negative_control')
        for name in set(rows) - set((*REQUIRED_METRICS, *ALL_GATES)):
            if rows[name].state not in ('passed', 'not_applicable'):
                raise ValueError(f'candidate rejected: {name}')


@dataclass(frozen=True)
class PresentationNode(GraphRecord):
    render_sha256: str
    source_wav: FileRef
    output: FileRef
    start_sample: int
    stop_sample: int
    gain: float

    def validate(self) -> None:
        _sha(self.render_sha256)
        if not 0 <= self.start_sample < self.stop_sample or not 0 < self.gain <= 16:
            raise ValueError('invalid bounded trim/gain transform')
        if not re.fullmatch(r'media/s[0-9a-f]{2,64}\.wav', self.output.path):
            raise ValueError('presentation requires opaque media path')
        if self.source_wav.path.casefold() == self.output.path.casefold():
            raise ValueError('presentation must not overwrite renderer output')


@dataclass(frozen=True)
class Stimulus(GraphRecord):
    id: str
    presentation_sha256: str

    def validate(self) -> None:
        if not re.fullmatch('s[0-9a-f]{2,64}', self.id):
            raise ValueError('opaque stimulus ID required')
        _sha(self.presentation_sha256)


@dataclass(frozen=True)
class Panel(GraphRecord):
    purpose: Literal['canonical', 'roles', 'parity', 'timing', 'tuning']
    stimuli: tuple[str, ...]

    def validate(self) -> None:
        if len(self.stimuli) != PANEL_SIZES[self.purpose]:
            raise ValueError('incorrect panel exposure count for schema revision')
        _unique(self.stimuli, 'panel stimulus')


@dataclass(frozen=True)
class ReviewBundle(GraphRecord):
    schema: Literal['aimusic.review-bundle/1']
    sources: tuple[SourceNode, ...]
    renders: tuple[RenderNode, ...]
    presentations: tuple[PresentationNode, ...]
    stimuli: tuple[Stimulus, ...]
    panels: tuple[Panel, ...]
    blind_map: FileRef
    assignment_directory: str
    finding_directory: str

    def validate(self) -> None:
        if not self.blind_map.path.startswith('private/'):
            raise ValueError('blind map must be separate and private')
        for path in (self.assignment_directory, self.finding_directory):
            safe_relative(path)
        if self.assignment_directory.casefold() == self.finding_directory.casefold():
            raise ValueError('findings and assignments must be separated')
        for nodes, label in ((self.sources, 'source'), (self.renders, 'render'),
                             (self.presentations, 'presentation')):
            if not nodes:
                raise ValueError(f'missing {label} nodes')
            _unique((n.sha256 for n in nodes), label)
        sources = {s.sha256: s for s in self.sources}
        renders = {r.sha256: r for r in self.renders}
        presentations = {p.sha256: p for p in self.presentations}
        for render in self.renders:
            if render.source_sha256 not in sources:
                raise ValueError('unknown source digest')
            state = next(m.state for m in render.metrics if m.name == 'reconciliation')
            if sources[render.source_sha256].kind == 'render_package' and state == 'not_applicable':
                raise ValueError('package reconciliation is applicable')
        if any(p.render_sha256 not in renders for p in self.presentations):
            raise ValueError('unknown render digest')
        for presentation in self.presentations:
            if presentation.source_wav not in renders[presentation.render_sha256].reports:
                raise ValueError('presentation source WAV is not linked to render evidence')
        if any(s.presentation_sha256 not in presentations for s in self.stimuli):
            raise ValueError('unknown presentation digest')
        _unique((s.id for s in self.stimuli), 'stimulus')
        _unique((p.output.path.casefold() for p in self.presentations), 'media path')
        _unique((p.purpose for p in self.panels), 'panel')
        if set(p.purpose for p in self.panels) != set(PANEL_SIZES):
            raise ValueError('all five panels required')
        exposures = tuple(s for panel in self.panels for s in panel.stimuli)
        if len(exposures) != 14 or set(exposures) != {s.id for s in self.stimuli}:
            raise ValueError('schema requires exactly fourteen exposures')
        _unique(exposures, 'exposure')
        by_id = {s.id: renders[presentations[s.presentation_sha256].render_sha256] for s in self.stimuli}
        tuning_ids = next(p.stimuli for p in self.panels if p.purpose == 'tuning')
        states = [{m.name: m.state for m in by_id[s].metrics} for s in tuning_ids]
        if sorted(row['controlled_tuning'] for row in states) != ['failed', 'passed']:
            raise ValueError('tuning panel needs measured valid and failed members')
        if sorted(row['negative_control'] for row in states) != ['not_applicable', 'passed']:
            raise ValueError('tuning panel needs an admitted calibration control')

    def references(self) -> tuple[FileRef, ...]:
        def walk(value: Any) -> list[FileRef]:
            if isinstance(value, FileRef):
                return [value]
            if isinstance(value, Record):
                return [r for f in fields(value) for r in walk(getattr(value, f.name))]
            if isinstance(value, tuple):
                return [r for v in value for r in walk(v)]
            return []
        return tuple(walk(self))

    def verify(self, root: Path, *, expected_sha256: str) -> None:
        self.__post_init__()
        if self.sha256 != expected_sha256:
            raise ValueError('bundle digest mismatch')
        seen: dict[str, FileRef] = {}
        for ref in self.references():
            key = ref.path.casefold()
            if key in seen and seen[key] != ref:
                raise ValueError('conflicting file references')
            if key not in seen:
                ref.read_verified(root)
                seen[key] = ref
        PrivateBlindMap.from_bytes(self.blind_map.read_verified(root)).validate_bundle(self)

    def public_projection(self, *, root: Path, expected_sha256: str) -> dict[str, Any]:
        """Only entry point for future pages; never expose the private graph."""
        self.__post_init__()
        for render in self.renders:
            render.require_admitted()
        self.verify(root, expected_sha256=expected_sha256)
        presentations = {p.sha256: p for p in self.presentations}
        # Never expose private pair orientation or global condition-index order.
        panels = [Panel(p.purpose, tuple(sorted(p.stimuli)) if len(p.stimuli) == 2 else p.stimuli)
                  for p in self.panels]
        return {'bundle_sha256': self.sha256, 'panels': [p.to_dict() for p in panels],
                'stimuli': [{'id': s.id, 'media': presentations[s.presentation_sha256].output.to_dict()}
                            for s in sorted(self.stimuli, key=lambda s: s.id)]}


@dataclass(frozen=True)
class PrivateBlindMap(GraphRecord):
    conditions: tuple[tuple[str, str], ...]
    order_seed: str | None = None  # Private entropy; legacy maps cannot generate assignments.

    def validate(self) -> None:
        if self.order_seed is not None:
            _sha(self.order_seed)
        _unique((s for s, _ in self.conditions), 'blind stimulus')
        for _, condition in self.conditions:
            _text(condition)

    def to_dict(self) -> dict[str, Any]:
        # The optional extension must not change canonical legacy bytes/digests.
        result = super().to_dict()
        if self.order_seed is None:
            del result['order_seed']
        return result

    @classmethod
    def from_dict(cls, data: Any) -> PrivateBlindMap:
        # Two wire shapes: legacy conditions-only, or conditions plus a SHA seed.
        # Null is not a third representation of the legacy map.
        if isinstance(data, dict):
            if 'order_seed' in data and data['order_seed'] is None:
                raise ValueError('private order seed must be omitted or a SHA-256 digest')
            if set(data) == {'conditions'}:
                data = {**data, 'order_seed': None}
        return super().from_dict(data)

    def validate_bundle(self, bundle: ReviewBundle) -> None:
        if {s for s, _ in self.conditions} != {s.id for s in bundle.stimuli}:
            raise ValueError('blind map must cover the exact stimulus set')
        presentations = {p.sha256: p for p in bundle.presentations}
        renders = {r.sha256: r for r in bundle.renders}
        conditions = dict(self.conditions)
        for stimulus in bundle.stimuli:
            render = renders[presentations[stimulus.presentation_sha256].render_sha256]
            control = any(m.name == 'negative_control' and m.state == 'passed' for m in render.metrics)
            if control != (conditions[stimulus.id] == 'flattened-12-calibration-control'):
                raise ValueError('calibration control must be labelled privately')


@dataclass(frozen=True)
class AssignmentReference(GraphRecord):
    bundle_sha256: str
    reviewer: str
    document: FileRef
    media: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        _sha(self.bundle_sha256)
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}', self.reviewer):
            raise ValueError('bounded reviewer pseudonym required')
        _unique((s for s, _ in self.media), 'assigned stimulus')
        for _, sha in self.media:
            _sha(sha)

    def validate_bundle(self, bundle: ReviewBundle) -> None:
        presentations = {p.sha256: p for p in bundle.presentations}
        expected = {s.id: presentations[s.presentation_sha256].output.sha256 for s in bundle.stimuli}
        if self.bundle_sha256 != bundle.sha256 or dict(self.media) != expected:
            raise ValueError('assignment bundle/media digest mismatch')
        if not self.document.path.startswith(bundle.assignment_directory + '/'):
            raise ValueError('assignment document outside assignment directory')


@dataclass(frozen=True)
class ExternalFindingReference(GraphRecord):
    bundle_sha256: str
    assignment_sha256: str
    reviewer: str
    document: FileRef

    def validate(self) -> None:
        _sha(self.bundle_sha256)
        _sha(self.assignment_sha256)
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}', self.reviewer):
            raise ValueError('bounded reviewer pseudonym required')

    def validate_assignment(self, assignment: AssignmentReference) -> None:
        if (self.bundle_sha256 != assignment.bundle_sha256 or self.assignment_sha256 != assignment.sha256
                or self.reviewer != assignment.reviewer):
            raise ValueError('finding assignment/bundle/reviewer mismatch')
