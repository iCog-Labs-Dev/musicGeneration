"""Private deterministic candidate matrix, not a presented/published review bundle.

Consumes a retained Method A RenderPackage (from ``aimusic.app.cli generate``).
Native FluidSynth and an explicit approved SoundFont are optional prerequisites;
there is no asset discovery, fallback, presentation transform or human approval.
Fixed source MIDI, exposure order and same-runtime PCM are reproducible. Authentic
package/groove timestamps remain in provenance, so whole-run hashes may differ.

python -m aimusic.audio.review_matrix --package RUN --soundfont APPROVED.sf2 --out PRIVATE
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Literal

from aimusic.audio.config import load_audio_config
from aimusic.audio.demo_sources import SOURCE_NAMES, write_source
from aimusic.audio.from_midi import from_midi
from aimusic.audio.from_score import AudioPackageContext, from_score
from aimusic.audio.inputs import InputSnapshot
from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.orchestrator import Orchestrator
from aimusic.audio.publication import pin_output
from aimusic.audio.render.registry import file_digest, resolve_renderer
from aimusic.audio.review_admission import CandidateRequest, read_candidate
from aimusic.audio.review_bundle import FileRef, GraphRecord, Panel, RenderNode, SourceNode, load_json, read_relative

_PANELS = (
    Panel('canonical', ('fixture-12', 'generated-12', 'reference-19', 'multitrack-19')),
    Panel('roles', ('fixture-drums', 'fixture-bass', 'fixture-comping', 'fixture-lead')),
    Panel('parity', ('package-12', 'imported-12')),
    Panel('timing', ('groove-off', 'groove-on')),
    Panel('tuning', ('valid-19', 'flattened-12-calibration-control')),
)


@dataclass(frozen=True)
class Exposure(GraphRecord):
    condition: str  # Private label. Opaque presentation IDs belong to task 6.3.
    render_sha256: str
    wav: FileRef


@dataclass(frozen=True)
class ReviewMatrix(GraphRecord):
    schema: Literal['aimusic.review-matrix/1']
    sources: tuple[SourceNode, ...]
    renders: tuple[RenderNode, ...]
    exposures: tuple[Exposure, ...]
    panels: tuple[Panel, ...]

    def validate(self) -> None:
        if self.panels != _PANELS:
            raise ValueError('matrix requires the exact five panels')
        if tuple(e.condition for e in self.exposures) != tuple(s for panel in _PANELS for s in panel.stimuli):
            raise ValueError('matrix requires exactly fourteen ordered exposures')
        sources = {s.sha256 for s in self.sources}
        renders = {r.sha256: r for r in self.renders}
        if len(sources) != len(self.sources) or len(renders) != len(self.renders):
            raise ValueError('duplicate matrix source/render')
        for render in self.renders:
            if render.source_sha256 not in sources:
                raise ValueError('unknown matrix source')
            render.require_admitted()
            if next(m.state for m in render.metrics if m.name == 'polyphonic_f0') != 'unmeasured':
                raise ValueError('matrix does not measure polyphonic f0')
        for exposure in self.exposures:
            linked = renders.get(exposure.render_sha256)
            if linked is None or exposure.wav not in linked.reports:
                raise ValueError('exposure must reference admitted render media')


def _ref(root: Path, path: str) -> FileRef:
    import hashlib
    data = read_relative(root, path)
    return FileRef(path, hashlib.sha256(data).hexdigest(), len(data))


def generate_matrix(output: Path, *, package: Path, soundfont: Path) -> ReviewMatrix:
    """Generate seven admitted renders and exactly fourteen private exposure slots.

    ``output`` must be a fresh, trusted local work directory. A failure may retain
    diagnostic source/renders, but never writes matrix.json. This is not atomic
    bundle publication (task 6.5). No source or renderer WAV is transformed.
    """
    config = load_audio_config('listening').to_dict()
    config['paths']['soundfont'] = str(Path(soundfont).absolute())
    config['stages']['expressivization'] = False
    config['groove'] = {'spec': {'swing': {'ratio': .6, 'subdivision': 8, 'applies_to': ['lead']}},
                        'light_cleanup': False}
    selected = resolve_renderer('lead', config=config, tuning='channel_pitch', soundfont=Path(soundfont).absolute())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'sources').mkdir()
    snapshot = InputSnapshot.capture(Path(package), root / 'sources/generated-12', package=True)
    generated = from_score(root / 'sources/generated-12')
    if generated.structure.edo != 12:
        raise ValueError('generated-12 requires a 12-EDO RenderPackage')
    for name in SOURCE_NAMES:
        write_source(name, root / 'sources' / name)
    original = 'sources/reference-19/source.mid'
    intent = _ref(root, 'sources/reference-19/expected.json')
    flat = MidiEvents.from_bytes(read_relative(root, original)).to_mido()
    for track in flat.tracks:
        for index, message in enumerate(track):
            if message.type == 'pitchwheel':
                track[index] = message.copy(pitch=0)
    # Only invocation-private sources are written here; original MIDI/intent stay unchanged.
    flat_path = root / 'sources/flattened-12.mid'
    with flat_path.open('xb') as stream:
        flat.save(file=stream)

    sources: dict[str, SourceNode] = {}
    renders: dict[str, RenderNode] = {}
    media: dict[str, tuple[FileRef, ...]] = {}
    inputs = (
        ('fixture-12', from_midi(root / 'sources/fixture-12/source.mid')),
        ('generated-12', generated),
        ('reference-19', from_midi(root / original)),
        ('multitrack-19', from_midi(root / 'sources/multitrack-19/source.mid')),
        ('imported-12', from_midi(generated.package.midi_path)),
        ('groove-on', from_midi(root / 'sources/fixture-12/source.mid')),
        ('flattened-12', from_midi(flat_path)),
    )
    for name, context in inputs:
        run_config = deepcopy(config)
        run_config['stages']['expressivization'] = name == 'groove-on'
        out = f'renders/{name}'
        try:
            Orchestrator(run_config, root / 'cache').run(context, output_dir=root / out)
            is_package = isinstance(context, AudioPackageContext)
            midi = context.package.midi_path if isinstance(context, AudioPackageContext) else context.midi_path
            identity = {'midi_sha256': file_digest(midi), **context.verified_identity}
            if isinstance(context, AudioPackageContext):
                identity.update(structure_sha256=file_digest(context.package.structure_path),
                                tuning_sha256=file_digest(context.package.tuning_path))
            source = context.package.root if isinstance(context, AudioPackageContext) else midi
            request = CandidateRequest(out, source.relative_to(root).as_posix(),
                                       'render_package' if is_package else 'imported_midi', identity,
                                       file_digest(root / out / 'render.json'), 'fluidsynth', selected.soundfont_sha256,
                                       intent=intent if name in ('reference-19', 'flattened-12') else None,
                                       original_midi=_ref(root, original) if name == 'flattened-12' else None,
                                       flattened=name == 'flattened-12',
                                       cache_manifest_sha256=file_digest(root / out / 'stems/cache_manifest.json'))
            candidate = read_candidate(root, request)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f'matrix candidate {name} failed: {exc}') from exc
        sources[candidate.source.sha256] = candidate.source
        renders[name], media[name] = candidate.render, candidate.media

    def exposure(condition: str, run: str, filename: str = 'rough_mix.wav') -> Exposure:
        wav = next(ref for ref in media[run] if Path(ref.path).name == filename)
        return Exposure(condition, renders[run].sha256, wav)

    exposures = [exposure(name, name) for name in _PANELS[0].stimuli]
    index = load_json(read_relative(root, 'renders/fixture-12/stems/role_index.json'))
    roles = {row['role']: row['wav'] for row in index['stems']}
    exposures += [exposure('fixture-' + role, 'fixture-12', roles[role]) for role in ('drums', 'bass', 'comping', 'lead')]
    exposures += [exposure(condition, run) for condition, run in (
        ('package-12', 'generated-12'), ('imported-12', 'imported-12'),
        ('groove-off', 'fixture-12'), ('groove-on', 'groove-on'),
        ('valid-19', 'reference-19'), ('flattened-12-calibration-control', 'flattened-12'),
    )]
    matrix = ReviewMatrix('aimusic.review-matrix/1', tuple(sources.values()), tuple(renders.values()),
                          tuple(exposures), _PANELS)
    snapshot.verify()
    selected.verify_asset()
    with tempfile.TemporaryDirectory(prefix='review-matrix-') as tmp:
        staging = Path(tmp)
        (staging / 'matrix.json').write_bytes(matrix.to_bytes())
        with pin_output(root) as pinned:
            pinned.publish(staging, ['matrix.json'], replace=False)
    return matrix


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, help='retained generated 12-EDO RenderPackage')
    parser.add_argument('--soundfont', type=Path, required=True, help='explicit local license-approved SoundFont')
    parser.add_argument('--out', type=Path, required=True, help='fresh private candidate directory; no overwrites')
    args = parser.parse_args()
    try:
        generate_matrix(args.out, package=args.package, soundfont=args.soundfont)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(args.out / 'matrix.json')


if __name__ == '__main__':
    main()
