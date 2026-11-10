"""Versioned, asset-free listening fixtures (not a replacement for Method A).

python -m aimusic.audio.demo_sources fixture-12 --seed 11 --out outputs/demo/fixture
The fourth source is freshly composed by the existing symbolic CLI; see
``docs/audio-demo-sources.md``. Source revisions change when musical intent does.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import tempfile
from typing import Any

import mido

from aimusic.core.rng import RNGKey, randint

SOURCE_NAMES = ('fixture-12', 'reference-19', 'multitrack-19')
REVISION = 'aimusic.listening-sources/1'
PPQN = 480


@dataclass(frozen=True)
class _Note:
    track: int
    channel: int
    start_tick: int
    end_tick: int
    step: int  # MIDI key in 12-EDO; signed 19-EDO step relative to A4 otherwise.
    velocity: int = 96


def _fixture(key: RNGKey) -> tuple[list[_Note], RNGKey]:
    notes = []
    for bar, root in enumerate((48, 53, 55, 48) * 2):
        start = bar * 1920
        for beat in range(4):
            notes.append(_Note(1, 9, start + beat * 480, start + beat * 480 + 120,
                               36 if beat % 2 == 0 else 38))
        for eighth in range(8):
            at = start + eighth * 240
            notes.append(_Note(1, 9, at, at + 90, 42, 72))
            velocity, key = randint(key, 82, 105)
            notes.append(_Note(4, 2, at, at + 180, root + (24, 28, 31, 28, 26, 29, 31, 26)[eighth], velocity))
        for half in range(2):
            at = start + half * 960
            notes.append(_Note(2, 0, at, at + 840, root - 12))
            for interval in (0, 4, 7):
                notes.append(_Note(3, 1, at, at + 840, root + 12 + interval, 80))
    return notes, key


def _micro_notes(reference: bool) -> list[_Note]:
    if reference:
        # 1-second tones with 0.5-second gaps; all substantially off 12-TET.
        return [_Note(1, 1, i * 1440, i * 1440 + 960, step) for i, step in enumerate((1, 4, 7))]
    notes = []
    for bar in range(2):
        at = bar * 1920
        notes.append(_Note(1, 1, at, at + 1680, -37 + bar * 3, 90))
        # Independent voices in one logical comping track; never share a channel.
        for channel, step in ((2, -18), (3, -12), (4, -7)):
            notes.append(_Note(2, channel, at, at + 1680, step + bar * 3, 76))
        for beat, step in enumerate((1, 4, 7, 11)):
            notes.append(_Note(3, 5, at + beat * 480, at + beat * 480 + 360, step + bar * 3))
    return notes


def _pitch(note: _Note, edo: int) -> tuple[int, int]:
    if edo == 12:
        return note.step, 0
    pitch = 69 + note.step * 12 / 19
    key = round(pitch)
    return key, round((pitch - key) / 2 * 8192)


def _track(events: list[tuple[int, Any]], end: int) -> Any:
    result = mido.MidiTrack()
    previous = 0
    # Stable same-tick order: setup, then per-note wheel, then note-on.
    for tick, message in sorted(events, key=lambda row: row[0]):
        result.append(message.copy(time=tick - previous))
        previous = tick
    result.append(mido.MetaMessage('end_of_track', time=end - previous))
    return result


def build_source(name: str, *, seed: int = 11) -> tuple[bytes, dict[str, Any]]:
    """Return deterministic SMF and original intent, without reading output.

    Only fixture-12 consumes RNG (velocity variation). The two controlled 19-EDO
    sources intentionally have fixed pitch/time/velocity targets, independent of
    seed. All random choices thread the immutable core RNGKey explicitly.
    """
    if name not in SOURCE_NAMES:
        raise ValueError(f'Unknown demo source {name!r}; choose from {SOURCE_NAMES}')
    key = RNGKey(seed)
    edo = 12 if name == 'fixture-12' else 19
    tracks: tuple[tuple[str, int], ...]
    if edo == 12:
        notes, _ = _fixture(key)
        tracks = (('drums', 0), ('bass', 33), ('comping', 0), ('lead', 80))
        end = 15360
    else:
        reference = name == 'reference-19'
        notes = _micro_notes(reference)
        tracks = (('reference', 80),) if reference else (('bass', 33), ('comping', 0), ('lead', 80))
        end = 4320 if reference else 3840
    notes.sort(key=lambda n: (n.start_tick, n.track, n.channel, n.step))
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQN)
    midi.tracks.append(_track([(0, mido.MetaMessage('track_name', name=name)),
                              (0, mido.MetaMessage('set_tempo', tempo=500000)),
                              (0, mido.MetaMessage('time_signature', numerator=4, denominator=4))], end))
    for index, (label, program) in enumerate(tracks, 1):
        own = [note for note in notes if note.track == index]
        events: list[tuple[int, Any]] = [(0, mido.MetaMessage('track_name', name=label))]
        for channel in sorted({n.channel for n in own}):
            events.append((0, mido.Message('program_change', channel=channel, program=program)))
            if edo == 19:
                for control, value in ((101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127)):
                    events.append((0, mido.Message('control_change', channel=channel, control=control, value=value)))
        for note in own:
            pitch, wheel = _pitch(note, edo)
            if edo == 19:
                events.append((note.start_tick, mido.Message('pitchwheel', channel=note.channel, pitch=wheel)))
            events.append((note.start_tick, mido.Message('note_on', channel=note.channel, note=pitch,
                                                        velocity=note.velocity)))
            events.append((note.end_tick, mido.Message('note_off', channel=note.channel, note=pitch, velocity=0)))
        midi.tracks.append(_track(events, end))
    rows = []
    for note in notes:
        row = asdict(note)
        row.pop('step')
        row['midi_note'] = _pitch(note, edo)[0]
        row['edo_step_from_a4'] = note.step if edo == 19 else note.step - 69
        # Expected Hz never comes from MIDI wheels or rendered audio.
        row['expected_hz'] = None if note.channel == 9 else 440 * 2 ** (row['edo_step_from_a4'] / edo)
        rows.append(row)
    windows = []
    if name == 'reference-19':
        windows = [{'name': f'A4+{n.step}/19', 'start_sample': i * 72000 + 12000,
                    'stop_sample': i * 72000 + 36000, 'expected_hz': 440 * 2 ** (n.step / 19),
                    'percussion': False} for i, n in enumerate(notes)]
    intent = {'schema': REVISION, 'source': name, 'edo': edo, 'reference_hz': 440,
              'reference_midi_note': 69, 'sample_rate': 48000, 'notes': rows, 'windows': windows,
              'window_scope': 'original dry mono timing; no automatic retiming',
              'audio_pitch_accuracy': 'unmeasured', 'planner_reconciliation': 'not_applicable'}
    stream = io.BytesIO()
    midi.save(file=stream)
    return stream.getvalue(), intent


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False) + '\n').encode('ascii')


def write_source(name: str, output: Path, *, seed: int = 11) -> Path:
    """Exclusively publish source + intent + digest manifest; never overwrite.

    Reuse the renderer's pinned-directory/exclusive publication safety. Failure
    rolls back this call's files; a process crash can leave partial files, which
    a later call refuses rather than reusing. An empty output directory may remain.
    """
    midi, intent = build_source(name, seed=seed)
    from aimusic.audio.publication import pin_output
    payloads = {'source.mid': midi, 'expected.json': _canonical(intent)}
    payloads['source.json'] = _canonical({
        'schema': REVISION, 'source': name, 'seed': seed if name == 'fixture-12' else None,
        'generator': 'python -m aimusic.audio.demo_sources',
        'files': {filename: hashlib.sha256(data).hexdigest() for filename, data in payloads.items()},
        'tuning_requirement': 'direct_12' if name == 'fixture-12' else 'channel_pitch',
    })
    with pin_output(output) as pinned:
        pinned.check_new_files(list(payloads))
        with tempfile.TemporaryDirectory(prefix='aimusic-demo-') as tmp:
            staging = Path(tmp)
            for filename, data in payloads.items():
                (staging / filename).write_bytes(data)
            pinned.publish(staging, list(payloads), replace=False)
    return output / 'source.mid'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCE_NAMES)
    parser.add_argument('--seed', type=int, default=11, help='fixture-12 velocities only (default: 11)')
    parser.add_argument('--out', type=Path, required=True, help='exact output directory; no overwrites')
    args = parser.parse_args()
    try:
        path = write_source(args.source, args.out, seed=args.seed)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(path)


if __name__ == '__main__':
    main()
