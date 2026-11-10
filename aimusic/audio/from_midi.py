"""Read-only imported-MIDI admission; no Score, planner claims or DSP imports."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from aimusic.audio.analysis.from_midi import infer_source_tracks
from aimusic.audio.midi_events import MidiEvents, MidiEventError

# Bound derived analysis memory/work, not MIDI resolution or denominator alone.
_MAX_BAR_BOUNDARIES = 100_000


@dataclass(frozen=True)
class ImportedMidiContext:
    midi_path: Path
    source_sha256: str
    events: MidiEvents
    provenance: str = 'inferred'

    @property
    def run_id(self) -> str:
        return f'imported_{self.source_sha256[:16]}'

    @property
    def verified_identity(self) -> dict[str, str]:
        return {'midi_sha256': self.source_sha256, 'input_kind': 'imported_midi'}

    @property
    def reconciliation(self) -> dict[str, str]:
        return {'status': 'not_applicable', 'reason': 'Imported MIDI has no planner/Score metadata'}

    def analysis_document(self) -> dict[str, Any]:
        source = self.events
        tempo = dict([(0, 500000), *source.tempo_map.changes])
        meters = _meter_map(source)
        evidence = []
        for event in source.events:
            msg = event.message
            if msg.type in ('pitchwheel', 'sysex') or (msg.type == 'control_change' and
                    msg.control in (6, 38, 96, 97, 98, 99, 100, 101, 121)):
                evidence.append({'tick': event.tick, 'track': event.track, 'message': msg.dict()})
        end = max(e.tick for e in source.events)
        return {
            'schema': 'aimusic.imported-midi/1', 'provenance': self.provenance,
            'origin': 'inferred', 'source_midi': self.midi_path.name,
            'midi_type': source.type, 'resolution': source.ticks_per_beat,
            'verified_identity': self.verified_identity, 'inputs_hash': self.source_sha256,
            'planner_reconciliation': self.reconciliation,
            'tempo_bpm': 60000000 / tempo[0],
            'tempo_map': [[float(source.tempo_map.seconds(t)), 60000000 / us] for t, us in tempo.items()],
            'meter_map': [[t, n, d] for t, (n, d) in meters.items()],
            'time_signature': dict(zip(('numerator', 'denominator'), meters[0])),
            'duration_sec': float(source.tempo_map.seconds(end)),
            'bar_boundaries_sec': _bar_boundaries(source, meters, end),
            'tracks': infer_source_tracks(source), 'total_notes': len(source.notes),
            'tuning': {'source': 'MIDI events', 'intended_lattice': 'unknown',
                       'audio_pitch_accuracy': 'unmeasured', 'events': evidence},
        }


def _meter_map(source: MidiEvents) -> dict[int, tuple[int, int]]:
    meters = {0: (4, 4)}
    for event in source.events:
        if event.message.type == 'time_signature':
            meters[event.tick] = (event.message.numerator, event.message.denominator)
    return meters


def _check_bar_expansion(source: MidiEvents, meters: dict[int, tuple[int, int]], end: int) -> None:
    """Count exactly with integers before allocating any expanded boundaries."""
    count = 1  # The final endpoint; each nonempty segment contributes its start.
    changes = list(meters.items())
    for i, (start, (numerator, denominator)) in enumerate(changes):
        if numerator <= 0 or denominator <= 0:
            raise MidiEventError(f'Invalid meter at tick {start}: numerator and denominator must be positive')
        stop = min(end, changes[i + 1][0]) if i + 1 < len(changes) else end
        bar_ticks = source.ticks_per_beat * numerator * 4
        count += (max(0, stop - start) * denominator + bar_ticks - 1) // bar_ticks
        if count > _MAX_BAR_BOUNDARIES:
            raise MidiEventError(
                f'Imported MIDI meter at tick {start} exceeds {_MAX_BAR_BOUNDARIES} bar boundaries; '
                'use a smaller meter denominator or a shorter MIDI excerpt')


def _bar_boundaries(source: MidiEvents, meters: dict[int, tuple[int, int]], end: int) -> list[float]:
    """A meter change starts a new inferred bar, including mid-bar changes."""
    _check_bar_expansion(source, meters, end)
    ticks = [Fraction(0)]
    changes = list(meters.items())
    for i, (start, (numerator, denominator)) in enumerate(changes):
        stop = min(end, changes[i + 1][0]) if i + 1 < len(changes) else end
        at = Fraction(start)
        if at > ticks[-1]:
            ticks.append(at)
        step = Fraction(source.ticks_per_beat * numerator * 4, denominator)
        while at + step < stop:
            at += step
            ticks.append(at)
    if end > ticks[-1]:
        ticks.append(Fraction(end))
    return [float(source.tempo_map.seconds(t)) for t in ticks]


def from_midi(midi_path: Path | str) -> ImportedMidiContext:
    """Admit one byte snapshot; the orchestrator independently rechecks its identity."""
    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f'MIDI file not found: {path}')
    payload = path.read_bytes()
    try:
        events = MidiEvents.from_bytes(payload)
    except (EOFError, ValueError, KeyError, IndexError, OSError) as exc:
        raise MidiEventError(f'Malformed MIDI or unsupported input {path}: {exc}') from exc
    if not events.notes:
        raise MidiEventError('No paired notes in imported MIDI; provide a nonempty type-0/type-1 source')
    _check_bar_expansion(events, _meter_map(events), max(e.tick for e in events.events))
    events.isolate_tracks()  # Reject cross-track controllers that corrupt another instrument's channel.
    return ImportedMidiContext(path, hashlib.sha256(payload).hexdigest(), events)
