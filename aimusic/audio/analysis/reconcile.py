"""Read-only planner/MIDI reconciliation, before any expressive transforms.

MIDI track chunks, not PrettyMIDI instruments, own source identity. This keeps
MPE member channels together and requires no synthesizer or DSP imports.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import mido

from aimusic.core.render_package import ContractViolation

if TYPE_CHECKING:
    from aimusic.audio.from_score import AudioPackageContext

TEMPO_TOLERANCE_BPM = 0.001


@dataclass(frozen=True)
class FieldCheck:
    track: str | None
    field: str
    expected: Any
    observed: Any
    tolerance: float
    outcome: str


@dataclass(frozen=True)
class ReconciliationReport:
    checks: tuple[FieldCheck, ...]
    tracks: tuple[dict[str, Any], ...]
    score_occupancy_sec: float
    planner_extent_sec: float

    @property
    def passed(self) -> bool:
        return all(c.outcome != "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "source": "original MIDI", **asdict(self)}


class ReconciliationError(ContractViolation):
    def __init__(self, report: ReconciliationReport):
        self.report = report
        failures = [
            f"{c.track or 'source'} {c.field}: expected={c.expected!r}, "
            f"observed={c.observed!r}, tolerance={c.tolerance}"
            for c in report.checks if c.outcome == "fail"
        ]
        super().__init__("MIDI reconciliation failed: " + "; ".join(failures))


@dataclass(frozen=True)
class _Tempo:
    ppq: int
    changes: tuple[tuple[float, float], ...]  # (absolute tick, BPM)

    def seconds(self, tick: float) -> float:
        result = 0.0
        previous_tick, bpm = self.changes[0]
        for at, next_bpm in self.changes[1:]:
            if at > tick:
                break
            result += (at - previous_tick) * 60.0 / bpm / self.ppq
            previous_tick, bpm = at, next_bpm
        return result + (tick - previous_tick) * 60.0 / bpm / self.ppq

    def one_tick(self, tick: float) -> float:
        bpm = self.changes[0][1]
        for at, value in self.changes:
            if at > tick:
                break
            bpm = value
        return 60.0 / bpm / self.ppq


def _within_tolerance(expected: float, observed: float, tolerance: float) -> bool:
    return abs(expected - observed) <= tolerance + 1e-9


def _match_note_intervals(
    expected: list[tuple[int, int]], observed: list[tuple[int, int]],
    planned: _Tempo, tempo: _Tempo,
) -> dict[int, int]:
    """Maximum one-to-one endpoint matching, deterministic in sorted input order.

    An augmenting path can move an earlier match so a later note with fewer
    candidates is not stranded. Channels stay aggregated by logical track.
    """
    expected_seconds = [(planned.seconds(start), planned.seconds(end)) for start, end in expected]
    observed_bounds = [(tempo.seconds(start), tempo.seconds(end), tempo.one_tick(start), tempo.one_tick(end))
                       for start, end in observed]
    candidates = [
        [j for j, (actual_start, actual_end, start_tol, end_tol) in enumerate(observed_bounds)
         if _within_tolerance(start, actual_start, start_tol) and _within_tolerance(end, actual_end, end_tol)]
        for start, end in expected_seconds
    ]
    matches: dict[int, int] = {}
    owners: dict[int, int] = {}
    for root in range(len(expected)):
        pending = [root]
        parents: dict[int, int] = {}
        free: int | None = None
        for i in pending:
            for j in candidates[i]:
                if j in parents:
                    continue
                parents[j] = i
                if j not in owners:
                    free = j
                    break
                pending.append(owners[j])
            if free is not None:
                break
        while free is not None:
            i = parents[free]
            previous = matches.get(i)
            matches[i] = free
            owners[free] = i
            free = previous
    return matches


def _effective_map(events: list[tuple[int, Any]], default: Any, field: str) -> list[tuple[int, Any]]:
    by_tick: dict[int, Any] = {}
    for tick, value in sorted(events, key=lambda e: e[0]):
        if tick in by_tick and by_tick[tick] != value:
            raise ContractViolation(f"conflicting MIDI {field} at tick {tick}")
        by_tick[tick] = value
    result = [(0, by_tick.pop(0, default))]
    for tick, value in by_tick.items():
        if value != result[-1][1]:
            result.append((tick, value))
    return result


def _planned_tempo(rows: list[list[float]], ppq: int) -> _Tempo:
    changes: list[tuple[float, float]] = []
    tick, previous_sec, previous_bpm = 0.0, 0.0, 120.0
    for index, (sec, bpm) in enumerate(rows):
        if not (math.isfinite(sec) and math.isfinite(bpm) and bpm > 0):
            raise ContractViolation(f"invalid structure tempo_map[{index}]: {rows[index]!r}")
        if (index == 0 and sec != 0) or (index > 0 and sec <= previous_sec):
            raise ContractViolation("structure tempo_map must start at zero and increase strictly")
        tick += (sec - previous_sec) * previous_bpm / 60.0 * ppq
        changes.append((tick, bpm))
        previous_sec, previous_bpm = sec, bpm
    if not changes:
        raise ContractViolation("structure tempo_map is empty")
    return _Tempo(ppq, tuple(changes))


def _planned_meter(rows: list[list[Any]], ppq: int) -> list[tuple[float, str]]:
    changes: list[tuple[float, str]] = []
    tick, previous_bar, beats = 0.0, 0, 4.0
    for index, (bar, label) in enumerate(rows):
        try:
            numerator, denominator = map(int, label.split('/'))
            valid = numerator > 0 and denominator > 0 and denominator & (denominator - 1) == 0
            valid = valid and bar == int(bar) and bar >= 0
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ContractViolation(f"invalid structure meter[{index}]: {rows[index]!r}")
        if (index == 0 and bar != 0) or (index > 0 and bar <= previous_bar):
            raise ContractViolation("structure meter must start at bar zero and increase strictly")
        tick += (bar - previous_bar) * beats * ppq
        changes.append((tick, label))
        previous_bar, beats = bar, numerator * 4.0 / denominator
    if not changes:
        raise ContractViolation("structure meter is empty")
    return changes


def _midi_observations(path: Any) -> tuple[Any, list[dict[str, Any]], _Tempo, list[tuple[int, str]], int]:
    try:
        midi = mido.MidiFile(path)
    except (OSError, ValueError, EOFError) as exc:
        raise ContractViolation(f"invalid source MIDI {path}: {exc}") from exc
    if midi.type not in (0, 1) or midi.ticks_per_beat <= 0:
        raise ContractViolation("source MIDI requires type 0/1 and positive ticks_per_beat")
    tracks: list[dict[str, Any]] = []
    events = []
    tempos = []
    meters = []
    conductor_end = 0
    for index, track in enumerate(midi.tracks):
        tick = 0
        names = {msg.name for msg in track if msg.type == 'track_name'}
        if len(names) > 1:
            raise ContractViolation(f"source track {index}: conflicting track_name events")
        name = next(iter(names), '')
        has_notes = any(msg.type == 'note_on' and msg.velocity > 0 for msg in track)
        last_payload_tick = 0
        for order, msg in enumerate(track):
            tick += msg.time
            if msg.type != 'end_of_track':
                last_payload_tick = tick
            events.append((tick, index, order, msg))
            if msg.type == 'set_tempo':
                if msg.tempo <= 0:
                    raise ContractViolation(f"invalid MIDI tempo at tick {tick}")
                tempos.append((tick, float(mido.tempo2bpm(msg.tempo))))
            if msg.type == 'time_signature':
                meters.append((tick, f"{msg.numerator}/{msg.denominator}"))
        if not has_notes:
            conductor_end = max(conductor_end, last_payload_tick)
        tracks.append({"name": name, "index": index, "end": tick, "notes": [], "programs": []})
    programs = [0] * 16
    program_changes: list[tuple[int, int]] = []
    active: dict[tuple[int, int], tuple[int, int]] = {}
    owners: dict[int, int] = {}
    for tick, index, _, msg in sorted(events, key=lambda e: e[:3]):
        if msg.type == 'program_change':
            programs[msg.channel] = msg.program
            program_changes.append((msg.channel, msg.program))
            tracks[index]['programs'].append((msg.channel, msg.program))
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.channel, msg.note)
            if key in active:
                raise ContractViolation(f"track {tracks[index]['name']!r}: ambiguous note pairing {key}")
            if msg.channel in owners and owners[msg.channel] != index:
                raise ContractViolation(f"source channel {msg.channel} belongs to multiple tracks")
            owners[msg.channel] = index
            active[key] = (tick, index)
            tracks[index]['programs'].append((msg.channel, programs[msg.channel]))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            start = active.pop(key, None)
            if start is None or start[1] != index or tick <= start[0]:
                raise ContractViolation(f"track {tracks[index]['name']!r}: unmatched note endpoint {key}")
            tracks[index]['notes'].append((start[0], tick, msg.channel))
    if active:
        raise ContractViolation(f"source MIDI has unterminated notes: {sorted(active)}")
    for source in tracks:
        source['program_changes'] = [p for ch, p in program_changes if owners.get(ch) == source['index']]
        source['programs'].extend((ch, p) for ch, p in program_changes if owners.get(ch) == source['index'])
    tempo = _Tempo(midi.ticks_per_beat, tuple(_effective_map(tempos, 120.0, 'tempo')))
    return midi, tracks, tempo, _effective_map(meters, '4/4', 'meter'), conductor_end


def reconcile(ctx: AudioPackageContext) -> ReconciliationReport:
    """Compare original MIDI to Score counts/ticks and planner declarations.

    Raises ReconciliationError with a complete field report for mismatches;
    malformed or ambiguous sources raise ContractViolation before comparison.
    """
    midi, observed, tempo, meters, conductor_end = _midi_observations(ctx.package.midi_path)
    planned = _planned_tempo(ctx.structure.tempo_map, ctx.score.ticks_per_beat)
    planned_meters = _planned_meter(ctx.structure.meter, ctx.score.ticks_per_beat)
    checks: list[FieldCheck] = []
    tracks: list[dict[str, Any]] = []

    def check(field: str, expected: Any, actual: Any, tolerance: float = 0.0,
              track: str | None = None) -> bool:
        same = expected == actual if tolerance == 0 else _within_tolerance(expected, actual, tolerance)
        checks.append(FieldCheck(track, field, expected, actual, tolerance, 'pass' if same else 'fail'))
        return same

    check('ticks_per_beat', ctx.score.ticks_per_beat, midi.ticks_per_beat)
    check('score.tempo_bpm', ctx.score.tempo_bpm, tempo.changes[0][1], TEMPO_TOLERANCE_BPM)
    check('tempo_map.count', len(planned.changes), len(tempo.changes))
    for i, ((expected_tick, bpm), (tick, actual_bpm)) in enumerate(zip(planned.changes, tempo.changes)):
        check(f'tempo_map[{i}].bpm', bpm, actual_bpm, TEMPO_TOLERANCE_BPM)
        check(f'tempo_map[{i}].time_sec', planned.seconds(expected_tick), tempo.seconds(tick), tempo.one_tick(tick))
    check('meter.count', len(planned_meters), len(meters))
    bar_meters = [(float(tick), label) for tick, label in meters]
    for i, ((expected_tick, label), (tick, actual_label)) in enumerate(zip(planned_meters, meters)):
        signature_matches = check(f'meter[{i}].signature', label, actual_label)
        timing_matches = check(f'meter[{i}].time_sec', planned.seconds(expected_tick),
                               tempo.seconds(tick), tempo.one_tick(tick))
        if signature_matches and timing_matches:
            # Keep raw timing in the report, but apply an accepted change at its
            # declared boundary so one late tick cannot delay it an entire bar.
            bar_meters[i] = (expected_tick, actual_label)

    declared_names = [t['name'] for t in ctx.structure.tracks]
    source_tracks = [t for t in observed if t['notes'] or t['name'] in declared_names]
    check('track_names', sorted(declared_names), sorted(t['name'] for t in source_tracks))
    check('score.track_names', sorted(ctx.track_note_counts), sorted(declared_names))
    expected_end = conductor_end
    for declaration in ctx.structure.tracks:
        name = declaration['name']
        candidates = [t for t in source_tracks if t['name'] == name]
        if len(candidates) != 1:
            check('source_track_identity', 1, len(candidates), track=name)
            continue
        source = candidates[0]
        role = ctx.track_roles[name]
        channels = sorted({n[2] for n in source['notes']})
        notes = sorted((n[0], n[1]) for n in source['notes'])
        expected_notes = sorted((n.ton, n.toff) for n in ctx.score if n.track == name)
        check('note_count', len(expected_notes), len(notes), track=name)
        for channel in channels:
            check(f'channel[{channel}].percussion', declaration['percussion'], channel == 9, track=name)
        if declaration['percussion'] or role == 'drums':
            check('drum.percussion', True, declaration['percussion'], track=name)
            check('drum.program', None, declaration['program'], track=name)
            check('drum.program_changes', [], source['program_changes'], track=name)
            check('drum.microtonal', False, declaration.get('microtonal'), track=name)
        else:
            used_programs = sorted({p for ch, p in source['programs'] if ch in channels})
            check('program', [declaration['program']], used_programs, track=name)
        inferred = next((r for r in ('drums', 'bass', 'lead', 'comp', 'pads', 'ornament', 'piano')
                         if r in name.lower()), None)
        if inferred == 'comp':
            inferred = 'comping'
        checks.append(FieldCheck(name, 'role_inference', role, inferred, 0.0,
                                 'unavailable' if inferred is None else 'informational'))
        matches = _match_note_intervals(expected_notes, notes, planned, tempo)
        check('note_matching.count', len(expected_notes), len(matches), track=name)
        # Pair unmatched leftovers only for endpoint diagnostics; never reuse an observation.
        unmatched = [i for i in range(len(expected_notes)) if i not in matches]
        unused = sorted(set(range(len(notes))) - set(matches.values()))
        matches.update(zip(unmatched, unused))
        for i, j in sorted(matches.items()):
            start, end = expected_notes[i]
            actual_start, actual_end = notes[j]
            for label, expected_tick, actual_tick in (('start', start, actual_start), ('end', end, actual_end)):
                check(f'note[{i}].{label}_sec', planned.seconds(expected_tick),
                      tempo.seconds(actual_tick), tempo.one_tick(actual_tick), name)
        last_note = max((end for _, end in expected_notes), default=0)
        check('end_of_track_sec', planned.seconds(last_note), tempo.seconds(source['end']),
              tempo.one_tick(source['end']), name)
        expected_end = max(expected_end, last_note)
        tracks.append({**declaration, 'role': role, 'source_track_index': source['index'],
                       'channels': channels, 'note_count': len(expected_notes),
                       'role_source': ctx.role_source, 'count_source': ctx.count_source})
    observed_end = max((t['end'] for t in observed), default=0)
    # Conductor automation may outlast the notes; planner silence is not an EOT.
    check('end_of_track_sec', tempo.seconds(expected_end), tempo.seconds(observed_end),
          tempo.one_tick(observed_end))
    score_end = max((n.toff for n in ctx.score), default=0)
    note_end = max((n[1] for t in observed for n in t['notes']), default=0)
    occupancy = planned.seconds(score_end)
    check('score_occupancy_sec', occupancy, tempo.seconds(note_end), tempo.one_tick(note_end))

    bars = ctx.structure.bar_table
    check('bar_table.indices', list(range(len(bars))), [b[0] for b in bars])
    tick, meter_index = 0.0, 0
    for i, (_, sec) in enumerate(bars):
        while meter_index + 1 < len(bar_meters) and bar_meters[meter_index + 1][0] <= tick:
            meter_index += 1
        check(f'bar_table[{i}].time_sec', sec, tempo.seconds(tick), tempo.one_tick(tick))
        numerator, denominator = map(int, bar_meters[meter_index][1].split('/'))
        tick += midi.ticks_per_beat * numerator * 4.0 / denominator
    extent = float(bars[-1][1]) if bars else 0.0
    check('planner_extent_covers_score', True, occupancy <= extent + tempo.one_tick(score_end) + 1e-9)
    if ctx.structure.sections:
        section_bounds = [s['bars'] for s in ctx.structure.sections]
        check('sections.end_bar', len(bars) - 1, max(end for _, end in section_bounds))
        check('sections.bar_indices_valid', True,
              all(start == int(start) and end == int(end) and 0 <= start < end <= len(bars) - 1
                  for start, end in section_bounds))
    report = ReconciliationReport(tuple(checks), tuple(tracks), occupancy, extent)
    if not report.passed:
        raise ReconciliationError(report)
    return report
