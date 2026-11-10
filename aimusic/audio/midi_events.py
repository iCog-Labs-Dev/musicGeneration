"""Audio-owned, immutable SMF events. No note-only serialization or DSP.

Times are absolute source ticks; message payloads are frozen and time-free.
Equal-tick order is (source track, source order). Type 2, SMPTE, multiple
ports and ambiguous note/channel ownership are deliberately unsupported.
"""
from __future__ import annotations

import io
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from functools import cached_property
from fractions import Fraction
from pathlib import Path
from typing import Any

import mido
from mido.frozen import freeze_message


class MidiEventError(ValueError):
    """Malformed or ambiguous source MIDI."""


class ChannelStateConflict(MidiEventError):
    """Retiming cannot preserve the source channel state."""


def rational(value: float | int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


@dataclass(frozen=True)
class MidiEvent:
    tick: int
    track: int
    order: int
    message: Any
    note_ids: tuple[int, ...] = ()
    sort_order: Fraction | None = None

    @property
    def channel(self) -> int | None:
        return getattr(self.message, 'channel', None)

    @property
    def key(self) -> tuple[int, int, Fraction]:
        return self.tick, self.track, self.sort_order if self.sort_order is not None else Fraction(self.order)


@dataclass(frozen=True)
class MidiNote:
    id: int
    track: int
    channel: int
    pitch: int
    velocity: int
    start: int
    end: int
    sounding_end: int
    on: int
    off: int


@dataclass(frozen=True)
class TempoMap:
    ticks_per_beat: int
    changes: tuple[tuple[int, int], ...]

    @cached_property
    def _segments(self) -> tuple[tuple[int, ...], tuple[Fraction, ...], tuple[int, ...]]:
        ticks, seconds, tempos = [0], [Fraction(0)], [500000]
        for at, tempo in self.changes:
            seconds.append(seconds[-1] + Fraction((at - ticks[-1]) * tempos[-1],
                                                   1000000 * self.ticks_per_beat))
            ticks.append(at)
            tempos.append(tempo)
        return tuple(ticks), tuple(seconds), tuple(tempos)

    def seconds(self, tick: int | Fraction) -> Fraction:
        ticks, seconds, tempos = self._segments
        index = max(0, bisect_right(ticks, tick) - 1)
        return seconds[index] + (tick - ticks[index]) * Fraction(tempos[index], 1000000 * self.ticks_per_beat)

    def tick(self, seconds: float | Fraction) -> Fraction:
        ticks, starts, tempos = self._segments
        value = rational(seconds)
        index = max(0, bisect_right(starts, value) - 1)
        return ticks[index] + (value - starts[index]) * Fraction(1000000 * self.ticks_per_beat, tempos[index])

    def shift(self, tick: int, seconds: float | Fraction) -> int:
        target = self.tick(max(Fraction(0), self.seconds(tick) + rational(seconds)))
        return tick + round(target - tick)


def is_expression(message: Any) -> bool:
    return message.type in ('pitchwheel', 'aftertouch', 'polytouch') or (
        message.type == 'control_change' and message.control == 74
    )


@dataclass(frozen=True)
class MidiEvents:
    type: int
    ticks_per_beat: int
    track_count: int
    charset: str
    events: tuple[MidiEvent, ...]
    notes: tuple[MidiNote, ...]

    @classmethod
    def load(cls, path: Path | str) -> MidiEvents:
        try:
            midi = mido.MidiFile(str(path))
        except (OSError, EOFError, ValueError, KeyError, IndexError) as exc:
            raise MidiEventError(f'Malformed MIDI {path}: {exc}') from exc
        return cls.from_mido(midi)

    @classmethod
    def from_bytes(cls, data: bytes) -> MidiEvents:
        return cls.from_mido(mido.MidiFile(file=io.BytesIO(data)))

    @classmethod
    def from_mido(cls, midi: Any) -> MidiEvents:
        if midi.type not in (0, 1):
            raise MidiEventError(f'Unsupported MIDI type {midi.type}; use type 0 or 1')
        if midi.ticks_per_beat <= 0:
            raise MidiEventError('SMPTE/invalid tick division is unsupported; use PPQN')
        if not midi.tracks or (midi.type == 0 and len(midi.tracks) != 1):
            raise MidiEventError('Invalid track count for MIDI type')
        events = []
        globals_seen: dict[tuple[int, str], Any] = {}
        for track_index, tr in enumerate(midi.tracks):
            tick, ended = 0, False
            for order, msg in enumerate(tr):
                if ended:
                    raise MidiEventError(f'track {track_index}: event after end_of_track')
                if not isinstance(msg.time, int) or msg.time < 0:
                    raise MidiEventError(f'track {track_index}: invalid delta time {msg.time}')
                tick += msg.time
                if msg.type == 'midi_port' and msg.port != 0:
                    raise MidiEventError('Multiple MIDI ports are unsupported')
                if msg.type in ('set_tempo', 'time_signature'):
                    key = tick, msg.type
                    payload = msg.copy(time=0)
                    if key in globals_seen and globals_seen[key] != payload:
                        raise MidiEventError(f'Conflicting {msg.type} at tick {tick}')
                    if msg.type == 'set_tempo' and msg.tempo <= 0:
                        raise MidiEventError('Invalid tempo')
                    globals_seen[key] = payload
                events.append(MidiEvent(tick, track_index, order, freeze_message(msg.copy(time=0))))
                ended = msg.type == 'end_of_track'
        events.sort(key=lambda e: e.key)
        active: dict[tuple[int, int], int] = {}
        owners: dict[int, int] = {}
        pending: dict[int, tuple[int, MidiEvent]] = {}
        notes: dict[int, MidiNote] = {}
        held: dict[int, list[int]] = {}
        sustain: dict[int, bool] = {}
        for i, event in enumerate(events):
            msg, channel = event.message, event.channel
            if channel is None:
                continue
            if msg.type == 'note_on' and msg.velocity > 0:
                if channel in owners and owners[channel] != event.track:
                    raise MidiEventError(f'Conflicting channel ownership: track {event.track}, channel {channel}')
                owners[channel] = event.track
                key = channel, msg.note
                if key in active:
                    raise MidiEventError(f'ambiguous overlapping note: track {event.track}, channel {channel}, note {msg.note}')
                note_id = len(pending)
                active[key] = note_id
                pending[note_id] = i, event
                events[i] = replace(event, note_ids=(note_id,))
            elif msg.type in ('note_off', 'note_on'):
                key = channel, msg.note
                if key not in active:
                    raise MidiEventError(f'unmatched note-off: track {event.track}, channel {channel}, note {msg.note}')
                note_id = active.pop(key)
                on_index, on_event = pending[note_id]
                if on_event.track != event.track or event.tick <= on_event.tick:
                    raise MidiEventError(f'Ambiguous note pairing: channel {channel}, note {msg.note}')
                notes[note_id] = MidiNote(note_id, event.track, channel, msg.note,
                                         on_event.message.velocity, on_event.tick,
                                         event.tick, event.tick, on_index, i)
                events[i] = replace(event, note_ids=(note_id,))
                if sustain.get(channel, False):
                    held.setdefault(channel, []).append(note_id)
            elif msg.type == 'control_change' and msg.control == 64:
                sustain[channel] = msg.value >= 64
                if not sustain[channel]:
                    for note_id in held.pop(channel, []):
                        notes[note_id] = replace(notes[note_id], sounding_end=event.tick)
        if active:
            raise MidiEventError(f'unmatched note-on: {sorted(active)}')
        if any(held.values()):
            raise MidiEventError('Unreleased sustain: sounding notes have no pedal release')
        ordered_notes = tuple(notes[i] for i in sorted(notes))
        events = _associate_expression(events, ordered_notes)
        return cls(midi.type, midi.ticks_per_beat, len(midi.tracks), midi.charset,
                   tuple(events), ordered_notes)

    @cached_property
    def tempo_map(self) -> TempoMap:
        return TempoMap(self.ticks_per_beat, tuple(
            (e.tick, e.message.tempo) for e in self.events if e.message.type == 'set_tempo'))

    def to_mido(self) -> Any:
        midi = mido.MidiFile(type=self.type, ticks_per_beat=self.ticks_per_beat, charset=self.charset)
        tracks: list[list[MidiEvent]] = [[] for _ in range(self.track_count)]
        for event in sorted(self.events, key=lambda e: e.key):
            tracks[event.track].append(event)
        for events in tracks:
            track = mido.MidiTrack()
            previous = 0
            for event in events:
                track.append(event.message.copy(time=event.tick - previous))
                previous = event.tick
            midi.tracks.append(track)
        return midi

    def save(self, path: Path | str) -> None:
        self.to_mido().save(str(path))

    def isolate_tracks(self, declared_tracks: tuple[int, ...] = ()) -> dict[int, MidiEvents]:
        """Project complete channel streams; keep global events and SMF ordering.

        Note-bearing tracks own their channels. A conductor/setup-only track may
        supply their state, but another logical instrument may not. Original SMF
        track slots (including now-empty ones) retain equal-tick merge ordering.
        """
        tracks = {n.track for n in self.notes} | set(declared_tracks)
        if any(not 0 <= t < self.track_count for t in tracks):
            raise MidiEventError('Invalid declared source track index')
        # Names, unused-channel controllers and tempo metadata do not establish
        # instrument identity. Only notes or explicit source declarations do.
        local: dict[int, set[int]] = {t: set() for t in tracks}
        for event in self.events:
            if event.track in tracks and event.channel is not None:
                local[event.track].add(event.channel)
        owners: dict[int, int] = {}
        for track in sorted(tracks):
            for channel in sorted(local[track]):
                if channel in owners and owners[channel] != track:
                    raise MidiEventError(f'Conflicting channel ownership: track {track} and '
                                         f'track {owners[channel]}, channel {channel}')
                owners[channel] = track
        result = {}
        for track in sorted(tracks):
            channels = local[track]
            # Unowned setup channels are conductor state too (no notes can be
            # duplicated); retain them rather than silently discarding setup.
            events = tuple(e for e in self.events if e.channel not in owners or e.channel in channels)
            positions = {(e.track, e.order): i for i, e in enumerate(events)}
            notes = tuple(replace(n, on=positions[(self.events[n.on].track, self.events[n.on].order)],
                                  off=positions[(self.events[n.off].track, self.events[n.off].order)])
                          for n in self.notes if n.track == track)
            # Note IDs index this projection's tuple; source identity remains
            # in track/order. Do not leave sparse IDs in a valid MidiEvents.
            identities = {n.id: i for i, n in enumerate(notes)}
            notes = tuple(replace(n, id=identities[n.id]) for n in notes)
            events = tuple(replace(e, note_ids=tuple(identities[i] for i in e.note_ids)) for e in events)
            result[track] = replace(self, events=events, notes=notes)
        return result


def _associate_expression(events: list[MidiEvent], notes: tuple[MidiNote, ...]) -> list[MidiEvent]:
    """Sweep active/held voices and same-tick setups, rather than scan all notes."""
    starts: dict[tuple[int, int], list[int]] = {}
    for note in notes:
        starts.setdefault((note.channel, note.start), []).append(note.id)
    active: dict[int, set[int]] = {}
    held_ends: list[tuple[int, int]] = []
    releases: dict[int, list[int]] = {}
    previous_tick = -1
    for i, event in enumerate(events):
        if event.tick != previous_tick:
            releases.clear()
            previous_tick = event.tick
        while held_ends and held_ends[0][0] <= event.tick:
            _, note_id = heappop(held_ends)
            active[notes[note_id].channel].discard(note_id)
        channel, msg = event.channel, event.message
        if channel is None:
            continue
        sounding = active.setdefault(channel, set())
        if msg.type == 'note_on' and msg.velocity:
            sounding.add(event.note_ids[0])
        elif msg.type in ('note_on', 'note_off'):
            note_id = event.note_ids[0]
            note = notes[note_id]
            if note.sounding_end > note.end:
                heappush(held_ends, (note.sounding_end, note_id))
            else:
                sounding.remove(note_id)
                releases.setdefault(channel, []).append(note_id)
        elif is_expression(msg):
            ids = sounding | {n for n in starts.get((channel, event.tick), []) if notes[n].on > i}
            if msg.type == 'polytouch':
                ids = {n for n in ids if notes[n].pitch == msg.note}
            if not ids:
                ids = {n for n in releases.get(channel, [])
                       if msg.type != 'polytouch' or notes[n].pitch == msg.note}
            events[i] = replace(event, note_ids=tuple(sorted(ids)))
    return events


def _channel_indices(events: tuple[MidiEvent, ...]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for i, event in enumerate(events):
        if event.channel is not None:
            result.setdefault(event.channel, []).append(i)
    return result


@dataclass(frozen=True)
class NoteTiming:
    start: int
    end: int
    velocity: int


# Parameter values are raw 14-bit MIDI values, not rounded semitone estimates.
State = dict[tuple[str, int], int]
_PARAMETER_CCS = (6, 38, 96, 97, 98, 99, 100, 101)
_DEFAULTS: State = {('pitchwheel', 0): 0, ('aftertouch', 0): 0,
                    ('program_change', 0): 0, ('cc', 0): 0, ('cc', 32): 0,
                    ('bank', 0): 0, ('bank', 32): 0,
                    ('cc', 7): 100, ('cc', 10): 64, ('cc', 11): 127,
                    ('cc', 64): 0, ('cc', 74): 0,
                    ('rpn', 0): 256, ('rpn', 1): 8192, ('rpn', 2): 8192,
                    ('cc', 100): 127, ('cc', 101): 127,
                    ('cc', 98): 127, ('cc', 99): 127, ('selection', 0): 0}


def _update(state: State, msg: Any) -> None:
    if msg.type in ('pitchwheel', 'aftertouch', 'program_change'):
        state[(msg.type, 0)] = getattr(msg, {'pitchwheel': 'pitch', 'aftertouch': 'value',
                                            'program_change': 'program'}[msg.type])
        if msg.type == 'program_change':
            for number in (0, 32):
                if ('cc', number) in state:
                    state[('bank', number)] = state[('cc', number)]
                else:
                    state.pop(('bank', number), None)
    elif msg.type == 'polytouch':
        state[('poly', msg.note)] = msg.value
    elif msg.type == 'control_change':
        cc, value = msg.control, msg.value
        if cc in (98, 99, 100, 101):
            state[('selection', 0)] = int(cc in (98, 99))
        if cc in (96, 97):
            raise ChannelStateConflict('Ambiguous relative parameter increment/decrement in retiming')
        if cc in (6, 38):
            nrpn = state.get(('selection', 0), 0)
            msb, lsb = (99, 98) if nrpn else (101, 100)
            if ('opaque', 0) in state and any(k not in state for k in
                                             (('selection', 0), ('cc', msb), ('cc', lsb))):
                raise ChannelStateConflict('Data entry targets unknown parameter selection after opaque SysEx')
            parameter = state.get(('cc', msb), 127) * 128 + state.get(('cc', lsb), 127)
            if parameter != 16383:
                key = ('nrpn' if nrpn else 'rpn', parameter)
                old = state.get(key)
                if old is None and ('opaque', 0) in state:
                    # Neither half of a parameter is known after an opaque reset.
                    state[(key[0] + ('_msb' if cc == 6 else '_lsb'), parameter)] = value
                    high = state.get((key[0] + '_msb', parameter))
                    low = state.get((key[0] + '_lsb', parameter))
                    if high is not None and low is not None:
                        state[key] = high * 128 + low
                else:
                    old = old or 0
                    state[key] = value * 128 + old % 128 if cc == 6 else old // 128 * 128 + value
                if key == ('rpn', 6) and state.get(key, value) != 0:
                    raise ChannelStateConflict('MPE master-zone interpretation is unsupported in retiming; use per-channel expression')
        else:
            state[('cc', cc)] = value


def _event_state(channels: dict[int, State], event: MidiEvent) -> State:
    if event.message.type == 'sysex':
        # Payloads (including MTS and reset messages) are opaque. No channel's
        # defaults or earlier state remain authoritative after this boundary.
        channels.update({channel: {('opaque', 0): 1} for channel in range(16)})
    return channels.setdefault(event.channel, dict(_DEFAULTS)) if event.channel is not None else {}


def _snapshots(events: tuple[MidiEvent, ...]) -> list[State]:
    channels: dict[int, State] = {}
    result = []
    for e in events:
        state = _event_state(channels, e)
        _update(state, e.message)
        result.append(dict(state))
    return result


def _protected(state: State, pitch: int) -> State:
    """Preserve pitch/expression, including partially known post-SysEx state."""
    result = {k: v for k, v in state.items()
              if k[0] in ('pitchwheel', 'aftertouch', 'program_change', 'bank', 'rpn', 'nrpn',
                          'rpn_msb', 'rpn_lsb', 'nrpn_msb', 'nrpn_lsb')
              or k in (('cc', 74), ('poly', pitch))
              or (('opaque', 0) in state and
                  (k == ('selection', 0) or k in (('cc', 98), ('cc', 99), ('cc', 100), ('cc', 101))))}
    result.setdefault(('poly', pitch), 0)
    return result


def _ordinary(state: State) -> State:
    return {k: v for k, v in state.items() if k[0] == 'cc' and k[1] not in (*_PARAMETER_CCS, 0, 32, 74)}


def _replay(current: State, wanted: State, channel: int) -> list[Any]:
    if ('opaque', 0) in current or ('opaque', 0) in wanted:
        if current != wanted:
            raise ChannelStateConflict('Cannot replay unknown channel state after opaque SysEx')
        return []
    result = []

    def cc(number: int, value: int) -> None:
        result.append(mido.Message('control_change', channel=channel, control=number, value=value))

    instrument_keys = (('bank', 0), ('bank', 32), ('program_change', 0))
    if any(current.get(k, 0) != wanted.get(k, 0) for k in instrument_keys):
        for number in (0, 32):
            cc(number, wanted.get(('bank', number), 0))
        result.append(mido.Message('program_change', channel=channel,
                                   program=wanted.get(('program_change', 0), 0)))
        # Bank selects latch on program change. Restore pending source selects,
        # not the active bank, for subsequent program automation.
        for number in (0, 32):
            cc(number, current.get(('cc', number), 0))
    # Bank selects precede programs; parameter selection/data entry are replayed
    # as transactions, never as an unordered bag of controller values.
    for key in sorted(current.keys() | wanted.keys()):
        kind, number = key
        value = wanted.get(key, _DEFAULTS.get(key, 0))
        if current.get(key, _DEFAULTS.get(key, 0)) == value:
            continue
        if kind == 'cc' and number not in (*_PARAMETER_CCS, 0, 32):
            cc(number, value)
        elif kind in ('rpn', 'nrpn'):
            if kind != 'rpn' or number not in (0, 1, 2):
                raise ChannelStateConflict(f'Cannot replay unknown/device-specific parameter {kind} {number}')
            msb, lsb = (101, 100) if kind == 'rpn' else (99, 98)
            cc(msb, number // 128); cc(lsb, number % 128)
            cc(6, value // 128); cc(38, value % 128)
        elif kind in ('pitchwheel', 'aftertouch'):
            field = {'pitchwheel': 'pitch', 'aftertouch': 'value', 'program_change': 'program'}[kind]
            result.append(mido.Message(kind, channel=channel, **{field: value}))
        elif kind == 'poly':
            result.append(mido.Message('polytouch', channel=channel, note=number, value=value))
    if result:
        # Preserve both selectors and which family was selected last.
        selectors = (101, 100, 99, 98) if current.get(('selection', 0)) else (99, 98, 101, 100)
        for number in selectors:
            cc(number, current.get(('cc', number), 127))
    return result


def _expression_tick(source: MidiEvents, note: MidiNote, timing: NoteTiming, tick: int) -> int:
    tempo = source.tempo_map
    old_start, old_end = tempo.seconds(note.start), tempo.seconds(note.end)
    new_start, new_end = tempo.seconds(timing.start), tempo.seconds(timing.end)
    when = tempo.seconds(tick)
    # Articulation scales key-down expression. Sustain-tail expression follows
    # the moved key release; the pedal release itself remains absolute.
    seconds = (new_start + (when - old_start) * (new_end - new_start) / (old_end - old_start)
               if tick <= note.end else new_end + when - old_end)
    return tick + round(tempo.tick(seconds) - tick)


def _conflict(note: MidiNote, detail: str) -> ChannelStateConflict:
    return ChannelStateConflict(f'track {note.track}, channel {note.channel}, note {note.pitch} '
                                f'(id {note.id}): {detail}; retry with groove disabled')


def _order_boundaries(source: MidiEvents, moved: list[MidiEvent]) -> list[MidiEvent]:
    """Stable order except release bundles that must precede a reused channel."""
    changed = {(e.track, e.tick) for e, old in zip(moved, source.events)
               if e.tick != old.tick and e.note_ids}
    groups: dict[tuple[int, int], list[MidiEvent]] = {}
    for event in moved:
        if (event.track, event.tick) in changed:
            groups.setdefault((event.track, event.tick), []).append(event)
    positions: dict[tuple[int, int], Fraction] = {}
    for (track, tick), members in groups.items():
        group = sorted(members, key=lambda e: e.key)
        starts = [e for e in group if e.message.type == 'note_on' and e.message.velocity]
        ends = [e for e in group if e.message.type == 'note_off' or
                (e.message.type == 'note_on' and not e.message.velocity)]
        for end in ends:
            note_id = end.note_ids[0]
            release = [e for e in group if note_id in e.note_ids]
            for start in starts:
                if start.channel != end.channel:
                    continue
                setup = [e for e in group if start.note_ids[0] in e.note_ids]
                before = min(group.index(e) for e in setup)
                if max(group.index(e) for e in release) < before:
                    continue
                if any(e in release for e in setup):
                    raise _conflict(source.notes[note_id], 'ambiguous shared expression at note boundary')
                group = [e for e in group if e not in release]
                before = min(group.index(e) for e in setup)
                group[before:before] = release
        positions.update({(track, e.order): Fraction(i * 2) for i, e in enumerate(group)})
    return [replace(e, sort_order=positions[(e.track, e.order)]) if (e.track, e.order) in positions else e
            for e in moved]


def retime(source: MidiEvents, timings: dict[int, NoteTiming]) -> MidiEvents:
    """Retime paired notes/associated expression, replay setup, audit every voice.

    No channel reallocation. Unsafe overlaps (including pedal-held notes),
    ambiguous shared expression and opaque state-boundary crossings fail closed.
    Original events retain their source track/order IDs; replay IDs are negative.
    """
    targets = {n.id: timings.get(n.id, NoteTiming(n.start, n.end, n.velocity)) for n in source.notes}
    if timings.keys() - targets.keys():
        raise MidiEventError('Unknown note identity in retiming request')
    for n in source.notes:
        t = targets[n.id]
        if (not all(isinstance(v, int) for v in (t.start, t.end, t.velocity))
                or t.start < 0 or t.end <= t.start or not 1 <= t.velocity <= 127):
            raise _conflict(n, 'invalid note timing/velocity')
    if all(targets[n.id] == NoteTiming(n.start, n.end, n.velocity) for n in source.notes):
        return source
    original_states = _snapshots(source.events)
    moved = []
    for i, e in enumerate(source.events):
        at, msg = e.tick, e.message
        if msg.type in ('note_on', 'note_off'):
            n = source.notes[e.note_ids[0]]
            t = targets[n.id]
            at = t.start if i == n.on else t.end
            if i == n.on:
                msg = freeze_message(msg.copy(velocity=t.velocity))
        elif is_expression(msg) and e.note_ids:
            ticks = {_expression_tick(source, source.notes[n], targets[n], e.tick) for n in e.note_ids}
            if len(ticks) != 1:
                raise _conflict(source.notes[e.note_ids[0]], 'ambiguous shared expression retiming')
            at = ticks.pop()
        elif msg.type == 'control_change' and msg.control in (66, 69, 120, 121, 123, 124, 125, 126, 127):
            raise ChannelStateConflict(f'Unsupported pedal/channel-mode CC {msg.control} in retiming')
        moved.append(replace(e, tick=at, message=msg))
    # EOT is a container boundary, not automation: extend only if necessary.
    track_ends: dict[int, int] = {}
    for e in moved:
        track_ends[e.track] = max(track_ends.get(e.track, 0), e.tick)
    for i, e in enumerate(moved):
        if e.message.type == 'end_of_track':
            moved[i] = replace(e, tick=track_ends[e.track])
    moved = _order_boundaries(source, moved)
    moved_by_id = {(e.track, e.order): e for e in moved}
    sysex_keys = [e.key for e in source.events if e.message.type == 'sysex']
    new_sysex_keys = sorted(e.key for e in moved if e.message.type == 'sysex')
    for old, new in zip(source.events, moved):
        if old.channel is not None and bisect_left(sysex_keys, old.key) != bisect_left(new_sysex_keys, new.key):
            raise ChannelStateConflict(f'track {old.track}, channel {old.channel}: opaque SysEx boundary crossing')
    # Independent ordinary-controller baseline: source automation in destination
    # order, before any replay can contaminate another voice's expected state.
    timeline = tuple(sorted(moved, key=lambda e: e.key))
    timeline_states = _snapshots(timeline)
    timeline_positions = {(e.track, e.order): i for i, e in enumerate(timeline)}
    control_indices: dict[int, list[int]] = {n.channel: [] for n in source.notes}
    for i, e in enumerate(timeline):
        if e.message.type == 'sysex':
            for indices in control_indices.values():
                indices.append(i)
        elif (e.message.type == 'control_change' and e.channel in control_indices
              and e.message.control not in (*_PARAMETER_CCS, 0, 32, 74)):
            control_indices[e.message.channel].append(i)
    channels: dict[int, State] = {}
    replayed: list[MidiEvent] = []
    replay_order = min(e.order for e in source.events) - 1
    sounding: dict[int, list[int]] = {}
    for e in timeline:
        state = _event_state(channels, e)
        if e.message.type == 'note_on' and e.message.velocity:
            n = source.notes[e.note_ids[0]]
            wanted = dict(original_states[n.on])
            for key, value in state.items():
                # Ordinary automation is absolute. Only an earlier onset needs
                # its future initial controls replayed; never rewind later CCs.
                if ((e.tick >= n.start and key[0] == 'cc' and key[1] not in (0, 32, 74))
                        or (key[0] == 'poly' and key[1] != n.pitch)):
                    wanted[key] = value
            setup = _replay(state, wanted, n.channel)
            releases = sounding.setdefault(n.channel, [])
            while releases and releases[0] <= e.tick:
                heappop(releases)
            if releases and any(m.type == 'control_change' and m.control not in (*_PARAMETER_CCS, 0, 32, 74)
                                for m in setup):
                raise _conflict(n, 'ordinary controller/sustain state replay over another sounding voice')
            heappush(releases, n.sounding_end if n.sounding_end > n.end else targets[n.id].end)
            for j, msg in enumerate(setup):
                anchor = e.sort_order if e.sort_order is not None else Fraction(e.order)
                replayed.append(MidiEvent(e.tick, e.track, replay_order,
                                          freeze_message(msg), (n.id,),
                                          anchor - Fraction(len(setup) - j, len(setup) + 1)))
                replay_order -= 1
                _update(state, msg)
        replayed.append(e)
        _update(state, e.message)
    replayed.sort(key=lambda e: e.key)
    candidate = replace(source, events=tuple(replayed))
    try:
        parsed = MidiEvents.from_mido(candidate.to_mido())
    except MidiEventError as exc:
        raise ChannelStateConflict(f'Invalid retimed channel state/pairing: {exc}') from exc
    # Pair by source event identity, not pitch or chronological note index.
    notes = []
    positions = {(e.track, e.order): i for i, e in enumerate(replayed)}
    parsed_by_key = {(n.track, n.channel, n.pitch, n.start): n for n in parsed.notes}
    for n in source.notes:
        t = targets[n.id]
        observed = parsed_by_key[(n.track, n.channel, n.pitch, t.start)]
        expected_release = n.sounding_end if n.sounding_end > n.end else t.end
        if observed.sounding_end != expected_release:
            raise _conflict(n, 'sustain lifetime changed')
        on = positions[(n.track, source.events[n.on].order)]
        off = positions[(n.track, source.events[n.off].order)]
        notes.append(replace(n, start=t.start, end=t.end, velocity=t.velocity,
                             sounding_end=observed.sounding_end, on=on, off=off))
    candidate = replace(candidate, notes=tuple(notes))
    actual_states = _snapshots(candidate.events)
    source_channels = _channel_indices(source.events)
    source_ticks = {channel: [source.events[i].tick for i in indices] for channel, indices in source_channels.items()}
    boundaries: dict[int, dict[int, int]] = {}
    for i, e in enumerate(candidate.events):
        if e.channel is not None:
            boundaries.setdefault(e.channel, {})[e.tick] = i
    boundary_ticks = {channel: list(ticks) for channel, ticks in boundaries.items()}
    for n, new_note in zip(source.notes, candidate.notes):
        timeline_on = timeline_positions[(n.track, source.events[n.on].order)]
        controls = _ordinary(original_states[n.on] if new_note.start < n.start else timeline_states[timeline_on])
        if _ordinary(actual_states[new_note.on]) != controls:
            raise _conflict(n, 'initial ordinary controller state changed by replay')
        automation = control_indices[n.channel]
        control_index = bisect_right(automation, timeline_on)
        expected = _protected(original_states[n.on], n.pitch)
        if _protected(actual_states[new_note.on], n.pitch) != expected:
            raise _conflict(n, 'initial pitch/expression state changed')
        changes = []
        previous = expected
        indices = source_channels[n.channel]
        first = bisect_right(indices, n.on)
        stop = bisect_left(source_ticks[n.channel], n.sounding_end)
        for i in indices[first:stop]:
            e = source.events[i]
            current = _protected(original_states[i], n.pitch)
            delta = {k: v for k, v in current.items() if previous.get(k) != v}
            if delta:
                at = moved_by_id[(e.track, e.order)].tick
                if not new_note.start <= at < new_note.sounding_end:
                    raise _conflict(n, 'pitch/expression trajectory crosses note lifetime')
                changes.append((at, i, delta))
            previous = current
        changes.sort(key=lambda change: change[:2])
        change_index = 0
        # Compare at every channel-event boundary, including other voices' bends,
        # replay transactions and automation; transient setup at one tick is atomic.
        channel_ticks = boundary_ticks[n.channel]
        audit_ticks = channel_ticks[bisect_left(channel_ticks, new_note.start):bisect_left(channel_ticks, new_note.sounding_end)]
        for tick in audit_ticks:
            while change_index < len(changes) and changes[change_index][0] <= tick:
                expected.update(changes[change_index][2]); change_index += 1
            last = boundaries[n.channel][tick]
            # Only this voice's earlier-onset replay is permitted as its initial
            # override. Subsequent source controls remain absolute, including
            # redundant writes; all other replay must match this expectation.
            while control_index < len(automation) and timeline[automation[control_index]].key <= candidate.events[last].key:
                event = timeline[automation[control_index]]
                if event.message.type == 'sysex':
                    controls.clear()
                else:
                    controls[('cc', event.message.control)] = event.message.value
                control_index += 1
            if _ordinary(actual_states[last]) != controls:
                raise _conflict(n, f'incompatible ordinary controller state at tick {tick}')
            actual = _protected(actual_states[last], n.pitch)
            if any(actual.get(k) != expected.get(k) for k in actual.keys() | expected.keys()):
                raise _conflict(n, f'incompatible simultaneous pitch/expression state at tick {tick}')
    return candidate


def add_ghosts(source: MidiEvents, ghosts: list[tuple[int, NoteTiming]]) -> MidiEvents:
    """Insert optional ghosts only in unambiguous channel gaps, restoring state.

    A duplicate key during an existing voice is not a portable MIDI note pair.
    Neither a new channel nor a shortened original voice is silently invented.
    """
    if not ghosts:
        return source
    snapshots = _snapshots(source.events)
    channels = _channel_indices(source.events)
    channel_keys = {channel: [source.events[i].key for i in indices] for channel, indices in channels.items()}
    sysex = [i for i, e in enumerate(source.events) if e.message.type == 'sysex']
    sysex_keys = [source.events[i].key for i in sysex]
    occupied: dict[int, list[tuple[int, int]]] = {}
    for n in source.notes:
        occupied.setdefault(n.channel, []).append((n.start, n.sounding_end))
    # Merge ghost reservations with original voices in one interval sweep.
    reservations: dict[int, list[tuple[int, int, int]]] = {}
    for note_id, timing in ghosts:
        note = source.notes[note_id]
        if timing.start < 0 or timing.end <= timing.start or not 1 <= timing.velocity <= 127:
            raise _conflict(note, 'invalid ghost timing/velocity')
        reservations.setdefault(note.channel, []).append((timing.start, timing.end, note_id))
    for channel, requested in reservations.items():
        intervals: list[tuple[int, int, int | None]] = [(start, end, None) for start, end in occupied.get(channel, [])]
        intervals.extend(requested)
        end, ghost_end = -1, -1
        for start, stop, owner in sorted(intervals, key=lambda item: item[:2]):
            if (owner is not None and start < end) or start < ghost_end:
                note = source.notes[owner if owner is not None else requested[0][2]]
                raise _conflict(note, 'ghost note overlap with an existing sounding note')
            end = max(end, stop)
            if owner is not None:
                ghost_end = stop
    events = list(source.events)
    order = max(e.order for e in events) + 1
    track_ends: dict[int, int] = {}
    added: list[tuple[MidiNote, NoteTiming, int, int]] = []
    for number, (note_id, timing) in enumerate(ghosts):
        note = source.notes[note_id]
        # Start follows source events in the note's track; release/restore precede
        # source setup at the ending tick. Use exact keys, including other tracks.
        start_key = timing.start, note.track, Fraction(order)
        end_key = timing.end, note.track, Fraction(-1)
        region = bisect_left(sysex_keys, start_key)
        if bisect_left(sysex_keys, source.events[note.on].key) != region:
            raise _conflict(note, 'ghost crosses opaque SysEx boundary')
        keys = channel_keys.get(note.channel, [])
        before = bisect_left(keys, start_key)
        if before < bisect_left(keys, end_key) or region < bisect_left(sysex_keys, end_key):
            raise _conflict(note, 'ghost crosses channel state automation')
        previous = channels[note.channel][before - 1] if before else -1
        reset = sysex[region - 1] if region else -1
        current = (snapshots[previous] if previous > reset else
                   {('opaque', 0): 1} if reset >= 0 else dict(_DEFAULTS))
        wanted = snapshots[note.on]
        setup = _replay(current, wanted, note.channel)
        ghost_state = dict(current)
        for msg in setup:
            _update(ghost_state, msg)
        restore = _replay(ghost_state, current, note.channel)
        bundle = [(timing.start, m) for m in setup]
        bundle.append((timing.start, mido.Message('note_on', channel=note.channel,
                                                 note=note.pitch, velocity=timing.velocity)))
        bundle.append((timing.end, mido.Message('note_off', channel=note.channel, note=note.pitch)))
        bundle.extend((timing.end, m) for m in restore)
        ghost_id = len(source.notes) + number
        for i, (tick, msg) in enumerate(bundle):
            priority = Fraction(i - len(bundle) - 1) if tick == timing.end else None
            events.append(MidiEvent(tick, note.track, order + i, freeze_message(msg), (ghost_id,), priority))
        added.append((note, timing, order + len(setup), order + len(setup) + 1))
        order += len(bundle)
        track_ends[note.track] = max(track_ends.get(note.track, 0), timing.end)
    for i, e in enumerate(events):
        if e.track in track_ends and e.message.type == 'end_of_track':
            events[i] = replace(e, tick=max(e.tick, track_ends[e.track]), sort_order=Fraction(order))
    candidate = replace(source, events=tuple(sorted(events, key=lambda e: e.key)))
    try:
        parsed = MidiEvents.from_mido(candidate.to_mido())
    except MidiEventError as exc:
        raise ChannelStateConflict(f'ghost pairing/state invalid: {exc}') from exc
    positions = {(e.track, e.order): i for i, e in enumerate(candidate.events)}
    notes = []
    for old in source.notes:
        on, off = source.events[old.on], source.events[old.off]
        notes.append(replace(old, on=positions[(on.track, on.order)], off=positions[(off.track, off.order)]))
    parsed_by_key = {(n.track, n.channel, n.pitch, n.start): n for n in parsed.notes}
    for note, timing, on_order, off_order in added:
        ghost = parsed_by_key[(note.track, note.channel, note.pitch, timing.start)]
        if ghost.sounding_end != timing.end:
            raise _conflict(note, 'ghost sustain lifetime exceeds its safe channel gap')
        notes.append(replace(ghost, id=len(notes), on=positions[(note.track, on_order)],
                             off=positions[(note.track, off_order)]))
    return replace(candidate, notes=tuple(notes))
