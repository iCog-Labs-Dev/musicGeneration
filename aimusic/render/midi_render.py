from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import mido

from aimusic.core.config import MicrotonalRendering
from aimusic.core.core_types import NoteEvent, Score
from aimusic.theory.edo import EDO


DEFAULT_DRUM_CHANNEL = 9
MELODIC_CHANNELS = tuple(channel for channel in range(1, 16) if channel != DEFAULT_DRUM_CHANNEL)


@dataclass(frozen=True)
class SymbolicNote:
    """A renderer-agnostic representation of a musical note."""

    pitch_height: int
    start_time: float
    end_time: float
    velocity: int = 64
    timbre: int | None = None
    pressure: int | None = None


@dataclass(frozen=True)
class _TrackNote:
    track_name: str
    note: SymbolicNote


@dataclass(frozen=True)
class _TrackSpec:
    track_name: str
    program: int | None
    is_drum: bool = False


@dataclass(frozen=True)
class TrackInstrumentConfig:
    """Optional per-track instrument override for score-based MIDI export."""

    program: int | None = None
    is_drum: bool = False


def _track_spec(
    track_name: str,
    track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
) -> _TrackSpec:
    normalized = track_name.strip().lower()
    if track_instruments is not None and normalized in track_instruments:
        override = track_instruments[normalized]
        return _TrackSpec(
            track_name=track_name,
            program=override.program,
            is_drum=override.is_drum,
        )
    specs = {
        "bass": _TrackSpec(track_name=track_name, program=33),
        "comping": _TrackSpec(track_name=track_name, program=4),
        "lead": _TrackSpec(track_name=track_name, program=81),
        "drums": _TrackSpec(track_name=track_name, program=None, is_drum=True),
    }
    return specs.get(normalized, _TrackSpec(track_name=track_name, program=0))


def _clamp_midi_value(value: int) -> int:
    return max(0, min(127, value))


def _unit_interval_to_midi(value: float) -> int:
    return _clamp_midi_value(round(max(0.0, min(1.0, value)) * 127))


def _score_event_to_symbolic_note(event: NoteEvent, ticks_per_beat: int) -> SymbolicNote:
    timbre = None if not event.e else _unit_interval_to_midi(event.e[0])
    return SymbolicNote(
        pitch_height=event.h,
        start_time=event.ton / ticks_per_beat,
        end_time=event.toff / ticks_per_beat,
        velocity=max(1, _unit_interval_to_midi(event.v)),
        timbre=timbre,
    )


def score_to_track_notes(score: Score) -> Tuple[_TrackNote, ...]:
    """Convert a symbolic score into stable per-track symbolic notes."""
    return tuple(
        _TrackNote(
            track_name=event.track,
            note=_score_event_to_symbolic_note(event, score.ticks_per_beat),
        )
        for event in score.note_events
    )


def _allocate_channels(track_notes: Sequence[_TrackNote]) -> List[Tuple[_TrackNote, int]]:
    """Allocate MIDI channels 1-15 globally across all tracks."""
    sorted_notes = sorted(
        track_notes,
        key=lambda item: (item.note.start_time, item.note.end_time, item.track_name, item.note.pitch_height),
    )

    allocated_notes = []
    channel_free_times: Dict[int, float] = {ch: 0.0 for ch in range(1, 16)}
    channel_last_tracks: Dict[int, str | None] = {ch: None for ch in range(1, 16)}
    channel_used: Dict[int, bool] = {ch: False for ch in range(1, 16)}

    for track_note in sorted_notes:
        free_channels = [
            ch
            for ch, free_time in channel_free_times.items()
            if (
                not channel_used[ch]
                or free_time < track_note.note.start_time
                or (
                    free_time == track_note.note.start_time
                    and channel_last_tracks[ch] == track_note.track_name
                )
            )
        ]

        if not free_channels:
            raise ValueError(
                "MPE polyphony limit exceeded: Attempted to play > 15 overlapping notes "
                f"at time {track_note.note.start_time}. No free channels available."
            )

        chosen_ch = free_channels[0]
        channel_free_times[chosen_ch] = track_note.note.end_time
        channel_last_tracks[chosen_ch] = track_note.track_name
        channel_used[chosen_ch] = True
        allocated_notes.append((track_note, chosen_ch))

    return allocated_notes


def _allocate_channels_in_pool(
    track_notes: Sequence[_TrackNote],
    channel_pool: Sequence[int],
) -> List[Tuple[_TrackNote, int]]:
    if not channel_pool:
        raise ValueError("channel_pool must not be empty.")

    sorted_notes = sorted(
        track_notes,
        key=lambda item: (item.note.start_time, item.note.end_time, item.note.pitch_height),
    )
    allocated_notes = []
    channel_free_times: Dict[int, float] = {channel: 0.0 for channel in channel_pool}

    for track_note in sorted_notes:
        free_channels = [
            channel
            for channel, free_time in channel_free_times.items()
            if free_time <= track_note.note.start_time
        ]
        if not free_channels:
            raise ValueError(
                "Track polyphony exceeded the reserved MIDI channel pool for "
                f"{track_note.track_name!r} at time {track_note.note.start_time}."
            )
        chosen_channel = free_channels[0]
        channel_free_times[chosen_channel] = track_note.note.end_time
        allocated_notes.append((track_note, chosen_channel))

    return allocated_notes


def _mpe_setup_events(channels: Sequence[int], pb_range: int) -> List[Tuple[int, int, str, int, int, int]]:
    events: List[Tuple[int, int, str, int, int, int]] = []
    for channel in channels:
        events.append((0, -4, "control_change", 101, 0, channel))
        events.append((0, -3, "control_change", 100, 0, channel))
        events.append((0, -2, "control_change", 6, pb_range, channel))
        events.append((0, -1, "control_change", 38, 0, channel))
    return events


def _program_change_events(program: int, channels: Iterable[int]) -> List[Tuple[int, int, str, int, int, int]]:
    return [
        (0, -5, "program_change", program, 0, channel)
        for channel in tuple(dict.fromkeys(channels))
    ]


def _note_events_for_track(
    track_notes: Sequence[Tuple[_TrackNote, int]],
    edo: EDO,
    ticks_per_beat: int,
    *,
    is_drum: bool = False,
) -> List[Tuple[int, int, str, int, int, int]]:
    events: List[Tuple[int, int, str, int, int, int]] = []
    for track_note, channel in track_notes:
        midi_note, pitch_bend = edo.to_midi(track_note.note.pitch_height)
        start_tick = int(track_note.note.start_time * ticks_per_beat)
        end_tick = int(track_note.note.end_time * ticks_per_beat)

        events.append((end_tick, 0, "note_off", midi_note, 0, channel))
        if not is_drum:
            events.append((start_tick, 1, "pitchwheel", pitch_bend, 0, channel))

        if track_note.note.timbre is not None and not is_drum:
            events.append((start_tick, 2, "control_change", 74, track_note.note.timbre, channel))

        if track_note.note.pressure is not None and not is_drum:
            events.append((start_tick, 3, "aftertouch", track_note.note.pressure, 0, channel))

        events.append((start_tick, 4, "note_on", midi_note, track_note.note.velocity, channel))

    events.sort()
    return events


def _append_midi_messages(
    track: mido.MidiTrack,
    events: Sequence[Tuple[int, int, str, int, int, int]],
) -> None:
    current_tick = 0
    for abs_tick, _, msg_type, val1, val2, channel in events:
        delta_tick = abs_tick - current_tick

        if msg_type == "pitchwheel":
            track.append(mido.Message("pitchwheel", pitch=val1, time=delta_tick, channel=channel))
        elif msg_type == "control_change":
            track.append(
                mido.Message(
                    "control_change",
                    control=val1,
                    value=val2,
                    time=delta_tick,
                    channel=channel,
                )
            )
        elif msg_type == "aftertouch":
            track.append(mido.Message("aftertouch", value=val1, time=delta_tick, channel=channel))
        elif msg_type == "program_change":
            track.append(
                mido.Message("program_change", program=val1, time=delta_tick, channel=channel)
            )
        else:
            track.append(
                mido.Message(msg_type, note=val1, velocity=val2, time=delta_tick, channel=channel)
            )

        current_tick = abs_tick


def _build_single_track_midi(
    notes: Sequence[SymbolicNote],
    edo: EDO,
    output_path: str,
    ticks_per_beat: int,
) -> None:
    allocated_notes = _allocate_channels(
        tuple(_TrackNote(track_name="default", note=note) for note in notes)
    )
    events = _mpe_setup_events(
        sorted({channel for _, channel in allocated_notes}),
        edo.config.pitch_bend_range,
    )
    events.extend(_note_events_for_track(allocated_notes, edo, ticks_per_beat))
    events.sort()

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track_name = f"{edo.config.n}-EDO Export"
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    _append_midi_messages(track, events)
    track.append(mido.MetaMessage("end_of_track", time=0))

    mid.save(output_path)


def _max_simultaneous_notes(track_notes: Sequence[_TrackNote]) -> int:
    events = []
    for track_note in track_notes:
        events.append((track_note.note.start_time, 1))
        events.append((track_note.note.end_time, -1))
    events.sort(key=lambda item: (item[0], item[1]))

    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def _allocate_score_track_channels(
    grouped_notes: "OrderedDict[str, List[_TrackNote]]",
    edo: EDO,
    *,
    track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
) -> tuple[dict[str, list[tuple[_TrackNote, int]]], tuple[int, ...]]:
    melodic_specs = []
    used_channels: list[int] = []
    allocations: dict[str, list[tuple[_TrackNote, int]]] = {}

    if edo.config.n == 12:
        next_melodic_channel = iter(MELODIC_CHANNELS)
        for track_name, track_notes in grouped_notes.items():
            spec = _track_spec(track_name, track_instruments)
            if spec.is_drum:
                allocations[track_name] = [
                    (track_note, DEFAULT_DRUM_CHANNEL) for track_note in track_notes
                ]
                used_channels.append(DEFAULT_DRUM_CHANNEL)
                continue
            channel = next(next_melodic_channel)
            allocations[track_name] = [(track_note, channel) for track_note in track_notes]
            used_channels.append(channel)
        return allocations, tuple(dict.fromkeys(used_channels))

    for track_name, track_notes in grouped_notes.items():
        spec = _track_spec(track_name, track_instruments)
        if spec.is_drum:
            allocations[track_name] = [
                (track_note, DEFAULT_DRUM_CHANNEL) for track_note in track_notes
            ]
            used_channels.append(DEFAULT_DRUM_CHANNEL)
            continue
        melodic_specs.append((track_name, track_notes, _max_simultaneous_notes(track_notes)))

    total_required = sum(required for _, _, required in melodic_specs)
    if total_required > len(MELODIC_CHANNELS):
        raise ValueError(
            "Score requires more simultaneous melodic channels than are available for "
            f"microtonal MPE rendering ({total_required} > {len(MELODIC_CHANNELS)})."
        )

    cursor = 0
    for track_name, track_notes, required in melodic_specs:
        channel_pool = MELODIC_CHANNELS[cursor : cursor + required]
        allocations[track_name] = _allocate_channels_in_pool(track_notes, channel_pool)
        used_channels.extend(channel_pool)
        cursor += required

    return allocations, tuple(dict.fromkeys(used_channels))


def _build_multitrack_midi(
    score: Score,
    edo: EDO,
    output_path: str,
    *,
    track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
) -> None:
    track_notes = score_to_track_notes(score)
    grouped_notes: "OrderedDict[str, List[_TrackNote]]" = OrderedDict()
    for track_note in track_notes:
        grouped_notes.setdefault(track_note.track_name, []).append(track_note)

    allocated_by_track, used_channels = _allocate_score_track_channels(
        grouped_notes,
        edo,
        track_instruments=track_instruments,
    )
    melodic_channels = tuple(channel for channel in used_channels if channel != DEFAULT_DRUM_CHANNEL)

    mid = mido.MidiFile(ticks_per_beat=score.ticks_per_beat)

    conductor = mido.MidiTrack()
    mid.tracks.append(conductor)
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    conductor.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.tempo_bpm), time=0)
    )
    if melodic_channels:
        _append_midi_messages(
            conductor,
            _mpe_setup_events(melodic_channels, edo.config.pitch_bend_range),
        )
    conductor.append(mido.MetaMessage("end_of_track", time=0))

    for track_name, notes_for_track in allocated_by_track.items():
        spec = _track_spec(track_name, track_instruments)
        midi_track = mido.MidiTrack()
        mid.tracks.append(midi_track)
        midi_track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        events: list[tuple[int, int, str, int, int, int]] = []
        if spec.program is not None:
            events.extend(_program_change_events(spec.program, (channel for _, channel in notes_for_track)))
        events.extend(
            _note_events_for_track(
                notes_for_track,
                edo,
                score.ticks_per_beat,
                is_drum=spec.is_drum,
            )
        )
        events.sort()
        _append_midi_messages(midi_track, events)
        midi_track.append(mido.MetaMessage("end_of_track", time=0))

    mid.save(output_path)


def render_midi(
    notes: Sequence[SymbolicNote] | Score,
    edo: EDO,
    output_path: str,
    ticks_per_beat: int = 480,
    *,
    track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
) -> None:
    """
    Render symbolic notes or a repository-native Score into MIDI.

    Score inputs preserve symbolic track labels as separate MIDI tracks.
    """
    if edo.config.microtonal_rendering_method == MicrotonalRendering.MTS:
        raise NotImplementedError(
            "MTS (MIDI Tuning Standard) rendering is currently deferred. "
            "Due to limited modern VST support, please use MicrotonalRendering.MPE."
        )

    if isinstance(notes, Score):
        _build_multitrack_midi(
            notes,
            edo,
            output_path,
            track_instruments=track_instruments,
        )
        return

    _build_single_track_midi(notes, edo, output_path, ticks_per_beat)


@dataclass(frozen=True)
class MidiSummary:
    """A statistical summary of a rendered MIDI file for quick inspection."""

    total_notes: int
    unique_channels: Tuple[int, ...]
    pitch_bend_events: int
    timbre_events: int
    pressure_events: int

    def print_report(self) -> None:
        """Print a human-readable console report of the MIDI file."""
        print("\n=== MIDI Rendering Summary ===")
        print(f"Total Notes Played:   {self.total_notes}")
        print(f"Unique Channels Used: {len(self.unique_channels)} {self.unique_channels}")
        print(f"Pitch Bend Events:    {self.pitch_bend_events}")
        print(f"Timbre (CC74) Events: {self.timbre_events}")
        print(f"Pressure Events:      {self.pressure_events}")
        print("==============================\n")


def summarize_midi(filepath: str) -> MidiSummary:
    """Read a MIDI file from disk and tally its expressive contents."""
    mid = mido.MidiFile(filepath)

    note_count = 0
    channels = set()
    pb_count = 0
    timbre_count = 0
    pressure_count = 0

    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                note_count += 1
                channels.add(msg.channel)
            elif msg.type == "pitchwheel":
                pb_count += 1
            elif msg.type == "control_change" and msg.control == 74:
                timbre_count += 1
            elif msg.type == "aftertouch":
                pressure_count += 1

    return MidiSummary(
        total_notes=note_count,
        unique_channels=tuple(sorted(channels)),
        pitch_bend_events=pb_count,
        timbre_events=timbre_count,
        pressure_events=pressure_count,
    )
