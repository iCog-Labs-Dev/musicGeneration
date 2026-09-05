from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple

_logger = logging.getLogger(__name__)

from aimusic.core.config import DecodeConfig
from aimusic.core.core_types import BeatState, NoteEvent, Score
from aimusic.core.rng import RNGKey, allocate_named_keys
from aimusic.core.vocab import (
    ChordToken,
    GrooveToken,
    Vocabularies,
    validate_vocabulary_compatibility,
)
from aimusic.theory.tonal import chord_pitch_classes, get_fifth_steps, pc


DEFAULT_TICKS_PER_BEAT = 480
DRUM_PITCHES = {
    "kick": 36,
    "snare": 38,
    "hat_closed": 42,
    "hat_open": 46,
}


def _require_non_empty_path(path: Sequence[BeatState]) -> Tuple[BeatState, ...]:
    items = tuple(path)
    if not items:
        raise ValueError("path must not be empty.")
    if any(not isinstance(state, BeatState) for state in items):
        raise TypeError("path must contain only BeatState instances.")
    return items


def _decode_states(
    path: Sequence[BeatState],
    *,
    include_terminal_state: bool,
) -> Tuple[BeatState, ...]:
    states = _require_non_empty_path(path)
    if include_terminal_state or len(states) == 1:
        return states
    return states[:-1]


def _groove_token(state: BeatState, vocabularies: Vocabularies) -> GrooveToken:
    return vocabularies.grooves.token_for_id(state.groove_id)


def _chord_token(state: BeatState, vocabularies: Vocabularies) -> ChordToken:
    return vocabularies.chords.token_for_id(state.chord_id)


def _role_label(state: BeatState, vocabularies: Vocabularies) -> str:
    return vocabularies.roles.token_for_id(state.role_id).label


def _head_label(state: BeatState, vocabularies: Vocabularies) -> str:
    return vocabularies.heads.token_for_id(state.head_id).label


def _meter_beats(state: BeatState, vocabularies: Vocabularies) -> int:
    return vocabularies.meters.token_for_id(state.meter_id).beats_per_bar


def _strong_beats(state: BeatState, vocabularies: Vocabularies) -> Tuple[int, ...]:
    return vocabularies.meters.token_for_id(state.meter_id).strong_beats


def _tension_level(state: BeatState, vocabularies: Vocabularies) -> float:
    role = _role_label(state, vocabularies)
    tension = 0.2 + (0.18 * state.boundary_lvl)
    if role == "prep":
        tension += 0.15
    elif role == "change":
        tension += 0.2
    elif role == "cad":
        tension += 0.28
    return min(1.0, tension)


def _unit_to_velocity(tension: float, decode_config: DecodeConfig) -> float:
    low, high = decode_config.tension_velocity_range
    return low + ((high - low) * tension)


def _unit_to_expression(tension: float, decode_config: DecodeConfig) -> float:
    low, high = decode_config.tension_expression_range
    return low + ((high - low) * tension)


def _should_emit(
    track_density: float, beat_index: int, tension: float, *, strong: bool
) -> bool:
    if track_density <= 0.0:
        return False
    activation = track_density + (0.15 * tension) + (0.2 if strong else 0.0)
    cycle = ((beat_index * 37) % 100) / 100.0
    return cycle < min(1.0, activation)


def build_subbeat_grid(
    path: Sequence[BeatState],
    *,
    decode_config: Optional[DecodeConfig] = None,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    include_terminal_state: bool = False,
) -> Tuple[Tuple[int, ...], ...]:
    states = _decode_states(path, include_terminal_state=include_terminal_state)
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    if ticks_per_beat % resolved_decode.subbeats_per_beat != 0:
        raise ValueError("ticks_per_beat must be divisible by subbeats_per_beat.")
    step = ticks_per_beat // resolved_decode.subbeats_per_beat
    grids = []
    for beat_index, _ in enumerate(states):
        start = beat_index * ticks_per_beat
        grids.append(
            tuple(
                start + (step * offset)
                for offset in range(resolved_decode.subbeats_per_beat)
            )
        )
    return tuple(grids)


def _family_offsets(groove: GrooveToken, subbeats_per_beat: int) -> Tuple[int, ...]:
    if groove.family == "straight":
        return (0, 2) if subbeats_per_beat >= 4 else (0,)
    if groove.family == "syncopated":
        return (0, 3) if subbeats_per_beat >= 4 else (0,)
    if groove.family == "swing":
        return (0, 2) if subbeats_per_beat >= 3 else (0,)
    return (0,)


def _fit_pitch_to_register(pitch_pc: int, register: tuple[int, int], edo: int) -> int:
    if not isinstance(edo, int) or isinstance(edo, bool) or edo < 1:
        raise ValueError("edo must be a positive integer.")
    low, high = register
    normalized_pc = pitch_pc % edo
    candidates = tuple(
        pitch for pitch in range(low, high + 1) if pitch % edo == normalized_pc
    )
    if not candidates:
        raise ValueError(
            f"register {register} contains no pitch with class {normalized_pc} "
            f"in {edo}-EDO."
        )
    return candidates[0]


def _nearest_pitch(
    prev_pitch: Optional[int],
    pitch_classes: Iterable[int],
    register: tuple[int, int],
    edo: int,
) -> int:
    pcs = tuple(pitch_classes)
    candidates = []
    for pitch_pc in pcs:
        base = _fit_pitch_to_register(pitch_pc, register, edo)
        for octave_shift in (-edo, 0, edo):
            pitch = base + octave_shift
            if register[0] <= pitch <= register[1]:
                candidates.append(pitch)
    if not candidates:
        return _fit_pitch_to_register(pcs[0], register, edo)
    if prev_pitch is None:
        return min(candidates)
    return min(candidates, key=lambda pitch: (abs(pitch - prev_pitch), pitch))


def _head_pitch_class(state: BeatState, vocabularies: Vocabularies, edo: int) -> int:
    chord = _chord_token(state, vocabularies)
    major_seventh = round(11 * edo / 12)
    minor_seventh = round(10 * edo / 12)
    diminished_seventh = round(9 * edo / 12)
    seventh_interval = {
        "maj": major_seventh,
        "maj7": major_seventh,
        "maj9": major_seventh,
        "6": major_seventh,
        "69": major_seventh,
        "dim": diminished_seventh,
        "dim7": diminished_seventh,
    }.get(chord.quality, minor_seventh)
    intervals = {
        "root": 0,
        "third": round(4 * edo / 12)
        if chord.quality != "min" and chord.quality != "dim"
        else round(3 * edo / 12),
        "fifth": get_fifth_steps(edo),
        "seventh": seventh_interval,
        "upper_approach": 1,
        "lower_approach": -1,
        "extension": round(2 * edo / 12),
        "rest": 0,
    }
    head = _head_label(state, vocabularies)
    interval = intervals.get(head, 0)
    return pc(chord.root_pc + interval, edo)


def _clamp_leap(
    prev_pitch: Optional[int],
    next_pitch: int,
    max_leap: int,
    *,
    edo: int,
    register: tuple[int, int],
) -> int:
    if prev_pitch is None:
        return next_pitch
    if abs(next_pitch - prev_pitch) <= max_leap:
        return next_pitch
    pitch_class = next_pitch % edo
    compatible = tuple(
        pitch
        for pitch in range(register[0], register[1] + 1)
        if pitch % edo == pitch_class and abs(pitch - prev_pitch) <= max_leap
    )
    if compatible:
        return min(compatible, key=lambda pitch: (abs(pitch - prev_pitch), pitch))
    # Never invent a different pitch class merely to satisfy the soft leap cap.
    return next_pitch


def _append_event(
    events: list[NoteEvent],
    *,
    ton: int,
    duration: int,
    pitch: int,
    velocity: float,
    expression: float,
    track: str,
) -> None:
    events.append(
        NoteEvent(
            ton=ton,
            toff=ton + duration,
            h=pitch,
            v=velocity,
            e=(expression,),
            track=track,
        )
    )


def _cleanup_events(events: Sequence[NoteEvent]) -> Tuple[NoteEvent, ...]:
    by_track = sorted(
        events, key=lambda event: (event.track, event.h, event.ton, event.toff)
    )
    cleaned: list[NoteEvent] = []
    last_by_voice: dict[tuple[str, int], NoteEvent] = {}
    for event in by_track:
        key = (event.track, event.h)
        previous = last_by_voice.get(key)
        if previous is not None and event.ton < previous.toff:
            if event.ton <= previous.ton:
                continue
            cleaned[-1] = NoteEvent(
                ton=previous.ton,
                toff=event.ton,
                h=previous.h,
                v=previous.v,
                e=previous.e,
                track=previous.track,
            )
            last_by_voice[key] = cleaned[-1]
        cleaned.append(event)
        last_by_voice[key] = event
    return tuple(
        sorted(cleaned, key=lambda event: (event.ton, event.track, event.h, event.toff))
    )


def generate_bass_events(
    path: Sequence[BeatState],
    *,
    key: RNGKey,
    decode_config: Optional[DecodeConfig] = None,
    vocabularies: Vocabularies,
    edo: int = 12,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    include_terminal_state: bool = False,
) -> tuple[Tuple[NoteEvent, ...], RNGKey]:
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    validate_vocabulary_compatibility(vocabularies, edo)
    states = _decode_states(path, include_terminal_state=include_terminal_state)
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    events: list[NoteEvent] = []
    prev_pitch: Optional[int] = None
    for beat_index, state in enumerate(states):
        tension = _tension_level(state, vocabularies)
        strong = state.beat_in_bar in _strong_beats(state, vocabularies)
        if not _should_emit(
            resolved_decode.bass_density, beat_index, tension, strong=strong
        ):
            continue
        chord = _chord_token(state, vocabularies)
        role = _role_label(state, vocabularies)
        pitch_pc = chord.root_pc if role != "prep" else pc(chord.root_pc - 1, edo)
        pitch = _nearest_pitch(
            prev_pitch,
            (pitch_pc, pc(chord.root_pc + get_fifth_steps(edo), edo)),
            resolved_decode.bass_register,
            edo,
        )
        if state.boundary_lvl > 0:
            pitch = _fit_pitch_to_register(
                chord.root_pc, resolved_decode.bass_register, edo
            )
        ton = beat_index * ticks_per_beat
        duration = ticks_per_beat if role != "change" else ticks_per_beat // 2
        _append_event(
            events,
            ton=ton,
            duration=duration,
            pitch=pitch,
            velocity=_unit_to_velocity(tension, resolved_decode),
            expression=_unit_to_expression(tension, resolved_decode),
            track="bass",
        )
        prev_pitch = pitch
    return _cleanup_events(events), key


def generate_comping_events(
    path: Sequence[BeatState],
    *,
    key: RNGKey,
    decode_config: Optional[DecodeConfig] = None,
    vocabularies: Vocabularies,
    edo: int = 12,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    include_terminal_state: bool = False,
) -> tuple[Tuple[NoteEvent, ...], RNGKey]:
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    validate_vocabulary_compatibility(vocabularies, edo)
    states = _decode_states(path, include_terminal_state=include_terminal_state)
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    events: list[NoteEvent] = []
    previous_voicing: Optional[Tuple[int, ...]] = None
    for beat_index, state in enumerate(states):
        if resolved_decode.comping_density <= 0.0:
            continue
        groove = _groove_token(state, vocabularies)
        offsets = _family_offsets(groove, resolved_decode.subbeats_per_beat)
        tension = _tension_level(state, vocabularies)
        chord = _chord_token(state, vocabularies)
        pitch_classes = tuple(
            sorted(chord_pitch_classes(chord.root_pc, chord.quality, edo))
        )
        voice_count = max(
            resolved_decode.min_comping_voices,
            min(resolved_decode.max_comping_voices, len(pitch_classes)),
        )
        if state.boundary_lvl > 0:
            voice_count = resolved_decode.max_comping_voices
        voices: list[int] = []
        for voice_idx in range(voice_count):
            target_pc = pitch_classes[voice_idx % len(pitch_classes)]
            prev_pitch = (
                None
                if previous_voicing is None or voice_idx >= len(previous_voicing)
                else previous_voicing[voice_idx]
            )
            voices.append(
                _nearest_pitch(
                    prev_pitch, (target_pc,), resolved_decode.comping_register, edo
                )
            )
        previous_voicing = tuple(sorted(voices))
        for offset in offsets[
            : max(1, round(resolved_decode.comping_density * len(offsets)))
        ]:
            ton = (beat_index * ticks_per_beat) + (
                (ticks_per_beat // resolved_decode.subbeats_per_beat) * offset
            )
            for pitch in previous_voicing:
                _append_event(
                    events,
                    ton=ton,
                    duration=ticks_per_beat // 2,
                    pitch=pitch,
                    velocity=_unit_to_velocity(tension, resolved_decode),
                    expression=_unit_to_expression(tension, resolved_decode),
                    track="comping",
                )
    return _cleanup_events(events), key


def generate_lead_events(
    path: Sequence[BeatState],
    *,
    key: RNGKey,
    decode_config: Optional[DecodeConfig] = None,
    vocabularies: Vocabularies,
    edo: int = 12,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    include_terminal_state: bool = False,
) -> tuple[Tuple[NoteEvent, ...], RNGKey]:
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    validate_vocabulary_compatibility(vocabularies, edo)
    states = _decode_states(path, include_terminal_state=include_terminal_state)
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    events: list[NoteEvent] = []
    prev_pitch: Optional[int] = None
    for beat_index, state in enumerate(states):
        tension = _tension_level(state, vocabularies)
        strong = state.beat_in_bar in _strong_beats(state, vocabularies)
        if not _should_emit(
            resolved_decode.lead_density, beat_index, tension, strong=strong
        ):
            continue
        head = _head_label(state, vocabularies)
        if head == "rest" and state.boundary_lvl == 0:
            continue
        head_pc = _head_pitch_class(state, vocabularies, edo)
        pitch = _nearest_pitch(
            prev_pitch, (head_pc,), resolved_decode.lead_register, edo
        )
        pitch = _clamp_leap(
            prev_pitch,
            pitch,
            resolved_decode.max_lead_leap_steps,
            edo=edo,
            register=resolved_decode.lead_register,
        )
        ton = beat_index * ticks_per_beat
        duration = ticks_per_beat if state.boundary_lvl > 0 else ticks_per_beat // 2
        _append_event(
            events,
            ton=ton,
            duration=duration,
            pitch=pitch,
            velocity=_unit_to_velocity(tension, resolved_decode),
            expression=_unit_to_expression(tension, resolved_decode),
            track="lead",
        )
        prev_pitch = pitch
    return _cleanup_events(events), key


def generate_drum_events(
    path: Sequence[BeatState],
    *,
    key: RNGKey,
    decode_config: Optional[DecodeConfig] = None,
    vocabularies: Vocabularies,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    include_terminal_state: bool = False,
) -> tuple[Tuple[NoteEvent, ...], RNGKey]:
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    states = _decode_states(path, include_terminal_state=include_terminal_state)
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    events: list[NoteEvent] = []
    step_ticks = ticks_per_beat // resolved_decode.subbeats_per_beat
    for beat_index, state in enumerate(states):
        tension = _tension_level(state, vocabularies)
        strong = state.beat_in_bar in _strong_beats(state, vocabularies)
        if not _should_emit(
            resolved_decode.drum_density, beat_index, tension, strong=strong
        ):
            continue
        groove = _groove_token(state, vocabularies)
        offsets = _family_offsets(groove, resolved_decode.subbeats_per_beat)
        offset_count = max(1, round(resolved_decode.drum_density * len(offsets)))
        for offset in offsets[:offset_count]:
            ton = (beat_index * ticks_per_beat) + (offset * step_ticks)
            pitch = (
                DRUM_PITCHES["kick"]
                if strong and offset == 0
                else DRUM_PITCHES["hat_closed"]
            )
            if offset == 0 and not strong:
                pitch = DRUM_PITCHES["snare"]
            if state.boundary_lvl > 1 and offset == 0:
                pitch = DRUM_PITCHES["kick"]
            _append_event(
                events,
                ton=ton,
                duration=max(step_ticks // 2, 1),
                pitch=pitch,
                velocity=_unit_to_velocity(tension, resolved_decode),
                expression=_unit_to_expression(tension, resolved_decode),
                track="drums",
            )
    return _cleanup_events(events), key


def decode_path_to_score(
    path: Sequence[BeatState],
    *,
    key: RNGKey,
    decode_config: Optional[DecodeConfig] = None,
    vocabularies: Vocabularies,
    edo: int = 12,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    tempo_bpm: float = 120.0,
    include_terminal_state: bool = False,
) -> tuple[Score, RNGKey]:
    """Decode a BeatState path into a multi-track symbolic score."""
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    validate_vocabulary_compatibility(vocabularies, edo)
    states = _decode_states(path, include_terminal_state=include_terminal_state)
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    track_keys, next_key = allocate_named_keys(
        key, ("decoder.comping", "decoder.bass", "decoder.lead", "decoder.drums")
    )
    comping, _ = generate_comping_events(
                states,
                key=track_keys["decoder.comping"],
                decode_config=resolved_decode,
                vocabularies=vocabularies,
                edo=edo,
                ticks_per_beat=ticks_per_beat,
                include_terminal_state=True,
            )
    bass, _ = generate_bass_events(
                states,
                key=track_keys["decoder.bass"],
                decode_config=resolved_decode,
                vocabularies=vocabularies,
                edo=edo,
                ticks_per_beat=ticks_per_beat,
                include_terminal_state=True,
            )
    lead, _ = generate_lead_events(
                states,
                key=track_keys["decoder.lead"],
                decode_config=resolved_decode,
                vocabularies=vocabularies,
                edo=edo,
                ticks_per_beat=ticks_per_beat,
                include_terminal_state=True,
            )
    drums, _ = generate_drum_events(
                states,
                key=track_keys["decoder.drums"],
                decode_config=resolved_decode,
                vocabularies=vocabularies,
                ticks_per_beat=ticks_per_beat,
                include_terminal_state=True,
            )
    events = list(comping) + list(bass) + list(lead) + list(drums)
    _logger.info(f"Decoded {len(events)} note events ({len(states)} beats)")
    return Score(
        note_events=_cleanup_events(events),
        ticks_per_beat=ticks_per_beat,
        tempo_bpm=tempo_bpm,
    ), next_key
