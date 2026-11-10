"""Asset-free renderer requirements from symbolic intent and MIDI evidence.

This is admission policy, not a pitch-fidelity proof. ``channel_pitch`` covers
independent channel wheels/RPN tuning (including per-channel MPE exports).
Explicit MPE zones require ``mpe`` instead: a future channel-wheel capability
must not accidentally certify master/member semantics. MTS and opaque SysEx
remain distinct, unverified requirements. No native capability is granted here.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, NoReturn

from aimusic.audio.midi_events import MidiEvents
from aimusic.core.core_types import Score


# Combined evidence can only strengthen admission, never erase earlier state.
_MODES = ('direct_12', 'channel_pitch', 'mpe', 'mts', 'sysex', 'unsupported')
_DEFAULT_RPN = {0: 2 * 128, 1: 64 * 128, 2: 64 * 128, 6: 0}


def _event_modes(midi: MidiEvents, *, drums: bool) -> set[str]:
    modes: set[str] = set()
    selectors = [{100: 127, 101: 127, 98: 127, 99: 127} for _ in range(16)]
    nrpn = [False] * 16
    values = [dict(_DEFAULT_RPN) for _ in range(16)]
    for event in midi.events:
        msg, channel = event.message, event.channel
        if msg.type == 'sysex':
            data = msg.data
            modes.add('mts' if len(data) >= 4 and data[0] in (0x7e, 0x7f) and data[2] == 8 else 'sysex')
        if drums or channel is None or channel == 9:
            continue
        if msg.type == 'pitchwheel' and msg.pitch != 0:
            modes.add('channel_pitch')
        if msg.type != 'control_change':
            continue
        control, value = msg.control, msg.value
        selected = selectors[channel]
        if control == 121:
            # Reset selection, not registered parameter values or prior evidence.
            for number in selected:
                selected[number] = 127
            nrpn[channel] = False
            continue
        if control in (98, 99, 100, 101):
            selected[control] = value
            nrpn[channel] = control in (98, 99)
        if control not in (6, 38, 96, 97):
            continue
        high, low = (99, 98) if nrpn[channel] else (101, 100)
        parameter = selected[high] * 128 + selected[low]
        if parameter == 16383:
            continue  # Null selection: Data Entry and increment/decrement do nothing.
        if nrpn[channel] or parameter not in _DEFAULT_RPN:
            modes.add('unsupported')
            continue
        if control in (96, 97):
            modes.add('mpe' if parameter == 6 else 'channel_pitch')
            continue  # Relative changes require a renderer, not a neutrality guess.
        old = values[channel][parameter]
        current = value * 128 + old % 128 if control == 6 else old // 128 * 128 + value
        values[channel][parameter] = current
        # Coarse tuning and zone member count use only the MSB. Fine tuning and
        # bend sensitivity use both bytes (sensitivity's low byte is cents).
        changed = (current // 128 != _DEFAULT_RPN[parameter] // 128 if parameter in (2, 6)
                   else current != _DEFAULT_RPN[parameter])
        if changed:
            modes.add('mpe' if parameter == 6 else 'channel_pitch')
    return modes


def _number(values: dict[str, Any], key: str, default: float) -> float:
    raw = values.get(key, default)
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid tuning {key}: {raw!r}') from exc
    if isinstance(raw, bool) or not math.isfinite(number):
        raise ValueError(f'invalid tuning {key}: {raw!r}')
    return number


def _metadata_views(metadata: dict[str, Any], track_name: str | None,
                    source_track_index: int | None) -> list[dict[str, Any]]:
    views = [metadata, metadata.get('tuning', {})]
    if track_name is not None:
        for view in tuple(views):
            per_track = view.get('per_track', {}).get(track_name)
            if per_track is not None:
                views.append(per_track if isinstance(per_track, dict) else {'method': per_track})
    for row in metadata.get('tracks', []):
        matches = (row.get('source_track_index') == source_track_index if source_track_index is not None
                   else track_name is not None and row.get('name') == track_name)
        if matches:
            views.append(row)
    return views


def required_tuning(midi: MidiEvents, metadata: dict[str, Any], *, score: Score | None = None,
                    track_name: str | None = None, source_track_index: int | None = None,
                    drums: bool = False) -> str:
    """Derive the strongest requirement for an isolated logical source stem.

    Base tuning is a MIDI-semitone anchor, not Hz: integer anchors are standard
    12-TET. Score heights are EDO steps, never drum-key transpositions. Without
    Score evidence a non-12 EDO is conservatively gated. With Score evidence a
    direct-labelled integer-pitch subset can be direct-12; this does not check
    Score/MIDI pitch agreement (the later transformation-integrity proof).

    Centered wheel resets and explicit default RPN values are neutral. Every
    nonzero wheel or nonstandard range/fine/coarse value requires channel pitch,
    even if later reset or outside sounding notes. Labels can strengthen, but
    cannot lower, evidence from another metadata layer or source events.
    """
    melodic = any(note.channel != 9 for note in midi.notes)
    if drums and melodic:
        raise ValueError('percussion declaration conflicts with melodic MIDI channels')
    drums = drums or (bool(midi.notes) and not melodic)
    modes = _event_modes(midi, drums=drums)
    if not drums:
        views = _metadata_views(metadata, track_name, source_track_index)
        heights = None if score is None else [note.h for note in score
                                             if track_name is None or note.track == track_name]
        if heights == [] and melodic:
            raise ValueError(f'Score is missing pitched notes for track {track_name!r}')
        # Inherit omitted coordinates, but check every explicit layer independently
        # so a contradictory direct label/track override cannot hide intent.
        coordinates = {key: value for view in views for key, value in view.items()
                       if key in ('edo', 'base_tuning')}
        for view in views:
            edo = _number(view, 'edo', coordinates.get('edo', 12))
            base = _number(view, 'base_tuning', coordinates.get('base_tuning', 0))
            bend_range = _number(view, 'pitch_bend_range', 2)
            if edo <= 0 or not edo.is_integer():
                raise ValueError(f'invalid tuning edo: {edo!r}')
            if bend_range <= 0:
                raise ValueError(f'invalid tuning pitch_bend_range: {bend_range!r}')
            if not base.is_integer() or bend_range != 2 or view.get('microtonal', False):
                modes.add('channel_pitch')
            if heights is None:
                if edo != 12:
                    modes.add('channel_pitch')
            else:
                for height in heights:
                    pitch = Fraction(str(base)) + Fraction(height * 12, int(edo))
                    if not 0 <= pitch < 128:
                        raise ValueError(f'Score pitch outside MIDI range: track={track_name!r}, pitch={float(pitch)}')
                    if pitch.denominator != 1:
                        modes.add('channel_pitch')
            method = str(view.get('method', 'direct_12')).lower()
            mode = {'direct_12': 'direct_12', 'mpe': 'channel_pitch', 'channel_pitch': 'channel_pitch',
                    'mts': 'mts'}.get(method, 'unsupported')
            modes.add(mode)
    return max(modes or {'direct_12'}, key=_MODES.index)


def channel_pitch_evidence(midi: MidiEvents) -> dict[str, Any]:
    """Validate the native-proven RPN0 +/-2 subset and reconstruct onset Hz.

    One sounding key per melodic channel; explicit full range setup/null and a
    fresh wheel before every note. No live bends, sustain, zones, NRPN, relative
    data, fine/coarse tuning or opaque events. Pan CC10 is allowed (the native
    simultaneous-channel probe separates mono measurement windows with it).
    This is event evidence only, never an audible-frequency certificate.
    """
    from aimusic.audio.render.registry import UnsupportedTuning
    setup = ((101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127))
    progress = [0] * 16
    ready = [False] * 16
    wheels: list[int | None] = [None] * 16
    active: list[int | None] = [None] * 16
    rows = []
    for event in midi.events:
        msg, ch = event.message, event.channel
        if msg.is_meta:
            continue

        def reject() -> NoReturn:
            raise UnsupportedTuning(f'Unsupported channel_pitch subset at track={event.track} '
                                    f'channel={ch} tick={event.tick}: {msg}; '
                                    'require explicit RPN0 +/-2, idle-channel wheels, one key/channel')

        if ch is None or ch == 9:
            reject()
        if msg.type == 'control_change':
            if msg.control == 10:
                continue
            if active[ch] is not None or (msg.control, msg.value) != setup[progress[ch]]:
                reject()
            ready[ch] = False
            progress[ch] += 1
            if progress[ch] == len(setup):
                progress[ch], ready[ch] = 0, True
        elif msg.type == 'pitchwheel':
            if active[ch] is not None or not ready[ch]:
                reject()
            wheels[ch] = msg.pitch
        elif msg.type == 'program_change':
            if active[ch] is not None or progress[ch]:
                reject()
        elif msg.type == 'note_on' and msg.velocity:
            wheel = wheels[ch]
            if not ready[ch] or wheel is None or active[ch] is not None:
                reject()
            rows.append({'track': event.track, 'channel': ch, 'tick': event.tick, 'midi_note': msg.note,
                         'wheel': wheel, 'hz': 440 * 2 ** ((msg.note - 69 + wheel * 2 / 8192) / 12)})
            active[ch], wheels[ch] = msg.note, None
        elif msg.type in ('note_off', 'note_on'):
            if active[ch] != msg.note:
                reject()
            active[ch] = None
        else:
            reject()
    if any(progress):
        raise UnsupportedTuning('Unsupported channel_pitch subset: incomplete RPN0 transaction')
    return {'subset': 'rpn0-2-semitones-v1', 'notes': rows, 'audio_pitch_accuracy': 'unmeasured'}
