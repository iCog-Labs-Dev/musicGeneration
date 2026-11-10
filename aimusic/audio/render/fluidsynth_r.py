"""Clean-room offline adapter to pyFluidSynth's public Synth API.

No m2a source used. API/license sources and limitations: docs/audio-fluidsynth.md.
Never starts an audio driver, MIDI player, sequencer, or wall-clock thread.
"""
from __future__ import annotations

import importlib
from ctypes import c_int
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.render.util import render_frames

if TYPE_CHECKING:
    from aimusic.audio.render.registry import RenderRequest

# Fixed, dry, single-thread settings. No platform/device defaults in rendering.
# https://www.fluidsynth.org/api/settings_synth.html
SETTINGS: dict[str, int | float | str] = {
    'synth.gain': 0.2, 'synth.midi-channels': 16,
    'synth.polyphony': 256, 'synth.cpu-cores': 1,
    'synth.reverb.active': 0, 'synth.chorus.active': 0,
    'synth.min-note-length': 0, 'synth.midi-bank-select': 'mma',
    'synth.audio-channels': 1, 'synth.audio-groups': 1,
    'synth.dynamic-sample-loading': 0,
}
BLOCK_FRAMES = 1024
SUPPORTED = {'note_on', 'note_off', 'program_change', 'control_change', 'pitchwheel'}


def load_binding() -> Any:
    """Load only on native selection; classify missing resources, not defects."""
    from aimusic.audio.render.registry import MissingRendererResource
    try:
        installed = version('pyfluidsynth')
    except PackageNotFoundError as exc:
        raise MissingRendererResource('FluidSynth bindings missing: pip install -e ".[audio-render]"; '
                                      'see docs/audio-fluidsynth.md') from exc
    if installed != '1.3.4':
        raise MissingRendererResource(f'FluidSynth requires pyfluidsynth==1.3.4, found {installed}; '
                                      'install the [audio-render] extra')
    try:
        return importlib.import_module('fluidsynth')
    except ImportError as exc:
        if str(exc) != "Couldn't find the FluidSynth library.":
            raise
        raise MissingRendererResource('FluidSynth native library missing; install it separately '
                                      '(docs/audio-fluidsynth.md)') from exc
    except OSError as exc:
        # This boundary is the binding's DLL load, never synthesis execution.
        raise MissingRendererResource(f'FluidSynth native library could not load: {exc}; '
                                      'check architecture/dependent DLLs (docs/audio-fluidsynth.md)') from exc


def versions(binding: Any) -> tuple[tuple[str, str], ...]:
    """Public fluid_version query, no synthesizer allocation or private handles."""
    major, minor, micro = c_int(), c_int(), c_int()
    binding.fluid_version(major, minor, micro)
    if major.value < 2:
        raise RuntimeError('FluidSynth >=2.0 is required by the pinned binding')
    return (('pyfluidsynth', version('pyfluidsynth')),
            ('libfluidsynth', f'{major.value}.{minor.value}.{micro.value}'),
            ('soundfile', sf.__version__), ('libsndfile', sf.__libsndfile_version__))


def _checked(synth: Any, method: str, *args: Any) -> None:
    result = getattr(synth, method)(*args)
    # A correctly paired key can already have decayed (especially percussion).
    # Official MIDI-message docs allow noteoff=-1 when no voices match.
    allowed = (0, -1) if method == 'noteoff' else (0,)
    if result not in allowed or result is False:
        raise RuntimeError(f'FluidSynth {method}{args} failed: status={result}')


def render(request: RenderRequest, *, binding: Any = None) -> Path:
    """Render exact absolute-sample event intervals, then a fixed release tail.

    Synth.pitch_bend takes signed MIDI values (unlike the native C API).
    CC transactions, including RPN selectors/data, stay ordered and untouched.
    program_select requests an exact preset: missing banks/programs fail rather
    than invoking FluidSynth program_change's substitute-preset policy.
    """
    frames = render_frames(request.sample_rate, request.duration_sec, request.release_tail_sec)
    if not 8000 <= request.sample_rate <= 96000:
        raise ValueError('FluidSynth sample rate must be between 8000 and 96000 Hz')
    source = MidiEvents.load(request.midi_path)
    if request.required_tuning == 'channel_pitch':
        from aimusic.audio.render.tuning import channel_pitch_evidence
        channel_pitch_evidence(source)
    for event in source.events:
        if not event.message.is_meta and event.message.type not in SUPPORTED:
            raise ValueError(f'Unsupported FluidSynth event {event.message.type} at tick {event.tick}; '
                             'the public Synth API subset cannot render it')
    if float(source.tempo_map.seconds(max(e.tick for e in source.events))) > request.duration_sec + 1e-9:
        raise ValueError('FluidSynth horizon ends before source MIDI events')
    if request.soundfont is None or not request.soundfont.is_file():
        raise ValueError('FluidSynth requires a local license-approved SoundFont')
    binding = binding if binding is not None else load_binding()
    settings = {**SETTINGS, 'synth.sample-rate': float(request.sample_rate)}
    options = {k: v for k, v in SETTINGS.items() if k not in ('synth.gain', 'synth.midi-channels')}
    audio = np.zeros((frames, 2), dtype=np.float32)
    synth = binding.Synth(gain=SETTINGS['synth.gain'], samplerate=request.sample_rate,
                          channels=16, **options)
    failure: BaseException | None = None
    try:
        # The pinned binding's setting() discards native return codes. Read back
        # the public settings so unsupported/rejected values cannot be claimed.
        for name, value in settings.items():
            if synth.get_setting(name) != value:
                raise RuntimeError(f'FluidSynth rejected setting {name}={value}')
        sfid = synth.sfload(str(request.soundfont), update_midi_preset=0)
        if sfid < 0:
            raise RuntimeError(f'FluidSynth sfload failed: {request.soundfont}')
        # Absence means no bank CC was received, distinct from explicit zero.
        # Each CC updates one half; the other retains its value (initially zero).
        banks: dict[int, list[int]] = {}
        assigned = set()
        cursor = 0

        def advance(target: int) -> None:
            nonlocal cursor
            while cursor < target:
                count = min(BLOCK_FRAMES, target - cursor)
                samples = np.asarray(synth.get_samples(count))
                if samples.dtype != np.int16 or samples.shape != (count * 2,):
                    raise RuntimeError('FluidSynth get_samples returned invalid stereo int16 block')
                audio[cursor:cursor + count] = samples.reshape(count, 2).astype(np.float32) / 32768.0
                cursor += count

        for event in source.events:
            msg, channel = event.message, event.channel
            if msg.is_meta:
                continue
            assert channel is not None  # Preflight admits only channel messages.
            # Fraction arithmetic + round-half-even at each absolute position;
            # never accumulate rounded delta times or poll a wall clock.
            advance(round(source.tempo_map.seconds(event.tick) * request.sample_rate))
            if msg.type == 'control_change':
                _checked(synth, 'cc', channel, msg.control, msg.value)
                if msg.control in (0, 32):
                    banks.setdefault(channel, [0, 0])[msg.control // 32] = msg.value
            elif msg.type == 'program_change':
                # Bank 128 is only the percussion default, never an override.
                msb, lsb = banks.get(channel, [1 if channel == 9 else 0, 0])
                bank = msb * 128 + lsb
                _checked(synth, 'program_select', channel, sfid, bank, msg.program)
                assigned.add(channel)
            elif msg.type == 'pitchwheel':
                _checked(synth, 'pitch_bend', channel, msg.pitch)
            elif msg.type == 'note_on' and msg.velocity:
                if channel not in assigned:
                    _checked(synth, 'program_select', channel, sfid, 128 if channel == 9 else 0, 0)
                    assigned.add(channel)
                _checked(synth, 'noteon', channel, msg.note, msg.velocity)
            else:
                _checked(synth, 'noteoff', channel, msg.note)
        advance(frames)
        if not source.notes:
            audio.fill(0)  # No notes: do not publish native int16 dither as music.
    except BaseException as exc:
        failure = exc
        raise
    finally:
        # delete() owns the synth/settings and loaded fonts; no driver exists.
        try:
            synth.delete()
        except Exception as exc:
            if failure is None:
                raise
            failure.add_note(f'FluidSynth cleanup also failed: {exc}')
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(request.output_path, audio, request.sample_rate, subtype='FLOAT')
    return request.output_path
