"""Deterministic MIDI audition, not the production render/mix/master pipeline.

Includes pitch bend at note onset (default two-semitone range) and tempo changes. Sustain and subsequent
per-note controller changes are not modeled. Percussion uses one-shot tails
independent of MIDI note-off, as a drum sampler would.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import wave

import mido
import numpy as np


@dataclass(frozen=True)
class PreviewNote:
    start: float
    duration: float
    pitch: float
    velocity: int
    channel: int


def read_midi_notes(path: Path | str) -> tuple[PreviewNote, ...]:
    midi = mido.MidiFile(path)
    tempo = 500000
    seconds = 0.0
    bends: dict[int, int] = defaultdict(int)
    active: dict[tuple[int, int], deque[tuple[float, int, int]]] = defaultdict(deque)
    notes = []
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "pitchwheel":
            bends[message.channel] = message.pitch
        elif message.type == "note_on" and message.velocity > 0:
            active[message.channel, message.note].append((seconds, message.velocity, bends[message.channel]))
        elif message.type in ("note_on", "note_off"):
            queue = active[message.channel, message.note]
            if not queue:
                raise ValueError("MIDI contains an unmatched note-off.")
            start, velocity, bend = queue.popleft()
            pitch = float(message.note) if message.channel == 9 else message.note + bend / 8192 * 2
            notes.append(PreviewNote(start, seconds - start, pitch, velocity, message.channel))
    if any(active.values()):
        raise ValueError("MIDI contains an unclosed note.")
    return tuple(notes)


def voice(note: PreviewNote, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    percussion = note.channel == 9
    kick = percussion and round(note.pitch) in (35, 36)
    duration = max(note.duration, 0.35 if kick else 0.18) if percussion else note.duration
    t = np.arange(max(1, round(duration * sample_rate))) / sample_rate
    if kick:
        # Fundamental supplies weight; harmonics and a short transient keep
        # the beat audible on speakers that cannot reproduce 50 Hz well.
        phase = 2 * np.pi * (52 * t + 75 * 0.022 * (1 - np.exp(-t / 0.022)))
        signal = (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(4 * phase)) * np.exp(-12 * t)
        signal += 0.12 * rng.normal(size=len(t)) * np.exp(-220 * t)
        gain = 0.65
    elif percussion:
        noise = rng.normal(0, 0.45, len(t))
        hat = round(note.pitch) in (42, 44, 46)
        signal = (noise - np.roll(noise, 1)) * np.exp(-(45 if hat else 20) * t)
        gain = 0.12 if hat else 0.25
    else:
        phase = 2 * np.pi * 440 * 2 ** ((note.pitch - 69) / 12) * t
        signal = (np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)) * np.exp(-1.8 * t)
        gain = 0.20
    envelope = np.minimum(1, t / 0.004) * np.minimum(1, (duration - t) / 0.025)
    return signal * np.maximum(0, envelope) * gain * note.velocity / 127


def render_midi_preview(midi_path: Path | str, output_path: Path | str, *, sample_rate: int = 44100) -> Path:
    notes = read_midi_notes(midi_path)
    if not notes:
        raise ValueError("Cannot preview a MIDI file without notes.")
    if sample_rate < 32000:
        raise ValueError("sample_rate must be at least 32000 Hz.")
    duration = max(note.start + max(note.duration, 0.35) for note in notes) + 0.1
    mix = np.zeros((int(np.ceil(duration * sample_rate)), 2), dtype=np.float64)
    rng = np.random.default_rng(11)
    for note in notes:
        signal = voice(note, sample_rate, rng)
        offset = round(note.start * sample_rate)
        mix[offset:offset + len(signal), :] += signal[:, None] / np.sqrt(2)
    mix *= 0.9 / max(float(np.max(np.abs(mix))), 1e-9)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes((mix * 32767).astype("<i2").tobytes())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi")
    parser.add_argument("wav")
    args = parser.parse_args()
    print(render_midi_preview(args.midi, args.wav))
