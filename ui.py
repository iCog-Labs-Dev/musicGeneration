from __future__ import annotations

import base64
import html
import json
import math
import shutil
import subprocess
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import mido
import numpy as np

from aimusic.app.cli import _build_structural_diagnostics, _json_ready
from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    MicrotonalRendering,
    StyleConfig,
)
from aimusic.core.diagnostics import RunManifest, SBDiagnostics
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_GROOVE_FAMILIES, DEFAULT_METER_SIGNATURES
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render import render_midi
from aimusic.render.midi_render import TrackInstrumentConfig
from aimusic.theory.edo import EDO


OUTPUT_DIR = Path("./outputs")
SOUNDFONT_PATH = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
PREVIEW_SAMPLE_RATE = 22050
PREVIEW_TAIL_SECONDS = 0.35
MIDI_DRUM_CHANNEL = 9
PITCH_CLASS_NAMES_12 = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
DRUM_NOTE_NAMES = {
    35: "Acoustic kick",
    36: "Kick",
    38: "Snare",
    40: "Electric snare",
    42: "Closed hat",
    44: "Pedal hat",
    46: "Open hat",
}


@dataclass(frozen=True)
class GenerationParams:
    seed: int
    beats: int
    edo: int
    meter: str
    groove_family: str
    tempo_bpm: float
    sample_path: bool
    drum_density: float
    bass_density: float
    comping_density: float
    lead_density: float
    pitch_bend_range: int
    rendering_method: str
    bass_program: int
    comping_program: int
    lead_program: int
    drum_track: list[str]


@dataclass(frozen=True)
class GeneratedArtifacts:
    run_id: str
    score_path: Path
    midi_path: Path
    manifest_path: Path
    wav_path: Path


@dataclass(frozen=True)
class MidiPreviewNote:
    start_time: float
    end_time: float
    midi_note: int
    velocity: int
    channel: int
    pitch_bend: int = 0
    pitch_bend_range: float = 2.0


class MidiAudioConversionError(RuntimeError):
    """Raised when no usable MIDI-to-WAV converter is available."""


def _as_int(label: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc

    if not numeric.is_integer():
        raise ValueError(f"{label} must be an integer.")

    integer = int(numeric)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and integer > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return integer


def _as_positive_float(label: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc

    if numeric <= 0:
        raise ValueError(f"{label} must be > 0.")
    return numeric


def _as_unit_float(label: str, value: Any) -> float:
    numeric = _as_positive_or_zero_float(label, value)
    if numeric > 1:
        raise ValueError(f"{label} must be <= 1.")
    return numeric


def _as_positive_or_zero_float(label: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc

    if numeric < 0:
        raise ValueError(f"{label} must be >= 0.")
    return numeric


def _normalize_inputs(
    seed: Any,
    beats: Any,
    edo: Any,
    meter: str,
    groove_family: str,
    tempo_bpm: Any,
    sample_path: bool,
    drum_density: Any,
    bass_density: Any,
    comping_density: Any,
    lead_density: Any,
    pitch_bend_range: Any,
    rendering_method: str,
    bass_program: Any,
    comping_program: Any,
    lead_program: Any,
    drum_track: list[str],
) -> GenerationParams:
    meter = str(meter).strip()
    groove_family = str(groove_family).strip()
    rendering_method = str(rendering_method).strip()

    if not meter:
        raise ValueError("meter must not be empty.")
    if not groove_family:
        raise ValueError("groove family must not be empty.")
    if not drum_track:
        raise ValueError("At least one track must be selected for drums.")
    supported_rendering_names = tuple(MicrotonalRendering.__members__)
    if rendering_method not in supported_rendering_names:
        raise ValueError(
            "rendering method must be one of "
            f"{', '.join(supported_rendering_names)}."
        )

    return GenerationParams(
        seed=_as_int("seed", seed),
        beats=_as_int("beats", beats, minimum=1),
        edo=_as_int("edo", edo, minimum=1),
        meter=meter,
        groove_family=groove_family,
        tempo_bpm=_as_positive_float("tempo bpm", tempo_bpm),
        sample_path=bool(sample_path),
        drum_density=_as_unit_float("drum density", drum_density),
        bass_density=_as_unit_float("bass density", bass_density),
        comping_density=_as_unit_float("comping density", comping_density),
        lead_density=_as_unit_float("lead density", lead_density),
        pitch_bend_range=_as_int("pitch bend range", pitch_bend_range, minimum=1),
        rendering_method=rendering_method,
        bass_program=_as_int("bass program", bass_program, minimum=0, maximum=127),
        comping_program=_as_int("comping program", comping_program, minimum=0, maximum=127),
        lead_program=_as_int("lead program", lead_program, minimum=0, maximum=127),
        drum_track=drum_track,
    )


def _drum_track_names(drum_track: list[str]) -> tuple[str, ...]:
    return tuple(name.strip().lower() for name in drum_track if name.strip())


def _build_track_instruments(params: GenerationParams) -> dict[str, TrackInstrumentConfig]:
    instruments = {
        "bass": TrackInstrumentConfig(program=params.bass_program),
        "comping": TrackInstrumentConfig(program=params.comping_program),
        "lead": TrackInstrumentConfig(program=params.lead_program),
    }
    for track_name in _drum_track_names(params.drum_track):
        existing = instruments.get(track_name)
        instruments[track_name] = TrackInstrumentConfig(
            program=None if existing is None else existing.program,
            is_drum=True,
        )
    return instruments


def _generate_artifacts(params: GenerationParams) -> GeneratedArtifacts:
    style_config = StyleConfig(
        allowed_meters=(params.meter,),
        groove_families=(params.groove_family,),
    )
    decode_config = DecodeConfig(
        drum_density=params.drum_density,
        bass_density=params.bass_density,
        comping_density=params.comping_density,
        lead_density=params.lead_density,
    )
    run_config = MethodARunConfig(
        total_beats=params.beats,
        seed=params.seed,
        use_sampling=params.sample_path,
        style_config=style_config,
        decode_config=decode_config,
        edo=params.edo,
    )

    plan_result, next_key = run_method_a(run_config, key=RNGKey(seed=params.seed))
    score, _ = decode_path_to_score(
        plan_result.path,
        decode_config=decode_config,
        vocabularies=plan_result.vocabularies,
        edo=params.edo,
        tempo_bpm=params.tempo_bpm,
        key=next_key,
    )
    structural_stats = _build_structural_diagnostics(
        plan_result.path,
        plan_result.vocabularies,
        edo=params.edo,
        sections=plan_result.endpoints.sections,
    )
    track_instruments = _build_track_instruments(params)
    manifest = RunManifest(
        seed=params.seed,
        config_dump=_json_ready(
            {
                "run_config": run_config,
                "meter": params.meter,
                "groove_family": params.groove_family,
                "tempo_bpm": params.tempo_bpm,
                "output_dir": str(OUTPUT_DIR),
                "pitch_bend_range": params.pitch_bend_range,
                "rendering_method": params.rendering_method,
                "track_instruments": track_instruments,
            }
        ),
        structural_stats=structural_stats,
        sb_stats=SBDiagnostics.from_solution(plan_result.sb_solution),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / f"{manifest.run_id}_score.json"
    midi_path = OUTPUT_DIR / f"{manifest.run_id}.mid"
    manifest_path = OUTPUT_DIR / f"{manifest.run_id}_manifest.json"
    wav_path = OUTPUT_DIR / f"{manifest.run_id}.wav"

    with score_path.open("w", encoding="utf-8") as f:
        json.dump(score.to_dict(), f, indent=2)

    render_midi(
        score,
        EDO(
            EDOConfig(
                n=params.edo,
                base_tuning=0,
                pitch_bend_range=params.pitch_bend_range,
                microtonal_rendering_method=MicrotonalRendering[params.rendering_method],
            )
        ),
        str(midi_path),
        track_instruments=track_instruments,
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return GeneratedArtifacts(
        run_id=manifest.run_id,
        score_path=score_path,
        midi_path=midi_path,
        manifest_path=manifest_path,
        wav_path=wav_path,
    )


def _run_converter(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _first_midi_tempo(midi_file: mido.MidiFile) -> int:
    for track in midi_file.tracks:
        for message in track:
            if message.type == "set_tempo":
                return int(message.tempo)
    return int(mido.bpm2tempo(120))


def _extract_midi_preview_notes(midi_path: Path) -> list[MidiPreviewNote]:
    midi_file = mido.MidiFile(midi_path)
    tempo = _first_midi_tempo(midi_file)
    seconds_per_tick = tempo / 1_000_000 / midi_file.ticks_per_beat
    preview_notes: list[MidiPreviewNote] = []

    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        active_notes: dict[
            tuple[int, int, int],
            list[tuple[int, int, int, float]],
        ] = {}
        pitch_bends: dict[int, int] = {}
        pitch_bend_ranges: dict[int, float] = {}
        rpn_selection: dict[int, tuple[int, int]] = {}
        rpn_msb: dict[int, int] = {}
        rpn_lsb: dict[int, int] = {}
        for message in track:
            absolute_tick += int(message.time)
            if hasattr(message, "channel") and message.type == "pitchwheel":
                pitch_bends[int(message.channel)] = int(message.pitch)
                continue
            if hasattr(message, "channel") and message.type == "control_change":
                channel = int(message.channel)
                if message.control == 101:
                    rpn_msb[channel] = int(message.value)
                elif message.control == 100:
                    rpn_lsb[channel] = int(message.value)
                rpn_selection[channel] = (
                    rpn_msb.get(channel, 127),
                    rpn_lsb.get(channel, 127),
                )
                if message.control == 6 and rpn_selection[channel] == (0, 0):
                    pitch_bend_ranges[channel] = float(message.value)
                continue
            if not hasattr(message, "channel") or not hasattr(message, "note"):
                continue

            channel = int(message.channel)
            key = (track_index, channel, int(message.note))
            if message.type == "note_on" and message.velocity > 0:
                active_notes.setdefault(key, []).append(
                    (
                        absolute_tick,
                        int(message.velocity),
                        pitch_bends.get(channel, 0),
                        pitch_bend_ranges.get(channel, 2.0),
                    )
                )
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                starts = active_notes.get(key)
                if not starts:
                    continue
                start_tick, velocity, pitch_bend, pitch_bend_range = starts.pop(0)
                if absolute_tick <= start_tick:
                    continue
                preview_notes.append(
                    MidiPreviewNote(
                        start_time=start_tick * seconds_per_tick,
                        end_time=absolute_tick * seconds_per_tick,
                        midi_note=int(message.note),
                        velocity=velocity,
                        channel=channel,
                        pitch_bend=pitch_bend,
                        pitch_bend_range=pitch_bend_range,
                    )
                )

    return preview_notes


def _midi_note_frequency(
    midi_note: int,
    pitch_bend: int = 0,
    pitch_bend_range: float = 2.0,
) -> float:
    bend_scale = 8191 if pitch_bend >= 0 else 8192
    sounding_pitch = midi_note + (pitch_bend / bend_scale) * pitch_bend_range
    return 440.0 * (2.0 ** ((sounding_pitch - 69) / 12.0))


def _note_envelope(sample_count: int, sample_rate: int) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float32)
    if sample_count <= 2:
        return envelope

    attack = min(max(1, int(sample_rate * 0.005)), sample_count // 3)
    release = min(max(1, int(sample_rate * 0.040)), sample_count // 3)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    envelope[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return envelope


def _render_drum_preview(note: MidiPreviewNote, sample_rate: int) -> np.ndarray:
    duration = max(0.04, min(note.end_time - note.start_time, 0.45))
    sample_count = max(1, int(duration * sample_rate))
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    amplitude = 0.22 * (note.velocity / 127.0)

    if note.midi_note in (35, 36):
        wave_data = np.sin(2.0 * math.pi * 58.0 * t) * np.exp(-8.0 * t)
    elif note.midi_note in (38, 40):
        rng = np.random.default_rng(note.midi_note * 1009 + sample_count)
        noise = rng.uniform(-1.0, 1.0, sample_count).astype(np.float32)
        tone = np.sin(2.0 * math.pi * 180.0 * t)
        wave_data = ((0.65 * noise) + (0.35 * tone)) * np.exp(-14.0 * t)
    elif note.midi_note in (42, 44, 46):
        wave_data = (
            np.sin(2.0 * math.pi * 2300.0 * t)
            * np.sin(2.0 * math.pi * 3100.0 * t)
            * np.exp(-32.0 * t)
        )
    else:
        wave_data = np.sin(2.0 * math.pi * 440.0 * t) * np.exp(-16.0 * t)

    return (amplitude * wave_data).astype(np.float32)


def _render_melodic_preview(note: MidiPreviewNote, sample_rate: int) -> np.ndarray:
    duration = max(0.01, note.end_time - note.start_time)
    sample_count = max(1, int(duration * sample_rate))
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    frequency = _midi_note_frequency(
        note.midi_note,
        note.pitch_bend,
        note.pitch_bend_range,
    )
    amplitude = 0.13 * (note.velocity / 127.0)
    wave_data = (
        np.sin(2.0 * math.pi * frequency * t)
        + 0.25 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
    )
    return (amplitude * wave_data * _note_envelope(sample_count, sample_rate)).astype(np.float32)


def _render_midi_preview_wav(
    midi_path: Path,
    wav_path: Path,
    *,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
) -> None:
    preview_notes = _extract_midi_preview_notes(midi_path)
    if not preview_notes:
        raise MidiAudioConversionError("MIDI preview failed because the file contained no notes.")

    duration = max(note.end_time for note in preview_notes) + PREVIEW_TAIL_SECONDS
    samples = np.zeros(max(1, int(duration * sample_rate)), dtype=np.float32)

    for note in preview_notes:
        rendered = (
            _render_drum_preview(note, sample_rate)
            if note.channel == MIDI_DRUM_CHANNEL
            else _render_melodic_preview(note, sample_rate)
        )
        start_index = max(0, int(note.start_time * sample_rate))
        end_index = min(samples.size, start_index + rendered.size)
        if end_index <= start_index:
            continue
        samples[start_index:end_index] += rendered[: end_index - start_index]

    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples * min(0.95 / peak, 1.0)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (samples * 32767.0).clip(-32768, 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _convert_midi_to_wav(midi_path: Path, wav_path: Path) -> str:
    midi_file = mido.MidiFile(midi_path)
    has_mts_tuning = any(
        message.type == "sysex"
        and tuple(message.data[:5]) == (0x7E, 0x7F, 0x08, 0x01, 0x00)
        for track in midi_file.tracks
        for message in track
    )
    if has_mts_tuning:
        raise MidiAudioConversionError(
            "The MIDI file uses MIDI Tuning Standard (MTS). Audio preview is "
            "disabled because the available preview converters cannot guarantee "
            "MTS tuning reproduction. Download the MIDI and play it with an "
            "MTS-compatible synthesizer."
        )

    conversion_errors: list[str] = []
    fluidsynth = shutil.which("fluidsynth")

    if fluidsynth is not None and SOUNDFONT_PATH.exists():
        try:
            _run_converter(
                [
                    fluidsynth,
                    "-ni",
                    str(SOUNDFONT_PATH),
                    str(midi_path),
                    "-F",
                    str(wav_path),
                    "-r",
                    "44100",
                ]
            )
            return "fluidsynth"
        except subprocess.CalledProcessError as exc:
            conversion_errors.append(f"fluidsynth failed: {exc.stderr or exc.stdout or exc}")
    elif fluidsynth is not None:
        conversion_errors.append(
            f"fluidsynth was found, but the soundfont is missing at {SOUNDFONT_PATH}."
        )

    timidity = shutil.which("timidity")
    if timidity is not None:
        try:
            _run_converter([timidity, str(midi_path), "-Ow", "-o", str(wav_path)])
            return "timidity"
        except subprocess.CalledProcessError as exc:
            conversion_errors.append(f"timidity failed: {exc.stderr or exc.stdout or exc}")

    try:
        _render_midi_preview_wav(midi_path, wav_path)
        return "built-in preview synth (MPE pitch bends applied)"
    except Exception as exc:
        conversion_errors.append(f"built-in preview synth failed: {exc}")

    details = "\n".join(conversion_errors)
    if details:
        details = f"\n\nConverter details:\n{details}"
    raise MidiAudioConversionError(
        "Could not create an audio preview for the generated MIDI. Install fluidsynth with "
        f"a General MIDI soundfont available at {SOUNDFONT_PATH}, or install timidity."
        "\n\nLinux example: sudo apt install fluidsynth fluid-soundfont-gm timidity"
        "\nmacOS example: brew install fluid-synth timidity"
        "\nWindows example: install FluidSynth and add fluidsynth.exe to PATH, or install TiMidity++."
        f"{details}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _note_label(note: dict[str, Any], edo: int) -> str:
    track = str(note.get("track", "")).lower()
    pitch_height = int(note.get("h", 0))
    if track == "drums":
        return DRUM_NOTE_NAMES.get(pitch_height, f"Drum {pitch_height}")
    if edo == 12:
        return f"{PITCH_CLASS_NAMES_12[pitch_height % 12]}{(pitch_height // 12) - 1}"
    return f"pc_{pitch_height % edo}"


def _dashboard_note_events(score_data: dict[str, Any], edo: int) -> list[dict[str, Any]]:
    ticks_per_beat = int(score_data.get("ticks_per_beat", 480))
    tempo_bpm = float(score_data.get("tempo_bpm", 120.0))
    seconds_per_tick = 60.0 / tempo_bpm / ticks_per_beat

    events = []
    for note in score_data.get("note_events", []):
        events.append(
            {
                "start": round(int(note.get("ton", 0)) * seconds_per_tick, 4),
                "end": round(int(note.get("toff", 0)) * seconds_per_tick, 4),
                "track": str(note.get("track", "default")),
                "label": _note_label(note, edo),
            }
        )
    return events


def _dashboard_chord_events(
    manifest_data: dict[str, Any],
    tempo_bpm: float,
) -> list[dict[str, Any]]:
    seconds_per_beat = 60.0 / tempo_bpm
    events = []
    for event in manifest_data.get("structure", {}).get("chord_timeline", []):
        events.append(
            {
                "start": round(float(event.get("start_time", 0.0)) * seconds_per_beat, 4),
                "end": round(float(event.get("end_time", 0.0)) * seconds_per_beat, 4),
                "label": str(event.get("label", "unknown")),
            }
        )
    return events


def _build_playback_dashboard(
    wav_path: Path,
    score_path: Path,
    manifest_path: Path,
) -> str:
    score_data = _load_json(score_path)
    manifest_data = _load_json(manifest_path)
    tempo_bpm = float(score_data.get("tempo_bpm", 120.0))
    edo = int(manifest_data.get("config", {}).get("run_config", {}).get("edo", 12))
    notes = _dashboard_note_events(score_data, edo)
    chords = _dashboard_chord_events(manifest_data, tempo_bpm)
    audio_base64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

    payload = json.dumps(
        {
            "audio": f"data:audio/wav;base64,{audio_base64}",
            "notes": notes,
            "chords": chords,
        }
    )
    iframe_document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #f8fafc;
      background: #0f1115;
    }}
    .panel {{
      border: 1px solid #2b3340;
      border-radius: 8px;
      padding: 12px;
      background: #141820;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }}
    audio {{
      width: 100%;
      height: 40px;
      margin-bottom: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 2fr;
      gap: 8px;
    }}
    .cell {{
      min-height: 54px;
      border: 1px solid #2b3340;
      border-radius: 6px;
      padding: 8px;
      background: #1b212c;
    }}
    .label {{
      color: #9aa4b2;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 18px;
      line-height: 1.25;
      font-weight: 650;
    }}
    .notes {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }}
    .chip {{
      border-radius: 6px;
      padding: 3px 6px;
      background: #243b63;
      color: #d7e7ff;
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
    }}
    .track {{
      color: #9fb5d4;
    }}
    .bar {{
      height: 5px;
      margin-top: 10px;
      border-radius: 99px;
      background: #2a303a;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      width: 0%;
      background: #7dd3fc;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <audio id="player" controls src=""></audio>
    <div class="grid">
      <div class="cell">
        <div class="label">Time</div>
        <div id="time" class="value">0.00s</div>
      </div>
      <div class="cell">
        <div class="label">Chord</div>
        <div id="chord" class="value">-</div>
      </div>
      <div class="cell">
        <div class="label">Notes</div>
        <div id="notes" class="notes"><span class="chip">-</span></div>
      </div>
    </div>
    <div class="bar"><div id="progress" class="fill"></div></div>
  </div>
  <script>
    const data = {payload};
    const player = document.getElementById("player");
    const timeEl = document.getElementById("time");
    const chordEl = document.getElementById("chord");
    const notesEl = document.getElementById("notes");
    const progressEl = document.getElementById("progress");
    player.src = data.audio;

    function currentChord(t) {{
      return data.chords.find((item) => item.start <= t && t < item.end);
    }}

    function currentNotes(t) {{
      return data.notes.filter((item) => item.start <= t && t < item.end).slice(0, 14);
    }}

    function render() {{
      const t = player.currentTime || 0;
      const chord = currentChord(t);
      const notes = currentNotes(t);
      timeEl.textContent = t.toFixed(2) + "s";
      chordEl.textContent = chord ? chord.label : "-";
      notesEl.innerHTML = notes.length
        ? notes.map((note) => `<span class="chip">${{note.label}} <span class="track">${{note.track}}</span></span>`).join("")
        : `<span class="chip">-</span>`;
      const duration = player.duration || 0;
      progressEl.style.width = duration ? `${{Math.min(100, (t / duration) * 100)}}%` : "0%";
    }}

    let timer = null;
    player.addEventListener("loadedmetadata", render);
    player.addEventListener("timeupdate", render);
    player.addEventListener("seeked", render);
    player.addEventListener("play", () => {{
      render();
      timer = window.setInterval(render, 100);
    }});
    player.addEventListener("pause", () => {{
      if (timer) window.clearInterval(timer);
      timer = null;
      render();
    }});
    player.addEventListener("ended", () => {{
      if (timer) window.clearInterval(timer);
      timer = null;
      render();
    }});
    render();
  </script>
</body>
</html>"""
    return (
        '<iframe title="MIDI playback dashboard" '
        'style="width:100%;height:185px;border:0;border-radius:8px;" '
        f'srcdoc="{html.escape(iframe_document, quote=True)}"></iframe>'
    )


def _score_summary(score_path: Path, manifest_path: Path) -> str:
    score_data = _load_json(score_path)
    manifest_data = _load_json(manifest_path)

    notes = score_data.get("note_events", [])
    ticks_per_beat = int(score_data.get("ticks_per_beat", 480))
    tempo_bpm = float(score_data.get("tempo_bpm", 120.0))
    track_counts = score_data.get("track_event_counts", {})
    max_tick = max((int(note.get("toff", 0)) for note in notes), default=0)
    duration_beats = max_tick / ticks_per_beat
    duration_seconds = duration_beats * 60.0 / tempo_bpm
    meter = manifest_data.get("config", {}).get("meter", "unknown")

    return (
        "### Analysis\n"
        f"- Tempo: {tempo_bpm:g} BPM\n"
        f"- Meter: {meter}\n"
        f"- Track count: {len(track_counts)}\n"
        f"- Duration: {duration_seconds:.2f}s ({duration_beats:.2f} beats)"
    )


def _score_table(score_path: Path, manifest_path: Path) -> str:
    score_data = _load_json(score_path)
    track_counts = score_data.get("track_event_counts", {})
    track_lines = "\n".join(
        f"| {track} | {count} |" for track, count in sorted(track_counts.items())
    )
    if not track_lines:
        track_lines = "| none | 0 |"
    return (
        "| Track | Notes |\n"
        "| --- | ---: |\n"
        f"{track_lines}"
    )


def _error_markdown(message: str, *, include_traceback: bool = False) -> str:
    if include_traceback:
        return f"### Error\n```text\n{message}\n```"
    return f"### Error\n{message}"


def generate_music(
    seed: Any,
    beats: Any,
    edo: Any,
    meter: str,
    groove_family: str,
    tempo_bpm: Any,
    sample_path: bool,
    drum_density: Any,
    bass_density: Any,
    comping_density: Any,
    lead_density: Any,
    pitch_bend_range: Any,
    rendering_method: str,
    bass_program: Any,
    comping_program: Any,
    lead_program: Any,
    drum_track: list[str],
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    try:
        params = _normalize_inputs(
            seed,
            beats,
            edo,
            meter,
            groove_family,
            tempo_bpm,
            sample_path,
            drum_density,
            bass_density,
            comping_density,
            lead_density,
            pitch_bend_range,
            rendering_method,
            bass_program,
            comping_program,
            lead_program,
            drum_track,
        )
        artifacts = _generate_artifacts(params)
        summary = _score_summary(artifacts.score_path, artifacts.manifest_path)
        table = _score_table(artifacts.score_path, artifacts.manifest_path)
    except Exception:
        return (
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            _error_markdown(traceback.format_exc(), include_traceback=True),
        )

    try:
        _ = _convert_midi_to_wav(artifacts.midi_path, artifacts.wav_path)
        dashboard = _build_playback_dashboard(
            artifacts.wav_path,
            artifacts.score_path,
            artifacts.manifest_path,
        )
    except MidiAudioConversionError as exc:
        return (
            "",
            gr.update(value=str(artifacts.midi_path), visible=True),
            gr.update(value=str(artifacts.score_path), visible=True),
            gr.update(value=str(artifacts.manifest_path), visible=True),
            gr.update(value=summary, visible=True),
            gr.update(value=table, visible=True),
            _error_markdown(str(exc)),
        )

    return (
        dashboard,
        gr.update(value=str(artifacts.midi_path), visible=True),
        gr.update(value=str(artifacts.score_path), visible=True),
        gr.update(value=str(artifacts.manifest_path), visible=True),
        gr.update(value=summary, visible=True),
        gr.update(value=table, visible=True),
        "",
    )


def _download_component(label: str) -> gr.components.Component:
    if hasattr(gr, "DownloadButton"):
        return gr.DownloadButton(label=label)
    return gr.File(label=label, interactive=False)


css = """
    .block { padding: 0 8px !important; }
    .form { gap: 2px !important; }
    .wrap { gap: 2px !important; padding: 0 !important; }
    label { margin-bottom: 0 !important; font-size: 11px !important; }
    .density-box { border: 1px solid #374151 !important; border-radius: 6px !important; padding: 2px 6px !important; }
    .density-box input[type=range] { height: 3px !important; }
    input[type=number] { padding: 1px 4px !important; }
    .drum-check .wrap { display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 12px !important; }
"""

with gr.Blocks(title="MIDI Generator", fill_height=True) as demo:
    gr.Markdown("## MIDI Generator <small style='font-weight:700;color:#9aa4b2;font-size:14px'>  | write configs and click 'generate'</small>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=0):
            seed = gr.Number(label="seed", value=11, precision=0)
            beats = gr.Number(label="beats", value=8, precision=0)
            edo = gr.Number(label="edo", value=12, precision=0)
            meter = gr.Dropdown(
                label="meter",
                choices=list(DEFAULT_METER_SIGNATURES),
                value="4/4",
                allow_custom_value=True,
            )
            groove_family = gr.Dropdown(
                label="groove-family",
                choices=list(DEFAULT_GROOVE_FAMILIES),
                value="straight",
            )
            tempo_bpm = gr.Number(label="tempo-bpm", value=120)
            pitch_bend_range = gr.Number(label="pitch-bend-range", value=2, precision=0)
            rendering_method = gr.Dropdown(
                label="rendering-method",
                choices=[method.name for method in MicrotonalRendering],
                value=MicrotonalRendering.MPE.name,
            )

        with gr.Column(scale=2):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    bass_program = gr.Number(label="track-program bass", value=34, precision=0)
                    comping_program = gr.Number(label="track-program comping", value=5, precision=0)
                    lead_program = gr.Number(label="track-program lead", value=88, precision=0)
                    drum_track = gr.CheckboxGroup(
                        label="drum-track",
                        choices=["drums", "bass", "comping", "lead"],
                        value=["drums"],
                        elem_classes="drum-check",
                    )
                with gr.Column(scale=2):
                    drum_density = gr.Slider(
                        label="drum-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.75,
                        elem_classes="density-box",
                    )
                    bass_density = gr.Slider(
                        label="bass-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.60,
                        elem_classes="density-box",
                    )
                    comping_density = gr.Slider(
                        label="comping-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.55,
                        elem_classes="density-box",
                    )
                    lead_density = gr.Slider(
                        label="lead-density",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.45,
                        elem_classes="density-box",
                    )
                    sample_path = gr.Checkbox(label="Choose Sample Path", value=False)
            generate_button = gr.Button("Generate", variant="primary")
            dashboard = gr.HTML()
            status = gr.Markdown()
            with gr.Row(equal_height=True):
                summary = gr.Markdown(visible=False, scale=1)
                table = gr.Markdown(visible=False, scale=1)
                with gr.Column(scale=1, min_width=160):
                    midi_download = _download_component("Download MIDI")
                    score_download = _download_component("Download score JSON")
                    manifest_download = _download_component("Download manifest JSON")
                    midi_download.visible = False
                    score_download.visible = False
                    manifest_download.visible = False

    generate_button.click(  # type: ignore[attr-defined]
        fn=generate_music,
        inputs=[
            seed,
            beats,
            edo,
            meter,
            groove_family,
            tempo_bpm,
            sample_path,
            drum_density,
            bass_density,
            comping_density,
            lead_density,
            pitch_bend_range,
            rendering_method,
            bass_program,
            comping_program,
            lead_program,
            drum_track,
        ],
        outputs=[
            dashboard,
            midi_download,
            score_download,
            manifest_download,
            summary,
            table,
            status,
        ],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="localhost", server_port=7860, inbrowser=True, css=css)
