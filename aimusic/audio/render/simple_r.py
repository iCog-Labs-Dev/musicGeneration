"""Deterministic numpy additive synth for CI when fluidsynth is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi as pm
import soundfile as sf

from aimusic.audio.manifest import write_manifest
from aimusic.audio.render.util import as_mono, peak_normalize


def _peak_dbfs(cfg: dict[str, Any]) -> float:
    render = cfg.get("render", {})
    if "peak_dbfs" in render:
        return float(render["peak_dbfs"])
    if "headroom_dbfs" in render:
        return float(render["headroom_dbfs"])
    return -12.0


def render_stem(
    midi_path: Path | str,
    *,
    soundfont: Path | str | None = None,
    sample_rate: int = 48000,
    output_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
    instrument_index: int | None = None,
) -> Path:
    """Render MIDI to a mono WAV via additive sines (not production quality)."""
    _ = soundfont  # unused; simple backend ignores SoundFont
    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")
    cfg = config or {}
    sr = int(cfg.get("render", {}).get("sample_rate", sample_rate))
    peak_dbfs = _peak_dbfs(cfg)

    midi = pm.PrettyMIDI(str(path))
    instruments = midi.instruments
    if instrument_index is not None:
        instruments = [midi.instruments[instrument_index]]

    duration = max(float(midi.get_end_time()) + 0.25, 0.25)
    n = int(duration * sr)
    audio = np.zeros(n, dtype=np.float64)
    for inst in instruments:
        for note in inst.notes:
            f0 = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            i0 = int(note.start * sr)
            i1 = int(note.end * sr)
            if i1 <= i0:
                i1 = i0 + 1
            i1 = min(i1, n)
            i0 = max(0, min(i0, n - 1))
            t = np.arange(i1 - i0) / sr
            amp = note.velocity / 127.0 * 0.2
            env = np.linspace(1.0, 0.2, i1 - i0)
            audio[i0:i1] += amp * env * np.sin(2 * np.pi * f0 * t)

    audio = peak_normalize(as_mono(audio), peak_dbfs)
    if output_path is None:
        out_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{path.stem}.wav"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio.astype(np.float32), sr, subtype="FLOAT")
    write_manifest(
        "render",
        output_path,
        [path],
        config_subset={"sample_rate": sr, "peak_dbfs": peak_dbfs, "backend": "simple"},
        metadata={"backend": "simple"},
        code_version=str(cfg.get("code_version", "0.1.0")),
    )
    return output_path
