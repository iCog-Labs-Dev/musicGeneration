"""Deterministic stem rendering helpers (simple backend)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi as pm
import soundfile as sf

from aimusic.audio.render.simple_r import _peak_dbfs, render_stem
from aimusic.audio.render.util import as_mono, peak_normalize


def render_click(
    bar_boundaries_sec: list[float],
    *,
    sample_rate: int = 48000,
    output_path: Path | str,
    peak_dbfs: float = -12.0,
) -> Path:
    """Simple metronome click at bar starts."""
    if len(bar_boundaries_sec) < 2:
        duration = 1.0
    else:
        duration = float(bar_boundaries_sec[-1])
    n = max(1, int(duration * sample_rate))
    audio = np.zeros(n, dtype=np.float64)
    click_len = int(0.01 * sample_rate)
    t = np.arange(click_len) / sample_rate
    click = np.sin(2 * np.pi * 1000.0 * t) * np.linspace(1.0, 0.0, click_len)
    for b in bar_boundaries_sec[:-1]:
        i = int(b * sample_rate)
        if 0 <= i < n:
            j = min(n, i + click_len)
            audio[i:j] += click[: j - i]
    audio = peak_normalize(audio, peak_dbfs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio.astype(np.float32), sample_rate, subtype="FLOAT")
    return out


def rough_mix(
    stem_paths: list[Path | str], *, output_path: Path | str, peak_dbfs: float = -12.0
) -> Path:
    """Sum stems to a rough mono mix for listening."""
    waves: list[np.ndarray] = []
    sr = None
    for p in stem_paths:
        data, file_sr = sf.read(str(p), always_2d=False)
        data = as_mono(np.asarray(data, dtype=np.float64))
        if sr is None:
            sr = int(file_sr)
        waves.append(data)
    if not waves or sr is None:
        raise ValueError("no stems to mix")
    max_len = max(w.shape[0] for w in waves)
    mix = np.zeros(max_len, dtype=np.float64)
    for w in waves:
        mix[: w.shape[0]] += w
    mix = peak_normalize(mix, peak_dbfs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.astype(np.float32), sr, subtype="FLOAT")
    return out


def write_bar_boundary_table(
    bar_boundaries_sec: list[float],
    *,
    output_path: Path | str,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"bar": i, "start_sec": float(t)} for i, t in enumerate(bar_boundaries_sec)]
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out


def render_role_stems(
    midi_path: Path | str,
    *,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Render one WAV per instrument/role; also click, mix, bar table."""
    path = Path(midi_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or {}
    sr = int(cfg.get("render", {}).get("sample_rate", 48000))
    peak_dbfs = _peak_dbfs(cfg)

    midi = pm.PrettyMIDI(str(path))
    stems: dict[str, Path] = {}
    for idx, inst in enumerate(midi.instruments):
        role = (inst.name or f"track_{idx}").strip().lower().replace(" ", "_") or f"track_{idx}"
        stem_path = out_dir / f"{role}.wav"
        render_stem(
            path,
            output_path=stem_path,
            config=cfg,
            instrument_index=idx,
            sample_rate=sr,
        )
        key = role
        n = 1
        while key in stems:
            n += 1
            key = f"{role}_{n}"
        stems[key] = stem_path

    bounds = (analysis or {}).get("bar_boundaries_sec")
    if not bounds:
        end = float(midi.get_end_time())
        beat = 0.5
        if hasattr(midi, "get_tempo_changes"):
            _t, tempos = midi.get_tempo_changes()
            if len(tempos):
                beat = 60.0 / float(tempos[0])
        bar = beat * 4.0
        n_bars = max(1, round(end / bar))
        bounds = [i * bar for i in range(n_bars + 1)]
        bounds[-1] = end

    click_path = out_dir / "click.wav"
    render_click(bounds, sample_rate=sr, output_path=click_path, peak_dbfs=peak_dbfs)
    stems["click"] = click_path

    mix_path = out_dir / "rough_mix.wav"
    rough_mix(
        [p for k, p in stems.items() if k != "click"],
        output_path=mix_path,
        peak_dbfs=peak_dbfs,
    )
    stems["rough_mix"] = mix_path

    table_path = out_dir / "bar_boundaries.json"
    write_bar_boundary_table(bounds, output_path=table_path)
    stems["bar_boundaries"] = table_path
    return stems
