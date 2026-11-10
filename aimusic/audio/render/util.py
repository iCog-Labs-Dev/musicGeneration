"""Shared audio render helpers (no fluidsynth dependency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def peak_normalize(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32, copy=False)
    target = 10.0 ** (peak_dbfs / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def as_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)


def render_frames(sample_rate: int, duration_sec: float, release_tail_sec: float) -> int:
    """One horizon for all outputs, rounded up once (never truncate the tail)."""
    import math
    from fractions import Fraction
    if sample_rate <= 0 or any(not math.isfinite(v) or v < 0 for v in (duration_sec, release_tail_sec)):
        raise ValueError('render sample rate/horizon/release tail must be finite and nonnegative')
    return max(1, math.ceil((Fraction(str(duration_sec)) + Fraction(str(release_tail_sec))) * sample_rate))


def limit_peak(audio: np.ndarray, ceiling: float = 0.95) -> tuple[np.ndarray, float]:
    """One attenuation-only gain; silence stays silent, FLOAT WAV respects ceiling."""
    import math
    if not math.isfinite(ceiling) or not 0 < ceiling <= 1:
        raise ValueError('peak ceiling must be finite and in (0, 1]')
    if not np.isfinite(audio).all():
        raise ValueError('audio contains nonfinite samples')
    target = np.float32(ceiling)
    if float(target) > ceiling:
        target = np.nextafter(target, np.float32(0))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    gain = min(1.0, float(target) / peak) if peak else 1.0
    return (audio * gain).astype(np.float32), gain


def write_float_wav(path: Path | str, audio: np.ndarray, sample_rate: int) -> None:
    """Write owned FLOAT WAV, canonicalizing only libsndfile's PEAK timestamp.

    RIFF chunks are length-delimited (including odd-byte padding); never search
    PCM for marker bytes. PEAK v1 stores version, Unix timestamp, then per-channel
    peak/position pairs. A zero timestamp is intentional, not render wall time.
    """
    import struct
    import soundfile as sf
    sf.write(path, audio, sample_rate, subtype='FLOAT', format='WAV')
    with open(path, 'r+b') as stream:
        header = stream.read(12)
        if header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('Expected RIFF WAVE output')
        end = struct.unpack('<I', header[4:8])[0] + 8
        while stream.tell() < end:
            kind, size = struct.unpack('<4sI', stream.read(8))
            position = stream.tell()
            if kind == b'PEAK':
                if size < 8 or stream.read(4) != b'\x01\x00\x00\x00':
                    raise ValueError('Unsupported WAV PEAK chunk')
                stream.write(b'\x00' * 4)
            stream.seek(position + size + size % 2)
