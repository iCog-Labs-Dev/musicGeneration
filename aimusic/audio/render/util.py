"""Shared audio render helpers (no fluidsynth dependency)."""

from __future__ import annotations

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
