"""Deterministic, non-destructive trim/gain copies of admitted review WAVs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
from pathlib import Path
import struct
import tempfile

import numpy as np
import soundfile as sf

from aimusic.audio.listening_review import _validate_wav
from aimusic.audio.publication import pin_output
from aimusic.audio.review_bundle import MAX_FILE_BYTES, FileRef, PresentationNode, RenderNode


def _without_peak(data: bytes) -> bytes:
    # libsndfile's optional PEAK chunk contains a wall-clock timestamp. Keep only
    # format, sample-count and audio chunks, never renderer or writer metadata.
    _validate_wav(data)
    pieces = []
    pos = 12
    while pos < len(data):
        name, size = struct.unpack_from('<4sI', data, pos)
        end = pos + 8 + size + size % 2
        if name != b'PEAK':
            pieces.append(data[pos:end])
        pos = end
    body = b'WAVE' + b''.join(pieces)
    return b'RIFF' + struct.pack('<I', len(body)) + body


def _check_signal(pcm: np.ndarray, ceiling: float) -> None:
    if not np.isfinite(pcm).all():
        raise ValueError('nonfinite presentation audio')
    if float(np.max(np.abs(pcm))) > ceiling:
        raise ValueError('presentation audio peak exceeds ceiling')


def write_presentation(root: Path, *, render: RenderNode, source_wav: FileRef,
                       output: str, start_sample: int, stop_sample: int,
                       gain: float = 1.0, max_bytes: int = MAX_FILE_BYTES,
                       max_duration_sec: float = 120.0, max_peak: float = .95) -> PresentationNode:
    """Create a presentation in an existing private tree; return its graph node.

    Paths are root-relative; output must be an opaque ``media/s<hex>.wav``.
    Caller supplies an admitted render from read_candidate/ReviewMatrix. This
    does not repeat render admission or publish a complete review bundle.
    Trim indices are frames (all channels), start inclusive and stop exclusive.
    Apply linear gain in float64, then encode in the original WAV subtype using
    libsndfile's conversion, without dither, resampling or normalization. Only
    fmt/fact/data chunks survive; source bytes, rate and channel order are intact.
    Repeatability is within the same numeric/libsndfile runtime.

    Sources are capped at 32 MiB, 120 seconds and peak .95, including trimmed-out
    audio. Output caps may only tighten those limits. Both the pre-encoding and
    quantized output peaks must pass; no clipping or tolerance is applied.
    All reads use the trusted-local-root/no-link contract of FileRef; output
    publication uses the shared retained-identity, create-only transaction.
    """
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_FILE_BYTES:
        raise ValueError('invalid presentation byte cap')
    for value, ceiling, label in ((max_duration_sec, 120, 'duration'), (max_peak, .95, 'peak')):
        if type(value) not in (float, int) or not 0 < value <= ceiling:
            raise ValueError(f'invalid presentation {label} cap')
    render.__post_init__()
    render.require_admitted()
    plan = PresentationNode(render.sha256, source_wav, FileRef(output, '0' * 64, 0),
                            start_sample, stop_sample, gain)
    if source_wav not in render.reports:
        raise ValueError('presentation source WAV must link to admitted render evidence')
    root = Path(root)
    data = source_wav.read_verified(root)
    _validate_wav(data)
    with sf.SoundFile(io.BytesIO(data)) as stream:
        if stop_sample > stream.frames:
            raise ValueError('trim exceeds source frames')
        rate, subtype = stream.samplerate, stream.subtype
        if stream.frames / rate > 120 or (stop_sample - start_sample) / rate > max_duration_sec:
            raise ValueError('presentation duration exceeds cap')
        pcm = stream.read(dtype='float64', always_2d=True)
        if len(pcm) != stream.frames:
            raise ValueError('unaligned presentation audio')
    _check_signal(pcm, .95)
    transformed = pcm[start_sample:stop_sample] * plan.gain
    _check_signal(transformed, max_peak)
    encoded = io.BytesIO()
    sf.write(encoded, transformed, rate, format='WAV', subtype=subtype)
    result = _without_peak(encoded.getvalue())
    _validate_wav(result)
    if len(result) > max_bytes:
        raise ValueError('presentation output byte cap exceeded')
    decoded, _ = sf.read(io.BytesIO(result), dtype='float64', always_2d=True)
    _check_signal(decoded, max_peak)
    node = replace(plan, output=FileRef(output, hashlib.sha256(result).hexdigest(), len(result)))
    target = root / output
    with pin_output(target.parent) as pinned:
        pinned.check_new_files([target.name])
        with tempfile.TemporaryDirectory(prefix='review-presentation-') as tmp:
            staging = Path(tmp)
            (staging / target.name).write_bytes(result)
            with pinned.publication(staging, [target.name], replace=False, sources=(root / source_wav.path,)):
                # A concurrent source edit invalidates the node; the shared
                # transaction removes only our output, never the changed source.
                source_wav.read_verified(root)
    return node
