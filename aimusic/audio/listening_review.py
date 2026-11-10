"""Offline, bounded four-mix review; no DSP, asset lookup or acceptance inference."""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import hashlib
from html import escape
import os
from pathlib import Path
import stat
import struct

MAX_TOTAL_WAV_BYTES = 32 * 1024 * 1024
DEMOS = (
    ('fixture-12', 'Eight-bar 12-EDO fixture', 'Drums, bass, comping & lead; seed 11, eight bars.'),
    ('generated-12', 'Fresh 12-EDO passage', 'Method A planner + decoder; seed 11, four beats, retained RenderPackage.'),
    ('reference-19', 'Controlled 19-EDO reference tones', 'Three dry tones: A4 + 1, 4 and 7 steps of 19-EDO. Controlled mono evidence only.'),
    ('multitrack-19', '19-EDO multitrack passage', 'Bass, three-voice comping & lead; per-channel pitch wheels, polyphonic f0 unmeasured.'),
)
CHECKLIST = (
    'Play all four mixes quietly first; check recognizable roles and percussion.',
    'Check gross timing, lost voices, stuck notes, clipping and truncated tails.',
    'Review controlled-tone reports separately; do not infer polyphonic accuracy.',
    'Compare the separate groove-on/off evidence at matched gain before claiming a groove finding.',
)


def _validate_wav(data: bytes) -> None:
    """Admit metadata-free PCM/float RIFF WAVs used by owned rough mixes.

    Reject, rather than strip, unknown chunks: embedded bytes must stay exact and
    ancillary text (e.g. paths in LIST/BEXT/iXML) must not leak into the page.
    This is container validation, not a signal-quality or provenance certificate.
    """
    def invalid() -> ValueError:
        return ValueError('expected a nonempty PCM/float WAV without ancillary metadata')

    if (len(data) < 12 or data[:4] != b'RIFF' or data[8:12] != b'WAVE'
            or struct.unpack_from('<I', data, 4)[0] != len(data) - 8):
        raise invalid()
    chunks: dict[bytes, tuple[int, int]] = {}
    pos = 12
    while pos < len(data):
        if pos + 8 > len(data):
            raise invalid()
        name, size = struct.unpack_from('<4sI', data, pos)
        start = pos + 8
        pos = start + size + size % 2
        if name not in (b'fmt ', b'fact', b'PEAK', b'data') or name in chunks or pos > len(data):
            raise invalid()
        chunks[name] = (start, size)
    if b'fmt ' not in chunks or b'data' not in chunks:
        raise invalid()
    start, size = chunks[b'fmt ']
    if size != 16:
        raise invalid()
    kind, channels, rate, byte_rate, align, bits = struct.unpack_from('<HHIIHH', data, start)
    if (kind not in (1, 3) or channels not in (1, 2) or not 1 <= rate <= 384000
            or bits not in ((8, 16, 24, 32) if kind == 1 else (32, 64))
            or align != channels * bits // 8 or byte_rate != rate * align):
        raise invalid()
    if not chunks[b'data'][1] or chunks[b'data'][1] % align:
        raise invalid()
    for name, expected in ((b'fact', 4), (b'PEAK', 8 + 8 * channels)):
        if name in chunks and chunks[name][1] != expected:
            raise invalid()


def _directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'expected a non-symlink directory: {path}')


def _read_mix(directory: Path, remaining: int) -> bytes:
    _directory(directory)
    _directory(directory / 'stems')
    path = directory / 'stems/rough_mix.wav'
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'expected a regular non-symlink WAV: {path}')
    if not info.st_mode & 0o444:
        raise PermissionError(f'WAV is not readable: {path}')
    if info.st_size > remaining:
        raise ValueError('total WAV byte cap exceeded (32 MiB)')
    # Nonblocking/no-follow protect the final component if replaced after lstat.
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        current = os.fstat(stream.fileno())
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f'expected a regular WAV: {path}')
        if current.st_size > remaining:
            raise ValueError('total WAV byte cap exceeded (32 MiB)')
        data = stream.read(remaining + 1)
    if len(data) > remaining:
        raise ValueError('total WAV byte cap exceeded (32 MiB)')
    _validate_wav(data)
    return data


def write_listening_review(outputs: Mapping[str, Path], out: Path, *, title: str = 'Four rough listening demos') -> Path:
    """Create a new HTML file from exactly four verified workflow output dirs.

    Keys are DEMOS IDs; each directory contains stems/rough_mix.wav. Callers own
    the directories and verify provenance separately. No manifests, assets or
    cache are read. Text is escaped, order/UTF-8 bytes fixed, total raw WAV input
    capped at 32 MiB. Existing destinations (including links) are never opened for
    overwrite. Parent must exist and must not itself be a symlink. Concurrent
    hostile replacement of ancestor directories is outside this local tool's
    contract. A write/process failure may leave a partial new file; inspect it
    and choose a fresh destination, never silently overwrite it.
    """
    if set(outputs) != {name for name, _, _ in DEMOS}:
        raise ValueError('exactly the four named demo workflow directories are required')
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        raise ValueError('title must contain 1–200 characters')
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(f'refusing existing output: {out}')
    _directory(out.parent)
    remaining = MAX_TOTAL_WAV_BYTES
    sections = []
    for name, label, description in DEMOS:
        data = _read_mix(Path(outputs[name]), remaining)
        remaining -= len(data)
        encoded = base64.b64encode(data).decode('ascii')
        sections.append(
            f'<section aria-labelledby="{escape(name, quote=True)}">\n'
            f'<h2 id="{escape(name, quote=True)}">{escape(label)}</h2>\n'
            f'<p>{escape(description)}</p>\n'
            f'<audio controls preload="metadata" aria-labelledby="{escape(name, quote=True)}" '
            f'src="data:audio/wav;base64,{encoded}">Your browser does not support WAV playback.</audio>\n'
            f'<p class="digest">WAV SHA-256: <code>{hashlib.sha256(data).hexdigest()}</code></p>\n'
            '</section>'
        )
    checks = '\n'.join(f'<li><label><input type="checkbox"> {escape(text)}</label></li>' for text in CHECKLIST)
    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink: #18212b; --paper: #fff; --rule: #bbc3ca; }}
body {{ font: 1rem/1.5 system-ui, sans-serif; color: var(--ink); background: var(--paper); margin: 0; }}
main {{ max-width: 52rem; margin: auto; padding: 1rem; }}
h1 {{ font-size: 1.75rem; }} h2 {{ font-size: 1.25rem; }}
section {{ border-top: 1px solid var(--rule); padding: .5rem 0; }}
audio {{ width: 100%; }} .digest {{ font-size: .75rem; overflow-wrap: anywhere; }}
ul {{ padding-left: 1.5rem; }} li {{ margin: .5rem 0; }}
:focus-visible {{ outline: 3px solid #005a9c; outline-offset: 3px; }}
</style>
</head>
<body><main>
<h1>{escape(title)}</h1>
<p>Four embedded rough mixes, no external media or network required. Start at low volume.</p>
<p>Not mastered; no CLAP/FAD scores; polyphonic f0 unmeasured. These demos use dry effects and a fixed 0.25 s release tail, not guaranteed complete decay.</p>
{chr(10).join(sections)}
<section aria-labelledby="checklist"><h2 id="checklist">Listening checklist</h2>
<p>Checks are local reminders, not saved acceptance. Record reviewer, date and findings separately. Generating this page does not certify quality or human approval.</p>
<ul>{checks}</ul></section>
</main></body>
</html>
'''
    # Exclusive creation also closes the exists-check race, including dangling links.
    with out.open('xb') as stream:
        stream.write(page.encode('utf-8'))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, help='native evidence root containing four-workflows/<name>/cold')
    for name, _, _ in DEMOS:
        parser.add_argument('--' + name, type=Path, help='verified output directory containing stems/rough_mix.wav')
    parser.add_argument('--out', type=Path, required=True, help='new external HTML file; parent must exist')
    parser.add_argument('--title', default='Four rough listening demos')
    args = parser.parse_args(argv)
    outputs = {name: getattr(args, name.replace('-', '_')) for name, _, _ in DEMOS}
    if args.evidence_root is not None:
        if any(value is not None for value in outputs.values()):
            parser.error('use --evidence-root OR all four workflow directory flags')
        outputs = {name: args.evidence_root / 'four-workflows' / name / 'cold' for name in outputs}
    elif any(value is None for value in outputs.values()):
        parser.error('use --evidence-root OR all four workflow directory flags')
    try:
        write_listening_review(outputs, args.out, title=args.title)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
