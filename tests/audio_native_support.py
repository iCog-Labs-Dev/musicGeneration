"""Local-only native test prerequisites; never downloads or blesses an asset."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from aimusic.audio.render.registry import file_digest

FIELDS = ('soundfont_path', 'soundfont_sha256', 'soundfont_name', 'soundfont_version',
          'source_url', 'source_commit', 'license_url', 'license_summary', 'native_version', 'binding_version')


def load_native_manifest() -> dict[str, Any]:
    path = os.environ.get('AIMUSIC_FLUIDSYNTH_MANIFEST')
    assert path, 'AIMUSIC_FLUIDSYNTH_MANIFEST must name a local approved-asset JSON manifest (docs/audio-testing.md)'
    assert Path(path).is_file(), 'AIMUSIC_FLUIDSYNTH_MANIFEST is not a file'
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    assert isinstance(data, dict), 'AIMUSIC_FLUIDSYNTH_MANIFEST must contain an object'
    for key in FIELDS:
        assert isinstance(data.get(key), str) and data[key].strip(), f'{key} is required'
    assert data.get('license_approved') is True, 'license_approved must be true (explicit local user approval)'
    for key in ('source_url', 'license_url'):
        assert data[key].startswith('https://'), f'{key} must be an HTTPS provenance URL'
    assert re.fullmatch('[a-f0-9]{40}', data['source_commit']), 'source_commit must be a pinned full commit'
    assert re.fullmatch('[a-f0-9]{64}', data['soundfont_sha256']), 'soundfont_sha256 must be a SHA-256'
    asset = Path(data['soundfont_path'])
    assert asset.is_absolute() and asset.is_file(), 'soundfont_path must be an absolute local file'
    actual = file_digest(asset)
    assert actual == data['soundfont_sha256'], f'soundfont_sha256 mismatch: expected={data["soundfont_sha256"]}, actual={actual}'
    return data


def verify_native_runtime(manifest: dict[str, Any]) -> dict[str, str]:
    from aimusic.audio.render import fluidsynth_r
    try:
        runtime = dict(fluidsynth_r.versions(fluidsynth_r.load_binding()))
    except Exception as exc:
        raise AssertionError(f'native prerequisites unavailable: {exc}') from exc
    for key, package in (('native_version', 'libfluidsynth'), ('binding_version', 'pyfluidsynth')):
        assert runtime.get(package) == manifest[key], f'{key}: expected={manifest[key]}, actual={runtime.get(package)}'
    return runtime
