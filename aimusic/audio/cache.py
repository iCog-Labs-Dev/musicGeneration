"""Complete content-addressed stage entries; no success marker without digests."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aimusic.audio.render.registry import file_digest
from aimusic.audio.publication import validate_names

MANIFEST = 'cache_manifest.json'
CACHE_VERSION = 1


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def output_digests(directory: Path, names: list[str]) -> dict[str, str]:
    validate_names(names)
    result = {}
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'missing/non-regular cache output: {name}')
        result[name] = file_digest(path)
    return result


def verified_entry(directory: Path, key: str, names: list[str]) -> bool:
    try:
        if directory.is_symlink() or not directory.is_dir() or (directory / MANIFEST).is_symlink():
            return False
        data = json.loads((directory / MANIFEST).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return False
        return (data.get('schema_version') == CACHE_VERSION and data.get('complete') is True
                and data.get('key') == key and fingerprint(data.get('fingerprint')) == key
                and set(p.name for p in directory.iterdir()) == set(names) | {MANIFEST}
                and data.get('outputs') == output_digests(directory, names))
    except (OSError, ValueError, TypeError):
        return False


def cached_stage(root: Path, identity: dict[str, Any], names: list[str],
                 produce: Callable[[Path], Any], *,
                 verify_inputs: Callable[[], None] = lambda: None) -> tuple[Path, bool]:
    """Stage off-path, check all expected artifacts, manifest last, atomic rename.

    A killed producer may leave a dot-prefixed scratch directory; it is never a
    cache candidate. An intact concurrent winner is reused rather than replaced.
    """
    validate_names([*names, MANIFEST])
    verify_inputs()
    key = fingerprint(identity)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    if verified_entry(destination, key, names):
        return destination, True
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=root) as tmp:
        staging = Path(tmp)
        produce(staging)
        verify_inputs()
        if set(p.name for p in staging.iterdir()) != set(names):
            raise ValueError('cache producer did not produce exactly the expected artifacts')
        digests = output_digests(staging, names)
        manifest = {'schema_version': CACHE_VERSION, 'complete': True, 'key': key,
                    'fingerprint': identity, 'outputs': digests}
        (staging / MANIFEST).write_text(canonical(manifest), encoding='utf-8')
        if not verified_entry(staging, key, names):
            raise ValueError('cache outputs changed during publication')
        verify_inputs()
        if verified_entry(destination, key, names):
            return destination, False
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        try:
            staging.rename(destination)
        except OSError:
            if not verified_entry(destination, key, names):
                raise
        return destination, False


@contextmanager
def restored_entry(source: Path, key: str, names: list[str]) -> Iterator[Path]:
    """Copy an untrusted cache into private scratch, then verify the request."""
    validate_names([*names, MANIFEST])
    with tempfile.TemporaryDirectory(prefix='.restore-') as tmp:
        staging = Path(tmp)
        if not verified_entry(source, key, names):
            raise ValueError('cache changed before restore; retry')
        for name in [*names, MANIFEST]:
            shutil.copyfile(source / name, staging / name)
        if not verified_entry(staging, key, names):
            raise ValueError('cache changed during restore; retry')
        yield staging
