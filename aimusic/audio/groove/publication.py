"""Transactional groove replacement through the shared publication policy."""
from __future__ import annotations

import tempfile
from pathlib import Path

from aimusic.audio.publication import pin_output, validate_names


def publish_bundle(source: Path, output: Path, payloads: dict[str, bytes]) -> None:
    """Preserve all old members until the complete new bundle is accepted.

    The context rolls back normal failures and retains/report recovery copies
    when restoration fails. It does not promise set-wide crash atomicity.
    """
    names = list(validate_names(output.name + suffix for suffix in payloads))
    with pin_output(output.parent) as root:
        with tempfile.TemporaryDirectory(prefix='aimusic-groove-') as tmp:
            staging = Path(tmp)
            for name, data in zip(names, payloads.values()):
                (staging / name).write_bytes(data)
            root.publish(staging, names, replace=True, sources=(source,))
