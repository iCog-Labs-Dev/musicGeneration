"""Optional MIDI→audio production pipeline (quarantined subpackage).

Install with::

    pip install -e ".[audio]"

Importing this package without heavy DSP deps is allowed in M0 (stub).
Calling :func:`require_audio_extra` before DSP entry points enforces the extra.
"""

from __future__ import annotations

__all__ = ["AudioExtraRequired", "require_audio_extra"]

__version__ = "0.1.0"


class AudioExtraRequired(ImportError):
    """Raised when ``aimusic.audio`` DSP entry points need optional deps."""


def require_audio_extra() -> None:
    """Assert that audio-stack packages are importable.

    Core symbolic code must never call this. Bridge / CLI audio subcommands call
    it before touching orchestrator or DSP modules.
    """
    missing: list[str] = []
    for mod in ("yaml",):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise AudioExtraRequired(
            "aimusic.audio requires optional dependencies. Install with: "
            'pip install -e ".[audio]". '
            f"Missing: {', '.join(missing)}"
        )
