"""CLI entry points for the audio quarantine (lazy-imported from ``aimusic.app.cli``)."""

from __future__ import annotations

import sys
from pathlib import Path

from aimusic.audio import require_audio_extra
from aimusic.audio.config import AudioConfig, load_audio_config
from aimusic.audio.from_score import ValidationReport, from_score, validate_package


def _format_validation_report(report: ValidationReport, config: AudioConfig) -> str:
    enabled_stages = [
        name
        for name, enabled in (
            ("analysis", config.stages.analysis),
            ("expressivization", config.stages.expressivization),
            ("render", config.stages.render),
            ("prompts", config.stages.prompts),
            ("restyle", config.stages.restyle),
            ("scoring", config.stages.scoring),
            ("mixmaster", config.stages.mixmaster),
        )
        if enabled
    ]
    microtonal = ", ".join(report.microtonal_tracks) if report.microtonal_tracks else "(none)"
    tracks = ", ".join(report.track_names) if report.track_names else "(none)"
    return (
        f"RenderPackage validation OK\n"
        f"  run_id: {report.run_id}\n"
        f"  schema: {report.schema}\n"
        f"  provenance: {report.provenance}\n"
        f"  edo: {report.edo}\n"
        f"  tracks ({report.track_count}): {tracks}\n"
        f"  microtonal: {microtonal}\n"
        f"  content_hash: {report.content_hash}\n"
        f"  config: {config.source_path}\n"
        f"  render.backend: {config.render.backend}\n"
        f"  stages enabled: {', '.join(enabled_stages) or '(none)'}\n"
    )


def render_audio(
    package_root: Path | str,
    *,
    profile: Path | str | None = None,
    validate_only: bool = True,
) -> int:
    """Validate a RenderPackage; full render deferred to M1 PR3."""
    require_audio_extra()
    config = load_audio_config(profile)
    ctx = from_score(package_root)
    report = validate_package(ctx)
    print(_format_validation_report(report, config), end="")
    if validate_only:
        return 0
    print(
        "Full render not implemented yet (M1 PR3). "
        "Re-run with --validate-only (default).",
        file=sys.stderr,
    )
    return 1
