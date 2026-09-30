"""CLI entry points for the audio quarantine (lazy-imported from ``aimusic.app.cli``)."""

from __future__ import annotations

import os
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
    """Validate a RenderPackage; optionally run M1 deterministic spine."""
    require_audio_extra()
    config = load_audio_config(profile)
    ctx = from_score(package_root)
    report = validate_package(ctx)
    print(_format_validation_report(report, config), end="")
    if validate_only:
        return 0

    from aimusic.audio.bridge import render_audio_from_package, resolve_backend
    from aimusic.audio.orchestrator import audio_config_to_dict

    backend = resolve_backend()
    out_dir = Path(config.paths.output_dir) / report.run_id
    if backend == "m2a":
        audio_out = render_audio_from_package(
            ctx.package,
            output_dir=out_dir,
            profile_path=config.source_path,
            config_path=None,
        )
    else:
        # Force native path explicitly for deterministic product default.
        os.environ.setdefault("AIMUSIC_AUDIO_BACKEND", "native")
        cfg = audio_config_to_dict(config)
        cfg.setdefault("paths", {})["cache_dir"] = config.paths.cache_dir
        cfg.setdefault("paths", {})["output_dir"] = str(out_dir)
        if config.paths.soundfont is not None:
            cfg.setdefault("paths", {})["soundfont"] = config.paths.soundfont
        from aimusic.audio.orchestrator import Orchestrator

        orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
        audio_out = orch.run(ctx.package.midi_path, output_dir=out_dir)

    stems = audio_out / "stems"
    mix = stems / "rough_mix.wav"
    print(f"Audio render OK (backend={backend})")
    print(f"  output: {audio_out}")
    if mix.is_file():
        print(f"  rough_mix: {mix}")
    return 0
