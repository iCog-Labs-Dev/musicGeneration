"""CLI entry points for the audio quarantine (lazy-imported from ``aimusic.app.cli``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aimusic.audio import require_audio_extra
from aimusic.audio.config import AudioConfig, load_audio_config, resolve_soundfont
from aimusic.audio.from_score import AudioPackageContext, ValidationReport, from_score, validate_package
from aimusic.audio.from_midi import ImportedMidiContext, from_midi


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
    output_dir: Path | str | None = None,
    soundfont: Path | str | None = None,
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
    cfg = _render_config(config, soundfont)
    backend = resolve_backend()
    out_dir = Path(output_dir) if output_dir is not None else Path(config.paths.output_dir) / report.run_id
    if backend == "m2a":
        audio_out = render_audio_from_package(
            ctx.package,
            output_dir=out_dir,
            config=cfg,
        )
    else:
        audio_out = _run_native(ctx, cfg, out_dir)
    _print_audio_result(audio_out, backend)
    return 0


def render_midi(
    midi_path: Path | str,
    *,
    profile: Path | str | None = None,
    validate_only: bool = True,
    output_dir: Path | str | None = None,
    soundfont: Path | str | None = None,
) -> int:
    """Validate imported event evidence; opt in to safe deterministic rendering."""
    require_audio_extra()
    config = load_audio_config(profile)
    ctx = from_midi(midi_path)
    doc = ctx.analysis_document()
    print('Imported MIDI validation OK')
    print(f'  provenance: {ctx.provenance}')
    print('  planner reconciliation: not applicable (no planner/Score metadata)')
    print(f'  MIDI type: {ctx.events.type}; tracks: {len(doc["tracks"])}; notes: {doc["total_notes"]}')
    print(f'  source_sha256: {ctx.source_sha256}')
    print(f'  render.backend: {config.render.backend}')
    print('  audio pitch accuracy: unmeasured; intended lattice: unknown')
    if validate_only:
        return 0

    from aimusic.audio.bridge import resolve_backend

    if resolve_backend() != 'native':
        raise ValueError('render-midi requires AIMUSIC_AUDIO_BACKEND=native; imported context is not supported by m2a')
    cfg = _render_config(config, soundfont)
    out_dir = Path(output_dir) if output_dir is not None else Path(config.paths.output_dir) / ctx.run_id
    audio_out = _run_native(ctx, cfg, out_dir)
    _print_audio_result(audio_out, 'native')
    return 0


def _render_config(config: AudioConfig, soundfont: Path | str | None) -> dict[str, Any]:
    cfg = config.to_dict()
    asset = resolve_soundfont(config, explicit=soundfont)
    cfg['paths']['soundfont'] = str(asset) if asset is not None else None
    return cfg


def _run_native(ctx: AudioPackageContext | ImportedMidiContext, cfg: dict[str, Any], out_dir: Path) -> Path:
    from aimusic.audio.orchestrator import Orchestrator

    cfg['paths']['output_dir'] = str(out_dir)
    return Orchestrator(config=cfg, cache_dir=Path(cfg['paths']['cache_dir'])).run(ctx, output_dir=out_dir)


def _print_audio_result(audio_out: Path, backend: str) -> None:
    stems = audio_out / "stems"
    mix = stems / "rough_mix.wav"
    print(f"Audio render OK (backend={backend})")
    print(f"  output: {audio_out}")
    if mix.is_file():
        print(f"  rough_mix: {mix}")
