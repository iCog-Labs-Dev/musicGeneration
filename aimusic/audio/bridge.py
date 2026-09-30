"""Bridge from RenderPackage to ``m2a`` or native ``aimusic.audio`` orchestrator."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from aimusic.audio import require_audio_extra
from aimusic.core.render_package import RenderPackage, load_render_package


def _default_pipeline_config() -> dict[str, Any]:
    return {
        "code_version": "0.1.0",
        "paths": {
            "cache_dir": ".cache/aimusic_audio",
            "output_dir": "outputs/audio",
            "soundfont": None,
            "reference_corpus": [],
        },
        "stages": {
            "analysis": True,
            "expressivization": True,
            "render": True,
            "prompts": False,
            "restyle": False,
            "scoring": False,
            "mixmaster": False,
        },
        "render": {
            "backend": "simple",
            "sample_rate": 48000,
            "peak_dbfs": -12.0,
            "allow_simple_fallback": True,
        },
        "budget": {"max_endpoint_calls": 40, "max_usd_estimate": 10.0},
        "clearml": {"enabled": False},
        "groove": {"light_cleanup": False},
    }


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    import yaml

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_backend() -> str:
    """Return ``native`` or ``m2a`` based on env and import availability."""
    forced = os.environ.get("AIMUSIC_AUDIO_BACKEND", "").strip().lower()
    if forced in {"native", "m2a"}:
        return forced
    try:
        import importlib.util

        spec = importlib.util.find_spec("aimusic.audio.orchestrator")
        if spec is not None:
            return "native"
    except ModuleNotFoundError:
        pass
    return "m2a"


def m2a_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("m2a") is not None
    except ModuleNotFoundError:
        return False


def render_audio_from_package(
    package: RenderPackage | Path | str,
    *,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    profile_path: Path | str | None = None,
    enable_restyle: bool = False,
) -> Path:
    """Run the audio spine on a RenderPackage; return the audio output directory."""
    require_audio_extra()
    if not isinstance(package, RenderPackage):
        package = load_render_package(package)

    cfg = _default_pipeline_config()
    cfg = _merge(cfg, _load_yaml_config(Path(config_path) if config_path else None))
    cfg = _merge(cfg, _load_yaml_config(Path(profile_path) if profile_path else None))
    if enable_restyle:
        cfg.setdefault("stages", {})["prompts"] = True
        cfg.setdefault("stages", {})["restyle"] = True
        cfg.setdefault("stages", {})["scoring"] = True

    out = (
        Path(output_dir)
        if output_dir
        else Path(cfg.get("paths", {}).get("output_dir", "outputs/audio"))
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package.midi_path, out / "score.mid")
    shutil.copy2(package.structure_path, out / "structure.json")

    backend = resolve_backend()
    if backend == "native":
        try:
            from aimusic.audio.orchestrator import Orchestrator

            orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            return orch.run(package.midi_path, output_dir=out)
        except Exception:
            if not m2a_available():
                raise
            backend = "m2a"

    if backend == "m2a":
        if not m2a_available():
            raise ImportError(
                "AIMUSIC_AUDIO_BACKEND=m2a requires optional bridge deps. "
                "Install with: pip install -e '.[audio-bridge]'"
            )
        from m2a.orchestrator import Orchestrator as M2AOrchestrator

        orch = M2AOrchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
        return orch.run(package.midi_path, output_dir=out)

    raise RuntimeError(f"unknown audio backend: {backend}")
