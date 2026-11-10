"""Bridge from RenderPackage to ``m2a`` or native ``aimusic.audio`` orchestrator."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

from aimusic.audio import require_audio_extra
from aimusic.audio.config import parse_yaml
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
    data = parse_yaml(path.read_text(encoding="utf-8"), path) or {}
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
    """Native is the product default; the bridge requires explicit selection."""
    forced = os.environ.get("AIMUSIC_AUDIO_BACKEND", "").strip().lower()
    if forced not in {"", "native", "m2a"}:
        raise ValueError(f"unknown audio backend: {forced}")
    return forced or "native"


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
    config: dict[str, Any] | None = None,
) -> Path:
    """Run the audio spine on a RenderPackage; return the audio output directory."""
    require_audio_extra()
    if not isinstance(package, RenderPackage):
        package = load_render_package(package)

    cfg = _default_pipeline_config()
    if config is not None:
        cfg = deepcopy(config)
    else:
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
    backend = resolve_backend()
    if backend == "native":
        from aimusic.audio.from_score import from_score
        from aimusic.audio.orchestrator import Orchestrator

        orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
        return orch.run(from_score(package.root), output_dir=out)

    if backend == "m2a":
        if not m2a_available():
            raise ImportError(
                "AIMUSIC_AUDIO_BACKEND=m2a requires optional bridge deps. "
                "Install with: pip install -e '.[audio-bridge]'"
            )
        return _run_m2a(package, cfg, out)

    raise RuntimeError(f"unknown audio backend: {backend}")


def _run_m2a(package: RenderPackage, cfg: dict[str, Any], out: Path) -> Path:
    """Keep the optional backend away from sources and the publication tree."""
    from aimusic.audio.inputs import InputSnapshot
    from aimusic.audio.publication import pin_output
    from m2a.orchestrator import Orchestrator as M2AOrchestrator

    if out.resolve() == package.root.resolve():
        raise ValueError(f"Audio output aliases the source package: {out}")
    with ExitStack() as stack:
        root = stack.enter_context(pin_output(out))
        root.check_new_files(["score.mid", "structure.json"])
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".m2a-run-")))
        inputs = InputSnapshot.capture(package.root, scratch / "inputs", package=True)
        staging = scratch / "output"
        staging.mkdir()
        for name in ("score.mid", "structure.json"):
            shutil.copyfile(scratch / "inputs" / name, staging / name)
        effective = deepcopy(cfg)
        effective.setdefault("paths", {})["output_dir"] = str(staging)
        orch = M2AOrchestrator(config=effective, cache_dir=Path(cfg["paths"]["cache_dir"]))
        orch.run(scratch / "inputs/score.mid", output_dir=staging)
        inputs.verify()
        # Discover the complete backend bundle before publishing any artifacts.
        directories: dict[Path, list[str]] = {staging: []}
        for path in sorted(staging.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"m2a output must not contain symlinks: {path}")
            if path.is_dir():
                directories[path] = []
            elif path.is_file():
                directories[path.parent].append(path.name)
            else:
                raise ValueError(f"m2a output must be regular files: {path}")
        publications = []
        for directory, names in directories.items():
            root.verify()
            target = root if directory == staging else stack.enter_context(
                pin_output(out / directory.relative_to(staging)))
            target.check_new_files(names)
            publications.append((target, directory, names))
        # All directory transactions stay open until the whole bundle succeeds.
        with ExitStack() as publishing:
            for target, directory, names in publications:
                root.verify()
                publishing.enter_context(target.publication(directory, names, replace=False))
            inputs.verify()
            for target, _, _ in publications:
                target.verify()
    return out
