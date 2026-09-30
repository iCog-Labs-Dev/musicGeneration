"""Load audio pipeline runtime config (``config/audio.default.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimusic.audio import require_audio_extra

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "audio.default.yaml"


@dataclass(frozen=True)
class AudioStagesConfig:
    analysis: bool
    expressivization: bool
    render: bool
    prompts: bool
    restyle: bool
    scoring: bool
    mixmaster: bool


@dataclass(frozen=True)
class AudioRenderConfig:
    backend: str
    sample_rate: int
    allow_simple_fallback: bool


@dataclass(frozen=True)
class AudioPathsConfig:
    cache_dir: str
    output_dir: str
    soundfont: str | None


@dataclass(frozen=True)
class AudioConfig:
    version: int
    mode: str
    render: AudioRenderConfig
    stages: AudioStagesConfig
    paths: AudioPathsConfig
    source_path: Path


def _require_mapping(label: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def load_audio_config(profile: Path | str | None = None) -> AudioConfig:
    """Load YAML audio config; default is ``config/audio.default.yaml``."""
    require_audio_extra()
    import yaml

    config_path = Path(profile) if profile is not None else _DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Audio config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Audio config root must be a mapping: {config_path}")

    render = _require_mapping("render", raw.get("render", {}))
    stages = _require_mapping("stages", raw.get("stages", {}))
    paths = _require_mapping("paths", raw.get("paths", {}))

    return AudioConfig(
        version=int(raw.get("version", 1)),
        mode=str(raw.get("mode", "corpus")),
        render=AudioRenderConfig(
            backend=str(render.get("backend", "simple")),
            sample_rate=int(render.get("sample_rate", 48000)),
            allow_simple_fallback=bool(render.get("allow_simple_fallback", True)),
        ),
        stages=AudioStagesConfig(
            analysis=bool(stages.get("analysis", True)),
            expressivization=bool(stages.get("expressivization", True)),
            render=bool(stages.get("render", True)),
            prompts=bool(stages.get("prompts", False)),
            restyle=bool(stages.get("restyle", False)),
            scoring=bool(stages.get("scoring", False)),
            mixmaster=bool(stages.get("mixmaster", False)),
        ),
        paths=AudioPathsConfig(
            cache_dir=str(paths.get("cache_dir", ".cache/aimusic_audio")),
            output_dir=str(paths.get("output_dir", "outputs/audio")),
            soundfont=paths.get("soundfont"),
        ),
        source_path=config_path,
    )
