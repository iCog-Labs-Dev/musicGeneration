"""Resource-backed audio profiles and immutable effective settings."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aimusic.audio import require_audio_extra

_PROFILES = {"preview", "listening"}


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
    source_path: Path | None
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Freeze a normalized snapshot, including dataclass replacement values."""
        raw = _thaw(self.raw)
        raw.update(version=self.version, mode=self.mode)
        for name in ("render", "stages", "paths"):
            raw[name].update(asdict(getattr(self, name)))
        _validate_json_values(raw)
        object.__setattr__(self, "raw", _freeze(raw))

    def to_dict(self) -> dict[str, Any]:
        """Return detached effective settings without reading the source file."""
        return _thaw(self.raw)


def _validate_json_values(
    value: Any, path: str = "config", ancestors: frozenset[int] = frozenset(),
) -> None:
    """Reject YAML-only values before coercion or freezing can hide their types."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: numbers must be finite")
        return
    if not isinstance(value, (Mapping, list)):
        raise ValueError(f"{path}: unsupported config value type {type(value).__name__}")
    if id(value) in ancestors:
        raise ValueError(f"{path}: recursive aliases are not JSON-like config values")
    ancestors = ancestors | {id(value)}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: mapping keys must be strings, got {type(key).__name__}")
            _validate_json_values(item, f"{path}.{key}", ancestors)
    else:
        for index, item in enumerate(value):
            _validate_json_values(item, f"{path}[{index}]", ancestors)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge for profile overrides; neither input is modified."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def _require_mapping(label: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def normalize_routing(render: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize one config layer, rejecting contradictory alias assignments."""
    normalized = dict(render)
    routes: dict[str, dict[str, str]] = {}
    origins: dict[tuple[str, str], str] = {}
    for form in ("routing", "role_map"):
        for role, route in _require_mapping(f"render.{form}", render.get(form, {})).items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"render.{form} role must be a nonempty string")
            canonical = "comping" if role == "comp" else role
            origin = f"render.{form}.{role}"
            if form == "role_map" and isinstance(route, str):
                route = {"default": route, "microtonal": route}
            route = _require_mapping(origin, route)
            target = routes.setdefault(canonical, {})
            for tuning, backend in route.items():
                if tuning not in {"default", "microtonal"}:
                    raise ValueError(f"{origin}: unknown tuning class {tuning!r}")
                if not isinstance(backend, str) or not backend.strip():
                    raise ValueError(f"{origin}.{tuning} renderer must be a nonempty string")
                key = (canonical, tuning)
                if tuning in target and target[tuning] != backend:
                    raise ValueError(
                        f"routing conflict for {canonical}.{tuning}: "
                        f"{origins[key]}={target[tuning]!r} versus {origin}={backend!r}"
                    )
                target[tuning] = backend
                origins[key] = origin
    normalized.pop("role_map", None)
    normalized["routing"] = routes
    return normalized


def parse_yaml(text: str, source: Path | str) -> Any:
    """Translate parser failures at the boundary, retaining source context."""
    import yaml

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {source}: {exc}") from exc


def load_audio_config(profile: Path | str | None = None) -> AudioConfig:
    """Load packaged defaults, optionally overlaid by a named or explicit YAML profile.

    A Path always means a file; only bare string names select packaged profiles.
    source_path is provenance for explicit files, never the settings authority.
    """
    require_audio_extra()
    resources = files("aimusic.audio").joinpath("resources")
    resource_label = "aimusic.audio/resources"
    raw = _require_mapping(
        "Audio config root", parse_yaml(resources.joinpath("audio.default.yaml").read_text(encoding="utf-8"),
                                        f"{resource_label}/audio.default.yaml")
    )
    config_path = None
    source: Path | str = "empty profile override"
    if isinstance(profile, str) and profile in _PROFILES:
        text = resources.joinpath(f"{profile}.yaml").read_text(encoding="utf-8")
        source = f"{resource_label}/{profile}.yaml"
    elif profile is not None:
        config_path = Path(profile).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Audio config not found: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        source = config_path
    else:
        text = "{}"
    override = _require_mapping("Audio config root", parse_yaml(text, source))
    _validate_json_values(raw)
    _validate_json_values(override)
    for section in ("render", "stages", "paths"):
        if section in override:
            _require_mapping(section, override[section])
    # Normalize each layer first: profile overrides may replace shipped defaults,
    # but two contradictory aliases in the same file must never mask each other.
    raw["render"] = normalize_routing(raw.get("render", {}))
    override["render"] = normalize_routing(override.get("render", {}))
    raw = merge_configs(raw, override)
    raw.setdefault("code_version", "0.1.0")

    render = _require_mapping("render", raw.get("render", {}))
    stages = _require_mapping("stages", raw.get("stages", {}))
    paths = _require_mapping("paths", raw.get("paths", {}))
    soundfont = paths.get("soundfont")
    if soundfont is not None:
        if not isinstance(soundfont, str) or not soundfont.strip():
            raise ValueError("SoundFont paths.soundfont must be a nonempty file path or null")
        origin = config_path.parent if config_path is not None else Path.cwd()
        paths["soundfont"] = str(_absolute_path(soundfont, origin))

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
        raw=raw,
    )


def _absolute_path(value: Path | str, origin: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else origin / path).resolve()


def resolve_soundfont(config: AudioConfig | dict[str, Any], *, explicit: Path | str | None = None) -> Path | None:
    """Resolve CLI > config/profile > environment, without loading a native library.

    Config-relative assets were anchored at load time. CLI/environment relative
    paths use the current directory. None means no asset was requested; renderer
    policy decides whether one is required. An invalid selection never falls back.
    """
    candidates = (
        ("--soundfont", explicit),
        ("paths.soundfont", config.paths.soundfont if isinstance(config, AudioConfig)
         else config.get("paths", {}).get("soundfont")),
        ("AIMUSIC_SOUNDFONT", os.environ.get("AIMUSIC_SOUNDFONT")),
    )
    for source, value in candidates:
        if value is None:
            continue
        guidance = (
            f"SoundFont from {source} ({value!s}) must be a readable file path; "
            "set --soundfont, paths.soundfont, or AIMUSIC_SOUNDFONT to a local "
            "license-approved SoundFont. No lower-precedence asset was selected."
        )
        try:
            if not str(value).strip():
                raise ValueError("empty path")
            path = _absolute_path(value, Path.cwd())
            if not path.is_file():
                raise ValueError("not a regular file")
            with path.open("rb") as stream:
                stream.read(1)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(guidance) from exc
        return path
    return None
