"""Role -> renderer routing (CI-safe simple backend only in this PR)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aimusic.audio.render import simple_r

RendererFn = Callable[..., Any]

_REGISTRY: dict[str, str] = {
    "drums": "simple",
    "bass": "simple",
    "comp": "simple",
    "lead": "simple",
    "pads": "simple",
    "ornament": "simple",
    "piano": "simple",
}

_BACKENDS: dict[str, RendererFn] = {
    "simple": simple_r.render_stem,
}


def get_renderer_name(role: str, *, config: dict[str, Any] | None = None) -> str:
    """Return the configured renderer backend name for an instrument role."""
    if config:
        overrides = config.get("render", {}).get("role_map", {})
        if role in overrides:
            name = str(overrides[role])
            if name == "fluidsynth":
                allow = bool(config.get("render", {}).get("allow_simple_fallback", True))
                if allow:
                    return "simple"
                raise NotImplementedError(
                    "fluidsynth backend is deferred; use render.backend: simple "
                    "or allow_simple_fallback: true"
                )
            return name
        backend = str(config.get("render", {}).get("backend", "simple"))
        if backend == "fluidsynth":
            if bool(config.get("render", {}).get("allow_simple_fallback", True)):
                return "simple"
            raise NotImplementedError(
                "fluidsynth backend is deferred to a later PR; "
                "set render.backend: simple for CI-safe stems"
            )
        return backend
    return _REGISTRY.get(role, "simple")


def get_renderer(role: str, *, config: dict[str, Any] | None = None) -> RendererFn:
    """Return the callable renderer for a role."""
    name = get_renderer_name(role, config=config)
    if name not in _BACKENDS:
        raise KeyError(f"renderer backend not registered: {name}")
    return _BACKENDS[name]


def list_roles() -> list[str]:
    return sorted(_REGISTRY.keys())
