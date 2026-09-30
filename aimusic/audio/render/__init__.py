"""Deterministic stem rendering (CI-safe simple backend)."""

from __future__ import annotations

from aimusic.audio.render.simple_r import render_stem
from aimusic.audio.render.stems import render_role_stems

__all__ = ["render_role_stems", "render_stem"]
