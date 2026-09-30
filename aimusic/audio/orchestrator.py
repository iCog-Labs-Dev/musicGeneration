"""Stage DAG, content-hash cache, and budget guard (M1 deterministic spine).

Stages communicate only via files + JSON sidecars — never in-memory objects
across the DAG boundary. Restyle / scoring / mixmaster remain gated off.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aimusic.audio.manifest import content_hash_files


STAGE_ORDER = (
    "analysis",
    "expressivization",
    "render",
    "prompts",
    "restyle",
    "scoring",
    "mixmaster",
)


@dataclass
class BudgetGuard:
    max_endpoint_calls: int = 40
    max_usd_estimate: float = 10.0
    calls_used: int = 0
    usd_used: float = 0.0

    def can_spend(self, *, calls: int = 1, usd: float = 0.0) -> bool:
        return (
            self.calls_used + calls <= self.max_endpoint_calls
            and self.usd_used + usd <= self.max_usd_estimate
        )

    def record(self, *, calls: int = 1, usd: float = 0.0) -> None:
        if not self.can_spend(calls=calls, usd=usd):
            raise RuntimeError(
                f"budget exceeded: calls={self.calls_used}/{self.max_endpoint_calls}, "
                f"usd={self.usd_used:.2f}/{self.max_usd_estimate:.2f}"
            )
        self.calls_used += calls
        self.usd_used += usd


@dataclass
class Orchestrator:
    """Pipeline runner with content-addressed cache under cache_dir."""

    config: dict[str, Any]
    cache_dir: Path
    budget: BudgetGuard = field(default_factory=BudgetGuard)
    cache_hits: dict[str, int] = field(default_factory=dict)
    cache_misses: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        budget_cfg = self.config.get("budget", {})
        self.budget = BudgetGuard(
            max_endpoint_calls=int(budget_cfg.get("max_endpoint_calls", 40)),
            max_usd_estimate=float(budget_cfg.get("max_usd_estimate", 10.0)),
        )

    def cache_key(self, stage: str, input_paths: list[Path | str], subset: dict[str, Any]) -> str:
        return content_hash_files(
            [Path(p) for p in input_paths],
            stage,
            json.dumps(subset, sort_keys=True),
            str(self.config.get("code_version", "0.1.0")),
        )

    def cached_path(self, stage: str, key: str, suffix: str) -> Path:
        return self.cache_dir / stage / f"{key[:16]}{suffix}"

    def _load_groove_spec(self) -> dict[str, Any]:
        groove_cfg = self.config.get("groove", {})
        spec_path = groove_cfg.get("spec_path") or groove_cfg.get("spec_file")
        if spec_path:
            p = Path(spec_path)
            if not p.is_file():
                repo_root = Path(__file__).resolve().parents[2]
                candidate = repo_root / spec_path
                if candidate.is_file():
                    p = candidate
            with p.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"groove spec must be a mapping: {p}")
            return data
        if "spec" in groove_cfg and isinstance(groove_cfg["spec"], dict):
            return groove_cfg["spec"]
        return {
            "swing": {
                "ratio": 0.58,
                "subdivision": 8,
                "applies_to": ["drums", "bass", "comp"],
            },
            "velocity": {
                "accent_map_16": [1.0, 0.7, 0.85, 0.7] * 4,
            },
        }

    def run(self, midi_path: Path | str, *, output_dir: Path | str | None = None) -> Path:
        """Execute analysis → expressivization → render (M1 spine)."""
        midi_path = Path(midi_path)
        if not midi_path.is_file():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        out = (
            Path(output_dir)
            if output_dir
            else Path(self.config.get("paths", {}).get("output_dir", "outputs/audio"))
        )
        out.mkdir(parents=True, exist_ok=True)
        stages = self.config.get("stages", {})
        code_version = str(self.config.get("code_version", "0.1.0"))

        from aimusic.audio.analysis.from_midi import analyze_midi
        from aimusic.audio.groove.apply import apply_groove
        from aimusic.audio.render.stems import render_role_stems

        analysis_doc: dict[str, Any] | None = None
        expressive_midi = midi_path

        if stages.get("analysis", True):
            subset: dict[str, Any] = {"code_version": code_version}
            key = self.cache_key("analysis", [midi_path], subset)
            cached = self.cached_path("analysis", key, ".analysis.json")
            dest = out / f"{midi_path.stem}.analysis.json"
            if cached.is_file():
                self.cache_hits["analysis"] = self.cache_hits.get("analysis", 0) + 1
                shutil.copy2(cached, dest)
                analysis_doc = json.loads(dest.read_text(encoding="utf-8"))
            else:
                self.cache_misses["analysis"] = self.cache_misses.get("analysis", 0) + 1
                cached.parent.mkdir(parents=True, exist_ok=True)
                analysis_doc = analyze_midi(
                    midi_path, config=self.config, output_path=cached
                )
                shutil.copy2(cached, dest)
                if Path(str(cached) + ".json").is_file():
                    shutil.copy2(Path(str(cached) + ".json"), Path(str(dest) + ".json"))

        if stages.get("expressivization", True):
            groove_spec = self._load_groove_spec()
            light = bool(self.config.get("groove", {}).get("light_cleanup", False))
            subset = {
                "groove": groove_spec,
                "light_cleanup": light,
                "code_version": code_version,
            }
            key = self.cache_key("expressivization", [midi_path], subset)
            cached = self.cached_path("expressivization", key, ".expressive.mid")
            dest = out / f"{midi_path.stem}.expressive.mid"
            if cached.is_file():
                self.cache_hits["expressivization"] = (
                    self.cache_hits.get("expressivization", 0) + 1
                )
                shutil.copy2(cached, dest)
            else:
                self.cache_misses["expressivization"] = (
                    self.cache_misses.get("expressivization", 0) + 1
                )
                cached.parent.mkdir(parents=True, exist_ok=True)
                apply_groove(
                    midi_path,
                    groove_spec,
                    analysis=analysis_doc,
                    output_path=cached,
                    config=self.config,
                    light_cleanup=light,
                )
                shutil.copy2(cached, dest)
                diff_src = Path(str(cached) + ".diff.json")
                if diff_src.is_file():
                    shutil.copy2(diff_src, Path(str(dest) + ".diff.json"))
            expressive_midi = dest

        if stages.get("render", True):
            render_cfg = self.config.get("render", {})
            render_subset = {
                "sample_rate": render_cfg.get("sample_rate"),
                "peak_dbfs": render_cfg.get("peak_dbfs", render_cfg.get("headroom_dbfs")),
                "backend": render_cfg.get("backend", "simple"),
                "soundfont": self.config.get("paths", {}).get("soundfont"),
                "code_version": code_version,
            }
            key = self.cache_key("render", [expressive_midi], render_subset)
            cached_dir = self.cache_dir / "render" / key[:16]
            stems_dir = out / "stems"
            marker = cached_dir / "_complete"
            if marker.is_file():
                self.cache_hits["render"] = self.cache_hits.get("render", 0) + 1
                if stems_dir.exists():
                    shutil.rmtree(stems_dir)
                shutil.copytree(cached_dir, stems_dir, ignore=lambda *_: {"_complete"})
            else:
                self.cache_misses["render"] = self.cache_misses.get("render", 0) + 1
                if cached_dir.exists():
                    shutil.rmtree(cached_dir)
                cached_dir.mkdir(parents=True, exist_ok=True)
                render_role_stems(
                    expressive_midi,
                    output_dir=cached_dir,
                    config=self.config,
                    analysis=analysis_doc,
                )
                marker.write_text("ok", encoding="utf-8")
                if stems_dir.exists():
                    shutil.rmtree(stems_dir)
                shutil.copytree(cached_dir, stems_dir, ignore=lambda *_: {"_complete"})

        return out


def load_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config (default.yaml or a profile)."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow-recursive merge for profile overrides."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def audio_config_to_dict(config: Any) -> dict[str, Any]:
    """Build orchestrator dict config from frozen AudioConfig (+ defaults)."""
    from aimusic.audio.config import AudioConfig

    if not isinstance(config, AudioConfig):
        raise TypeError("expected AudioConfig")
    raw = load_config(config.source_path) if config.source_path.is_file() else {}
    # Ensure code_version for cache keys
    raw.setdefault("code_version", "0.1.0")
    return raw
