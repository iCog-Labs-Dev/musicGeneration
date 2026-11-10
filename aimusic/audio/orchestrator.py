"""Stage DAG, content-hash cache, and budget guard (M1 deterministic spine).

Stages communicate only via files + JSON sidecars — never in-memory objects
across the DAG boundary. Restyle / scoring / mixmaster remain gated off.
"""

from __future__ import annotations

import json
import shutil
import platform
from importlib import metadata
import tempfile
from contextlib import ExitStack
from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aimusic.audio.config import parse_yaml
from aimusic.audio.from_score import AudioPackageContext, ValidationReport, from_score, validate_package
from aimusic.audio.from_midi import ImportedMidiContext, from_midi
from aimusic.audio.manifest import content_hash_files
from aimusic.audio.inputs import InputSnapshot
from aimusic.audio.cache import MANIFEST, cached_stage, fingerprint, restored_entry
from aimusic.audio.render.registry import file_digest
from aimusic.audio.publication import pin_output
from aimusic.core.render_package import ContractViolation


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

    @staticmethod
    def _copy_stage_artifacts(
        cached: Path, dest: Path, *sidecar_suffixes: str
    ) -> None:
        """Exclusively publish the primary file and present sidecars as a bundle."""
        sources = {dest.name: cached}
        for suffix in sidecar_suffixes:
            source = Path(str(cached) + suffix)
            if source.is_symlink():
                raise ValueError('stage artifacts must not be symlinks')
            if source.is_file():
                sources[dest.name + suffix] = source
        with pin_output(dest.parent) as root:
            root.check_new_files(list(sources))
            with tempfile.TemporaryDirectory(prefix='.stage-') as tmp:
                staging = Path(tmp)
                for name, source in sources.items():
                    shutil.copyfile(source, staging / name)
                root.publish(staging, list(sources), replace=False)

    def _cached_stage(self, stage: str, identity: dict[str, Any], names: list[str],
                      produce: Callable[[Path], Any], *, verify_inputs: Callable[[], None]) -> Path:
        directory, hit = cached_stage(self.cache_dir / stage, identity, names, produce, verify_inputs=verify_inputs)
        verify_inputs()
        counts = self.cache_hits if hit else self.cache_misses
        counts[stage] = counts.get(stage, 0) + 1
        return directory

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
            data = parse_yaml(p.read_text(encoding="utf-8"), p) or {}
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

    def run(
        self, midi_path: Path | str | AudioPackageContext | ImportedMidiContext, *,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Execute analysis → expressivization → render (M1 spine)."""
        with ExitStack() as stack:
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.audio-run-')))
            return self._run_snapshot(midi_path, output_dir, scratch, stack)

    def _run_snapshot(self, midi_path: Path | str | AudioPackageContext | ImportedMidiContext, output_dir: Path | str | None,
                      scratch: Path, stack: ExitStack) -> Path:
        context: AudioPackageContext | ImportedMidiContext | None = None
        validation = None
        is_package = isinstance(midi_path, AudioPackageContext)
        source = (midi_path.package.root if isinstance(midi_path, AudioPackageContext) else
                  midi_path.midi_path if isinstance(midi_path, ImportedMidiContext) else Path(midi_path))
        inputs = InputSnapshot.capture(source, scratch / 'inputs', package=is_package)
        if isinstance(midi_path, AudioPackageContext):
            # Parse, validate and reconcile one private, verified package snapshot.
            context = from_score(scratch / 'inputs')
            if context.package.content_hash != midi_path.package.content_hash:
                raise ContractViolation(
                    f"package identity changed since admission: expected={midi_path.package.content_hash}, "
                    f"actual={context.package.content_hash}; reload the context"
                )
            validation = validate_package(context)
            midi_path = context.package.midi_path
        elif isinstance(midi_path, ImportedMidiContext):
            context = from_midi(inputs.files[0].path)
            if context.source_sha256 != midi_path.source_sha256:
                raise ValueError('imported MIDI identity changed since admission; reload the context')
            midi_path = context.midi_path
        else:
            midi_path = inputs.files[0].path
        inputs.verify()
        midi_path = Path(midi_path)
        if not midi_path.is_file():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        out = (
            Path(output_dir)
            if output_dir
            else Path(self.config.get("paths", {}).get("output_dir", "outputs/audio"))
        )
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems, stem_artifact_names
        stages = self.config.get('stages', {})
        names = []
        if context:
            names += [f'{midi_path.stem}.reconciliation.json', f'{midi_path.stem}.analysis.json']
            if isinstance(context, ImportedMidiContext):
                names += ['source.mid']
        elif stages.get('analysis', True):
            names += [f'{midi_path.stem}.analysis.json', f'{midi_path.stem}.analysis.json.json']
        if stages.get('expressivization', True):
            names += [f'{midi_path.stem}.expressive.mid{suffix}' for suffix in ('', '.diff.json', '.json')]
        if stages.get('render', True):
            names += ['render.json']
        root = stack.enter_context(pin_output(out))
        root.check_new_files(names)
        stem_root = None
        if stages.get('render', True):
            metadata = ({'tracks': list(validation.reconciliation.tracks)} if validation else
                        context.analysis_document() if context else None)
            logical = isolate_source_stems(MidiEvents.load(midi_path), analysis=metadata)
            stem_names = stem_artifact_names(logical) + ['render.json', MANIFEST]
            stem_root = stack.enter_context(pin_output(out / 'stems'))
            stem_root.check_new_files(stem_names)
        self._run_stages(midi_path, scratch, context, validation, inputs.verify)
        inputs.verify()
        if stem_root:
            stem_names = sorted(p.name for p in (scratch / 'stems').iterdir() if p.name != MANIFEST) + [MANIFEST]
            stem_root.check_new_files(stem_names)
        with root.publication(scratch, names, replace=False):
            if stem_root:
                stem_root.publish(scratch / 'stems', stem_names, replace=False)
        return out

    def _run_stages(self, midi_path: Path, out: Path, context: AudioPackageContext | ImportedMidiContext | None,
                    validation: ValidationReport | None, verify_inputs: Callable[[], None]) -> None:
        """Compute/restore privately; only run() may publish into the user tree."""
        stages = self.config.get("stages", {})
        code_version = str(self.config.get("code_version", "0.1.0"))

        from aimusic.audio.analysis.from_midi import analyze_midi
        from aimusic.audio.groove.apply import apply_groove
        from aimusic.audio.render.stems import (
            STEM_LAYOUT_VERSION, prepare_stem_render, render_role_stems,
        )

        code_identity = {str(p.relative_to(Path(__file__).parent)): file_digest(p)
                         for p in sorted(Path(__file__).parent.rglob('*.py'))}
        source_identity = {'midi_sha256': file_digest(midi_path),
                           **(context.verified_identity if context else {})}
        if isinstance(context, AudioPackageContext):
            source_identity['structure_sha256'] = file_digest(context.package.structure_path)
            source_identity['tuning_sha256'] = file_digest(context.package.tuning_path)
        analysis_doc: dict[str, Any] | None = None
        expressive_midi = midi_path

        if context is not None:
            analysis_doc = context.analysis_document()
            if isinstance(context, ImportedMidiContext):
                shutil.copyfile(midi_path, out / 'source.mid')
                (out / f"{midi_path.stem}.reconciliation.json").write_text(
                    json.dumps(context.reconciliation, indent=2, sort_keys=True), encoding='utf-8')
            if validation is not None:
                analysis_doc["tracks"] = list(validation.reconciliation.tracks)
                (out / f"{midi_path.stem}.reconciliation.json").write_text(
                    json.dumps(validation.reconciliation.to_dict(), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            (out / f"{midi_path.stem}.analysis.json").write_text(
                json.dumps(analysis_doc, indent=2, sort_keys=True), encoding="utf-8"
            )
        elif stages.get("analysis", True):
            identity: dict[str, Any] = {'source': source_identity, 'code': code_identity, 'runtime': _runtime_identity('analysis'),
                        'code_version': code_version}
            names = [f'{midi_path.stem}.analysis.json', f'{midi_path.stem}.analysis.json.json']
            identity['artifacts'] = names
            cached = self._cached_stage('analysis', identity, names,
                                        lambda directory: analyze_midi(midi_path, config=self.config,
                                                                       output_path=directory / names[0]),
                                        verify_inputs=verify_inputs)
            dest = out / f"{midi_path.stem}.analysis.json"
            with restored_entry(cached, fingerprint(identity), names) as verified:
                self._copy_stage_artifacts(verified / names[0], dest, '.json')
            analysis_doc = json.loads(dest.read_text(encoding='utf-8'))

        if stages.get("render", True):
            # Admit original channel evidence before any transform/cache lookup
            # can replace it. Expressive MIDI is independently checked below.
            prepare_stem_render(midi_path, config=self.config, analysis=analysis_doc,
                                score=context.score if isinstance(context, AudioPackageContext) else None)

        if stages.get("expressivization", True):
            groove_spec = self._load_groove_spec()
            light = bool(self.config.get("groove", {}).get("light_cleanup", False))
            identity = {'source': source_identity, 'groove': groove_spec, 'light_cleanup': light,
                        'analysis': _semantic_analysis(analysis_doc), 'code': code_identity,
                        'runtime': _runtime_identity('expressivization'), 'code_version': code_version}
            names = [f'{midi_path.stem}.expressive.mid{suffix}' for suffix in ('', '.diff.json', '.json')]
            identity['artifacts'] = names
            cached = self._cached_stage('expressivization', identity, names,
                lambda directory: apply_groove(midi_path, groove_spec, analysis=analysis_doc,
                                                output_path=directory / names[0],
                                                config=self.config, light_cleanup=light), verify_inputs=verify_inputs)
            dest = out / f"{midi_path.stem}.expressive.mid"
            with restored_entry(cached, fingerprint(identity), names) as verified:
                self._copy_stage_artifacts(verified / names[0], dest, '.diff.json', '.json')
            expressive_midi = dest

        if stages.get("render", True):
            plan = prepare_stem_render(expressive_midi, config=self.config, analysis=analysis_doc,
                                       score=context.score if isinstance(context, AudioPackageContext) else None)
            effective = deepcopy(plan.config)
            effective.pop('source_path', None)
            effective['paths'] = {k: v for k, v in effective.get('paths', {}).items()
                                  if k not in ('cache_dir', 'output_dir', 'soundfont')}
            for key in ('spec_path', 'spec_file'):
                effective.get('groove', {}).pop(key, None)
            if stages.get('expressivization', True):
                effective.setdefault('groove', {})['spec'] = groove_spec
            render_subset = {
                'stem_layout_version': STEM_LAYOUT_VERSION, 'source': source_identity,
                'expressive_midi_sha256': plan.source_sha256,
                'analysis': _semantic_analysis(analysis_doc), 'effective_config': effective,
                'application_backend': {'requested_backend': 'native', 'effective_backend': 'native',
                                        'fallback_reason': None},
                'renderers': [dict(choice.identity(), source_track_index=stem.source_track_index,
                                   role=stem.role, role_source=stem.role_source, tuning_evidence=evidence)
                              for stem, choice, evidence in zip(plan.stems, plan.renderers, plan.tuning_evidence)],
                'duration_sec': plan.duration_sec, 'release_tail_sec': plan.release_tail_sec,
                'code': code_identity, 'runtime': _runtime_identity('render'),
            }

            def verify_render_inputs() -> None:
                verify_inputs()
                plan.verify_inputs(expressive_midi)

            def produce(directory: Path) -> None:
                render_role_stems(expressive_midi, output_dir=directory, plan=plan)
                (directory / 'render.json').write_text(json.dumps(render_subset, indent=2, sort_keys=True), encoding='utf-8')

            with pin_output(out) as root:
                root.check_new_files(['render.json'])
                cached_dir = self._cached_stage('render', render_subset, plan.names + ['render.json'], produce,
                                                verify_inputs=verify_render_inputs)
                with restored_entry(cached_dir, fingerprint(render_subset), plan.names + ['render.json']) as verified:
                    with pin_output(out / 'stems') as stem_root:
                        stem_root.publish(verified, plan.names + ['render.json', MANIFEST], replace=False)
                    verify_render_inputs()
                    root.publish(verified, ['render.json'], replace=False)


# Versions are stage-specific: an unrelated validator change must not invalidate
# analysis or rendering when the expressive MIDI bytes remain identical.
_STAGE_DEPENDENCIES = {
    'analysis': ('mido', 'numpy', 'pretty_midi', 'six'),
    'expressivization': ('mido', 'pydantic', 'pydantic_core', 'annotated-types',
                        'typing_extensions', 'typing-inspection', 'PyYAML'),
    'render': ('mido', 'numpy', 'pretty_midi', 'six', 'soundfile', 'cffi', 'pycparser'),
}


def _runtime_identity(stage: str) -> dict[str, Any]:
    dependencies = {}
    for name in _STAGE_DEPENDENCIES[stage]:
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = 'not-installed'  # Optional/transitive across supported versions.
    return {'python': platform.python_version(), 'platform': platform.platform(), 'dependencies': dependencies}


def _semantic_analysis(analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Source paths are display provenance, not semantic content identity."""
    return {k: v for k, v in (analysis or {}).items() if k != 'source_midi'}


def load_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config (default.yaml or a profile)."""
    data = parse_yaml(Path(path).read_text(encoding="utf-8"), path) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Retain the orchestrator API while sharing profile merge semantics."""
    from aimusic.audio.config import merge_configs as merge

    return merge(base, override)


def audio_config_to_dict(config: Any) -> dict[str, Any]:
    """Build orchestrator dict config from frozen AudioConfig (+ defaults)."""
    from aimusic.audio.config import AudioConfig

    if not isinstance(config, AudioConfig):
        raise TypeError("expected AudioConfig")
    return config.to_dict()
