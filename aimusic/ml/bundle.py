from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import Vocabularies, build_default_vocabularies
from aimusic.ml.vocab_export import export_vocabularies_json, vocabularies_from_export
from aimusic.scoring.priors import (
    NeuralPriorManifest,
    load_neural_prior_manifest,
    save_neural_prior_manifest,
)


@dataclass(frozen=True)
class PriorBundle:
    """Loaded prior artifact bundle."""

    manifest: NeuralPriorManifest
    vocabularies_export: dict[str, Any]
    style_config: StyleConfig
    vocabularies: Vocabularies
    count_state: dict[str, Any]
    bundle_dir: Path


def _counts_checkpoint_dir(bundle_dir: Path) -> Path:
    return bundle_dir / "counts"


def save_vocabularies_json(path: Path, vocabularies: Vocabularies, style_config: StyleConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(export_vocabularies_json(vocabularies, style_config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_vocabularies_json(path: Path) -> tuple[dict[str, Any], StyleConfig, Vocabularies]:
    data = json.loads(path.read_text(encoding="utf-8"))
    export, style_config = vocabularies_from_export(data)
    return export, style_config, build_default_vocabularies(style_config)


def save_count_state(checkpoint_dir: Path, count_state: Mapping[str, Any]) -> None:
    try:
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise ImportError("Saving count checkpoints requires the optional [ml] extra.") from exc

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    serializable = {name: jnp.asarray(array) for name, array in count_state.items()}
    checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=os.name != "nt")
    checkpointer.save(checkpoint_dir, serializable, force=True)


def load_count_state(checkpoint_dir: Path) -> dict[str, Any]:
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise ImportError("Loading count checkpoints requires the optional [ml] extra.") from exc

    checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=os.name != "nt")
    restored = checkpointer.restore(checkpoint_dir)
    if not isinstance(restored, dict):
        raise ValueError("Count checkpoint must restore to a dict of arrays.")
    return dict(restored)


def save_prior_bundle(
    bundle_dir: str | Path,
    *,
    manifest: NeuralPriorManifest,
    vocabularies: Vocabularies,
    style_config: StyleConfig,
    count_state: Mapping[str, Any],
) -> Path:
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)

    vocab_path = root / "vocabularies.json"
    manifest_path = root / "manifest.json"
    counts_dir = _counts_checkpoint_dir(root)

    save_vocabularies_json(vocab_path, vocabularies, style_config)
    save_count_state(counts_dir, count_state)

    manifest_to_save = NeuralPriorManifest(
        manifest_version=manifest.manifest_version,
        model_family=manifest.model_family,
        model_version=manifest.model_version,
        factorization_mode=manifest.factorization_mode,
        token_streams=manifest.token_streams,
        checkpoint_path="counts",
        tokenizer_path="vocabularies.json",
        supports_batch_scoring=manifest.supports_batch_scoring,
        expected_edo=manifest.expected_edo,
        metadata=manifest.metadata,
    )
    save_neural_prior_manifest(manifest_to_save, str(manifest_path))
    return root


def load_prior_bundle(bundle_dir: str | Path) -> PriorBundle:
    root = Path(bundle_dir)
    manifest = load_neural_prior_manifest(str(root / "manifest.json"))
    export, style_config, vocabularies = load_vocabularies_json(root / "vocabularies.json")
    counts_rel = manifest.checkpoint_path or "counts"
    count_state = load_count_state(root / counts_rel)
    return PriorBundle(
        manifest=manifest,
        vocabularies_export=export,
        style_config=style_config,
        vocabularies=vocabularies,
        count_state=count_state,
        bundle_dir=root,
    )
