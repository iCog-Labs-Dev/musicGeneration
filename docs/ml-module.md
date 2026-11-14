# ML Module — Recent Changes

This document collects and summarizes the recent commits that introduced and
iterated on the `aimusic.ml` package — the machine-learning training and
inference pipeline for neural transition priors that plugs into the existing
symbolic-music generator.

> **Scope:** commits that added the `aimusic/ml` package, its CLI, training
> pipeline, inference shim, dataset requirements documentation, optional
> ClearML monitoring, and CUDA-enabled JAX install with progress logs.

---

## 1. Commit timeline

| Commit | Date | Author | Subject |
|---|---|---|---|
| `90067c1` | 2026-07-03 | Arseniy Losev | feat(ml): add JAX count-prior training pipeline |
| `50834e7` | 2026-07-07 | Arseniy Losev | docs(ml): add training dataset requirements |
| `6982996` | 2026-07-07 | Arseniy Losev | feat(ml): add optional ClearML training monitoring |
| `d23aa97` | 2026-07-09 | Arseniy Losev | feat(ml): add training progress logs and CUDA-enabled JAX install |

The first commit introduces the module end-to-end (MIDI ingest → beat
annotation → JAX count tables → Orbax bundle → CLI integration). The
subsequent three iterate on it: dataset documentation, optional experiment
tracking, and operator ergonomics (timing logs, GPU install default).

---

## 2. What was added

### 2.1 New `aimusic/ml` package

The first commit (`90067c1`) introduces the entire module in one PR (22
files changed, +1550 / −1). Layout:

| File | Responsibility |
|---|---|
| `aimusic/ml/__init__.py` | Package marker, empty `__all__` |
| `aimusic/ml/midi_ingest.py` | Parse SMF into beat-quantized harmonic + drum activity (`MidiBeatGrid`, `BeatWindow`) |
| `aimusic/ml/annotate.py` | Convert `MidiBeatGrid` → `tuple[BeatState, ...]` using vocabulary-aware heuristics |
| `aimusic/ml/dataset.py` | Build transition examples, pack tokenized batches for JAX, expose `STREAM_FIELD_MAP` |
| `aimusic/ml/count_prior.py` | JAX bincount factorized transition tables, `CountPriorState`, JIT-compiled `CountPriorModel` |
| `aimusic/ml/train.py` | End-to-end entry point `train_prior_from_corpus` returning a `TrainResult` |
| `aimusic/ml/bundle.py` | Orbax-backed save/load of the prior bundle (`manifest.json` + `vocabularies.json` + `counts/`) |
| `aimusic/ml/vocab_export.py` | JSON snapshot of `Vocabularies` + `StyleConfig` for portable artifacts |
| `aimusic/ml/inference.py` | `load_trained_neural_prior(bundle_dir)` — turn a bundle into a `NeuralPrior` for the planning pipeline |
| `aimusic/ml/cli.py` | `python -m aimusic.ml.cli train …` entry point |
| `aimusic/ml/monitoring.py` | Optional ClearML experiment tracking (added in `6982996`) |

### 2.2 New tests

The same commit also adds the matching test suite:

- `tests/test_annotate.py` — chord / head / role / boundary / groove heuristics
- `tests/test_midi_ingest.py` — beat-grid parsing
- `tests/test_count_prior.py` — JAX bincount tables and JIT scoring
- `tests/test_ml_dataset.py` — transition example + JAX batch packing
- `tests/test_ml_train.py` — end-to-end training on a tiny fixture
- `tests/test_ml_bundle.py` — round-trip save/load of artifact bundles
- `tests/test_ml_integration.py` — bundle → `NeuralPrior` → batch scoring
- `tests/test_vocab_export.py` — JSON round-trip of vocabulary snapshot
- `tests/test_ml_monitoring.py` — ClearML monitor (added in `6982996`)
- `tests/conftest.py` — `requires_jax` marker for tests needing `[ml]`

### 2.3 Pipeline integration

- `aimusic/app/cli.py` — new `--prior-bundle` flag on `generate`. When set,
  the CLI lazily imports `aimusic.ml.inference.load_trained_neural_prior`,
  wraps the bundle as a `NeuralPrior`, and threads it into `run_method_a`.
- `.github/workflows/ci.yml` — new `ml-tests` job that installs
  `.[ml,dev]` and runs the ML test files separately from the core suite.
- `pyproject.toml` — new `ml` optional extra
  (`jax`, `jaxlib`, `flax`, `optax`, `orbax-checkpoint`) and an `ml` pytest
  marker. Core dependencies remain NumPy-only; ML is strictly optional.

### 2.4 Dataset requirements documentation (`50834e7`)

`docs/ml-training-dataset-requirements.md` (454 lines) is added to document:

- Corpus directory layout (recursive `**/*.mid` and `**/*.midi`)
- SMF Type 0/1 requirements, tempo and time-signature handling
- Beat quantization rules (`midi_tick // ticks_per_beat`)
- Channels 0–8/10–15 = harmonic, channel 9 = drum
- Annotation inclusion rules: meter in `allowed_meters`, ≥1 pitch class,
  vocabulary chord overlap
- The 8-stream token vocabulary (`meter`, `beat_position`, `boundary`,
  `key`, `chord`, `role`, `head`, `groove`)
- EDO handling and `expected_edo` recorded in the manifest
- Recommended corpus characteristics and content to exclude
- A 9-step validation checklist and a minimal two-beat fixture pattern
- Known v1 limitations (heuristic annotation, no sustain carry-over,
  no per-file error isolation, factorized independence, etc.)

### 2.5 Optional ClearML monitoring (`6982996`)

Adds `aimusic/ml/monitoring.py` and wires it through the CLI:

- `ClearMLConfig` — `enabled`, `project_name`, `task_name`, `tags`,
  `upload_artifacts`
- `TrainingMonitor` Protocol + `NullTrainingMonitor` no-op
- `ClearMLTrainingMonitor` logs:
  - Hyperparameters (`edo`, `alpha`, `model_version`, full `StyleConfig`)
  - Corpus metrics (`transition_count`, `midi_files_processed`,
    `transitions_per_midi_file`, zero-yield file count)
  - Per-stream count-table stats (vocab size, total transitions,
    unique `(prev, next)` pairs, sparsity)
  - A per-file corpus table (path, annotated_beats, transitions)
  - Artifact uploads: trained bundle directory, `manifest.json`,
    `vocabularies.json` (skippable via `--no-clearml-artifacts`)
- `pyproject.toml` adds a separate `clearml` extra (`clearml`) so the
  monitoring import is lazy and the core `[ml]` install stays lean.
- `cli.py` gains `--clearml`, `--clearml-project`, `--clearml-task-name`,
  `--clearml-tag`, `--no-clearml-artifacts` and a friendly install hint
  on `ImportError`.
- `docs/ml-training-dataset-requirements.md` §8 gains a ClearML
  usage block (install, example invocation, what gets logged).
- `tests/test_ml_monitoring.py` covers both the no-op and ClearML paths.

### 2.6 CUDA JAX + progress logs (`d23aa97`)

Operator-facing improvements so a fresh install "just works" on GPU and
the 4–5 minute corpus run no longer looks like a hang:

- `pyproject.toml` — `ml` extra switches from `jax` / `jaxlib` to
  `jax[cuda12]`, and adds `tqdm`.
- `aimusic/ml/cli.py` — `os.environ.setdefault("JAX_PLATFORMS", "cuda")`
  at import time to suppress the `libtpu.so` probe warning; new
  `_configure_logging` with `logging.basicConfig(level=INFO, force=True,
  stream=sys.stderr)`; JAX/absl loggers silenced; `--quiet` flag for
  test-friendly output.
- `aimusic/ml/dataset.py` — wraps the per-file loop in a `tqdm` progress
  bar (`desc="parse+annotate"`) and converts any single-file failure
  into a logged warning so one bad MIDI no longer aborts the run;
  emits a final `Loaded N transitions from M files in Xs` summary.
- `aimusic/ml/train.py` — per-stage `[stage i/5]` timings plus a
  final `TOTAL training time` line, all using `logger.info`.

The commit message documents the design choice: training remains
single-device because `mido` parsing dominates the wall time
(~51 ms/file Python vs. <1 s of GPU bincount on a single RTX 3090),
and multi-GPU sharding would save ~50 ms total. Verified locally:
**237/237 tests pass.**

---

## 3. Pipeline shape

```text
                    +------------------------------+
                    |   MIDI corpus directory      |
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    | parse_midi_beats()           |  mido → MidiBeatGrid
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    | annotate_beat_grid()         |  vocab-aware heuristics
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    | build_transition_examples()  |  (prev, next) BeatStates
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    | examples_to_tokenized()      |  8-stream token ids
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    | train_counts()               |  JAX bincount tables
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    | save_prior_bundle()          |  Orbax counts/ + manifest
                    +---------------+--------------+
                                    |
                                    v
                    +------------------------------+
                    |   aimusic/app/cli generate   |
                    |   --prior-bundle <dir>       |
                    +------------------------------+
```

History/future-hint lengths match Method A (`history_len=1`,
`future_hint_len=1`); factorization is
`PriorFactorization.FACTORIZED` — one independent
`vocab_size × vocab_size` table per stream.

The eight structural streams (from `aimusic/ml/dataset.py::STREAM_FIELD_MAP`)
are: `meter`, `beat_position`, `boundary`, `key`, `chord`, `role`,
`head`, `groove`.

---

## 4. CLI usage

### 4.1 Train

```bash
pip install -e ".[ml,dev]"

python -m aimusic.ml.cli train \
  --midi-dir ./corpus \
  --output ./artifacts/prior_v1 \
  --edo 12 \
  --alpha 1.0 \
  --model-version v1 \
  --meter 4/4 --meter 5/4 \
  --groove-family straight --groove-family syncopated
```

Optional flags: `--quiet` (silence per-stage logs, useful in tests),
`--clearml` (enable tracking), `--clearml-project`,
`--clearml-task-name`, `--clearml-tag` (repeatable),
`--no-clearml-artifacts`.

### 4.2 Generate using a trained bundle

```bash
pip install -e ".[ml]"

python -m aimusic.app.cli generate \
  --beats 8 \
  --prior-bundle ./artifacts/prior_v1
```

The CLI lazy-imports `aimusic.ml.inference.load_trained_neural_prior`,
wraps the bundle as a `NeuralPrior`, and passes it through
`run_method_a(run_config, prior=prior)`.

### 4.3 Programmatic API

```python
from aimusic.core.config import StyleConfig
from aimusic.ml.train import train_prior_from_corpus
from aimusic.ml.inference import load_trained_neural_prior

train_prior_from_corpus(
    "./corpus",
    "./artifacts/prior_v1",
    style_config=StyleConfig(
        allowed_meters=("4/4",),
        groove_families=("straight", "syncopated"),
    ),
    edo=12,
    alpha=1.0,
)

prior = load_trained_neural_prior("./artifacts/prior_v1")
score = prior.score_transition_batch(queries)
```

---

## 5. Bundle layout

A saved bundle is a self-contained directory:

```text
prior_v1/
├── manifest.json           # model family, edo, factorization, paths, metadata
├── vocabularies.json       # StyleConfig snapshot + 8 token tables
└── counts/                 # Orbax StandardCheckpointer of JAX count tables
```

`manifest.json` is created via
`aimusic.scoring.priors.save_neural_prior_manifest` and matches the
existing `NeuralPriorManifest` contract used by the planning pipeline.

### 5.1 Manifest versions and migration

- **Version 1 (`manifest_version: 1`)**:
  Default factorized token streams: 8 structural streams (`meter`, `beat_position`,
  `boundary`, `key`, `chord`, `role`, `head`, `groove`).
- **Version 2 (`manifest_version: 2`)**:
  Adds support for the optional `"section"` token stream in `token_streams`. When
  present, the prior wrapper provides `TokenizedPriorQuery.section_tokens` with
  section-level conditioning (`section_index`, `section_name`, `offset_in_section`,
  `section_length`, `is_section_start`, `target_tension`, `groove_family`,
  `allowed_meters`, `key_region`).

**Backward compatibility**: Manifests specifying `manifest_version: 1` or omitting
the field continue to load seamlessly. Bundles trained under v1 remain fully
operational, and unknown token streams continue to be rejected with `ValueError`.
Placeholder priors (`NullPrior`, placeholder `NeuralPrior`) do not consume
`section_tokens`, leaving legacy graph weights invariant.

---

## 6. Optional dependencies

`pyproject.toml` keeps the core install NumPy-only and gates ML behind
two extras:

```toml
[project.optional-dependencies]
ml = ["jax[cuda12]", "flax", "optax", "orbax-checkpoint", "tqdm"]
clearml = ["clearml"]
dev = ["pytest", "flake8", "mypy"]
```

Install recipes:

| Goal | Command |
|---|---|
| Train only | `pip install -e ".[ml]"` |
| Train + tests | `pip install -e ".[ml,dev]"` |
| Train + ClearML | `pip install -e ".[ml,clearml]"` |
| Everything | `pip install -e ".[ml,clearml,dev]"` |

CI runs two separate jobs: the core test suite without ML deps, and a
dedicated `ml-tests` job that installs `.[ml,dev]` and runs
`pytest` against the ML-only test files.

---

## 7. Test coverage

ML module changes added nine new test files / helpers:

| File | Covers |
|---|---|
| `tests/conftest.py` | `requires_jax` marker + `pytest_configure` |
| `tests/test_annotate.py` | chord / head / role / boundary / groove heuristics |
| `tests/test_midi_ingest.py` | `parse_midi_beats` beat grid, tempo, meter |
| `tests/test_count_prior.py` | JAX bincount tables, JIT scoring |
| `tests/test_ml_dataset.py` | `STREAM_FIELD_MAP`, batch packing |
| `tests/test_ml_train.py` | end-to-end training on a fixture |
| `tests/test_ml_bundle.py` | save/load round-trip |
| `tests/test_ml_integration.py` | bundle → `NeuralPrior` → batch scoring |
| `tests/test_vocab_export.py` | JSON round-trip of vocabulary snapshot |
| `tests/test_ml_monitoring.py` | `NullTrainingMonitor` + `ClearMLTrainingMonitor` |

The `d23aa97` commit message records the local verification:
**237/237 tests pass.**

---

## 8. Design notes carried over from the commits

- **Keep core NumPy-only.** The ML pipeline is optional and lives behind
  the `ml` extra; the planning/decode/render stages never import JAX.
- **Bundle is the artifact of record.** Counts live in `counts/`,
  vocabulary in `vocabularies.json`, model identity in `manifest.json`.
  This lets the generator load the prior without re-running training
  and without re-deriving vocabulary tables from `StyleConfig`.
- **MIDI corpus, not a derived dataset.** v1 does not accept
  pre-annotated JSON/CSV/NPZ — the trainer always walks a SMF tree.
  Future dataset formats are listed as "future work" in the dataset doc.
- **Factorized count prior.** One independent `vocab × vocab` table per
  stream, smoothed with Dirichlet `alpha` (default `1.0`). Cross-stream
  interactions are not modeled — explicitly called out as a v1 limit.
- **Single-device training.** Per the `d23aa97` message, `mido` parsing
  dominates the wall time, so multi-GPU sharding is not worth the
  implementation cost in v1.
- **Observability is opt-in.** ClearML is lazy-loaded and gated by
  `--clearml`; without it, `NullTrainingMonitor` is used.

---

## 9. Files touched across the four commits

| Path | 90067c1 | 50834e7 | 6982996 | d23aa97 |
|---|---|---|---|---|
| `aimusic/ml/__init__.py` | +5 | – | – | – |
| `aimusic/ml/midi_ingest.py` | +144 | – | – | – |
| `aimusic/ml/annotate.py` | +152 | – | – | – |
| `aimusic/ml/dataset.py` | +150 | – | – | +52 |
| `aimusic/ml/count_prior.py` | +166 | – | – | – |
| `aimusic/ml/train.py` | +60 | – | – | +60 |
| `aimusic/ml/bundle.py` | +125 | – | – | – |
| `aimusic/ml/vocab_export.py` | +101 | – | – | – |
| `aimusic/ml/inference.py` | +26 | – | – | – |
| `aimusic/ml/cli.py` | +65 | – | +60 | +34 |
| `aimusic/ml/monitoring.py` | – | – | +274 | – |
| `aimusic/app/cli.py` | +20 | – | – | – |
| `pyproject.toml` | +18 | – | +3 | ±2 |
| `.github/workflows/ci.yml` | +22 | – | – | – |
| `tests/conftest.py` | +25 | – | – | – |
| `tests/test_*.py` (9 files) | +499 | – | +119 | – |
| `docs/ml-training-dataset-requirements.md` | – | +454 | +30 | – |
| `docs/ml-module.md` | – | – | – | +new (this file) |

Net change across the four commits: **+2153 / −1** in ~30 files, all
additive on top of the existing symbolic-music generator.

---

## 10. Where to read next

- Dataset contract: [`docs/ml-training-dataset-requirements.md`](ml-training-dataset-requirements.md)
- End-to-end trainer: `aimusic/ml/train.py::train_prior_from_corpus`
- Inference shim: `aimusic/ml/inference.py::load_trained_neural_prior`
- Monitor API: `aimusic/ml/monitoring.py`
- CLI help: `python -m aimusic.ml.cli train --help`
