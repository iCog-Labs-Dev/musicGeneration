# EPICs and Phases for the Full Music Generation Pipeline

## Planning Basis

This plan is based on the current repository state plus the project design documents:

- `Software Design GTTM SB.pdf`
- `dev_plan.md`
- `readme.md`
- `reports.md`

It targets the full symbolic music generation pipeline:

1. Configs, vocabularies, tonal system, and priors
2. Endpoint planning (Method A first)
3. Sparse layered `BeatState` graph construction
4. Schrodinger Bridge inference
5. Sampling or MAP trajectory extraction
6. `BeatState` path decoding into a symbolic score
7. MIDI rendering and export
8. Diagnostics, CLI, and evaluation

The immediate output remains symbolic score plus MIDI. Full generative audio
synthesis remains out of scope for the **symbolic MVP**. Optional post-MVP
MIDI→audio integration (RenderPackage + `aimusic.audio`) is tracked as
**EPIC 11** — see [`docs/audio-roadmap.md`](docs/audio-roadmap.md).

## Current Repository Baseline

The repo already contains useful foundations, but it is still well short of the full pipeline:

- `config.py` currently defines only `EDOConfig` and `MicrotonalRendering`.
- `edo.py` provides EDO pitch-class math and MIDI conversion helpers.
- `tonal.py` provides EDO-generic chord templates, functional classification, and tonal distance utilities.
- `gttm_features.py` contains an early structural prototype built around `MusicalEvent`, grouping, time-span reduction, and prolongational stubs.
- `rhythm_features.py` introduces a `BeatState`-like structure and simple meter/boundary/groove scoring heuristics.
- Tests currently cover EDO math, tonal utilities, early GTTM logic, and rhythm scoring.

The major missing pieces are:

- Unified core types for `BeatState`, `NoteEvent`, `Score`, `Layer`, and `Edge`
- Vocabulary layer for meters, keys, chords, roles, groove families, and head tokens
- Prior interfaces and corpus-backed priors
- BeatState-centric GTTM energy library
- Candidate generation and sparse graph expansion
- Schrodinger Bridge solver, sampling, and MAP extraction
- Endpoint planning and orchestration logic
- Decoder and track generators
- MIDI renderer, CLI, diagnostics, and end-to-end evaluation

## Delivery Principles

- Keep EDO as a first-class parameter, with 12-EDO and 19-EDO both supported.
- Use immutable dataclasses and pure functions wherever practical.
- Thread RNG explicitly through all sampling code.
- Keep numerics backend-agnostic, with NumPy first and JAX-compatible design later.
- Make Method A the first end-to-end target before Method B and full neural-prior integration.
- Treat diagnostics and tests as part of the pipeline, not post-hoc work.

## EPIC Backlog

### EPIC 1: Core Domain Model and Configuration Spine

**Goal:** Establish the shared types, vocabularies, and config objects that every downstream module will depend on.

**Primary modules:** `config.py`, `core_types.py`, `vocab.py`, `edo.py`, `tonal.py`

**Tasks**

- [x] Add immutable config dataclasses for `StyleConfig`, `PriorWeights`, `SBConfig`, `DecodeConfig`, and `PlanConfig`.
- [x] Define canonical immutable types for `BeatState`, `NoteEvent`, `Score`, `Layer`, `Edge`, and endpoint distributions.
- [x] Create token vocabularies for meters, beat positions, boundaries, keys, chord ids, role ids, head ids, and groove ids.
- [x] Define serialization and pretty-print helpers so states and scores can be inspected in logs and tests.
- [x] Unify the repository around one `BeatState` definition instead of the current split between `gttm_features.py` and `rhythm_features.py`.
- [x] Decide where musical role semantics live (`hold`, `prep`, `change`, `cad`) and encode them in the vocabulary layer.
- [x] Add validation helpers to reject impossible config combinations early.
- [x] Add RNG utility helpers so future sampling functions share one reproducible interface.
- [x] Refactor existing modules to import shared types instead of local placeholders.
- [x] Add unit tests for equality, hashing, config validation, and vocabulary integrity.

**Definition of done**

- [x] Every downstream module can depend on one shared set of frozen types.
- [x] All config objects required by the PDF spec exist and validate correctly.
- [x] Existing tests still pass after the type consolidation.

### EPIC 2: BeatState-Centric GTTM Energy Library

**Goal:** Replace the current prototype scoring logic with a reusable feature library that computes `Egttm(s -> s'; t)` over `BeatState` transitions.

**Primary modules:** `gttm_features.py`, `rhythm_features.py`, `tonal.py`

**Tasks**

- [x] Refactor feature functions to accept `BeatState`, `next_state`, time index, and optional local window context.
- [x] Implement meter features for legal meter progression, beat-position validity, and boundary placement on strong positions.
- [x] Implement grouping features for phrase/subgroup boundaries aligned with bar and subdivision structure.
- [x] Implement harmonic features driven by tonal distance, chord compatibility, cadence pressure, and key motion.
- [x] Implement prolongational role features for `hold`, `prep`, `change`, and `cad` aligned with metrical strength.
- [x] Implement melodic head features for chord-tone anchoring, weak-beat approach tones, and resolution behavior.
- [x] Implement groove features for groove-family continuity and controlled change near boundaries.
- [x] Create a weighted feature registry so `Egttm` is an explicit sum of named feature functions.
- [x] Merge or clearly separate the current `rhythm_features.py` logic so there is no duplicated scoring authority.
- [x] Add caching for repeated tonal-neighbor and tonal-distance lookups.
- [x] Add unit tests for each feature family and for aggregate energy monotonicity on known musical examples.

**Definition of done**

- [x] `Egttm` operates on `BeatState` transitions rather than `MusicalEvent` stubs.
- [x] The six feature families described in the spec are represented in code.
- [x] Feature weights can be configured without editing scoring logic.

### EPIC 3: Prior Interfaces and Neural Prior Seams

**Goal:** Build the data-side transition prior layer so the graph scorer can mix external learned priors with GTTM energy without blocking the algorithmic MVP.

**Primary modules:** `priors.py`, `vocab.py`, `config.py`

**Tasks**

- [x] Define a minimal prior protocol such as `logp_next(prev_state, next_state, t, context) -> float`.
- [x] Implement `NullPrior` for algorithmic-only mode.
- [x] Define a neural-prior contract for scalar and optional batched scoring so graph code does not depend on model internals.
- [x] Design a structural tokenization and context contract from symbolic examples into chord, key, role, groove, head, and boundary streams for a future neural model.
- [x] Decide whether the neural prior interface is whole-state, factorized, or mixed, and document the decision.
- [x] Add immutable config/artifact metadata for external neural checkpoints, tokenizers, and model-version manifests.
- [x] Implement a placeholder `NeuralPrior` adapter that satisfies the production interface but returns neutral or deterministic mock scores until the external model exists.
- [x] Add model manifest save/load helpers for externally produced prior artifacts.
- [x] Add tests for deterministic placeholder scores, scalar/batch parity, manifest loading, and scorer integration.

**Definition of done**

- [x] The scorer can run with `NullPrior` or placeholder `NeuralPrior` without changing graph code.
- [x] The structural tokenization and model-I/O contract are documented well enough for an external team to target.
- [x] The prior API is stable enough to accept an externally implemented neural model and artifact bundle.

### EPIC 4: Candidate Generation and Sparse Graph Builder

**Goal:** Construct a bounded layered graph of plausible `BeatState` candidates over the planning horizon.

**Primary modules:** `candidates.py`, `graph.py`, `core_types.py`, `vocab.py`, `tonal.py`, `priors.py`, `gttm_features.py`

**Tasks**

- [x] Implement hard gating rules for meter continuity, beat positions, role legality, and boundary constraints.
- [x] Implement key proposals that mostly stay local and only modulate near permitted boundaries or cadential contexts.
- [x] Implement chord proposals from tonal neighbors, cadence targets, and corpus top-K suggestions.
- [x] Implement melodic head proposals consistent with chord, key, and metrical strength.
- [x] Implement groove proposals that preserve groove families but allow constrained boundary-driven switches.
- [x] Deduplicate candidate states using hashable `BeatState` objects.
- [x] Implement edge scoring as `lambda_data * logp_data - lambda_gttm * Egttm`.
- [x] Build layer expansion logic for `t = 0..T-1`.
- [x] Enforce `K_max` per layer and `D_max` per source state.
- [x] Add heuristic pruning using incoming log-mass and distance-to-go toward endpoint constraints.
- [x] Return graph diagnostics such as layer sizes, rejected candidates, and prune reasons.
- [x] Add unit tests for no-illegal-transition guarantees and pruning behavior.

**Definition of done**

- [x] The graph builder can create a sparse layered graph for a small horizon without exploding in state count.
- [x] Every retained edge can be traced back to a legal proposal and a scored transition.
- [x] The graph builder exposes enough diagnostics to explain why states were kept or pruned.

### EPIC 5: Schrodinger Bridge Solver, Sampling, and MAP

**Goal:** Infer globally coherent trajectories over the sparse graph using SB.

**Primary modules:** `sb.py`, `graph.py`, `config.py`

**Tasks**

- [x] Define the sparse graph input contract for layers, edges, and endpoint distributions.
- [x] Implement log-space forward and backward scaling updates for SB.
- [x] Implement sparse scatter log-sum-exp helpers behind a small backend abstraction.
- [x] Add numerical safeguards for underflow, empty support, and non-convergence cases.
- [x] Compute bridge-modified transition probabilities from the solver outputs.
- [x] Implement pure path sampling from the bridge transitions.
- [x] Implement pure MAP extraction using Viterbi-style dynamic programming.
- [x] Return optional bridge marginals and convergence traces.
- [x] Add deterministic reproducibility tests for solver inputs and RNG-controlled sampling.
- [x] Add solver convergence tests on tiny hand-built graphs.

**Definition of done**

- [x] SB converges on controlled sparse examples.
- [x] Sampling and MAP can be called independently on the solved bridge.
- [x] The solver API is pure and compatible with later backend swaps.

### EPIC 6: Endpoint Planning and Method A Orchestration

**Goal:** Turn the lower-level graph and solver pieces into a usable generation plan centered on Method A.

**Primary modules:** `plans.py`, `config.py`, `graph.py`, `sb.py`

**Tasks**

- [x] Implement endpoint distribution generators for `pi0` and `piT`.
- [x] Define how start and end passages are sampled or selected from style constraints.
- [x] Encode section metadata needed by future section-wise planning even if the first MVP is single-section.
- [x] Implement Method A as a top-level pure orchestration flow.
- [x] Add run configuration objects that bundle style, priors, decoding, and planning choices.
- [x] Surface planning diagnostics such as chosen endpoint states, section tags, and target tension arcs.
- [x] Add smoke tests for an end-to-end Method A run over a short horizon.

**Definition of done**

- [x] A single function can run config -> endpoints -> graph -> SB -> path for Method A.
- [x] Endpoint choices are inspectable and reproducible from a seed.
- [x] Method A is the stable MVP path before Method B work begins.

### EPIC 7: Decoder from BeatState Path to Symbolic Score

**Goal:** Realize the structural trajectory into playable multi-track symbolic music.

**Primary modules:** `decode.py`, `tonal.py`, `core_types.py`

**Tasks**

- [x] Define the symbolic `Score` container and track-level event assembly rules.
- [x] Build sub-beat grids from groove tokens and decode quantization settings.
- [x] Implement comping realization with voicing rules and voice-leading constraints.
- [x] Implement bass-line generation with register control and approach-tone logic.
- [x] Implement lead generation anchored to head tokens with contour and resolution constraints.
- [x] Implement drum generation tied to groove families and section boundaries.
- [x] Map structural tension and roles into velocity, density, articulation proxies, and expressive controls.
- [x] Handle rests, sustain, and overlap cleanup across tracks.
- [x] Add decoder tests for chord-tone anchoring, register limits, voice-leading sanity, and track density behavior.

**Definition of done**

- [x] A `BeatState` path can be decoded into a coherent multi-track `Score`.
- [x] Decoder behavior is modular by track generator and testable without MIDI export.
- [x] The score remains EDO-native until the rendering stage.

### EPIC 8: MIDI Rendering and Export

**Goal:** Convert the symbolic score into inspectable MIDI output for 12-EDO and microtonal EDOs.

**Primary modules:** `midi_render.py`, `edo.py`, `config.py`

**Tasks**

- [x] Implement 12-EDO direct mapping from pitch heights to MIDI note numbers.
- [x] Implement 19-EDO and other N-EDO rendering via MPE-style pitch bends.
- [x] Add a rendering path for MTS-style tuning support if practical, or clearly document it as deferred.
- [x] Design track-to-channel allocation and channel reuse rules.
- [x] Encode per-note expressive information into MIDI-friendly controls where available.
- [x] Generate valid `.mid` files with deterministic ordering and metadata.
- [x] Add rendering smoke tests and small fixture exports.
- [x] Add inspection helpers to summarize rendered note counts, channels, and pitch-bend usage.

**Definition of done**

- [x] The system can export playable MIDI from the decoded score.
- [x] 12-EDO and 19-EDO runs have separate verified rendering paths.
- [x] Rendering stays isolated from structural planning logic.

### EPIC 9: CLI, Diagnostics, Evaluation, and Reproducibility

**Goal:** Make the pipeline usable, debuggable, and measurable.

**Primary modules:** `cli.py`, `diagnostics.py`, `tests/` or existing test files

**Tasks**

- [x] Add a CLI entry point for `generate`, `inspect`, and `export`.
- [x] Emit diagnostics for chord timeline, key timeline, groove timeline, boundary sequence, and role sequence.
- [x] Compute a tension curve from tonal distances and prolongational role behavior.
- [x] Capture SB convergence logs, effective entropy, layer sizes, and prune statistics.
- [x] Add seed reporting and run manifests so outputs are reproducible.
- [x] Add end-to-end fixtures that produce stable short passages for regression testing.
- [x] Define basic evaluation criteria for structural coherence, cadence plausibility, and decode validity.
- [x] Document failure modes and fallback behaviors for empty layers, solver instability, and decode dead ends.

**Definition of done**

- [x] A generation run produces both output files and diagnostics.
- [x] Regressions can be caught with short deterministic fixtures.
- [x] Users can inspect why a generation succeeded or failed.

### EPIC 10: Long-Form Scaling, Method B, and Hybrid Extensions

**Goal:** Extend the MVP into the full research direction described in the spec.

**Primary modules:** `plans.py`, `priors.py`, `sb.py`, `decode.py`

**Tasks**

- [ ] Implement Method B with start -> middle and middle -> return SB passes.
- [ ] Implement section-wise SB for intro, theme, solo, bridge, and return structures.
- [ ] Add soft stitching constraints between sections.
- [ ] Add performance improvements for long horizons, including cached neighbor sets and memoized distances.
- [ ] Implement the optional `NeuralPrior` integration path.
- [ ] Add evaluation runs on longer forms and compare Method A vs Method B behavior.
- [ ] Expand diagnostics to section-level summaries and long-run memory/performance metrics.

**Definition of done**

- [ ] Long-form planning no longer relies on one monolithic SB pass.
- [ ] Method B and hybrid priors are integrated without destabilizing Method A.
- [ ] The project can target 5-15 minute forms with bounded compute.

### EPIC 11: Optional MIDI→Audio Production Pipeline (post-MVP)

**Goal:** Quarantined audio path behind a RenderPackage file contract without
compromising symbolic purity or core install weight.

**Primary modules:** `aimusic/core/render_package.py`, `aimusic/audio/*`,
`config/audio.default.yaml`

**Docs:** `docs/audio-pipeline.md`, `docs/audio-roadmap.md`, `DECISIONS.md`

**Tasks**

- [x] M0: RenderPackage contract + architecture quarantine + always-emit from `generate`
- [ ] M1: Deterministic spine (groove + stem render + orchestrator)
- [ ] M1.5–M2: Microtonal routing + scorers
- [ ] M3: One restyle endpoint end-to-end (mock in CI)
- [ ] M4–M6: References, breadth, single-ref seams
- [ ] B4 host-conditioned planning (deferred next quarter)

**Definition of done**

- [ ] `pip install aimusic` stays free of torch/librosa/demucs
- [ ] Every `generate` run emits a valid `run_<hash>/` RenderPackage
- [ ] Optional `[audio]` path can produce stems/master from fixtures under CI budget

## Phase Plan

### Phase 1: Foundation and Type Consolidation

**Objective:** Build the shared data model and config backbone before any new pipeline module is added.

**EPICs covered:** EPIC 1

**Tasks**

- [ ] Create `core_types.py` and move canonical `BeatState`, `NoteEvent`, `Score`, `Layer`, and `Edge` there.
- [ ] Expand `config.py` with all missing immutable config dataclasses.
- [ ] Create `vocab.py` with first-pass 12-EDO vocabularies and an extension point for 19-EDO.
- [ ] Refactor `rhythm_features.py` and `gttm_features.py` to use shared types.
- [ ] Add validation, serialization, and hashing tests.

**Exit criteria**

- [ ] There is exactly one shared `BeatState` definition.
- [ ] All config objects required for later phases exist.
- [ ] Existing tests pass after refactoring.

### Phase 2: Structural Energies and Priors

**Objective:** Make transition scoring musically meaningful and data-aware.

**EPICs covered:** EPIC 2, EPIC 3

**Tasks**

- [ ] Finish the BeatState-centric GTTM feature registry.
- [ ] Merge rhythm/meter/groove logic into the main energy story.
- [ ] Implement `NullPrior` and placeholder `NeuralPrior`.
- [ ] Define structural tokenization, context, and model-artifact contracts for the future neural prior.
- [ ] Add tests for feature-family behavior and prior scoring.

**Exit criteria**

- [ ] `Egttm` can score a `BeatState` transition with configurable weights.
- [ ] A corpus prior can be plugged in without changing scorer interfaces.
- [ ] Scoring is deterministic under fixed inputs.

### Phase 3: Candidate Generation and Sparse Graph MVP

**Objective:** Build a legal, sparse, inspectable candidate graph.

**EPICs covered:** EPIC 4

**Tasks**

- [ ] Implement hard gating rules.
- [ ] Implement proposal functions for key, chord, role-compatible head, and groove choices.
- [ ] Implement edge scoring and graph expansion.
- [ ] Add deduplication, `K_max`, `D_max`, and endpoint-aware pruning.
- [ ] Add graph diagnostics and illegal-transition tests.

**Exit criteria**

- [ ] Small-horizon graphs build successfully.
- [ ] Layer growth is bounded by configured caps.
- [ ] All retained candidates are legal and explainable.

### Phase 4: Schrodinger Bridge Solver and Path Extraction

**Objective:** Solve for globally coherent trajectories over the sparse graph.

**EPICs covered:** EPIC 5

**Tasks**

- [ ] Implement log-space sparse SB updates.
- [ ] Add bridge transition extraction.
- [ ] Add sampling and MAP path extraction.
- [ ] Add convergence diagnostics and tiny-graph tests.

**Exit criteria**

- [ ] SB converges on known toy examples.
- [ ] Both sampled and MAP paths can be extracted from solved bridges.
- [ ] Solver outputs are numerically stable on small sparse graphs.

### Phase 5: Method A End-to-End Structural Planning

**Objective:** Deliver the first complete planning pipeline through `BeatState` path generation.

**EPICs covered:** EPIC 6

**Tasks**

- [ ] Implement endpoint distribution generation for start and end passages.
- [ ] Implement Method A orchestration from config to solved path.
- [ ] Add run manifests and seed control.
- [ ] Add a short-horizon end-to-end smoke test.

**Exit criteria**

- [ ] The project can generate a valid `BeatState` trajectory from Method A.
- [ ] Endpoint choices and seeds are logged and reproducible.
- [ ] Failures are diagnosable at the planning stage.

### Phase 6: Decode Structural Paths into Music and Export MIDI

**Objective:** Convert the abstract path into audible symbolic output.

**EPICs covered:** EPIC 7, EPIC 8

**Tasks**

- [ ] Implement the symbolic decoder for drums, bass, comping, and lead.
- [ ] Build EDO-native `Score` assembly.
- [ ] Implement MIDI rendering for 12-EDO and 19-EDO.
- [ ] Add fixture exports and decode/render validation tests.

**Exit criteria**

- [ ] A generated `BeatState` path becomes a multi-track score and MIDI file.
- [ ] Decode logic is modular and separately testable.
- [ ] Rendering works for both direct and microtonal modes.

### Phase 7: Diagnostics, CLI, and Regression Safety

**Objective:** Make the full pipeline usable in day-to-day development.

**EPICs covered:** EPIC 9

**Tasks**

- [ ] Add CLI commands for generation, export, and inspection.
- [ ] Emit timelines, tension curves, prune stats, and solver logs.
- [ ] Add deterministic regression fixtures for short passages.
- [ ] Document common failure modes and debug workflows.

**Exit criteria**

- [ ] A user can run generation from the command line and inspect what happened.
- [ ] Short regression fixtures protect core pipeline behavior.
- [ ] Diagnostics explain both successful and failed runs.

### Phase 8: Long-Form Scaling and Research Extensions

**Objective:** Extend the MVP into the full long-form design space from the spec.

**EPICs covered:** EPIC 10

**Tasks**

- [ ] Implement Method B.
- [ ] Add section-wise SB with soft stitching.
- [ ] Add performance and caching improvements for long horizons.
- [ ] Integrate the externally implemented neural prior.
- [ ] Evaluate longer forms and compare generation strategies.

**Exit criteria**

- [ ] The project supports more than the MVP single-pass Method A workflow.
- [ ] Long-form runs are computationally bounded and diagnosable.
- [ ] Advanced modes do not regress the Method A baseline.

## Recommended MVP Cut

If the goal is to deliver one working pipeline before expanding scope, the first hard MVP should include:

- Phase 1 through Phase 6
- Method A only
- `NullPrior` plus placeholder `NeuralPrior`
- NumPy backend only
- 12-EDO and 19-EDO MIDI rendering
- Basic diagnostics and at least one deterministic end-to-end fixture

Everything in Phase 8 should be treated as second-wave work unless the pipeline is already stable.

## Suggested Success Criteria for the First Complete Pipeline

- [ ] A user can choose a seed and config, run Method A, and receive a MIDI file plus diagnostics.
- [ ] The generated output has a valid `BeatState` path, valid decoded score, and valid MIDI render.
- [ ] Layer sizes and outdegree remain within configured limits.
- [ ] SB converges or fails with actionable diagnostics.
- [ ] The decoder respects register, chord-tone anchoring, and groove constraints.
- [ ] The full short-form pipeline is covered by automated tests.
