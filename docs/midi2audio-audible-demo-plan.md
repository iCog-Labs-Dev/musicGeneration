# MIDI-to-Audio Audible Demo Implementation Plan

**Branch:** `feature/midi2audio-audible-demo`  
**Scope:** Planning only in this commit; no implementation phases have started.  
**Target:** Deterministic, reproducible conversion of generated or imported MIDI into playable instrument stems and a rough WAV mix.

## Goal and stopping point

Complete a dependable audio demonstration pipeline:

```text
RenderPackage or imported MIDI
  -> validation and reconciliation
  -> optional groove processing
  -> tuning-aware instrument rendering
  -> role stems
  -> playable rough_mix.wav
```

The milestone stops at reliable audible examples. It does not include paid APIs,
generative restyling, mastering, or host-conditioned symbolic planning.

## Current repository state

The repository already contains a functioning M1 CI-safe audio spine:

- `generate` emits a RenderPackage containing MIDI, Score, planner structure,
  tuning metadata, and a manifest.
- `render-audio` lazily loads `aimusic.audio`, validates a RenderPackage, and can
  run the native audio orchestrator.
- The native orchestrator performs MIDI analysis, optional groove application,
  simple additive rendering, stem generation, rough mixing, and caching.
- An optional pinned `midi2audio_generative` bridge is available.
- The focused audio and architecture suite currently passes (22 tests in the
  planning-session environment).

The existing simple backend already writes WAV files, but it is a sine-wave CI
renderer rather than a dependable instrument renderer. The next milestone is
therefore not merely "write audio"; it is "produce recognizable, correctly
routed, timing- and tuning-preserving audio examples."

## Linear alignment

The relevant issues assigned to Arseniy Losev are:

| Ticket | Status | Plan relationship |
| --- | --- | --- |
| MUS-29 — Optional MIDI-to-audio production pipeline | Backlog | Parent epic; partially implemented |
| MUS-30 — RenderPackage contract and quarantine | Done | Existing foundation; harden integrity |
| MUS-31 — Audio entry and validation spine | Done | Existing foundation |
| MUS-32 — Groove, simple render, orchestrator, bridge | Done | Existing foundation |
| MUS-33 — Upstream M1 integration | Done | Upstream PR #47 is attached and merged |
| MUS-34 — FluidSynth backend and reconcile | Backlog | Primary implementation target |
| MUS-35 — Microtonal spine and tuning check | Backlog | Implement dry-render integrity needed for examples |
| MUS-36 — Audio scorers and corruption suite | Backlog | Deferred from this branch |
| MUS-37 — One restyle endpoint E2E | Backlog | Deferred; blocked on MUS-36 |
| MUS-38 — References, mastering, endpoint breadth | Backlog | Deferred |

Required roadmap order remains:

```text
MUS-34 -> MUS-35 -> MUS-36 -> MUS-37 -> MUS-38
```

This feature stops after the reliable audible-demo portion of MUS-35. MUS-35
must not be marked fully complete if its remaining Scala or diffusion-gating
acceptance criteria have not been delivered.

## Gaps found during analysis

1. `render_role_stems` invokes the simple renderer directly, bypassing the
   renderer registry.
2. The default YAML uses `render.routing`, while the registry expects
   `render.role_map`; generated `comping` also differs from registry `comp`.
3. Groove rewriting reconstructs PrettyMIDI notes and instruments but does not
   preserve all pitch bends, RPN bend range, MTS SysEx, controllers, or other
   expressive metadata.
4. The simple renderer computes frequency from integer MIDI note numbers and is
   therefore always effectively 12-TET.
5. RenderPackage structure and tuning are validated, but the native
   orchestrator subsequently receives only the MIDI path and re-infers data.
6. Render-cache identity does not fully cover SoundFont bytes, effective
   routing, planner structure, tuning, or fallback policy.
7. The default audio YAML resides outside the installed Python package while
   package data is disabled, so wheel installs may not find it.
8. Native bridge fallback catches broad exceptions and can mask a native
   rendering regression when the external bridge is installed.
9. Existing tests establish artifact production and cache behavior, but not
   rendered pitch accuracy, routing compatibility, SoundFont invalidation, or
   installed-wheel configuration behavior.

## Phase 0 — Isolated baseline

Implementation must continue on this dedicated branch (preferably in a separate
worktree) rather than in `main`.

Tasks:

1. Confirm the branch base and preserve unrelated local changes outside the
   worktree.
2. Establish three verification lanes:
   - symbolic/core tests;
   - lightweight audio tests using the simple backend;
   - optional real-FluidSynth integration tests.
3. Record baseline failures separately from feature regressions.
4. Keep each phase in focused, reviewable commits.

Exit criterion: the development environment and relevant baseline suites are
reproducible without changing the symbolic core dependency boundary.

## Phase 1 — Dependable package, configuration, and reconciliation

Primary areas:

- `aimusic/core/render_package.py`
- `aimusic/audio/from_score.py`
- `aimusic/audio/config.py`
- `aimusic/audio/orchestrator.py`
- package-data configuration

Tasks:

1. Pass the full RenderPackage audio context into orchestration, including
   authoritative planner structure, tuning, package identity, and track roles.
2. Add planner-versus-MIDI reconciliation for:
   - track names, roles, programs, and note counts;
   - tempo and meter;
   - duration and bar boundaries;
   - field-level mismatch diagnostics.
3. Recompute and verify package/source hashes when loading or reusing a package.
4. Move the default audio configuration into package data and load it through
   `importlib.resources` or an equivalent wheel-safe mechanism.
5. Define one routing schema and normalize role names, including `comping`.
6. Make backend fallback explicit and record the effective backend in manifests.
7. Extend render-cache identity with effective configuration, package/tuning
   identity, renderer version, routing, and SoundFont content hash.
8. Avoid broad exception fallback that can hide native backend defects.

Exit criteria:

- Installed-wheel default configuration loads outside the source checkout.
- A structure, tuning, routing, or SoundFont-content change causes a cache miss.
- Reconciliation failures identify the exact mismatched fields.
- Audio still imports only across the documented quarantine boundary.

## Phase 2 — FluidSynth instrument rendering (MUS-34)

Primary areas:

- new `aimusic/audio/render/fluidsynth_r.py`
- `aimusic/audio/render/registry.py`
- `aimusic/audio/render/stems.py`
- optional dependency extras and audio profiles

Tasks:

1. Selectively port the tested FluidSynth adapter from the pinned
   `midi2audio_generative` implementation; do not copy known stale stem-collision
   behavior.
2. Add an optional `[audio-render]` extra while keeping the symbolic core at its
   current lightweight dependency boundary.
3. Resolve a SoundFont from explicit argument, profile, or environment with
   actionable errors.
4. Connect stem generation to the renderer registry and authoritative package
   roles.
5. Preserve General MIDI programs and percussion-channel semantics.
6. Preserve tempo, note timing, sustain, pitch-bend, and relevant controller
   behavior through per-stem extraction.
7. Produce consistent-length, non-silent stems and a headroom-controlled rough
   mix.
8. Retain `simple` as the lightweight default/fallback, but add a listening-demo
   profile that requires FluidSynth and cannot silently fall back.
9. Document the separate requirements for Python bindings, native FluidSynth,
   and a user-supplied appropriately licensed SoundFont.

Exit criteria:

- An eight-bar 12-EDO fixture produces recognizable drums, bass, comping, and
  lead stems plus `rough_mix.wav`.
- Output sample rate, duration, finiteness, non-silence, and peak headroom are
  checked.
- Reconcile tests pass within documented tolerances.
- Default CI remains green without FluidSynth or a SoundFont.

## Phase 3 — Timing and microtonal integrity (scoped MUS-35)

Tasks:

1. Preserve pitch-wheel, RPN bend range, MTS/tuning data, controllers, tempo
   changes, and channel associations through groove processing and stem
   isolation.
2. Consume `tuning.json` and `structure.json` as renderer inputs rather than
   merely validating and discarding them.
3. Define renderer capability checks for direct 12-TET, MPE, and MTS paths.
4. Fail clearly when a backend cannot render the requested tuning; never
   silently flatten 19-EDO to 12-TET.
5. Add deterministic `tuning_check` analysis for controlled reference tones.
6. Exclude percussion tracks from pitched-frequency checks.
7. Ensure tuning metadata participates in cache identity and provenance.

Acceptance tests:

- A 19-EDO MPE fixture retains bend-range and pitch-wheel behavior after groove
  processing; reconstructed pitches stay within one cent at the MIDI stage.
- A controlled dry 19-EDO render targets approximately five-cent accuracy to
  the intended lattice.
- Configured mean/P95 tuning floors are enforced; P95 must not exceed the
  documented 15-cent floor.
- A deliberately 12-TET-flattened version fails tuning validation.
- Percussion remains GM drum-key based and is excluded from f0 validation.
- Unsupported MTS rendering is rejected unless a verified, lossless supported
  path is implemented.

Exit criterion: 12-EDO and the supported 19-EDO path can be demonstrated without
unreported tuning loss.

## Phase 4 — User-facing example workflow

Support two entry paths:

1. Existing product path: `generate` -> RenderPackage -> `render-audio`.
2. Imported MIDI path: a lazy-loaded audio command such as `render-midi`, with
   inferred metadata contained entirely in `aimusic.audio`.

Tasks:

1. Add lightweight-preview and FluidSynth-listening profiles.
2. Add explicit output-directory and SoundFont options.
3. Use safe output names and reject malformed or unsupported input with clear
   diagnostics.
4. Add a reproducible example command or script and Windows/Linux setup notes.
5. Produce a listening set containing:
   - the existing eight-bar 12-EDO fixture;
   - one freshly generated 12-EDO passage;
   - controlled 19-EDO reference tones;
   - one short 19-EDO multitrack passage.
6. Each example run should retain:
   - source MIDI and package metadata;
   - expressive MIDI when groove is enabled;
   - named stem WAV files;
   - `rough_mix.wav`;
   - backend, SoundFont hash, and effective configuration provenance;
   - reconciliation and tuning reports where applicable.

SoundFonts, generated WAV files, credentials, and large model assets must not be
committed. Commit only fixture sources, profiles, expected metrics, and
reproduction instructions.

Exit criterion: documented local commands reliably produce files that can be
played and reviewed as actual sound examples.

## Verification strategy

### Lightweight CI

- RenderPackage contract and hash-tamper tests.
- Architecture/import quarantine tests.
- Config/routing and installed-wheel smoke tests.
- Groove property tests, including expressive-event preservation.
- Simple-render artifact, cache, sidecar, and duplicate-role tests.
- Mocked FluidSynth selection, fallback, and error-path tests.
- Raw-MIDI safety and malformed-input tests.
- No live keys, paid calls, SoundFont, GPU, or native FluidSynth required.

### Optional real-render integration lane

- Use a pinned, license-approved test SoundFont.
- Run a real 12-EDO FluidSynth end-to-end fixture.
- Check stems/mix for sample rate, duration, finite values, RMS, and headroom.
- Run controlled 19-EDO tuning checks only for explicitly supported paths.
- Target the roadmap's approximately 60-second fixture budget where practical.

### Human listening audit

- Listen for recognizable role assignment and gross timing errors.
- Compare groove-disabled and groove-enabled output.
- Review the 12-EDO musical example and supported 19-EDO example.
- Treat numerical metrics as fidelity evidence, not a claim of production
  quality.

## Definition of done

The feature is complete when:

- Fresh and wheel-style installations can load the shipped configuration.
- Generated RenderPackages and supported imported MIDI both render.
- Real-FluidSynth mode emits correctly named role stems and a playable mix.
- Timing, sustain, programs, percussion, and supported tuning survive the path.
- Duplicate names cannot overwrite stems.
- Cold and warm cache results are reproducible in a pinned environment.
- Relevant input/config/SoundFont/tuning changes invalidate cached rendering.
- Unsupported microtonal paths fail explicitly rather than flattening pitches.
- Lightweight CI does not need external assets or services.
- Real rendering is covered by a separate integration lane.
- Reproducible playable examples and their provenance are available for review.

## Explicitly deferred

This branch does not implement:

- MUS-36 CLAP/FAD and full objective scoring;
- MUS-37 Stable Audio or other paid/generative endpoints;
- prompt generation, candidate search, or generative restyling;
- MUS-38 Matchering, mastering, references, endpoint breadth, or HTML audit;
- Gradio UI expansion;
- Method B or unrelated planner/ML refactoring;
- B4 host-conditioned planning or closed-loop GTTM optimization.

These remain separate milestones after reliable deterministic rendering and
tuning integrity are established.
