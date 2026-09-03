# DECISIONS.md

Agent-maintained decision log for audio-pipeline integration into `aimusic`.
Canonical narrative: sibling `midi2audio_generative/AUDIO_PIPELINE_ARCHITECTURE.pdf`
and [docs/audio-pipeline.md](docs/audio-pipeline.md).

## 2026-08-21 — Audio pipeline integration (M0 / PR1)

### Package quarantine (B2)

- **Decision:** All audio DSP / restyle / scoring lives under `aimusic.audio`,
  behind optional extra `[audio]`. Core install stays `numpy` + `mido` only.
- **Rationale:** Preserves functional purity and install weight. Enforced by
  `tests/test_architecture.py`.
- **Alt considered:** Vendoring `m2a` into the root package — rejected.

### Import rules

- **Decision:**
  1. No module outside `aimusic.audio` may import `aimusic.audio`.
  2. `aimusic.audio` may import only `aimusic.core.*` and `aimusic.theory.*`.
  3. Importing `aimusic.core` must not pull `torch`, `librosa`, `madmom`, or `demucs`.
- **Rationale:** PDF §2.4; B4 feedback travels as data files, never reverse imports.

### Always emit structure.json / RenderPackage

- **Decision:** `generate` always emits a `RenderPackage` (`structure.json` +
  `tuning.json` + score/MIDI/manifest) into `run_<hash>/`. Legacy flat artifacts
  are also written for Gradio/UI compatibility.
- **Rationale:** Cost is negligible; every symbolic run is audio-ready (PDF §10 #6).

### Tick rounding (groove apply)

- **Decision:** Round-half-even to nearest tick; offsets are absolute per note
  relative to the quantized onset (not cumulative across successive transforms).
- **Rationale:** Matches m2a M1 property tests; stable under Hypothesis.

### Crossfade at chunk seams

- **Decision:** Equal-power crossfade; default overlap 2 bars, seams chosen by
  descending `boundary_lvl` then snapped to downbeats.
- **Rationale:** PDF §4.3 / §10 #2.

### LUFS targets

- **Decision:** drums −12, bass −13, comping −15, lead −12, master −10 LUFS
  (integrated), matching `config/audio.default.yaml`.
- **Rationale:** PDF appendix B; adjustable via config.

### Microtonal cents tolerance

- **Decision:** A track is `microtonal: true` when any pitch class is more than
  **8 cents** from the nearest 12-TET pitch. Tuning floor: mean and P95 f0
  deviation from the intended EDO lattice must stay within **15 cents** (P95)
  or the candidate is rejected.
- **Rationale:** PDF appendix B defaults.

### Backend-interface exemption

- **Decision:** Audio DSP (librosa/torch/…) is exempt from the NumPy→JAX backend
  protocol used by the symbolic core. Recorded here so agents do not force a
  shared numeric backend across the quarantine.
- **Rationale:** PDF §1.3 / §10 #5.

### E_host normalization (B4 — deferred)

- **Decision:** Deferred to next quarter. When implemented, `E_host` participates
  in the same weighted energy sum as GTTM features (not a post-hoc edge mask).
- **Rationale:** Keeps SB theory clean; PDF §10 #7.

### Bridge then port

- **Decision:** Phase 1 may consume `midi2audio-generative` (`m2a`) via optional
  git dependency. Phase 2 ports modules into `aimusic.audio` and drops the git
  dep once parity tests pass. M0 does **not** add the git dependency yet.
- **Rationale:** Fast E2E while preserving reviewable PRs (ADR-003).

### Fork-first contribution path

- **Decision:** Integration PRs land on `arsenylosev/musicGeneration` (`main`)
  first. Opening a PR to `iCog-Labs-Dev/musicGeneration` is a separate,
  explicit human step after fork CI is green. Do not add iCog as a git
  `upstream` remote for this workstream unless asked.
- **Rationale:** Keeps CI and review ownership on the fork during migration.
