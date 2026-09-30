# Implementation Plan: MIDI→Audio Integration

## M0 — PR1 (merged)

Land the RenderPackage contract and PDF-derived audio-pipeline documentation on fork
`main` (`arsenylosev/musicGeneration`). Org PR to iCog is out of scope for agents.

### Checkpoint: M0

- [x] Docs cover B1–B4, M0–M6, stages, contract
- [x] `generate` emits valid RenderPackage
- [x] Architecture + package tests pass; core CI green
- [x] Fork PR #2 merged

---

## M1 PR2 — Native audio entry spine (this PR)

### Overview

Add a dependency-light native entry point inside `aimusic.audio`: load and validate
RenderPackage via `from_score`, load audio YAML config, and expose lazy
`render-audio --validate-only` CLI. Declare `[audio-bridge]` extra; runtime bridge
deferred to PR3.

### Task List

- [x] Branch `feature/m1-audio-spine` from fork `main`
- [x] `aimusic/audio/config.py` — frozen dataclass YAML loader
- [x] `aimusic/audio/from_score.py` — `AudioPackageContext` + `validate_package`
- [x] `aimusic/audio/entry.py` + CLI `render-audio` lazy import
- [x] `[audio-bridge]` extra in `pyproject.toml` (packaging only)
- [x] `tests/test_audio_entry.py`; CI tests job installs `.[audio]`
- [x] Update `docs/audio-next-steps.md`, `tasks/todo.md`
- [ ] Fork PR → `main`; CI green (push blocked: no GitHub credentials here)

### Checkpoint: M1 PR2

- [ ] `generate` → `render-audio --validate-only` works
- [ ] Architecture quarantine tests still green
- [ ] No heavy deps in core install
- [ ] Ready for PR3 (groove + simple render + bridge runtime)

### Not doing in PR2

- Groove apply, fluidsynth/simple render, orchestrator, reconcile
- Restyle, scoring, mixmaster
- Runtime m2a bridge delegation

### Risks

| Risk | Mitigation |
|------|------------|
| Scope creep into DSP | `--validate-only` default; full render exits non-zero |
| Bridge git URL unknown | Extra skeleton only; pin in PR3 |
