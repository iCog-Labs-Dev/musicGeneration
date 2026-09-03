# Implementation Plan: MIDI→Audio Integration (PR1 / M0)

## Overview

Land the RenderPackage contract and PDF-derived audio-pipeline documentation on
`feature/midi2audio-integration`, then open a PR into fork `main`
(`arsenylosev/musicGeneration`). Org PR to iCog is out of scope for agents.

## Architecture Decisions

- File contract (`RenderPackage`) is the sole symbolic↔audio boundary (B1–B2).
- Core install stays `numpy` + `mido`; `aimusic.audio` is a quarantine stub.
- Always emit `run_<hash>/` from `generate`; keep legacy flat artifacts for UI.
- Fork-first: no iCog git upstream remote; PR target is fork `main`.

## Task List

### Phase 1: Foundation (this PR)

- [x] Branch + tasks files
- [ ] PDF-derived docs + DECISIONS + AGENTS + ADRs
- [ ] Architecture quarantine + tests
- [ ] RenderPackage producer + CLI emit + tests
- [ ] Fork PR + CI green

### Checkpoint: M0

- [x] Docs cover B1–B4, M0–M6, stages, contract
- [x] `generate` emits valid RenderPackage
- [x] Architecture + package tests pass; core CI green
- [x] Fork PR opened; CI green

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Recovered code vs current APIs | Med | Diff against live Score/BeatState |
| Dual artifacts confuse consumers | Low | Document legacy vs contract |
| Scope creep into M1 DSP | High | Hard non-goals in PR1 |

## Open Questions

- None for PR1; bridge/git dep deferred to PR2.
