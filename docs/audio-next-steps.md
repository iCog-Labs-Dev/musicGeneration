# Audio next steps (agents)

Ordered backlog. Prefer one P0/P1 item per session. Full roadmap:
[audio-roadmap.md](audio-roadmap.md).

## P0 — M0 contract (this PR)

- [x] Document architecture / ADRs / contract / roadmap in this repo
- [x] Architecture quarantine tests + `aimusic.audio` stub
- [x] RenderPackage producer + always emit from `generate`
- [ ] Merge PR to fork `main`; keep CI green

## P1 — Bridge / deterministic spine

- [ ] Optional `[audio-bridge]` on tagged `midi2audio-generative@v0.1.0`
- [ ] CLI `render-audio` with lazy import; mock restyle only in CI
- [ ] Port M1: `from_score`, groove apply, simple/fluidsynth render, orchestrator
- [ ] Prefer `from_score` when `structure.json` exists; CI reconcile on fixtures

## P2 — Safe generative loop

- [ ] Profile with prompts+restyle+scoring (mock only)
- [ ] Optional CI job with `.[audio]` + `tests/test_audio_*.py`
- [ ] Never require live API keys in default CI

## P3 — B3 microtonal integrity

- [ ] Enforce `allow_microtonal_diffusion` before endpoint calls
- [ ] Hard-reject via `tuning_check` floors
- [ ] 19-EDO fixture through render + tuning_check

## P4 — M4–M6 / B4

- [ ] Matchering when installed; FAD vs corpus; ClearML tags
- [ ] Live Suno/MusicGen behind flags + vcrpy
- [ ] B4 host lattice → planning energy — next quarter only

## Explicitly not now

- Closed-loop optimization of GTTM weights via audio scorers
- Replacing symbolic purity with in-process audio buffers
- Git submodules of `midi2audio_generative`
- Opening org PRs to `iCog-Labs-Dev/musicGeneration` without an explicit ask
