# Audio next steps (agents)

Ordered backlog. Prefer one P0/P1 item per session. Full roadmap:
[audio-roadmap.md](audio-roadmap.md).

## P0 — M0 contract (this PR)

- [x] Document architecture / ADRs / contract / roadmap in this repo
- [x] Architecture quarantine tests + `aimusic.audio` stub
- [x] RenderPackage producer + always emit from `generate`
- [x] Merge PR to fork `main`; keep CI green

## P1 — Bridge / deterministic spine

- [x] Optional `[audio-bridge]` extra declared (runtime bridge deferred to PR3)
- [x] CLI `render-audio` with lazy import; validate-only default in PR2
- [x] Native `from_score` loader; prefer planner `structure.json`
- [ ] Port M1 remainder: groove apply, simple/fluidsynth render, orchestrator
- [ ] CI reconcile on fixtures

## P2 — Safe generative loop

- [ ] Profile with prompts+restyle+scoring (mock only)
- [x] CI tests install PyYAML via `requirements.txt` for `tests/test_audio_*.py`
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
