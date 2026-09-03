# Todo: MIDI→Audio PR1 (M0)

- [x] Create `feature/midi2audio-integration` (fork only; no iCog upstream)
- [x] Land PDF-derived docs (pipeline, contract, roadmap, stages, B4, testing, config, next-steps)
- [x] Seed `DECISIONS.md`, `AGENTS.md`, `docs/decisions/*`, readme/epics pointers
- [x] Add `aimusic/audio` stub + `[audio]` extra placeholder + `tests/test_architecture.py`
- [x] Port `aimusic/core/render_package.py` + thin `aimusic/render/package.py`
- [x] Wire `handle_generate` to emit RenderPackage (keep legacy flats)
- [x] Add `tests/test_render_package.py`; run lint/typecheck/tests
- [x] Push branch; `gh pr create` → fork `main`; monitor CI (all green)
