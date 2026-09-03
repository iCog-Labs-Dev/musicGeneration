# Audio config skeleton

Runtime defaults live in [`config/audio.default.yaml`](../config/audio.default.yaml)
(PDF Appendix B). Profiles (future): `config/profiles/{funk,prog,jazz_fusion,single_ref,microtonal}.yaml`.

Key groups:

- `edo` — `tolerance_cents`, `allow_microtonal_diffusion`
- `groove` — author mode, tension coupling, MIDI-GPT/GrooVAE flags (off)
- `render` — sample rate, routing table, CI-safe `simple` backend
- `restyle` — endpoints, strength bases, chunk seams
- `scoring` — weights + hard floors
- `search` — strength grid, successive halving
- `budget` — max calls / USD
- `mixmaster` — LUFS targets, cover blend
- `stages` — enable flags (deterministic spine on by default; restyle off)
- `paths` — cache/output/soundfont/refs

M0 does not load this config from the symbolic CLI; it seeds the future
`aimusic.audio` orchestrator.
