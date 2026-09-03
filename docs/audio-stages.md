# Audio stages (STAGE 1–6)

Design digest from `AUDIO_PIPELINE_ARCHITECTURE.pdf` §4. Implementation lives
under `aimusic.audio` after M1+ (see [audio-roadmap.md](audio-roadmap.md)).

## STAGE 1 — Symbolic expressivization

Generated MIDI is grid-quantized with near-flat velocities. STAGE 1 is the
difference between a mock-up and music.

- Authored `GrooveSpec` (pydantic) + measured templates from references
- Pure applicator: `(Score|MIDI, GrooveSpec, rng_key) -> (MIDI, DiffReport)`
- Extensions: track-name keys, `section_overrides`, tension coupling to SB curve,
  `edo_safe` for microtonal tracks, `groove_id_hint` from planner

## STAGE 2 — Deterministic stem rendering

Routing table keyed by `(role, microtonal, policy)`:

| Role | 12-EDO default | Microtonal |
|------|----------------|------------|
| drums | fluidsynth / SFZ | same |
| bass | sfizz | sfizz + `.scl` / MPE |
| comping | sfizz / Pianoteq | Pianoteq / Surge + `.scl` |
| lead | sfizz / MIDI-DDSP | MIDI-DDSP (f0-driven) |

Also emit click, rough mix, and `bar_table` timestamps (from structure).
Run `tuning_check` on **dry** microtonal stems before spending endpoint calls.

## STAGE 3 — Generative restyle

- Chunk on descending `boundary_lvl` (2-bar overlap, equal-power crossfade on downbeats)
- Strength schedule by role / tension / microtonal penalty
- Hard rule: microtonal + diffusion ⇒ skip unless `allow_microtonal_diffusion`

## STAGE 4 — Prompt synthesis

Seed prompts with planner facts from `structure.json` (tempo, meter including
odd meters, key, section, role). Explicit meter + negative `four-on-the-floor`
for non-4/4 prog sections.

## STAGE 5 — Scoring, selection, search

Axes: rhythmic adherence, note fidelity, **tuning preservation** (new),
CLAP prompt match, FAD / reference proximity. Hard floors before weighted
objective. Search: coarse strength grid → successive halving; budget guard
before every paid call.

## STAGE 6 — Mix and master

LUFS by role, cross-correlation align, static mix, optional cover blend,
Matchering, final audit (rhythm + tuning). A master that drifts to 12-TET is a
**failed** run.

## CLI surface (target)

```bash
python -m aimusic.app.cli generate --seed 11 --beats 256 --edo 19 --out ./outputs
python -m aimusic.app.cli render-audio ./outputs/run_<hash>/ --profile config/profiles/prog.yaml
python -m aimusic.app.cli expressivize ./outputs/run_<hash>/ --groove-spec ./specs/funk.json
python -m aimusic.app.cli audit ./outputs/audio/run_<hash>/
```

Audio subcommands must **lazy-import** `aimusic.audio` (never static from core).
