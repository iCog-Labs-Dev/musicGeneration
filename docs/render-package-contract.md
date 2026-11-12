# RenderPackage contract (`aimusic.structure/1`)

Sole boundary between symbolic `aimusic` and audio (`m2a` / `aimusic.audio`).
Producer: `aimusic.core.render_package` (re-exported from `aimusic.render.package`).
Design source: `AUDIO_PIPELINE_ARCHITECTURE.pdf` §3.

## Directory layout

```text
run_<runId8>_<packageHash12>/
├── score.mid           # required
├── score.json          # required (Score serialization)
├── structure.json      # required — schema aimusic.structure/1
├── tuning.json         # required
├── manifest.json       # required (RunManifest / provenance)
└── beatstates.jsonl    # optional (one BeatState JSON per line)
```

`source_hash` digests score + MIDI + EDO only (stored on `structure.source_hash`).
`package_hash` digests `source_hash` + canonical structure + tuning + beatstates
and is the directory / `RenderPackage.content_hash` identity. Manifest
`render_package` records both hashes. Re-writing the same package is idempotent;
a conflicting path raises `ContractViolation` (no destructive overwrite).
## `structure.json` (schema v1)

Key fields: `schema`, `provenance` (`planner` | `inferred` | `host`), `source_hash`,
`edo`, `base_tuning`, `tempo_map`, `meter`, `key`, `chords`, `sections` (with
`boundary_lvl`), `bar_table`, `tracks` (name/role/program/`microtonal`/
`onset_profile_16`).

Default programs: bass 33, comping 4, lead 81, drums `null`.

A track is `microtonal: true` when any pitch class is more than **8 cents** from
the nearest 12-TET pitch (see `DECISIONS.md`). Percussion tracks
(`percussion: true`, e.g. drums / GM drum keys) are always `microtonal: false`
— drum note numbers are sample indices, not EDO pitch heights.

## `tuning.json`

EDO size, base tuning, pitch-bend range, global `method`, and per-track methods.
Microtonal tracks must never use `method: direct_12`.

## Invariants (`assert_contract_invariants`)

1. Every `NoteEvent.track` appears exactly once in `structure.tracks`.
2. `bar_table` is monotonic; `tempo_map` and `meter` are non-empty.
3. `tuning.json.edo == structure.edo`.
4. `score.mid` exists and is non-empty.
5. Microtonal tracks are not routed as `direct_12`.

Violations raise `ContractViolation` and abort the run.

## Emission policy

### Supplied renderer requirement

REQ-10 workflows receive a single `ScoreRenderer(Score, RenderRequest) -> RenderResult` selected by the shared application factory. The request and result are format-neutral. The concrete MIDI adapter owns tuning and instrument settings and delegates to the established `render_midi` function, whose direct signature and None return are unchanged.

CLI packaging and UI preview still request valid MIDI. The neutral adapter supplies a fresh sibling temporary path, requires a regular completed file and matching path/format/byte-size report, then atomically replaces the final destination. Missing files, non-regular files, incorrect reports and renderer exceptions prevent a new success manifest/package and preserve any previous final MIDI. Already-written score JSON and unrelated artifacts are not rolled back.

CLI and UI manifests add `config.components` with effective factory selections, implementation identities, explicitly public options and known MIDI construction settings. Constructor options are private by default; `public_options` names shareable values and the rest contribute only an unpublished-option count. Runtime overrides have honest unknown-options markers. Built-in MIDI renderers must match the run's settings, including through our artifact wrappers. The top-level manifest/package layout is unchanged.

The CLI owns score JSON, manifest, structure, tuning and package emission. The UI helper emits flat score/MIDI/manifest files for preview/download and does not create a RenderPackage. See the [component extension guide](extending-components.md).

### Product output

**Always** emit a RenderPackage from product `generate` (cheap; every symbolic run
is audio-ready). Legacy flat artifacts (`{run_id}_score.json`, `.mid`,
`_manifest.json`) remain for Gradio/UI compatibility.

Prefer `provenance: planner` via `build_structure(score, beatstate_path, …)`.
Audio may fill `inferred` / `host` only when planner structure is missing; then
`reconcile` against MIDI analysis in CI fixtures (M1+).

## Tests

- `tests/test_render_package.py` — write/load/invariants + CLI generate emits package
- `tests/test_architecture.py` — import quarantine
- `tests/test_artifact_injection.py` — supplied decoder/renderer output reaches real MIDI packaging
