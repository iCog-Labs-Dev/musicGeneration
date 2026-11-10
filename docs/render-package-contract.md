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
## Integrity verification (v1 identity unchanged)

`load_render_package` recomputes both SHA-256 aggregates before returning:

- Source: UTF-8 `json.dumps(score_payload, sort_keys=True)` + original MIDI
  bytes + UTF-8 decimal EDO.
- Package: the unprefixed source digest text + structure and tuning serialized
  with `indent=2, sort_keys=True` + exact `beatstates.jsonl` bytes.

The manifest's `source_hash` and `structure.source_hash` (`sha256:` prefixed)
must agree. `package_hash` and its `content_hash` alias must agree when both
are present; either package alias alone is accepted. Missing legacy beatstates
means empty bytes **only if the aggregate still matches**. A directory name is
not evidence of integrity. Reused directories are verified, never repaired or
overwritten on mismatch.

New manifests add `render_package.file_digests`: unprefixed SHA-256 digests of
all five payload files, including empty beatstates. These check exact file bytes
before JSON parsing. The mutable manifest is deliberately **not self-hashed**.
Existing aggregate-only packages need no rewrite and retain their identities.

Diagnostics distinguish `score.mid digest mismatch: expected=…, actual=…`
from `source aggregate (score.json, score.mid, structure.edo)` or
`package aggregate (source digest, structure.json, tuning.json, beatstates.jsonl)`
failures. Aggregate-only evidence cannot identify one corrupt file. Restore or
regenerate the affected source package; editing the manifest to silence errors
is not a repair. These hashes detect corruption, not malicious rewriting of
both payloads and declarations.

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

**Always** emit a RenderPackage from product `generate` (cheap; every symbolic run
is audio-ready). Legacy flat artifacts (`{run_id}_score.json`, `.mid`,
`_manifest.json`) remain for Gradio/UI compatibility.

Prefer `provenance: planner` via `build_structure(score, beatstate_path, …)`.
Package audio validation reconciles the original MIDI before groove or cache
reuse. `AudioPackageContext` carries parsed tuning, verified source/package
identity, planner roles (`structure.json`) and counts computed from `score.json`.
Inference never replaces these facts. Native orchestration accepts the context
and supplies authoritative analysis to downstream stages even when the analysis
stage is disabled. Imported MIDI without a package remains a separate inferred
path; it is not evidence of planner agreement.

## Source-MIDI reconciliation

`aimusic.audio.analysis.reconcile` uses mido track chunks, not PrettyMIDI's
per-channel instrument count. One uniquely named source track may own multiple
MPE channels. Duplicate/missing track identities, ambiguous note pairing and
cross-track note-channel ownership fail explicitly. `comp` normalizes to
`comping`; name-based role guesses are informational, or `unavailable` when no
role can be inferred. They do not veto the authoritative role.

- Exact: source-track names, note counts, PPQ, programs, drum metadata, meter
  signatures, effective map counts and bar indices. Drum keys are not pitches;
  percussion must use channel 9 (zero-based) and no pitched program declaration.
- Tempo: **0.001 BPM**, inclusive, at every effective change point. MIDI's
  default 120 BPM / 4/4 applies when the corresponding event is absent.
- Boundaries: **one source MIDI tick** at the local tempo, inclusive, for map
  changes, note onset/release pairs, bar times, Score occupancy and EOT.
  Meter rows use start-bar indices; tempo rows and bar times use seconds.
- Score occupancy is the last note release, not the final planner bar. Trailing
  planned silence is allowed. The current Score exporter ends each note track
  at its own last note, so its EOT is checked against that bound, not the planned
  section end. Conductor events may extend the overall MIDI horizon.

Successful full native runs write `score.reconciliation.json` with field,
expected/observed values, tolerance and outcome, plus `score.analysis.json`
containing tuning, hashes and source-track/channel associations. Failed field
comparisons raise `ReconciliationError` with the same structured report on
`.report`; no source files are rewritten. Validate-only creates no WAVs and
imports no synth, PrettyMIDI or soundfile. Reconciliation is not a pitch-fidelity
or audio-quality certificate; event transformations and tuning evidence are
separate checks.

## Tests

- `tests/test_render_package.py` — write/load/invariants + CLI generate emits package
- `tests/test_audio_reconcile.py` — field reports, tolerances, MPE source identity
- `tests/test_audio_entry.py` — context handoff, validation-first entry
- `tests/test_architecture.py` — import quarantine
