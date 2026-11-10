# Imported MIDI rendering

For the versioned four-source reproduction recipe and bounded native acceptance,
see [listening demo sources](audio-demo-sources.md).

`render-midi` accepts local type-0/type-1 PPQN MIDI without inventing a Score or
planner metadata. Both audio commands **validate only by default**; use
`--no-validate-only` to create audio. Install Python dependencies with
`pip install -e '.[audio]'`. Preview requires no FluidSynth, SoundFont, keys or
network service.

```bash
python -m aimusic.app.cli render-midi song.mid --profile preview
python -m aimusic.app.cli render-midi song.mid --profile preview --out ./audio-preview --no-validate-only
python -m aimusic.app.cli render-audio outputs/run_<hash> --profile preview --output-dir ./package-preview --no-validate-only
```

- `--out` / `--output-dir` selects the **exact** destination, not a parent to which
  a run ID is appended. Without it, existing `render-audio` behavior remains
  `<configured output_dir>/<run_id>`; imports use `imported_<source SHA-256 prefix>`.
- `--profile` accepts packaged `preview` / `listening` or a YAML path. Use
  `./preview` to explicitly select a file named `preview` rather than the profile.
  Malformed YAML reports the config/profile path (or packaged resource label),
  without an uncaught parser traceback.
- `--soundfont` overrides profile/config `paths.soundfont`, then
  `AIMUSIC_SOUNDFONT`. Relative config assets are anchored to the config file.
  A bad higher-precedence selection fails; it never falls through to another.
- Validation checks input/config only, writes nothing, and does not load the
  synthesizer or probe native assets. Renderer/asset checks occur on full render.
- `listening` requires the separately installed bindings/native library and a
  locally supplied, license-approved SoundFont; it never falls back to preview:

```bash
python -m aimusic.app.cli render-midi song.mid --profile listening --soundfont /path/to/approved.sf2 --out ./audio-listening --no-validate-only
```

See [FluidSynth setup and evidence limits](audio-fluidsynth.md). The four-demo
[native listening acceptance](audio-listening-review.md) does not certify arbitrary
imports; this command or the preview tests alone cannot establish it.
Imported context is supported by the application backend `native` only (unset
`AIMUSIC_AUDIO_BACKEND` or set it to `native`); the instrument renderer can still
be `simple`. Existing explicit package-to-`m2a` dispatch remains supported.

## Evidence and safety

Full imported rendering publishes a byte-identical `source.mid`, optional
`<source>.expressive.mid` plus groove sidecars, `<source>.analysis.json`,
`<source>.reconciliation.json`, `render.json`, and `stems/` containing named
source-track MIDI/WAVs, renderer sidecars, `role_index.json`, `click.wav`,
`bar_boundaries.json`, and **`rough_mix.wav`**. The click is excluded from the mix.

The inferred context records the source hash/type, tempo and meter changes,
source-track/channel identities, effective programs and program changes, note
counts, inferred roles, and raw pitch-wheel/RPN/SysEx tuning evidence. Absent
initial tempo/meter use MIDI defaults (120 BPM, 4/4); meter changes start a new
inferred bar, even mid-bar. Type-0 remains one source-track stem; multiple channels
are preserved, and heterogeneous roles are labeled `mixed`, not split by guesswork.
Planner reconciliation is explicitly **not applicable**, never a passing report.
The intended tuning lattice is unknown and audio pitch accuracy is unmeasured.

Malformed/type-2/SMPTE/multiple-port MIDI, invalid note pairing, unreleased sustain,
and conflicting cross-track channel ownership fail clearly. Unsupported tuning
is retained as evidence during validation but rejected by renderer capability
checks before successful publication. No simple fallback flattens bends/MTS.
Source changes after admission fail even with a warm cache. Track names are not
paths: duplicate/traversal/reserved names get stable safe identity-based names.
Existing artifacts and stem-directory symlinks are refused, not overwritten.
Use a fresh output directory for each publication. The package-to-`m2a` bridge
also preflights `score.mid`/`structure.json`, rejects source-directory aliases,
runs against a private package snapshot, and exclusively publishes the complete
output bundle with rollback on failure (empty output directories may remain).

Imported meter expansion is limited to **100,000 bar boundaries**, cumulatively
across meter changes. Exact integer admission checks run before expansion, so an
extreme MIDI denominator cannot trigger an effectively unbounded loop. This is a
bound on derived analysis, not on PPQN or the denominator alone: high-resolution
and short fine-meter inputs remain valid. Shorten the excerpt or use a smaller
meter denominator if the limit is exceeded.

Simple preview produces additive sine audio with source event timing, including
non-track-zero tempo changes. It is **not** recognizable instrument rendering,
controller fidelity, tuning measurement, or mastered/CLAP/FAD-scored audio.

## Recorded local preview check (2026-11-02)

From `$SCRATCH/imported-cli.8drK4f` (`SCRATCH` is the caller-owned external
scratch parent), using the existing `musicgeneration-venv` (Python 3.12.12),
a mido type-1 source contained a conductor (480 PPQN, 120 BPM, 4/4), bass (program 33/channel 0/note 45),
and lead (program 80/channel 1/note 69), both velocity 100 and ticks 0–960.

```bash
AIMUSIC_AUDIO_BACKEND=native python -m aimusic.app.cli render-midi source.mid --profile preview --out audio --no-validate-only
```

Exit **0**; source unchanged; source SHA-256
`4627813ab8b3056b713e70711dbbe73c0a0a4fa0b9f9a4fe0e7e9c557761d25f`.
SoundFile read all WAVs successfully. Each source stem and mix: **48 kHz,
60,000 frames, 240,080 bytes** (1 second plus 0.25 second release horizon).

| Artifact under `audio/stems/` | RMS | Peak |
|---|---:|---:|
| `track_0001_bass.wav` | 0.102321 | 0.251189 |
| `track_0002_lead.wav` | 0.102194 | 0.251189 |
| `rough_mix.wav` | 0.144614 | 0.483803 |

Exact hashes/metrics are local in `wav-evidence.json`. No native assets were
used. WAVs remain untracked; no human listening/native acceptance is claimed.
OpenSpec **10.3–10.4** now have [four-example native evidence](audio-demo-sources.md);
the later [human rough-demo and timing acceptance](audio-listening-review.md)
completes task 11.3 without claiming mastered quality. [Final review (11.4)](audio-testing.md#final-acceptance-review-114)
now completes the change at **42/42** tasks.

### Verification

Commands in the existing asset-free audio/dev environment:

```bash
python -m pytest tests/test_audio_imported.py tests/test_audio_cli.py tests/test_audio_entry.py tests/test_architecture.py tests/test_render_package.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
```

Focused: **43 passed**, 30 subtests. Lightweight/architecture/contract (including
wheel smoke): **319 passed**, 596 subtests, **1 optional bridge skip**. Full
unittest: **652 run**, **9 optional skips**, no failures (69.865 s). Lint and mypy
clean (131 files; existing untyped-body note retained). Skips are not native
acceptance evidence.

TDD red runs established missing imported context/CLI and caught incorrect
preview timing for non-track-zero tempo changes. Reversing the new stale-source
identity gate failed its regression; original code was restored before final
checks. Tests cover malformed/type-2/unsafe channel inputs, default validation,
source-copy identity, symlink/collision safety, readable nonempty full-render
WAVs, both package/import options, missing optional dependencies and mocked
listening prerequisites. Logs: `imported-context-red.log`,
`imported-render-red.log`, `imported-cli-red.log`, `imported-mutation.log`,
`imported-audio.log`, `imported-full.log` under the approved scratch parent.
