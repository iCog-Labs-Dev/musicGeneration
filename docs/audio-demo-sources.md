# Four reproducible native listening demos (v1)

**Local native reproduction is complete** for all four sources below, using the
approved GeneralUser GS 2.0.3 / FluidSynth 2.6.2 / pyfluidsynth 1.3.4 lane.
[Machine-readable evidence](audio-native-listening-evidence.json) and
[commands/results](audio-testing.md#four-native-listening-workflows-103--104--112)
cover OpenSpec **10.3, 10.4 and 11.2**. The user subsequently
[accepted the four rough demos](audio-listening-review.md) on 2026-11-02.
[Final integration review (11.4) is complete](audio-testing.md#final-acceptance-review-114):
**42/42** tasks, including **7.5 / 9.4 / 11.3**.
The separately named timing pair was reviewed with no perceived difference and
“No defects heard”; no aesthetic groove improvement or mastered-quality claim is made.

## Exactly four sources

| Source | Versioned producer / seed | Musical intent (all 480 PPQN, 120 BPM, 4/4, SMF type 1) |
|---|---|---|
| Eight-bar 12-EDO fixture | `aimusic.audio.demo_sources fixture-12`, seed **11** | 16 s; drums/bass/comping/lead; 96/16/48/64 notes. Zero-based programs 0/33/0/80; channels 9/0/1/2. C–F–G–C twice; deterministic lead velocity variation. |
| Fresh 12-EDO passage | Existing **Method A planner + decoder** CLI, seed **11**, **4 beats**, default MAP path | 2 s; 28 notes; bass/comping/drums/lead = 4/12/8/4 notes. Programs 33/4/0/81; channels 1/2/9/3. Retains the generated RenderPackage, not a randomized fixture substitute. |
| Controlled 19-EDO reference tones | `demo_sources reference-19`, fixed targets, no RNG | A4 + **1, 4, 7** steps of 19-EDO; one logical track, channel 1, program 80. One-second tones at 0/1.5/3 s, 0.5 s gaps, 4.5 s file horizon. |
| Short 19-EDO multitrack MPE passage | `demo_sources multitrack-19`, fixed targets, no RNG | Two bars / 4 s; bass (2 notes, channel 1, program 33), three-voice comping (6 notes, channels 2–4, program 0), lead (8 notes, channel 5, program 80). Polyphonic audio pitch accuracy **unmeasured**. |

The three fixture definitions are original source code, revision
`aimusic.listening-sources/1`. Their `source.mid`, `expected.json`, and
`source.json` are reproducible bytes in the pinned runtime. JSON uses sorted keys,
compact separators, ASCII, finite numbers and a trailing newline; the manifest
hashes MIDI and expected intent. It contains no timestamps or output paths. Only
`fixture-12` consumes explicitly threaded immutable `RNGKey` streams; changing its
seed changes velocities, not form/pitches. Fixed 19-EDO sources record seed `null`.

The fresh source uses the **existing** core generator and its explicit RNG
threading. Its MIDI and Score bytes repeat for the same seed/runtime. Existing
core run IDs, directory names and run manifests include execution identity; do
not claim those are byte-identical or replace them to fake determinism. Preserve
all package metadata. Fixture changes require a source revision bump and review
of intent/expected values; no checked-in binary MIDI is needed.

## Installation and runtime

Python >=3.11; from the repository, in an activated virtual environment:

```bash
python -m pip install -e '.[audio,dev]'
```

Verified locally: Python **3.12.12**, macOS arm64, NumPy **2.5.4**, mido **1.3.3**,
PrettyMIDI **0.2.11.post0**, SoundFile **0.14.0**, PyYAML **6.0.3**. Record your exact
`python -VV`, `python -m pip freeze`, OS/architecture, git commit and commands
alongside local outputs. No GPU, service key, model, network call or licensed asset
is needed after installation. Source generation is near-instant; the documented
short source/preview sequence took seconds locally. The **~60-second eight-bar
native target is measured separately** in [audio-testing.md](audio-testing.md),
not inferred from simple preview timing.

Bindings (`.[audio-render]`, pyfluidsynth 1.3.4), the **native FluidSynth library**,
and a user-selected **license-approved SoundFont** are three separate prerequisites.
See [Windows/Linux installation and license record requirements](audio-fluidsynth.md).
Do not download an arbitrary SoundFont. Record its filename, supplier/source URL,
license/redistribution terms, SHA-256, native/binding versions and hardware before
native acceptance. The configured native synth disables chorus/reverb for dry tests.

## Generate and preview locally

POSIX shell (bash/zsh; on Windows use WSL or translate shell variables to PowerShell).
Run from the checkout with the activated environment. `set -e` plus a **fresh**
directory guards the existing core generator against accidental reruns. New fixture
publication also refuses every existing target (including dangling links), never
opens a destination for overwrite, and rolls back its own files on errors. Empty
folders can remain on failure; process-crash leftovers must be inspected, not reused.

```bash
set -eu
export AIMUSIC_AUDIO_BACKEND=native   # application backend; NOT the instrument renderer
unset AIMUSIC_SOUNDFONT
mkdir -p outputs
DEMO="$(mktemp -d "$PWD/outputs/listening-v1.XXXXXX")"
export DEMO

python -m aimusic.audio.demo_sources fixture-12 --seed 11 --out "$DEMO/fixture-12"
mkdir "$DEMO/generated-12"           # deliberately no -p; refuse an existing generation destination
python -m aimusic.app.cli generate --seed 11 --beats 4 --edo 12 --tempo-bpm 120 --meter 4/4 --out "$DEMO/generated-12"
python -m aimusic.audio.demo_sources reference-19 --out "$DEMO/reference-19"
python -m aimusic.audio.demo_sources multitrack-19 --out "$DEMO/multitrack-19"

python -m aimusic.app.cli render-midi "$DEMO/fixture-12/source.mid" --profile preview --out "$DEMO/fixture-preview" --no-validate-only
python -m aimusic.app.cli render-midi "$DEMO"/generated-12/run_*/score.mid --profile preview --out "$DEMO/generated-preview" --no-validate-only
```

Each preview produces copied `source.mid`, expressive MIDI/diff, inferred analysis,
reconciliation marked **not applicable**, `render.json` (effective config,
application and instrument backends, versions/code and asset hashes), logical
stem MIDI/WAVs/renderer sidecars, role index, click, and `stems/rough_mix.wav`.
The click is excluded from the mix. No SoundFont means an explicit **null** asset
hash, not missing provenance. Default expressivization has no nontrivial groove
spec; it must not be described as audible groove improvement.

For the fresh passage's authoritative planner reconciliation, use the retained
package instead (this is the **same** source, not a fifth example):

```bash
python -m aimusic.app.cli render-audio "$DEMO"/generated-12/run_* --profile preview --out "$DEMO/generated-package-preview" --no-validate-only
```

An imported MIDI run does **not** ingest `expected.json` or automatically issue a
lattice/f0 certificate. Keep the fixture sidecars with the source directory. The
existing `preview` and `listening` profiles are versioned packaged resources.

### Expected 19-EDO refusal, not preview audio

```bash
for source in reference-19 multitrack-19; do
  if python -m aimusic.app.cli render-midi "$DEMO/$source/source.mid" --profile preview --out "$DEMO/$source-preview" --no-validate-only; then
    echo 'ERROR: microtonal preview unexpectedly succeeded' >&2
    exit 1
  fi
done
```

Expect `simple does not support tuning channel_pitch`, no successful `render.json`
or WAVs. Validation-only can admit the source without allocating a renderer:

```bash
python -m aimusic.app.cli render-midi "$DEMO/reference-19/source.mid" --profile listening
```

### Listening commands (approved native prerequisites required)

After explicit approval and installation, set `SF` to the approved local file;
**do not** execute with a placeholder. SoundFont selection is CLI > profile/config
> environment, and listening never falls back to simple.

```bash
SF=/absolute/path/to/license-approved.sf2
python -m aimusic.app.cli render-midi "$DEMO/fixture-12/source.mid" --profile listening --soundfont "$SF" --out "$DEMO/fixture-listening" --no-validate-only
python -m aimusic.app.cli render-audio "$DEMO"/generated-12/run_* --profile listening --soundfont "$SF" --out "$DEMO/generated-listening" --no-validate-only
# These require the proven native 2.6.2 channel_pitch subset:
python -m aimusic.app.cli render-midi "$DEMO/reference-19/source.mid" --profile listening --soundfont "$SF" --out "$DEMO/reference-listening" --no-validate-only
python -m aimusic.app.cli render-midi "$DEMO/multitrack-19/source.mid" --profile listening --soundfont "$SF" --out "$DEMO/multitrack-listening" --no-validate-only
```

All four have now been reproduced locally; missing resources and unsupported
tuning still fail. The opt-in integration test retains each source, three runs
(cold, independent cold-repeat, warm cache), effective configuration, asset/backend
provenance and signal checks. The generated passage uses its **RenderPackage**,
not the inferred MIDI route. Reference f0 is checked against independent original
Hz; multitrack f0 is **unmeasured**, with event pitch integrity <=1 cent. The test
also retains separate reference groove-on/off and rejected flattened comparisons.
These measured native results are separate from the later
[human rough-demo acceptance](audio-listening-review.md).

For the single-page embedded player command (all four mixes; no relative WAV
URLs), see [self-contained listening review](audio-listening-review.md#reproduce-the-portable-page-no-native-dependency-needed).

## Supported source-event subset and independent expectations

“MPE” here means independent **per-channel pitch wheels**, not full MPE zones.
Every melodic channel gets program setup then RPN **0** (CC101=0, CC100=0), Data
Entry **2 semitones / 0 cents** (CC6=2, CC38=0), then RPN null (CC101=127,
CC100=127). Before each note-on, set wheel to
`round((69 + step*12/19 - nearest_midi_key) / 2 * 8192)`. Each channel has at most
one sounding note, with gaps before reuse; no sustain, zone/master expression,
NRPN, relative Data Entry, fine/coarse tuning, SysEx or MTS. Comping's three member
channels remain **one logical source-track stem**. Admission is `channel_pitch`,
not zone `mpe` and not `direct_12`. **MTS remains unsupported** and is rejection-tested.

Expected Hz is **440 × 2^(step/19)** from original intended integer steps, never
from rendered MIDI, wheel values, measured audio or nearest-lattice inference.
Reference `expected.json` contains explicit half-open sample windows at **48 kHz**:
`[12000,36000)`, `[84000,108000)`, `[156000,180000)`. They exclude note attacks and
releases, assuming original timing. They are not portable to a shifted groove or
another sample rate without independently recomputing window positions from the
source timing transformation; do not move pitch targets to match bad output.
The multitrack sidecar deliberately has **no monophonic windows**.

Once a native **dry original-timing** reference stem exists, run the standalone
checker against original targets (this does not implement native capability or
wire the checker into orchestrator publication):

```bash
python - "$DEMO/reference-19/expected.json" "$DEMO/reference-listening/stems" "$DEMO/reference-tuning.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from aimusic.audio.scoring.tuning_check import ToneWindow, analyze_tuning
expected = json.loads(Path(sys.argv[1]).read_text())
stems = Path(sys.argv[2])
index = json.loads((stems / 'role_index.json').read_text())
assert len(index['stems']) == 1
pcm, rate = sf.read(stems / index['stems'][0]['wav'], always_2d=True)
assert rate == expected['sample_rate']
report = analyze_tuning(np.mean(pcm, axis=1), rate,
                        tuple(ToneWindow(**row) for row in expected['windows']))
with Path(sys.argv[3]).open('x') as stream:
    json.dump(report.to_dict(), stream, sort_keys=True, allow_nan=False, indent=2)
report.require_pass()
PY
```

Use the **role-index filename** for the reference stem; no general polyphonic
pitch claim follows from this check. Repeat with independently shifted windows
for a controlled groove-on run, and with deliberately flattened audio using the
**same original Hz targets** (must fail).

## Listening checklist and floors (not mastered quality)

- Verify source/sidecar SHA-256, seed, code/runtime versions and effective routing;
  planner counts/programs exact, tempo within **0.001 BPM**, boundaries within
  **one source tick** where planner reconciliation applies.
- Source-event pitch after groove/serialization/isolation: **<=1 cent**; preserve
  counts, channels, program, tempo/meter and source-track identities.
- Stems and mix: configured **48 kHz**, equal frame counts including **0.25 s**
  release tail, finite samples, active-region **RMS >1e-5**, peak **<=0.95**.
- Controlled mono tones: **100% window coverage**, **<=5 cents every tone**;
  mean and P95 **<=15 cents**, stricter settings allowed. Silence/low-confidence
  windows fail. Polyphonic f0 remains **unmeasured**.
- Listen quietly to each stem, then mix: bass/comping/lead recognizable with the
  approved native asset, GM percussion not pitched sine notes, no clipping,
  truncated/stuck notes, lost voices or gross timing errors. Compare groove on/off
  at matched gain; record reviewer/date/findings and any asset-specific failure.
- Preview is additive sine audio, **not instrument recognizability**. Full rhythm
  F-score **>=0.85** and notes F1 **>=0.70** are future scorer floors, **not measured
  here**. No CLAP/FAD, mastering/LUFS or production quality is claimed; the
  later human acceptance is bounded as recorded in [the listening review](audio-listening-review.md).

## Local verification / artifact policy

```bash
python -m pytest tests/test_audio_demo_sources.py tests/test_audio_cli.py tests/test_audio_wheel.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
openspec validate deliver-midi2audio-audible-demo --strict
```

Wheel smoke builds/installs offline, excludes checkout/PYTHONPATH, executes the
installed fixture module and preview CLI. Tests cover all four sources, byte
repeats, canonical sidecars/digests, independent pitch math, safe refusal,
serialization/groove/isolation, readable preview audio and microtonal/MTS rejection.

Keep generated MIDI/WAVs, full reports, caches, runtime freeze files and all licensed
assets **local/untracked**. Only the compact, path-relative measured JSON summary
is versioned; it links the full local evidence by SHA-256. Use `outputs/` (already ignored) or an external scratch
directory. SoundFonts live outside the repository. Never commit credentials,
weights, WAVs or vendor assets; never add broad ignores for source fixtures or
OpenSpec. `.gitignore` needs **no change**. Check `git status` before staging;
stage source/test/doc paths explicitly. No upload or Linear update is authorized.

### Recorded asset-free evidence (2026-11-02)

- Focused sources/CLI/wheel/architecture/contract: **36 passed**, 44 subtests.
- All lightweight audio + architecture + contract: **340 passed**, 642 subtests,
  **1 skipped** (optional bridge, not native evidence).
- Full unittest: **673 run**, **9 optional skips**, no failures (79.892 s).
- Flake8 clean; mypy clean (133 files, pre-existing untyped-body note); strict
  OpenSpec validation and `git diff --check` clean. Wheel builds/installs offline
  and runs the installed module and preview CLI outside the checkout.
- TDD first failed on the absent module; a later red test caught incompatible
  window field names before aligning the sidecar with `ToneWindow`. Inverting
  the direct-12/MPE encoding branch was killed by the event-integrity regression;
  the original was restored before the full checks.

Actual CLI preview (same pinned runtime above), **four stems each**, every stem
finite/readable/aligned and active RMS >1e-5, all peaks <=0.95:

| Source | Frames at 48 kHz, including tail | Mix RMS (0.1–0.4 s) | Mix peak |
|---|---:|---:|---:|
| Eight-bar fixture | 780000 | 0.182776 | 0.823289 |
| Fresh Method A passage | 108000 | 0.165412 | 0.719104 |

Source MIDI SHA-256 (fixture seed 11; no native/audio-f0 certification):

```text
fixture-12     e97c8db92fc495ff79810a76af38c0aec3d28ac901580462c69a6e744c8f215f
generated-12   a607e8c6ec3daeb2687056b7f150aacd1f85f192946584ac8078557613ea6f41
reference-19   0b7f7debe5b2c95d360cc25247534293be1c361f193442cf56c7f733ade23db6
multitrack-19  6e76a9b1b7dde2ee97455c6d5044dc94ec2a94be65f9587566e2444b26a29fe8
```

Local logs: `demo-sources-red.log`, `demo-sources-mutation.log`,
`demo-sources-audio.log`, `demo-sources-full.log` in the session's approved scratch
parent. Preview measurements are in `listening-sources.zR6Ojr/evidence.json` there.
These historical outputs are **simple** only and did not complete native tasks.
The later native summary above supersedes the former four-workflow blocker;
the later [human review](audio-listening-review.md) supports proposing MUS-34
completion. MUS-35 is still partial, and tickets remain unchanged.
