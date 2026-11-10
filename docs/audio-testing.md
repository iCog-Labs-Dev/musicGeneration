# Audio pipeline testing and risks

From `AUDIO_PIPELINE_ARCHITECTURE.pdf` §§8–9.

**Current:** all four pinned native workflows are locally reproduced; see
[10.3 / 10.4 / 11.2 evidence](#four-native-listening-workflows-103--104--112).
The later [human rough-demo acceptance and portable report](audio-listening-review.md)
complete **7.5 / 9.4 / 11.3**. [Approved final review (11.4)](#final-acceptance-review-114)
completes the change at **42/42** tasks. The timing-pair review
records no perceived difference and “No defects heard”, not aesthetic groove
improvement. Earlier checkpoints below are historical measurements, not current
blockers or human acceptance.

## Testing strategy

### Verification lanes

1. **Core / architecture (required):** run the symbolic suite, contract tests,
   lint, and type checks with no audio-native dependencies loaded.
2. **Lightweight audio (required):** run config, MIDI-event, reconciliation,
   cache, safety, and mocked-renderer tests with the CI-safe `simple` backend.
   This lane requires no FluidSynth installation, SoundFont, GPU, credentials,
   or network service.
3. **Real FluidSynth (explicit opt-in):** set
   `AIMUSIC_RUN_FLUIDSYNTH_TESTS=1` and provide the approved SoundFont path and
   SHA-256 via the manifest contract below. Once explicitly selected, missing
   bindings/native library/SoundFont or provenance are setup failures, not skips. This lane records native
   versions, asset license/hash, signal metrics, tuning evidence, and elapsed
   time.

The real lane must use a locally supplied, license-approved SoundFont selected
by the reviewer. The repository does not download, bundle, or commit one.

| Component | Test approach |
|-----------|---------------|
| `render_package` | Contract invariants on every fixture; schema validation |
| `analysis/reconcile` | Planner vs inferred analysis on own MIDI, per-field tolerances |
| `groove/apply` | Hypothesis: offsets within 1 tick; note-count rules; velocity idempotence |
| Metrics | Monotonicity under synthetic corruption; failing metrics leave the objective |
| `tuning_check` | Pure 19-EDO tones ≈ 0¢; 12-TET flattening exceeds threshold |
| `chunking` | Seams on downbeats; equal-power; unmodified reassembly sample-identical |
| Endpoint wrappers | vcrpy record/replay; never hit paid APIs in default CI |
| Orchestrator | Cache hit ⇒ bit-identical; budget abort reports best-so-far |
| Architecture | Import-direction + core-import-weight tests |
| E2E | 8-bar fixture through M1+M2 in CI, fluidsynth/simple, under ~60s |

Human audit: HTML report per run (stems, spectrograms, metrics, prompts).

## Windows verification: simulation, required native CI, optional renderer

The historical macOS renderer measurements below are **not Windows evidence**.
Hardening adds three distinct forms of Windows verification:

| Lane | Required? | What it establishes |
|---|---|---|
| Host-independent ABI/policy simulation | Yes, existing Linux suite | ctypes widths, flags, failure handling, reparse/identity decisions; not real NTFS behavior |
| `Windows audio (asset-free wheel)` in `ci.yml` | Intended required gate; temporarily skipped | Actual Windows NTFS publication, collision, rollback, Unicode/long paths, simple preview and cold/warm cache; no real FluidSynth audio |
| `windows-audio-evidence.yml` / `-Renderer` | Explicit opt-in only | Pinned Windows FluidSynth canonical workflows, controlled tuning, output digests and same-machine cold/repeat/warm results; no inherited listening approval |

A repository owner must select **Windows audio (asset-free wheel)** as a required
branch-protection check after its temporary deferral is removed. YAML cannot
configure GitHub branch protection. No workflow dispatch or branch-rule change is
performed by these scripts.

> **Temporary deferral (2026-11-10):** fork PR #5 run `42789164898` failed in
> the standard-user `Start-Process` bootstrap before any Windows tests executed.
> The job remains defined but is explicitly skipped in `ci.yml`; this is not
> Windows acceptance evidence. OpenSpec task 8.0a schedules repair and restoration
> of the required gate in a later iteration.

### Required asset-free Windows job

The runner builds a wheel from an external source snapshot and installs
`[audio,dev]` into a fresh venv beneath a generated **space + non-ASCII** path.
It copies the test harness (not `aimusic`) to a separate working directory and
clears editable/PYTHONPATH/native-asset overrides. `tests.windows_ci` proves the
loaded package is inside that venv and outside the checkout before running:

- Package-wide imports, excluding unpackaged Gradio UI and optional ML modules.
  The existing Linux import job still covers UI. No FluidSynth binding may be
  installed in this lane.
- `test_architecture.py` against the **loaded package's actual sources**, with a
  nonempty-source guard, and `test_render_package.py`.
- `test_audio_windows_native.py`: real local NTFS publication and workflow tests,
  plus the host-independent ABI and policy substitutes.

The **actual build and checks require a standard-user/restricted token**, not an
administrator shell. `tests.windows_token` queries the real Windows token before
build and again inside `tests.windows_ci`: elevation must be false, effective
Administrators membership false, integrity at most medium. Token API errors or
optimized Python fail closed. `windows-ci-results.json` records the token facts.
No Developer Mode, FluidSynth, SoundFont, GPU, supplied credentials, asset downloads,
or machine execution-policy changes are needed. Package install may use PyPI;
synth execution and external asset acquisition do not occur.

GitHub `windows-latest` starts as administrator. The separate hosted-only
`test-windows-audio-as-standard-user.ps1` bootstrap uses that privilege **only for
setup/cleanup**: it creates an expiring ephemeral account in the localized Users
group (identified by SID), grants checkout/interpreter/shell read+execute and a
fresh external scratch tree Modify, then launches the unchanged local script with
`Start-Process -Credential`. No fallback executes checks as administrator. The
child gets a fresh environment with explicit external TEMP/TMP; it builds and
installs its own wheel beneath a space+Unicode path. Git trusts only the supplied
read-only checkout via a command-scoped `safe.directory`, never a global wildcard.

The random password exists only as a SecureString passed through the credential
API: never argv, environment, transcript, output, or a credential file. Paths are
JSON/base64 data carried by `-EncodedCommand`, not executable interpolation.
The bootstrap captures the child's checked exit status and logs, disables GitHub
workflow-command interpretation while replaying those logs, removes the account,
and restores read ACLs in `finally`. Timeout/VM termination can bypass cleanup;
account expiry and disposal of the hosted VM are the backstop. This is a token
regression test, **not a sandbox for malicious repository code**. Do not run the
bootstrap on local or asset-bearing self-hosted machines.
Local reproduction from a trusted checkout (Python 3.11 x64, local NTFS):

```powershell
$Python = (Get-Command python).Source  # Or your explicitly selected Python executable.
$Scratch = [System.IO.Path]::GetTempPath()  # Existing external local NTFS directory.
# Per-process script permission only; do not change machine execution policy.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./scripts/test-windows-audio.ps1 -Python $Python -ScratchParent $Scratch
if ($LASTEXITCODE -ne 0) { throw 'Windows wheel verification failed' }
```

GitHub's bootstrap runs the same script with `pwsh` as the standard user. Local
users need no setup account or elevation. Every native subprocess exit is checked,
not only the last command. The fresh external build directory retains
`commands.log`, `build.json` (source revision and wheel digest), and
`checks/windows-ci-results.json`; failures retain diagnostics too. Never add that
directory to Git. Build/install/test logs are not evidence of native renderer
quality or of passing unrelated Linux jobs.

Local remediation verification on **2026-11-05**, Windows build **10.0.26300**,
Python **3.11.14 x64**, `pwsh` **7.6.9**, non-administrator token:

- Exact local wheel command above, using `pwsh.exe`: **67 package modules imported**;
  **56 tests run, 55 passed, 1 narrowly scoped symlink skip** (WinError 1314).
- Token recorded twice: `elevated=false`, `administrator=false`,
  `integrity_rid=8192` (medium). All required publication/orchestrator/cache checks
  passed; `unexpected_skips=[]`, `asset_environment_clean=true`.
- External build identifier: `7b2cdd99ce844e118d0453fce1e2773d`; wheel SHA-256
  `fdad16e3df70e5f7c018689c94b08c51e39e2720eadc692a210a894c0ac86edd`.
- The exact documented Windows PowerShell **5.1** command also passed: **56 tests,
  55 passed, the same one symlink skip**. External build `5a012611c5a749998688818a02fe68e9`,
  wheel SHA-256 `08de1419bf1896082d9eba9e7d34308646807efe179306781cf7fb4c7f9fa9b7`.
- Combined native/architecture/contract/CI run: **72 passed, 1 symlink skip,
  149 subtests**. Final CI/safety regressions (including the added optimized-Python
  guard): **18 passed, 78 subtests**. Full-repository flake8 and mypy (**159 files**)
  and PowerShell parsing passed. In-memory token-gate and
  old-null-removal mutations failed their regression tests; no mutation remains.
- Regression tests exercise absent, empty and nonempty asset overrides, exact skip
  reasons, token rejection, and Unicode/apostrophe/metacharacter argument transport
  with success and failure exit paths on Windows PowerShell 5.1 and `pwsh`.
- **Hosted account creation / alternate-credential launch has not been executed
  locally** (this session has no administrator token). A real `windows-latest` run
  is still required to verify hosted ACL, logon/profile and provisioning behavior.
  No hosted CI, Linux job or FluidSynth rendering success is claimed.

Root cause of the isolated end-to-end failure: current `pwsh` can turn
`[Environment]::SetEnvironmentVariable(name, $null, 'Process')` into a **present
empty string**; Windows PowerShell 5.1 returned an absent key for the same call.
`AIMUSIC_SOUNDFONT=''` is correctly rejected by configuration validation, even in a
simple preview. The harness now uses `Remove-Item Env:...` and the installed runner
requires all four renderer/asset keys to be absent (not merely unequal to `1`).
The helper restores the caller's original absent/empty/nonempty state; an explicit
Win32 empty assignment preserves original empty values even on PowerShell 5.1.
Production configuration validation is unchanged. See the official PowerShell
[environment-variable semantics](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_environment_variables)
and [Start-Process contract](https://learn.microsoft.com/powershell/module/microsoft.powershell.management/start-process).

### Capability probes and mandatory substitutes

Before native tests, `tests.windows_capabilities` separately creates and checks a
hard link, directory junction, and directory symlink. Only WinErrors **5, 50,
1314** from a probe mean unavailable; unexpected failures are fatal. Hard links
are a **required local NTFS capability**, not an optional privileged skip.

Junction/symlink cases skip only their exact test when their probe is unavailable.
After a successful probe, a failed creation during a test is a real failure, not
a new skip. Logs show `WINDOWS_CAPABILITY ... exercised/unavailable`, exact skipped
test IDs, and the result JSON's `substituted` list. The installed-wheel runner
rejects every skip outside the four named junction/symlink tests. Even those four
must use exactly `CAPABILITY_UNAVAILABLE <kind>: WinError <code>; substitutes mandatory`,
with the matching kind and code in **{5, 50, 1314}**. Unknown codes, malformed numbers,
missing suffixes, trailing text/whitespace/newlines and mismatched kinds fail.

Whether or not links are available, ordinary directory/leaf access-denial,
identity handoff, substituted-object cleanup, collision and rollback tests must
run. Mocked reparse attributes and changed identity must prevent content writes
and public renames. These are **invariant substitutes**, not a claim that an
unavailable reparse type was exercised. Unsupported volumes cannot turn the
whole Windows lane green through skips.

### Optional pinned Windows renderer evidence

Dry/preflight uses **only the standard library**, works on any host, reads no
asset/secret, imports no audio dependency and never asserts renderer success:

```powershell
& $Python -S -m tests.windows_fluidsynth_evidence --preflight
```

Enabled prerequisites: Python **3.11 x64**, FluidSynth **2.6.2** and its dependent
DLLs, pyfluidsynth **1.3.4**, SoundFile **0.14.0** / libsndfile **1.2.2**, an explicit
approved local SoundFont manifest using the contract below, and a reviewed DLL
SHA-256. Python patch/build and every queried version are recorded, not inferred
from the CLI. Other dependency versions remain recorded by `pip-freeze.txt`;
this is not a fully locked package environment. See [Windows setup](audio-fluidsynth.md#windows-powershell).
No library or SoundFont is downloaded by the evidence harness.

Set `$Manifest`, `$Dll`, `$DllSha256` and `$EvidenceParent` to reviewed **absolute
local paths/digest in your shell**, never in a tracked script. The evidence parent
must already exist outside the checkout. Do not manufacture license approval.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./scripts/test-windows-audio.ps1 -Python $Python -ScratchParent $Scratch -Renderer -Manifest $Manifest -Dll $Dll -DllSha256 $DllSha256 -EvidenceParent $EvidenceParent
if ($LASTEXITCODE -ne 0) { throw 'Pinned renderer evidence failed; inspect external logs' }
```

Once selected, missing manifest/approval/asset/DLL, wrong digest/runtime,
non-Windows/non-x64 Python, assertion optimization, skipped native tests, or failed
semantic gates are **failures**, not skips. The public `fluid_version` function's
loaded Windows module must match the supplied DLL identity and digest. All five
existing `test_audio_fluidsynth_integration.py` tests run unchanged in a child
process; every canonical workflow includes independent cold, cold-repeat and
warm-cache runs on this machine. No Windows/macOS PCM-hash comparison is made.

External `windows-fluidsynth-*` evidence contains `runtime.json` (OS build,
architecture, Python, loaded DLL/binding digests, runtime versions, approved asset
and manifest digest, commands), `pip-freeze.txt`, `native-tests.log`,
`worker-results.json`, and `summary.json`. The summary hashes retained outputs
and links the canonical report's per-workflow cache hits/misses, elapsed times,
and repetition equality. The existing evidence subtree retains sources, expected
intent, configs, render/cache manifests, gate reports, WAVs and source command
logs. A failed test produces no successful summary. Keep the companion build log
and wheel digest with the evidence. Human listening remains **pending**; CLAP/FAD,
LUFS mastering, rhythm/note scores and polyphonic f0 are not inferred.

The manual workflow defaults to hosted **preflight only**. Opt-in execution needs
a trusted, dedicated self-hosted runner labelled `aimusic-fluidsynth-2.6.2`, with
assets pre-provisioned and a reviewer-protected `windows-audio-evidence`
environment configured by the owner. Only dispatch trusted, reviewed refs; never
attach that asset-bearing runner to untrusted PR execution. Absent runner means
an unscheduled job, not native proof. Inputs enter PowerShell through environment
variables, not interpolated executable text. No secrets are required and no
artifacts/media are automatically uploaded. A path is local input, not an asset
retrieval URL.

**Repository hygiene:** never track generated WAV/MIDI/cache/review bundles,
SoundFonts, DLLs, credentials, private manifests, pip freezes, or machine-specific
absolute paths. Reproduction docs use variables; portable reports use relative
artifact names plus digests. License approval for local assets is not permission
to publish them or the generated evidence wholesale. Inspect `git diff --check`
and `git status --short` before staging; static tests are not a license audit.

## Package integrity / reconciliation regression lane

```bash
python -m pytest tests/test_render_package.py tests/test_audio_entry.py tests/test_audio_reconcile.py tests/test_architecture.py -q
```

These tests need no native synth or SoundFont. Contract tests corrupt each
payload, exercise aggregate-only v1 packages and aliases, and verify that a
reused directory is not trusted or overwritten. New manifests identify the
file and expected/actual SHA-256; legacy failures identify the aggregate's
input group instead (see [contract](render-package-contract.md)).

Reconciliation fixtures check original MIDI before groove: exact discrete
fields, inclusive **0.001 BPM** and **one local source tick** tolerances,
complete tempo/meter maps, per-track EOT, and trailing planner silence distinct
from Score occupancy. A multi-channel 19-EDO fixture verifies logical source
track counts, not audible tuning. Unknown name-derived roles are explicitly
unavailable; declared planner roles remain authoritative. Program/count/drum
mismatches fail even with valid package hashes. Warm-cache and stale-context
tests prove validation cannot be bypassed; validate-only is checked in a fresh
process for absence of synth/DSP imports.

This lane does not certify downstream expressive-MIDI fidelity, recognizable
instruments or f0 accuracy. Those require the later event/render/tuning checks
and the separately enabled native lane.

## Event-preserving groove regression lane

The implemented [event/groove semantics](audio-midi-events.md) have asset-free
round-trip, tempo-map, expression, sustain-conflict and independent 19-EDO pitch
reconstruction tests. These are MIDI-event checks, not native audio measurements.

```bash
python -m pytest tests/test_audio_midi_events.py tests/test_audio_midi_scaling.py tests/test_audio_groove.py tests/test_architecture.py -q
```

### Deterministic scaling guard

`test_audio_midi_scaling.py` profiles module operation counts, not elapsed time,
for approximately 2k/4k source events (500/1000 notes, bends, CC74, tempo changes).
Parsing, retiming and one ghost per note must each grow by at most **2.7×** when
input doubles. Groove anticipation separately counts event visits to guard meter
lookups. Batched ghosts are also compared with incremental insertion, and the
expression sweep is checked against the lifetime-based association definition.

Measured with Python 3.12 during review remediation:

| Phase | Before 2k → 4k calls | Indexed 2k → 4k calls |
|-------|---------------------|-----------------------|
| Parse | 510,546 → 2,021,086 | 9,548 → 19,088 |
| Retime | 3,625,270 → 14,300,500 | 93,297 → 186,547 |
| Ghost batch | not used as a timing gate | 104,045 → 208,175 |

These are deterministic regression signals, not synth throughput or audio-quality
claims. Work still scales with actual simultaneous voice/event intersections;
unbounded sustained polyphony is not promised linear cost.

## Logical-stem review gate A (C4)

[Layout and migration](audio-stems.md) documents source IDs, role lookup,
reserved artifacts and refusal of existing output paths. Review evidence is
asset-free and concerns **MIDI events**, not native listening or audio f0:

| Regression evidence (`tests/test_audio_stems.py`) | Assertion |
|---|---|
| Multi-channel logical track plus drums | Two stems, all member-channel notes retained; authoritative `comp` → `comping`, independent of a misleading name |
| Serialized isolated MIDI | Exact absolute ticks, source-track slots, event payloads and global streams; banks/programs, RPN range, bends, sustain, aftertouch and opaque MTS payload retained |
| Sustain / tempo | Both held notes release at tick 700; tempo/meter changes at tick 480 unchanged; drum keys remain on channel 9 |
| Independent channel-state oracle | Full before/after program, bank, pedal and pitch-offset trajectories equal (12.5-semitone RPN range included) |
| Three 19-EDO tones through retiming + serialization/isolation | Correct note identities/counts; each reconstructed onset within **1 cent** of independent EDO targets |
| Cross-track ownership | Conflicting notes/programs/bends/CCs fail instead of duplicating or silently changing channels |
| Artifact names and mix | 13 duplicate/reserved/Unicode/separator/traversal/long labels remain unique and indexed; mix equals normalized sum of every source WAV |
| Pre-existing artifacts | Seven output kinds × file/directory/inside/outside/dangling-link cases refused without modification; symlink root refused |
| Integration | Silent stem retained; all MPE members feed simple preview; cold/warm cache copies retain index and unrelated files, refuse collisions |

Reproduce with the project's lightweight audio/dev environment:

```bash
python -m pytest tests/test_audio_stems.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
```

Recorded on 2026-11-01 with Python 3.12.12, mido 1.3.3, NumPy 2.5.4,
PrettyMIDI 0.2.11.post0 and SoundFile 0.14.0:

- Focused stems: **18 passed**, 44 subtests.
- Lightweight audio + architecture + RenderPackage: **173 passed**, 172 subtests;
  **1 skipped** optional `audio-bridge` test (not installed).
- Full unittest discovery: **506 run**, **9 skipped**, no failures (60.840 s).
- `flake8 aimusic tests ui.py`: clean; `mypy aimusic tests ui.py`: no issues in
  115 files; strict OpenSpec validation and `git diff --check`: clean.
- One retained PrettyMIDI warning identifies the deliberate non-track-0 key
  signature fixture; exact global-event preservation is separately asserted.
- TDD red runs reproduced missing isolation, unsafe/reserved names and overwrite
  behavior before fixes. An inverted ownership guard was killed by regression
  tests; the guard was restored before final verification.

This evidence is presented for human review of event fidelity (task 5.3). It is
**not** human approval, native instrument recognizability, monophonic/polyphonic
f0 measurement, CLAP/FAD scoring or a mastered-audio quality claim. At that checkpoint native asset
selection and C6+ work were outstanding; later native results are recorded below.

## Registry/cache contract checkpoint C5

[Renderer/cache contracts](audio-renderer-cache.md) describe routing, classified
fallback, actual-backend provenance and complete digest manifests. C5 tightens
simple admission: the C4 MPE/MTS event-isolation tests still preserve all payloads,
but simple rendering of those unsupported pitches is now rejected, not flattened.
The C4 multi-channel/silent preview test uses direct-12 channel content instead.

Recorded C5 verification (2026-11-01, existing Python 3.12.12 audio/dev venv):

- Lightweight audio + architecture + RenderPackage: **208 passed**, 224 subtests,
  **1 skipped** optional bridge integration; one retained non-track-0 metadata
  warning from the event-fidelity fixture.
- Full unittest discovery: **541 run**, **9 skipped**, no failures (63.732 s).
- `flake8 aimusic tests ui.py`: clean; `mypy aimusic tests ui.py`: no issues in
  118 files. Strict OpenSpec validation and `git diff --check`: clean.
- TDD reproduced registry bypass, broad bridge fallback, stale/corrupt cache reuse,
  output-report overwrite and valid-entry substitution during copying. An inverted
  capability condition was killed by registry tests and restored.
- A real subprocess exit during cache production left no complete entry; a retry
  produced a new verified result. Existing planner-handoff coverage now wraps
  actual stages instead of pretending missing sidecars are successful output.
- `python -m aimusic.app.cli generate --seed 11 --beats 4 --out <scratch>` followed
  by preview orchestration produced a 48 kHz, 108000-frame rough mix (RMS 0.05272).
  Cold expression/render misses became warm hits; mix bytes were identical.

No native FluidSynth lane, SoundFont, audio-f0 certification or human listening
acceptance is claimed by C5. Only tasks 6.1–6.4 are completed in this checkpoint.

## Clean-room FluidSynth / aligned-mix checkpoint C6 partial

[Provenance, public API sources, licensing and Windows/Linux setup](audio-fluidsynth.md)
cover **7.1–7.3 only**. Bindings are optional `pyfluidsynth==1.3.4`; wheel metadata
checks core requirements remain only `numpy` and `mido`. No pinned-repository
adapter/tests source was copied.

Recorded on 2026-11-01 in the existing Python 3.12.12 audio/dev environment:

```bash
python -m pytest tests/test_audio_fluidsynth.py tests/test_audio_mix.py tests/test_audio_stems.py tests/test_audio_wheel.py tests/test_audio_registry.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
openspec validate deliver-midi2audio-audible-demo --strict
```

- Focused renderer/mix/stems/wheel/registry: **49 passed**, 85 subtests.
- Config regression after the default-comment update: **23 passed**, 27 subtests.
- Lightweight audio + architecture + RenderPackage: **239 passed**, 290 subtests,
  **1 skipped** optional bridge integration; retained non-track-0 metadata warning.
- Full unittest: **572 run**, **9 skipped**, no failures (61.884 s).
- Lint/type checks, strict OpenSpec validation and `git diff --check`: clean.
- TDD red runs exposed the missing adapter, unequal frames, wrong mix gain and
  overly strict note-off status handling. An inverted peak-gain condition was
  killed by mix tests and restored before the green runs.
- Mocked native calls check tempo changes, absolute fractional-tick sample
  positions, same-tick order, banks/programs/drums, sustain/RPN/bends, fixed tail,
  rejected events/settings, cleanup and no realtime driver. They are **not**
  native sound, pitch, latency or instrument-quality evidence.

### Generation → simple WAV smoke

Reproduce with a fresh local `$SCRATCH` directory (never commit its outputs):

```bash
python -m aimusic.app.cli generate --seed 11 --beats 4 --out "$SCRATCH/generated"
python - "$SCRATCH" <<'PY'
import sys
from pathlib import Path
from aimusic.audio.config import load_audio_config
from aimusic.audio.from_score import from_score
from aimusic.audio.orchestrator import Orchestrator
root = Path(sys.argv[1])
context = from_score(next((root / 'generated').glob('run_*')))
for name in ('cold', 'warm'):
    runner = Orchestrator(load_audio_config('preview').to_dict(), root / 'cache')
    runner.run(context, output_dir=root / name)
    print(name, runner.cache_hits, runner.cache_misses)
PY
```

Measured on macOS arm64, mido 1.3.3, NumPy 2.5.4, PrettyMIDI 0.2.11.post0,
SoundFile 0.14.0. Four stems, click and mix are all **48 kHz / 108000 frames**
(2.0-second horizon + 0.25-second fixed tail), finite and peak <=0.95.

| Signal | Active RMS (0.1–0.8 s) | Peak |
|---|---:|---:|
| Bass | 0.104148 | 0.251189 |
| Comping | 0.039570 | 0.251189 |
| Drums (simple preview, not native percussion) | 0.044356 | 0.251189 |
| Lead | 0.092371 | 0.251189 |
| Rough mix | 0.151110 | 0.719104 |

Whole-mix RMS: **0.150934**. Cold expression/render missed; warm expression/render
hit and all six WAVs were byte-identical. Every effective renderer was `simple`,
with no fallback. This also exercises package reconciliation and publication;
unit tests independently assert click exclusion and exact-zero silent stems.

The local detailed report is `c6-smoke.KwN21L/report.json` under the session's
approved scratch directory. Native FluidSynth/SoundFont rendering, the eight-bar
native benchmark, controlled audio tuning and human listening were **not run**.
At that checkpoint **7.4/7.5 were unchecked** and 7.5/MUS-34 acceptance still
required human review. Later native and human acceptance evidence supersedes those gates.

## Tuning requirements checkpoint (8.1 only)

`tests/test_audio_tuning.py` covers Score/EDO/base intent, misleading direct-12
labels, channel-local RPN/default/null/partial writes, all nonzero bends, MPE zone
and MTS evidence, percussion exclusion, and pre-groove/pre-publication admission.
See [capability policy and migration](audio-renderer-cache.md). The two cache
metadata tests use an explicitly fake tuning-capable provider; production simple
and FluidSynth then advertised only `direct_12` (before the native proof below).

Recorded on 2026-11-01 in the existing Python 3.12.12 audio/dev environment:

- Focused tuning/registry/reconcile/orchestrator: **74 passed**, 112 subtests.
- `python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q`:
  **266 passed**, 342 subtests, **1 skipped** optional bridge test; retained
  non-track-0 metadata warning.
- `python -m unittest discover -s tests -p 'test_*.py'`: **599 run**, **9 skipped**,
  no failures (64.361 s).
- `flake8 aimusic tests ui.py` and `mypy aimusic tests ui.py`: clean (125 files).
- TDD red runs demonstrated range/MTS/Score/metadata-identity gaps and groove
  running before admission. Inverting the RPN default comparison failed tests;
  the original source and its Python bytecode cache were restored before checks.

These were asset-free admission tests, not native tuning/f0 evidence. At that
checkpoint only **8.1** was complete; the later bounded channel-pitch proof below
completed **8.2/8.3** without advertising general MPE.

## Controlled tuning checker (9.1–9.2 only)

`aimusic/audio/scoring/tuning_check.py` is a standalone NumPy-only checker. Supply
**original intended Hz**, independently derived from Score/EDO/base tuning or a
known reference sidecar, and half-open sample windows in real mono PCM. Select
steady regions excluding attacks/releases; do not select a new target from the
rendered MIDI/audio. Percussion uses `expected_hz=None, percussion=True` and is
explicitly excluded. There is no channel mixing, automatic window selection,
transcription, dependency/model download, or renderer allocation.

```python
from aimusic.audio.scoring.tuning_check import (
    ToneWindow, TuningCheckConfig, analyze_tuning,
)

# pcm is the caller's normalized mono buffer; intended_hz is original intent.
report = analyze_tuning(
    pcm, 48000, (ToneWindow("reference", 4800, 19200, intended_hz),),
    config=TuningCheckConfig(max_tone_cents=4, max_mean_cents=10, max_p95_cents=10),
)
data = report.to_dict()  # JSON-safe, including failures and exclusions
report.require_pass()   # raises TuningCheckError with .report on rejection
```

- Target-independent normalized autocorrelation compares **interpolated peak
  heights**, avoiding integer-lag bias toward period multiples. Near-best period
  candidates must also have spectral support; a Hann FFT with at least 8× zero
  padding and log-parabolic interpolation refines f0. Search defaults are
  40–2000 Hz, at least four periods/64 samples, centered RMS >1e-5, and
  periodic-similarity confidence >=0.9. Confidence is **not a probability**.
- Inclusive frequency-range admission allows a small interpolation uncertainty:
  **0.002 × sample_rate / window_samples Hz**, or 0.2% of an unpadded Fourier bin
  (0.004 Hz for a half-second window). The report records this bin fraction and
  method `interpolated-autocorrelation+hann-fft-v2`. Actual measured Hz is never
  clamped to a boundary: cents and their unchanged hard floors use that value.
  This margin is not an accuracy guarantee for arbitrary signals.
- Each designated window is one measurement unit: its coverage is 1 when reliably
  measured, otherwise 0. Aggregate coverage is measured / expected **pitched
  windows**, not voiced time or polyphonic-note coverage. Every expected pitched
  window must be measured; empty/percussion-only sets cannot pass.
- Reports retain original targets, measured Hz, confidence, absolute cents,
  sample bounds, settings, exclusions and failure reasons. Missing measurements
  are null; mean/P95 use only available measurements and are null when none exist.
  Partial coverage fails even if its available statistics look good.
- Public result/report constructors validate finite positive measurements/targets,
  finite confidence in [0, 1], settings and consistent missing/percussion reasons.
  The report independently applies its configured minimum confidence and range to
  coverage/statistics and rejects any missing/nonfinite metric. A directly
  constructed low-confidence observation cannot pass or be hidden by good rows.
  Standalone `TuningWindowResult.coverage` indicates measurement presence; report
  coverage (including each serialized row) additionally applies these gates.
- Defaults enforce <=5 cents **for every controlled tone**, plus mean <=15 and
  P95 <=15 cents. `controlled=False` applies only the aggregate gates to the same
  explicit mono windows; it does not enable polyphonic analysis. P95 uses NumPy's
  linear percentile, with equal weight per measured window. Typed config can
  tighten each cents limit down to zero, never relax it. Inclusive comparisons
  allow only 1e-9 cents for floating-point roundoff, not DSP error tolerance.
- This slice does **not** wire the checker into orchestrator publication/cache or
  promote renderer capabilities. A caller must use `require_pass()` before
  claiming a successful validated check; failure preserves its structured report.

Recorded **2026-11-02**, Python 3.12.12 / NumPy 2.5.4, entirely synthetic PCM:

| 19-EDO steps above A4 (1, 4, 7) | Coverage | Mean absolute cents | P95 absolute cents | Result |
|---|---:|---:|---:|---|
| Original intended sine tones | 1.0 | 0.000120 | 0.000147 | pass |
| Nearest 12-TET corruption, same original targets | 1.0 | 42.105396 | 46.842246 | reject |

The corruption tones individually deviate by 36.842224, 47.368563 and 42.105401
cents; none relies on a near-grid choice. `tests/test_audio_tuning_check.py` also
covers exact boundaries, detuning signs/octaves, stronger second harmonics, DC,
44.1/48 kHz buffers, silence, short/truncated/nonfinite/noisy windows, percussion,
partial evidence, strict settings and a bad tone hidden by aggregate statistics.

Initial feature checkpoint verification (existing asset-free audio/dev environment):

```bash
python -m pytest tests/test_audio_tuning_check.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
```

- Focused: **22 passed**, 90 subtests. Lightweight + architecture + contract:
  **292 passed**, 456 subtests, **1 skipped** optional bridge test; retained
  non-track-0 metadata warning. Full unittest: **625 run**, **9 skipped**, no
  failures (63.410 s). Lint/type checks clean (128 files).
- TDD red runs exposed absent analysis/gates, negative-correlation peak selection,
  extreme-target arithmetic and invalid policy flags. Inverting the cents gate
  failed the suite; original code was restored before final verification.
- Local report/logs: `tuning-check-synthetic.json`, `tuning-check-audio.log`,
  `tuning-check-full.log` under the session's approved scratch directory.

Review corrections verified **2026-11-02**, same asset-free environment/commands:

- Focused checker: **34 passed**, 213 subtests; lightweight/architecture/contract:
  **304 passed**, 579 subtests, **1 optional bridge skip**, retained metadata warning.
  Full unittest: **637 run**, **9 skipped**, no failures (63.672 s). Lint and mypy
  clean (128 files). No task checkboxes changed.
- Red regressions reproduced invalid direct construction, NaN/missing metrics,
  confidence-zero acceptance, all 18 high-19-EDO/fractional-period phase cases,
  and inclusive frequency-boundary failures. Disabling the report confidence
  gate failed the regression suite; original code was restored before full checks.
- Coverage includes `440 * 2**(40/19)` at 16 kHz, periods 8.5/9.5 samples, six
  phases each, and 40/2000 Hz boundaries across six phases and three lengths.
  Values remain identical to estimation with wider bounds. At 0.5 seconds,
  genuinely outside 39.99/2000.01 Hz tones reject despite errors smaller than the
  5-cent floor; stricter cents limits still reject admitted boundary measurements.
- Logs: `tuning-evidence-red.log`, `tuning-fractional-red.log`,
  `tuning-range-red.log`, `tuning-evidence-mutation.log`,
  `tuning-evidence-audio.log`, `tuning-evidence-full.log` in approved scratch.

**Evidence limits:** dry controlled mono tones only. Octave-ambiguous, vibrato,
inharmonic, overlapping or arbitrary instrument audio is unproven; every report
labels polyphonic accuracy **unmeasured**. That synthetic checkpoint produced no
native hashes or listening samples. The later native checkpoint completed **9.3**;
**9.4 remained open** until the later [human review](audio-listening-review.md).
MUS-35 is still partial: Scala/MTS-helper breadth and pre-endpoint
diffusion gating remain follow-up work, with MUS-34 → MUS-35 → MUS-36 → MUS-37 →
MUS-38 dependencies unchanged. No native acceptance, human listening approval,
CLAP/FAD or production-quality claim is made.

## Final asset-free verification (11.1)

Re-run on 2026-11-02 with Python 3.12.12 and no FluidSynth library, SoundFont,
GPU, credentials, or live service required:

```bash
python -m pytest tests/test_architecture.py tests/test_render_package.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
python -m pytest tests/test_audio_wheel.py -q
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
openspec validate --all --strict --no-interactive
```

- Core architecture/contract: **17 passed**, 13 subtests.
- Lightweight audio/architecture/contract: **340 passed**, 642 subtests;
  **1 skipped** optional bridge test.
- Full unittest discovery: **673 run**, **9 skipped**, no failures (79.701 s).
- Isolated wheel/profile/demo-source smoke: **1 passed**.
- Flake8, mypy (133 source files), and strict OpenSpec validation: clean.

These results completed only the asset-free cross-lane check. Skips are not native
evidence. The subsequently approved asset, native proofs and four-workflow results
are recorded below; human listening was still pending at this checkpoint.

## Risks

| Risk | Mitigation |
|------|------------|
| Diffusion pulls 19-EDO → 12-TET | B3 routing; f0-following; tuning floor; skip-restyle default |
| Endpoint terms/pricing change | RestyleEndpoint protocol ≥2 impls; local MusicGen for search logic |
| Spend overruns | Budget guard; open-weight for broad search |
| madmom/demucs install pain | Extras groups; light deterministic spine; pin/containerize heavy path |
| Symbolic blockers | Fixture corpus + contract decoupling |
| Invalid LLM groove specs | pydantic validate; reject-and-retry; measured templates override |
| Scorer gaming | Hard floors; monotonicity tests; milestone human audits |
| Rights / watermarks | Provenance manifest; MIDI-GPT flagged off by default |

## Native opt-in manifest contract and 12-EDO evidence (2026-11-02)

`tests/test_audio_fluidsynth_integration.py` skips **only** when
`AIMUSIC_RUN_FLUIDSYNTH_TESTS` is not exactly `1`. Set
`AIMUSIC_FLUIDSYNTH_MANIFEST` to an external UTF-8 JSON object. All fields below
are required and nonempty; `license_approved` must be boolean true. The asset must
be an absolute existing local file, and its actual bytes must match the lowercase
64-digit SHA-256. Source commit must be a full lowercase commit hash. Source and
license URLs must be HTTPS. Native/binding versions must match runtime queries.
Missing binding/library/path/license/expected hash, modified asset bytes or wrong
versions **fail** the selected lane, never skip. The manifest records the user's
approval; it cannot independently establish legal ownership of samples.

Example (replace only the local path with the already-approved external asset):

```json
{
  "soundfont_path": "/absolute/local/path/GeneralUser-GS.sf2",
  "soundfont_sha256": "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe",
  "soundfont_name": "GeneralUser GS",
  "soundfont_version": "2.0.3",
  "source_url": "https://github.com/mrbumpy409/GeneralUser-GS",
  "source_commit": "684543d5e5efaef08d02be50dcda8d552478fa60",
  "license_url": "https://github.com/mrbumpy409/GeneralUser-GS/blob/684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt",
  "license_summary": "Private/commercial music and software-project use/modification permitted; historical sample-origin uncertainty disclosed; local-only approval",
  "license_approved": true,
  "native_version": "2.6.2",
  "binding_version": "1.3.4"
}
```

```bash
# Activated audio-render/dev venv. No download, service key or GPU.
export AIMUSIC_RUN_FLUIDSYNTH_TESTS=1
export AIMUSIC_FLUIDSYNTH_MANIFEST="$LOCAL_ASSETS/native-manifest.json"
export AIMUSIC_FLUIDSYNTH_ARTIFACTS="$LOCAL_OUTPUTS"  # existing external directory
python -m pytest tests/test_audio_fluidsynth_integration.py -q -s
# Default/asset-free lane, including subprocess proof of fail-closed opt-in:
unset AIMUSIC_RUN_FLUIDSYNTH_TESTS AIMUSIC_FLUIDSYNTH_MANIFEST AIMUSIC_SOUNDFONT
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
```

`AIMUSIC_FLUIDSYNTH_ARTIFACTS` is optional (defaults to OS temp); each run prints a
fresh `fluidsynth-evidence-*` directory and retains source/expected JSON, actual
asset manifest, config/backend provenance, original/expressive/isolated MIDI,
WAVs, separate caches and reports. No machine-specific path is tracked.

First passing native checkpoint: `fluidsynth-evidence-wnd_i84v` in local scratch.
All **780000 frames / 48 kHz**: 16-second eight-bar horizon + fixed **0.25 s** tail.
Roles/programs/channels/counts match `demo_sources fixture-12` exactly:
drums 0/9/96, bass 33/0/16, comping 0/1/48, lead 80/2/64 (zero-based).
Source and isolated note endpoints/velocities match; the no-op expression path is
not claimed as audible groove improvement. Click is generated but excluded by an
exact reconstructed-mix assertion. Every designated active window passes >1e-5;
all samples finite and peaks <=0.95.

| Signal | RMS, 0.1–0.4 s | Peak |
|---|---:|---:|
| Drums | 0.000967488 | 0.064514160 |
| Bass | 0.014692156 | 0.041717529 |
| Comping | 0.012433238 | 0.054138184 |
| Lead | 0.010266098 | 0.033386230 |
| Rough mix | 0.021610964 | 0.118316650 |

Cold **0.931 s**, independent cold-repeat **0.852 s**, warm **0.354 s**;
all six WAVs byte-identical across runs. Both cold runs miss expression/render
caches, warm hits both. The test retains timings and fails above **60 s**, rather
than relaxing the target. Native/binding/asset provenance is checked on each run.

**7.5 remained open at this checkpoint**: samples were available, not yet human-reviewed.
No MUS-34 completion, role recognizability, mastered quality, CLAP/FAD, or broad
polyphonic tuning certification follows from these signal measurements.

## Native controlled tuning checkpoint (8.2 / 8.3 / 9.3)

Final targeted run on the pinned asset/runtime above: **4 passed**, no skips
(3.42 s), `fluidsynth-evidence-qjrwn8k9` in local scratch. The existing eight-bar
checks still pass: cold **0.978 s**, independent cold **0.918 s**, warm **0.357 s**;
all six WAV hashes match. Signal metrics are unchanged from the table above.

Production dry `reference-19` is rendered through the registry/orchestrator with
original timing, then with a controlled **+12.5 ms (+12 ticks)** microtiming offset
via `apply_groove`. Original target frequencies never change. Windows shift
independently by exactly **600 samples** at 48 kHz; notes' event evidence is checked
at the corresponding original or shifted ticks. Every run has **228000 frames**
(4.5-second horizon + 0.25-second tail); dry reverb/chorus remain off.

| Original target Hz | Original measured Hz / abs cents | Groove measured Hz / abs cents |
|---:|---:|---:|
| 456.348220 | 456.450112 / 0.386501 | 456.449698 / 0.384934 |
| 509.128504 | 509.551902 / 1.439118 | 509.548268 / 1.426770 |
| 568.013247 | 568.882467 / 2.647250 | 568.870632 / 2.611233 |

- Original: coverage **1.0**, mean **1.490956¢**, P95 **2.526436¢**.
- Groove: coverage **1.0**, mean **1.474312¢**, P95 **2.492787¢**.
- **Simultaneous independent channels**, same synth, hard-panned mono windows:
  456.450105 / **0.386478¢**, 568.882500 / **2.647350¢**, coverage **1.0** each.
  A shared/global wheel would detune a voice; these are opposite-signed offsets.
  This is separated controlled measurement, **not polyphonic pitch scoring**.
- Deliberately flattened wheels, **same original targets/windows**, remain
  measurable (coverage **1.0**) but reject: mean **41.678582¢**, P95 **47.877602¢**;
  per-tone **37.442129 / 48.897800 / 38.695818¢**. Both controlled <=5¢ and
  aggregate-only mean/P95 <=15¢ checks reject. Floors/config remain unchanged.
- Reference warm-cache WAV bytes and provenance match the original; groove and
  flattened sources miss its cache. Cache fingerprint equals the recorded
  `render.json`, including original tuning events and reconstructed onset pitches.
  Native MTS and cross-track-ownership rejection tests publish no success.

Reports: `reference-19-report.json`, `{original,groove,flat}/tuning-check.json`,
`flat/tuning-check-general.json`, `simultaneous-tuning.json`; each retains expected
and measured Hz, confidence, coverage and unchanged checker configuration. The
reference report also hashes the original `expected.json`. Raw public-adapter
probe MIDI/WAV/reports are retained separately; no native capability was promoted
until those measurements passed. Asset-free red tests then required the exact
subset, idle-channel pitch state, no fallback and pre-cache/pre-allocation refusal.
An inverted note-admission state was killed by regression tests before restoration.

**Limits / open gates:** this is controlled monophonic f0 and source-event evidence.
Arbitrary imports and polyphonic audio remain **unmeasured**. Orchestration does
not automatically ingest arbitrary expected-frequency sidecars or certify f0;
the native harness explicitly runs `require_pass()` before treating its controlled
reports as passing. Flattened renders are intentionally retained as rejected
counterexamples, never labelled validated. At this checkpoint the complete four
examples were still pending; they are recorded below. Human listening,
Scala/MTS-helper breadth, diffusion gating, CLAP/FAD and mastering remain open.
This checkpoint alone did not complete 7.5/9.4/10.3/10.4/11.x or change Linear.

Final checkpoint verification (native opt-in unset for all asset-free commands):

```bash
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
openspec validate --all --strict --no-interactive
```

- Lightweight + architecture/contract (including isolated wheel smoke):
  **349 passed**, **681 subtests passed**, **5 skipped** (four deliberately opt-in
  native tests and the pre-existing optional bridge test), 32.62 s.
- Full unittest: **686 run**, **13 skipped**, no failures, 79.786 s. Native skips
  in this lane are not counted as acceptance; the separate native run has no skips.
- Flake8 clean; mypy clean (**137 files**, unchanged untyped-body informational
  note); all strict OpenSpec checks and `git diff --check` clean.
- That continuation completed **1.3, 7.4, 8.2, 8.3, 9.3** only. Its remaining
  four-workflow work is covered below. **7.5 / 9.4** still required human review then;
  no change to the already-completed historical 11.1 check.
  MUS-35 remains partial and the blocking chain is unchanged. No Linear/push/PR.

## Four native listening workflows (10.3 / 10.4 / 11.2)

Recorded **2026-11-02** after `0eb35c1`, with the additional
`test_four_native_listening_workflows` harness. No production code changed.
[Compact machine-readable manifest](audio-native-listening-evidence.json) records
measured metrics, all WAV hashes, source/package digests, per-run elapsed times,
render-manifest hashes, runtime bytes and the full evidence digest. Full local
outputs are retained in **`fluidsynth-evidence-g_19b5em`** under the approved
external scratch directory (the test prints its absolute path).

Runtime: **Python 3.12.12**, **macOS 27.2 arm64**, **Apple M4 Pro / 14 cores /
24 GiB RAM**, native **FluidSynth 2.6.2**, binding **pyfluidsynth 1.3.4**. Asset:
approved **GeneralUser GS 2.0.3**, official source commit
`684543d5e5efaef08d02be50dcda8d552478fa60`, SF2 SHA-256
`9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe`.
Sibling `LICENSE.txt` SHA-256:
`7b32efefdf95ce38a043799f0659853ddc00fbaa14d8c50f0aca16b9b8b405be`.
[License terms/caveat](audio-fluidsynth.md#approved-local-only-asset--installed-runtime-2026-11-02)
remain local-only approval, not a redistribution warranty.

### Reproduce and locate the listening set

Use the existing venv and approved manifest; do not download assets. From the
checkout in a POSIX shell (WSL on Windows), first set these **caller-owned absolute
paths**: `SCRATCH` to an existing external output directory, `PYTHON` to the
venv's Python executable, and `AIMUSIC_FLUIDSYNTH_MANIFEST` to the approved JSON
manifest. For an already activated venv, `PYTHON="$(python -c 'import sys; print(sys.executable)')"`
is a portable derivation. No particular venv name or directory layout is assumed.

```bash
set -eu
: "${SCRATCH:?Set an absolute existing external output directory}"
: "${PYTHON:?Set the absolute path to the audio-render venv Python executable}"
: "${AIMUSIC_FLUIDSYNTH_MANIFEST:?Set the absolute approved manifest path}"
for path in "$SCRATCH" "$PYTHON" "$AIMUSIC_FLUIDSYNTH_MANIFEST"; do
  case "$path" in /*) ;; *) echo "Absolute path required: $path" >&2; exit 1 ;; esac
done
[ -d "$SCRATCH" ] && [ -x "$PYTHON" ] && [ -f "$AIMUSIC_FLUIDSYNTH_MANIFEST" ] || {
  echo 'SCRATCH, Python executable and manifest must already exist' >&2; exit 1;
}
# Optional Homebrew library discovery; never assume an Apple Silicon prefix.
if command -v brew >/dev/null 2>&1; then
  HOMEBREW_PREFIX="$(brew --prefix)"
  export HOMEBREW_PREFIX
fi
export AIMUSIC_RUN_FLUIDSYNTH_TESTS=1
export AIMUSIC_FLUIDSYNTH_MANIFEST
export AIMUSIC_FLUIDSYNTH_ARTIFACTS="$SCRATCH"
"$PYTHON" -m pytest tests/test_audio_fluidsynth_integration.py -q -s
```

The recorded native invocation passed **5 tests, no skips, 10.29 s**. The test
runs fresh source-generator subprocesses and the real `Orchestrator` (no fake
renderer). Its exact source argv/logs are retained in `four-workflows/evidence.json`
and `command-*.log`. It uses the versioned `listening` profile plus the approved
asset path and explicit empty groove spec. Render elapsed times exclude generation
and measurement. Effective config, dry native settings and code/dependency hashes
are retained in every `render.json`. Actual author-machine paths remain only in
external evidence, not these reproduction commands.

#### Supplemental runtime capture (after a passing native run)

The test does **not** generate `runtime.json` or `pip-freeze.txt`. Run the following
separately, in the same environment and checkout. Set `EVIDENCE` to the **absolute
fresh directory printed by the test**, not the scratch parent. Set
`FLUIDSYNTH_LIBRARY` to the absolute shared-library file used by your installation
(not the CLI executable). This is a caller-identified binary path for hashing;
version verification alone cannot prove which of two same-version binaries loaded.
The snippet records platform/CPU information (additional hardware fields on macOS),
binding/library/license bytes, asset provenance and invocation. Files are created
exclusively: existing evidence is never overwritten. Keep both files external.

```bash
set -eu
: "${PYTHON:?Use the same venv Python as the native run}"
: "${EVIDENCE:?Set the absolute fresh evidence directory printed by the test}"
: "${FLUIDSYNTH_LIBRARY:?Set the absolute installed shared-library file path}"
"$PYTHON" - "$EVIDENCE" "$FLUIDSYNTH_LIBRARY" <<'PY'
import json, os, platform, subprocess, sys
from pathlib import Path
from aimusic.audio.render import fluidsynth_r
from aimusic.audio.render.registry import file_digest
from tests.audio_native_support import verify_native_runtime
root, native = map(Path, sys.argv[1:])
assert root.is_absolute() and root.is_dir()
assert native.is_absolute() and native.is_file()
assert not any((root / name).exists() for name in ('runtime.json', 'pip-freeze.txt'))
asset = json.loads((root / 'asset.json').read_text())
evidence = json.loads((root / 'four-workflows/evidence.json').read_text())
assert evidence['checks_passed'] and evidence['runtime'] == verify_native_runtime(asset)
assert file_digest(Path(asset['soundfont_path'])) == asset['soundfont_sha256']
license_path = Path(asset['soundfont_path']).with_name('LICENSE.txt')
binding = Path(fluidsynth_r.load_binding().__file__)
hardware = dict(machine=platform.machine(), processor=platform.processor(), logical_cpus=os.cpu_count())
if sys.platform == 'darwin':
    hardware.update({k: subprocess.check_output(['sysctl', '-n', k], text=True).strip()
                     for k in ('machdep.cpu.brand_string', 'hw.ncpu', 'hw.memsize')})
data = dict(python_executable=sys.executable, python_version=sys.version, platform=platform.platform(),
            hardware=hardware, asset=asset, native_versions=evidence['runtime'],
            runtime_files_sha256={str(p): file_digest(p) for p in (binding, native, license_path)},
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            native_command=[sys.executable, '-m', 'pytest', 'tests/test_audio_fluidsynth_integration.py', '-q', '-s'],
            env={k: os.environ.get(k) for k in ('HOMEBREW_PREFIX', 'AIMUSIC_RUN_FLUIDSYNTH_TESTS',
                 'AIMUSIC_FLUIDSYNTH_MANIFEST', 'AIMUSIC_FLUIDSYNTH_ARTIFACTS')})
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
with (root / 'runtime.json').open('x') as stream:
    json.dump(data, stream, indent=2, allow_nan=False)
with (root / 'pip-freeze.txt').open('xb') as stream:
    stream.write(freeze)
PY
```

The earlier [CLI commands](audio-demo-sources.md#listening-commands-approved-native-prerequisites-required)
remain available for individual renders; the harness adds verified cache evidence.

Under the retained directory, listen to:

| Workflow | Mix (relative path) | Frames at 48 kHz | Cold / cold-repeat / warm seconds | Mix active RMS / peak |
|---|---|---:|---:|---:|
| Eight-bar fixture | `four-workflows/fixture-12/cold/stems/rough_mix.wav` | 780000 | 0.888 / 0.872 / 0.351 | 0.021611 / 0.118317 |
| Fresh Method A, seed 11, 4 beats | `four-workflows/generated-12/cold/stems/rough_mix.wav` | 108000 | 0.676 / 0.660 / 0.281 | 0.031423 / 0.190491 |
| Controlled reference | `four-workflows/reference-19/cold/stems/rough_mix.wav` | 228000 | 0.199 / 0.194 / 0.089 | 0.009882 / 0.026077 |
| 19-EDO multitrack | `four-workflows/multitrack-19/cold/stems/rough_mix.wav` | 204000 | 0.522 / 0.519 / 0.220 | 0.022462 / 0.077621 |

`role_index.json` beside each mix names all source stems and isolated MIDI. Sources
and independent `expected.json` remain one level above `cold/`; the fresh passage
retains **`generated-12/run_12cb5ba3_8e1cec6bed3c/`**, all six package files, and its
original flat CLI outputs. `score.analysis.json` retains planner provenance,
`structure.json` roles and `score.json` counts. `score.reconciliation.json` passes
original-MIDI comparison with exact discrete fields, **0.001 BPM** and **one-tick**
boundary tolerances. Other sources explicitly report reconciliation not applicable.

### Measured gates and limits

- All stems/mixes are finite, **48 kHz**, aligned including **0.25 s** release tail,
  active **0.1–0.4 s RMS >1e-5**, and peak **<=0.95**. The smallest stem RMS is
  **0.000967488** (fixture drums); largest stem peak **0.086364746** (generated
  comping). The mix equals the FLOAT sum of source stems with attenuation-only
  headroom, **excluding the separately generated nonzero click**, exactly.
- Every cold run misses expression/render caches; warm runs hit both. **All warm
  output bytes match**, including MIDI, WAV, analysis/reconciliation, renderer
  provenance, cache manifests and reference tuning reports. Independent cold
  **WAVs match exactly** too. One intentional cold metadata difference is retained:
  expressive sidecar `wall_clock_utc`. The test compares every other field/byte,
  records both hashes, and never rewrites that timestamp to fake equality.
- Source/expressive/isolated event payloads (ticks, notes, channels, programs,
  controller state and conductor metadata) match for the explicitly **no-op groove**
  four-run set. 19-EDO stem pitches are independently reconstructed and compared
  to **440 × 2^(original step/19)**: max **0.008995¢** reference, **0.011565¢**
  multitrack, both <=1¢. Every actual 19-EDO renderer is **channel_pitch FluidSynth
  2.6.2 / pyfluidsynth 1.3.4**, with no fallback, not a simple preview certificate.
- Reference `cold/tuning-check.json` uses the original `expected.json` windows and
  existing independent-target checker: **coverage 1.0**, per-tone **0.386501 /
  1.439118 / 2.647250¢**, mean **1.490956¢**, P95 **2.526436¢**; all unchanged gates
  pass. Imported expected-frequency sidecars are not automatically read by the
  product; this harness explicitly calls `require_pass()`.
- Additional `original/`, `groove/`, `flat/` and `warm-reference/` outputs in the
  same retained directory reproduce the earlier controlled **+12.5 ms** comparison.
  Groove f0 coverage **1.0**, mean **1.474312¢**, P95 **2.492787¢**; the deliberately
  flattened counterexample rejects against unchanged original targets (mean
  **41.678582¢**, P95 **47.877602¢**). Flat WAVs are diagnostic failures, not accepted
  demos. Use `original/stems/rough_mix.wav` and `groove/stems/rough_mix.wav` for human
  timing comparison; no audible groove improvement is asserted.
- The four-set MTS counterexample still fails with `fluidsynth does not support
  tuning mts`, **no successful files or complete cache**. Separate native tests
  also reject ambiguous ownership. Evidence completion is written only after all
  four workflows and MTS assertions pass.
- Three mutations of copied real outputs were rejected by the new checks: adding
  click to mix, replacing planner provenance with inferred, and flattening comping
  pitch wheels. Sources/production code were not changed. The initial harness red
  run identified only the intentional cold-expression timestamp; this was resolved
  by recording that distinction, not relaxing audio floors.

**At this checkpoint human listening was pending**; the later
[acceptance record](audio-listening-review.md) supersedes that status. Polyphonic
f0 (especially multitrack 19-EDO) is
**unmeasured**; mono reference results do not certify arbitrary SoundFont notes.
Role recognizability, subjective groove, complete envelope decay, rhythm F-score,
notes F1, CLAP/FAD and mastering were unmeasured at this checkpoint. Tasks
**7.5, 9.4, 11.3, 11.4** were then unchecked; MUS-35 remains partial
(Scala/helpers and diffusion gating), and
MUS-34 → MUS-35 → MUS-36 → MUS-37 → MUS-38 is unchanged. No Linear, push or PR.

### Cross-lane verification for this continuation

With native opt-in, manifest, SoundFont and Homebrew-discovery variables **unset**
(the library remains installed, but this lane does not require or load it):

```bash
python -m pytest tests/test_architecture.py tests/test_render_package.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
openspec validate --all --strict --no-interactive
```

- Architecture/contract: **17 passed**, 13 subtests (1.11 s).
- Lightweight audio including CLI and isolated installed-wheel/profile/demo smoke:
  **351 passed**, 697 subtests, **6 skipped** (five opt-in native tests and the
  optional bridge), 32.42 s. These skips are **not** native acceptance evidence.
- Full unittest: **689 run**, **14 skipped**, no failures (80.262 s).
- Flake8 clean; mypy clean (**137 files**, unchanged untyped-body informational
  note). Strict OpenSpec validation passes. Logs remain beside the native evidence.

Only **10.3, 10.4, 11.2** were completed in that checkpoint; its four review tasks
were left open. No WAV/SF2/cache, runtime freeze, credential or unrelated tooling
file was part of that commit. See the later [human outcome and task decisions](audio-listening-review.md).

## Final acceptance review (11.4)

Finalized **2026-11-03**, after `ed06d58` (authoritative source-track roles through
groove processing) and `134ff43` (portable imported-preview evidence path). The final
reviewer reported **no Critical/Required findings** and authorized task **11.4**.
Only that checkbox changes in this finalization: **42/42 tasks complete**.

### Cross-lane evidence and final reviewer confirmation

- Implementation verification: asset-free audio/architecture/contract (including
  installed-wheel checks) **376 passed / 6 optional skips**; full unittest
  **714 run / 14 optional skips**, no failures. `flake8 aimusic tests ui.py`,
  `mypy aimusic tests ui.py` (**140 files**, existing untyped-body informational
  note) and `openspec validate --all --strict --no-interactive` were clean.
  External logs: `role-review-{asset-free,full,flake8,mypy,openspec}.log`.
- The final reviewer independently confirmed **376 asset-free passes / 6 optional
  skips** and **5 native passes / no skips** with the approved GeneralUser GS 2.0.3,
  FluidSynth 2.6.2 and pyfluidsynth 1.3.4. Native evidence remains external in
  **`fluidsynth-evidence-afttn_6e`**. Asset-free skips are not native acceptance.
- The reviewer independently confirmed the authoritative-role regression's exact
  **[12,252] note-on/off ticks**, rather than name-based groove selection. Complete
  source-track mappings, canonical `comping` / legacy groove `comp`, inferred
  imports and expression-cache identity are covered by the role regressions.
- The **four mixes**, separate original/shifted **timing WAVs**, and generated
  **listening page** remain byte-identical to their respective pre-fix artifacts.
  This does not mean original and shifted audio equal one another. The generated
  page's SHA-256 remains
  `61554dbae92ce96cd63485c4657804ac5b51695a4bb0b5d23c428307e0aee454`;
  the earlier manual page has different HTML but the same four embedded mixes.
- Final review found **no tracked author-machine paths, prohibited SoundFonts,
  generated WAVs, credentials or submodules** in the delivery. Only documentation
  and the OpenSpec task file are included in this finalization; unrelated
  `AGENTS.md` and local tooling changes remain excluded.

### Per-capability acceptance mapping

| Capability | Accepted behavior and evidence |
|---|---|
| **Input integrity** (`audio-input-integrity`) | Package digests and pre-groove reconciliation retain planner roles/counts (exact discrete fields, 0.001 BPM / one-tick tolerances); supported imports remain inferred and immutable. `test_render_package.py`, `test_audio_reconcile.py`, `test_audio_imported.py` and `test_audio_entry.py`; authoritative-role **[12,252]** tick probe and generated-package native report. |
| **Instrument rendering** (`audio-instrument-rendering`) | Wheel-safe profiles, explicit per-role selection/fallback, event-preserving collision-safe aligned stems, click-free rough mix and verified content-based caches. Config/wheel, groove/events, registry/stems/mix/orchestrator tests plus five native tests; all four mixes finite, 48 kHz, active RMS >1e-5, peak <=0.95, cold-repeat/warm WAV bytes equal in the pinned lane. |
| **Bounded tuning** (`audio-tuning-integrity`) | Proven channel-pitch subset only; MTS/conflicts/lossy fallback reject. Event pitch error <=1 cent; controlled mono coverage 1.0, every tone <=5 cents and mean/P95 <=15 cents; flattened original-target counterexample rejects. `test_audio_tuning.py`, `test_audio_tuning_check.py` and native original/groove/flat reports. Polyphonic f0 remains **unmeasured**. |
| **Demo workflow** (`audio-demo-workflow`) | Lazy validation-first package/import CLI, optional-audio quarantine, four reproducible sources, separate asset-free/native lanes and portable deterministic embedded players. CLI/architecture/demo-source/docs/listening-review/wheel tests and retained native reports; [human rough-demo and timing acceptance](audio-listening-review.md) is recorded separately from automation. |

Test filenames above are under `tests/`; measured details and reproduction
commands remain in the preceding sections. Human acceptance does **not** assert
an audition of every isolated stem, aesthetic groove improvement, full envelope
decay, mastered quality, full rhythm/notes scores or CLAP/FAD. Mono reference
measurements do not certify every polyphonic note.

**Handoff:** MUS-34 completion **may be proposed** on this evidence; no Linear
mutation is made. **MUS-35 remains partial**: Scala/MTS helpers and pre-endpoint
diffusion gating are still outstanding. Polyphonic f0, later full scorers/CLAP/FAD
(MUS-36), restyle (MUS-37), mastering/references/breadth (MUS-38), and B4 remain
outside this milestone. **MUS-34 → MUS-35 → MUS-36 → MUS-37 → MUS-38** is unchanged;
partial MUS-35 does not unblock later tickets. No push, PR, upload or archive.

### Documentation-only finalization checks

`python -m pytest tests/test_audio_docs.py -q`: **3 passed, 12 subtests**. Strict
OpenSpec validation, local Markdown link-target checks and `git diff --check` pass. Task-diff checks confirm
only **11.4** changed from unchecked to checked; OpenSpec reports **42/42** complete.
The full implementation/native counts above are recorded evidence, not claimed
as reruns by this documentation-only finalization.
