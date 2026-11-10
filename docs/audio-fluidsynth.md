# Optional offline FluidSynth renderer

All four 12/19-EDO workflows now have [pinned local native evidence](audio-demo-sources.md),
including planner reconciliation and controlled f0. The user
[accepted the four rough demos](audio-listening-review.md) on 2026-11-02; task 7.5
is complete and MUS-34 completion can be proposed. This is separate from metrics;
MUS-35 stays partial. The separate human timing review is now recorded with
“No defects heard”; no aesthetic groove improvement is claimed. [Final review
(task 11.4) is complete](audio-testing.md#final-acceptance-review-114): **42/42** demo tasks,
without changing Linear or the remaining MUS-35 scope.

## Clean-room provenance and dependency review (2026-11-01)

- Historical pin: `arsenylosev/midi2audio_generative` at
  `46b3dfea37e15fecb84386ee91912cdeebcd9aec` (`46b3dfe`). The earlier apply-session
  adapter/tests review is recorded in the OpenSpec design §4. Relevant paths were
  `m2a/render/fluidsynth_r.py`, `tests/test_fluidsynth_render.py`, and
  `pyproject.toml`. That review found no license/permission notice. The public
  [recursive tree](https://api.github.com/repos/arsenylosev/midi2audio_generative/git/trees/46b3dfe?recursive=1)
  and [package metadata](https://github.com/arsenylosev/midi2audio_generative/blob/46b3dfe/pyproject.toml)
  were independently checked here: no LICENSE/COPYING/NOTICE or package license.
  Public availability is **not** permission to copy. This implementation did not
  read/copy that adapter or its tests; the pin is historical provenance only.
  The historical review also found a PrettyMIDI solo/delegated render, not a
  lossless event dispatcher. Native tests skipped without prerequisites and
  checked rate/peak, not event ordering, tempo-map sample scheduling or native
  lifecycle. Those findings explain why reuse would not satisfy this contract
  even with permission; they are review evidence, not implementation source.
- `aimusic/audio/render/fluidsynth_r.py` is newly written against this repo's
  `MidiEvents`/renderer contracts and the documented public APIs below. No
  submodule, vendored adapter, subprocess synth, or private synth handle is used.
- Optional distribution **`pyfluidsynth==1.3.4`**, imported as **`fluidsynth`**:
  [PyPI release metadata](https://pypi.org/pypi/pyFluidSynth/1.3.4/json),
  [tagged README/license statement](https://github.com/nwhitehead/pyfluidsynth/blob/v1.3.4/README.md#license),
  [tagged LICENSE](https://github.com/nwhitehead/pyfluidsynth/blob/v1.3.4/LICENSE).
  LGPL-2.1-or-later; copyright Nathan Whitehead and contributors. Only NumPy is
  required by this wheel. No PyAudio/device dependency is added.
  The downloaded wheel was inspected without installation/execution:
  `pyfluidsynth-1.3.4-py3-none-any.whl`, SHA-256
  `c6990329db7cfb35f5e65d523dd4f0c971d928e70df3a6bceec8864827edf246`.
  Its bundled `dist-info/LICENSE` is LGPL-2.1; embedded README specifies “or any
  later version”. PyPI's bare `license` field is null, not evidence of no license.
- Deliberately **not** selecting 1.4.0: its wheel metadata declares MIT while
  `dist-info/licenses/LICENSE` and README still say LGPL. Resolve that upstream
  discrepancy before upgrading. This is not a claim that the newer package lacks
  all licensing, nor a legal opinion.
- Native FluidSynth is a separate LGPL-2.1-or-later shared library; see the
  [official license](https://github.com/FluidSynth/fluidsynth/blob/master/LICENSE).
  It and each SoundFont have their own distribution obligations. We bundle
  neither. Keep notices/source availability obligations when redistributing
  dependencies; an optional extra does not waive their licenses.

### Authoritative API sources

1. [pyFluidSynth v1.3.4 “Managing Audio”](https://github.com/nwhitehead/pyfluidsynth/blob/v1.3.4/README.md#managing-audio):
   create `Synth`, **do not call `start()`**, call `get_samples(n)` for stereo
   interleaved frames, and `delete()` after use.
2. [Tagged public Synth API/docstrings](https://github.com/nwhitehead/pyfluidsynth/blob/v1.3.4/fluidsynth.py):
   constructor settings, `get_setting`, `sfload`, `program_select`, `cc`,
   `pitch_bend`, `noteon`, `noteoff`, `get_samples`, `delete`; public
   `fluid_version` version query. `pitch_bend` takes **-8192..8191**, unlike the
   native C API's unsigned 0..16383. Its semitone amount comes from RPN range,
   not the binding's historical fixed-range example.
3. [FluidSynth MIDI channel API](https://www.fluidsynth.org/api/group__midi__messages.html):
   CC/bank/program behavior and status returns. A `noteoff` failure may merely
   mean no voices matched (a drum/sample already decayed); this status is allowed
   only for already-paired note-offs. Other failed dispatch statuses abort.
4. [Synth settings](https://www.fluidsynth.org/api/settings_synth.html):
   MMA bank selection `CC0*128+CC32`, CPU cores, sample rate, effects and polyphony.
5. [Native audio rendering API](https://www.fluidsynth.org/api/group__audio__rendering.html):
   signed 16-bit frames/dithering. Public `get_samples` returns those samples;
   convert by `/32768` and write FLOAT WAV using existing SoundFile/libsndfile.

## Behavior and evidence limits

- A fresh synth per isolated source stem; no realtime driver/player/sequencer,
  sleeps, background playback, network, or fallback during rendering.
- Absolute tick → rational tempo-map seconds → round-half-even sample position.
  Stable same-tick order; blocks at most 1024 frames split at every event.
  This proves requested dispatch positions, **not** sub-block native acoustic
  latency (FluidSynth has its own internal synthesis buffering).
- Pending bank CCs latch at program changes, using the public `program_select`
  API and **MMA `CC0*128+CC32` on every channel**, including channel **9**.
  With neither bank controller received, program changes use bank 0 on melodic
  channels and bank 128 on channel 9. Explicit zero selects bank 0, not a default.
  CC0/CC32 are independent, channel-local state: an unseen half is zero; a partial
  update retains the other half. Thus CC32=3 alone selects bank 3; CC0=1 plus CC32=3
  selects bank 131; subsequent CC0=2 selects 259, then CC32=4 selects 260.
  Program changes and CC121 do not clear bank state. Bank CCs alone never change
  the current preset: without a program change, the initial note uses program 0
  in the default bank. Channel 9 retains drum keys. Missing exact bank/program
  presets fail; the adapter never substitutes bank 128 for an explicit bank.
- All CCs, including sustain and RPN select/data, are delivered in source order;
  pitch wheels remain signed. No hand-reconstructed RPN approximation.
- SysEx (including MTS/reset) and pressure messages fail before allocation. The
  pinned public `Synth` has no pressure methods; we do not bypass it with private
  handles. Events are still preserved by groove/isolation, not silently discarded.
- The registry advertises `channel_pitch` **only for native 2.6.2** with the
  pinned 1.3.4 binding, after the real fixtures below passed. Other native versions
  remain `direct_12` only. No general MPE/MTS capability is advertised.
- Fixed settings: gain 0.2, 16 MIDI channels, polyphony 256, one CPU, one stereo
  output/group, reverb/chorus off, minimum note length 0, MMA banks, dynamic sample
  loading off. Public `get_setting` readback must match. Settings/code/native and
  binding versions enter cache identity; sample rate, duration/tail, actual frame
  count, post-render peak gain and hashes enter per-stem sidecars. Preset envelopes
  and supported CC effects still depend on the supplied SoundFont.
- Cleanup is guaranteed for each returned `Synth`, including load, setting,
  dispatch and sample errors. A cleanup exception cannot replace a primary error.
  WAV writing happens after native cleanup. Constructor-internal failures and
  native process crashes remain limitations of the binding, not recoverable Python
  render failures; no native success is published after such a failure.

## Common horizon and rough mix

`render.duration_sec` optionally extends (never shortens) the greater of MIDI EOT
and planner bar extent. Add `render.release_tail_sec` (default **0.25 s**) once;
use `ceil((horizon + tail) * sample_rate)` for every source WAV, mix, and click.
A short backend output is zero-padded; an overlong one is rejected, not truncated.
This fixed tail may cut off long SoundFont envelopes: choose a longer tail when
needed rather than claiming complete decay. Silent declared tracks remain present;
no-note native output is zeroed to avoid treating int16 dither as music.

`render.mix_peak` defaults to **0.95** (linear amplitude, finite and in `(0, 1]`).
Stems are capped before hashing. Sum each source stem once in source order,
including distinct same-role stems, average stereo channels for the existing mono
rough mix, and apply **one attenuation-only gain if needed**. Do not include the
separate click. Quiet content/silence is not boosted by mixing. Simple's legacy
preview normalization remains separate. `headroom_dbfs`/`peak_dbfs` still control
simple preview/click level, not rough-mix normalization. Layout revision 3
invalidates older fixed-normalization/unequal-length caches; filenames are unchanged.

Tests check designated nonempty **active windows** have RMS >1e-5, finite samples,
peak <=0.95, equal frames and fixed tails; exact-zero silent stems, click exclusion,
stereo handling, padding, rate mismatch, deterministic mix bytes and error cleanup
are also covered. The asset-free cases use simple/mock signals; the separate native
lane measures all four demos and controlled mono f0. Neither lane certifies
CLAP/FAD, LUFS mastering or subjective quality.

## Installation: Python, native library and asset are separate

The historical measured renderer runtime below is macOS. Native Windows
**asset-free publication** and **optional pinned renderer evidence** are separate
lanes; see [Windows verification](audio-testing.md#windows-verification-simulation-required-native-ci-optional-renderer).
The required lane only installs `.[audio,dev]` and does not require a synth or
asset. A passing publication job does not establish Windows FluidSynth sound.

### Linux (Debian/Ubuntu example)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[audio-render]"
# Optional native library, installed separately by the user:
sudo apt-get install --no-install-recommends libfluidsynth3
export AIMUSIC_SOUNDFONT=/absolute/path/to/your-approved.sf2
sha256sum "$AIMUSIC_SOUNDFONT"
```

Use your distribution's shared library package on other systems; official
[distribution instructions](https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions).
The package does not acquire a SoundFont. Record the library package version and
user-selected asset's source URL, license text/permission, version and SHA-256.

### Windows (PowerShell)

For asset-free verification, use the wheel-building PowerShell command in
[audio-testing.md](audio-testing.md#required-asset-free-windows-job). It needs no
administrator rights or Developer Mode and exercises real NTFS, not API
simulation. Use a trusted local NTFS directory; network shares and reparse roots
remain unsupported by [publication policy](audio-publication-safety.md).

For **optional renderer evidence**, separately obtain an official
[FluidSynth 2.6.2 release](https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.2)
matching Python 3.11 x64, including dependent DLLs. Review the binary source and
record its SHA-256 before running it. A CLI executable alone is not the shared
library. No Windows binary or SoundFont is bundled or acquired by our scripts.

Pinned pyFluidSynth 1.3.4 unconditionally registers **`C:\tools\fluidsynth\bin`**
on Windows, so that upstream-required directory must exist and contain the
trusted release DLLs. It is an upstream search-path requirement, not a tracked
author-machine path. Provision it separately using your normal security policy;
**the required asset-free CI job never creates or needs it**. Keep DLLs and the
working directory trusted (the binding also searches the working directory).
Do not install the unrelated PyPI package named `fluidsynth`.

The optional script builds a fresh wheel/venv, installs `[audio-render,dev]` and
SoundFile 0.14.0, and requires the queried libsndfile version 1.2.2. Provide the
approved asset manifest, exact DLL path and reviewed digest as shell variables;
the enabled lane fails on missing/wrong prerequisites instead of skipping.
Other Python packages are recorded, not fully version-locked. Preflight needs
only Python and no asset, native library, package installation, or secret:

```powershell
python -S -m tests.windows_fluidsynth_evidence --preflight
```

The wrapper identifies the actual DLL containing the binding's public
`fluid_version` function via
[GetModuleHandleExW (FROM_ADDRESS, UNCHANGED_REFCOUNT)](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-getmodulehandleexw)
and [GetModuleFileNameW](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-getmodulefilenamew).
It checks file identity and SHA-256 against your explicit DLL input; the API
version alone cannot identify one of two same-version binaries. OS build,
architecture, Python, binding/DLL/libsndfile versions, asset digest, commands,
outputs and same-machine cache/repetition evidence remain in **external** logs.
This is not new Windows listening approval or cross-platform byte identity.
Never track runtime paths, DLLs, SoundFonts, generated WAVs, credentials or caches.

### Opt-in selection

`load_audio_config('listening')` selects FluidSynth and forbids simple fallback;
use `paths.soundfont` in an explicit profile or `AIMUSIC_SOUNDFONT`. See
[audio-config-skeleton.md](audio-config-skeleton.md) for precedence. Both `render-audio` and `render-midi` support `--profile listening`, `--soundfont`,
`--out` and `--no-validate-only`; see [demo commands](audio-demo-sources.md). Missing bindings/library/asset
fail with setup guidance. Native execution defects never trigger a bridge.

The user approved the local-only GeneralUser GS asset below. Other assets still
require explicit license review/approval. Keep WAVs, SoundFonts and credentials untracked.

## Approved local-only asset / installed runtime (2026-11-02)

- **GeneralUser GS 2.0.3**, author Chris (S. Christian) Collins / `mrbumpy409`.
  Official repository: <https://github.com/mrbumpy409/GeneralUser-GS>.
  Pinned commit: `684543d5e5efaef08d02be50dcda8d552478fa60`.
  Filename: `GeneralUser-GS.sf2`; SHA-256:
  `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe`.
- [License v2.0 at that commit](https://github.com/mrbumpy409/GeneralUser-GS/blob/684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt)
  permits unrestricted private/commercial music creation, software-project use
  and modification, and says all contained samples allow full music-production
  use, including profit from recordings. **Caveat:** the author discloses historical
  uncertainty about some samples' origins and relies on original suppliers'
  rights information. Do not translate this into an unconditional redistribution
  warranty. User approval is **local-only**; no SF2/sample/WAV is bundled here.
- Measured on Apple **M4 Pro**, macOS **27.2 arm64**, Python **3.12.12**;
  Homebrew FluidSynth/shared libfluidsynth **2.6.2**, pyfluidsynth **1.3.4**,
  NumPy **2.5.4**, mido **1.3.3**, SoundFile **0.14.0**, libsndfile **1.2.2**.
  Bindings were installed in the user's external virtual environment. On this
  Homebrew installation, `HOMEBREW_PREFIX=/opt/homebrew` is needed for discovery.
  These are versions actually queried by the native lane, not CLI-version guesses.

macOS setup (separate from Linux/Windows above):

```bash
brew install fluidsynth             # native library, NOT a SoundFont
python -m pip install -e '.[audio-render,dev]'
export HOMEBREW_PREFIX="$(brew --prefix)"
```

No automatic asset installation. Store its absolute local path only in the
external manifest described in [audio-testing.md](audio-testing.md).

Cold-repeat verification found identical PCM but a changing libsndfile FLOAT-WAV
`PEAK` timestamp. Owned published WAVs now canonicalize only that timestamp to
zero, scanning RIFF chunk lengths/padding (not searching audio bytes). Samples,
peak values and positions remain unchanged. Asset-free tests inject different
wall times; the native lane checks all six cold-repeat and warm-restored WAVs
byte-for-byte. Cross-platform/native-version byte identity is not promised.

## Proven channel-pitch subset (native 2.6.2 / binding 1.3.4)

`channel_pitch` is **not general MPE**. Supported source events match
`demo_sources reference-19` / `multitrack-19`:

- Every melodic channel explicitly sends the complete ordered transaction
  **CC101=0, CC100=0, CC6=2, CC38=0, CC101=127, CC100=127**.
  Only this RPN0 **±2 semitones, 0 cents** range is proven.
- Program setup; a fresh signed pitchwheel before every note; at most one
  key sounding per channel; wheel/range/program changes only on idle channels.
  Notes may sound simultaneously on independent channels in one logical stem.
- CC10 pan is the sole additional admitted controller, used to separate left/right
  monophonic windows in the single-synth simultaneous-channel proof. It does not
  broaden pitch semantics. Conductor metadata remains preserved.
- No sustain, modulation, pressure, live bends, NRPN, increment/decrement,
  fine/coarse tuning, other ranges, zone/master expression, SysEx or MTS.
  Incomplete/out-of-order range setup, missing wheels and channel overlaps reject.
  This deliberately rejects many otherwise valid MIDI/MPE files rather than
  advertise unmeasured semantics. Use direct-12 for its separately supported
  instrument/controller path, never relabel microtonal content to bypass admission.
- Registry modes are version-pinned; original and expressive source preflight,
  per-request execution and direct adapter calls enforce the subset. Unsupported
  tuning cannot use simple fallback. MTS and conflicting channel ownership fail
  before successful WAV/cache publication.

Event evidence in `render.json` and stem sidecars includes the subset revision,
track/channel/key/tick/wheel and reconstructed Hz, with original tuning metadata
retained. It enters the render-cache fingerprint. MIDI-event checks compare the
original fixture's **440 × 2^(step/19)** targets, after real groove and isolation,
within **1 cent**. Inferred imports still have no known intended lattice; emitted
frequencies are event reconstruction, **not** automatically an audio-f0 certificate.

The real dry reference uses the original **GM program 80** (zero-based square
lead), without changing fixture bytes/pitches or choosing an easier target.
GeneralUser's sample timbre passes the existing checker; no alternate preset,
attack-window adjustment, tuning correction or relaxed threshold was needed.
Both simultaneous voices are also program 80, using opposite-signed wheels and
hard-pan CC10 in one synth. See [measured results](audio-testing.md).

Additional local runtime-byte provenance (not a cross-platform requirement):

- Installed `fluidsynth.py` SHA-256:
  `45fdf0b9e00b77080d6c5e0a1d57498d27a6b3504d3a34225a1e2334ab7a42a4`.
- Homebrew libfluidsynth 2.6.2 dylib SHA-256:
  `9c6732846300884c1d001e0d6f98acd9314c6a59cc096568db815570855e9dc8`.

Exact external paths and Python package freeze are retained in local
`runtime-files.json` / `pip-freeze.txt`, not tracked. No licensed asset is copied.
