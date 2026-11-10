# Verified renderer and cache boundary (C5 / C6 partial)

All contracts live in `aimusic.audio`; core imports and validate-only remain
asset-free. `render_role_stems` first builds a `StemRenderPlan`: isolated source
tracks, canonical roles (`comp` → `comping`), tuning requirements and **resolved**
renderers. Orchestration does this even on a warm cache lookup. There is no
hard-coded simple-render bypass.

## Renderer contract and failure policy

`audio/render/registry.py` defines frozen `RenderRequest`, `RenderResult`,
`RendererCapabilities`, `RendererBackend` and `ResolvedRenderer` records.

- Requests carry event-preserving isolated MIDI, effective tuning, sample rate,
  common duration/release-tail intent, output path, config and resolved asset.
- Providers probe resources without allocating a synth and return backend/code/
  native versions plus capabilities. Adapters classify only known resource-load
  failures as `MissingRendererResource`. FluidSynth DLL-load `OSError` is handled
  at import; arbitrary import defects and synthesis-time errors propagate.
- Routing uses `render.routing.<canonical-role>.default` or `.microtonal`, then
  the global backend for a missing route. Unknown backend names are errors.
- Only an explicit `allow_simple_fallback: true` permits selection-time fallback,
  and only if simple supports the request. `listening` forbids fallback.
- Execution never retries with a different renderer, even if a resource disappears
  after selection. Each result must be a regular WAV at the requested path with
  nonempty finite samples and the requested sample rate. Short outputs are padded
  to the common frame count; overlong outputs fail. Peak limiting precedes hashing.
- Application backend selection is separate: native is the default;
  `AIMUSIC_AUDIO_BACKEND=m2a` explicitly selects the optional bridge. Native defects
  never trigger bridge failover. Unknown environment values now fail clearly.

The lightweight default remains the **simple integer-pitch sine preview**.
Its capability is `direct_12`, not instrument/controller fidelity. Task **8.1**
adds `audio/render/tuning.py`: requirements use Score heights (`base + h*12/EDO`),
all applicable metadata layers and source channel events. A `direct_12` label
cannot override fractional base tuning, any nonzero bend, or nonstandard RPN
bend range/fine/coarse tuning. Integer base anchors remain on the 12-TET lattice.
A known integer-pitch Score subset can use direct-12; without Score evidence,
non-12 EDO metadata is conservatively gated.

Centered wheel resets and explicit defaults (2-semitone range, zero fine/coarse
offset) are neutral. **The former 48-semitone centered-wheel exception is removed.**
Nonstandard values require `channel_pitch`, even if later reset. RPN selection
is channel-local, null selection is a no-op, and unknown RPN/NRPN writes require
`unsupported`. Per-channel MPE exports require `channel_pitch`; explicit MPE zone
configuration requires the distinct `mpe` capability. MTS SysEx/metadata require
`mts`, while other SysEx requires `sysex`. Neither backend advertises zone `mpe`, `mts`, `sysex` or `unsupported`.
FluidSynth 2.6.2 now advertises only the proven `channel_pitch` subset below. Percussion stays drum-key based and is excluded from
pitched intent; global unverified SysEx is still gated.

Orchestration preflights original MIDI before groove, then checks expressive MIDI
again before render-cache lookup. Native admission also validates the exact
channel-pitch subset before groove/cache, and independently before native allocation.
Source-fixture original-Hz tests compare event reconstruction after groove/isolation
within one cent; this is not a general Score/imported-MIDI or audio-f0 certificate.

C6 tasks 7.1–7.3 add the [clean-room FluidSynth adapter](audio-fluidsynth.md),
optional pinned bindings, common horizon/fixed tail, aligned WAVs and attenuation-only
0.95 mix ceiling. Providers query versions without allocating a synth; fixed native
settings enter identity and are checked on creation. The opt-in native lane now
measures real 12-EDO and controlled 19-EDO output;
[results and limits](audio-testing.md) do not constitute human listening acceptance.

SoundFont selection reuses config precedence (`paths.soundfont`, then
`AIMUSIC_SOUNDFONT`; the resolver also supports explicit CLI input). An explicitly
invalid higher-precedence path is an error, not an excuse to choose another asset.
Asset **bytes**, never just a filename, enter identity. The exact resolved asset
is checked before and after each backend call, before cache publication, and
after warm restoration. Detected mutation aborts; it never publishes under the
old asset digest. These checks do not lock a SoundFont against a writer that
changes and restores it between checks. No SoundFont is needed for the default
preview.

## Identity and provenance

`render.json` at the run root (also in `stems/`) records:

- verified source/package and structure/tuning digests when a package is supplied;
- original and expressive MIDI digests and semantic planner/inferred analysis;
- normalized config/routes, fallback policy and duration/tail settings;
- each source identity/role and its `requested_backend`, `effective_backend`,
  `fallback_reason`, required tuning and capability/backend/code/native versions;
- SoundFont content digests, audio source-code digests and runtime identity;
- per-native-channel-pitch stem `tuning_evidence`: exact subset revision,
  source-track/channel/key/onset/wheel and reconstructed Hz. Stem WAV sidecars
  retain `required_tuning`, original tuning metadata, and the same event evidence.
  This data participates in the render fingerprint and verified warm restore.

The orchestrator copies original MIDI or all package payloads into private
scratch **before** parsing, package validation or reconciliation. It checks both
originals and snapshots at cache boundaries; detected mutation aborts before a
complete entry is published. A stem plan parses and hashes the same MIDI byte
buffer, and execution rejects a different input. No stage rereads user output
files as its input.

Runtime identities are stage-specific: analysis includes `mido`, `numpy`,
`pretty_midi` and `six`; expression includes `mido`, `pydantic`, `pydantic_core`,
validation support packages and `PyYAML`; rendering includes MIDI/audio dependency
versions plus each backend's declared native-library versions (including
`libsndfile` for simple). Missing optional transitive packages have an explicit
`not-installed` identity, not a newly required dependency.

Per-WAV `.json` files repeat the actual renderer/capability/asset identity,
`midi_sha256` and `output_sha256`, using relative output names. They also include
render settings, final aligned frame count and the applied peak-limiting gain. Application
backend provenance is separate from per-stem renderer provenance. Output/cache directory paths, config
source paths and SoundFont display paths are excluded from the fingerprint; equal
asset bytes at a new path can reuse a result. Upstream cache artifacts use their
final basenames, included in those stage identities, so sidecar `output_path`
and `diff_report` references resolve beside the sidecar on cold and warm runs.
No provenance serializes temporary `.pending-*` or run-scratch paths.
Legacy/canonical equivalent routes normalize to the same identity. No cross-platform byte-equality claim is made.

## Complete caches, safe restoration

Analysis, expression and render use full SHA-256 directory keys. Producers run in
unique `.pending-*` directories, never the final key directory. Every expected
artifact (including sidecars) must exist as a regular file; the exact output set
and digests are verified before `cache_manifest.json` is written **last** and the
entry directory is atomically renamed into place.

Reuse requires a complete manifest, matching fingerprint/key, exact expected
files and matching SHA-256 for **every** file. Old `_complete` markers, missing or
modified WAVs/sidecars, invalid manifests, and symlink entries are misses. A killed
process may leave a `.pending-*` scratch directory; it is never selected as a
cache hit and may be removed when no producers are running.

Warm copies retain the original bytes of WAVs and all provenance. Every stage
is copied into private scratch and reverified against the requested fingerprint
and every output digest before use. Substituting another valid entry also fails.
The run preflights all artifact names (including package reports and sidecars)
and pins both output directories before stage execution. All computation stays
outside the output tree; complete bundles publish exclusively through the same
pinned mechanism, with the stem cache manifest last. Existing files, symlinks
and hardlink/source aliases are refused, not overwritten; choose a fresh output
directory to rerun. Existing C4 safe-stem naming,
role index and `stems/rough_mix.wav` remain unchanged.

## Asset-free evidence

```bash
python -m pytest tests/test_audio_tuning.py tests/test_audio_registry.py tests/test_audio_orchestrator.py tests/test_audio_bridge.py tests/test_audio_stems.py tests/test_audio_stem_publication.py tests/test_audio_run_safety.py -q
```

Coverage includes canonical role overrides, forbidden tuning/fallback, missing
resources versus execution defects, resource availability changing before warm
lookup, each identity component, same-path SoundFont replacement, valid package
metadata changes with identical MIDI, corrupt/missing sidecars, process death,
invalid WAV results, run-wide output collision safety, byte-identical warm
restoration, MIDI/last-stem SoundFont/package mutation races, mixed package
snapshots, stable sidecar references and dependency-only invalidation.
These are contract/cache checks, **not** FluidSynth or audible tuning acceptance.
