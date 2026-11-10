# Tasks

Implementation and approved final review are complete: **42/42 tasks**. The proposed commit names below are planning checkpoints; actual evidence and continuation commits are recorded in `docs/audio-testing.md`. Only checked boxes are delivered. No Linear updates, pushes or PR creation are authorized here.

Each implementation checkbox is a focused slice with its own tests. Write failing regression tests first, implement, then verify without deleting assertions or weakening floors. The named new test files are planned outputs, not existing coverage. `python` below means the interpreter in the chosen virtual environment; this planning shell exposes `python3`, not `python`.

Dependency chain: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11**. Group 3 config work can run alongside group 2 only with coordinated edits to shared files; otherwise keep the linear sequence. Each group after baseline ends in a tested commit, and groups 5, 7, 9 and 11 have human review gates.

## 1. Baseline and isolated execution (no implementation commit required)

Scope: existing branch/worktree, environment and verification records. Do not recreate completed MUS-30–33 work.

- [x] 1.1 Confirm explicit implementation approval, branch base and a clean isolated worktree on `feature/midi2audio-audible-demo` or its approved continuation; verify `git status --short` preserves the existing unrelated `AGENTS.md` and tooling changes outside the implementation diff.
- [x] 1.2 Establish the core and lightweight-audio environment without native assets; run `python -m pytest tests/test_architecture.py tests/test_render_package.py -q`, `python -m unittest discover -s tests -p 'test_*.py'`, `flake8 aimusic tests ui.py`, and `mypy aimusic tests ui.py`; record versions and any pre-existing failures separately, without claiming skipped tests passed.
- [x] 1.3 Record three verification lanes and the real-lane opt-in contract in `docs/audio-testing.md`; verify the default lane requires no FluidSynth/SoundFont/keys and the requested native lane will fail missing prerequisites. Obtain the user's local license-approved SoundFont choice before native acceptance, not by downloading an arbitrary asset.

## 2. Package integrity and planner reconciliation — commit C1

Suggested message: `fix(audio): verify RenderPackage integrity and reconcile source MIDI`

Files: `aimusic/core/render_package.py`, `aimusic/audio/from_score.py`, `entry.py`, `analysis/from_midi.py`, new `analysis/reconcile.py`, focused tests and contract docs. Split each numbered slice if it exceeds a focused session; commit only once this vertical package-validation path is green.

- [x] 2.1 Add aggregate hash recomputation and additive per-file manifest digests while retaining valid v1 identities; verify new tests in `tests/test_render_package.py` cover score/MIDI/structure/tuning/beatstates corruption, manifest aliases, reused directories and legacy aggregate-only packages, with expected/actual digest diagnostics.
- [x] 2.2 Carry parsed tuning, verified identity and authoritative roles/count sources in the audio-owned context; verify `tests/test_audio_entry.py` proves full rendering receives planner data, validate-only allocates no synth, and architecture imports remain quarantined.
- [x] 2.3 Implement field-level reconciliation against original MIDI, aggregating MPE channels by source track and distinguishing Score duration from trailing planner silence; verify new `tests/test_audio_reconcile.py` covers exact discrete fields, 0.001 BPM and one-tick tolerances, meter/tempo maps, mismatched programs/counts, unavailable inferred roles and invalid drum metadata.
- [x] 2.4 Document aggregate versus per-file diagnostics and reconciliation tolerances in `docs/render-package-contract.md` / `docs/audio-testing.md`; run focused contract, entry, reconcile and architecture tests and inspect the C1 diff before committing.

## 3. Wheel-safe effective configuration — commit C2

Suggested message: `fix(audio): package profiles and normalize effective routing`

Files: `audio/config.py`, `audio/orchestrator.py` config conversion, new `audio/resources/` profiles, `pyproject.toml`, new config/wheel tests and setup docs.

- [x] 3.1 Ship resource-backed defaults plus `preview`/`listening` profiles and retain effective settings rather than rereading `source_path`; verify `tests/test_audio_config.py` covers packaged resources, explicit YAML profiles and source-path-independent config conversion.
- [x] 3.2 Normalize `routing`/legacy `role_map` and `comp`/`comping`, rejecting contradictory aliases; fix imported pad/lead heuristic precedence and verify config/analysis tests cover identical aliases, conflicts and program boundaries.
- [x] 3.3 Implement CLI-ready SoundFont resolution with explicit > profile/config > `AIMUSIC_SOUNDFONT` precedence, config-relative paths and actionable invalid-path errors; verify mocked config tests include a bad higher-precedence path and do not access a native library.
- [x] 3.4 Add an isolated wheel smoke test that builds/installs the wheel and loads defaults/profiles with checkout and `PYTHONPATH` excluded; verify new `tests/test_audio_wheel.py` and document profile/asset precedence in audio setup docs before committing C2.

## 4. Event-preserving groove — commit C3

Suggested message: `fix(audio): preserve MIDI channel state through groove transforms`

Files: new `audio/midi_events.py`, `audio/groove/apply.py`, event fixtures/tests and groove documentation.

- [x] 4.1 Introduce lossless absolute-tick event parsing/serialization and note pairing; verify new `tests/test_audio_midi_events.py` covers type-0/type-1, stable same-tick order, tempo/meter changes, RPN, bends, sustain, aftertouch and MTS SysEx round trips, rejecting ambiguous unsupported arrangements.
- [x] 4.2 Replace note-only PrettyMIDI reconstruction with tempo-aware event transforms and round-half-even tick handling; verify existing groove tests plus new cases retain all non-note event payloads, expected note count/velocity behavior, tempo maps and note-associated expression.
- [x] 4.3 Detect retiming conflicts involving bend trajectories, channel reuse and sustained notes; verify deterministic overlap fixtures reject incompatible pitch states and safe 19-EDO retiming reconstructs pitches within one cent without changing intended note identities.
- [x] 4.4 Document supported expression semantics and explicit conflict diagnostics; run the groove/event and architecture suites, preserving existing property tests, before committing C3.

## 5. Safe event-preserving stem isolation — commit C4 / review gate A

Suggested message: `fix(audio): isolate logical stems without event or artifact collisions`

Files: `audio/render/stems.py`, `audio/midi_events.py`, stem tests/fixtures and output-layout docs.

- [x] 5.1 Isolate source-track stems with conductor events and complete channel setup, consolidating MPE member channels and retaining authoritative roles; verify `tests/test_audio_stems.py` compares programs, drum semantics, sustain, tempo and pitch-state trajectories before/after extraction and rejects conflicting cross-track channel ownership.
- [x] 5.2 Introduce stable identity-based safe names and a role index; verify duplicate, reserved `click`/`rough_mix`/`bar_boundaries`, Unicode, separators, traversal and pre-existing symlink cases cannot overwrite or escape output roots, and every source stem remains included in the mix.
- [x] 5.3 Document the stem naming compatibility change and generated-artifact namespace; run all lightweight audio tests plus architecture checks and present event-fidelity evidence for human review before committing C4 and proceeding to the native-render stage.

## 6. Registry, fallback and complete cache identity — commit C5

Suggested message: `refactor(audio): route stems through verified renderer and cache contracts`

Files: `audio/render/registry.py`, `stems.py`, `orchestrator.py`, `bridge.py`, renderer/cache tests and provenance docs.

- [x] 6.1 Define renderer request/result and capability contracts, resolve each role before cache lookup, and route simple rendering through the registry; verify mocked `tests/test_audio_registry.py` proves role selection and rejects unsupported tuning rather than flattening it.
- [x] 6.2 Remove catch-all native-to-m2a failover while retaining explicit backend selection and compatible missing-resource fallback; verify bridge/registry tests distinguish missing dependencies, native runtime defects, listening no-fallback and requested/effective backend records.
- [x] 6.3 Expand fingerprints to verified inputs, metadata, effective config/routes, fallback policy, renderer versions and SoundFont bytes; verify orchestrator tests mutate each input independently, including same-path asset replacement and changed tuning/structure, and require cache misses.
- [x] 6.4 Publish complete cache manifests last, verify output digests before reuse and restore provenance sidecars; verify interrupted/corrupt-cache tests and byte-identical warm restores, document cache invalidation and effective-backend fields, then commit C5 with the full lightweight lane green.

## 7. FluidSynth instrument path — commit C6 / review gate B (MUS-34)

Suggested message: `feat(audio): add optional FluidSynth instrument rendering`

Files: new `audio/render/fluidsynth_r.py`, registry adapter, `pyproject.toml`, profiles, native/mocked tests and native setup docs. The exact pinned adapter was reviewed and found to have no discoverable license; do not add a submodule or copy its source. Implement clean-room from public APIs per the user's 2026-11-01 approval.

- [x] 7.1 Record the adapter/license/tests review at `midi2audio_generative@46b3dfe`, including the absence of discoverable licensing, and add optional `[audio-render]` bindings without changing core dependencies; verify wheel metadata and that the clean-room implementation copies no source from the pinned repository.
- [x] 7.2 Implement offline event-to-sample dispatch with native lifecycle cleanup, programs/banks, drums, sustain and pitch-wheel/RPN handling; verify mocked `tests/test_audio_fluidsynth.py` checks call order, tempo-map timing, resource cleanup and errors without native assets.
- [x] 7.3 Align frame counts to one configured horizon/release tail and implement deterministic mix headroom; verify stem/mix tests cover tail, finiteness, 0.95 default peak ceiling, active-window RMS >1e-5, click exclusion and preservation of silent tracks.
- [x] 7.4 Add opt-in `tests/test_audio_fluidsynth_integration.py` with a pinned local SoundFont hash/version record; verify an explicitly requested lane fails missing prerequisites, then run the eight-bar 12-EDO fixture checking roles/programs, sample rate, equal frames, RMS/peaks, cold-repeat/warm-cache evidence and elapsed time against the ~60-second target.
- [x] 7.5 Document Python/native/asset installation separately for Windows and Linux and record the asset license; run all lightweight checks without the native asset, present 12-EDO stems/mix for human listening, and commit C6 only with evidence sufficient to propose MUS-34 completion (do not update Linear automatically).

## 8. Supported tuning capability enforcement — commit C7

Suggested message: `feat(audio): enforce supported MPE tuning and reject lossy routes`

Files: new `audio/render/tuning.py`, context/registry integration, MPE fixtures and tuning-policy tests/docs.

- [x] 8.1 Derive required tuning from Score/EDO/base tuning and source channel events; verify new `tests/test_audio_tuning.py` covers ordinary direct-12, misleading fractional-base labels, pitch bends and nonstandard RPN range, keeping percussion drum-key based.
- [x] 8.2 Prove the FluidSynth per-channel MPE capability using a controlled native fixture and forbid simple fallback for it; verify unsupported MTS, ambiguous channel ownership and unavailable MPE backend failures occur before successful output publication.
- [x] 8.3 Carry tuning identity into manifests/cache and compare expected versus event-reconstructed pitch after groove and isolation; verify the 19-EDO path stays within one cent, document the exact supported MPE subset and unsupported MTS path, and commit C7 after lightweight and targeted native tests pass.

## 9. Controlled dry-render tuning evidence — commit C8 / review gate C (partial MUS-35)

Suggested message: `feat(audio): validate dry 19-EDO renders with tuning evidence`

Files: new `audio/scoring/tuning_check.py`, controlled-tone source/expected-frequency fixtures, checker/integration tests and metric docs.

- [x] 9.1 Implement deterministic monophonic-window f0 analysis and structured expected/measured/coverage reports; verify new `tests/test_audio_tuning_check.py` with synthetic exact, detuned, silent, insufficient-confidence and percussion cases, without transcription/model dependencies.
- [x] 9.2 Enforce controlled per-tone <=5-cent acceptance and default mean/P95 <=15-cent hard floors, honoring stricter configured limits; verify boundary tests, flattened-12-TET corruption and missing-evidence rejection use original targets rather than inferring new targets from the bad output.
- [x] 9.3 Run native dry 19-EDO reference tones through groove-disabled and groove-enabled paths; verify intended frequency windows pass and a deliberately flattened version fails, record asset/runtime hashes, and label polyphonic frequency accuracy unmeasured.
- [x] 9.4 Document evidence limits and remaining MUS-35 Scala/helpers/diffusion-gating work; present reports and controlled listening samples for human review and commit C8, explicitly leaving MUS-35 partially delivered and later blocking relations intact.

## 10. Imported MIDI and reproducible CLI examples — commit C9

Suggested message: `feat(audio): expose safe MIDI demo workflows and listening examples`

Files: `aimusic/app/cli.py` lazy dispatch, `audio/entry.py`, inferred-input adapter, CLI safety tests, example generators and reproduction docs. Keep changes inside the audio boundary except lazy CLI glue.

- [x] 10.1 Add audio-owned imported-MIDI context with provenance, tempo/meter maps, programs and tuning evidence; verify type-0/type-1 fixtures render without pretending to have planner reconciliation, while malformed/type-2/unsupported arrangements fail clearly and leave sources unchanged.
- [x] 10.2 Add lazy `render-midi`, output/SoundFont options and named profiles to both entry paths while preserving validate-only defaults; verify `tests/test_audio_entry.py` and new CLI tests exercise help, successful explicit rendering, missing dependencies, invalid inputs, safe outputs and core-only imports.
- [x] 10.3 Add reproducible source generators/commands for the eight-bar fixture, fresh seeded 12-EDO passage, 19-EDO reference tones and short 19-EDO multitrack passage; verify each produces source/expressive MIDI as applicable, stems/mix, config/asset/backend provenance and applicable reconciliation/tuning reports.
- [x] 10.4 Document executable local commands, seed/runtime/asset requirements and listening checklist; verify example sources/profiles/expected metrics are versioned while WAVs/SoundFonts/credentials are not, then commit C9 with CLI and packaging smoke tests passing.

Asset-free continuation (2026-11-02): `aimusic/audio/demo_sources.py` and
`docs/audio-demo-sources.md` provide three versioned fixture generators plus the
existing seeded Method A command, exactly the four requested sources. Tests
exercise byte repeats, independent Hz/windows, expressive event integrity,
exclusive output publication, real 12-EDO preview stems/mixes and installed-wheel
commands. That checkpoint left **10.3/10.4** open for native evidence. The later
approved GeneralUser GS 2.0.3 / FluidSynth 2.6.2 / binding 1.3.4 run now retains
all four workflows; see `docs/audio-native-listening-evidence.json` and
`docs/audio-testing.md` (four native workflows). That checkpoint claimed no human
listening approval. The later 2026-11-02 [human acceptance](../../../docs/audio-listening-review.md)
completes **7.5 / 9.4**: all four embedded players worked and rough demos were
accepted, with no specific audible defect reported. The subsequent named timing
pair review on the same date completes **11.3**: the user could not perceive a
difference between `timing_original.wav` and `timing_shifted_12_5ms.wav`, then
explicitly reported “No defects heard” when asked about clipping, stuck/lost
notes, truncation or broken timing. A uniform 12.5 ms shift may be imperceptible;
no aesthetic groove improvement is claimed. **11.4 stayed open at that checkpoint**;
the subsequent approved final review below completes it.
The demos remain rough and dry, not mastered, with no CLAP/FAD scores and
polyphonic f0 unmeasured. MUS-35 remains partial and Linear is unchanged.

## 11. Final cross-lane verification — commit C10 / review gate D

Finalization message: `test(audio): finalize audible demo acceptance`

Scope: integration evidence and any integration-only harness adjustments, not delayed unit coverage. No quality floor may be relaxed to finish this milestone.

- [x] 11.1 Re-run core/architecture, all lightweight audio, lint/type checks and clean-wheel commands from group 1/3 in an asset-free environment; verify no new failures or quarantine violations and record separately any unchanged baseline failures.
- [x] 11.2 Reproduce all four examples in the pinned native lane, recording signal metrics, reconciliation, controlled tuning, provenance, cache equivalence and elapsed time; verify unsupported tuning still fails and no skipped native test is presented as acceptance evidence.
- [x] 11.3 Obtain human listening review of recognizable 12-EDO roles, groove on/off timing and the supported 19-EDO examples; record findings and the explicit limitation that these are rough audible demos, not mastered/CLAP/FAD-scored production audio.
- [x] 11.4 Review the final diff and per-capability acceptance mapping, confirm no unrelated user files or prohibited assets are staged, and commit only the integration evidence/harness changes as C10; present MUS-34 completion evidence, partial MUS-35 status and remaining follow-up scope without changing tickets or opening PRs.

Finalization (2026-11-03, after `ed06d58` / `134ff43`): no Critical/Required
findings; reviewer authorized **11.4**. [Per-capability acceptance and verification](../../../docs/audio-testing.md#final-acceptance-review-114)
record **376 asset-free passes / 6 optional skips**, **5 native passes / no skips**,
and implementation full unittest **714 run / 14 optional skips**; lint/type/strict
OpenSpec clean. Reviewer independently confirmed the two lanes, exact authoritative
**[12,252]** ticks, unchanged four mixes/timing WAVs/listening-page bytes and no
tracked author-machine paths or prohibited assets/credentials/submodules. External
evidence: `fluidsynth-evidence-afttn_6e`. This finalization changes only **11.4**
to checked; no implementation scope or quality floor changed.

## Workflow follow-up

- Final review is approved; archiving remains a separately authorized follow-up.
- MUS-34 completion may now be proposed, without mutating Linear. MUS-35 remains partial (Scala/MTS helpers and diffusion gating); polyphonic f0, later scorers/restyle/mastering remain outside this milestone. MUS-35 and MUS-29 must not be closed by this partial milestone.
- Preserve MUS-34 → MUS-35 → MUS-36 → MUS-37 → MUS-38; do not begin restyle before the scorer milestone merges.
- Fork-first PR delivery is separate from this task list; any org PR to `iCog-Labs-Dev/musicGeneration` requires an explicit user request.
