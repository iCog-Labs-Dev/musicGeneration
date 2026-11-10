# Design

## Context

See `proposal.md` for motivation and the four delta specs for acceptance behavior. This is planning only; no baseline test pass or real audio measurement is claimed by this session.

### Observed repository state

Read-only inspection was against `feature/midi2audio-audible-demo`, HEAD `a0e2532` (the supplied plan). Existing unrelated changes include `AGENTS.md` and untracked tooling/OpenSpec configuration; implementation must preserve them.

| Evidence | Consequence for this design |
| --- | --- |
| `audio/entry.py` loads a package then passes only MIDI to orchestration; `from_score.py` omits parsed tuning | Carry full audio-owned context, not another MIDI-only entry |
| `core/render_package.py:358-485` writes aggregate hashes but trusts them on load | Verify legacy aggregate identities; add per-file digests for actionable new-package diagnostics |
| `structure.tracks` omits note counts | Reconcile counts against `score.json`; do not manufacture planner fields from MIDI |
| `audio/config.py` resolves checkout-relative YAML; effective config is reread from disk | Package resources and retain normalized configuration in memory |
| `render/stems.py` bypasses registry; `routing` and `role_map`, `comp` and `comping` disagree | Normalize once, route every stem through registry |
| Groove reconstructs PrettyMIDI notes, losing tempo maps, controllers, RPN, bends and SysEx | Use a lossless event representation for transformations |
| `simple_r.py` uses integer pitches only | Keep it preview-only for supported direct-12 material; do not certify microtonal output with it |
| Reserved names and path separators are not handled by stem naming | Identity-based safe names and a reserved artifact namespace |
| Render cache omits effective routes, tuning and SoundFont bytes; bridge catches broad native exceptions | Resolve the renderer before hashing, record effective behavior, narrow fallback |
| `build_tuning` can label fractional-base 12-EDO direct-12; drum overrides can disagree with metadata | Audit actual events and fail disagreements, rather than trusting labels |

Existing tests cover artifacts, simple cache hits, ordinary duplicate names and basic groove properties, not real instruments or channel-state fidelity. The supplied document's “22 tests pass” is historical, not a result from this planning session. No sibling checkout or installed copy of the pinned `midi2audio_generative@46b3dfe` was available; its adapter cannot yet be claimed reviewed or suitable for direct copying.

### Linear reconciliation (read on 2026-11-01)

Workspace: MusicAI (`https://linear.app/singularitynet`); assignee resolved to Arseniy Losev, ID `5fcdd45f-8c8d-465f-ba2a-d1c731b33c86`. All ten assigned issues were read in full, with relations and comments; the list had no further page.

| Ticket | Live status | Treatment / acceptance boundary |
| --- | --- | --- |
| [MUS-29](https://linear.app/singularitynet/issue/MUS-29) | Backlog | Parent epic, not closed by this demo |
| [MUS-30](https://linear.app/singularitynet/issue/MUS-30) | Done | Contract/quarantine foundation; integrity hardening is follow-on work |
| [MUS-31](https://linear.app/singularitynet/issue/MUS-31) | Done | Keep lazy entry and validation-first behavior |
| [MUS-32](https://linear.app/singularitynet/issue/MUS-32) | Done | Existing simple/groove/bridge foundation, not a new implementation task |
| [MUS-33](https://linear.app/singularitynet/issue/MUS-33) | Done | PR #47 attached; comment confirms opening; no new upstream-PR task |
| [MUS-34](https://linear.app/singularitynet/issue/MUS-34) | Backlog | Complete selectable optional FluidSynth, reconcile tolerances and asset-free default CI |
| [MUS-35](https://linear.app/singularitynet/issue/MUS-35) | Backlog | Deliver render tuning, controlled 19-EDO checks and rejection floors only |
| [MUS-36](https://linear.app/singularitynet/issue/MUS-36) | Backlog | Defer full rhythm/notes/CLAP/FAD/combine and corruption suite |
| [MUS-37](https://linear.app/singularitynet/issue/MUS-37) | Backlog | Defer restyle; explicitly blocked until MUS-36 is merged |
| [MUS-38](https://linear.app/singularitynet/issue/MUS-38) | Backlog | Defer mastering, references, breadth and seams |

MUS-32/33 descriptions still say upstream not opened; MUS-33 status, attachment and comment supersede that historical text for planning. GitHub merge state was not independently checked. MUS-34 → MUS-35 → MUS-36 → MUS-37 → MUS-38 is also the live blocking chain. MUS-35's Scala/MTS-helper breadth and pre-endpoint diffusion gate remain follow-up work; partial completion does not unblock MUS-36 in Linear. No ticket mutations are part of this plan execution.

## Goals / Non-Goals

**Goals:**
- Keep RenderPackage as the sole symbolic/audio file boundary; all DSP, inferred-input models and renderer lifecycle stay in `aimusic.audio`.
- Establish event fidelity before adding a native renderer, then prove instrument and tuning behavior separately.
- Make failures and evidence visible in local JSON reports without adding external telemetry.

**Non-Goals:**
- No general polyphonic f0 transcription, arbitrary imported-MIDI tuning inference, real-time synth playback or automatic channel reallocation.
- No correction of unrelated planner/ML bugs. Inconsistent package drum overrides or unsupported base-tuning metadata fail clearly; broad symbolic CLI changes are not prerequisites.
- No SoundFont download/bundling, global tool installation, paid calls, full LUFS mastering or production-quality claims.

## Decisions

### 1. Verified audio input context and reconciliation before groove

Extend the audio-owned frozen input context with parsed tuning, score-derived note counts, planner structure, verified hashes, stable source-track identities and provenance. The package loader passes it into orchestration; imported MIDI uses the same downstream shape with `origin=inferred` and no claimed planner data. This replaces re-inference as the primary source without introducing an audio import into core.

Verify the existing hash algorithm without changing valid v1 identity: source hash is canonical Score bytes + source MIDI bytes + EDO bytes; package hash is source digest text + canonical structure + canonical tuning + beatstates bytes. Verify manifest aliases and structure source hash agree. Absent legacy optional beatstates is treated as empty only if the resulting aggregate matches. New manifests add per-file SHA-256 digests for all contract payloads (not a self-hash of the mutable manifest). Legacy aggregate failures identify the involved file group and expected/actual aggregate; exact single-file localization is only possible when per-file digests exist. Hashes detect corruption, not malicious rewriting of both payload and manifest.

Reconciliation uses Score counts and source-track/channel association, not PrettyMIDI instrument count (MPE can split a logical track). Compare exact programs, meter, discrete counts and identities; normalize `comp` to `comping`. Role agreement means association to the declared source track, not a mandatory agreement with a heuristic. Report unavailable heuristic fields explicitly. Compare tempo maps at serialized change points with 0.001 BPM tolerance, and time/bar boundaries within one source tick at the local tempo. Score occupancy duration and planner section/bar extent are distinct: trailing planned silence is allowed, but note endpoints and MIDI end-of-track are compared to their corresponding expected bounds. Reconcile original MIDI before groove; audit groove separately so intentional offsets do not trigger planner mismatch errors.

Alternative rejected: expand structure schema with duplicated note counts or silently “fix” metadata from MIDI. Both create more sources of truth.

### 2. One effective configuration and explicit failure policy

Move shipped defaults/profiles under `aimusic/audio/resources/` and declare package data. Read via `importlib.resources`, keeping normalized raw/effective values in the frozen configuration instead of reopening `source_path`. Keep explicit YAML-path profiles compatible; add named `preview` and `listening` profiles.

Canonical routing is `render.routing.<role>.{default,microtonal}`. Accept legacy scalar `role_map` by expanding it; equivalent aliases merge, contradictions fail. Normalize `comp` to `comping` everywhere; fix the overlapping pad/lead program heuristic in imported analysis. Resolve SoundFont precedence as CLI `--soundfont` > selected profile/config `paths.soundfont` > `AIMUSIC_SOUNDFONT`. A specified but invalid higher-precedence path fails instead of silently choosing another asset. Resolve config-relative paths against their config origin.

`preview` defaults to simple with explicit compatible fallback permitted; `listening` selects native FluidSynth and forbids fallback. Both keep restyle/MIDI-GPT off. The application backend (`native` versus explicitly selected `m2a`) is separate from the instrument renderer (`simple` versus `fluidsynth`). Remove catch-all native-to-bridge failover: missing optional render dependencies are classified and handled by renderer policy; execution defects propagate. Preserve explicit bridge selection and the pin, but do not route this verified native demo through an unverified bridge path.

Alternative rejected: a second incompatible schema or globally defaulting to FluidSynth, which would break lightweight installations.

### 3. Event-preserving transform and stem isolation

Introduce an audio-local mido-based absolute-tick event stream carrying original track index, channel, event order and note association. PrettyMIDI can remain an analysis convenience but not the serialization authority. Preserve conductor/global events and all non-note payloads. Merge at stable tick order; preserve original equal-tick ordering except for explicit retimed note bundles, where prior note-off and necessary setup precede note-on.

Use tempo-map-aware seconds/ticks conversion and round-half-even offsets. Move note endpoints and associated per-note expression together; persistent controllers keep their source timeline, with initial state replayed before an earlier shifted note when needed. MPE pitch trajectories and setup must remain attached to their note lifetime. Validate note-pairing and channel-state equivalence after transformation. If shifted notes overlap incompatible bend states (including held sustain), reject with a specific conflict; do not guess a new MPE allocation. Unsupported ambiguous imports get the same diagnostic and can be retried with groove disabled if the original is valid.

Isolate logical source tracks while carrying their complete channel state and global events; consolidate MPE member channels back into the logical source stem. Distinct tracks sharing a role retain separate stems and a role index. Stable filenames include source ordinal plus a sanitized role/name; reserve generated names and check resolved containment including existing symlinks. Do not trust MIDI names as paths. Reject conflicting cross-track channel ownership rather than copying a channel's notes into two stems.

Alternative rejected: copying only PrettyMIDI pitch bends/CCs. It still cannot guarantee SysEx, exact ordering, channel identity or tempo-map preservation.

### 4. Registry-driven rendering and FluidSynth offline adapter

Define an audio-only renderer request containing isolated event MIDI, effective tuning, sample rate, common duration/tail, output path and resolved asset; results include actual backend/version, capability record and output hashes. Every stem uses registry selection before cache lookup.

The exact pinned adapter at `46b3dfe` was reviewed during apply and contains no license file, package license metadata, or permission notice. Per the user's 2026-11-01 decision, do not copy or port that source. Implement the adapter clean-room from public FluidSynth/pyFluidSynth APIs and this repository's event contracts, recording the external commit only as historical provenance. Use offline native sample generation through optional Python bindings: dispatch events by absolute sample positions derived from the tempo map, honor programs/banks, CC/RPN/pitch wheel, note on/off and percussion channel 10 (zero-based 9). Do not initialize a real-time audio device. Capture native/binding versions and deterministic synth settings; disable uncontrolled effects for reference-tone tests. Unsupported SysEx/tuning rendering fails before success, although event transforms preserve its payload.

Compute one render horizon from score/planned end, MIDI events and a configured fixed release tail; render or pad all stems to the same frame count. Sum stems without the generated click and apply one deterministic gain when needed to respect peak 0.95 by default. This is rough mixing, not LUFS mastering. Tests use active-window RMS > 1e-5 for designated nonempty fixtures; silent tracks are not artificially made audible.

Alternative rejected: invoke an external synthesizer with opaque defaults or claim the current simple sine renderer is a recognizable instrument backend.

### 5. Cache identity follows actual execution

Fingerprint verified source/package/structure/tuning, expressive MIDI, normalized config and routes, requested/effective backend, fallback policy/reason, renderer capability/code/native versions, SoundFont bytes and relevant output settings. Resolve availability and tuning compatibility before computing the key. Cache identity excludes output path and display-only source paths when bytes and effective settings are identical.

Stage outputs in a temporary run directory, verify expected files and their hashes, then publish a complete cache manifest last. Warm reuse verifies all outputs/sidecars; partial or corrupt entries miss. Source packages remain immutable; audio results and reports live under the selected audio output directory. Preserve `stems/rough_mix.wav`, adding run-level reconciliation, tuning, routes, effective configuration and hash reports. Do not promise cross-platform byte equality; cold-repeat and warm-hit evidence is for pinned versions/assets.

Alternative rejected: SoundFont path-only keys or caching requested FluidSynth results that actually came from simple fallback.

### 6. Tuning capability gate and bounded f0 evidence

Add `audio/render/tuning.py` for pitch-intent reconstruction/capability policy and `audio/scoring/tuning_check.py` for controlled dry tones only. Infer required capability from both metadata and actual channel events; fractional base tuning and bends override misleading direct-12 labels. Simple remains incapable of tuning beyond its proven path. FluidSynth MPE support means the per-channel pitch-wheel/RPN path proven by fixtures, not a claim that every MPE feature is supported. MTS rendering stays explicitly unsupported in this milestone.

For generated fixtures, expected frequencies come from Score pitch heights plus EDO/base tuning, independently of rendered MIDI. Imported reference fixtures supply a known expected-frequency sidecar; general imports without an intended lattice get event-based provenance, not a fabricated lattice certificate. Compare event-reconstructed pitch within 1 cent after every transformation. Estimate f0 on known monophonic steady-state windows using deterministic NumPy signal analysis; record window, expected/measured Hz, confidence/coverage, mean and P95 absolute cents. Every designated reference window must be measurable. Controlled-tone acceptance is <=5 cents per tone; general default hard rejection uses mean <=15 and P95 <=15 cents, with stricter configured floors supported. Select off-12-TET tones so the flattened corruption actually breaches the floor. Exclude percussion and label polyphonic f0 unmeasured.

Alternative rejected: heavy transcription dependencies, nearest-lattice-only scoring that can accept wrong notes, or pretending scalar metadata verifies audible pitch.

### 7. CLI and evidence-first demo delivery

Extend the existing lazy `render-audio` command with output/SoundFont options and named profiles; add lazy `render-midi` with the same validate-only convention. Proposed full-render invocation: `python -m aimusic.app.cli render-midi <source.mid> --profile listening --soundfont <asset.sf2> --out <directory> --no-validate-only`. Confirm argument placement against existing CLI tests during implementation; preserve existing flags and defaults.

Produce the four listening examples from versioned source fixtures/generators, recording seeds, commands, platform, versions, asset license/hash and reports. Keep WAVs/SoundFonts local and untracked. Default tests use mocks and simple preview; explicitly enabled native tests fail if their promised prerequisites are missing. Human gates review 12-EDO instruments first, then 19-EDO tones/groove and the final set. Metrics are fidelity evidence, not a claim of mastered quality.

## Risks / Trade-offs

- Native library/binding behavior varies → record/pin versions and test offline event dispatch on a small controlled fixture before broader instrument work.
- The pinned adapter has no discoverable license → retain its hash/provenance but use a clean-room implementation based only on public APIs; do not copy unlicensed source.
- Channel retiming is subtle → event-state tests precede rendering; unsupported conflicts fail closed. No automatic channel reallocation in this milestone.
- SoundFont quality and licensing determine instrument recognizability → user supplies a license-approved asset, hash-pinned in the real lane; human listening remains a gate.
- Old package hashes are aggregate, not authenticated per-file hashes → retain aggregate verification and clearly scoped diagnostics; new additive file hashes improve localization without changing legacy identity.
- User-chosen exotic imports may exceed supported semantics → explicitly support type-0/1 with consistent channel ownership; reject type-2, malformed or ambiguous arrangements.
- Verification could become “green by skipping” → separate opt-in real tests from default CI and fail missing prerequisites when that lane is enabled; never count a skipped native test as demo evidence.
- The roadmap's ~60-second fixture budget is environment-sensitive → record elapsed time and hardware; investigate overruns rather than lowering fidelity floors or calling the target proven.

## Migration Plan

Implementation begins only after a new explicit apply request. Continue the dedicated branch, preferably in a separate worktree, preserving current unrelated edits. Each commit in `tasks.md` includes tests; commits are proposed messages, not commits made by this workflow.

1. Baseline and contract/config compatibility, then event preservation and safe stems.
2. Registry/cache integration, then optional FluidSynth with a 12-EDO listening checkpoint (MUS-34 boundary).
3. Supported MPE tuning checks (partial MUS-35 boundary), then imported CLI and the complete listening set.
4. Retain existing valid v1 packages and YAML aliases; document stricter errors and explicit bridge selection. Old incomplete cache entries become misses, not migrated successful results.
5. Roll back a failed implementation slice by reverting that slice on the feature branch. For supported direct-12 inputs use preview while diagnosing native failures; do not roll back by silently flattening microtonal input or re-enabling broad exception fallback.

Only after evidence review should MUS-34 completion be proposed; MUS-35 remains partial. Fork PRs and any org PR are separate human-authorized actions.

## Open Questions

- Which local license-approved SoundFont and native FluidSynth version will be pinned for the real-render lane? This affects asset setup/evidence, not the interface or acceptance floors. Record the choice before running native acceptance.
- Where should local listening WAVs be shared for human review without committing them? Default is the user's selected output directory; no upload is authorized by this plan.
