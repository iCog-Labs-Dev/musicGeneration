# Proposal

## Why

The existing MIDI-to-audio spine writes CI-safe sine-wave WAVs, but does not yet establish recognizable instrument rendering or preservation of routing and microtonal intent. Deliver the stopping point in `docs/midi2audio-audible-demo-plan.md`: reproducible playable stems and a rough mix, completing MUS-34 and only the dry-render portion of MUS-35.

## What Changes

- Carry verified RenderPackage metadata through native rendering and reconcile it against source MIDI before groove processing.
- Ship wheel-safe audio defaults, normalize legacy routing aliases, and make fallback and cache provenance explicit.
- Add optional FluidSynth instrument rendering, collision-safe stems, GM percussion/program preservation, and a listening profile that never silently falls back to the simple renderer.
- Preserve MIDI timing, channel state, controllers and tuning through groove and stem extraction; support a verified 19-EDO MPE path and reject unsupported tuning paths.
- Add controlled-tone tuning checks with hard rejection, plus generated-package and imported-MIDI example workflows.
- Add asset-free tests, an explicitly enabled real-render lane, reproducible fixture sources, and human listening checkpoints. Commit checkpoints are defined in `tasks.md`; this change currently contains planning only.
- **BREAKING (intentional validation tightening):** invalid/tampered packages, unsupported microtonal rendering, conflicting routing aliases, and native runtime failures previously hidden by bridge fallback become explicit errors. Existing valid simple-preview and validate-only workflows remain supported.

## Capabilities

### New Capabilities

No main OpenSpec capabilities currently exist. These introduce requirements for the existing pipeline and its proposed extensions:

- `audio-input-integrity`: package verification, planner/MIDI reconciliation, and safe imported-MIDI admission.
- `audio-instrument-rendering`: configuration, renderer selection, event-preserving stems/mix, cache identity and provenance.
- `audio-tuning-integrity`: tuning capability negotiation, lossless event handling, and controlled dry-render tuning validation.
- `audio-demo-workflow`: lazy CLI entry paths, reproducible listening sets and separated verification lanes.

### Modified Capabilities

None.

## Impact

- Targets `aimusic/audio/`, the lazy audio commands in `aimusic/app/cli.py`, optional packaging/configuration, relevant tests and audio documentation. Core contract changes, if necessary for hash verification, remain dependency-light in `aimusic/core/render_package.py`; no DSP crosses the file-contract boundary.
- Adds optional `[audio-render]`; native FluidSynth and a separately supplied, appropriately licensed SoundFont are required only for real instrument rendering. No SoundFonts, generated WAVs, secrets, weights, submodules or new core DSP dependencies are committed.
- Linear snapshot (2026-11-01, MusicAI workspace, assignee Arseniy Losev): [MUS-29](https://linear.app/singularitynet/issue/MUS-29) remains the Backlog epic; MUS-30–33 are Done; [MUS-34](https://linear.app/singularitynet/issue/MUS-34) and [MUS-35](https://linear.app/singularitynet/issue/MUS-35) are Backlog implementation targets. MUS-35 stays incomplete until its remaining Scala/helpers and diffusion-gating scope is separately delivered.
- Preserve the dependency chain MUS-34 → MUS-35 → MUS-36 → MUS-37 → MUS-38. MUS-36 full scorers/CLAP/FAD, MUS-37 restyle, MUS-38 mastering/references/breadth, UI expansion, B4 and unrelated symbolic work are out of scope. This milestone does not unblock later tickets by declaring partial MUS-35 complete.
- No tickets, commits, branches, PRs or implementation files are changed by this proposal workflow. Fork-first delivery and separately authorized upstream PRs remain mandatory.
