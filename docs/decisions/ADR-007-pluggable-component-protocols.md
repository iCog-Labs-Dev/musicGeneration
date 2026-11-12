# ADR-007: Pluggable component protocols (REQ-10)

## Status

Implemented for PR #51 review. The mentor revision supersedes the original two-renderer-protocol and Python-only selection decisions.

## Date

2026-10-31

## Context

The original issue requires replaceable tonal metrics, candidate generation, whole-score decoding and rendering, with independent typed contracts, default compatibility and proof of substitution. The original implementation retained the existing callable shapes, including a None-returning renderer. A presentation follow-up added a second reporting protocol. The mentor's review then required one neutral renderer protocol, application-owned construction, configuration-based selection and recorded implementations, together with stronger data and test validation.

The current upstream also includes section-wise planning and a section-wise UI default. Both planning strategies must receive the selected components. Existing direct MIDI/decoder APIs and default musical algorithms remain intact.

## Decision

- Define `TonalMetric`, `ScoreDecoder` and a single `ScoreRenderer` in `core.protocols`, with the existing candidate callable in `planning.protocols`.
- The renderer accepts `Score` and `RenderRequest(output_path, format)` and returns `RenderResult(output_path, format, size_bytes)`. Neither request nor result exposes MIDI/EDO settings. Remove the earlier reporting protocol and renderer union.
- Bind EDO and instrument configuration in the concrete `MidiRenderer`, using `MidiRenderSettings`. It delegates to the unchanged `render_midi` function, whose existing direct signature and None return remain available.
- Make `RenderResultAdapter` generic: render to a fresh sibling temporary path, validate a regular new file and its exact report, then atomically replace the destination. Reject missing, inconsistent and non-regular outputs. Failure preserves previous final bytes.
- Resolve defaults, configured constructors and supplied runtime objects in `app.components.create_components`. CLI and UI share this factory. Local JSON names trusted installed `module:export` constructors and JSON keyword options. Candidate constructors receive the selected metric; renderer constructors receive effective MIDI settings as construction inputs, outside the neutral protocol.
- Construct fresh objects per run, copy nested options before constructor calls, snapshot provenance independently, and check required positional/keyword call shapes without executing placeholder calls. These checks complement static fixtures; they cannot prove semantics or return annotations.
- Record effective role sources, factories, implementation identities, explicitly public options and known renderer settings under manifest `config.components`. Constructor options are private by default; `public_options` selects shareable values. Record only an omitted-option count for the rest. Explicit runtime objects have honest unknown-options markers. Reject simultaneous file and object selection for one role.
- Reject a built-in MIDI renderer whose bound settings differ from the run, including through our artifact wrappers. This prevents actual tuning/programs from disagreeing with package metadata. Opaque custom implementations must honor their supplied backend settings.
- Move executable demo composition to `app.demos`; retain the old planning entry as a lazy delegator. Planning does not import or select concrete decoder/rendering defaults. Carry metric/candidate choices through every section as well as single-pass planning.
- Validate all shared data classes and recheck candidate objects at the graph boundary. Require coherent unique outcomes/counters, finite aligned scores, correct RNG/source/time, legal vocabulary IDs and transitions. Give endpoint checks accurate internal counters without inflating raw-generator diagnostics. Validate graph `d_max` agreement with SB configuration, leaving proposal budgets separate.
- Negative typing fixtures mark each invalid assignment. Match errors to its exact file/line, rather than accepting an error total alone.

## Alternatives and consequences

A named constructor registry is easier to enumerate for a browser catalog but requires host registration before the stock CLI can select an installed alternative. Trusted local factory references meet configuration-only selection without coordinator edits. We do not add package discovery, a browser-supplied Python loader or a mutable global registry. The operator controls startup configuration; selected factories execute in process and are not sandboxed.

Keeping two new renderer protocols would preserve a PR-internal intermediate API but contradict the requested single boundary. Only the established direct `render_midi` function is retained for compatibility. New request/result wrappers are limited to the renderer; candidate and decoder callable types remain established project types.

A simple filesystem-timestamp check cannot reliably distinguish fresh identical writes on every platform. Fresh staging avoids stale-file success and preserves valid overwrites. The adapter validates artifact facts, not file-format correctness or music quality; product workflows still require valid MIDI.

Existing manifests gain a component-provenance config member. Score and MIDI output remain subject to fixed-seed comparison against the current unmodified upstream, whose unrelated changes may differ from the earlier October snapshot. Custom runtime-object construction cannot be reconstructed from identity alone.

See the [extension guide](../extending-components.md) for constructors and contracts and the [implementation record](../req-10-implementation.md) for exact verification and review-comment coverage.
