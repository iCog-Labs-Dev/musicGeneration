# ADR-007: Pluggable component protocols with existing types (REQ-10)

## Status

Accepted

## Date

2026-10-31

## Context

Priors already had a substantial typed replacement boundary. Tonal metrics,
candidate generation, whole-score decoding, and rendering still depended on
directly selected implementations. Experiments needed changes to their callers.
The issue requires substitution, small contracts, default compatibility,
checked conformance, test doubles, and extension documentation. The mentor
also requires the new types to match the existing implementations.

## Decision

- Define `TonalMetric`, `ScoreDecoder`, and `ScoreRenderer` in
  `aimusic.core.protocols`. Define `CandidateGenerator` in
  `aimusic.planning.protocols`, with type-only imports for prior/context types.
- Preserve the existing candidate, decoder, and renderer callable shapes.
  Decoder output remains `(Score, RNGKey)`. Rendering accepts both `Score`
  and symbolic-note sequences, takes the project `EDO`, and returns `None`.
  Add a separate `ReportingScoreRenderer` with identical inputs and a
  `RenderResult` return. Both are accepted at workflow entry points.
- Move candidate result/rejection and symbolic note/instrument dataclasses
  to `aimusic.core.component_types`. Old module imports alias the same classes.
- Supply runtime components through optional workflow arguments, separately
  from serializable configuration. Use existing implementations when omitted.
  Bind default candidate proposals to the selected metric with
  `MetricBoundCandidates`; expose existing tonal functions with
  `ExistingTonalMetric`. The decoder function conforms directly.
  `RenderResultAdapter` wraps legacy or reporting renderers and returns verified
  artifact details without changing `render_midi` or `ScoreRenderer`.
- Route the selected metric through proposals, pruning, structural scoring,
  retained-edge diagnostics, cadence, and tension reports. Preserve default
  caches and bypass them for custom implementations.
- Validate supplied candidate results where they enter the graph. Keep final
  endpoint restrictions, legality, edge scoring, and retained-edge limits in
  the graph. Empty results stay empty; component failures propagate.

## Alternatives considered

New request wrappers would shorten signatures but require conversions
and change the existing input shapes. The original design omitted a renderer
result. The mentor required all presented features, including slide 7, so that
decision is superseded: a separate reporting adapter now returns the saved
path, format, actual size and tuning. Legacy signatures stay unchanged.
A configuration adapter is unnecessary because `EDO` is a project type, not
a MIDI-library class. Global registries or a process-wide active component would hide
selection state. These options were rejected for this issue.

## Consequences

Components can be substituted without editing orchestration. Static positive
and negative fixtures check signatures; behavior and integration tests check
returned results, errors, defaults, and actual artifacts. Existing per-beat
`TrackGenerator` behavior and prior protocols remain separate.

The reporting adapter renders into a fresh temporary path beside the final
destination. It checks that the new file exists and that supplied metadata
matches the actual file size, temporary destination, MIDI format and tuning.
It then atomically replaces the final destination and reports the final path.
Failures discard staged output and preserve an existing final file. A simple
timestamp comparison was rejected because valid overwrites can retain the
same observed filesystem timestamps. The adapter does not
parse MIDI or guarantee arbitrary plugin correctness. CLI/UI output still
requires valid MIDI at the requested destination.
Python components are trusted in-process code, not sandboxed plugins.
Existing manifests do not serialize component objects or their provenance;
custom experiments must record their implementation version and settings.
No CLI registry or UI plugin selector is added.

See the [extension guide](../extending-components.md) for contracts and a
checked example, and the [implementation record](../req-10-implementation.md)
for file changes, compatibility evidence, and verification limits.
