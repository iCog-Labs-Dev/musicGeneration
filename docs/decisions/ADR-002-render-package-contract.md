# ADR-002: RenderPackage file contract as the only inter-system boundary

## Status
Accepted

## Date
2026-08-20

## Context
Symbolic and audio halves must evolve independently with provenance
(content hashes, manifests) and without circular imports.

## Decision
Inter-system boundary is a versioned directory `run_<hash>/` (RenderPackage):
MIDI, score JSON, structure.json, tuning.json, manifest, optional beatstates.
Always emit from product `generate`. Stages communicate via files + sidecars.

## Alternatives considered
- In-memory Score/BeatState into audio APIs — rejected
- Opt-in `--emit-structure` only — rejected (always emit)

## Consequences
Contract tests; fixture corpora; B1 can fill structure from planner fields.
