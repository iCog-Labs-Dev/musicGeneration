# Spec Delta

## Purpose

Ensure audio rendering uses trustworthy source files and retains the distinction between authoritative planner metadata and inferred imported-MIDI metadata.

## ADDED Requirements

### Requirement: Verify package content before use
The audio entry path SHALL validate required files and recompute declared source and package hashes before rendering or cache reuse. It SHALL reject mismatches without changing source files, naming the file when per-file digests exist or the involved file group for legacy aggregate-only hashes.

#### Scenario: Source tampering despite a warm cache
- **WHEN** a package with per-file digests has its MIDI or hashed metadata changed without the declared digest being updated
- **THEN** validation fails naming the file and expected versus actual digest, even if cached audio exists

#### Scenario: Legacy aggregate-only package
- **WHEN** a legacy package has no per-file digests and recomputed source or package identity differs from its declared aggregate
- **THEN** validation rejects it with the expected/actual aggregate and involved file group, without falsely identifying a single changed file

### Requirement: Retain authoritative planner context
Package rendering SHALL retain planner structure, tuning, source identity and track-role associations through the audio pipeline. MIDI inference SHALL cross-check rather than silently replace package metadata.

#### Scenario: Planner roles differ from name heuristics
- **WHEN** a valid package identifies a track as comping but its MIDI name does not contain a recognizable role
- **THEN** the audio route uses the package role and records its planner origin

### Requirement: Reconcile source MIDI before expression
Package validation SHALL compare track identity, role association, programs, note counts, tempo, meter, duration and bar boundaries with source MIDI before groove processing. Reports SHALL identify each field, expected and observed values, tolerance and outcome. Non-inferable roles SHALL be identified rather than invented.

#### Scenario: Within-tolerance serialization
- **WHEN** source MIDI matches planner values within one source tick for serialized boundaries and 0.001 BPM for tempo, with exact discrete counts, programs and meter
- **THEN** reconciliation succeeds and records the applied tolerances

#### Scenario: Mismatch is not hidden by inference
- **WHEN** a track program or note count differs, or a time boundary exceeds tolerance
- **THEN** validation fails with the affected track and field, without rendering or silently rewriting planner intent

### Requirement: Admit supported imported MIDI safely
Imported MIDI SHALL be treated as inferred input, not as a generated RenderPackage. Malformed files, ambiguous unsupported event/channel arrangements, and unsupported MIDI types SHALL fail with actionable diagnostics. Input files SHALL remain unchanged and artifact names SHALL stay inside the chosen output directory.

#### Scenario: Supported import
- **WHEN** a supported type-0 or type-1 MIDI file is supplied without planner metadata
- **THEN** the system records inferred tempo, meter, tracks, programs and tuning evidence, and labels planner reconciliation not applicable

#### Scenario: Unsafe name or unsupported input
- **WHEN** track names contain traversal syntax or duplicate names, or input is malformed or type-2 MIDI
- **THEN** names cannot escape the output directory or overwrite one another, and malformed or unsupported input is rejected before successful artifact publication
