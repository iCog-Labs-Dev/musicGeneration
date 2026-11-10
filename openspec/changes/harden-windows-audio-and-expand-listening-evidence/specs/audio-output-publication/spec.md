# Spec Delta

## Purpose

Provide dependable cross-platform publication of complete audio artifacts without permitting path escapes, silent overwrites, partial-success claims, or loss of recoverable prior outputs.

## ADDED Requirements

### Requirement: Platform-appropriate publication
The audio pipeline SHALL publish supported artifacts on native Windows and POSIX systems using operations whose containment and collision semantics are verified on the executing platform.

#### Scenario: Native Windows audio run
- **WHEN** a supported audio workflow runs in a trusted local NTFS output location on native Windows
- **THEN** groove artifacts, demo sources, stems, cache restorations, and final mix artifacts are published without invoking unsupported POSIX descriptor APIs

#### Scenario: Unsupported publication environment
- **WHEN** the executing filesystem or runtime cannot provide the required publication guarantees
- **THEN** publication fails with an actionable diagnostic before reporting success and does not silently fall back to overwrite-capable copying

### Requirement: Complete-file exclusive publication
Artifacts whose contract forbids replacement SHALL become visible only after their complete bytes are staged and SHALL refuse every existing destination entry, including files, directories, links, reparse points, and case-insensitive aliases.

#### Scenario: Clean exclusive destination
- **WHEN** all destination names are absent and every staged artifact is complete
- **THEN** each destination receives exactly the staged bytes and the published identity is recorded for rollback verification

#### Scenario: Concurrent destination collision
- **WHEN** another writer creates a destination before this invocation publishes it
- **THEN** this invocation preserves the other writer's entry, rolls back only artifacts it owns, and reports failure

### Requirement: Transactional groove bundle replacement
Groove publication SHALL support replacement of its related MIDI and sidecar bundle while retaining the previous leaf entries until the new complete bundle can be accepted or rolled back.

#### Scenario: Successful bundle replacement
- **WHEN** valid previous groove outputs exist and every new payload has been staged and flushed
- **THEN** all public bundle names refer to the new complete payloads and no stale backup remains

#### Scenario: Publication failure after partial progress
- **WHEN** one bundle member fails after earlier members were published
- **THEN** the previous bundle is restored where it existed, newly introduced members are removed, and the source MIDI remains unchanged

#### Scenario: Rollback failure
- **WHEN** the filesystem refuses restoration during rollback
- **THEN** the operation fails, retains the only recovery copies, and reports their recoverable locations without claiming a valid bundle

### Requirement: Pinned containment and reparse safety
Publication SHALL keep the selected output identity under verification for the operation, reject disallowed links or reparse points, validate portable leaf names, and prevent output from escaping through path-component replacement.

#### Scenario: Output component changes identity
- **WHEN** an output root or ancestor is replaced, renamed, or converted to a disallowed reparse point during publication
- **THEN** the operation fails closed and does not publish into the substituted target

#### Scenario: Unsafe artifact name
- **WHEN** an artifact name contains traversal, separators, reserved device names, forbidden Windows characters, trailing spaces or dots, or control characters
- **THEN** the name is rejected before any public artifact is written

### Requirement: Owned rollback and cleanup
Failed publication SHALL remove only entries proven to belong to that invocation and SHALL preserve unrelated pre-existing files and entries introduced by other writers.

#### Scenario: Failure after one owned artifact
- **WHEN** publication fails after creating one verified artifact and before completing the set
- **THEN** the verified owned artifact and private temporary entries are removed while unrelated entries remain byte-identical

#### Scenario: Process-crash residue
- **WHEN** a prior process leaves private staging or recovery entries
- **THEN** a later invocation refuses ambiguous reuse and reports the residue for explicit inspection or recovery

### Requirement: Native Windows verification
Default verification SHALL exercise publication on a native Windows runner independently of simulated Windows APIs, while privilege-dependent link tests SHALL retain meaningful fallback coverage rather than turning the lane green through broad skips.

#### Scenario: Asset-free Windows CI
- **WHEN** the default Windows CI job runs without FluidSynth, a SoundFont, administrator rights, or Developer Mode
- **THEN** native ordinary publication, collisions, rollback, path handling, wheel execution, and mocked reparse safety are tested and required to pass

#### Scenario: Optional native renderer lane
- **WHEN** the explicitly enabled Windows native-render lane has its pinned library and approved asset prerequisites
- **THEN** it runs the canonical workflows and records runtime, architecture, DLL, asset, output, and cache evidence without comparing Windows PCM hashes to another operating system
