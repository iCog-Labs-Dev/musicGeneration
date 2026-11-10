# Spec Delta

## Purpose

Provide reproducible instrument stems and a playable rough mix with explicit renderer selection, safe artifacts, and auditable cache provenance.

## ADDED Requirements

### Requirement: Portable effective configuration
Audio defaults and named preview/listening profiles SHALL load from installed distributions outside a source checkout. Legacy role alias `comp` SHALL normalize to `comping`; legacy routing forms SHALL normalize to one effective route map, with conflicting definitions rejected.

#### Scenario: Wheel installation
- **WHEN** audio rendering is invoked outside the repository from a wheel installation
- **THEN** defaults and both profiles load, and effective configuration is available without rereading a checkout-relative YAML file

#### Scenario: Conflicting aliases
- **WHEN** `comp` and `comping`, or legacy and canonical route definitions, prescribe different renderers for the same role and tuning class
- **THEN** configuration fails with a diagnostic identifying the conflict

### Requirement: Explicit renderer selection and fallback
Stem rendering SHALL honor effective per-role routes. Simple preview SHALL remain the lightweight default. Missing optional render resources SHALL permit simple fallback only when explicitly allowed and tuning-compatible; listening mode SHALL fail instead. Native runtime defects SHALL NOT silently switch to the external bridge.

#### Scenario: Missing SoundFont in listening mode
- **WHEN** FluidSynth listening is requested without a usable SoundFont, bindings or native library
- **THEN** rendering fails with setup guidance and publishes no successful listening result

#### Scenario: Permitted preview fallback
- **WHEN** an optional renderer is unavailable and simple fallback is enabled for supported direct-12-TET material
- **THEN** preview renders with requested backend, effective backend and fallback reason recorded separately

### Requirement: Preserve instrument and timing semantics
Instrument rendering and stem isolation SHALL preserve source tempo/meter maps, note timing, programs, percussion-channel behavior, sustain and supported controller state. Stems SHALL follow authoritative package roles or explicitly inferred imported roles, not lose notes through channel splitting.

#### Scenario: Multitrack instrument fixture
- **WHEN** the eight-bar 12-EDO fixture is rendered with FluidSynth
- **THEN** drums, bass, comping and lead retain their programs, timing and note counts, including sustain releases and tempo changes in event-preservation fixtures

### Requirement: Collision-safe aligned output
Stem artifacts SHALL have stable unique safe names, distinct from click, mix and boundary artifacts. Stems and mix SHALL share the configured sample rate and sample count, include the configured release tail, contain finite samples and respect configured peak headroom. Nonempty audible fixtures SHALL produce non-silent output.

#### Scenario: Duplicate and reserved names
- **WHEN** inputs contain duplicate, `click`, `rough_mix`, `bar_boundaries`, separator-containing or traversal-like names
- **THEN** every source stem remains separately addressable within the output root, no artifact overwrites another, and all source stems participate in the mix

#### Scenario: Listening signal checks
- **WHEN** the approved nonempty listening fixture is rendered
- **THEN** stems and mix have equal frame counts, RMS greater than 0.00001 on active regions, finite samples and absolute peaks no greater than the configured ceiling, whose default is 0.95

### Requirement: Content-based cache and provenance
Render cache identity SHALL cover source bytes, package/structure/tuning identity, effective configuration and routes, requested/effective renderer and version, fallback policy, and SoundFont content hash. Reuse SHALL verify cached output integrity and restore matching sidecars. Determinism SHALL be claimed only within a pinned runtime and asset environment.

#### Scenario: SoundFont replacement at the same path
- **WHEN** SoundFont bytes change without its path changing
- **THEN** rendering misses the previous cache and records the new content digest

#### Scenario: Semantic input change
- **WHEN** validated structure, tuning, routing, renderer version or fallback policy changes
- **THEN** the changed effective request cannot reuse the previous render result

#### Scenario: Verified warm reuse
- **WHEN** all effective inputs match a complete intact cache entry
- **THEN** restored WAVs are byte-identical to the stored result and provenance identifies the effective backend and asset hashes

#### Scenario: Incomplete cache
- **WHEN** a cached stem is missing or its bytes disagree with its recorded digest
- **THEN** the entry is not reported as a successful cache hit
