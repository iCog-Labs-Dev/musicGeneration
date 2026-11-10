# Spec Delta

## Purpose

Prevent silent tuning loss between symbolic input and audible output, with explicit supported paths and measured evidence for controlled 19-EDO material.

## ADDED Requirements

### Requirement: Check tuning against renderer capabilities
Rendering SHALL evaluate actual tuning intent, base tuning and source events rather than trusting EDO or microtonal labels alone. Direct 12-TET and verified 19-EDO MPE rendering SHALL be supported by appropriate routes. Unsupported MTS or other tuning arrangements SHALL fail explicitly; incompatible fallback SHALL NOT flatten pitches.

#### Scenario: Unsupported MTS path
- **WHEN** MTS input is selected for a backend without a verified MTS implementation
- **THEN** rendering rejects that path with an actionable tuning-capability diagnostic

#### Scenario: Nonstandard base tuning
- **WHEN** 12-EDO input has nonstandard base tuning or source pitch bends
- **THEN** capability selection evaluates those pitches rather than declaring it direct 12-TET solely because its EDO is 12

### Requirement: Preserve channel-scoped tuning through transforms
Groove and stem isolation SHALL preserve channel associations, pitch wheels, RPN bend range, MTS payloads and relevant controller/tempo events. A transformed note SHALL retain its intended channel pitch state. If retiming would create incompatible simultaneous pitch states on one channel, the transform SHALL reject the conflict rather than detune a note.

#### Scenario: Expressive 19-EDO MPE fixture
- **WHEN** groove and stem isolation process a supported 19-EDO MPE fixture
- **THEN** note pitches reconstructed from channel state differ from intended pitches by at most one cent, note counts remain correct, and tuning setup precedes affected notes

#### Scenario: Channel conflict after groove
- **WHEN** retiming causes notes needing different simultaneous bends to overlap on the same channel
- **THEN** processing rejects the conflict with track/channel/note diagnostics instead of producing silently detuned output

### Requirement: Measure controlled dry-render tuning
Tuning checks SHALL compare measured pitched-tone frequencies with explicit intended frequencies, report mean and P95 absolute cents errors and measurement coverage, and exclude percussion. Controlled reference tones SHALL be within five cents each. General check defaults SHALL reject mean or P95 errors above 15 cents; stricter configured limits SHALL be enforced.

#### Scenario: Correct controlled render
- **WHEN** supported dry 19-EDO reference tones with known steady-state measurement windows are rendered
- **THEN** each measured reference tone is within five cents of its intended frequency and the report includes expected/measured frequencies, mean, P95 and coverage

#### Scenario: Flattened corruption
- **WHEN** a reference set containing sufficiently non-12-TET pitches is deliberately rendered at nearest 12-TET pitches
- **THEN** the tuning check rejects it using the original intended frequencies, not a newly inferred 12-TET target

#### Scenario: Missing pitch evidence
- **WHEN** expected pitched reference windows are silent or have insufficient reliable frequency estimates
- **THEN** the check reports insufficient evidence and does not treat missing measurements as zero error or a passing result

#### Scenario: Percussion exclusion
- **WHEN** reference material includes a GM drum track
- **THEN** the track remains drum-key based and is excluded from pitched-frequency statistics with the exclusion recorded

### Requirement: Bound tuning claims to their evidence
Reports SHALL distinguish MIDI-event integrity, controlled monophonic audio measurements and unmeasured polyphonic passages. Failed tuning checks SHALL prevent a successful validated-demo result, and tuning metadata/check configuration SHALL participate in provenance and cache identity.

#### Scenario: Polyphonic listening example
- **WHEN** a multitrack 19-EDO example passes event-integrity checks but has no reliable polyphonic frequency measurement
- **THEN** the report labels its audio pitch accuracy unmeasured and does not claim that controlled-tone results certify every polyphonic note
