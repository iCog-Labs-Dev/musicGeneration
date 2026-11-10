# Spec Delta

## Purpose

Provide bounded, reproducible human-listening evaluations whose exact audio stimuli, objective admission evidence, blinding assignments, and reviewer findings remain traceable without implying unsupported quality claims.

## ADDED Requirements

### Requirement: Bounded representative review set
The review workflow SHALL produce a versioned, bounded set covering canonical mixes, role stems, package/import parity, a meaningful timing comparison, a valid-versus-flattened tuning control, and the supported polyphonic microtonal boundary.

#### Scenario: Complete broad review generation
- **WHEN** approved native prerequisites are available and every required source renders successfully
- **THEN** the bundle contains all declared panels with stable stimulus identities and no more than the configured reviewer exposure limit

#### Scenario: Required panel cannot be produced
- **WHEN** any required stimulus fails rendering or admission
- **THEN** no complete review bundle is reported and the failed panel remains explicitly identified

### Requirement: Objective admission before presentation
Every presented stimulus SHALL pass its applicable integrity, provenance, renderer, signal, reconciliation, event, and controlled-tuning gates before entering a reviewer-visible page.

#### Scenario: Admitted direct-12 stimulus
- **WHEN** a direct-12 render has verified source identity, expected backend and asset, preserved event/routing evidence, finite aligned audio, non-silence where expected, and bounded peak
- **THEN** the stimulus may be included with its gate evidence and exact WAV digest

#### Scenario: Controlled tuning stimulus
- **WHEN** a controlled 19-EDO reference has complete measurable windows and satisfies the configured per-tone and aggregate cent-error floors
- **THEN** it may be included as the valid member of the tuning comparison

#### Scenario: Flattened tuning control
- **WHEN** a deliberately flattened control is measured against the unchanged original frequency targets
- **THEN** it is included only if it fails the acceptance floors as expected and is labelled privately as a calibration control

#### Scenario: Unmeasured metric
- **WHEN** a metric such as CLAP, FAD, LUFS, rhythm F-score, note F1, or polyphonic f0 has not been implemented and run
- **THEN** the bundle records it as `unmeasured` rather than absent, passed, zero, or inferred from human feedback

### Requirement: Blinded comparison presentation
Comparison panels SHALL use opaque stimulus identifiers and reviewer-specific balanced ordering so that labels, file names, paths, and page text do not reveal the intended control or production route.

#### Scenario: Reviewer opens a comparison page
- **WHEN** a reviewer receives a generated assignment
- **THEN** the page presents opaque IDs in that assignment's order and excludes the private blind map and source-revealing metadata

#### Scenario: Review closes
- **WHEN** all intended reviewer findings are finalized
- **THEN** an authorized unblinded report may map findings to stimuli while preserving the original assignments and hashes

### Requirement: Structured human findings
The workflow SHALL collect findings separately from machine evidence and SHALL link each response to the bundle, assignment, reviewer pseudonym, stimulus or pair, and exact media digests.

#### Scenario: Reviewer records a pairwise judgment
- **WHEN** a reviewer evaluates an A/B pair
- **THEN** the finding records difference perception, defects, preference, confidence, optional notes, and completion state without modifying the source bundle

#### Scenario: Incomplete form
- **WHEN** required judgments are missing or a finding references an unknown or changed stimulus
- **THEN** the finding remains incomplete or is rejected and cannot contribute to acceptance totals

### Requirement: Immutable provenance graph
The review bundle SHALL provide machine-readable links from each stimulus to its source, render, presentation transform, effective configuration, backend, asset, runtime, applicable reports, and cryptographic digests.

#### Scenario: Stimulus verification
- **WHEN** a recipient verifies a stimulus from the bundle
- **THEN** its bytes and every referenced evidence document match the declared digest chain or verification fails

#### Scenario: Presentation gain is applied
- **WHEN** a clip is gain-matched, trimmed, or otherwise transformed for presentation
- **THEN** the transform parameters, source WAV digest, output digest, and non-destructive relationship are recorded

### Requirement: Portable bounded delivery
The workflow SHALL deliver bounded reviewer media and purpose-grouped offline pages as one deterministic sealed archive published exclusively as a single file.

#### Scenario: Sealed archive publication
- **WHEN** every required candidate, assignment, page, and media member has passed admission and archive verification
- **THEN** one complete archive is published create-only, its opened public leaf identity matches the transaction-recorded staged `Ownership` before size/content are verified against the sealed bytes inside the publication transaction before commit, and its SHA-256 is conveyed independently; no reviewer-visible directory is used as the publication boundary

#### Scenario: Backend copy corrupts the archive
- **WHEN** the backend/private second copy differs from the sealed archive despite successful external staging verification
- **THEN** public-byte verification fails before commit, normal exception rollback removes the invocation-owned archive, and unrelated entries remain unchanged

#### Scenario: POSIX readback during temporary parent substitution
- **WHEN** a corrupt backend/private copy is published and a valid decoy directory temporarily occupies the selected parent pathname between path checks
- **THEN** public-byte verification opens the validated leaf relative to the retained output directory descriptor without reopening the parent pathname, enforces no-follow regular-file exact-size/content and bounded overrun checks, fails before commit, and rolls back the actual owned archive without changing the decoy or unrelated entries

#### Scenario: Correct-byte decoy replaces a corrupt published leaf temporarily
- **WHEN** backend/private copy corruption is followed by temporary public leaf substitution with correct decoy bytes and restoration of the corrupt owned leaf before the commit identity check
- **THEN** readback rejects the opened POSIX descriptor or Windows retained-compatible read handle unless its identity matches the transaction-recorded published ownership before size/content checks, the publication fails before commit, and rollback removes only the restored owned archive while preserving the decoy and unrelated entries

#### Scenario: Portable archive contracts in native Windows CI
- **WHEN** the asset-free installed-wheel Windows lane runs the portable archive suites from its external checks directory
- **THEN** archive publication, verification, extraction, copy-corruption rollback, and completion integrity are required to pass using synthetic fixtures, helper imports do not depend on the checkout, and POSIX-only tests do not broaden the skip policy; simulated adapter checks remain distinct from native execution

### Requirement: Strict archive verification and extraction
The archive SHALL contain a versioned canonical manifest with exact sizes and SHA-256 digests for every payload member. The manifest SHALL be protected by an independently conveyed whole-archive SHA-256, and the complete archive SHALL be verified before extraction.

#### Scenario: Invalid or over-budget archive
- **WHEN** an archive has a wrong trusted digest, malformed or unsupported ZIP feature, duplicate or case-fold-colliding name, unsafe path, link or special entry, undeclared or missing member, member digest/size mismatch, or cumulative count/compressed/expanded-budget violation
- **THEN** verification and extraction fail before any destination directory is created or any member is exposed

### Requirement: Trusted-local extraction custody
Extraction, extracted-tree verification, and serving SHALL require trusted-local custody without concurrent namespace or payload mutation. Hostile multi-user locations and network shares are unsupported. The extracted tree is a disposable view, not a publication boundary, immutable snapshot, or custody proof. If custody is uncertain, the recipient SHALL verify the sealed archive and re-extract into a fresh trusted location rather than serve or repair the old view.

#### Scenario: Portable extracted bundle
- **WHEN** a recipient verifies the sealed archive against its independently conveyed trusted SHA-256 and extracts it into a new destination under trusted-local custody without concurrent mutation
- **THEN** all declared admitted media and pages are created exclusively and an archive-digest completion record is written last; only after `verify_extracted` succeeds immediately before serving may the pages be served without network access, credentials, repository checkout, or the licensed SoundFont, with custody maintained throughout serving

#### Scenario: Namespace mutation during path-based directory creation
- **WHEN** another actor swaps a parent or substitutes an ordinary directory between path-based mkdir and pinning
- **THEN** the operation is outside the trusted-local extraction contract: a post-check may detect the swap only after creation outside the selected parent, and an ordinary substitute may be adopted; neither completion nor passing byte verification proves custody or prevents that escape

#### Scenario: Existing or unsafe output
- **WHEN** under the supported custody contract the selected archive or extraction destination exists, aliases a source, or has a disallowed link or reparse ancestor
- **THEN** publication or extraction fails without overwriting the existing entry

### Requirement: Verify extracted views before serving
Immediately before serving, `verify_extracted` SHALL fully verify the archive with the independently trusted digest, require exact canonical completion and inventory, and compare every extracted member with the verified archive bytes, digests, and reviewer-safe templates. Even exact completion alone SHALL NOT establish current tree integrity. Enumeration and reads SHALL be bounded and reject symbolic links, reparse points, and special entries.

#### Scenario: Exact completion beside an altered tree
- **WHEN** the completion record is exact but a payload is corrupted or missing, or an extra file, directory, or publication residue is present
- **THEN** extracted-tree verification rejects the view and it must not be served

#### Scenario: Read-only complete-tree verification
- **WHEN** an extracted view is checked immediately before serving
- **THEN** verification changes no files, requires the exact archive-member-plus-completion inventory and only required directories, uses exact per-file read limits and bounded enumeration, and does not follow symbolic links or traverse extra directories

### Requirement: Claims remain scoped to new evidence
Generated reports SHALL distinguish objective gates from subjective findings, retain disagreements, and SHALL NOT transfer historical acceptance to new audio hashes or describe rough mixes as mastered or production quality.

#### Scenario: New render after earlier acceptance
- **WHEN** any stimulus bytes, source, configuration, backend, runtime, asset, or presentation transform change
- **THEN** prior human acceptance is historical only and the new stimulus begins with no human approval

#### Scenario: Multiple reviewers disagree
- **WHEN** reviewers provide conflicting defect or preference judgments
- **THEN** the report presents reviewer counts and individual linked findings without converting disagreement into unanimous acceptance

### Requirement: Windows review evidence
The canonical and broad review workflows SHALL be reproducible on native Windows with platform-specific provenance while semantic gates remain comparable across supported operating systems.

#### Scenario: Windows bundle generation
- **WHEN** the pinned Windows native-render lane completes using paths with spaces and non-ASCII characters from an installed wheel
- **THEN** it produces a verifiable review bundle and records OS build, architecture, Python, native DLLs, binding, libsndfile, SoundFont, commands, browser checks, and same-machine repeat evidence

#### Scenario: Cross-platform comparison
- **WHEN** Windows and another supported platform render the same declared sources
- **THEN** reports compare semantic gates and provenance without requiring cross-platform PCM or WAV byte equality
