# Tasks

## 1. Planning and branch integration

- [x] 1.1 Validate this change with `openspec validate harden-windows-audio-and-expand-listening-evidence --strict`, commit only the approved planning artifacts as `docs(openspec): plan Windows audio hardening`, and verify `git show --stat --oneline HEAD` lists the complete change directory.
- [x] 1.2 Fetch the current fork remotes, integrate `origin/main` into `feature/midi2audio-audible-demo` without dropping feature commits, resolve conflicts against the approved specs/design, run `python -m pytest tests/test_architecture.py tests/test_render_package.py -q` plus the existing audio publication/review tests, and record the integration as the Conventional Commit checkpoint `chore(sync): integrate current main for audio hardening` (use that message for the merge commit if a merge is required).
- [x] 1.3 Re-audit affected publication, cache, source, orchestration, evidence, and CI call sites after integration; update this change's design/tasks only if paths changed without altering scope, rerun strict OpenSpec validation, and amend the integration checkpoint only when it has not been shared.

## 2. Shared publication contracts and POSIX preservation

- [x] 2.1 Add failing contract tests for portable names, create-only collisions, case-fold aliases, complete-file visibility, source aliasing, invocation-owned rollback, transaction residue, and actionable unsupported-environment errors; verify the focused tests fail for the intended missing abstraction rather than unrelated setup.
- [x] 2.2 Introduce the internal publication policy, identity/ownership records, journal phases, and platform backend interface under `aimusic.audio`, then verify unit tests cover canonical serialization and refuse separators, traversal, controls, Windows reserved names, and trailing spaces/dots.
- [x] 2.3 Move the existing descriptor-relative POSIX mechanics behind the shared interface without weakening no-follow, pinned-parent, replacement-backup, or rollback behavior; verify all existing groove/stem/source/cache publication tests and new adversarial POSIX tests pass.
- [x] 2.4 Migrate groove bundles, stem/final mix publication, demo-source and MIDI-preview output, and cache restoration to explicit create-only or transactional-replacement policies; verify failure injection preserves unrelated bytes and restores prior groove bundles.
- [x] 2.5 Document the publication contract, supported modes, ownership journal, residue/recovery procedure, and non-atomic crash boundary beside the existing audio safety docs; verify every documented POSIX command/test target runs as written.
- [x] 2.6 Run `flake8 aimusic tests ui.py`, `mypy aimusic tests ui.py`, and the focused publication suite, then commit this group as `refactor(audio): centralize safe output publication` and verify the working tree contains no uncommitted group-2 files.

## 3. Security-preserving native Windows publication

- [x] 3.1 Add a native Windows API wrapper with explicit ctypes signatures, checked error propagation, handle lifetime management, volume/file identity queries, reparse/NTFS capability checks, and retained-handle same-directory rename/disposition operations; verify mocked API tests cover each success and failure branch on every host.
- [x] 3.2 Implement pinned ancestor/publication-directory acquisition that rejects reparse components and unsupported filesystems while preventing directory-object replacement or reparse conversion; document that child namespace mutation remains possible, and verify native tests detect identity changes and cannot convert the pinned directory into a junction during the operation.
- [x] 3.3 Reserve empty create-only invocation-private leaves directly in the publication directory, retain their handles, acquire the pin with a matching directory identity, then write and flush complete contents only through those handles before same-directory atomic no-replace publication; include case-insensitive collision refusal and identity-owned cleanup that tolerates unrelated child mutation, and verify native Windows tests cover existing files/directories/links/reparse points, concurrent collisions, substitutions, Unicode, spaces, long paths, identity-handoff races, no pre-handoff disclosure, and no partial public bytes.
- [x] 3.4 Implement transactional Windows replacement for groove bundles using retained old-leaf handles, flat invocation-owned recovery names, and a private journal; verify injected identity-handoff, mid-publish, and rollback failures restore old bytes or retain/report the only recovery copies without changing source MIDI.
- [x] 3.5 Replace the current Windows hard-link/path branch in all migrated callers and add end-to-end asset-free native tests for groove, stems/final mix, generated sources, MIDI previews, and cache restoration; verify the original `WinError 32` and `os.O_DIRECTORY` reproducers now pass on local NTFS.
- [x] 3.6 Add Windows threat-model and supported-filesystem documentation, including why `FILE_SHARE_WRITE`, path-based copy/link/rename fallbacks, network shares, and automatic residue deletion are rejected; verify unsupported cases fail with the documented diagnostic.
- [x] 3.7 Run the full native Windows publication suite plus lint/typecheck, commit this group as `fix(audio): publish outputs safely on Windows`, and verify `git show --check HEAD` reports no whitespace errors.

## 4. Required Windows CI and optional renderer evidence lane

- [x] 4.1 Add a required asset-free `windows-latest` CI job that builds a wheel, installs it into a path containing spaces and non-ASCII characters, runs import/architecture checks, and executes native publication/workflow tests without FluidSynth, SoundFont, administrator rights, or Developer Mode; verify the workflow with the same PowerShell commands locally or in a fork run.
- [x] 4.2 Split privileged reparse/link coverage into an explicit capability probe, required invariant substitutes, and narrowly reported optional cases so unavailable privilege cannot broadly skip the Windows job; verify CI logs distinguish exercised, substituted, and unavailable cases.
- [x] 4.3 Add an opt-in pinned Windows FluidSynth evidence workflow/script with explicit approved asset input and records for OS build, architecture, Python, binding/DLL/libsndfile versions, SoundFont digest, commands, outputs, cache behavior, and same-machine repetition; verify its dry/preflight mode needs no secret or asset and fails clearly when enabled prerequisites are absent.
- [x] 4.4 Update Windows setup/testing documentation to separate simulation, required native asset-free CI, and optional native renderer evidence; verify no generated WAV, SoundFont, credential, or machine-specific absolute path is tracked.
- [x] 4.5 Run a fork CI workflow or equivalent clean Windows job and the existing Linux jobs, commit this group as `ci(audio): verify native Windows publication`, and attach run identifiers/results to the implementation notes without opening an organization PR.

## 5. Review-bundle model and objective admission

- [x] 5.1 Add failing schema/validation tests for immutable source-render-presentation provenance nodes, SHA-256 links, effective config/backend/runtime/asset identity, four-state metrics, panel definitions, blind-map separation, assignment references, and external findings; verify malformed, missing, non-finite, and path-escaping records are rejected.
- [x] 5.2 Implement canonical review-bundle models and digest calculation with schema revisioning and safe relative paths; verify serialization is byte-stable and any referenced-file or metadata tamper changes identity or fails verification.
- [x] 5.3 Implement candidate readers that verify existing render manifests/reports and common admission gates for source identity, expected backend/asset, finite aligned audio, bounded peak, expected signal, event/routing evidence, and package reconciliation; verify rejected candidates cannot appear in public pages.
- [x] 5.4 Implement controlled 19-EDO admission using complete declared windows and the configured per-tone/aggregate 15-cent P95 floor, plus a deliberately flattened control measured against unchanged 19-EDO targets that must fail that floor; verify synthetic positive, missing-window, wrong-target, and expected-negative tests.
- [x] 5.5 Require `passed`, `failed`, `unmeasured`, or `not_applicable` records for CLAP, FAD, LUFS, rhythm F-score, note F1, polyphonic f0, and every applicable existing gate; verify omissions and inferred human-to-machine results are invalid.
- [x] 5.6 Document the manifest graph, objective-admission policy, rough-mix claim limits, tuning-control interpretation, and historical-hash non-transfer rule; verify examples pass the same schema validator.
- [x] 5.7 Run focused schema/admission tests, lint, and typecheck, then commit this group as `feat(audio): add review bundle admission` and verify the checkpoint contains no generated media.

## 6. Fixed 14-exposure generation and blinded presentation

- [x] 6.1 Extend deterministic source/workflow tooling to generate the exact matrix of four canonical mixes, four fixture role stems, package/import parity pair, groove-on/off timing pair, and valid/flattened tuning pair; verify a manifest test counts exactly 14 exposures and includes the polyphonic 19-EDO boundary without claiming its f0 accuracy.
- [x] 6.2 Implement bounded, deterministic trim/gain presentation transforms that preserve source files and record parameters plus source/output digests; verify transformed WAVs retain allowed containers, bounded peaks/bytes/duration, and no ancillary metadata.
- [x] 6.3 Implement opaque stimulus names, private condition mapping, and deterministic reviewer-specific balanced A/B order derived from bundle identity and reviewer pseudonym; verify public HTML/media/filenames reveal none of the condition labels, original paths, or blind-map fields.
- [x] 6.4 Generate purpose-grouped offline pages for canonical listening, stems, parity, timing, and tuning with accessible controls and no network dependency; verify browser smoke tests, CSP, missing-media behavior, keyboard access, and extracted-bundle playback.
- [x] 6.5 Publish deterministic reviewer-safe archives create-only with transaction-owned identity/byte verification and rollback. Require independently trusted digests, bounded strict archive validation, create-only extraction, and exact extracted-view verification before serving under trusted-local custody. Cover tampering, malformed archives, path/alias/reparse hazards, resource limits, copy/namespace faults, residue and false-completion prevention, including portable installed-wheel Windows tests.
- [x] 6.6 Provide explicit installed-wheel matrix → prepare → assemble commands and separate offline verification/extraction in POSIX and PowerShell runbooks. Keep preparation lazy, create-only and fail-closed; document prerequisites, exposure limits, trusted digests and private/serving custody. Verify synthetic generation, stdlib-only archive operations, UTF-8 subprocesses and native Windows runner selection.
- [x] 6.7 Run matrix, blinding, browser, tamper, and publication tests, then commit this group as `feat(audio): generate blinded listening panels` and verify only source/tests/docs—not generated bundles—are tracked.

## 7. Structured findings and scoped reporting

- [x] 7.1 Add failing tests for reviewer pseudonyms, bundle/assignment/stimulus digest links, single and pair judgments, perceived difference, defects, preference, confidence, notes, completion state, duplicate responses, changed media, and unknown stimuli.
- [x] 7.2 Implement append-only finding templates and validation that keeps incomplete responses out of acceptance totals without mutating the bundle; verify valid disagreements remain separate and invalid/tampered findings fail closed.
- [x] 7.3 Implement authorized unblinding and report generation that combines machine gates with counts and individual linked human findings while preserving assignments, disagreements, negative-control disclosure, `unmeasured` metrics, and rough-demo limits; verify snapshots contain no inherited approval or unanimous claim when findings disagree.
- [x] 7.4 Document reviewer instructions, low-volume safety, defect vocabulary, completion/withdrawal handling, privacy limits for pseudonyms, unblinding procedure, and explicit prohibition on fabricated findings; verify a fresh template validates as incomplete and a completed fixture validates as complete.
- [x] 7.5 Run finding/report tests, lint, and typecheck, then commit this group as `feat(audio): record provenance-linked listening findings` and verify test fixtures contain no real reviewer identity.

## 8. Fresh native evidence and human review

- [ ] 8.0a Repair the hosted `windows-latest` standard-user `Start-Process` bootstrap failure from fork run `42789164898`, remove the explicit false job condition, and require the complete asset-free Windows job to pass before relying on native Windows evidence; do not replace it with elevated execution or broader skips.
- [ ] 8.0b Resolve MUS-40 by preventing reviewer-visible canonical media from serving as a digest oracle for blinded pair members; add a regression test that attempts condition recovery from the complete reviewer-visible assignment/media inventory before distributing any assignments.
- [ ] 8.1 On the approved native Windows renderer environment, execute the pinned canonical and 14-exposure workflows from an installed wheel in a Unicode/space path, including cold/warm cache and same-machine repeat runs; verify all required provenance, admission, output, and cache reports and their digests before sealed-archive publication.
- [ ] 8.2 Run sealed-archive and extraction verification, deliberate archive/media/manifest tamper checks, offline browser checks, and Windows-versus-supported-platform semantic comparisons; verify semantic gates are reported without cross-platform PCM/WAV hash requirements.
- [ ] 8.3 Distribute blinded assignments to the approved reviewer set and collect actual completed finding files without exposing the private map or fabricating responses; verify every included finding links to the exact bundle/assignment/media digests and preserve incomplete or disagreeing responses as specified.
- [ ] 8.4 Generate the authorized unblinded evidence report, record measured objective and human results plus every `unmeasured` metric, and update audio status/testing/listening docs without transferring the prior four-demo acceptance or claiming mastering/production quality; verify all claims cite scorer/gate records or reviewer counts.
- [ ] 8.5 Run strict OpenSpec validation, `python -m pytest tests/test_architecture.py tests/test_render_package.py -q`, the full core suite, focused audio suites, lint, typecheck, wheel/import checks, and native Windows verification; commit source/test/docs evidence (never WAVs, SoundFonts, secrets, or machine-private maps) as `docs(audio): record Windows listening evidence` and verify the branch remains ready only for fork review.

## Workflow follow-up

- Push to the fork or open a fork PR only when separately requested; never open an organization PR without explicit approval.
- After implementation and review are complete, run `/opsx-archive` to synchronize and archive this change.
