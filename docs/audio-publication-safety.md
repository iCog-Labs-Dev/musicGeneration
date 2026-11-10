# Audio output publication safety

aimusic.audio.publication is the publication boundary for externally selected audio outputs. It stages complete regular files privately on the output filesystem and fails closed when required primitives are unavailable. This is a trusted-local-output protocol, not protection from an attacker able to mutate the private transaction directory.

## Modes and portable names

- **Create-only** (replace=False) covers stems, final mixes, previews, generated sources, and cache restores. Every public leaf must be absent. Existing files, directories, links/reparse entries, and differently-cased casefold aliases are collisions. Atomic no-replace publication exposes either no leaf or the complete staged bytes.
- **Transactional replacement** (replace=True) is reserved for related groove MIDI/sidecar bundles. Every old leaf is retained privately before the first replacement. Complete new leaves are replaced one at a time; backups are discarded only after the caller accepts the publication and identities are revalidated.

Names must be single nonempty portable leaf components: no dot or dot-dot components, separators, traversal, controls, Windows-forbidden characters, reserved device basenames (including COM/LPT superscript forms), trailing space/dot, or reserved publication prefix. Duplicate names and Unicode casefold aliases are rejected within a plan and against existing entries, even on case-sensitive filesystems. Inputs and outputs must not alias by resolved path or file identity.

## Pinned containment and no-follow behavior

On currently supported POSIX hosts, every output ancestor and the output directory is opened with O_DIRECTORY | O_NOFOLLOW. Descriptors stay open through publication, rollback, and cleanup, while (st_dev, st_ino) identities are rechecked. Leaf stat, open, hard-link, replace, unlink, and private-directory operations are descriptor-relative and do not follow leaf links. Staged inputs must be regular files opened with O_NOFOLLOW; the selected output root cannot be a symlink.

This requires a local POSIX filesystem/runtime with those flags, directory-relative operations, same-filesystem hard links, atomic leaf replacement, and meaningful device/inode identities. Unsupported primitives produce an actionable error; no path-copy or overwrite-capable fallback is used.

## Immutable ownership, phases, and exception rollback

FileIdentity, Ownership, and Journal are frozen records. Journal updates create new values. Canonical JSON records the plan; original, staged, backup, and published identities; and these normal phases:

preparing -> prepared -> publishing -> published -> committed

The internal `transaction(...)` and `PinnedOutput.publication(...)` contexts yield
`tuple[Ownership, ...]` in plan order after all public names exist and the
`published` phase is recorded, but before commit. These are the transaction's
recorded staged identities, not a fresh pathname snapshot. Callers may use them
for verification or continue ignoring the context value. The full journal and
mutation handles are not exposed through this context value.

Exception handling records rolling_back; incomplete rollback records recovery when possible. Journal writes are flushed and fsynced, but interruption can leave a partial or stale record.

Any pre-commit exception, including one raised by the caller inside the context, replays ownership in reverse. A new public leaf is removed only while its identity matches. An original is restored only if its destination is absent or still has the identity published by this invocation. Unrelated and substituted entries are preserved. If restoration fails, publication fails and retains journal and recovery backups. A backup that may be the only old copy is never deleted merely to make cleanup succeed; the error identifies its transaction directory.

## Crash residue and explicit recovery

Process exit and power loss bypass exception rollback. Any inactive .stems-*, .groove-*, or .publication-* entry is ambiguous residue. Later pin_output() calls report exact names and refuse all publication; they never reuse, commit, restore, or delete residue automatically.

Recovery is an explicit operator decision:

1. Stop writers and preserve or copy the complete output tree.
2. Inspect each reported transaction directory. Treat journal.json as possibly truncated or one ownership step behind; inventory public leaves and private staged/backup-* entries by no-follow identity and content.
3. Reconcile the inventory with original, staged, backup, and published ownership records. Never overwrite an unrelated or identity-mismatched public leaf. Preserve every backup that may be the only old copy.
4. Explicitly choose and validate one coherent state: restore verified old leaves, or retain verified complete new leaves. A mixed groove bundle is not complete.
5. Archive evidence as needed, then manually remove only verified transaction-owned residue. Publication stays blocked until every ambiguous private-prefix entry is resolved.

There is no automatic recovery command because a damaged or partial journal cannot safely choose between old and new operator data.

## Non-atomic crash boundary and Windows support

Each public leaf has complete-file visibility, but a related set is **not** a single crash-atomic filesystem transaction. During replacement or after a crash, a set may contain old leaves, new leaves, absent leaves, and private backups. Normal exceptions provide rollback; process/filesystem crashes do not. A committed journal with failed cleanup is also residue to inspect, not permission to roll back after old copies may have been discarded.

Windows publication is supported only on writable, fixed, local NTFS volumes. UNC, network/device, read-only, reparse-point, and non-NTFS output paths fail closed with `Publication requires ...` diagnostics. No fallback attempts to approximate the contract.

The Windows backend reserves empty private leaves with `CREATE_NEW`, retains their handles, pins and revalidates the publication-directory identity and reparse state, then writes and flushes content only through those handles. Public names are created by same-directory retained-handle atomic no-replace renames. Directory pins do not lock the child namespace, so case-fold checks, atomic collision refusal, and identity-owned rollback handle competing child changes.

`FILE_SHARE_WRITE` on directory pins is rejected because it permits directory replacement or reparse conversion. Path-based copy, hard-link, `MoveFileExW`, and overwrite-capable rename fallbacks are rejected because they reopen ancestor races, expose partial public bytes, conflict with the pin, or can replace unrelated entries. Windows residue uses flat invocation-prefixed leaves; it is ambiguous after interruption and is never deleted automatically. Use the same explicit inventory and recovery procedure above, preserving every possible only copy.

The original `WinError 32` hard-link conflict and unavailable `os.O_DIRECTORY` route are regression-tested through native retained-handle publication. This native support does not make a related multi-file set crash-atomic, and native CI evidence remains distinct from optional FluidSynth renderer evidence.

## POSIX verification

From the repository root in the project dev environment, on a POSIX host:

    python -m pytest tests/test_audio_publication.py tests/test_audio_publication_records.py tests/test_audio_stem_publication.py -q
    python -m pytest tests/test_audio_groove.py tests/test_audio_stems.py tests/test_audio_demo_sources.py tests/test_audio_imported.py tests/test_audio_orchestrator.py tests/test_audio_mix.py -q

No FluidSynth, SoundFont, GPU, credentials, or network is needed. The first suite covers policy, residue, rollback, and adversarial POSIX mechanics. The second verifies migrated groove, stem/mix, source/preview, orchestration, and cache-facing workflows.

## Sealed reviewer archives (task 6.5)

The revised delivery boundary is **one ZIP file plus an independently conveyed
SHA-256**, implemented in `aimusic.audio.review_archive`. The abandoned directory
publisher and its red race suite were removed; they are not supported APIs.
Directory promotion was rejected because checking a mutable tree before renaming
it does not seal its children. Parent descriptors and extra scans do not repair
that verify-to-rename gap. This decision replaces that attempt, not its findings.

### Assembly and publication

`assemble_archive` takes the private `ReviewBundle`, reviewer `ReviewAssignment`,
their independently expected digests, and an explicit local input root. It checks
admission for every render, assignment/bundle links, and every unique declared
input's size/digest. It reads only those references, never discovers assets or
walks/copies a source tree. Private evidence and the blind map remain organizer
inputs, not reviewer payloads. Input reads retain the existing trusted-local,
no-link/no-reparse `review_bundle.read_relative` contract.

The archive contains exactly 22 regular members: `assignment.json`, six generated
HTML pages, fourteen opaque `media/s<hex>.wav` files, and `manifest.json`. The
versioned canonical manifest lists every payload's exact size and SHA-256, **not
itself**. Whole-archive trust protects the manifest. Pages are checked against the
reviewer-safe templates; WAVs must retain metadata-free bounded PCM/float RIFF
containers and the assignment's exact digests. Source paths, render provenance,
condition mapping, and private entropy are never copied into delivery.

ZIP members use sorted ASCII paths, stored compression (method 0), the timestamp
1980-01-01 00:00:00, Unix regular-file mode 0644, and fixed versions/attributes.
There are no directory entries, extras, comments, encryption, data descriptors,
ZIP64, appended bytes, or alternative compression methods. The full archive is
reopened and verified, including exact canonical re-encoding, before publication.
`write_review_archive` stages those checked bytes privately and calls the existing
single-file `pin_output` create-only layer. Inside `PinnedOutput.publication(...)`,
it reads the actual newly published leaf, requires its opened-handle identity to
match the yielded transaction `Ownership`, and compares it with the sealed bytes
**before transaction commit**. Backend/private second-copy corruption therefore
raises while normal exception rollback can remove the invocation-owned output;
checking only the first external staging copy would miss this fault. It returns
the archive SHA-256 only after success. Recipient verification against the
independently conveyed digest is still mandatory, including after interruption.
On POSIX this readback opens only the validated leaf relative to the retained
`PosixBackend.fd`, using `O_NOFOLLOW` and `O_NONBLOCK`. It never reopens
`pinned.path`. The opened descriptor must be a regular file whose
`(st_dev, st_ino)` matches the recorded staged identity before checking exact
expected size; reads are capped at 1 MiB and the remaining expected bytes plus one to
reject overrun, truncation, and differing content. Pre/post pathname checks alone
could accept a temporary valid decoy while leaving corrupt owned bytes behind.
The parent-substitution regression checks rollback through the retained
descriptor while the decoy directory still occupies the selected pathname. A
separate corruption-plus-leaf-substitution regression temporarily replaces the
public leaf with correct decoy bytes and restores the corrupt owned leaf before
commit/rollback. Identity-bound readback must reject that decoy; rollback removes
only the restored owned archive and preserves the decoy and unrelated entries.
On Windows the read-only leaf handle uses sharing compatible with the retained
write/delete transaction handle. Its volume/file identity must match the same
transaction-owned identity before checking size/content. This does not loosen
directory pins or retained handles. Existing destinations, casefold aliases,
links/reparse ancestors, and output locations inside the input root (including
filesystem aliases) fail.
Parents must already exist. Existing single-file transaction/recovery semantics
above still apply; no new mutable-directory atomicity claim is made.

### Verification and resource limits

`verify_archive(archive_bytes_or_path, expected_sha256=trusted_digest)` requires a
digest obtained independently of the received archive. A digest computed from an
untrusted download is not an authenticity check. Do not use a generic unzip tool
as the verification boundary.

Verification checks the archive digest, bounded end/central-directory metadata,
member count, portable paths, duplicate/casefold/prefix collisions, regular-file
types, supported ZIP fields, exact declared inventory, and all payload sizes and
digests. It rejects inconsistent local/central headers, offsets, CRCs and metadata.
All counts and byte totals are checked before reading ZIP payloads; compressed
members are rejected without invoking a decompressor.

`ArchiveLimits` can tighten, not enlarge, these ceilings:

- 32 members, including the manifest; 128 MiB whole-archive bytes.
- 127 MiB total stored bytes and 127 MiB total expanded bytes, including metadata.
- 32 MiB per member; 64 KiB each for manifest and assignment; 120 seconds per WAV.
- Assembly: 512 unique input files and 512 MiB total declared input bytes.

Memory is bounded (up to four archive-sized byte buffers plus one input file and
bounded metadata), not streaming/unlimited. Staged and extracted readbacks also
have exact per-file bounds. No network, renderer, SoundFont lookup or asset
discovery occurs during assembly, verification or extraction.

### Create-only extraction and interruption

`extract_archive(archive_bytes_or_path, new_directory, expected_sha256=trusted_digest)`
verifies **everything before creating the destination**. The local parent must
exist and pass no-link/reparse checks. The new root and `media` directory are
created exclusively, and each file is published create-only through `pin_output`.
The copied bytes are checked before publishing `complete.json` **last**, containing
only schema `aimusic.review-extraction/1` and the trusted `archive_sha256`.

Even an **exact canonical completion record with the correct digest is not proof
of current tree integrity**. An interrupted extraction may retain an incomplete
view or publication residue. Never serve it, resume it, overwrite it, or delete
ambiguous entries automatically. Preserve it for inspection and extract afresh
under a different name. No directory rollback or directory promotion is attempted.

### Verify the current view immediately before serving

Call `verify_extracted(archive_bytes_or_path, extracted_directory,
expected_sha256=trusted_digest)` immediately before opening or serving the pages.
It is read-only and requires the same independently trusted whole-archive digest;
it never learns trust from `complete.json`. It fully verifies the sealed archive,
then requires:

- The exact canonical completion bytes for that digest.
- The exact archive-member-plus-completion inventory, with only the root and
  `media` directories. Missing members, extra files/directories, and publication
  residue all fail.
- Every extracted member's exact bytes against the verified archive, including
  manifest/assignment digests, media digests, and reviewer-safe page templates.
- Bounded enumeration and exact per-file read limits, rejecting symbolic links,
  reparse points, and special entries without traversing extra directories.

For an already extracted view, the API sequence is:

```python
from pathlib import Path
from aimusic.audio.review_archive import verify_extracted

# trusted_digest comes from the independent trusted channel, not the download.
view = verify_extracted(Path("review.zip"), Path("review-view"),
                        expected_sha256=trusted_digest)
# Only now open/serve view; maintain trusted custody for the entire session.
```

**Trusted-local custody excludes concurrent namespace OR payload mutation** during
extraction, verification, and serving, even on a single-user machine. Hostile
multi-user locations and network shares are unsupported. A passing check is not a
snapshot, serving-time lock, or custody proof. If custody is uncertain, stop
serving, verify the sealed archive, and re-extract into a fresh trusted location;
do not repair the old view or merely replace its marker. Keep the private
graph/blind map outside the served tree.

The current path-based `mkdir` then pin sequence is **not safely pinned child
creation**. A swapped parent can cause creation outside the intended directory
before the post-check rejects it. An ordinary directory substitute can be adopted
before pinning, even yielding a byte-valid view. The deterministic POSIX tests in
`tests/test_audio_review_archive_security.py` demonstrate both unsupported cases;
post-checks do not prevent that escape. This limitation is not repaired by marker
or byte verification. The authoritative delivery remains the single sealed ZIP.
Native Windows execution remains separate evidence from macOS tests and simulated
reparse/sharing decisions.

### Installed-wheel operator commands (task 6.6)

The [sealed-archive runbook](audio-review-bundles.md#sealed-archive-commands-task-66)
contains exact POSIX and PowerShell commands for `assemble`, `verify`, `extract`,
and `verify-extracted`, independent digest recording, separate loopback-only
stdlib serving/browser opening, the 14-exposure limit, and private blind-map
custody. The CLI is `python -m aimusic.audio.review_archive`; it requires explicit
paths and trusted digests, checks bounded canonical bundle/assignment records,
and emits canonical JSON only on completed operations. It performs no network or
asset discovery and works from an installed wheel without audio extras.

`verify-extracted` calls the API `verify_extracted` immediately before serving;
it does not start a server or lock the view. Keep the authoritative archive and
independently authenticated digest separate from the disposable view. Neither
`manifest.json` nor `complete.json` supplies the trust anchor. No command grants
custody guarantees beyond the limitations above or automates recovery.

### Focused verification

Use the project development environment. On macOS, set `TMPDIR` to a physical,
non-symlink directory (the `/var` alias is intentionally refused by no-follow
publication). These tests use synthetic bytes and no licensed assets:

    python -m pytest tests/test_audio_review_archive.py tests/test_audio_review_archive_security.py tests/test_audio_review_archive_cli.py tests/test_audio_review_archive_docs.py tests/test_audio_review_bundle.py tests/test_audio_review_blinding.py tests/test_audio_review_pages.py tests/test_audio_publication.py tests/test_audio_publication_records.py -q
    flake8 aimusic tests ui.py
    mypy aimusic tests ui.py


The required installed-wheel Windows lane (`tests/windows_ci.py`) also selects
`test_audio_review_archive` and the portable security classes
`TestVerifiedExtraction`, `TestBackendCopyIntegrity`, and
`TestWindowsPublicReadback`. The ordinary publication/extraction and copy-fault
cases exercise the native Windows backend when run there; the last class remains
an explicitly simulated adapter check. POSIX-only descriptor/swap cases are not
selected, and the allowed-skip policy is unchanged.

`scripts/test-windows-audio.ps1` copies the entire `tests` package to the external
checks directory, preserving the archive → blinding → bundle fixture imports.
These helpers are not packaged in the product wheel. A clean `[audio,dev]` wheel
smoke on macOS passed all 30 selected archive tests outside the checkout, including
subprocess interruption, with no skips or FluidSynth binding. This verifies
packaging and helper imports, **not native Windows execution**; the expanded
Windows lane still needs its own run.
