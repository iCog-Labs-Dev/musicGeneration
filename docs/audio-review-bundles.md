# Review bundle graph and objective admission (schema 1)

The model/admission, assignment, page-rendering and sealed-archive layers are
implemented. External [finding collection and authorized reporting](audio-review-findings.md)
are available; this is not evidence of a new human review. The historical four-demo tool and
its [recorded acceptance](audio-listening-review.md) remain separate.

## Immutable graph and custody

`aimusic.audio.review_bundle` uses frozen dataclasses. Nested configuration,
backend and runtime JSON is copied into read-only mappings/tuples. Constructors
and `from_bytes` reject unknown/missing fields, invalid enums, non-finite numbers,
mutable typed sequences, malformed digests and unsafe paths. The shared
`review_identity` validator checks nested effective config/backend/runtime shapes
without coercion or defaults, and is used before candidate payload reads too.

```text
source files + identity -> SourceNode digest
                             |
render manifest + all reports + effective config/backend/runtime/assets
                             -> RenderNode digest + explicit machine metrics
                                           |
render WAV digest + sample trim bounds + gain + output WAV digest
                             -> PresentationNode digest
                                           |
                             opaque Stimulus -> purpose Panel
                                           |
                             immutable ReviewBundle digest
                                           |
             external assignment reference -> external finding reference
```

- `SourceNode` binds the MIDI file bytes to its identity. RenderPackage sources
  retain the package/source aggregates and payload files, not just the MIDI.
- `RenderNode` names the exact source node, render manifest, report files,
  effective config, requested/effective renderer, fallback, capabilities, backend
  code/version, runtime dependency versions, and SoundFont SHA-256 (or explicit
  `null` for an asset-free backend). The manifest additionally retains the
  orchestrator's full code inventory. No licensed asset is copied or discovered.
- `PresentationNode` links its source WAV to render evidence. The output uses an
  opaque `media/s<hex>.wav` path; it cannot overwrite the source path. Only explicit
  half-open sample trim bounds and finite positive gain (maximum 16) are modeled.
  `review_presentation.write_presentation` creates these copies from an admitted
  render and a verified source `FileRef` in an existing private tree. It preserves
  rate, channels and WAV subtype, applies gain in float64, and retains only
  fmt/fact/data chunks (no timestamp-bearing PEAK chunk). Source files are never
  rewritten. Source and output are capped at 32 MiB, 120 seconds and peak .95;
  output caps may be tightened. Both pre-encoding and quantized peaks must pass.
  It returns the exact digest-linked node after create-only publication and a
  source recheck; failure rolls back only owned output. Repeatability is within
  the same numeric/libsndfile runtime. Merely declaring a `PresentationNode` still
  does not certify samples. Sealed-archive assembly below verifies the complete delivery.
- Schema 1 fixes panel sizes: canonical 4, roles 4, parity 2, timing 2, tuning 2.
  Exactly 14 unique exposures are required. The tuning pair must have one measured
  valid result and one failed tuning result with a passed negative-control gate.
  `ReviewMatrix` fixes the selected fixtures; assignment generation is described below.
- The **whole provenance manifest is private**, not a page data source. Its
  `blind_map` references a separate file under `private/`; it never embeds that
  mapping. `PrivateBlindMap` must cover precisely the stimulus set and label the
  negative member `flattened-12-calibration-control`. Keep both graph and map away
  from reviewers until authorized unblinding. Blinding cannot prevent inference
  from audible musical content.
- `public_projection(root=..., expected_sha256=...)` refuses rejected gates and
  verifies all referenced bytes and the private map before returning only bundle
  digest, purpose panels, opaque IDs and media references. The new generation
  layer must use it, not directly serialize the private graph. Public pair inventories
  and the global stimulus inventory are sorted by opaque ID, never private condition
  order. Only an assignment determines A/B playback order. It returns data,
  not HTML, and does not modify the historical four-demo page.

Every path is a portable, nonempty relative path. Traversal, absolute paths, URLs,
backslashes, empty components, percent escapes, Windows device names, controls,
trailing dots/spaces and alternate streams are rejected. Readers reject
nonregular files and link/reparse components, including ancestors, and bound each
file to 32 MiB. Candidate audio is bounded to 120 seconds. Cumulative exposure/media
budgets and create-only single-file publication are enforced by the sealed-archive
assembler, not promised by these read-only models.

Use trusted local roots. These readers check identities, bounded reads and final
hashes; they do **not** claim retained-handle protection against hostile concurrent
ancestor replacement. The shared [publication contract](audio-publication-safety.md)
is the authority for writing archives and extracting local views.

### Strict render provenance (same native manifest shape)

`RenderNode` validates nested identities as well as its outer fields. The same
non-coercing validator runs immediately after the admission reader pins
`render.json`, before reading its cache or audio payloads. It does not load a
renderer or normalize/fill in missing provenance. Mappings remain deeply frozen;
the serialized field names and native manifest pair-array format are unchanged.

- Effective config requires `render`, all seven boolean `stages`, and the sanitized
  `paths` object. Render requires a backend, positive integer sample rate (at most
  768000), boolean fallback policy, routing map, nonnegative release tail and
  positive mix peak at most 0.95. Only the existing optional config sections and
  their declared fields are accepted. An enabled expressivization stage requires
  its effective groove spec. No implicit config defaults are supplied here.
- Application backend requires exactly `requested_backend`, `effective_backend`
  (`native` or `m2a`) and explicit `fallback_reason`. Unchanged selections require
  a null reason; changed selections require nonempty text.
- Runtime requires exactly `python`, `platform`, and a nonempty dependency map.
  Every dependency name/version is bounded nonempty text. Versions are strings,
  not coerced numbers; native revision strings and `not-installed` dependency
  sentinels remain valid (no invented semantic-version requirement).
- Every renderer requires the native selection, backend, asset, source-track/role,
  and tuning-evidence fields. The backend requires name/version, lowercase SHA-256
  code digest, unique nonempty native-version pairs, settings pairs, and capabilities.
  Settings values are finite numbers or bounded strings, not booleans/null.
  Capabilities require a revision, boolean instrument-fidelity flag and unique
  nonempty tuning modes (`direct_12`, `channel_pitch`, or `mts`).
- Backend name must equal effective selection; required tuning must be supported;
  requested selection must match config routing. A renderer fallback must be
  explicitly allowed and target `simple`. FluidSynth requires a SoundFont digest;
  the asset field otherwise remains explicit null-or-digest, preserving manifests
  that record a selected asset even with a simple/fallback renderer. Channel-pitch
  FluidSynth additionally requires well-typed pitch-event evidence.

All identity text is limited to 4096 characters without controls. Required values
cannot be missing/null, except explicitly nullable fields above. JSON booleans
are never accepted as numeric/string identities. Unknown identity fields require
an explicit schema decision instead of silently surviving as unchecked JSON.
These are structural/consistency checks, not independent proof of honest version
claims or permission to bypass the objective admission gates.

### Canonical identity and external references

Canonical JSON is UTF-8 with ASCII escapes, sorted object keys, compact separators,
no whitespace/newline, and no NaN/Infinity. Array order remains meaningful (notably
panel/assignment order). SHA-256 is lowercase, unprefixed, 64 hex characters.
Identity is over canonical record bytes, without a self-hash field; `sha256` on
graph records is derived. `FileRef.sha256` instead hashes the exact referenced file
bytes, independent of whether the file happens to be JSON.

`ReviewBundle.from_bytes(data, expected_sha256=trusted_digest)` validates metadata
identity. `bundle.verify(root, expected_sha256=trusted_digest)` additionally checks
every referenced file's exact size/digest and the private map. Save the approved
bundle digest separately. Hashes detect change relative to a trusted identity;
they are not signatures or proof of honest authorship. Rewriting every payload
and hash produces a *different* identity, not a repair of the original review.

Assignment and finding records are **external** so their addition never changes
the bundle digest or introduces a hash cycle. The bundle declares safe assignment
and finding directories. `AssignmentReference` links bundle digest, pseudonym,
external document FileRef, and exact stimulus/media digests; `validate_bundle`
checks those links. `ExternalFindingReference` links bundle, assignment-reference
digest, pseudonym and another FileRef; `validate_assignment` checks those links.
Verify each external document with `document.read_verified(root)` before using it.
These are provenance references, not completed responses: append-only response
schemas, required judgments, authorized unblinding and acceptance counts are later
work. No field in this bundle grants human approval.

## Opaque names and reviewer assignments (task 6.3)

`aimusic.audio.review_blinding` adds read-only graph/assignment helpers, not pages
or complete-directory publication:

1. `create_blind_map(matrix)` allocates a separate `s<64 lowercase hex>` ID for
   every exposure, including repeated audio, plus a private 256-bit `order_seed`.
   Neither IDs nor filenames encode conditions, indices, source hashes or paths.
   Entropy collisions retry at most 32 times per token, then fail closed. Retain
   this map; fresh allocation intentionally creates a different bundle identity.
   The injectable entropy callback is for synthetic tests, not production seeding.
2. Pass `media/<ID>.wav` to `write_presentation` for each mapped exposure, with
   explicit trim/gain parameters. `bind_presentations(matrix, blind, presentations)`
   checks exact coverage and each render/source link, then returns the private
   `ReviewBundle`. It does not write or verify files. The caller retains map bytes
   at `private/blind-map.json` and the complete private provenance graph.
3. `create_assignment(bundle, pseudonym, root=..., expected_sha256=...)` checks
   admission and all bundle references through `public_projection`. It returns a
   frozen, strict `ReviewAssignment` (`aimusic.review-assignment/1`). There are no
   writes, network calls, asset discovery, HTML or human judgments.

Each reviewer gets 14 exposures once, grouped by purpose. Canonical and role
orders stay fixed. The three pair arrays are A then B. HMAC-SHA256 uses the private
seed and a versioned message containing bundle identity, pseudonym and a ranking
label. The ranked first one or two pairs are reversed, selected by another keyed
bit. Thus forward/reverse counts differ by one **within each reviewer**; arbitrary
reviewer sets are not guaranteed equal aggregate counts. No global RNG is used.
Retaining the same bundle/map and pseudonym reproduces identical assignment bytes.
The secret seed prevents reviewers from reconstructing condition order using the
public bundle digest and pseudonym alone. It is never included in reviewer data.

`assignment.reference(bundle)` returns the existing `AssignmentReference` contract,
checking that this is the assignment's exact bundle. Its file is
`<bundle.assignment_directory>/a<assignment SHA-256>.json`; the default directory
is `assignments`, but nested declared directories are supported. The opaque leaf
contains no pseudonym or condition. Choose neutral directory names for reviewer
references. The directory and private bundle fields are not added to assignment
documents. Its digest links cover all 14 media files. Only schema, bundle digest,
pseudonym, purpose panels and opaque media references enter assignment bytes.
`assignment.validate_bundle(bundle)` checks private graph links;
`assignment.verify(reviewer_root, expected_sha256=trusted_assignment_digest)` checks
identity and all media without needing any private files. Verify serialized
assignment bytes with `reference.document.read_verified(root)` before parsing.
Hashes detect tampering relative to trusted references; they are not signatures.

Pre-assignment maps with only `conditions` preserve their canonical bytes and
trusted digest on parsing, reserialization and nested serialization. Seedless maps
omit `order_seed`; seeded maps add a SHA-256-shaped value. Explicit JSON null is
rejected rather than introducing a third wire shape. All canonical helpers use
the model's `to_dict` wire shape. Seedless maps cannot generate assignments.
Adding a seed changes map bytes
and bundle identity; it cannot inherit old approval. Never distribute the map,
seed or whole graph to reviewers. Pseudonyms must be nonempty ASCII letters,
digits, `_` or `-`, at most 64 characters; do not supply real identities. Audible
content and equal-media digests can still allow inference. This layer does not
claim to prevent that, certify unverified presentation samples, or publish a
sealed distributable archive (task 6.5).

## Offline reviewer pages (task 6.4)

`aimusic.audio.review_pages.render_review_pages(assignment,
expected_sha256=approved_assignment_digest)` returns a dictionary of six fixed
relative filenames to UTF-8 bytes: `index.html`, `canonical.html`, `roles.html`,
`parity.html`, `timing.html`, and `tuning.html`. It accepts only a strict
`ReviewAssignment`, produced through the admitted public projection, and checks
its trusted metadata digest again. It does not read the graph, map or media, write
files, publish a bundle, or establish new admission/acceptance. The archive assembler
verifies media and publishes a sealed reviewer-only ZIP. Never serve the
private provenance tree alongside these pages.

All media links are `media/s<hex>.wav`; page filenames contain no reviewer identity.
Only opaque clip/media IDs, purpose labels and bundle/assignment hashes are shown.
The pseudonym is deliberately not echoed: even a valid pseudonym could disclose a
condition. Pair playback order comes only from the assignment, not sorted inventory.

Pages use no scripts, dependencies or remote assets. Their early meta CSP forbids
scripts, connections, objects, base URLs and forms; it allows only same-origin
media and the exact hash of the inline stylesheet. A CSP meta tag cannot enforce
`frame-ancestors`; these pages are intended for trusted local serving, not an
internet-hosted application. The index explains loopback-only serving from the
extracted reviewer directory with Python's standard HTTP server.

Players have labelled native controls, no autoplay or preload, visible keyboard
focus, a skip link, panel navigation, and low-volume guidance. Recovery instructions
and opaque WAV downloads remain visible outside every audio element, even with
missing/unsupported media and scripting disabled. A failed download means re-extract
or request a verified copy and mark the clip **not reviewed**, never substitute
another file. Static pages cannot diagnose file integrity or save findings.

`tests/test_audio_review_pages.py` checks HTML structure, keyboard-accessible names
and links, CSP/style hashes, privacy, and an extracted synthetic bundle served on
loopback (all pages, decodable WAV bytes, downloads, and a missing-media 404).
These asset-free tests require neither Chrome nor a renderer; they do not claim
real-browser playback, screen-reader, or human-listening evidence.

## Installed-wheel generation and preparation (task 6.6)

Provision Python 3.11+ and the wheel's `[audio-render]` extra before generation
(`[audio]` suffices for preparation). Native generation additionally requires the
approved local FluidSynth library/binding and a license-approved SoundFont; follow
[audio-fluidsynth.md](audio-fluidsynth.md) or the pinned Windows setup. Installation
and asset approval are separate steps, never automatic discovery or fallback.
Supply a retained **12-EDO RenderPackage directory** and the exact SoundFont file.
Do not use the schema-only JSON example as candidate evidence.

The following sequence runs from an installed wheel, without a checkout.
`-X utf8` explicitly enables UTF-8 for redirected Unicode-path output. It makes
seven admitted renders / 14 exposures, records the freshly generated matrix's
SHA-256 in a separate create-only private ledger, then prepares and assembles it.
Hashing is justified **only because you just generated it under continuous trusted
custody**. For a received matrix, use an independently approved matrix digest
instead; hashing received bytes does not establish trust.

`review_archive prepare` requires explicit `--input-root`, `--matrix`,
`--expected-matrix-sha256` and `--reviewer`. The matrix must be directly inside the
fresh private matrix root. It verifies bounded canonical matrix bytes, allocates
opaque IDs/secret ordering entropy, and calls `write_presentation` with frame zero
to the full clip end and gain **1.0**, without resampling, normalization or trimming.
It retains the source/render bytes and verifies the complete graph before creating
`private/blind-map.json`, `bundle.json` and `assignments/a<digest>.json`.
`prepared.json` is the final canonical result, identical to successful stdout:
`schema`, `operation: "prepare"`, `input_root`, `matrix`, `matrix_sha256`,
`blind_map`, `blind_map_sha256`, `bundle`, `bundle_sha256`, `assignment`, and
`assignment_sha256`. Paths are absolute; records have no BOM or trailing newline.
Only `prepare` lazily loads audio/matrix modules; archive help/assembly/verification/
extraction do not load renderers, numpy or soundfile. Preparation opens no SoundFont.

All outputs are **create-only**. `prepare-started.json` reserves the one-shot run;
it is not completion. On any failure stop: no success stdout is emitted. Preserve
residue for inspection, regenerate the matrix in a **fresh private root**, and
never resume, repair, overwrite or infer completeness from partial metadata/media.
The set is not crash-atomic. These commands require the same private local custody
and absence of concurrent namespace or payload mutation as archive operations
below; `mkdir`/post-checks do not make hostile storage safe. Use private temporary
storage and separate existing delivery/ledger parents. Never serve the private root.
Pseudonyms are bounded ASCII identifiers, not real reviewer identities.

### POSIX organizer: matrix → prepare → assemble

<!-- archive-posix-generate -->
```sh
set -eu
umask 077
PYTHON='/absolute/venv/bin/python'
PACKAGE='/absolute/retained/run_hash'
SOUNDFONT='/absolute/approved/instrument.sf2'
INPUT_ROOT='/absolute/private-inputs'
MATRIX='/absolute/private-inputs/matrix.json'
MATRIX_RECORD='/absolute/trusted-ledger/matrix.sha256'
ARCHIVE='/absolute/delivery/review-r01.zip'
DIGEST_RECORD='/absolute/trusted-ledger/review-r01.sha256'
REVIEWER='r01'
"$PYTHON" -X utf8 -m aimusic.audio.review_matrix --package "$PACKAGE" --soundfont "$SOUNDFONT" --out "$INPUT_ROOT"
MATRIX_SHA256=$("$PYTHON" -X utf8 -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "$MATRIX")
"$PYTHON" -X utf8 -c "import os,sys; from pathlib import Path; f=Path(sys.argv[1]).open('xb'); f.write(sys.argv[2].encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" "$MATRIX_RECORD" "$MATRIX_SHA256"
PREPARED=$("$PYTHON" -X utf8 -m aimusic.audio.review_archive prepare --input-root "$INPUT_ROOT" --matrix "$MATRIX" --expected-matrix-sha256 "$MATRIX_SHA256" --reviewer "$REVIEWER")
BUNDLE=$("$PYTHON" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])['bundle'])" "$PREPARED")
BUNDLE_SHA256=$("$PYTHON" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])['bundle_sha256'])" "$PREPARED")
ASSIGNMENT=$("$PYTHON" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])['assignment'])" "$PREPARED")
ASSIGNMENT_SHA256=$("$PYTHON" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])['assignment_sha256'])" "$PREPARED")
RESULT=$("$PYTHON" -X utf8 -m aimusic.audio.review_archive assemble --input-root "$INPUT_ROOT" --bundle "$BUNDLE" --assignment "$ASSIGNMENT" --expected-bundle-sha256 "$BUNDLE_SHA256" --expected-assignment-sha256 "$ASSIGNMENT_SHA256" --out "$ARCHIVE")
ARCHIVE_SHA256=$("$PYTHON" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])['archive_sha256'])" "$RESULT")
"$PYTHON" -X utf8 -m aimusic.audio.review_archive verify --archive "$ARCHIVE" --expected-sha256 "$ARCHIVE_SHA256"
"$PYTHON" -X utf8 -c "import os,sys; from pathlib import Path; f=Path(sys.argv[1]).open('xb'); f.write(sys.argv[2].encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" "$DIGEST_RECORD" "$ARCHIVE_SHA256"
printf '\nIndependent-channel SHA-256: %s\n' "$ARCHIVE_SHA256"
```

### PowerShell organizer: matrix → prepare → assemble

Use existing private local NTFS parents with appropriate ACLs. Every native exit
code is checked explicitly; `$ErrorActionPreference` alone is insufficient.
These blocks avoid version-specific PowerShell syntax, but that is not evidence
of execution on both PowerShell 5.1 and 7. Record the actual shell/version tested.

<!-- archive-powershell-generate -->
```powershell
$ErrorActionPreference = 'Stop'
$Python = 'C:\Review Tools\venv\Scripts\python.exe'
$Package = 'C:\Review Retained\run_hash'
$SoundFont = 'C:\Review Approved\instrument.sf2'
$InputRoot = 'C:\Review Private\inputs'
$Matrix = 'C:\Review Private\inputs\matrix.json'
$MatrixRecord = 'C:\Review Trust\matrix.sha256'
$Archive = 'C:\Review Delivery\review-r01.zip'
$DigestRecord = 'C:\Review Trust\review-r01.sha256'
$Reviewer = 'r01'
& $Python -X utf8 -m aimusic.audio.review_matrix --package $Package --soundfont $SoundFont --out $InputRoot
if ($LASTEXITCODE -ne 0) { throw 'Matrix generation failed; preserve residue and use a fresh private root.' }
$MatrixSHA256 = & $Python -X utf8 -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" $Matrix
if ($LASTEXITCODE -ne 0) { throw 'Matrix digest failed; stop.' }
& $Python -X utf8 -c "import os,sys; from pathlib import Path; f=Path(sys.argv[1]).open('xb'); f.write(sys.argv[2].encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" $MatrixRecord $MatrixSHA256
if ($LASTEXITCODE -ne 0) { throw 'Private matrix digest recording failed; stop.' }
$Prepared = & $Python -X utf8 -m aimusic.audio.review_archive prepare --input-root $InputRoot --matrix $Matrix --expected-matrix-sha256 $MatrixSHA256 --reviewer $Reviewer
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed; preserve residue and use a fresh private root.' }
$Preparation = $Prepared | ConvertFrom-Json
$Bundle = $Preparation.bundle
$BundleSHA256 = $Preparation.bundle_sha256
$Assignment = $Preparation.assignment
$AssignmentSHA256 = $Preparation.assignment_sha256
$Result = & $Python -X utf8 -m aimusic.audio.review_archive assemble --input-root $InputRoot --bundle $Bundle --assignment $Assignment --expected-bundle-sha256 $BundleSHA256 --expected-assignment-sha256 $AssignmentSHA256 --out $Archive
if ($LASTEXITCODE -ne 0) { throw 'Archive assembly failed; stop.' }
$ArchiveSHA256 = ($Result | ConvertFrom-Json).archive_sha256
& $Python -X utf8 -m aimusic.audio.review_archive verify --archive $Archive --expected-sha256 $ArchiveSHA256
if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed; stop.' }
& $Python -X utf8 -c "import os,sys; from pathlib import Path; f=Path(sys.argv[1]).open('xb'); f.write(sys.argv[2].encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" $DigestRecord $ArchiveSHA256
if ($LASTEXITCODE -ne 0) { throw 'Independent digest recording failed; stop.' }
Write-Output "Independent-channel SHA-256: $ArchiveSHA256"
```

Send only the sealed ZIP, convey the archive digest through an authenticated
independent channel, and use the reviewer extraction/serving sequence below.
Keep the matrix ledger, preparation result and blind map private. No successful
command constitutes human approval or fresh native-render evidence by itself.

## Sealed-archive commands (task 6.6)

This separate **archive-only offline lane** uses already prepared/admitted records.
Retain `bundle` and `assignment` paths/digests from successful preparation above
(or independently approved canonical API outputs), and every bundle reference
under one private input root. It does not render candidates or invent approval.
JSON must be canonical UTF-8/ASCII-escaped bytes, with no BOM or trailing newline.
The schema-only example in this document is not an assembly input. Candidate
rendering (`review_matrix`) is a separate prerequisite, not an archive operation.

Use Python 3.11+ with the product wheel already installed. No checkout, test
helpers, `[audio]` extra, renderer, SoundFont, credentials, or network is needed
for the four archive commands. Installation is a separate provisioning step;
when using a pre-provisioned wheelhouse it can also be offline:
`python -m pip install --no-index --find-links "/absolute/wheelhouse" "/absolute/wheelhouse/aimusic-0.1.0-py3-none-any.whl"`
(PowerShell: use quoted Windows paths). Normal product dependencies must already
be installed or present in that wheelhouse. Do not run `pip` during verification.

The CLI is `python -m aimusic.audio.review_archive` with explicit subcommands:

| Command | Required arguments | Effect |
|---|---|---|
| `assemble` | `--input-root`, `--bundle`, `--assignment`, `--expected-bundle-sha256`, `--expected-assignment-sha256`, `--out` | Verify inputs, create one new sealed ZIP. |
| `verify` | `--archive`, `--expected-sha256` | Read-only strict whole-archive verification. |
| `extract` | `--archive`, `--expected-sha256`, `--out` | Verify first, then create a new disposable view. |
| `verify-extracted` | `--archive`, `--expected-sha256`, `--directory` | Call `verify_extracted`; check current view immediately before serving. |

There are no inferred paths/digests, response files, abbreviated options, serving
or browser subcommands. Record reads reject links/reparse points and nonregular
files before parsing, cap the bundle at 32 MiB and assignment at 64 KiB, check the
trusted byte digest, then require strict canonical schema round trips. Archive
budgets and threat-model limits are in [publication safety](audio-publication-safety.md#sealed-reviewer-archives-task-65).

Each successful command writes exactly one canonical JSON object to stdout,
without a newline: `archive_sha256`, `operation`, and
`schema: "aimusic.review-archive-result/1"`. Operational failures exit 1, write a
canonical object with `error` instead of `archive_sha256` to stderr, and emit no
success stdout. Usage errors exit 2 with argparse diagnostics. Treat **any**
nonzero exit as failure; do not continue to extraction or serving. Output may
contain private diagnostic paths; keep organizer logs private.

### Custody and trust prerequisites

- Choose existing, access-controlled local parent directories. No symlink/reparse
  ancestors, network shares, shared hostile locations, or concurrent namespace
  **or payload mutation** during assembly, extraction, verification or serving.
  Windows publication requires writable fixed local NTFS. On macOS use physical
  paths (`/private/...`, not the `/var` or `/tmp` aliases). No path is auto-resolved
  to bypass link checks. Keep temporary storage private too.
- Keep the input root, delivery directory, digest ledger, and reviewer view
  separate. The archive output cannot be inside the input root. Output archive,
  digest-record file and extraction directory must be **new names**. Parent
  directories must already exist; these commands do not provision permissions.
- **Organizer:** retain the archive unchanged and record its successful assembly
  digest in a separate trusted ledger. Send that digest through an authenticated
  independent channel (for example an independently authenticated message or
  in-person confirmation), separate from the archive download. A hash is not a
  signature; transport of an untrusted hash alongside an untrusted ZIP adds no
  authenticity. Never put the trusted digest inside the ZIP.
- **Reviewer:** enter the digest confirmed through that independent channel, not
  one computed from the received ZIP, read from its manifest/`complete.json`, or
  copied from an unauthenticated neighboring checksum file. Retain the received
  archive and the independent trust record separately for later verification.
- Keep the **whole provenance graph**, `private/blind-map.json`, `order_seed`,
  source/render evidence and reviewer identity-to-pseudonym mapping in organizer
  custody. Never send them with the archive or put them below a served directory.
  Limit access/backups to authorized organizers; unblind only when authorized
  after findings are finalized. Retain the map to reproduce assignment order.
  Audible content and equal-media digests can still reveal similarities.
- A marker is not proof of integrity. An interrupted extraction or uncertain
  custody requires stopping the server, preserving the old view for inspection,
  verifying the retained sealed archive, and extracting under a **fresh name** in
  trusted storage. Never resume, repair, overwrite, or automatically delete a
  partial view or publication residue. Re-run `verify-extracted` before every
  serving session; a pass is neither a snapshot nor a serving-time lock.

Replace every `/absolute/...` or `C:\Review ...` example path and uppercase digest
placeholder below with your explicit local value. Do not use the example tokens
as digests; the CLI rejects them. The Python executable must be from the installed
wheel environment. Keep each shell's variables in the same terminal session.

### POSIX organizer: assemble, verify and record independently

<!-- archive-posix-assemble -->
```sh
set -eu
umask 077
PYTHON='/absolute/venv/bin/python'
INPUT_ROOT='/absolute/private-inputs'
BUNDLE='/absolute/private-inputs/bundle.json'
ASSIGNMENT='/absolute/private-inputs/reviewer.json'
BUNDLE_SHA256='REPLACE_WITH_APPROVED_BUNDLE_SHA256'
ASSIGNMENT_SHA256='REPLACE_WITH_APPROVED_ASSIGNMENT_SHA256'
ARCHIVE='/absolute/delivery/review-r01.zip'
DIGEST_RECORD='/absolute/trusted-ledger/review-r01.sha256'
RESULT=$("$PYTHON" -m aimusic.audio.review_archive assemble --input-root "$INPUT_ROOT" --bundle "$BUNDLE" --assignment "$ASSIGNMENT" --expected-bundle-sha256 "$BUNDLE_SHA256" --expected-assignment-sha256 "$ASSIGNMENT_SHA256" --out "$ARCHIVE")
ARCHIVE_SHA256=$("$PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])['archive_sha256'])" "$RESULT")
"$PYTHON" -m aimusic.audio.review_archive verify --archive "$ARCHIVE" --expected-sha256 "$ARCHIVE_SHA256"
"$PYTHON" -c "import os,sys; from pathlib import Path; from aimusic.audio.review_bundle import FileRef; d=sys.argv[2]; FileRef('digest',d,0); f=Path(sys.argv[1]).open('xb'); f.write(d.encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" "$DIGEST_RECORD" "$ARCHIVE_SHA256"
printf '\nIndependent-channel SHA-256: %s\n' "$ARCHIVE_SHA256"
```

The ledger file is ASCII: 64 lowercase hex characters plus LF, created
exclusively and flushed. Do not store it in the reviewer view or repackage the
ZIP to include it. Transfer the ZIP through the chosen delivery channel; convey
and authenticate the displayed digest separately. No command here transfers data.

### POSIX reviewer: independent trust, strict extraction, then local serving

<!-- archive-posix-extract -->
```sh
set -eu
umask 077
PYTHON='/absolute/venv/bin/python'
ARCHIVE='/absolute/received/review-r01.zip'
VIEW='/absolute/reviewer-views/review-r01-new'
DIGEST_RECORD='/absolute/trusted-ledger/received-r01.sha256'
TRUSTED_SHA256='REPLACE_WITH_INDEPENDENTLY_CONFIRMED_ARCHIVE_SHA256'
"$PYTHON" -c "import os,sys; from pathlib import Path; from aimusic.audio.review_bundle import FileRef; d=sys.argv[2]; FileRef('digest',d,0); f=Path(sys.argv[1]).open('xb'); f.write(d.encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" "$DIGEST_RECORD" "$TRUSTED_SHA256"
"$PYTHON" -m aimusic.audio.review_archive verify --archive "$ARCHIVE" --expected-sha256 "$TRUSTED_SHA256"
"$PYTHON" -m aimusic.audio.review_archive extract --archive "$ARCHIVE" --expected-sha256 "$TRUSTED_SHA256" --out "$VIEW"
```

Keep custody; in that same terminal verify the **current** view, immediately
followed by the separate standard-library HTTP server. Serve only `VIEW`, never
its parent, the input root, digest ledger or archive directory:

<!-- archive-posix-serve -->
```sh
"$PYTHON" -m aimusic.audio.review_archive verify-extracted --archive "$ARCHIVE" --expected-sha256 "$TRUSTED_SHA256" --directory "$VIEW" &&
"$PYTHON" -m http.server 8000 --bind 127.0.0.1 --directory "$VIEW"
```

After the server reports that it is listening, open a **second terminal**:

<!-- archive-posix-open -->
```sh
PYTHON='/absolute/venv/bin/python'
"$PYTHON" -m webbrowser -t 'http://127.0.0.1:8000/index.html'
```

### PowerShell organizer: assemble, verify and record independently

Native exit codes are checked explicitly (record the shell/version actually tested): `$ErrorActionPreference` alone does not stop failed Python commands.
Use existing private NTFS parents with appropriate ACLs; POSIX `umask` is not a
Windows access-control mechanism.

<!-- archive-powershell-assemble -->
```powershell
$ErrorActionPreference = 'Stop'
$Python = 'C:\Review Tools\venv\Scripts\python.exe'
$InputRoot = 'C:\Review Private\inputs'
$Bundle = 'C:\Review Private\inputs\bundle.json'
$Assignment = 'C:\Review Private\inputs\reviewer.json'
$BundleSHA256 = 'REPLACE_WITH_APPROVED_BUNDLE_SHA256'
$AssignmentSHA256 = 'REPLACE_WITH_APPROVED_ASSIGNMENT_SHA256'
$Archive = 'C:\Review Delivery\review-r01.zip'
$DigestRecord = 'C:\Review Trust\review-r01.sha256'
$Result = & $Python -m aimusic.audio.review_archive assemble --input-root $InputRoot --bundle $Bundle --assignment $Assignment --expected-bundle-sha256 $BundleSHA256 --expected-assignment-sha256 $AssignmentSHA256 --out $Archive
if ($LASTEXITCODE -ne 0) { throw 'Archive assembly failed; stop.' }
$ArchiveSHA256 = ($Result | ConvertFrom-Json).archive_sha256
& $Python -m aimusic.audio.review_archive verify --archive $Archive --expected-sha256 $ArchiveSHA256
if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed; stop.' }
& $Python -c "import os,sys; from pathlib import Path; from aimusic.audio.review_bundle import FileRef; d=sys.argv[2]; FileRef('digest',d,0); f=Path(sys.argv[1]).open('xb'); f.write(d.encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" $DigestRecord $ArchiveSHA256
if ($LASTEXITCODE -ne 0) { throw 'Independent digest recording failed; stop.' }
Write-Output "Independent-channel SHA-256: $ArchiveSHA256"
```

Send only the ZIP through the delivery channel; independently convey and
authenticate the displayed digest. Keep the ledger and private inputs separate.

### PowerShell reviewer: independent trust, strict extraction, then local serving

<!-- archive-powershell-extract -->
```powershell
$ErrorActionPreference = 'Stop'
$Python = 'C:\Review Tools\venv\Scripts\python.exe'
$Archive = 'C:\Review Received\review-r01.zip'
$View = 'C:\Review Views\review-r01-new'
$DigestRecord = 'C:\Review Trust\received-r01.sha256'
$TrustedSHA256 = 'REPLACE_WITH_INDEPENDENTLY_CONFIRMED_ARCHIVE_SHA256'
& $Python -c "import os,sys; from pathlib import Path; from aimusic.audio.review_bundle import FileRef; d=sys.argv[2]; FileRef('digest',d,0); f=Path(sys.argv[1]).open('xb'); f.write(d.encode('ascii')+bytes([10])); f.flush(); os.fsync(f.fileno()); f.close()" $DigestRecord $TrustedSHA256
if ($LASTEXITCODE -ne 0) { throw 'Independent digest recording failed; stop.' }
& $Python -m aimusic.audio.review_archive verify --archive $Archive --expected-sha256 $TrustedSHA256
if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed; stop.' }
& $Python -m aimusic.audio.review_archive extract --archive $Archive --expected-sha256 $TrustedSHA256 --out $View
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed; preserve it and choose a fresh name.' }
```

In the same terminal, under continuous custody, verify immediately before the
separate standard-library server:

<!-- archive-powershell-serve -->
```powershell
& $Python -m aimusic.audio.review_archive verify-extracted --archive $Archive --expected-sha256 $TrustedSHA256 --directory $View
if ($LASTEXITCODE -ne 0) { throw 'View verification failed; do not serve.' }
& $Python -m http.server 8000 --bind 127.0.0.1 --directory $View
if ($LASTEXITCODE -ne 0) { throw 'Local server failed.' }
```

After the server reports that it is listening, open a **second PowerShell window**:

<!-- archive-powershell-open -->
```powershell
Start-Process 'http://127.0.0.1:8000/index.html'
```

### Reviewer session and exposure limit

The HTTP server/browser are **separate, explicitly local** operations, not part of
assembly, verification or extraction. Bind only to `127.0.0.1`; no internet
service, credentials or remote assets are needed. Use the exact loopback URL
above, not a public interface or `file://` as a verification shortcut. Stop the
server with Ctrl-C at the end. If the port is occupied, stop and investigate;
do not assume an existing server is serving your verified view.

Schema 1 has **exactly 14 exposures per reviewer, each assigned once**: four
canonical mixes, four role stems, and three A/B pairs (parity, timing, tuning).
Follow the assigned page and A/B order. Do not add alternate files, another
assignment or calibration examples; expanding the set requires a new schema
revision. Clips are individually at most 120 seconds; 14 exposures is a workload
cap, not a claim of safe listening duration. Start at low volume, pause/take breaks
or stop if uncomfortable. Controls never autoplay. If playback fails, try the
opaque WAV download; if that fails, mark it **not reviewed**, stop and request a
verified replacement or fresh extraction. Never substitute another clip.

Pages do not save judgments. Findings/authorized reporting are later tasks; a
successful command, completion marker or playback is not a human finding or
acceptance. This is a rough-mix workflow, not mastering, polyphonic-f0 proof, or
new CLAP/FAD/LUFS evidence. Historical four-demo approval does not transfer.

## Candidate selection and admission

`aimusic.audio.review_admission.read_candidate(root, request)` consumes an explicit
`CandidateRequest`: relative output/source paths, source kind, independently
approved source identity, exact render-manifest **and cache-manifest** digests,
expected backend and expected asset digest. `cache_manifest_sha256` is required
and keyword-only: it anchors output bytes independently of render configuration.
Do not learn expected values from an untrusted output and
then call that verification. Package identity uses the full native manifest source
mapping (`midi_sha256`, `source_hash`, `package_hash`, `structure_sha256`,
`tuning_sha256`); imported identity uses `midi_sha256` and `input_kind`.

The reader makes no network/asset-discovery call, allocates no synth, and writes no
media. It reads bounded source bytes into private temporary files for the existing
package/import validators. Before returning an admitted candidate it:

1. Pins both manifests to independently approved digests; verifies the complete
   cache manifest/key/fingerprint/output set, every output digest, and the
   duplicate `stems/render.json` against the top-level manifest. Before output
   payload reads it rejects case-fold aliases, missing/extra entries, more than
   64 declared outputs or more than 128 MiB in the stems directory. Repeated
   reads and final rechecks consume a 512 MiB reader work budget.
2. Recomputes source identity and import analysis or RenderPackage reconciliation
   using the existing validators. Analysis, render-source declarations and stored
   reconciliation must agree exactly. Imports are explicitly `not_applicable` for
   planner reconciliation, never evidence of planner agreement.
3. Replays the declared groove through the existing pure event transform and
   compares expressive MIDI events and its diff report. It compares every isolated
   stem event and source-track/role/channel/note-count association. It checks role
   index membership, selected tuning capability, requested/effective backend,
   absence of fallback, asset digest, native/backend versions/code and per-stem
   settings/sidecars. Channel-pitch event evidence is reconstructed independently.
4. Reads allowed metadata-free WAV containers. All stems, rough mix and click must
   be finite, aligned in sample rate/frame horizon, at or below the configured peak
   and caller ceiling (never above 0.95), and above the expected RMS floor (default
   `1e-5`). A stem's expected signal window spans its first onset to last release;
   this is a gross non-silence gate, not a per-note audibility scorer. It also checks
   the exact float rough sum of source stems, with attenuation and click excluded.
5. Applies controlled-tuning admission when explicit original intent is supplied,
   then rechecks all referenced bytes. Missing/failed evidence raises an error;
   no partially admitted candidate or public page is returned.

`read_native_candidate(root, report_ref, workflow=..., run=...,
expected_backend=..., expected_asset_sha256=...)` adapts the existing
`aimusic.native-listening-evidence/1` four-workflow report. The independently pinned
report must identify every retained source/run artifact by digest, including
both render and cache manifests. The adapter checks those bytes, derives the pinned
request, runs the same admission reader, and retains the report as another private
reference. It does not inherit the
report's `checks_passed` or `human_listening` value. The `reference-19` workflow
requires its retained original intent. New harness reports emit POSIX relative
paths on every platform. Only inside this pinned legacy-report adapter, retained
Windows relative paths such as `stems\cache_manifest.json` are converted to `/`.
Drive-relative/absolute, UNC/root, mixed separators, dot/traversal and empty
components are rejected **before** normalization; the usual portable-path checks
then apply. Normalized and casefold aliases are rejected as a collection. Report
bytes/digests are unchanged, and all returned FileRefs remain POSIX-only. This
compatibility rule does not relax the general bundle/candidate path contract.
A native runtime/asset report is evidence
about that run only, not a cross-platform WAV hash requirement.

The asset-free `simple` backend may be explicitly selected for tests/previews.
That admission proves the recorded gates only; it does not promote the simple
renderer to FluidSynth instrument/controller fidelity. Renderer expectations
cannot silently fall back to simple. Report/scorer values are not inferred from
successful file creation, playback or a human's listening answers.

## Four-state metrics and controlled tuning

All required metrics are explicit: `CLAP`, `FAD`, `LUFS`, `rhythm_f_score`, `note_f1`,
`polyphonic_f0`, the common gates above, reconciliation, controlled tuning and the
negative-control gate. A missing/duplicate record is invalid.

| State | Meaning |
|---|---|
| `passed` | A named machine method ran, with digest-linked evidence. |
| `failed` | A named machine method ran and failed; evidence is retained. |
| `unmeasured` | Not implemented/run here; no number, method or fake zero. |
| `not_applicable` | Explicit reason why that metric/gate does not apply. |

Measured records require a method and evidence references. Human/listening methods
cannot supply machine results. The admission reader always marks the unfinished
six scorer metrics `unmeasured`; controlled monophonic f0 cannot set polyphonic
f0, note F1, rhythm F-score, CLAP, FAD or LUFS to passed. Applicable common gates
must pass. A failed controlled-tuning result is allowed **only** for a privately
identified calibration control whose separate negative-control gate passes.

Controlled admission uses every original declared steady mono window, unchanged
19-EDO frequency targets, and the original sample rate. It validates the original
MIDI note set/timing/pitch against intent, rejects polyphony or out-of-note windows,
and refuses dropped, changed, silent, truncated, low-confidence or unmeasurable
windows. Frequency estimation remains target-independent.

**Floor interpretation:** existing `TuningCheckConfig` is retained unchanged:
per-tone absolute error **<=5 cents** (stricter than 15), aggregate mean and P95
**<=15 cents**, inclusive. Configuration may only tighten these floors. There is
one frequency observation per declared steady window; the per-tone P95 is that
single absolute error, not a claim about frame-by-frame tracking over a whole
performance. Coverage must be complete; a good subset cannot hide a bad tone.

For the negative member, original MIDI is retained separately and the flattened
input must equal it with only pitch wheels zeroed. Targets are never replaced by
nearest 12-TET values. The control must be fully measurable, fail acceptance, and
**exceed the aggregate P95 floor**; silence or merely missing evidence does not
count as successful calibration. Existing `tuning-check.json`, when present, must
match recomputed measurements. The control is synthetic calibration, not a failed
production demo and not proof that a reviewer will hear the difference.

## Claims and historical non-transfer

These artifacts are **rough mixes**, not mastered or production-quality audio.
Peak/alignment/event checks do not measure perceptual quality. The fixed release
tail does not guarantee full decay; dry synthesis does not establish mix/mastering
quality. Controlled mono evidence does not certify arbitrary instrument notes,
polyphonic f0, aesthetic groove improvements, or general human discrimination.

Any change to source, rendered/presented bytes, config, backend, runtime, asset or
transform changes a node and therefore bundle identity. Earlier listening approval
is historical only, even if the music sounds similar. New bundles start with no
human findings; neither an agent nor a gate supplies a review. Disagreements,
incomplete responses and counts must remain separate in the later reporting layer.

## Validator-checked example and tests

[audio-review-bundle-example.json](audio-review-bundle-example.json) is a **schema-only
synthetic fixture**, not evidence of an actual render, valid media or a human
review. Its small placeholder file digests intentionally do not constitute a
distributable bundle. The test validates it using the exact public schema parser:

```python
from pathlib import Path
from aimusic.audio.review_bundle import ReviewBundle

bundle = ReviewBundle.from_bytes(Path("docs/audio-review-bundle-example.json").read_bytes())
assert bundle.to_bytes() == ReviewBundle.from_bytes(bundle.to_bytes()).to_bytes()
# For a real retained bundle: bundle.verify(root, expected_sha256=approved_digest)
```

Focused verification (activated environment, no native asset required):

```text
python -m pytest tests/test_audio_review_bundle.py tests/test_audio_review_admission.py tests/test_audio_review_presentation.py tests/test_audio_review_blinding.py tests/test_audio_tuning_check.py tests/test_audio_review_archive.py tests/test_audio_review_archive_security.py tests/test_audio_review_archive_cli.py tests/test_audio_review_prepare.py tests/test_audio_review_archive_docs.py -q
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
```

Tests include malformed schemas, deep immutability, canonical round trips,
metadata/file tampering, wrong hashes/backends/assets, private-map containment,
public refusal, package reconciliation, meaningful groove replay, altered routing
and sidecar settings, explicit unmeasured states, and positive/negative/missing/
wrong-target tuning. The synthetic native adapter test is **simulation only**; it
is not fresh native Windows FluidSynth or human-listening evidence.

The CLI tests cover required arguments/digests, canonical JSON output, malformed,
oversized, link/reparse and nonregular records, changed archives/views, and
create-only refusal. The runbook test builds and installs a wheel in an external
Unicode/space path, with no product or audio dependencies, then executes the
archive-only commands with network/asset/renderer access forbidden. A separate
installed-wheel lane uses provisioned audio dependencies and a synthetic renderer
to execute matrix → prepare → assemble; no real FluidSynth is required. Child
Python explicitly enables UTF-8 after inherited encoding overrides are removed.
The Windows runner supplies the wheel, docs and fixture helpers outside the
checkout and selects these tests without broader skips. It separately smoke-tests
verify-then-loopback serving. POSIX blocks run on POSIX; PowerShell blocks run
when that shell is available, otherwise only their exact Python argument lists
and inline recording scripts are executed. The latter is not native PowerShell
or Windows/NTFS evidence. Browser launch commands are checked, not automatically
opened; no new browser-playback or human-listening evidence is claimed here.
