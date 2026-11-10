# Provenance-linked listening findings (schema 1)

This is collection/reporting tooling, **not new human listening evidence**.
Never fabricate findings, fill in answers for an absent reviewer, or reuse historical
acceptance for different hashes. Only tests contain synthetic responses.

## Reviewer instructions

1. Obtain your pseudonymous assignment and its exact document SHA-256 through an
   independent trusted channel. Follow the [archive verification/extraction
   runbook](audio-review-bundles.md) before serving the reviewer view. Never use
   the private graph or condition map during blinded listening.
2. Start at a **low safe listening volume**. Stop if uncomfortable. Pause one clip
   before starting another. Keep **fixed volume** across both members of each pair;
   do not normalize, replace, edit or process the files. Take breaks between
   panels and whenever fatigue affects attention. Resume only when comfortable.
3. Follow assignment order: **8 singles** (canonical then roles), **3 pairs**
   (parity, timing, tuning), 14 exposures. A/B labels are local to each pair and
   reviewer. Do not guess conditions or equate preference with lack of defects.
4. On **playback failure**, try the linked WAV in a local player at low volume.
   If it still fails, stop the panel and request a verified fresh extraction.
   Leave required fields unanswered; optionally explain in notes. A missing or
   unplayable file is not silence, an audio defect, or a completed judgment.
5. Record only what you actually heard. Disagreements and apparently inconsistent
   subjective combinations are retained, not corrected by the tool.

### Response fields

The whole-assignment template retains exact media `FileRef`s (ID, path, SHA-256,
size). Do not edit links, pseudonym, schema or row order.

- Single: `audibility` = `audible`, `partly_audible`, `inaudible`, `unsure`, or
  `null`; `defects`; `confidence` = `low`, `medium`, `high`, or `null`; `notes`.
- Pair: `perceived_difference` = `heard`, `not_heard`, `unsure`, or `null`;
  `defects_a`, `defects_b`; `preference` = `A`, `B`, `no_preference`, `unsure`,
  or `null`; `confidence`; `notes`.
- Defects: `null` means unanswered; `[]` means explicitly no defects heard.
  Otherwise use an array of unique codes. Never infer an answer from notes.
- Notes are optional (`null`), at most 1024 characters, nonblank when present,
  with no control characters. `other` requires explanatory notes.

| Defect | Meaning |
|---|---|
| `click_pop` | Audible click or pop |
| `distortion` | Unwanted audible distortion |
| `dropout` | An interruption in the sound |
| `missing_voice` | A voice appears absent |
| `missing_notes` | Notes appear missing |
| `stuck_note` | A note persists unexpectedly |
| `timing_issue` | Perceived timing defect, not merely a difference |
| `pitch_issue` | Perceived pitch defect, not an objective f0 measurement |
| `cut_off_tail` | An ending/release seems cut off |
| `other` | Another defect, explained in notes |

### Completion and withdrawal

A fresh template validates as `incomplete`. `complete` requires every response
field in all 11 rows answered; `unsure` and empty defect arrays are answers.
Notes remain optional except for `other`. All incomplete snapshots are excluded
from judgment totals, even when some or all response fields are populated.

Never edit a published digest-named file. Copy it to an external working draft,
make your own responses, and set `previous_finding_sha256` to that exact previous
file's digest. Start with one incomplete root. Permitted chains are:

- incomplete → incomplete (zero or more times) → complete → withdrawn
- incomplete → withdrawn

Complete → complete corrections and changes after withdrawal are prohibited.
To withdraw, append a whole-assignment snapshot with `state: "withdrawn"` and a
nonblank `withdrawal_reason` (same 1024-character bound). Preserve prior snapshots.
Withdrawal excludes **all** of that assignment's judgments, including earlier
completed responses. It does not erase retained data; contact the organizer about
privacy/retention separately. Two roots, forks, missing predecessors, mixed
assignments or ambiguous duplicates fail closed. Identical replay counts once.
Up to 128 ordinary snapshots (including the incomplete root and any completion)
are allowed, plus one reserved final withdrawal. A 129th ordinary snapshot is
rejected; reaching the ordinary limit does not prevent withdrawal.

## Privacy, custody and identity

Pseudonyms allow 1–64 ASCII letters, digits, `_` and `-`. They are not anonymity:
reuse creates linkability across assignments; musical choices and free text may
identify someone. Do not include names, emails or identifying details in notes.
Keep any identity-to-pseudonym mapping separately under organizer control.

`ReviewFinding.assignment_document_sha256` links the **ReviewAssignment document**,
not its `AssignmentReference` wrapper. The older `ExternalFindingReference` still
links the wrapper digest; this workflow deliberately does not use or redefine it.

Maintain trusted-local **single-writer custody**, with no concurrent namespace or
payload mutation during verification, collection or reporting. Digests require
independent trust and do not authenticate people. Create-only publication is
**not tamper-proof** storage or a filesystem lock. Preserve trusted digests and
backups separately; missing chain records reject rather than being reconstructed.
Inspect residue manually. Never automatically delete, repair or resume residue.

Findings, working drafts, closures, reports, archives and assets stay outside Git.
Keep them outside both the private input root and the sealed archive's extracted
view. Never add them to the archive or page inventory; `verify-extracted` requires
that inventory to stay exact. Reports reveal provenance and conditions: keep them
private until explicitly authorized for distribution.

## Offline CLI collection

Use Python 3.11+ with the installed `aimusic` wheel. These commands use only the
stdlib; no audio extra, renderer, SoundFont, network or asset discovery is needed.
They accept an explicit assignment plus media root, so the reviewer never needs
the private graph. Reverify the extracted archive before listening as described
in the archive runbook. Collection rechecks current media bytes independently.

Set these shell variables to explicit local values (PowerShell also uses `$Name`):
`Python` = installed-wheel interpreter; `Assignment` = extracted `assignment.json`;
`AssignmentSHA256` = independently trusted **document** digest; `View` = verified
extracted view; `History` = existing empty external directory for this assignment.
Later set `Finding` = your canonical submitted draft; `FindingSHA256` = its
reviewer-confirmed exact digest. For received findings convey that digest through
an independent trusted channel, not by trusting a digest supplied beside altered
bytes. Commands return canonical success JSON; operational failures produce
canonical error JSON on stderr, no success stdout, and exit 1 (usage errors: 2).

### 1. Publish an unanswered template (and retain its returned digest/path)

<!-- findings-posix-template -->
```sh
"$Python" -m aimusic.audio.review_findings template --assignment "$Assignment" --expected-assignment-sha256 "$AssignmentSHA256" --media-root "$View" --history "$History"
```

<!-- findings-powershell-template -->
```powershell
& $Python -m aimusic.audio.review_findings template --assignment $Assignment --expected-assignment-sha256 $AssignmentSHA256 --media-root $View --history $History
if ($LASTEXITCODE -ne 0) { throw 'Template failed' }
```

Work on a copy **outside** `History` and `View`, not on the published file. Enter
actual responses; set the predecessor digest and completion state as above.
All record fields are required, including explicit nulls. Submit exact canonical
JSON: sorted keys, compact separators, ASCII-escaped UTF-8, no BOM or trailing newline.
An editor's pretty-printed draft must be canonicalized before submission; the
validator never silently normalizes it. With `Draft` set to your working JSON and
`Finding` a new external filename, the following saves only structurally valid
answers, rejects duplicate keys and checks the 128 KiB canonical serialized size
**before opening the output file**. ASCII escaping can exceed that limit even when
all notes fit their character bounds and the UTF-8 draft fits. It prints the exact
digest (it does not invent
answers or attest to listening). Do not use shell redirection to rewrite records.

<!-- findings-posix-canonicalize -->
```sh
"$Python" -c 'import sys; from pathlib import Path; from aimusic.audio.review_bundle import load_json, read_relative; from aimusic.audio.review_findings import ReviewFinding, MAX_DOCUMENT_BYTES; p=Path(sys.argv[1]); f=ReviewFinding.from_dict(load_json(read_relative(p.parent,p.name,limit=MAX_DOCUMENT_BYTES))); data=f.to_bytes(); out=Path(sys.argv[2]); out.open("xb").write(data); print(f.sha256)' "$Draft" "$Finding"
```

<!-- findings-powershell-canonicalize -->
```powershell
& $Python -c 'import sys; from pathlib import Path; from aimusic.audio.review_bundle import load_json, read_relative; from aimusic.audio.review_findings import ReviewFinding, MAX_DOCUMENT_BYTES; p=Path(sys.argv[1]); f=ReviewFinding.from_dict(load_json(read_relative(p.parent,p.name,limit=MAX_DOCUMENT_BYTES))); data=f.to_bytes(); out=Path(sys.argv[2]); out.open("xb").write(data); print(f.sha256)' $Draft $Finding
if ($LASTEXITCODE -ne 0) { throw 'Canonical submission failed' }
```

This only creates an untrusted working submission. Hardened history publication
happens through `append`, after full verification.

### 2. Validate the snapshot, then append it

Validation reports structural `state`, not history eligibility or acceptance.
`append` additionally verifies the entire predecessor chain and refuses ambiguity.

<!-- findings-posix-validate -->
```sh
"$Python" -m aimusic.audio.review_findings validate --assignment "$Assignment" --expected-assignment-sha256 "$AssignmentSHA256" --media-root "$View" --finding "$Finding" --expected-finding-sha256 "$FindingSHA256"
```

<!-- findings-powershell-validate -->
```powershell
& $Python -m aimusic.audio.review_findings validate --assignment $Assignment --expected-assignment-sha256 $AssignmentSHA256 --media-root $View --finding $Finding --expected-finding-sha256 $FindingSHA256
if ($LASTEXITCODE -ne 0) { throw 'Finding validation failed' }
```

<!-- findings-posix-append -->
```sh
"$Python" -m aimusic.audio.review_findings append --assignment "$Assignment" --expected-assignment-sha256 "$AssignmentSHA256" --media-root "$View" --finding "$Finding" --expected-finding-sha256 "$FindingSHA256" --history "$History"
```

<!-- findings-powershell-append -->
```powershell
& $Python -m aimusic.audio.review_findings append --assignment $Assignment --expected-assignment-sha256 $AssignmentSHA256 --media-root $View --finding $Finding --expected-finding-sha256 $FindingSHA256 --history $History
if ($LASTEXITCODE -ne 0) { throw 'Finding append failed' }
```

Only `f<sha256>.json` files belong in a history directory. Each must be exact
canonical bytes with a matching name. Limits: 128 KiB per serialized finding,
128 ordinary snapshots plus one final withdrawal per assignment. Size checks also
apply to constructed Python records, before append and report inclusion. Unknown
entries, symlinks/reparse points and residue reject the history. Hard-linked files
are rejected when the local filesystem reports `st_nlink > 1` (including local
APFS/NTFS); this is not a universal alias-detection guarantee or protection against
concurrent namespace changes. Keep the single-writer custody requirement above.

## Organizer: close the intended roster and authorize unblinding

Retain each exact assignment as `<input-root>/<bundle.assignment_directory>/a<document-sha256>.json`.
Place every collected history under the external
`<findings-root>/<assignment-document-sha256>/` directory. Keep all snapshots, not
only final responses. Confirm the complete **intended** pseudonym roster from your
independent allocation record; do not omit unfinished or disagreeing reviewers.

Set `InputRoot`, `Bundle`, `BundleSHA256` (independently trusted), `FindingsRoot`,
and an existing external private `Reports` directory. `Reviewer` is the assigned
pseudonym. `FindingSHA256` must now identify that assignment's final snapshot.
Repeat `--entry ASSIGNMENT_DOCUMENT_SHA256 PSEUDONYM FINAL_FINDING_SHA256` for **every**
intended reviewer, in your chosen roster order (maximum 32).

<!-- findings-posix-close -->
```sh
"$Python" -m aimusic.audio.review_report close --bundle-sha256 "$BundleSHA256" --input-root "$InputRoot" --entry "$AssignmentSHA256" "$Reviewer" "$FindingSHA256" --out "$Reports"
```

<!-- findings-powershell-close -->
```powershell
& $Python -m aimusic.audio.review_report close --bundle-sha256 $BundleSHA256 --input-root $InputRoot --entry $AssignmentSHA256 $Reviewer $FindingSHA256 --out $Reports
if ($LASTEXITCODE -ne 0) { throw 'Closure publication failed' }
```

`close` freezes the declaration as `c<digest>.json`, **not a claim that histories
are valid or complete**. Inspect the entire roster/final digests against your
trusted records, then retain its digest independently. Set `Closure` to that file
and `ClosureSHA256` to the independently confirmed digest. A new closure cannot
silently change what a trusted earlier closure meant.

<!-- findings-posix-generate -->
```sh
"$Python" -m aimusic.audio.review_report generate --bundle "$Bundle" --expected-bundle-sha256 "$BundleSHA256" --closure "$Closure" --expected-closure-sha256 "$ClosureSHA256" --input-root "$InputRoot" --findings-root "$FindingsRoot" --out "$Reports" --authorize-unblinding
```

<!-- findings-powershell-generate -->
```powershell
& $Python -m aimusic.audio.review_report generate --bundle $Bundle --expected-bundle-sha256 $BundleSHA256 --closure $Closure --expected-closure-sha256 $ClosureSHA256 --input-root $InputRoot --findings-root $FindingsRoot --out $Reports --authorize-unblinding
if ($LASTEXITCODE -ne 0) { throw 'Authorized report failed' }
```

Authorization is operator acknowledgement/private-map custody, **not authentication**.
The tool verifies the exact bundle, its referenced private map (no substitute map
option), assignment documents, current media/provenance, full histories and every
closure final digest. Missing or incomplete intended reviewers block reporting.
Withdrawn reviewers are retained and excluded. Reports publish create-only as
`r<digest>.json`; an existing report is never overwritten. No `order_seed` is
serialized. Inputs are bounded by archive provenance limits; aggregate finding
input is capped at 8 MiB and report output at 32 MiB. The multi-history reader
shares the byte/file budgets across the whole roster (at most 32 × 129 files),
checks file sizes before reading, and charges actual bytes before JSON parsing.
It stops at the first excess file without reading that file or any later history;
these limits are not merely checked after loading all histories.

Reports preserve the original assignment, individual history and disagreements;
category counts are not a vote for consensus or approval. Only final complete
snapshots contribute eligible reviewer/single/pair counts. Pair preference totals
refer to **stimulus IDs**, not aggregate A/B labels with different orientations.
Pair-side defect counts likewise refer to each exact stimulus ID, regardless of
that reviewer's A/B orientation. Defect counts count each selected category, not
mutually exclusive outcomes.
Every machine `Metric` (including `unmeasured`) and source/render/presentation
link is preserved separately. The flattened tuning control is disclosed as a
calibration control. Fixed limits remain: rough-demo only, no historical acceptance,
no mastering, no production-quality claim, and no inferred polyphonic-f0 accuracy.
