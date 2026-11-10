# Self-contained listening review and human outcome

The new [review-bundle graph and objective admission layer](audio-review-bundles.md)
is separate from this historical four-demo acceptance. It does not transfer the
findings below to new source/render/presentation identities.

## Human observation — 2026-11-02

The user confirmed **all four embedded players worked**, then selected **“Accept
rough demos”**. The recognizable-role / gross-timing gate was accepted, with **no
specific audible defect reported**. This is a human rough-demo acceptance, not an
independently measured defect-free certificate or an agent listening claim.

The accepted page was the manually assembled `listening-review-embedded.html` in
the retained external evidence directory `fluidsynth-evidence-g_19b5em`. It used
`data:audio/wav;base64,` because relative local WAV URLs did not play in Review.
Its SHA-256 is
`1e4f2240626bc4ca99666ec8735f3d26e46eb0efca53203ffe48dbf70c23557e`.
The four mixes and their source/runtime/asset and measured fidelity evidence are
identified in [the native evidence](audio-native-listening-evidence.json) and
[the four-workflow report](audio-testing.md#four-native-listening-workflows-103--104--112).

**Limits remain:** rough mixing, not mastered audio; no CLAP/FAD, full rhythm
F-score, notes F1 or LUFS/mastering measurements; polyphonic f0 **unmeasured**.
Native chorus/reverb are disabled (dry effects), with a fixed **0.25 s** release
tail, not guaranteed full envelope decay. Controlled mono f0 results do not certify
all instrument notes. No specific stem-by-stem finding or subjective groove
improvement was reported. The four mixes use **no-op groove**; the timing review
below concerns the separate original versus shifted pair.

### Human timing review — 2026-11-02

The timing files were presented with the clear names **`timing_original.wav`**
and **`timing_shifted_12_5ms.wav`**. The user reported that they **could not perceive
a difference**. When subsequently asked about clipping, stuck/lost notes,
truncation or broken timing, the user explicitly answered **“No defects heard”**.
This completes the remaining human timing review for task **11.3**, alongside
the earlier acceptance of all four rough demos.

A uniform **12.5 ms** shift may be imperceptible in this comparison: shifting all
events equally does not change their relative timing. The response is a human
listening observation, not proof of identical samples or a general perceptual
threshold. **No aesthetic groove improvement is claimed.** The demos remain
rough and dry, not mastered, with no CLAP/FAD scores and polyphonic f0 unmeasured.
“No defects heard” is not an objective defect-free guarantee.

## Reproduce the portable page (no native dependency needed)

`aimusic.audio.listening_review` uses only the Python standard library and never
renders, downloads or contacts a service. Supply already verified output files;
page generation itself does **not** certify native provenance or human acceptance.

From an activated environment with the package installed, set `EVIDENCE` to the
absolute external directory printed by the native harness. Use a **new** filename
(the accepted manual page must not be overwritten):

```bash
python -m aimusic.audio.listening_review \
  --evidence-root "$EVIDENCE" \
  --out "$EVIDENCE/listening-review-reproduced.html"
```

Alternatively, use all four explicit directory flags for the
[individual CLI workflows](audio-demo-sources.md#listening-commands-approved-native-prerequisites-required):

```bash
python -m aimusic.audio.listening_review \
  --fixture-12 "$DEMO/fixture-listening" \
  --generated-12 "$DEMO/generated-listening" \
  --reference-19 "$DEMO/reference-listening" \
  --multitrack-19 "$DEMO/multitrack-listening" \
  --out "$DEMO/listening-review.html"
```

Each explicit input contains `stems/rough_mix.wav`; evidence-root mode maps to
`four-workflows/<workflow>/cold/`. The modes cannot be mixed. Open the HTML in
Review or an offline browser. There are exactly four labelled native audio
controls, descriptions, WAV digests and an unchecked listening checklist. No
JavaScript, external media, paths, manifest contents, SoundFont or cache are
embedded. The checklist is not persisted or treated as consent. An optional
`--title` is HTML-escaped and limited to 200 characters; do not put private data
in display text. The page never copies prior human acceptance onto a new render.

The total raw WAV cap is **32 MiB**, enforced before and during bounded reads.
Inputs must be readable regular non-symlink files in non-symlink workflow/stems
directories. Container checks admit nonempty mono/stereo PCM/float RIFF WAVs with
16-byte `fmt ` and only `fmt ` / `fact` / `PEAK` / `data` chunks. Unknown ancillary
metadata is **rejected**, not stripped or copied, preserving byte identity without
leaking WAV text metadata. This is intentionally not a general audio-file converter.

Output parent must already exist and not itself be a symlink. Existing files,
directories, live/dangling symlinks and creation races fail without overwrite;
there is no force option. All inputs validate before exclusive creation. Write
errors/process crashes may leave a partial **new** file: inspect it and choose a
fresh filename. Use caller-owned directories; hostile concurrent replacement of
ancestor directories is outside this local-tool contract.

Identical WAVs and title produce identical HTML bytes irrespective of mapping
order, destination name, wall clock or input directory location. The regenerated
accepted set is **7,044,236 bytes**, SHA-256
`61554dbae92ce96cd63485c4657804ac5b51695a4bb0b5d23c428307e0aee454`.
Its four data URLs decode **byte-identically** to the accepted manual page and
original rough mixes. The HTML presentation differs; the audio does not.

The native harness now generates `listening-review-embedded.html` at its fresh
external evidence root only after all four workflow and MTS-refusal assertions
pass. It parses the HTML, decodes all four WAVs and checks exact equality, then
records the page's digest in `four-workflows/evidence.json`. New native runs still
report `human_listening: pending`: automation cannot transfer the historical
human decision to different audio. Generated HTML/WAVs remain **untracked**.

## Task decisions

| Task | Decision / evidence |
|---|---|
| **7.5** | Complete: separate Python/native/asset Windows/Linux setup and approved license record in [FluidSynth docs](audio-fluidsynth.md), asset-free checks, retained/presented 12-EDO stems and accepted mixes. Evidence is sufficient to **propose MUS-34 completion**, not mutate its ticket. No claim of a human audition of every isolated stem. |
| **9.4** | Complete: controlled samples/reports presented and rough set accepted; measured mono-only limits remain explicit. **MUS-35 remains partial**: Scala/MTS-helper breadth and pre-endpoint diffusion gating are outstanding. |
| **11.3** | Complete: four rough demos accepted, then the separately named original / +12.5 ms timing pair reviewed on 2026-11-02. User could not perceive a difference and explicitly reported “No defects heard” when asked about clipping, stuck/lost notes, truncation or broken timing. No aesthetic groove improvement claimed. |
| **11.4** | Complete: final reviewer reported no Critical/Required findings and authorized closure; [cross-lane evidence and per-capability acceptance](audio-testing.md#final-acceptance-review-114) complete **42/42** tasks. |

MUS-34 → MUS-35 → MUS-36 → MUS-37 → MUS-38 is unchanged. No Linear update,
push, upload, PR, archive, mastered-quality claim or expansion into later stages.

## Verification — 2026-11-02

Commands in the same Python 3.12.12 environment (native opt-in, asset and Homebrew
variables unset for asset-free checks):

```bash
python -m pytest tests/test_audio_listening_review.py tests/test_audio_wheel.py tests/test_architecture.py tests/test_render_package.py -q
python -m pytest tests/test_audio_*.py tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
openspec validate --all --strict --no-interactive
```

- Report unit/CLI tests: **12 passed**, 6 subtests. Coverage includes four decoded
  byte-identical data URLs, fixed order, deterministic bytes, escaped text, PCM and
  native FLOAT WAV, bad/missing/unreadable/nonregular/metadata-bearing input,
  cumulative/exact byte-cap boundary, sparse oversized input, collisions, live and
  dangling links, output-parent link, and a racing output creator.
- Installed-wheel smoke passes offline under **`-I -S`**, no checkout/PYTHONPATH
  or audio/native dependencies available to the report command; all four decoded
  byte comparisons pass. Existing wheel/profile/preview checks remain intact.
- Asset-free audio/architecture/contract: **365 passed**, **715 subtests**, **6
  skipped** (five native opt-in tests and optional bridge). Full unittest:
  **703 run**, **14 optional skips**, no failures, **78.676 s**.
- Flake8 clean; mypy clean (**140 files**, unchanged untyped-body informational
  note); strict OpenSpec and `git diff --check` passed. At that implementation
  checkpoint OpenSpec was **40/42**, with 11.3 and 11.4 unchecked. The later human
  timing review completed **11.3**, reaching **41/42**. The subsequent
  [approved final review](audio-testing.md#final-acceptance-review-114) completes **11.4**: **42/42**.
- Separate approved GeneralUser GS 2.0.3 / FluidSynth 2.6.2 / pyfluidsynth 1.3.4
  native lane: **5 passed, no skips, 10.24 s**. Fresh external evidence is
  `fluidsynth-evidence-sev5rca1`; its generated page is byte-identical to the page
  generated from the accepted evidence (digest above). No audio thresholds changed.
- TDD first failed on the missing report module. An inverted four-workflow gate
  was killed by the tests. Restoring same-size source within one timestamp second
  left stale mutation bytecode; instruction inspection confirmed cached `==`
  versus source `!=`. Only that module's generated bytecode was removed before the
  final green asset-free/full checks. No assertions or quality floors were relaxed.
- Browser preview was disconnected in this implementation subagent session, so
  **no new browser playback observation** is claimed. The human observation above
  belongs to the accepted manual page; the new page has parsed/decoded-byte and
  native-harness verification. Parent-session visual/playback verification remains
  available using the regenerated external page.

Logs are external `listening-review-{red,mutation,asset-free-final,full,native}.log`
in the approved scratch parent; no generated HTML/WAV or machine-specific path is
part of this commit.

### Timing-review documentation check

That timing-review follow-up changed only documentation and task status. Strict OpenSpec
validation, local Markdown link-target checks, checkbox-diff checks (only 11.3
changed) and `git diff --check` pass. No runtime/audio tests were rerun for this
timing-review docs-only update; the implementation results above remain historical evidence.
Final-review evidence and documentation checks are recorded in [audio-testing.md](audio-testing.md#final-acceptance-review-114).
