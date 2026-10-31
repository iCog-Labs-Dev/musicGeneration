# REQ-10 implementation record

Date: 31 October 2026. Base revision: `741a874c45cd36f4d01d327bf3711e78407409f0`. This report describes local working-tree changes. No implementation commit, remote CI run, or pull request was created.

An earlier follow-up review closed an integration-test coverage gap without production changes. A later presentation audit identified a required renderer-result feature, now implemented; see the renderer-result section at the end. See the review update at the end of this record for its scope, evidence, and security limits. Counts in the original verification table describe the initial handover; the review update records the later checkout. Paths to `REQ-10-planning` refer to the author's local evidence archive outside the repository, not files required by a fresh clone.

## What changed, in plain English

The system already generated a sequence of musical states, turned those states into notes, and wrote a MIDI file. However, several steps directly called one particular implementation. Trying a different method meant changing the code that coordinates the process.

The coordinating code now accepts replacements as arguments [dependency injection]. A replacement can calculate musical distance [tonal metric], propose possible next states [candidate generator], turn a complete state sequence into notes [whole-score decoder], or write those notes to a file [renderer]. Leaving an argument out uses the original method.

Each replacement has a declared input and output shape [protocol]. It also has behavior it must follow [contract]. For example, a distance must be finite and nonnegative, and a candidate generator must return allowed next states for the requested source and time. Python type annotations describe types. Mypy checks those descriptions before execution; annotations alone do not check every value while the program runs.

Settings such as tempo and note density remain ordinary configuration values. Executable replacement objects are separate arguments. The implementation does not put Python functions into saved settings.

The original bridge solver and music-generation algorithms remain in use. This change makes the selected parts replaceable and checks that the old implementations fit the new interfaces. It does not claim that an experimental method produces better music.

## Requirements and evidence

| Requirement | Implementation and proof |
|---|---|
| Small typed interfaces for the four components | Four component interfaces, with separate legacy/reporting renderer signatures in core and the candidate interface in planning. Direct assignments check the actual defaults and independent doubles. |
| Types match pre-existing implementations | Candidate, decoder, and renderer protocols preserve the existing names, parameter kinds, defaults, and result types. The renderer still takes `EDO` and returns `None`. |
| Replace components through orchestration | Graph and planner accept metric/candidate replacements. CLI generation, the UI helper, and demo accept all four. Export accepts a renderer. Tests verify actual graph and output changes. |
| Keep contracts independent of backend libraries | Shared definitions use project types and the standard library. A fresh-interpreter import test verifies that loading contracts does not import NumPy, mido, audio, candidates, priors, decoder, or renderer implementations. |
| Keep existing behavior | Unchanged baseline, full regression suite, cache tests, old/new class identity checks, and real default 12-EDO/19-EDO output comparisons. |
| Test replacements without running every component | Controlled candidates, fixed metrics, a known-score decoder, and a recording renderer isolate behavior. The artifact renderer double also writes valid MIDI for integration tests. |
| Explain errors, diagnostics, setup, and configuration | Contract details below and a checked [extension guide](extending-components.md). |

## Before and after

Previously, `run_method_a` called the graph, which directly called `get_valid_next_states`. Tonal calculations inside proposals, pruning, scoring, and reporting selected the built-in formula independently. CLI and UI workflows directly called the built-in decoder and renderer.

Now `run_method_a` passes the selected metric and candidate generator to the graph. The graph selects `MetricBoundCandidates(tonal_metric)` when no generator was supplied. That adapter calls the original proposal function with the selected metric. The graph uses a supplied candidate callable directly for nonfinal steps. It retains ownership of the requested final endpoints.

CLI, UI, and demo workflows select their decoder and renderer once per invocation and call the selected functions. CLI generation retains score JSON, MIDI, manifest, and RenderPackage output. The UI artifact helper retains flat score/MIDI/manifest output; the demo writes MIDI and returns its result.

The selected metric reaches these paths:

1. Key and chord neighbor proposals in default candidate generation.
2. Endpoint-distance estimates used in edge retention and layer pruning.
3. Raw GTTM features and their weighted contributions, through surrounding prior-scoring functions.
4. Window-based rescoring of retained graph edges and the resulting diagnostics.
5. Harmonic neighbor and cadence features.
6. Realized tension curves in CLI and UI diagnostics.

The existing prior interface, `Prior.logp_next`, does not change. Neither does the existing per-track `TrackGenerator` interface. Whole-score decoding and per-track note generation have different responsibilities.

## Final interfaces

The exact source signatures are in `aimusic/core/protocols.py` and `aimusic/planning/protocols.py`.

| Interface | Input and output |
|---|---|
| `TonalMetric` | `tonal_distance(a_root, b_root, edo=12) -> float` and `basic_space_distance(a_root, a_quality, b_root, b_quality, edo=12) -> float`. Roots and EDO are integers; qualities are strings. |
| `CandidateGenerator` | Existing source `BeatState`, time, `RNGKey`, `d_max`, optional style/vocabulary/prior/context/EDO/budget settings and prior-guided flag. Returns `tuple[CandidateGenerationResult, RNGKey]`. |
| `ScoreDecoder` | `Sequence[BeatState]`, with keyword-only random key, vocabulary, decode configuration, EDO, timing, tempo, and terminal-state flag. Returns `tuple[Score, RNGKey]`. |
| `ScoreRenderer` | `Sequence[SymbolicNote] | Score`, project `EDO`, string destination, optional ticks per beat, and keyword-only instrument mapping. Writes the file and returns `None`. |
| `ReportingScoreRenderer` | The same inputs; writes the file and returns `RenderResult(output_path, format, size_bytes, edo)`. Workflow arguments accept either signature through `Renderer`. |

The candidate and decoder return the next random key, preserving explicit random-state handling. The legacy renderer remains unchanged. The added reporting adapter returns a frozen `RenderResult` as required by the presentation scope. Score ticks default to 480 and decoder tempo defaults to 120.0 BPM, as before. Decoder terminal-state inclusion still defaults to false.

The shared frozen dataclasses moved into `aimusic/core/component_types.py`. Imports at their old modules refer to the same class objects. Their fields, validation, and properties remain the same. This preserves normal imports and `isinstance` checks. Newly serialized Python pickle class paths would name the new owner; pickle compatibility was not a project requirement or a verified artifact format. JSON Score/MIDI/manifest formats remain covered.

## Runtime contracts

Custom distances pass through `root_distance` or `chord_distance`. These reject booleans and non-real values with `TypeError`, and negative or non-finite values with `ValueError`. The original default calculations keep their shared caches. Custom implementations bypass those caches, so unhashable replacement objects work and one experiment does not reuse another experiment's values.

Root-neighbor ties use ascending root order. Cadence checks retain the existing comparison with distance `1`. A different distance scale can therefore change cadence decisions. Fixed intervals and chord templates remain existing musical rules; they are not new plugin operations.

At the graph boundary, custom candidate results must identify the requested source and time, contain a tuple of deduplicated legal states, use known vocabulary identifiers, provide nonnegative integer proposal/unique counts, align optional scores, and return an `RNGKey`. Rejection records must have the requested source and time. The graph does not reconstruct a plugin's proposal work from those declared counts.

An empty valid result remains empty. A plugin exception propagates. Invalid supplied results raise clear `TypeError` or `ValueError` exceptions instead of being silently filtered or replaced. Final transitions still use the requested endpoint layer and existing legality rules. The existing final-target helper's zero proposal counters remain unchanged. There is no new blanket rule that accepted states must be fewer than the proposal count.

The caller creates component objects and owns expensive setup or cleanup. Stateless defaults require no lifecycle hooks. Renderers own their file handles; the application owns destinations and packaging. No process-wide active plugin or mutable registry was added. Python annotations do not validate arbitrary decoder or renderer behavior at runtime. The reporting adapter checks file existence and metadata consistency; protocol checks and behavior tests cover the remaining promises.

CLI packaging and UI preview/download require a valid MIDI file at the supplied path. Supporting arbitrary file formats in those workflows is outside this issue. Renderer exceptions stop subsequent success-manifest creation and CLI packaging, although a score or partial output file may already exist. There is no automatic fallback or transaction rollback.

Existing candidate rejection reasons/counts, structural score breakdowns, tension reports, manifests, and RenderPackage values remain the diagnostic formats. Runtime component identities are not serialized into manifests. A custom experiment must record its component version and constructor settings separately for reproduction.

## File changes

### Production files

| File | Change and reason |
|---|---|
| `aimusic/core/component_types.py` | Added the neutral owner for candidate result/rejection and symbolic note/instrument data. Avoids importing an implementation just to name its data. |
| `aimusic/core/protocols.py` | Added metric, whole-score decoder, and renderer interfaces. Implementation-only types use type-checking imports. |
| `aimusic/planning/protocols.py` | Added the candidate callable interface, preserving its existing prior/context types without executing those imports. |
| `aimusic/theory/tonal.py` | Added `ExistingTonalMetric`, checked custom-distance dispatch, and selected-metric neighbor ordering. Existing functions remain callable. |
| `aimusic/planning/candidates.py` | Reexports the moved candidate data; routes neighbor proposals through the metric; adds `MetricBoundCandidates`. |
| `aimusic/planning/graph.py` | Accepts replacements, validates custom candidate results, and routes the metric through pruning, edge priority, scoring, and rescoring. |
| `aimusic/planning/plans.py` | Passes metric/candidate choices through Method A; lets the demo select its decoder and renderer. |
| `aimusic/scoring/gttm_features.py` | Adds a keyword-only metric after the seven existing feature arguments. Uses it consistently in harmonic features and score aggregation, preserving default caches. |
| `aimusic/scoring/priors.py` | Passes the metric through structural scoring and diagnostic breakdown helpers. Leaves the prior protocol unchanged. |
| `aimusic/scoring/tension.py` | Uses the selected metric for motion and realized tension reports. |
| `aimusic/render/adapters.py` | Adds `RenderResultAdapter`, preserving legacy calls while returning verified artifact details. |
| `aimusic/render/__init__.py` | Exports the reporting adapter. |
| `aimusic/render/midi_render.py` | Reexports moved note/instrument data. Rendering algorithm and signature remain unchanged. |
| `aimusic/app/cli.py` | Accepts supplied components in generation, a renderer in export, and the selected metric in tension diagnostics. |
| `ui.py` | Adds the same choices to the artifact helper and uses them. User-facing controls retain the existing defaults. |

### Tests and documentation

| File | Purpose |
|---|---|
| `tests/test_render_result.py` | Verifies real outputs and metadata in both tunings, both accepted input shapes, legacy byte preservation, demo exposure, missing outputs, mismatched reports and failures. |
| `tests/test_component_protocols.py` | Verifies shared class identity and independent contract imports. |
| `tests/test_metric_injection.py` | Verifies custom feature values, candidate binding, default adapter equivalence in both tunings, separate A/B/A metric runs, invalid distances, cadence scale, and tension reporting. |
| `tests/test_planning_injection.py` | Verifies controlled successors, final endpoint restriction, empty results, invalid results, raw/weighted metric diagnostics, metric-driven edge/layer pruning, random keys, and deterministic default replay. |
| `tests/test_artifact_injection.py` | Verifies CLI 12/19 output substitution and packaging, error propagation, export, and UI-helper flat output. The combined test selects all four replacements together. |
| `tests/typecheck_protocols.py` | Explicitly assigns every real default and independent double to the intended protocol. No `Any` or cast stands in for conformance. |
| `tests/typing_fixtures/incompatible.py.txt` | Holds five intentionally incompatible assignments outside normal Python test discovery. |
| `tests/test_protocol_typing.py` | Runs positive mypy checks and verifies five expected assignment errors, including rejection of a legacy None-returning renderer where a reporting renderer is required. |
| `docs/extending-components.md` | Explains selection points, contracts, and a complete metric/decoder/renderer example. The exact example is executed and type-checked. |
| `docs/req-10-implementation.md` | This detailed implementation record. |

No dependencies or configuration files changed. No production files were removed. The separate `../REQ-10-planning` folder retains the design, baseline, work log, commands, and logs. It also contains rerunnable evidence tools: `run_check.py`, `inspect_baseline_artifacts.py`, `compare_artifacts.py`, and `check_extension_example.py`. Older slides and scripts remain untouched.

## Test-first work and verification

Focused red/green cycles are recorded in `../REQ-10-planning/evidence/implementation`:

| Behavior | Initial failure | Passing evidence |
|---|---|---|
| Shared contracts | `contracts-red`: protocol module absent | `contracts-green` |
| Metric in harmonic features | `metric-features-red`: unsupported metric keyword | `metric-features-green`, `metric-final` |
| Metric in candidate proposals | `metric-candidates-red`: adapter absent | `metric-candidates-green` |
| Candidate substitution | `planning-red`: unsupported candidate keyword | `planning-green-integrated` |
| Decoder/renderer substitution | `artifact-injection-red`: unsupported decoder keyword | `artifact-green` |
| Checked type conformance | `typing-tests-red`: negative fixture absent | `types-green` |

Additional checks were added after these units passed, including invalid distances, A/B/A cache isolation, actual pruning decisions, and cadence scale. These are compatibility/coverage checks, not claimed red/green cycles.

The baseline ran before source edits and verified hashes of all 143 tracked files. It passed 365 discovered tests, including 9 skips, and 10 architecture/package pytest tests. The first completed implementation suite passed 385 discovered tests with the same 9 skips. Two extra pruning/cadence checks were then added and their 13-test metric/planning group passed. A final complete run includes these additions; see the final verification record below.

From the repository root, use the isolated interpreter at `../.req10-work/venv/Scripts/python.exe`, or a Python 3.11 environment with the declared requirements and `.[dev]` installed:

```powershell
python -m compileall -q aimusic tests ui.py
python -m tests.import_all
python -m unittest discover -s tests -p 'test_*.py'
python -m pytest tests/test_architecture.py tests/test_render_package.py -q
python -m flake8 aimusic tests ui.py
python -m mypy aimusic tests ui.py
python -m unittest tests.test_protocol_typing -v
```

The author also executed and type-checked the exact extension-guide example with a task-local script. That script belongs to the local evidence archive. In a fresh clone, use the complete example in [the extension guide](extending-components.md).

Each recorded command uses the absolute isolated-interpreter path. JSON records under `../REQ-10-planning/evidence/final` contain exact commands, working directory, timestamps, duration, and exit status. Their matching `.log` files contain output. Broad mypy retains the repository's pre-existing ignored modules. The separate positive/negative fixtures are checked code and protect against treating that broad success alone as proof of compatibility.

## Real output comparison

Both the unchanged baseline and final implementation ran these commands with the same interpreter and dependencies, using separate output folders:

```powershell
python -m aimusic.app.cli generate --seed 11 --beats 4 --edo 12 --out <output-folder>
python -m aimusic.app.cli generate --seed 11 --beats 4 --edo 19 --out <output-folder>
```

`../REQ-10-planning/evidence/final/artifact-comparison.json` records these results:

- Score JSON and MIDI bytes match the baseline exactly in both tuning modes.
- Each run contains 28 notes across bass, comping, drums, and lead.
- Manifests match except for run ID, timestamp, and configured output directory. The comparison checks that each output directory is the actual expected folder before excluding it.
- The package loader validates each new RenderPackage. Packaged scores match the standalone Score values, packaged MIDI matches the standalone MIDI bytes, and tuning metadata identifies the requested EDO.
- Hashes are saved in the comparison record. Final artifacts remain in `evidence/final/artifacts-12` and `artifacts-19`; originals remain in the baseline folders.

The comparison tool initially exposed two assumptions in the verification script: manifests include the chosen output directory, and packaged JSON uses different whitespace. Those checks now compare only run-independent manifest content and parsed Score values inside packages. The standalone old/new JSON and MIDI comparison still requires exact bytes. No product output was changed to satisfy this check.

## Environment, review, and limits

Tests use task-local CPython 3.11.18 on Windows. Setup used uv 0.13.6, with no system package replacement. Relevant installed versions include NumPy 2.4.6, mido 1.3.3, mypy 2.4.0, SciPy 1.17.1, and Gradio 6.31.0. The full package list and dependency consistency check are preserved in baseline evidence. The recorder itself runs under Python 3.13.13, but launches all project checks under the isolated Python 3.11 interpreter. CI was not run.

Nine tests skip because optional dependencies are absent: eight JAX/ML tests and one external audio-bridge test. Those skips also occurred in the unchanged baseline. They are not counted as passing tests. The original automated implementation checks did not exercise optional model training, external audio rendering, paid services, or browser interaction. The UI artifact helper was exercised directly. Later browser/download and user playback confirmation are recorded in the combined-coverage section below.

Some UI imports hang inside the restricted environment while initializing local sockets. Full-suite/import checks therefore used authorized local execution. Existing Gradio imports also make their own telemetry/version requests; those are unrelated to component replacement. No source workaround was introduced for the local socket issue.

The separate automated review helper was rejected by automatic approval review because it might send repository code to an external Codex service. It was not bypassed. Local review covered the complete diff, callers, import direction, type fixtures, metric dispatch, final endpoints, error paths, and real artifact comparison. This was a local review, not an independent automated-review approval. No remaining actionable defect was identified by that local review.

Known scope limits are deliberate: components are selected through Python arguments; no CLI plugin registry or UI plugin selector was added. A custom candidate generator must manage its own metric if it needs one. Custom component provenance is not added to the existing manifest schema. Whole-repository strict typing, alternate packaged file formats, and optional audio/ML functionality remain separate work.

## Initial handover verification record

| Check | Result | Evidence under `../REQ-10-planning/evidence/final` |
|---|---|---|
| Complete unittest discovery | 387 discovered, 378 passed, 9 skipped; no failures/errors | `unit-tests-complete.json` and `.log` |
| Focused injection/type checks after final assertion review | 17 passed | `injection-checks.json` and `.log` |
| Architecture and RenderPackage | 10 pytest tests passed | `architecture-package.json` and `.log` |
| Syntax | Passed | `syntax.json` and `.log` |
| Executable-module imports | 50 imported | `imports.json` and `.log` |
| flake8 | Passed | `lint-complete.json` and `.log` |
| Mypy | Passed for 114 source files | `mypy-complete.json` and `.log` |
| Exact extension-guide example | Type checking and real generation passed | `extension-example.json` and `.log` |
| Real default output comparisons, both EDO modes | Exact Score JSON/MIDI preservation; valid packages and stable manifests | `compare-artifacts-validated.json`, `.log`, and `artifact-comparison.json` |

All listed commands returned exit code 0. The implementation adds 22 test methods across five test modules. The local change contains 13 production files, seven test/fixture files, and two documentation files. Ten pre-existing files are modified and twelve files are new. The separate planning/evidence folder is outside the repository change.

The final command records and decision history are maintained in `../REQ-10-planning/WORK-LOG.md`. `final-inventory.json` in final evidence records changed-file hashes and the unchanged base revision. The optional skips and rejected external review remain the limits listed above.

## Follow-up review with updated skills

The 29 October follow-up reviewed the original issue, plan, standards, full implementation diff, new files, connected callers, and evidence. Separate reviewers checked standards and requirements. A focused reviewer used the updated Cloudflare security-audit guidance to inspect source trust boundaries. The skill revisions and hashes are preserved outside the repository in `REQ-10-planning/skill-sources-2026-10-29/versions.json`.

One test gap was found and closed. Candidate substitution was already tested inside the graph, but forwarding through the main planner and application helpers lacked dedicated protection. Three new test methods and an expanded UI test now cover the planner, demo, CLI, and UI helper. The planner test requires distinctive diagnostics from the supplied generator's returned result. Failure tests require the supplied generator's exact exception before decoding or output creation.

A disposable-copy proof deliberately removed candidate forwarding at each entry point. All four variants failed for the intended missing replacement behavior. The correct code passed. No implementation function was patched in the kept tests, and repository production code was not mutated by the proof.

The review made no production-code changes. There are now 25 added test methods relative to the original repository. Initial evidence remains preserved; review evidence is in `REQ-10-planning/evidence/review-2026-10-29`, with updated file hashes and results in `review-inventory.json`.

The reviewed suite ran 390 tests: 381 passed and the same 9 optional tests skipped, with no failures or errors. The candidate/artifact group passed 14 tests; all four deliberately omitted forwarding arguments were detected in disposable copies. Syntax, lint, mypy for 114 files, and all 10 architecture/RenderPackage tests passed. Saved default outputs were compared again, and production file hashes still match the original verified handover.

The focused security review found no confirmed introduced vulnerability within the changed code and immediate consumers. It did not run exploit probes, a dependency advisory scan, deployed-service tests, or Cloudflare's full six-phase audit. The current runner has not demonstrated that skill's required isolation. The user approved ordinary local regression testing while security probing remained source-only. This is a bounded review result, not a security certification.

The local follow-up review record distinguishes standards, requirements, security scope, historical proof, new execution, and the coverage improvement. The outcomes and limitations above are included here so this repository record remains usable without the author's separate evidence archive.

## Existing documentation reconciliation

After the implementation and clean standalone autoreview, an additional documentation pass updated the repository's existing entry points. The initial handover had added new guides and updated the separate planning folder, but had missed these existing reference pages.

| Existing document | Update |
|---|---|
| `readme.md` | Lists the four protocols and default implementations, explains selection points, links the guides/ADR, adds shared modules to the architecture map, documents development type checks, and separates completed Method A/track decoding from future Method B. |
| `docs/gttm-transition-context.md` | Describes selected-metric routing, cache separation, and the unchanged cadence-distance convention. |
| `docs/tension-diagnostics-spec.md` | Marks the original implementation complete, labels historical problem/scope text, updates the callable signatures and actual manifest placement, and explains metric selection/provenance. |
| `docs/render-package-contract.md` | States the supplied renderer's valid-MIDI requirement and the application's continued packaging responsibility. |
| `docs/decisions/README.md` | Indexes the new component decision and corrects the relative link to root `DECISIONS.md`. |
| `docs/decisions/ADR-006-track-generator-pure-functions.md` | Cross-references the separate whole-score decoder contract while preserving the per-beat decision. |
| `epics_and_phases.md` | Labels its original baseline/checklists as historical and links the completed component-interface work. |

Added [ADR-007](decisions/ADR-007-pluggable-component-protocols.md) records the selected interfaces, shared data ownership, exact type compatibility, runtime injection, alternatives, and limits. The `beat_tension` docstring now describes its actual distance normalization, `distance / (1 + distance)`, instead of obsolete fixed divisors.

This pass changes prose and one docstring only. A source-AST comparison after removing docstrings verifies that executable tension logic is unchanged. Local Markdown links and file paths in the affected pages were checked. No behavior test or autoreview rerun was needed for these documentation edits. The earlier test and review results remain historical evidence for the implementation; the documentation-pass record is in `REQ-10-planning/evidence/documentation-2026-10-29`.

## Combined replacement coverage and manual acceptance

The earlier runs proved individual replacement behavior, successful decoder/renderer substitution through CLI/UI helpers, a metric/decoder/renderer demo, and candidate substitution through planning. They did not include a single successful application run selecting all four replacements together. That limitation was clarified and the missing combined case was added.

`test_all_four_replacements_complete_cli_and_ui_workflows` now covers four successful cases: CLI and UI helper, each in 12-EDO and 19-EDO. Every case selects a fixed metric, an alternative candidate strategy, a known-score decoder, and a recording renderer that writes real MIDI. The test runs the real graph construction and bridge solver, then checks:

- The alternative candidate strategy's third-head states occur on the decoder's selected path, and path transitions belong to its returned support.
- Metric calls use the selected EDO. Saved tension values match an independent calculation using the supplied distances.
- The decoder's distinctive one-note Score reaches the renderer and saved JSON. The generated MIDI is parsed and contains the expected note.
- CLI RenderPackages reload, match the standalone score and MIDI, and identify the selected EDO. The UI helper's flat score/MIDI/manifest files are checked separately; it does not emit a RenderPackage.

The combined case passed all four scenarios. The first fixture attempt filtered for third-head proposals, but the default proposal policy does not offer that head on weak beats, leaving no feasible path. The test double was corrected to propose legal third-head anchors directly and deduplicate them. No production implementation was changed to make the fixture pass.

There are now 26 added test methods relative to the original repository. Combined-run command records and subsequent full-suite results are archived in `REQ-10-planning/evidence/combined-2026-10-29`. An independent source recheck found no actionable issue in the combined test.

The latest full suite ran 391 tests: 382 passed, nine optional tests skipped, with no failures/errors. Mypy and flake8 passed. Disposable-copy checks independently omitted each selected component and the combined test caught all four omissions. Production logic was unchanged; the addition is test coverage and documentation.

The [manual acceptance guide](req-10-manual-checks.md) records browser generation/download as verified from the user's six downloaded files and settings screenshots. Both runs use seed 11, four beats, 105 BPM, and the correct EDO. Each has 28 notes across four tracks. MIDI timing, velocity, pitch data, and note closure match the saved scores; all downloads match the generated files exactly. The 19-EDO file contains nonzero pitch bends with a maximum measured pitch-representation error of about 0.0103 cents. This validates file data, not the playback behavior of a particular synthesizer.

The user subsequently confirmed playing both examples, completing the requested manual playback check with no issue reported. The playback application and its microtonal tuning behavior were not independently inspected. Automated component tests and this basic listening check do not establish musical quality or full-length performance. Optional audio/ML functionality and remote CI retain their earlier scope limits. The manual-download verification, archived files, and subsequent user playback confirmation are retained in the author's local `REQ-10-planning/evidence/manual-browser-2026-10-29` archive.


## Renderer result required by the presentation

The mentor required everything presented, including slide 7's proposal to return the saved file location and result details. Earlier documentation treated that feature as optional. That interpretation was incorrect for the expanded scope and is superseded here and in ADR-007.

The new frozen `RenderResult` contains `output_path: str`, `format: str`, `size_bytes: int`, and `edo: int`. `ReportingScoreRenderer` uses the existing renderer input signature and returns that value. The legacy `ScoreRenderer` and `render_midi` still return `None`; no old input or return type changed. Workflow arguments accept either callable through the `Renderer` union.

`RenderResultAdapter` calls the selected renderer once, checks the requested file exists, reads its actual size, and returns MIDI artifact details. A supplied report must equal those factual details or it raises `ValueError`. Renderer and file errors propagate without a success result. The adapter renders to a fresh temporary path in the destination directory, with the same filename. After checking the new file and report, it atomically replaces the final destination and reports that final path. It removes temporary output on failure and preserves the previous final MIDI file. This prevents a no-op replacement from reporting an old file as a new render. The adapter does not parse MIDI or sandbox components; packaging and behavior tests still check valid MIDI. There are no retries or rollback of score JSON and other workflow artifacts.

CLI generation/export, the bridge demo, and the UI helper all use the adapter. CLI generation uses the reported path for output handling. The demo and UI artifact results expose `.render_result`, with a default of `None` for existing constructors. CLI handler return types, JSON schemas, manifest fields, output filenames and MIDI algorithms remain unchanged.

Seven new test methods cover both tunings, both renderer input forms, byte-identical legacy output, nested reporting adapters, incorrect metadata, missing files, raised errors, stale existing files, preservation of old output after errors or invalid reports, temporary-file cleanup and real demo results. The existing combined CLI/UI test now supplies a reporting adapter for all four replacement cases and checks UI metadata against the saved file. Existing legacy-renderer tests remain. Positive type fixtures cover both renderer interfaces; negative fixtures reject five assignments, including the legacy renderer where a reporting renderer is required. The extension example now returns a report and verifies it against its actual output file.

TDD evidence includes a missing-adapter failure before implementation, a passing adapter test run, and a failing UI integration assertion before exposing `.render_result`. An initial restricted UI run stalled during local socket initialization and was stopped; the authorized local rerun produced the expected missing-field failure. The local archive `REQ-10-planning/evidence/render-result` records commands, logs and subsequent verification. Autoreview found the stale-destination issue. A regression test reproduced it. An initial filesystem-timestamp check also rejected a legitimate overwrite during testing, so the final adapter uses fresh staged output and atomic replacement. Final verification is recorded below.


### Final renderer-result verification

The final suite ran 398 tests: 389 passed and nine optional ML/audio tests skipped, with no failures or errors. All ten architecture/RenderPackage checks passed. Mypy passed across 116 source files, flake8 passed, syntax compilation passed, and the import smoke check loaded 51 executable modules. The exact updated extension example passed mypy and produced a real 19-EDO MIDI file with matching report details.

Fresh default CLI runs for seed 11 and four beats in both 12-EDO and 19-EDO still match the original baseline Score JSON and MIDI byte for byte. Manifests match except run ID, timestamp and output directory, and packages validate. The final autoreview, rerun after the stale-output fix, returned no actionable findings. Its command used the installed helper with `--mode local --engine codex --model gpt-5.6-sol --thinking high --no-web-search` and the standalone Codex CLI.

The local `REQ-10-planning/evidence/render-result-final` archive contains the final command/exit records, logs, structured review, output comparisons and completion snapshot. A source-only AST comparison also checks that legacy candidate, decoder and renderer signatures still match the original revision and that the four moved data classes are unchanged. Local documentation links and diff whitespace are checked before submission. There are 33 added test methods relative to the original repository, including seven for this result feature.

No further manual test is required for this follow-up: the new result and failure behavior are automated, and default music files remain byte-identical. Prior browser downloads and user playback confirmation remain the manual acceptance evidence. Optional audio/ML execution, full-length performance and remote CI remain unverified. The work is ready locally for the existing one-PR submission plan; this follow-up did not create a commit, branch, push or PR. Original slides/scripts remain as fallbacks and still need presentation-status wording updates if used to describe completed work.
