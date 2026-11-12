# REQ-10 implementation record

Original implementation: 31 October 2026. Mentor-review revision: 11 November 2026. Pull request: [#51](https://github.com/iCog-Labs-Dev/musicGeneration/pull/51).

The original submission used base `741a874c45cd36f4d01d327bf3711e78407409f0`. This revision incorporates upstream `44902dc0fc01abec62b9a566391dd4c6bf42b12d`, including section-wise planning and its UI default. Default-output comparisons use that unmodified upstream commit. The earlier report, slides, scripts, design, raw review comments and command logs remain in the author's separate local archive, `../REQ-10-planning`. A fresh clone does not need that archive to run the code or tests.

## What changed

The program already proposed musical states, selected a path through them, turned that path into notes and wrote MIDI. Several steps directly selected a built-in method. Trying another method therefore required editing the code coordinating the pipeline.

Four steps can now be replaced: calculating musical distance [tonal metric], proposing possible next states [candidate generator], turning the path into a complete score [decoder], and writing that score to a file [renderer]. Protocols describe the required calls and return types. Contracts also specify valid values and behavior. Runtime checks validate values at component boundaries; mypy checks declared type compatibility before execution. Python annotations alone do not enforce these promises.

An application factory selects components once per run. CLI generation/export, UI generation and the artifact demo share it. Trusted local JSON can select installed constructors and settings. Callers can also supply constructed Python objects. Omitted roles use existing methods. Configuring a factory and supplying an object for the same role is an error.

This revision replaces the original PR's two renderer protocols with the mentor's single neutral interface. MIDI tuning/instruments belong to the MIDI implementation. Direct `render_midi` keeps its established inputs and `None` return; `MidiRenderer` adapts it. The bridge solver and musical algorithms remain in use. Component replacement does not establish that an experiment produces better music.

## Mentor feedback coverage

All thirteen inline comments and the general instruction to update tests are covered below. Repeated renderer comments share one implementation.

| Comment ID | Implementation | Verification |
|---|---|---|
| 4500992731, 4500993466, 4500993803 | One `ScoreRenderer(Score, RenderRequest) -> RenderResult`, without MIDI/EDO arguments | Positive/negative typing; text renderer; actual MIDI adapter |
| 4501001064 | Validate all five named shared classes and the new request | Invalid-value tables and valid legacy values |
| 4501007994 | Generic artifact adapter; backend-specific MIDI construction | Text output, metadata mismatch, missing/stale output, nonregular file and cleanup tests |
| 4501023237 | `--components` selection and manifest identities/settings | Real configured CLI subprocesses and UI generation, all four roles in 12/19 EDO |
| 4501042527, 4501405404 | Shared `app.components.create_components` factory | Same config in CLI/UI; defaults, overrides, constructor errors and call shapes |
| 4501075236 | Artifact composition/default construction moves to `app`; planners receive selected metric/candidates | Demo tests; old demo import delegates to the application entry point |
| 4501083602 | Consistent counters, unique outcomes, finite scores and rejection diagnostics | Shared-value and graph-boundary tests |
| 4501089622 | Reject disagreement between graph `d_max` and supplied `SBConfig.d_max` | Mismatch fails before candidate generation |
| 4501394871 | Every requested malformed-candidate/RNG case | Separate cases for counters, duplicates, NaN/infinity, RNG and rejection source/time |
| 4501400614 | Match errors to each marked invalid assignment's exact file/line | Four incompatible role assignments independently rejected |

Comments are accessible at `https://github.com/iCog-Labs-Dev/musicGeneration/pull/51#discussion_r<comment-ID>`. Their exact text and action mapping are preserved in the local archive. The [ADR](decisions/ADR-007-pluggable-component-protocols.md) records the resulting design.

## Runtime flow and ownership

1. CLI/UI read run settings and optional trusted component JSON. The UI accepts the JSON path at process startup, not through browser input.
2. `create_components` constructs the selected metric, then injects it into the chosen candidate constructor. It constructs the decoder and injects effective MIDI settings into the renderer constructor.
3. The factory checks structural membership and callable argument shapes, records provenance and wraps the renderer with the generic artifact adapter. These checks do not execute the component or prove annotations/results.
4. `run_method_a` forwards the selected metric/candidates to the graph. Section-wise planning forwards the same objects to every section while retaining upstream endpoint/stitching behavior.
5. The selected decoder returns a `Score`. The selected renderer receives the score and a request containing destination and format.
6. The adapter stages a fresh sibling file, requires a regular file and checks reported path/format/size against the actual artifact. It atomically replaces the destination after validation. Failure cleans staging and preserves an old destination.
7. CLI generation writes the existing RenderPackage. UI generation offers flat score/MIDI/manifest downloads. These flows still request MIDI because packaging and preview consume MIDI.

The metric also reaches neighbor proposals, endpoint-distance estimates, edge retention/layer pruning, GTTM features, prior diagnostics, edge rescoring, cadence features and realized tension. Existing prior `logp_next` and per-track `TrackGenerator` interfaces stay unchanged. Whole-score decoding and per-track note generation have different responsibilities.

Pure planners do not construct decoder/renderer implementations. The old `planning.plans.render_exact_bridge_demo` import remains a compatibility delegate to `app.demos`; artifact orchestration now belongs to the application layer.

## Interfaces, configuration and provenance

Exact signatures live in `core/protocols.py` and `planning/protocols.py`.

| Role | Required call/result |
|---|---|
| `TonalMetric` | `tonal_distance(a_root, b_root, edo=12)` and `basic_space_distance(a_root, a_quality, b_root, b_quality, edo=12)`, returning finite nonnegative distances |
| `CandidateGenerator` | Existing source/time/RNG and candidate settings; returns `tuple[CandidateGenerationResult, RNGKey]` |
| `ScoreDecoder` | Existing state sequence and keyword-only decode/RNG/vocabulary/timing settings; returns `tuple[Score, RNGKey]` |
| `ScoreRenderer` | `(score: Score, request: RenderRequest) -> RenderResult` |

`RenderRequest` contains `output_path` and `format`. `RenderResult` contains factual `output_path`, `format` and `size_bytes`. Neither carries EDO. `MidiRenderSettings` binds backend EDO configuration and instruments; `MidiRenderer` calls the unchanged legacy renderer.

Each optional JSON role has `factory: "module:export"`, an `options` object and optional `public_options` names. Candidate constructors receive reserved keyword `tonal_metric`; renderer constructors receive `midi_settings`. Options cannot overwrite these dependencies. Unknown roles/fields, malformed references, non-JSON/non-finite options, invalid public names, constructor mistakes and incompatible callable shapes fail during setup. Constructor options and provenance have separate deep copies so mutation does not change later runs or the saved selection.

`manifest.config.components` records each role's selection source, factory, actual implementation and explicitly public option values. All constructor options are private by default; `public_options` opts shareable settings into downloadable output. Unpublished options contribute only a count, so credentials and signed URLs are not copied automatically. A public option must never contain secrets, including nested secrets. Renderer provenance also records supplied construction settings and wrapper. Runtime objects have unknown options recorded as `null`. Built-in MIDI settings must match the current run even through artifact wrappers, preventing misleading package tuning/program metadata. Opaque custom providers must honor their supplied settings. Identities are module/qualified names, not code/version hashes; external experiments must still pin code and environment.

The [extension guide](extending-components.md) provides runnable constructors and CLI/UI commands.

## Value and diagnostic contracts

- Request/result paths and formats are nonblank strings. Size is an integer of at least zero; booleans are rejected. File existence/type/size are checked when rendering.
- Rejections require a nonnegative integer time, `BeatState` source/candidate and nonblank reason.
- Candidate results require tuples of distinct accepted/rejected states with no overlap. Rejections match their result's source/time. Counters are nonnegative integers, `proposed_count >= unique_count`, and `unique_count == len(states) + len(rejections)`. Scores are empty or aligned with accepted states and finite real values.
- The graph revalidates supplied results, then checks requested source/time, vocabulary, transition legality and returned `RNGKey`. A valid empty result stays empty. Invalid output raises without hidden fallback.
- Final transitions still target requested endpoints. Their internal outcome counters are now truthful. Terminal endpoint enumeration stays excluded from raw proposal totals, preserving existing diagnostic meaning.
- Graph `d_max` is positive and agrees with a supplied SB configuration. Without one, the graph builds an SB configuration using that value. Proposal budget remains separate.
- Symbolic notes permit signed integer pitches and require finite nonnegative start, later finite end and velocity/timbre/pressure in 0..127. Instrument configuration checks optional program in 0..127 and an actual boolean drum flag.
- Custom distances reject boolean/non-real, negative and non-finite values. Built-ins retain caches; custom metrics bypass them so experiments do not share cached answers. Existing cadence comparisons retain their scale-dependent meaning.

Shared classes live in `core.component_types`. Existing candidate/MIDI imports reexport the same objects, preserving ordinary imports and `isinstance`. Pickle class-path compatibility is outside the verified JSON/MIDI formats.

## Files and responsibilities

| Files | Responsibility |
|---|---|
| `core/component_types.py`, `core/protocols.py`, `planning/protocols.py` | Validated shared values and typed interfaces without concrete backend imports |
| `theory/tonal.py`, `planning/candidates.py` | Existing-method adapters, checked metric dispatch and neighbor proposals |
| `planning/graph.py`, `planning/plans.py` | Injection, graph checks, consistent retention settings and both planning modes |
| `scoring/gttm_features.py`, `scoring/priors.py`, `scoring/tension.py` | Selected metric in musical features and diagnostics |
| `app/components.py` | Trusted config loader, per-run construction, call-shape checks and provenance |
| `app/cli.py`, `app/demos.py`, root `ui.py` | Selected-component artifact workflows |
| `render/adapters.py`, `render/midi_adapter.py` | General verified publication and concrete MIDI adaptation |
| `render/midi_render.py`, `render/__init__.py` | Existing MIDI entry point/reexports and adapter export |
| `tests/test_component_protocols.py`, `typecheck_protocols.py`, `typing_fixtures/incompatible.py.txt`, `test_protocol_typing.py` | Import independence, class identity, default types and per-assignment negative checks |
| `tests/test_metric_injection.py`, `test_planning_injection.py` | Isolated behavior, pruning, invalid values/RNG, replay and section forwarding |
| `tests/test_component_validation.py`, `test_component_factory.py` | Shared-value invariants, config errors, snapshots, provenance and call shapes |
| `tests/test_renderer_contract.py`, `test_midi_component.py`, `test_render_result.py` | Neutral format, legacy bytes, result verification and failure cleanup |
| `tests/test_artifact_injection.py`, `test_component_selection_workflows.py`, `component_providers.py` | Object/JSON-based four-role CLI/UI tests with artifact inspection |
| README, extension guide, render-package contract, ADR-007, this record and manual-check guide | Current usage, design and verification limits |

Production paths in this table are under `aimusic/` unless stated otherwise. No dependency was added. Generated artifacts, environments, temporary examples, evidence, planning notes, slides and PDFs stay outside the PR. The negative `.py.txt` typing fixture is intentional test source and belongs in the PR.

## Verification

Small red/green units covered neutral rendering, MIDI adaptation, factory selection, CLI/UI composition, value validation and graph configuration. Initial failures and passing results remain in `../REQ-10-planning/evidence/mentor-revision`. Independent review found constructor-option mutation, incompatible callable acceptance and nonregular staged output. Regression checks reproduced reachable cases before fixing those boundaries. An initial section test timed out because Mock instrumentation was expensive; lightweight counters proved section forwarding in both tunings without changing production behavior.

Structured review then found private constructor options entering artifacts, a built-in MIDI renderer disagreeing with run metadata, and one obsolete README sentence. Regression tests reproduced both code defects before the fixes. Explicitly public provenance and MIDI-setting agreement now address them. A separate thermonuclear maintainability review found graph checks duplicated after the shared validator. Removing sixteen lines leaves one owner for intrinsic candidate rules; the graph still checks current-call context and legality. All12 planning-injection tests passed after that cleanup, followed by static/import/architecture checks. The detailed local quality review stays outside the source PR.

A follow-up review reproduced mutation of the instrument dictionary inside frozen MIDI settings. Its copied mapping is now read-only, so constructors cannot mutate both sides of the agreement check or change programs after metadata capture. Separate tests check caller-alias isolation, direct mutation refusal and configured-constructor mutation refusal. The final full-suite run starts after both this fix and the thermonuclear cleanup.

The last compatibility check also reproduced a false mismatch between absent and empty instrument mappings. Settings now normalize both to the same no-override value. All16 focused MIDI/factory/configured-workflow tests pass after that normalization, including both agreement directions and byte-equivalent MIDI in both tunings. The README now states the public-option publication rule explicitly. The full-suite run began before this final normalization; its result is supplemented by those post-change tests and fresh artifact/UI/type checks.

Run from the repository root with Python 3.11, declared dependencies/development tools and Gradio for UI tests:

```bash
python -m compileall -q aimusic tests ui.py
python -m tests.import_all
python -m unittest discover -s tests -p 'test_*.py'
python -m pytest tests/test_architecture.py tests/test_render_package.py -q
python -m flake8 aimusic tests ui.py
python -m mypy aimusic tests ui.py
python -m unittest tests.test_protocol_typing -v
```

The exact complete extension-guide example is separately type-checked/executed. Broad mypy retains existing ignored modules; explicit positive/negative assignments provide narrower component conformance evidence. Project checks use isolated CPython 3.11; command recording uses CPython 3.13. Records include exact commands, working directory, duration, exit status and logs.

Final measured local results on 12 November 2026:

| Check | Result |
|---|---|
| Complete unittest discovery | 451 run, 441 passed, 10 skipped; no failures/errors |
| Post-normalization MIDI/factory/configured workflows | All 16 test methods pass; includes four real CLI/UI generation cases |
| Post-thermonuclear planning boundary and section forwarding | All 12 test methods pass |
| Architecture and RenderPackage pytest checks | All 10 pass |
| Static typing | No issues in 129 source files; all four deliberately incompatible assignments independently rejected |
| Lint, syntax and import smoke | Pass; 56 executable modules import |
| Default real 12/19-EDO generation | Score/MIDI byte-identical to upstream 44902dc; manifests, public provenance and packages validate |
| Actual configured Gradio server | Startup config, real HTTP callbacks and three downloads pass in both tunings; private constructor values absent |
| Exact extension-guide example | Passes mypy and writes real 19-EDO MIDI |
| Structured autoreview | Post-reset review of code commit `1bf1a3e` completed cleanly with zero actionable findings. |

The command archive is `../REQ-10-planning/evidence/mentor-revision-closeout`; final normalization checks/artifacts are under `mentor-revision-canonical`. The full run began before the last absent/empty normalization, followed by the 16-method focused suite and fresh final-source type, lint, syntax, guide, default-artifact and live-UI checks. The earlier 447/449-case runs and all review findings remain in their historical evidence folders.

An earlier external review attempt reached a service usage limit during read-only inspection. Its automatic approval layer rejected that inspection and the process exited without a verdict. After the reported reset, a new review of code commit `1bf1a3e` completed with zero actionable findings and a "patch is correct" result. Both the failed attempt and successful retry are preserved. The final documentation update records that measured result without changing production code.

Default CLI regression uses seed 11/four beats in 12/19 EDO. Score JSON/MIDI must match current upstream byte for byte, with 28 notes each. Stable manifests must match except run ID, timestamp, output directory and separately validated new component provenance. RenderPackages must load and agree with standalone scores/MIDI and requested tuning.

Combined replacement tests use a known one-note decoder and marked MIDI renderer. They prove all four selected components are consumed, not musical quality. Separate tests prove metric effects, controlled successors, pruning and per-section forwarding.

## Limits and manual checks

Nine optional JAX/ML/audio-bridge tests are unavailable. A further symlink-output test needs a Windows privilege this account lacks; an actual directory also checks the regular-file guard. Skips are reported separately from passes. Security probing stays source-only; no deployed-service penetration test is claimed.

The user previously downloaded and played default 12/19-EDO browser examples. [Manual checks](req-10-manual-checks.md) preserves that historical acceptance. No subjective manual test is needed to prove selection/metadata. If presenting the latest upstream planner's music, generate and listen to those presentation examples; earlier playback does not certify a newly generated performance or a player's tuning behavior.

Optional models, external synthesis, long-piece performance, alternate formats in MIDI-specific packaging/preview and whole-repository strict typing remain outside REQ-10. Trusted constructors execute Python and own their resources; the factory is not a sandbox or lifecycle manager. Check remote CI after pushing; local success does not mean GitHub CI passed.
