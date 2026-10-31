# Long-form evaluation suite (MUS-28 / REQ-18) — Milestone 1

`aimusic.evaluation` is a reproducible benchmark framework that sits **downstream** of
generation. It calls the public planning/scoring APIs (`run_method_a`, `SparseGraph`
diagnostics, `SBDiagnostics`, tension scoring) and never changes generation behavior.
Nothing under `aimusic/` outside `aimusic/evaluation/` imports it (enforced by
`tests/test_evaluation.py`).

## One command

```bash
python -m aimusic.evaluation run \
    --config config/benchmarks/m1_64_128.json \
    --out outputs/benchmarks/m1_64_128 \
    --workers 2
```

* Re-running the same command **resumes**: finished runs are skipped, nothing is overwritten.
* An output directory created from a different config is **refused**.
* Subsets: `--conditions`, `--horizons`, `--edos`, `--seeds`; `--retry-failed` re-runs failures;
  `--dry-run` / `list` print the run ids without running.
* `python -m aimusic.evaluation report --out DIR` rebuilds the aggregate and report from raw runs.

Each run executes in its own subprocess, which gives a per-run peak-RSS reading, a hard
`timeout_s`, and crash isolation. Failures, timeouts and crashes are stored as ordinary results
(`status` = `failed` / `timeout` / `crashed`) and never abort the matrix.

## Output layout

| file | content |
|---|---|
| `manifest.json` | config (verbatim + hash), environment, every expected run id and spec |
| `runs/<run_id>.json` | one raw result per run (schema `mus28-result/1`) |
| `results.jsonl` | all raw results, one per line |
| `aggregate.json` | per-cell summaries, CIs, paired ablation deltas (schema `mus28-aggregate/1`) |
| `report.md` | concise human-readable report |

Every result records the full `RunSpec` (condition, horizon, seed, EDO, path mode, prior mode,
`lambda_*`, bridge mode, `K_max`, `D_max`, proposal budget, SB settings, generation method),
`git_commit`/`git_dirty`, a `source_sha256` over `aimusic/**/*.py`, Python/numpy versions, and a
`path_sha256` of the generated state path. No run depends on an undocumented default.

## Conditions (ablations)

Each ablation changes **one** factor from `full`, with the same seed, graph limits, horizon, EDO and
endpoints (verified by `tests/test_evaluation.py`).

| condition | difference from `full` | what it isolates |
|---|---|---|
| `full` | GTTM on, placeholder structured prior, SB bridge | — |
| `no_gttm` | `lambda_gttm = 0`; legality constraints unchanged | GTTM energies |
| `no_prior` | `NullPrior` | prior |
| `no_sb` | same graph; uniform random walk over end-reachable edges instead of the SB solution | SB reweighting |
| `trained_prior` | reserved; reports `prior_unavailable` unless a bundle loads | trained prior (MUS-27) |

Prior states are never conflated: `prior.status` is one of `off_intentional`,
`neutral_algorithmic`, `placeholder_not_trained`, `requested_unavailable`, `loaded`.
**`full` uses the placeholder structured prior — a deterministic heuristic, not a trained or
corpus-backed model.** The production default (`prior=None`) is `NullPrior`, so without this choice
`full` and `no_prior` would be identical.

`no_sb` is only defined for sampled paths (a uniform walk has no meaningful MAP path), so the main
matrix uses `path_modes: ["sample"]`.

## Metrics (all from existing diagnostics)

* **Graph support** — layer width, proposed/legal/unique/scored/retained counts and fractions
  (REQ-13 counters), prune/reject reason counts, inactive GTTM features, support variance across seeds.
* **SB** — iterations, convergence, final delta, effective entropy.
* **Path** — length, endpoint reached, distinct-state fraction, chord/key/role/groove change rates,
  MAP path score finiteness. These are sanity statistics, **not** musical-quality claims.
* **Tension** — mean/max absolute error, shape correlation, peak offset against the planned arc.
* **GTTM** — mean per-edge GTTM score along the chosen path. It uses per-family feature weights, not
  `lambda_gttm`, so it is comparable between `full` and `no_gttm`.
* **Runtime** — wall, CPU, peak process RSS.

## Statistics

Seeds are replicates within a cell. Per metric: n, mean, sample sd, 95% Student-t CI, median, 5th/95th
percentile, min/max. Failure rates carry a Wilson 95% interval. Ablation deltas are **paired by seed**
(`aggregate.json: paired_vs_control`). Raw per-seed results are always kept. Cells never mix EDOs,
horizons or path modes. t-intervals on bounded metrics can overshoot their range; read the median and
percentile range for skewed metrics such as runtime.

No pass/fail thresholds are defined yet. The issue requires explicit thresholds before any long-form
claim; they should be set from these distributions in a later step, with justification.

## Profiling (before attempting 512/1,024 beats)

Method A, `K_max=64, D_max=8, proposal_budget=256`, 12-EDO, single process per run (2 runs in parallel on 2 cores).
Wall time and run-only RSS growth, mean of 5 seeds (64/128) or 1 seed (256, `m1_profile_256`):

| beats | `full` wall | `no_prior` wall | RSS growth |
|---|---|---|---|
| 64 | 77 s | 46 s | 53 MB |
| 128 | 164 s | 94 s | 101 MB |
| 256 | 339 s | 197 s | 194 MB |

Time and memory both scale linearly (about 2.1x per doubling; about 0.76 MB/beat). A cProfile of a 16-beat run
attributes >99% of the time to `build_sparse_graph`: transition scoring
(`calculate_transition_score_breakdowns` -> `transition_feature_vector` / `weighted_feature_breakdown`) is
about 60% and candidate generation / legality checks (`get_valid_next_states`) about 28%. The SB solve is
negligible. The placeholder structured prior adds about 70% to wall time over `NullPrior`; 19-EDO is about
20% slower than 12-EDO.

Extrapolation (**not yet measured**): 512 beats is about 12 min (`full`) and 1,024 beats about 23-25 min per
run, with roughly 0.4 GB and 0.8 GB of memory growth. A 1,024-beat matrix of 4 conditions x 2 EDOs x 5 seeds
(40 runs) would take about 8 hours of CPU, so run it with a reduced seed count first.

## Observations from the first matrix (`docs/benchmarks/m1_64_128/`)

80/80 runs succeeded (5 seeds, 12- and 19-EDO, 64 and 128 beats). These are descriptive, not quality claims:

* `no_gttm` roughly doubles tension MAE (about 0.33-0.37 vs 0.18-0.19) and flips tension-shape correlation
  negative; its mean GTTM score per edge is lower. Paired CIs exclude 0.
* `no_prior` is indistinguishable from `full` on these metrics (the placeholder prior is a weak heuristic),
  but about 40% faster.
* `no_sb` differs little from `full` here, which is expected: with singleton endpoints the bridge is nearly
  degenerate (about one iteration). It should not be read as "SB does not help" until MUS-15 lands.
* 12-EDO and 19-EDO give numerically identical graph and score summaries even though vocabularies differ
  (12 vs 19 keys, 48 vs 76 chords) and paths differ. Cells are still kept separate; the EDO-sensitive
  musical metrics of Milestone 2 are what should discriminate them.
* GTTM feature `beat_position_validity` had no non-zero activation on any retained edge in 38 of 80 runs
  (relevant to the MUS-23 "registered but never activated" check).
* Chord changes are rare (about 0.05 per beat) in every condition.

## MUS-15 endpoint-distribution evaluation (`m2_distribution_method_a`)

MUS-15 made Method A a true distribution-to-distribution bridge. The evaluation measures whether it handles
multi-state start and terminal distributions correctly. It is a **correctness/fidelity check at small horizons
(8 and 16 beats), not a long-form quality benchmark**, and it defines no pass/fail thresholds.

**Endpoint mode.** `RunSpec.endpoint_mode` is `singleton` or `distribution`, set in the config with
`"endpoint_mode"` (absent = `singleton`, so the M1 configs, their hashes and their run ids are unchanged).
The generation method is derived from it and cannot disagree: `method_a_singleton_endpoints` vs
`method_a_distribution_endpoints`. Every run sets the mode explicitly on `MethodARunConfig`, because the MUS-15
library default is `distribution`; a singleton run never inherits it and a distribution run is never collapsed.
Other distribution-only config keys: `endpoint_policy` (an `EndpointFallbackPolicy` value, default
`drop_unreachable_only`, the pipeline default), `endpoint_marginal_tolerance` (default 1e-6) and
`calibration_samples` (extra repeated samples per run, default 0). The `no_sb` condition is rejected in
distribution mode (a uniform walk has no endpoint-marginal bridge). Singleton and distribution cells have
different group keys (`|ep=...`), are aggregated separately, and are never paired in ablation deltas.

**MUS-15 APIs used (nothing is re-implemented).** `MethodARunConfig` (`endpoint_mode`, `unreachable_policy`,
`endpoint_marginal_tolerance`), `run_method_a`, and from the plan result `start_mass` / `end_mass`
(`EndpointMassReport`: `requested`, `effective`, `requested_mass`, `feasible_mass`, `removed_state_mass`,
`trimmed_mass`, `realized_probabilities`, `total_variation_shift`), `endpoint_resolution`
(`EndpointResolution`: policy, `request_modified`, `modifications`, flow checks, `termination_reason`),
`diagnostics.endpoint_mode` / `bridge_converged`, the `EndpointFeasibilityError` family (strict failures), and
`sample_bridge_path` on the solved bridge for calibration. If a plan lacks these diagnostics, or reports a
different endpoint mode than requested, the run fails with kind `endpoint_diagnostics_unavailable`.

**Metrics** (`aimusic/evaluation/endpoint_metrics.py`, `metrics.endpoint`; JSON only):

* Identity and mass accounting: `generation_method`, `endpoint_mode`, `endpoint_support_size_start|terminal`
  (effective support), `requested_|feasible_|removed_|trimmed_|effective_|realized_ start|terminal _mass`.
  The same names exist per side under `start` / `terminal`.
* Outcome (`accounting_status`, per side and overall): `no_modification` (all support feasible, nothing removed),
  `removed_and_renormalized` (unreachable states dropped, survivors renormalized, mass reported),
  `trimmed_and_renormalized`, `removed_trimmed_and_renormalized`, `interior_blend`, and `strict_failure`
  (the policy forbade changing the request: the run fails before solving, with `failure_kind`
  `level1_unreachable_support` or `level2_mass_infeasible`).
* Marginal error, per side: L1 and max-abs of realized vs requested (includes removed/trimmed mass) and vs
  effective (the solver's own error).
* Residuals: per-state rows `{state, requested, effective, realized, residual_vs_effective,
  residual_vs_requested, removed, within_tolerance}`, plus `residual_max_abs`, `residual_mean_abs`,
  `residual_within_tolerance` and `requested_equals_realized_within_tolerance` against
  `endpoint_marginal_tolerance`.
* Entropy (nats): requested, effective, realized and, when samples exist, empirical.
* Calibration: draws `calibration_samples` paths from the single solved bridge (independent derived RNG keys
  `eval.calibration`) and compares empirical start/terminal frequencies with the effective and requested
  distributions (L1, max-abs, max z-score, mass outside support, and the L1 a perfectly calibrated sampler
  would show from sampling noise alone). With fewer than 30 samples (including a MAP path or a single sampled
  path) the status is `unavailable_*` and no numbers are written. A single path is never called calibrated.

In singleton mode the endpoint section is only `{status: "not_applicable"}` plus the support sizes; the dashboard
shows "Distribution metrics unavailable: singleton endpoint mode." rather than zeros.

Reports gain `## Endpoint-distribution evaluation` (handling outcome, start/terminal mass accounting, marginal
error and residuals, entropy, calibration status); the dashboard gains an *Endpoint distribution* tab, endpoint
rows in the run inspector and the config panel, and an `EVALUATION` config type.

Run it (about 90 s; results in `docs/benchmarks/m2_distribution_method_a/`):

```bash
python -m aimusic.evaluation run --config config/benchmarks/m2_distribution_method_a.json \
  --out outputs/benchmarks/m2_distribution_method_a --workers 1
python -m aimusic.evaluation report --out outputs/benchmarks/m2_distribution_method_a
```

Controlled tests: `python -m unittest tests.test_evaluation_method_a_distribution` (hand-built graphs through
`solve_endpoint_bridge`: feasible multi-state, singleton compatibility, strict vs drop/renormalize vs trim,
determinism, repeated sampling, dashboard views).

## Not measured / deferred

* Distribution-to-distribution Method A (MUS-15) is evaluated only by `m2_distribution_method_a` (8/16 beats,
  12-EDO, `full`, sampled paths). The M1 matrices are singleton-endpoint runs, tagged
  `generation_method = method_a_singleton_endpoints`, and have no endpoint-marginal, residual or calibration
  metrics. Not covered for MUS-15: other policies (trim, conditional) on the real pipeline, 19-EDO, long
  horizons, ablations in distribution mode (no `no_sb`), and calibration of the terminal marginal beyond what
  one solved bridge shows (the sampler is checked against that bridge, not against an independent oracle).
* Method B (MUS-18) and section-wise SB with soft stitching (MUS-16): the config enums exist but
  `plans.py` does not branch on them, so no A/B or monolithic/section-wise comparison.
* NGramPrior (MUS-26) and trained neural prior (MUS-27): the `trained_prior` condition reports an
  explicit failure rather than substituting another prior.
* Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`),
  decoder-density ablation, Markov/Viterbi baselines, musical-structure metrics (cadence, motif
  recurrence, voice leading, groove continuity), and `K_max`/`D_max`/budget sweeps (the config supports a
  `graph_grid`, but only one point is run).
* Horizons 256–1,024 and the listening-test package (Milestones 2 and 3).

## Evaluation dashboard (Gradio)

`python ui.py` serves the generation UI at `/` and the evaluation dashboard at `/evaluation`
(implemented in `aimusic/evaluation/ui.py`; needs `pip install -e ".[ui]"`).

* **Runs go through the CLI.** *Run*, *Resume*, *List Runs* and *Generate Report* launch
  `python -m aimusic.evaluation run|report ...` in a subprocess (stdout+stderr are kept in
  `<out>/ui_run.log` and shown in the page), so UI and command-line results are identical. The page polls
  while a job runs and never blocks; *Stop* terminates the CLI and its workers, and *Resume* continues.
  A directory created from a different config is refused before anything starts.
* **Config-aware controls (MUS-28 UI).** Selecting a config shows a *Benchmark Configuration* panel read straight
  from the JSON (purpose, version, type, horizons, EDOs, seeds, conditions with explanations, path modes,
  `K_max`/`D_max`/proposal budget, SB settings, timeout, expected run count) and a `SMOKE` / `PROFILE` /
  `BENCHMARK` / `EVALUATION` label and the endpoint mode. The type is inferred from the config name/description, or set with an optional
  `"benchmark_type"` key (do not add it to shipped configs: it would change their `config_hash`). Condition
  descriptions live in `ui.CONDITION_DESCRIPTIONS` and can be overridden per config with an optional
  `"condition_descriptions": {name: text}` object. Condition/horizon/EDO/seed filters are checkbox groups that
  only offer values from the selected config; switching config clears ticks that no longer exist, and an empty
  group means "all values". Under the filters the page previews the exact matrix (`2 conditions × 1 horizon × 2
  seeds × 1 EDO = 4 runs`, computed with `expand_matrix`/`select_specs`), the output directory, whether Run
  would start, resume or skip, and a rough runtime when the directory has finished runs at the same horizon
  and EDO. Invalid or zero-run selections are refused before any subprocess starts, with a message naming the
  config and its valid values. Help text for each term is in `ui.TERM_HELP`.
* **Data sources.** `manifest.json` + `runs/*.json` (falling back to `results.jsonl`) for run level,
  `aggregate.json` for cell level with its 95% intervals. If `aggregate.json` is missing or out of date
  (e.g. mid-run) it is recomputed with `stats.aggregate`, the same code the CLI uses. `report.md` is shown as a
  reference view only.
* **Views.** Overview cards, runtime/memory bars with CIs, run-status chart, paired ablation deltas vs `full`,
  tension target-vs-realized, runtime/memory scaling by horizon, 12- vs 19-EDO (EDOs are never pooled), and a
  per-run inspector (including failure records), and an *Endpoint distribution* tab (mass accounting, support
  sizes, marginal error, residuals, entropy, calibration status; distribution cells only).
* **Tension curves.** New runs record `metrics.tension.target_curve` / `realized_curve` (per-beat lists, ignored by
  the aggregator). Results produced before this change have only summary tension metrics; the dashboard shows those
  instead and says so. Re-run to get curves.
