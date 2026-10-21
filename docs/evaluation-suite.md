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

## Not measured / deferred

* True distribution-to-distribution Method A (MUS-15). `run_method_a` still collapses to one start and
  one terminal state, so SB converges in about one iteration and endpoint-marginal error, residuals and
  calibration are meaningless today. Results are tagged `generation_method = method_a_singleton_endpoints`.
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
  `BENCHMARK` label. The type is inferred from the config name/description, or set with an optional
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
  per-run inspector (including failure records).
* **Tension curves.** New runs record `metrics.tension.target_curve` / `realized_curve` (per-beat lists, ignored by
  the aggregator). Results produced before this change have only summary tension metrics; the dashboard shows those
  instead and says so. Re-run to get curves.
