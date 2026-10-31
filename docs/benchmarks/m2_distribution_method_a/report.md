# MUS-28 benchmark report: m2_distribution_method_a

- Config: `m2_distribution_method_a` version `1.0.0`, hash `daf7c73aca66`
- Code: commit `521dc4bc1672` on `feature/req-18-evaluation-suite` (dirty working tree); Python 3.13.16, numpy 2.5.3
- Runs: 6/6 completed, 6 ok, 0 failed/timeout/crashed
- Generation: `method_a_distribution_endpoints`, single pass. `full` uses the **placeholder** structured prior (a deterministic heuristic, not a trained or corpus-backed model).
- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, path modes or endpoint modes. No pass/fail thresholds are defined yet, so nothing here is a quality claim.

## Success and failure

| cell | ok/runs | failure rate [Wilson 95%] | failure kinds |
|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 3/3 | 0.00 [0.00, 0.56] | - |
| full / 16b / 12-EDO / sample / distribution | 3/3 | 0.00 [0.00, 0.56] | - |

## Runtime and memory

| cell | wall s | median s | p95 s | peak RSS MB (process) | RSS growth MB (run only) |
|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 10.2 [9.0, 11.4] | 10.3 | 10.6 | 49 [47, 52] | 16 [13, 18] |
| full / 16b / 12-EDO / sample / distribution | 23.2 [21.9, 24.5] | 22.9 | 23.7 | 55 [51, 59] | 21 [17, 25] |

## Graph support

| cell | layer width | legal/proposed | unique/legal | retained/raw edges | unique cands | support sd (seeds) | inactive GTTM features |
|---|---|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 16.4 [8.0, 24.9] | 0.878 [0.849, 0.908] | 0.025 [0.019, 0.031] | 0.014 [0.008, 0.021] | 2,184 | 227.8 | 1.3 [-0.1, 2.8] |
| full / 16b / 12-EDO / sample / distribution | 10.9 [6.3, 15.6] | 0.895 [0.874, 0.915] | 0.027 [0.025, 0.029] | 0.009 [0.006, 0.012] | 5,513 | 230.5 | 1.3 [-0.1, 2.8] |

`scored/legal` is 0 unless `prior_guided_proposals` is enabled. Prune/reject reason counts are in the raw `metrics.graph.prune_reason_counts` fields. Registered GTTM features with no non-zero activation on any retained edge: `beat_position_validity` (6 runs), `head_resolution` (2 runs). Intervals on fractions can overshoot [0, 1] (t-interval).

## Schrodinger bridge

| cell | converged rate | iterations | effective entropy |
|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 5.3 [-0.9, 11.6] | 1.932 [1.667, 2.196] |
| full / 16b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 4.3 [0.5, 8.1] | 1.571 [0.973, 2.169] |

## Path, tension and GTTM

| cell | length ok | distinct states | chord change rate | tension MAE | tension shape corr | mean GTTM score/edge |
|---|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 0.250 [-0.061, 0.561] | 0.260 [0.141, 0.380] | 0.339 [-0.009, 0.687] | 7.743 [7.260, 8.225] |
| full / 16b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 0.208 [-0.150, 0.567] | 0.199 [0.140, 0.258] | 0.332 [0.140, 0.524] | 8.187 [7.806, 8.568] |

`mean GTTM score/edge` uses the per-family feature weights, not `lambda_gttm`, so it is comparable between `full` and `no_gttm`.

## Endpoint-distribution evaluation

### Endpoint handling outcome

How each run's requested endpoints were treated by the MUS-15 policy (`no_modification` = all support feasible, nothing removed; `removed_and_renormalized` = unreachable states dropped and survivors renormalized; `strict_failure` = the policy forbade changing the request, so the run failed before solving).

| cell | policy | outcome (runs) |
|---|---|---|
| full / 8b / 12-EDO / sample / distribution | `drop_unreachable_only` | removed_and_renormalized x3 |
| full / 16b / 12-EDO / sample / distribution | `drop_unreachable_only` | removed_and_renormalized x3 |

### Start boundary: mass accounting and support

| cell | support requested | support effective | requested mass | feasible mass | removed mass | trimmed mass | effective mass | realized mass |
|---|---|---|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 3.3 [1.9, 4.8] | 1.000 [1.000, 1.000] | 0.417 [0.237, 0.596] | 0.583 [0.404, 0.763] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 16b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 2.7 [1.2, 4.1] | 1.000 [1.000, 1.000] | 0.333 [0.154, 0.513] | 0.667 [0.487, 0.846] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

### Terminal boundary: mass accounting and support

| cell | support requested | support effective | requested mass | feasible mass | removed mass | trimmed mass | effective mass | realized mass |
|---|---|---|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 16b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Masses are probability mass; `effective` is what the solver was asked to match (1 after renormalization) and `realized` is the bridge's measured marginal.

### Marginal error and residuals

| cell | side | L1 vs requested | max abs vs requested | L1 vs effective | max abs vs effective | max per-state residual | within tolerance (rate) | realized = requested (rate) |
|---|---|---|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | start | 1.17e+00 | 1.81e-01 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00 | 0.00 |
| full / 8b / 12-EDO / sample / distribution | terminal | 6.26e-09 | 1.53e-09 | 6.26e-09 | 1.53e-09 | 1.53e-09 | 1.00 | 1.00 |
| full / 16b / 12-EDO / sample / distribution | start | 1.33e+00 | 2.64e-01 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00 | 0.00 |
| full / 16b / 12-EDO / sample / distribution | terminal | 4.33e-09 | 8.24e-10 | 4.33e-09 | 8.24e-10 | 8.24e-10 | 1.00 | 1.00 |

`vs requested` includes any mass removed or trimmed by the policy; `vs effective` is the solver's own marginal error (residual = realized - effective per state, checked against `endpoint_marginal_tolerance`). Per-state rows are in `metrics.endpoint.<side>.residuals`.

### Entropy (nats)

| cell | side | requested | effective | empirical (sampled) |
|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | start | 2.079 [2.079, 2.079] | 1.195 [0.782, 1.607] | 1.190 [0.782, 1.598] |
| full / 8b / 12-EDO / sample / distribution | terminal | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | 2.057 [2.032, 2.082] |
| full / 16b / 12-EDO / sample / distribution | start | 2.079 [2.079, 2.079] | 0.963 [0.382, 1.545] | 0.960 [0.379, 1.541] |
| full / 16b / 12-EDO / sample / distribution | terminal | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | 2.067 [2.048, 2.085] |

### Calibration (repeated samples from the solved bridge)

| cell | side | status (runs) | samples/run | L1 vs effective | expected L1 noise | max abs z |
|---|---|---|---|---|---|---|
| full / 8b / 12-EDO / sample / distribution | start | available x3 | 200 | 0.072 [-0.058, 0.202] | 0.086 [0.060, 0.111] | 1.12 [-0.88, 3.13] |
| full / 8b / 12-EDO / sample / distribution | terminal | available x3 | 200 | 0.167 [0.066, 0.267] | 0.149 [0.149, 0.149] | 2.07 [0.44, 3.69] |
| full / 16b / 12-EDO / sample / distribution | start | available x3 | 200 | 0.069 [-0.058, 0.195] | 0.072 [0.038, 0.106] | 1.01 [-0.87, 2.89] |
| full / 16b / 12-EDO / sample / distribution | terminal | available x3 | 200 | 0.133 [0.039, 0.227] | 0.149 [0.149, 0.149] | 1.35 [0.02, 2.69] |

Calibration needs at least 30 repeated samples; otherwise it is reported as unavailable (shown as `-`), never as zero. A single MAP or sampled path is not a calibration. `expected L1 noise` is the L1 a perfectly calibrated sampler would show from finite sampling alone.

## Ablations: paired differences vs `full` (same seeds)

No ablation cell has a matching `full` cell yet.

## Not measured / deferred

- Endpoint-distribution evaluation is a correctness/fidelity check of Method A at small horizons (8/16 beats here), not a long-form quality benchmark; calibration compares repeated samples from one solved bridge and says nothing about MAP paths.
- Method B start-middle-return planning (MUS-18): not implemented in `plans.py`; no A/B comparison.
- Section-wise SB with soft stitching (MUS-16): not implemented; no monolithic vs section-wise comparison.
- Factorized corpus NGramPrior (MUS-26) and a trained neural prior (MUS-27): `trained_prior` reports an explicit `prior_unavailable` failure unless a bundle is configured.
- Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`), decoder-density ablation, ordinary Markov/Viterbi baselines, and the musical-structure metrics (cadence, motif recurrence, voice leading, groove continuity) are Milestone 2.
- Listening-test package: Milestone 3.
