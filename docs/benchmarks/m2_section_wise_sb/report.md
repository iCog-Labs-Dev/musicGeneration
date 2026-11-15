# MUS-28 benchmark report: m2_section_wise_sb

- Config: `m2_section_wise_sb` version `1.0.0`, hash `29e992680808`
- Code: commit `ec44d8c27313` on `feature/req-18-evaluation-suite` (dirty working tree); Python 3.13.18, numpy 2.5.3
- Runs: 12/12 completed, 12 ok, 0 failed/timeout/crashed
- Generation: `method_a_distribution_endpoints`, single pass and section-wise (MUS-16, reported as separate cells). `full` uses the **placeholder** structured prior (a deterministic heuristic, not a trained or corpus-backed model).
- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, path modes, endpoint modes or sectioning modes. No pass/fail thresholds are defined yet, so nothing here is a quality claim.

## Success and failure

| cell | ok/runs | failure rate [Wilson 95%] | failure kinds |
|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 3/3 | 0.00 [0.00, 0.56] | - |
| full / 16b / 12-EDO / sample / distribution | 3/3 | 0.00 [0.00, 0.56] | - |
| full / 32b / 12-EDO / sample / distribution / section_wise | 3/3 | 0.00 [0.00, 0.56] | - |
| full / 32b / 12-EDO / sample / distribution | 3/3 | 0.00 [0.00, 0.56] | - |

## Runtime and memory

| cell | wall s | median s | p95 s | peak RSS MB (process) | RSS growth MB (run only) |
|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 9.9 [7.9, 11.8] | 10.1 | 10.5 | 64 [63, 64] | 29 [29, 29] |
| full / 16b / 12-EDO / sample / distribution | 16.0 [14.4, 17.5] | 15.6 | 16.6 | 65 [64, 67] | 31 [30, 32] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 27.3 [25.4, 29.3] | 27.7 | 27.9 | 85 [84, 86] | 50 [49, 51] |
| full / 32b / 12-EDO / sample / distribution | 32.2 [30.6, 33.7] | 32.4 | 32.6 | 84 [84, 84] | 49 [49, 49] |

## Graph support

| cell | layer width | legal/proposed | unique/legal | retained/raw edges | unique cands | support sd (seeds) | inactive GTTM features |
|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 35.6 [34.1, 37.1] | 0.835 [0.834, 0.836] | 0.016 [0.015, 0.016] | 0.041 [0.040, 0.042] | 1,625 | 4.6 | 1.0 [1.0, 1.0] |
| full / 16b / 12-EDO / sample / distribution | 26.1 [22.3, 30.0] | 0.837 [0.811, 0.864] | 0.019 [0.018, 0.020] | 0.017 [0.016, 0.018] | 3,613 | 109.5 | 1.0 [1.0, 1.0] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 32.8 [31.9, 33.8] | 0.831 [0.824, 0.838] | 0.017 [0.016, 0.017] | 0.024 [0.022, 0.025] | 5,249 | 60.9 | 1.0 [1.0, 1.0] |
| full / 32b / 12-EDO / sample / distribution | 23.3 [21.6, 24.9] | 0.838 [0.832, 0.845] | 0.020 [0.019, 0.021] | 0.014 [0.014, 0.015] | 8,055 | 213.8 | 1.0 [1.0, 1.0] |

`scored/legal` is 0 unless `prior_guided_proposals` is enabled. Prune/reject reason counts are in the raw `metrics.graph.prune_reason_counts` fields. Registered GTTM features with no non-zero activation on any retained edge: `beat_position_validity` (12 runs). Intervals on fractions can overshoot [0, 1] (t-interval).

## Schrodinger bridge

| cell | converged rate | iterations | effective entropy |
|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 1.00 [1.00, 1.00] | 42.0 [35.4, 48.6] | 2.963 [2.954, 2.972] |
| full / 16b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 5.0 [2.5, 7.5] | 2.482 [2.165, 2.800] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 1.00 [1.00, 1.00] | 36.7 [27.9, 45.4] | 2.794 [2.647, 2.941] |
| full / 32b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 4.3 [-1.9, 10.6] | 2.029 [1.551, 2.506] |

## Path, tension and GTTM

| cell | length ok | distinct states | chord change rate | tension MAE | tension shape corr | mean GTTM score/edge |
|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 1.00 [1.00, 1.00] | 0.896 [0.448, 1.344] | 0.146 [-0.033, 0.325] | 0.333 [0.333, 0.333] | -0.015 [-0.410, 0.381] | 7.386 [7.111, 7.662] |
| full / 16b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 0.958 [0.869, 1.048] | 0.104 [-0.133, 0.341] | 0.363 [0.356, 0.371] | 0.009 [-0.315, 0.333] | 7.907 [7.799, 8.015] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 1.00 [1.00, 1.00] | 0.927 [0.808, 1.046] | 0.073 [-0.046, 0.192] | 0.372 [0.335, 0.408] | -0.011 [-0.293, 0.270] | 7.789 [7.593, 7.984] |
| full / 32b / 12-EDO / sample / distribution | 1.00 [1.00, 1.00] | 0.917 [0.872, 0.961] | 0.104 [0.059, 0.149] | 0.346 [0.329, 0.363] | -0.040 [-0.343, 0.263] | 8.105 [7.968, 8.241] |

`mean GTTM score/edge` uses the per-family feature weights, not `lambda_gttm`, so it is comparable between `full` and `no_gttm`.

## Endpoint-distribution evaluation

### Endpoint handling outcome

How each run's requested endpoints were treated by the MUS-15 policy (`no_modification` = all support feasible, nothing removed; `removed_and_renormalized` = unreachable states dropped and survivors renormalized; `strict_failure` = the policy forbade changing the request, so the run failed before solving).

| cell | policy | outcome (runs) |
|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | `drop_unreachable_only` | no_modification x3 |
| full / 16b / 12-EDO / sample / distribution | `drop_unreachable_only` | removed_and_renormalized x3 |
| full / 32b / 12-EDO / sample / distribution / section_wise | `drop_unreachable_only` | no_modification x3 |
| full / 32b / 12-EDO / sample / distribution | `drop_unreachable_only` | removed_and_renormalized x3 |

### Start boundary: mass accounting and support

| cell | support requested | support effective | requested mass | feasible mass | removed mass | trimmed mass | effective mass | realized mass |
|---|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 16b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 6.3 [4.9, 7.8] | 1.000 [1.000, 1.000] | 0.792 [0.612, 0.971] | 0.208 [0.029, 0.388] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 32b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 5.3 [1.5, 9.1] | 1.000 [1.000, 1.000] | 0.667 [0.192, 1.141] | 0.333 [-0.141, 0.808] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

### Terminal boundary: mass accounting and support

| cell | support requested | support effective | requested mass | feasible mass | removed mass | trimmed mass | effective mass | realized mass |
|---|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 16b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| full / 32b / 12-EDO / sample / distribution | 8.0 [8.0, 8.0] | 8.0 [8.0, 8.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Masses are probability mass; `effective` is what the solver was asked to match (1 after renormalization) and `realized` is the bridge's measured marginal.

### Marginal error and residuals

| cell | side | L1 vs requested | max abs vs requested | L1 vs effective | max abs vs effective | max per-state residual | within tolerance (rate) | realized = requested (rate) |
|---|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | start | 4.63e-16 | 1.39e-16 | 4.63e-16 | 1.39e-16 | 1.39e-16 | 1.00 | 1.00 |
| full / 16b / 12-EDO / sample / distribution / section_wise | terminal | 2.50e-08 | 4.38e-09 | 2.50e-08 | 4.38e-09 | 4.38e-09 | 1.00 | 1.00 |
| full / 16b / 12-EDO / sample / distribution | start | 4.17e-01 | 1.25e-01 | 1.76e-16 | 2.78e-17 | 2.78e-17 | 1.00 | 0.00 |
| full / 16b / 12-EDO / sample / distribution | terminal | 4.65e-09 | 9.19e-10 | 4.65e-09 | 9.19e-10 | 9.19e-10 | 1.00 | 1.00 |
| full / 32b / 12-EDO / sample / distribution / section_wise | start | 9.99e-16 | 2.50e-16 | 9.99e-16 | 2.50e-16 | 2.50e-16 | 1.00 | 1.00 |
| full / 32b / 12-EDO / sample / distribution / section_wise | terminal | 1.53e-08 | 3.16e-09 | 1.53e-08 | 3.16e-09 | 3.16e-09 | 1.00 | 1.00 |
| full / 32b / 12-EDO / sample / distribution | start | 6.67e-01 | 1.25e-01 | 6.48e-17 | 9.25e-18 | 9.25e-18 | 1.00 | 0.00 |
| full / 32b / 12-EDO / sample / distribution | terminal | 1.89e-09 | 4.12e-10 | 1.89e-09 | 4.12e-10 | 4.12e-10 | 1.00 | 1.00 |

`vs requested` includes any mass removed or trimmed by the policy; `vs effective` is the solver's own marginal error (residual = realized - effective per state, checked against `endpoint_marginal_tolerance`). Per-state rows are in `metrics.endpoint.<side>.residuals`.

### Entropy (nats)

| cell | side | requested | effective | empirical (sampled) |
|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | start | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | - |
| full / 16b / 12-EDO / sample / distribution / section_wise | terminal | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | - |
| full / 16b / 12-EDO / sample / distribution | start | 2.079 [2.079, 2.079] | 1.843 [1.622, 2.064] | - |
| full / 16b / 12-EDO / sample / distribution | terminal | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | - |
| full / 32b / 12-EDO / sample / distribution / section_wise | start | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | - |
| full / 32b / 12-EDO / sample / distribution / section_wise | terminal | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | - |
| full / 32b / 12-EDO / sample / distribution | start | 2.079 [2.079, 2.079] | 1.647 [0.947, 2.347] | - |
| full / 32b / 12-EDO / sample / distribution | terminal | 2.079 [2.079, 2.079] | 2.079 [2.079, 2.079] | - |

### Calibration (repeated samples from the solved bridge)

| cell | side | status (runs) | samples/run | L1 vs effective | expected L1 noise | max abs z |
|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | start | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 16b / 12-EDO / sample / distribution / section_wise | terminal | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 16b / 12-EDO / sample / distribution | start | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 16b / 12-EDO / sample / distribution | terminal | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 32b / 12-EDO / sample / distribution / section_wise | start | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 32b / 12-EDO / sample / distribution / section_wise | terminal | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 32b / 12-EDO / sample / distribution | start | unavailable_no_repeated_samples x3 | 0 | - | - | - |
| full / 32b / 12-EDO / sample / distribution | terminal | unavailable_no_repeated_samples x3 | 0 | - | - | - |

Calibration needs at least 30 repeated samples; otherwise it is reported as unavailable (shown as `-`), never as zero. A single MAP or sampled path is not a calibration. `expected L1 noise` is the L1 a perfectly calibrated sampler would show from finite sampling alone.

## Section-wise SB evaluation

Monolithic Method A (`single_pass`) against section-wise Method A with soft stitching (`section_wise`, sections `intro`, `theme`, `bridge`, `return`). Pairs are matched on seed, horizon, EDO, endpoint mode, graph limits, prior, path mode, style and the authored section plan; only the sectioning mode differs. `single_pass` and `section_wise` cells are never pooled.

> **Comparison type: descriptive.** Matched seeds per pair: 3, 3; a paired confidence interval needs at least 5. Differences below are `section_wise - single_pass` shown as mean and range over matched seeds, not as significance tests.

### Success and failure

Three levels are reported separately: every **section solved**, every **join valid** (within the stitching tolerance) and the **whole-song path assembled**. A song can pass one and fail the next.

| cell | all sections solved | all joins valid | whole-song path assembled | failure rate [Wilson 95%] |
|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 3/3 | 3/3 | 3/3 | 0.00 [0.00, 0.56] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 3/3 | 3/3 | 3/3 | 0.00 [0.00, 0.56] |

| matched cell | single_pass ok | section_wise ok | both ok | only single_pass ok | only section_wise ok | both failed |
|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution | 3/3 | 3/3 | 3 | 0 | 0 | 0 |
| full / 32b / 12-EDO / sample / distribution | 3/3 | 3/3 | 3 | 0 | 0 | 0 |

### Runtime and memory

| matched cell | single_pass wall s | section_wise wall s | delta wall s | single_pass peak RSS MB | section_wise peak RSS MB | delta peak RSS MB |
|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution | 16.0 [14.4, 17.5] | 9.9 [7.9, 11.8] | -6.1 (range -7.7 to -5.0, n=3) | 65 [64, 67] | 64 [63, 64] | -1.4 (range -1.8 to -0.7, n=3) |
| full / 32b / 12-EDO / sample / distribution | 32.2 [30.6, 33.7] | 27.3 [25.4, 29.3] | -4.8 (range -6.2 to -3.8, n=3) | 84 [84, 84] | 85 [84, 86] | 1.0 (range 0.5 to 1.4, n=3) |

Wall time includes graph construction and bridge solving for every section. Peak RSS is the process high-water mark of one run (each run is its own process), so it is a whole-run figure only: per-section peak memory is not measurable.

### Tension, path and endpoints

Both modes are scored against the **same authored section-plan targets** (`tension_section_plan`), because the monolithic planner itself targets a single whole-song arc. Whole-song `tension` against each planner's own targets is in `## Path, tension and GTTM` above.

| matched cell | metric | single_pass | section_wise | delta (sw - sp) |
|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution | tension MAE (section plan) | 0.378 [0.314, 0.442] | 0.333 [0.333, 0.333] | -0.045 (range -0.074 to -0.028, n=3) |
| full / 16b / 12-EDO / sample / distribution | tension shape corr (section plan) | 0.015 [-0.440, 0.471] | -0.015 [-0.410, 0.381] | -0.030 (range -0.346 to 0.198, n=3) |
| full / 16b / 12-EDO / sample / distribution | distinct states | 0.958 [0.869, 1.048] | 0.896 [0.448, 1.344] | -0.062 (range -0.312 to 0.062, n=3) |
| full / 16b / 12-EDO / sample / distribution | chord change rate | 0.104 [-0.133, 0.341] | 0.146 [-0.033, 0.325] | 0.042 (range -0.125 to 0.188, n=3) |
| full / 16b / 12-EDO / sample / distribution | role change rate | 0.729 [0.550, 0.908] | 0.833 [0.744, 0.923] | 0.104 (range 0.062 to 0.125, n=3) |
| full / 16b / 12-EDO / sample / distribution | groove change rate | 0.271 [0.181, 0.360] | 0.208 [0.119, 0.298] | -0.062 (range -0.125 to 0.000, n=3) |
| full / 16b / 12-EDO / sample / distribution | mean GTTM score/edge | 7.907 [7.799, 8.015] | 7.386 [7.111, 7.662] | -0.521 (range -0.660 to -0.434, n=3) |
| full / 16b / 12-EDO / sample / distribution | SB effective entropy | 2.482 [2.165, 2.800] | 2.963 [2.954, 2.972] | 0.481 (range 0.338 to 0.567, n=3) |
| full / 32b / 12-EDO / sample / distribution | tension MAE (section plan) | 0.353 [0.305, 0.400] | 0.372 [0.335, 0.408] | 0.019 (range -0.012 to 0.049, n=3) |
| full / 32b / 12-EDO / sample / distribution | tension shape corr (section plan) | 0.036 [-0.266, 0.339] | -0.011 [-0.293, 0.270] | -0.048 (range -0.235 to 0.102, n=3) |
| full / 32b / 12-EDO / sample / distribution | distinct states | 0.917 [0.872, 0.961] | 0.927 [0.808, 1.046] | 0.010 (range -0.062 to 0.062, n=3) |
| full / 32b / 12-EDO / sample / distribution | chord change rate | 0.104 [0.059, 0.149] | 0.073 [-0.046, 0.192] | -0.031 (range -0.062 to 0.031, n=3) |
| full / 32b / 12-EDO / sample / distribution | role change rate | 0.719 [0.563, 0.874] | 0.750 [0.672, 0.828] | 0.031 (range -0.062 to 0.094, n=3) |
| full / 32b / 12-EDO / sample / distribution | groove change rate | 0.250 [-0.061, 0.561] | 0.281 [0.204, 0.359] | 0.031 (range -0.125 to 0.188, n=3) |
| full / 32b / 12-EDO / sample / distribution | mean GTTM score/edge | 8.105 [7.968, 8.241] | 7.789 [7.593, 7.984] | -0.316 (range -0.422 to -0.226, n=3) |
| full / 32b / 12-EDO / sample / distribution | SB effective entropy | 2.029 [1.551, 2.506] | 2.794 [2.647, 2.941] | 0.766 (range 0.630 to 0.929, n=3) |

| section-wise cell | outer endpoint | requested mass | effective mass | realized mass | L1 vs effective |
|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | whole-song start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 4.63e-16 |
| full / 16b / 12-EDO / sample / distribution / section_wise | whole-song terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.50e-08 |
| full / 32b / 12-EDO / sample / distribution / section_wise | whole-song start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 9.99e-16 |
| full / 32b / 12-EDO / sample / distribution / section_wise | whole-song terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.53e-08 |

Outer endpoints only; per-section requested/effective/realized mass (including the shared interior boundaries) is in the per-section table and in `per_section_endpoint_metrics`. The SB entropy of a section-wise run summarizes independent section solves and is not the entropy of a joint whole-song bridge. Lower tension error or fewer changes are not, by themselves, musical quality claims.

### Per-section summary

One row per section of each section-wise cell, aggregated over seeds whose run succeeded. Tension is measured on the whole-song realized curve sliced to the section; `mono MAE` is the monolithic run's error on the same slice and the same target.

| cell | section | beats | status (runs) | runtime s | layer width | retained states | SB iters | SB converged | SB entropy | tension MAE | mono MAE | tension corr | chord change rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution | intro | 4 [4, 4] | ok x3 | 2.56 [2.29, 2.83] | 31.9 [28.4, 35.4] | 196 [185, 207] | 11.7 [7.9, 15.5] | 1.00 [1.00, 1.00] | 2.872 [2.826, 2.918] | 0.230 [0.188, 0.272] | 0.298 [-0.032, 0.627] | 0.724 [0.598, 0.850] | 0.250 [0.250, 0.250] |
| full / 16b / 12-EDO / sample / distribution | theme | 4 [4, 4] | ok x3 | 2.38 [1.47, 3.29] | 31.1 [30.4, 31.9] | 188 [188, 188] | 11.3 [7.5, 15.1] | 1.00 [1.00, 1.00] | 2.833 [2.725, 2.942] | 0.398 [0.231, 0.565] | 0.380 [0.267, 0.493] | -0.341 [-1.069, 0.388] | 0.000 [0.000, 0.000] |
| full / 16b / 12-EDO / sample / distribution | bridge | 4 [4, 4] | ok x3 | 2.27 [1.55, 3.00] | 31.4 [29.7, 33.1] | 188 [188, 188] | 12.0 [9.5, 14.5] | 1.00 [1.00, 1.00] | 2.842 [2.739, 2.945] | 0.356 [0.202, 0.510] | 0.440 [0.358, 0.522] | -0.459 [-0.647, -0.271] | 0.167 [-0.192, 0.525] |
| full / 16b / 12-EDO / sample / distribution | return | 4 [4, 4] | ok x3 | 2.55 [1.97, 3.12] | 31.4 [27.8, 35.0] | 188 [188, 188] | 7.0 [7.0, 7.0] | 1.00 [1.00, 1.00] | 2.775 [2.715, 2.836] | 0.349 [0.205, 0.492] | 0.395 [0.156, 0.633] | -0.035 [-0.914, 0.844] | 0.000 [0.000, 0.000] |
| full / 32b / 12-EDO / sample / distribution | intro | 8 [8, 8] | ok x3 | 6.76 [5.83, 7.69] | 31.4 [28.1, 34.8] | 452 [441, 463] | 10.0 [7.5, 12.5] | 1.00 [1.00, 1.00] | 2.755 [2.647, 2.863] | 0.315 [0.200, 0.429] | 0.308 [0.245, 0.372] | 0.102 [-0.714, 0.918] | 0.042 [-0.138, 0.221] |
| full / 32b / 12-EDO / sample / distribution | theme | 8 [8, 8] | ok x3 | 7.08 [6.11, 8.06] | 30.7 [27.5, 33.9] | 444 [444, 444] | 10.0 [3.4, 16.6] | 1.00 [1.00, 1.00] | 2.781 [2.485, 3.078] | 0.383 [0.337, 0.429] | 0.368 [0.220, 0.515] | -0.140 [-0.601, 0.320] | 0.083 [-0.275, 0.442] |
| full / 32b / 12-EDO / sample / distribution | bridge | 8 [8, 8] | ok x3 | 6.70 [6.29, 7.12] | 30.2 [28.3, 32.1] | 444 [444, 444] | 10.0 [10.0, 10.0] | 1.00 [1.00, 1.00] | 2.703 [2.560, 2.845] | 0.397 [0.316, 0.477] | 0.370 [0.272, 0.467] | -0.189 [-0.952, 0.573] | 0.042 [-0.138, 0.221] |
| full / 32b / 12-EDO / sample / distribution | return | 8 [8, 8] | ok x3 | 6.63 [5.97, 7.29] | 30.7 [26.0, 35.5] | 444 [444, 444] | 6.7 [3.8, 9.5] | 1.00 [1.00, 1.00] | 2.701 [2.490, 2.911] | 0.392 [0.386, 0.398] | 0.365 [0.310, 0.419] | 0.212 [0.161, 0.264] | 0.042 [-0.138, 0.221] |

#### Per-section endpoint mass

| cell | section | side | requested mass | effective mass | realized mass | L1 vs effective |
|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution | intro | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 4.63e-16 |
| full / 16b / 12-EDO / sample / distribution | intro | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.76e-07 |
| full / 16b / 12-EDO / sample / distribution | theme | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.61e-15 |
| full / 16b / 12-EDO / sample / distribution | theme | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.40e-07 |
| full / 16b / 12-EDO / sample / distribution | bridge | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.11e-15 |
| full / 16b / 12-EDO / sample / distribution | bridge | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.47e-07 |
| full / 16b / 12-EDO / sample / distribution | return | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 4.63e-16 |
| full / 16b / 12-EDO / sample / distribution | return | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.50e-08 |
| full / 32b / 12-EDO / sample / distribution | intro | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 9.99e-16 |
| full / 32b / 12-EDO / sample / distribution | intro | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 7.70e-08 |
| full / 32b / 12-EDO / sample / distribution | theme | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.22e-16 |
| full / 32b / 12-EDO / sample / distribution | theme | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 8.98e-08 |
| full / 32b / 12-EDO / sample / distribution | bridge | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 6.11e-16 |
| full / 32b / 12-EDO / sample / distribution | bridge | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 9.31e-08 |
| full / 32b / 12-EDO / sample / distribution | return | start | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 6.11e-16 |
| full / 32b / 12-EDO / sample / distribution | return | terminal | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.53e-08 |

Each interior boundary is one authored distribution shared by both neighbours; a section may not trim or drop it, so requested = effective = realized there unless that section failed.

### Stitching and joins

Costs are the planner's `StitchCost` components (mismatch in [0, 1]); the **worse** of the incoming edge (last transition of the left section) and the outgoing edge (first transition of the right section) is reported. Continuity is `1 - mismatch`. `within tolerance` compares the stitching cost with the configured `max_cost`.

| cell | joins | key jump | chord jump | meter cont. | groove cont. | role cont. | head cont. | boundary cont. | cost (mean) | cost (worst) | within tolerance (rate) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | 3.000 [3.000, 3.000] | 0.000 [0.000, 0.000] | 0.111 [-0.128, 0.350] | 1.000 [1.000, 1.000] | 0.778 [0.300, 1.256] | 0.000 [0.000, 0.000] | 0.222 [-0.256, 0.700] | 0.333 [0.333, 0.333] | 2.778 [1.916, 3.640] | 3.000 [1.566, 4.434] | 1.000 [1.000, 1.000] |
| full / 32b / 12-EDO / sample / distribution / section_wise | 3.000 [3.000, 3.000] | 0.000 [0.000, 0.000] | 0.167 [-0.247, 0.581] | 1.000 [1.000, 1.000] | 0.444 [-0.034, 0.923] | 0.000 [0.000, 0.000] | 0.333 [-1.101, 1.768] | 0.333 [0.333, 0.333] | 2.889 [1.847, 3.931] | 3.333 [2.616, 4.051] | 1.000 [1.000, 1.000] |

#### Join detail

| cell | join | beat | boundary support | key jump | chord jump | groove cont. | role cont. | head cont. | boundary cont. | cost | tolerance margin | within tolerance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full / 16b / 12-EDO / sample / distribution / section_wise | intro -> theme | 4 [4, 4] | 8.0 [8.0, 8.0] | 0.000 [0.000, 0.000] | 0.333 [-0.384, 1.050] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.33 [-1.10, 1.77] | 0.33 [0.33, 0.33] | 2.667 [1.424, 3.909] | 4.333 [3.091, 5.576] | 1.00 [1.00, 1.00] |
| full / 16b / 12-EDO / sample / distribution / section_wise | theme -> bridge | 8 [8, 8] | 8.0 [8.0, 8.0] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.33 [0.33, 0.33] | 2.667 [2.667, 2.667] | 4.333 [4.333, 4.333] | 1.00 [1.00, 1.00] |
| full / 16b / 12-EDO / sample / distribution / section_wise | bridge -> return | 12 [12, 12] | 8.0 [8.0, 8.0] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.33 [-1.10, 1.77] | 0.00 [0.00, 0.00] | 0.33 [-1.10, 1.77] | 0.33 [0.33, 0.33] | 3.000 [1.566, 4.434] | 4.000 [2.566, 5.434] | 1.00 [1.00, 1.00] |
| full / 32b / 12-EDO / sample / distribution / section_wise | intro -> theme | 8 [8, 8] | 8.0 [8.0, 8.0] | 0.000 [0.000, 0.000] | 0.167 [-0.550, 0.884] | 0.33 [-1.10, 1.77] | 0.00 [0.00, 0.00] | 0.33 [-1.10, 1.77] | 0.33 [0.33, 0.33] | 3.167 [1.924, 4.409] | 3.833 [2.591, 5.076] | 1.00 [1.00, 1.00] |
| full / 32b / 12-EDO / sample / distribution / section_wise | theme -> bridge | 16 [16, 16] | 8.0 [8.0, 8.0] | 0.000 [0.000, 0.000] | 0.167 [-0.550, 0.884] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.33 [-1.10, 1.77] | 0.33 [0.33, 0.33] | 3.167 [1.924, 4.409] | 3.833 [2.591, 5.076] | 1.00 [1.00, 1.00] |
| full / 32b / 12-EDO / sample / distribution / section_wise | bridge -> return | 24 [24, 24] | 8.0 [8.0, 8.0] | 0.000 [0.000, 0.000] | 0.167 [-0.550, 0.884] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.33 [-1.10, 1.77] | 0.33 [0.33, 0.33] | 2.333 [0.899, 3.768] | 4.667 [3.232, 6.101] | 1.00 [1.00, 1.00] |

The planner enforces `max_cost` while building the section graphs, so a run that completes always has every join within tolerance; a join that cannot satisfy it is a section/join failure, listed below, not a silent pass.

### Section-local failures

No section-wise run failed.

### Limitations of this comparison

- **Not a quality ranking.** Section-wise generation succeeding does not make it better; the tables above are trade-offs (runtime, memory, success rate, tension and path statistics, join continuity), each with its own direction.
- Section-wise Method A solves each section independently against shared boundary marginals, so it approximates, and can differ from, the joint monolithic bridge; entropy and MAP scores are not comparable one-to-one.
- Tension error is a proxy against hand-authored arcs, not a listening result; the monolithic planner's own target is a single whole-song arc, which is why both are also scored on the shared section plan.
- Per-section runtime comes from timing the public graph-build and bridge-solve calls; boundary planning, joint path extraction and join validation are reported once as `section_wise_overhead_s`. Per-section peak memory is not available.
- Cells are small (few seeds, short horizons, one EDO, one meter) and the placeholder prior is not trained. A restricted 4/4 style is used because short sections cannot preserve the shared boundary under the default mixed 4/4-5/4-7/4 style; that is a MUS-16 limitation, recorded in the docs, not hidden here.

## Ablations: paired differences vs `full` (same seeds)

No ablation cell has a matching `full` cell yet.

## Not measured / deferred

- Endpoint-distribution evaluation is a correctness/fidelity check of Method A at small horizons (8/16 beats here), not a long-form quality benchmark; calibration compares repeated samples from one solved bridge and says nothing about MAP paths.
- Section-wise SB with soft stitching (MUS-16) is evaluated on the cells above only; see `## Section-wise SB evaluation` for what that does and does not show.
- Method B start-middle-return planning (MUS-18): not implemented in `plans.py`; no A/B comparison.
- Factorized corpus NGramPrior (MUS-26) and a trained neural prior (MUS-27): `trained_prior` reports an explicit `prior_unavailable` failure unless a bundle is configured.
- Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`), decoder-density ablation, ordinary Markov/Viterbi baselines, and the musical-structure metrics (cadence, motif recurrence, voice leading, groove continuity) are Milestone 2.
- Listening-test package: Milestone 3.
