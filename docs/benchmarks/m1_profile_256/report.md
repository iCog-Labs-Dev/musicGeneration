# MUS-28 benchmark report: m1_profile_256

- Config: `m1_profile_256` version `1.0.0`, hash `a953c8bbefae`
- Code: commit `084c4a4d55ee` on `feature/req-18-evaluation-suite` (dirty working tree); Python 3.13.16, numpy 2.5.3
- Runs: 2/2 completed, 2 ok, 0 failed/timeout/crashed
- Generation: `method_a_singleton_endpoints`, single pass. `full` uses the **placeholder** structured prior (a deterministic heuristic, not a trained or corpus-backed model).
- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, or path modes. No pass/fail thresholds are defined yet, so nothing here is a quality claim.

## Success and failure

| cell | ok/runs | failure rate [Wilson 95%] | failure kinds |
|---|---|---|---|
| full / 256b / 12-EDO / sample | 1/1 | 0.00 [0.00, 0.79] | - |
| no_prior / 256b / 12-EDO / sample | 1/1 | 0.00 [0.00, 0.79] | - |

## Runtime and memory

| cell | wall s | median s | p95 s | peak RSS MB (process) | RSS growth MB (run only) |
|---|---|---|---|---|---|
| full / 256b / 12-EDO / sample | 339.0 (n=1) | 339.0 | 339.0 | 227 (n=1) | 194 (n=1) |
| no_prior / 256b / 12-EDO / sample | 196.6 (n=1) | 196.6 | 196.6 | 228 (n=1) | 195 (n=1) |

## Graph support

| cell | layer width | legal/proposed | unique/legal | retained/raw edges | unique cands | support sd (seeds) | inactive GTTM features |
|---|---|---|---|---|---|---|---|
| full / 256b / 12-EDO / sample | 63.3 (n=1) | 0.907 (n=1) | 0.029 (n=1) | 0.005 (n=1) | 110,062 | - | 1.0 (n=1) |
| no_prior / 256b / 12-EDO / sample | 63.3 (n=1) | 0.905 (n=1) | 0.030 (n=1) | 0.005 (n=1) | 110,778 | - | 1.0 (n=1) |

`scored/legal` is 0 unless `prior_guided_proposals` is enabled. Prune/reject reason counts are in the raw `metrics.graph.prune_reason_counts` fields. Registered GTTM features with no non-zero activation on any retained edge: `beat_position_validity` (2 runs). Intervals on fractions can overshoot [0, 1] (t-interval).

## Schrodinger bridge

| cell | converged rate | iterations | effective entropy |
|---|---|---|---|
| full / 256b / 12-EDO / sample | 1.00 (n=1) | 1.0 (n=1) | 0.506 (n=1) |
| no_prior / 256b / 12-EDO / sample | 1.00 (n=1) | 1.0 (n=1) | 0.296 (n=1) |

With singleton endpoints the bridge is degenerate (about one iteration); see Deferred.

## Path, tension and GTTM

| cell | length ok | distinct states | chord change rate | tension MAE | tension shape corr | mean GTTM score/edge |
|---|---|---|---|---|---|---|
| full / 256b / 12-EDO / sample | 1.00 (n=1) | 0.793 (n=1) | 0.074 (n=1) | 0.186 (n=1) | 0.071 (n=1) | 8.564 (n=1) |
| no_prior / 256b / 12-EDO / sample | 1.00 (n=1) | 0.895 (n=1) | 0.164 (n=1) | 0.199 (n=1) | -0.061 (n=1) | 8.695 (n=1) |

`mean GTTM score/edge` uses the per-family feature weights, not `lambda_gttm`, so it is comparable between `full` and `no_gttm`.

## Ablations: paired differences vs `full` (same seeds)

| cell | metric | paired n | delta vs full [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| no_prior / 256b / 12-EDO / sample | runtime.wall_s | 1 | -142.387 (n=1) | - |
| no_prior / 256b / 12-EDO / sample | tension.mean_absolute_error | 1 | 0.013 (n=1) | - |
| no_prior / 256b / 12-EDO / sample | tension.shape_correlation | 1 | -0.132 (n=1) | - |
| no_prior / 256b / 12-EDO / sample | gttm.mean_gttm_score | 1 | 0.131 (n=1) | - |
| no_prior / 256b / 12-EDO / sample | path.distinct_state_fraction | 1 | 0.102 (n=1) | - |
| no_prior / 256b / 12-EDO / sample | path.chord_change_rate | 1 | 0.090 (n=1) | - |

With few seeds these intervals are wide; read them as descriptive, not as significance tests.

## Not measured / deferred

- True distribution-to-distribution Method A (MUS-15): generation still collapses to one start and one terminal state, so SB convergence is trivial and endpoint-marginal error, residuals and calibration are **not measured**.
- Method B start-middle-return planning (MUS-18): not implemented in `plans.py`; no A/B comparison.
- Section-wise SB with soft stitching (MUS-16): not implemented; no monolithic vs section-wise comparison.
- Factorized corpus NGramPrior (MUS-26) and a trained neural prior (MUS-27): `trained_prior` reports an explicit `prior_unavailable` failure unless a bundle is configured.
- Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`), decoder-density ablation, ordinary Markov/Viterbi baselines, and the musical-structure metrics (cadence, motif recurrence, voice leading, groove continuity) are Milestone 2.
- Listening-test package: Milestone 3.
