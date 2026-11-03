# MUS-28 benchmark report: m1_64_128

- Config: `m1_64_128` version `1.0.0`, hash `9232a310f929`
- Code: commit `084c4a4d55ee` on `feature/req-18-evaluation-suite` (dirty working tree); Python 3.13.16, numpy 2.5.3
- Runs: 80/80 completed, 80 ok, 0 failed/timeout/crashed
- Generation: `method_a_singleton_endpoints`, single pass. `full` uses the **placeholder** structured prior (a deterministic heuristic, not a trained or corpus-backed model).
- Values are `mean [95% t-interval over seeds]`. Cells never mix EDOs, horizons, or path modes. No pass/fail thresholds are defined yet, so nothing here is a quality claim.

## Success and failure

| cell | ok/runs | failure rate [Wilson 95%] | failure kinds |
|---|---|---|---|
| full / 64b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_gttm / 64b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_prior / 64b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_sb / 64b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| full / 128b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_gttm / 128b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_prior / 128b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_sb / 128b / 12-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| full / 64b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_gttm / 64b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_prior / 64b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_sb / 64b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| full / 128b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_gttm / 128b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_prior / 128b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |
| no_sb / 128b / 19-EDO / sample | 5/5 | 0.00 [0.00, 0.43] | - |

## Runtime and memory

| cell | wall s | median s | p95 s | peak RSS MB (process) | RSS growth MB (run only) |
|---|---|---|---|---|---|
| full / 64b / 12-EDO / sample | 76.7 [75.5, 77.9] | 77.0 | 77.5 | 87 [84, 89] | 53 [50, 56] |
| no_gttm / 64b / 12-EDO / sample | 77.2 [75.3, 79.1] | 76.7 | 78.8 | 108 [107, 110] | 75 [73, 76] |
| no_prior / 64b / 12-EDO / sample | 46.3 [44.5, 48.1] | 46.0 | 47.9 | 87 [84, 89] | 53 [50, 56] |
| no_sb / 64b / 12-EDO / sample | 78.7 [78.1, 79.3] | 78.9 | 79.2 | 87 [84, 89] | 53 [50, 56] |
| full / 128b / 12-EDO / sample | 164.4 [163.1, 165.7] | 164.5 | 165.6 | 134 [132, 137] | 101 [99, 103] |
| no_gttm / 128b / 12-EDO / sample | 159.3 [157.7, 160.9] | 159.3 | 160.8 | 193 [189, 198] | 160 [155, 164] |
| no_prior / 128b / 12-EDO / sample | 93.6 [92.5, 94.6] | 93.6 | 94.5 | 136 [133, 139] | 102 [99, 105] |
| no_sb / 128b / 12-EDO / sample | 161.0 [158.3, 163.6] | 160.6 | 163.7 | 134 [132, 136] | 101 [99, 102] |
| full / 64b / 19-EDO / sample | 94.5 [92.2, 96.8] | 94.8 | 96.2 | 87 [84, 89] | 53 [50, 56] |
| no_gttm / 64b / 19-EDO / sample | 93.6 [90.8, 96.4] | 92.9 | 96.4 | 108 [107, 110] | 75 [73, 76] |
| no_prior / 64b / 19-EDO / sample | 50.9 [49.4, 52.3] | 50.5 | 52.4 | 87 [84, 89] | 53 [50, 56] |
| no_sb / 64b / 19-EDO / sample | 97.7 [96.5, 99.0] | 97.9 | 98.6 | 87 [84, 90] | 53 [50, 56] |
| full / 128b / 19-EDO / sample | 192.1 [189.9, 194.2] | 191.4 | 194.1 | 134 [133, 136] | 101 [99, 103] |
| no_gttm / 128b / 19-EDO / sample | 190.4 [187.0, 193.7] | 190.9 | 193.0 | 193 [189, 198] | 160 [155, 164] |
| no_prior / 128b / 19-EDO / sample | 99.6 [98.0, 101.1] | 99.5 | 101.1 | 136 [133, 139] | 102 [100, 105] |
| no_sb / 128b / 19-EDO / sample | 194.8 [192.9, 196.7] | 195.0 | 196.1 | 134 [133, 136] | 101 [99, 103] |

## Graph support

| cell | layer width | legal/proposed | unique/legal | retained/raw edges | unique cands | support sd (seeds) | inactive GTTM features |
|---|---|---|---|---|---|---|---|
| full / 64b / 12-EDO / sample | 60.9 [60.8, 61.1] | 0.902 [0.894, 0.910] | 0.028 [0.028, 0.029] | 0.006 [0.005, 0.006] | 25,565 | 489.4 | 0.6 [-0.1, 1.3] |
| no_gttm / 64b / 12-EDO / sample | 61.0 [60.8, 61.1] | 0.862 [0.858, 0.866] | 0.021 [0.021, 0.022] | 0.011 [0.011, 0.012] | 18,276 | 188.0 | 0.2 [-0.4, 0.8] |
| no_prior / 64b / 12-EDO / sample | 61.0 [60.8, 61.1] | 0.901 [0.893, 0.909] | 0.029 [0.028, 0.029] | 0.005 [0.005, 0.006] | 25,832 | 698.9 | 0.6 [-0.1, 1.3] |
| no_sb / 64b / 12-EDO / sample | 60.9 [60.8, 61.1] | 0.902 [0.894, 0.910] | 0.028 [0.028, 0.029] | 0.006 [0.005, 0.006] | 25,565 | 489.4 | 0.6 [-0.1, 1.3] |
| full / 128b / 12-EDO / sample | 62.5 [62.4, 62.6] | 0.904 [0.901, 0.907] | 0.029 [0.028, 0.029] | 0.005 [0.005, 0.006] | 53,279 | 637.3 | 0.6 [-0.1, 1.3] |
| no_gttm / 128b / 12-EDO / sample | 62.5 [62.4, 62.5] | 0.853 [0.848, 0.859] | 0.020 [0.020, 0.020] | 0.013 [0.013, 0.014] | 34,526 | 596.1 | 0.0 [0.0, 0.0] |
| no_prior / 128b / 12-EDO / sample | 62.5 [62.4, 62.6] | 0.901 [0.896, 0.906] | 0.029 [0.029, 0.029] | 0.005 [0.005, 0.006] | 53,352 | 604.8 | 0.6 [-0.1, 1.3] |
| no_sb / 128b / 12-EDO / sample | 62.5 [62.4, 62.6] | 0.904 [0.901, 0.907] | 0.029 [0.028, 0.029] | 0.005 [0.005, 0.006] | 53,279 | 637.3 | 0.6 [-0.1, 1.3] |
| full / 64b / 19-EDO / sample | 60.9 [60.8, 61.1] | 0.902 [0.894, 0.910] | 0.028 [0.028, 0.029] | 0.006 [0.005, 0.006] | 25,565 | 489.4 | 0.6 [-0.1, 1.3] |
| no_gttm / 64b / 19-EDO / sample | 61.0 [60.8, 61.1] | 0.862 [0.858, 0.866] | 0.021 [0.021, 0.022] | 0.011 [0.011, 0.012] | 18,276 | 188.0 | 0.2 [-0.4, 0.8] |
| no_prior / 64b / 19-EDO / sample | 61.0 [60.8, 61.1] | 0.901 [0.893, 0.909] | 0.029 [0.028, 0.029] | 0.005 [0.005, 0.006] | 25,832 | 698.9 | 0.6 [-0.1, 1.3] |
| no_sb / 64b / 19-EDO / sample | 60.9 [60.8, 61.1] | 0.902 [0.894, 0.910] | 0.028 [0.028, 0.029] | 0.006 [0.005, 0.006] | 25,565 | 489.4 | 0.6 [-0.1, 1.3] |
| full / 128b / 19-EDO / sample | 62.5 [62.4, 62.6] | 0.904 [0.901, 0.907] | 0.029 [0.028, 0.029] | 0.005 [0.005, 0.006] | 53,279 | 637.3 | 0.6 [-0.1, 1.3] |
| no_gttm / 128b / 19-EDO / sample | 62.5 [62.4, 62.5] | 0.853 [0.848, 0.859] | 0.020 [0.020, 0.020] | 0.013 [0.013, 0.014] | 34,526 | 596.1 | 0.0 [0.0, 0.0] |
| no_prior / 128b / 19-EDO / sample | 62.5 [62.4, 62.6] | 0.901 [0.896, 0.906] | 0.029 [0.029, 0.029] | 0.005 [0.005, 0.006] | 53,352 | 604.8 | 0.6 [-0.1, 1.3] |
| no_sb / 128b / 19-EDO / sample | 62.5 [62.4, 62.6] | 0.904 [0.901, 0.907] | 0.029 [0.028, 0.029] | 0.005 [0.005, 0.006] | 53,279 | 637.3 | 0.6 [-0.1, 1.3] |

`scored/legal` is 0 unless `prior_guided_proposals` is enabled. Prune/reject reason counts are in the raw `metrics.graph.prune_reason_counts` fields. Registered GTTM features with no non-zero activation on any retained edge: `beat_position_validity` (38 runs). Intervals on fractions can overshoot [0, 1] (t-interval).

## Schrodinger bridge

| cell | converged rate | iterations | effective entropy |
|---|---|---|---|
| full / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.630 [0.465, 0.794] |
| no_gttm / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 2.124 [1.992, 2.255] |
| no_prior / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.653 [0.436, 0.870] |
| full / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.553 [0.385, 0.721] |
| no_gttm / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 2.352 [2.263, 2.442] |
| no_prior / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.597 [0.528, 0.666] |
| full / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.630 [0.465, 0.794] |
| no_gttm / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 2.124 [1.992, 2.255] |
| no_prior / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.653 [0.436, 0.870] |
| full / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.553 [0.385, 0.721] |
| no_gttm / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 2.352 [2.263, 2.442] |
| no_prior / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 1.0 [1.0, 1.0] | 0.597 [0.528, 0.666] |

With singleton endpoints the bridge is degenerate (about one iteration); see Deferred.

## Path, tension and GTTM

| cell | length ok | distinct states | chord change rate | tension MAE | tension shape corr | mean GTTM score/edge |
|---|---|---|---|---|---|---|
| full / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.903 [0.838, 0.968] | 0.050 [-0.013, 0.113] | 0.184 [0.174, 0.194] | 0.296 [0.237, 0.354] | 8.472 [8.265, 8.678] |
| no_gttm / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.909 [0.893, 0.926] | 0.016 [-0.004, 0.035] | 0.333 [0.317, 0.350] | -0.142 [-0.264, -0.020] | 7.187 [6.955, 7.420] |
| no_prior / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.891 [0.826, 0.955] | 0.062 [-0.033, 0.158] | 0.183 [0.176, 0.191] | 0.263 [0.188, 0.338] | 8.485 [8.403, 8.567] |
| no_sb / 64b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.912 [0.870, 0.955] | 0.050 [-0.016, 0.116] | 0.191 [0.185, 0.197] | 0.252 [0.158, 0.346] | 8.307 [8.149, 8.464] |
| full / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.861 [0.761, 0.961] | 0.062 [0.015, 0.110] | 0.188 [0.182, 0.194] | 0.182 [0.106, 0.259] | 8.567 [8.452, 8.681] |
| no_gttm / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.755 [0.716, 0.793] | 0.005 [-0.004, 0.013] | 0.370 [0.366, 0.375] | -0.240 [-0.380, -0.100] | 7.176 [7.039, 7.313] |
| no_prior / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.864 [0.795, 0.933] | 0.102 [0.040, 0.163] | 0.187 [0.181, 0.192] | 0.156 [0.043, 0.269] | 8.580 [8.517, 8.643] |
| no_sb / 128b / 12-EDO / sample | 1.00 [1.00, 1.00] | 0.872 [0.829, 0.914] | 0.070 [0.013, 0.128] | 0.192 [0.182, 0.202] | 0.134 [0.065, 0.202] | 8.431 [8.351, 8.511] |
| full / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.903 [0.838, 0.968] | 0.050 [-0.013, 0.113] | 0.184 [0.174, 0.194] | 0.296 [0.237, 0.354] | 8.472 [8.265, 8.678] |
| no_gttm / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.909 [0.893, 0.926] | 0.016 [-0.004, 0.035] | 0.333 [0.317, 0.350] | -0.142 [-0.264, -0.020] | 7.187 [6.955, 7.420] |
| no_prior / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.891 [0.826, 0.955] | 0.062 [-0.033, 0.158] | 0.183 [0.176, 0.191] | 0.263 [0.188, 0.338] | 8.485 [8.403, 8.567] |
| no_sb / 64b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.912 [0.870, 0.955] | 0.050 [-0.016, 0.116] | 0.191 [0.185, 0.197] | 0.252 [0.158, 0.346] | 8.307 [8.149, 8.464] |
| full / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.861 [0.761, 0.961] | 0.062 [0.015, 0.110] | 0.188 [0.182, 0.194] | 0.182 [0.106, 0.259] | 8.567 [8.452, 8.681] |
| no_gttm / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.755 [0.716, 0.793] | 0.005 [-0.004, 0.013] | 0.370 [0.366, 0.375] | -0.240 [-0.380, -0.100] | 7.176 [7.039, 7.313] |
| no_prior / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.864 [0.795, 0.933] | 0.102 [0.040, 0.163] | 0.187 [0.181, 0.192] | 0.156 [0.043, 0.269] | 8.580 [8.517, 8.643] |
| no_sb / 128b / 19-EDO / sample | 1.00 [1.00, 1.00] | 0.872 [0.829, 0.914] | 0.070 [0.013, 0.128] | 0.192 [0.182, 0.202] | 0.134 [0.065, 0.202] | 8.431 [8.351, 8.511] |

`mean GTTM score/edge` uses the per-family feature weights, not `lambda_gttm`, so it is comparable between `full` and `no_gttm`.

## Ablations: paired differences vs `full` (same seeds)

| cell | metric | paired n | delta vs full [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| no_gttm / 128b / 12-EDO / sample | runtime.wall_s | 5 | -5.112 [-5.833, -4.391] | yes |
| no_gttm / 128b / 12-EDO / sample | tension.mean_absolute_error | 5 | 0.182 [0.173, 0.191] | yes |
| no_gttm / 128b / 12-EDO / sample | tension.shape_correlation | 5 | -0.422 [-0.588, -0.256] | yes |
| no_gttm / 128b / 12-EDO / sample | gttm.mean_gttm_score | 5 | -1.391 [-1.522, -1.259] | yes |
| no_gttm / 128b / 12-EDO / sample | path.distinct_state_fraction | 5 | -0.106 [-0.180, -0.032] | yes |
| no_gttm / 128b / 12-EDO / sample | path.chord_change_rate | 5 | -0.058 [-0.107, -0.009] | yes |
| no_gttm / 128b / 19-EDO / sample | runtime.wall_s | 5 | -1.714 [-6.407, 2.979] | no |
| no_gttm / 128b / 19-EDO / sample | tension.mean_absolute_error | 5 | 0.182 [0.173, 0.191] | yes |
| no_gttm / 128b / 19-EDO / sample | tension.shape_correlation | 5 | -0.422 [-0.588, -0.256] | yes |
| no_gttm / 128b / 19-EDO / sample | gttm.mean_gttm_score | 5 | -1.391 [-1.522, -1.259] | yes |
| no_gttm / 128b / 19-EDO / sample | path.distinct_state_fraction | 5 | -0.106 [-0.180, -0.032] | yes |
| no_gttm / 128b / 19-EDO / sample | path.chord_change_rate | 5 | -0.058 [-0.107, -0.009] | yes |
| no_gttm / 64b / 12-EDO / sample | runtime.wall_s | 5 | 0.533 [-1.844, 2.911] | no |
| no_gttm / 64b / 12-EDO / sample | tension.mean_absolute_error | 5 | 0.150 [0.126, 0.173] | yes |
| no_gttm / 64b / 12-EDO / sample | tension.shape_correlation | 5 | -0.438 [-0.607, -0.268] | yes |
| no_gttm / 64b / 12-EDO / sample | gttm.mean_gttm_score | 5 | -1.284 [-1.599, -0.970] | yes |
| no_gttm / 64b / 12-EDO / sample | path.distinct_state_fraction | 5 | 0.006 [-0.062, 0.074] | no |
| no_gttm / 64b / 12-EDO / sample | path.chord_change_rate | 5 | -0.034 [-0.110, 0.041] | no |
| no_gttm / 64b / 19-EDO / sample | runtime.wall_s | 5 | -0.866 [-4.632, 2.901] | no |
| no_gttm / 64b / 19-EDO / sample | tension.mean_absolute_error | 5 | 0.150 [0.126, 0.173] | yes |
| no_gttm / 64b / 19-EDO / sample | tension.shape_correlation | 5 | -0.438 [-0.607, -0.268] | yes |
| no_gttm / 64b / 19-EDO / sample | gttm.mean_gttm_score | 5 | -1.284 [-1.599, -0.970] | yes |
| no_gttm / 64b / 19-EDO / sample | path.distinct_state_fraction | 5 | 0.006 [-0.062, 0.074] | no |
| no_gttm / 64b / 19-EDO / sample | path.chord_change_rate | 5 | -0.034 [-0.110, 0.041] | no |
| no_prior / 128b / 12-EDO / sample | runtime.wall_s | 5 | -70.813 [-72.522, -69.104] | yes |
| no_prior / 128b / 12-EDO / sample | tension.mean_absolute_error | 5 | -0.002 [-0.009, 0.006] | no |
| no_prior / 128b / 12-EDO / sample | tension.shape_correlation | 5 | -0.026 [-0.141, 0.089] | no |
| no_prior / 128b / 12-EDO / sample | gttm.mean_gttm_score | 5 | 0.014 [-0.110, 0.137] | no |
| no_prior / 128b / 12-EDO / sample | path.distinct_state_fraction | 5 | 0.003 [-0.133, 0.140] | no |
| no_prior / 128b / 12-EDO / sample | path.chord_change_rate | 5 | 0.039 [-0.047, 0.125] | no |
| no_prior / 128b / 19-EDO / sample | runtime.wall_s | 5 | -92.515 [-94.571, -90.460] | yes |
| no_prior / 128b / 19-EDO / sample | tension.mean_absolute_error | 5 | -0.002 [-0.009, 0.006] | no |
| no_prior / 128b / 19-EDO / sample | tension.shape_correlation | 5 | -0.026 [-0.141, 0.089] | no |
| no_prior / 128b / 19-EDO / sample | gttm.mean_gttm_score | 5 | 0.014 [-0.110, 0.137] | no |
| no_prior / 128b / 19-EDO / sample | path.distinct_state_fraction | 5 | 0.003 [-0.133, 0.140] | no |
| no_prior / 128b / 19-EDO / sample | path.chord_change_rate | 5 | 0.039 [-0.047, 0.125] | no |
| no_prior / 64b / 12-EDO / sample | runtime.wall_s | 5 | -30.377 [-32.679, -28.076] | yes |
| no_prior / 64b / 12-EDO / sample | tension.mean_absolute_error | 5 | -0.001 [-0.018, 0.016] | no |
| no_prior / 64b / 12-EDO / sample | tension.shape_correlation | 5 | -0.033 [-0.142, 0.076] | no |
| no_prior / 64b / 12-EDO / sample | gttm.mean_gttm_score | 5 | 0.014 [-0.136, 0.163] | no |
| no_prior / 64b / 12-EDO / sample | path.distinct_state_fraction | 5 | -0.013 [-0.093, 0.068] | no |
| no_prior / 64b / 12-EDO / sample | path.chord_change_rate | 5 | 0.013 [-0.038, 0.063] | no |
| no_prior / 64b / 19-EDO / sample | runtime.wall_s | 5 | -43.617 [-45.390, -41.844] | yes |
| no_prior / 64b / 19-EDO / sample | tension.mean_absolute_error | 5 | -0.001 [-0.018, 0.016] | no |
| no_prior / 64b / 19-EDO / sample | tension.shape_correlation | 5 | -0.033 [-0.142, 0.076] | no |
| no_prior / 64b / 19-EDO / sample | gttm.mean_gttm_score | 5 | 0.014 [-0.136, 0.163] | no |
| no_prior / 64b / 19-EDO / sample | path.distinct_state_fraction | 5 | -0.013 [-0.093, 0.068] | no |
| no_prior / 64b / 19-EDO / sample | path.chord_change_rate | 5 | 0.013 [-0.038, 0.063] | no |
| no_sb / 128b / 12-EDO / sample | runtime.wall_s | 5 | -3.403 [-5.472, -1.334] | yes |
| no_sb / 128b / 12-EDO / sample | tension.mean_absolute_error | 5 | 0.004 [-0.002, 0.010] | no |
| no_sb / 128b / 12-EDO / sample | tension.shape_correlation | 5 | -0.049 [-0.131, 0.034] | no |
| no_sb / 128b / 12-EDO / sample | gttm.mean_gttm_score | 5 | -0.135 [-0.241, -0.030] | yes |
| no_sb / 128b / 12-EDO / sample | path.distinct_state_fraction | 5 | 0.011 [-0.052, 0.074] | no |
| no_sb / 128b / 12-EDO / sample | path.chord_change_rate | 5 | 0.008 [-0.033, 0.049] | no |
| no_sb / 128b / 19-EDO / sample | runtime.wall_s | 5 | 2.689 [0.702, 4.676] | yes |
| no_sb / 128b / 19-EDO / sample | tension.mean_absolute_error | 5 | 0.004 [-0.002, 0.010] | no |
| no_sb / 128b / 19-EDO / sample | tension.shape_correlation | 5 | -0.049 [-0.131, 0.034] | no |
| no_sb / 128b / 19-EDO / sample | gttm.mean_gttm_score | 5 | -0.135 [-0.241, -0.030] | yes |
| no_sb / 128b / 19-EDO / sample | path.distinct_state_fraction | 5 | 0.011 [-0.052, 0.074] | no |
| no_sb / 128b / 19-EDO / sample | path.chord_change_rate | 5 | 0.008 [-0.033, 0.049] | no |
| no_sb / 64b / 12-EDO / sample | runtime.wall_s | 5 | 2.040 [0.863, 3.217] | yes |
| no_sb / 64b / 12-EDO / sample | tension.mean_absolute_error | 5 | 0.007 [-0.002, 0.017] | no |
| no_sb / 64b / 12-EDO / sample | tension.shape_correlation | 5 | -0.044 [-0.134, 0.047] | no |
| no_sb / 64b / 12-EDO / sample | gttm.mean_gttm_score | 5 | -0.165 [-0.226, -0.103] | yes |
| no_sb / 64b / 12-EDO / sample | path.distinct_state_fraction | 5 | 0.009 [-0.033, 0.052] | no |
| no_sb / 64b / 12-EDO / sample | path.chord_change_rate | 5 | 0.000 [-0.027, 0.027] | no |
| no_sb / 64b / 19-EDO / sample | runtime.wall_s | 5 | 3.242 [0.495, 5.989] | yes |
| no_sb / 64b / 19-EDO / sample | tension.mean_absolute_error | 5 | 0.007 [-0.002, 0.017] | no |
| no_sb / 64b / 19-EDO / sample | tension.shape_correlation | 5 | -0.044 [-0.134, 0.047] | no |
| no_sb / 64b / 19-EDO / sample | gttm.mean_gttm_score | 5 | -0.165 [-0.226, -0.103] | yes |
| no_sb / 64b / 19-EDO / sample | path.distinct_state_fraction | 5 | 0.009 [-0.033, 0.052] | no |
| no_sb / 64b / 19-EDO / sample | path.chord_change_rate | 5 | 0.000 [-0.027, 0.027] | no |

With few seeds these intervals are wide; read them as descriptive, not as significance tests.

## Not measured / deferred

- True distribution-to-distribution Method A (MUS-15): generation still collapses to one start and one terminal state, so SB convergence is trivial and endpoint-marginal error, residuals and calibration are **not measured**.
- Method B start-middle-return planning (MUS-18): not implemented in `plans.py`; no A/B comparison.
- Section-wise SB with soft stitching (MUS-16): not implemented; no monolithic vs section-wise comparison.
- Factorized corpus NGramPrior (MUS-26) and a trained neural prior (MUS-27): `trained_prior` reports an explicit `prior_unavailable` failure unless a bundle is configured.
- Proposal recall (needs an exhaustive-search reference; see `scripts/benchmark_proposal_budget.py`), decoder-density ablation, ordinary Markov/Viterbi baselines, and the musical-structure metrics (cadence, motif recurrence, voice leading, groove continuity) are Milestone 2.
- Listening-test package: Milestone 3.
