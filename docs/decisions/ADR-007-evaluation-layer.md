# ADR-007: Evaluation suite is a separate layer downstream of generation

## Status
Accepted

## Date
2026-10-11

## Context
MUS-28 (REQ-18) asks for a reproducible long-form evaluation, ablation and listening-test suite.
Evaluation code that lives inside the planning/scoring modules, or that re-derives graph/SB/prior
logic, can silently change what is being measured and makes ablations untrustworthy.

## Decision
* `aimusic.evaluation` only calls public generation APIs (`run_method_a`) and reads existing
  diagnostics. No generation module imports it (test-enforced).
* Ablations are expressed as `RunSpec` knobs that already exist (`PriorWeights.lambda_gttm`, the `prior`
  argument, `SBConfig`), one factor at a time. The one exception is the no-SB baseline, a small
  evaluation-side uniform end-reachable walk over the *same* generated graph
  (`aimusic/evaluation/baselines.py`); it is a baseline, not a second generator.
* Every run executes in a subprocess so memory, timeouts and crashes are per-run, and a failed run is a
  result. Results are versioned JSON with the full spec, git state and a source hash.
* `full` uses the placeholder structured prior and is labelled as such; unavailable priors are reported
  as failures, never substituted.
* Milestone 1 does not wait for MUS-15/16/18 or the NGramPrior; results state the generation method they
  were produced with so they cannot be mistaken for those features.

## Consequences
* Evaluation can be re-run on any commit, and old results stay interpretable.
* Metrics that need unimplemented generation features are explicitly listed as deferred.
* The no-SB baseline ignores edge weights by design; if a weight-aware non-SB baseline (Viterbi/Markov)
  is wanted it is a Milestone 2 addition.
