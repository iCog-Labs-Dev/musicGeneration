# ADR-005: Separate candidate proposal budget from retained outdegree (D_max)

## Status
Accepted

## Date
2026-09-04

## Context
`aimusic.planning.candidates.get_valid_next_states` stopped consuming the
shuffled candidate generator as soon as it had accumulated `d_max` *legal*
states. `aimusic.planning.graph.build_sparse_graph` then scored that
already-`d_max`-sized pool and trimmed it to `d_max` again by score. Because
the pool was never larger than `d_max` to begin with, the second trim was a
no-op: `d_max` was doing two different jobs (a proposal-stopping condition,
and a retained-edge cap), and any higher-quality candidate that would have
appeared later in the shuffle order was never seen or scored. Callers had no
way to search more broadly without also retaining more edges (which changes
the outdegree of the planning graph and the k_max/branching behavior
downstream).

## Decision
- `get_valid_next_states` takes an independent `proposal_budget` (beam width
  over *raw* proposals). It generates -> deduplicates -> validates the full
  budgeted pool, and, when `prior_guided_proposals=True`, batch-scores and
  ranks the legal pool via the prior (still strictly after legality checks --
  scoring never bypasses them). `d_max` is not consulted anywhere in this
  function.
- `proposal_budget` defaults to a fixed constant
  (`DEFAULT_PROPOSAL_BUDGET = 256`), *not* a multiple of `d_max`, so the two
  knobs stay independently tunable. `SBConfig` gained `proposal_budget: int`
  and `prior_guided_proposals: bool` fields.
- `build_sparse_graph` threads `proposal_budget` through, and applies
  `d_max` exactly once: after `calculate_transition_log_weights` scores the
  full (budgeted) candidate pool for each source state, sorted with the
  existing deterministic tie-break (`_state_sort_key`), trimmed to `d_max`.
  This is "D_max controls retained outgoing edges only."
- `CandidateGenerationResult` and `LayerBuildDiagnostics` report `proposed`,
  `legal`, `unique`, and `scored` counts as separate fields end-to-end
  (`LayerBuildDiagnostics.kept_edge_count` / `.retained_edge_count` is the
  "retained" count, and is the only one `d_max` moves).
- Fixed as part of the same change: a source with zero surviving candidates
  no longer triggers a wasteful zero-item batch call into the prior.

## Seed-sensitivity: support variance vs. bridge-sampling variance
These are two independent RNG concerns and REQ-13 asks that reporting keep
them distinguishable:
- **Support variance**: which candidates make it into the (budget-bounded)
  pool at all. Driven entirely by the `candidate_proposal` RNG stream
  (the `key` passed into `get_valid_next_states` / `build_sparse_graph`).
  Fixing that key reproduces an identical graph (support) byte-for-byte;
  varying it, under a tight budget, changes which states/edges are kept.
- **Bridge-sampling variance**: which path is drawn *from* a solved bridge.
  Driven by the separate RNG stream passed to `sample_bridge_path`. It can
  vary a sampled path without touching the graph/support at all -- the same
  `SolvedBridge` produces different trajectories for different
  bridge-sampling keys, and the same key always replays the same trajectory.

See `tests/test_candidate_proposal_budget.py::TestSeedSensitivityDistinguishesSupportFromBridgeVariance`
for a runnable demonstration of both properties, and
`TestTinyVocabularyRecall` for exhaustive-vs-bounded top-k recall on a small
vocabulary (edo=3, single meter, single groove family).

## Benchmark: quality/runtime tradeoff for proposal budgets
`scripts/benchmark_proposal_budget.py` measures average top-8 recall (against
an exhaustive `proposal_budget=8192` search, 10 candidate-proposal seeds) and
wall-clock time per `get_valid_next_states` call, with `prior_guided_proposals=True`,
on the default (full-size) vocabulary from a mid-progression source state
(~3,300 legal transitions). Measured on the CI container (single core,
`NeuralPrior` placeholder scoring):

| proposal_budget | avg top-8 recall | avg ms/call | p95 ms/call |
|-----------------:|------------------:|------------:|-------------:|
|  32 | 0.000 |   3.4 |   5.3 |
|  64 | 0.000 |   6.3 |  10.2 |
| 128 | 0.050 |  11.4 |  15.9 |
| 256 | 0.050 |  20.6 |  27.6 |
| 512 | 0.100 |  37.9 |  45.6 |
| 1024 | 0.200 |  74.6 |  81.2 |
| 2048 | 0.500 | 147.2 | 151.7 |

Takeaways:
- Runtime scales roughly linearly with `proposal_budget` (dominated by
  legality checks + batch scoring over the pool).
- Exact top-8 recall against a ~3,300-candidate legal space climbs slowly
  and needs a large budget to get close to exhaustive; scores cluster in
  many near-ties (see the tiny-vocab score histogram in
  `TestTinyVocabularyRecall`), so exact top-k membership is a demanding
  metric even when the *retained* edges (after `d_max` trims to a handful)
  are of similar quality across budgets.
- The default `DEFAULT_PROPOSAL_BUDGET = 256` favors latency; callers doing
  offline/batch generation, or using a small `d_max`, can raise
  `proposal_budget` (via `SBConfig.proposal_budget`) for meaningfully better
  candidate quality at a roughly proportional runtime cost.

## Alternatives considered
- Derive `proposal_budget` as a fixed multiple of `d_max` (e.g. `8 * d_max`)
  -- rejected, because it re-couples the two knobs and makes "D_max controls
  retained edges only" untrue in spirit even if technically true in code.
- Score every raw proposal (including illegal ones) for ranking -- rejected;
  legality is a hard filter, not a soft ranking signal, so only the legal
  pool is scored.
- Exhaustive (unbounded) generation by default -- rejected for production
  configs with large vocabularies, where the raw combinatorial space can be
  very large; kept as an explicit opt-in via a large `proposal_budget`.

## Consequences
- `get_valid_next_states` and `build_sparse_graph` gained `proposal_budget`
  and `prior_guided_proposals` parameters (both optional, backward
  compatible via `SBConfig` defaults).
- `CandidateGenerationResult.proposed_count` changed meaning: it now reports
  raw (pre-dedup, budget-bounded) proposals rather than
  `len(states) + len(rejections)`. No production caller relied on the old
  value; `tests/test_candidates.py` was updated for the new
  `proposal_budget` parameter.
- `LayerBuildDiagnostics` gained `legal_candidate_count`, `scored_candidate_count`,
  `scored_edge_count`, and `retained_edge_count`.
