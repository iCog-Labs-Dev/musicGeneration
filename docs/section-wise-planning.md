# Section-wise Schrödinger Bridge (REQ-02)

`SectioningStrategy.SECTION_WISE` executes one graph build and one SB solve
for every `PlanningSection`. Single-pass generation retains its existing path.

The UI (`python ui.py`) defaults to section-wise planning with intro, theme,
bridge and return, and 16 beats (four per section). Uncheck **Use section-wise
planning** for single-pass generation. Four-section runs require at least four
beats; section names and solver reports are saved in the downloaded manifest.
The CLI keeps its existing default; pass `--sections` to select section-wise planning.

```powershell
python -m aimusic.app.cli generate --seed 11 --beats 16 --sections intro theme bridge return --max-horizon-per-solve 4 --out ./outputs
```

The existing section planner divides the beat count evenly, assigning remainder
beats to the earlier sections. Each local solver uses times `0..section_length`.
Endpoint beat positions follow cumulative song time, so the meter does not
restart at each section. Names, global start/end, index, boundary level and
target tension arc reach priors through `PriorContext`; `graph_time` is local.
Tension metadata is context, not a new enforced tension objective.

## Boundary contract

Each interior boundary retains a shared distribution over ending candidates,
bounded by `PlanConfig.endpoint_top_k`. The next section's starting preferences
are generated at the same global beat position. Since starting and ending
roles can differ, we transfer those preferences onto ending candidates with a
soft distance kernel:

`q(s) ∝ p_left_end(s) × sum_r p_right_start(r) exp(-stitch_cost(s, r))`.

This uses both endpoint preferences and configurable stitching weights without
an additional graph or solve. The candidate comparison costs O(endpoint_top_k²)
per boundary. The hard tolerance applies to actual transitions, not this
comparison between same-time endpoint preferences. Section-aware priors still
score actual transitions in each section graph.

After building every section graph once, each section calls
`solve_endpoint_bridge()` to validate reachability, joint mass feasibility,
convergence, and realized endpoint marginal errors. Adjacent sections receive
identical shared supports and probabilities, with only local time indices
changed. Independent removal, trimming, renormalization or interior blending
of a shared boundary is rejected with section-local context, even when the
configured fallback permits changes at the outer song endpoints. There is no
cross-section retry loop.

Only after solving do we choose actual connections. Sampling carries the
preceding sampled endpoint into the next solved bridge. MAP uses one Viterbi
pass across all solved transition kernels, allowing following sections to
influence the selected connection. Neither procedure reruns SB. The MAP pass
is linear in retained edges and does not construct a monolithic graph.
`boundary_choices` records the retained distribution and final selected state;
join diagnostics expose support sizes, states and probabilities. Song-level
endpoints honor `endpoint_mode`: distribution mode uses the generated positive
probability distributions, while explicit singleton mode alone calls
`select_singleton_endpoints()`. Interior joins remain shared distributions in
both modes.

On downbeats, the section's authored boundary level is applied. Off downbeats,
the endpoint generator's meter-legal local boundary level is retained; section
metadata still records the requested structural boundary. The planner does not
silently add beats to align sections with bars.

Soft stitching penalizes the transition entering an internal anchor and the
transition leaving it. It does not compare the anchor against itself. Costs
are subtracted from edge log weights before pruning and after context-aware
rescoring, so they influence both retained alternatives and SB probabilities.
An edge touching two joins in a one-beat section is penalized only once.

`PlanConfig.stitching` accepts a frozen `StitchingConfig`:

* `key_weight`: normalized circular pitch-class root distance in the chosen EDO.
* `chord_weight`: mean of normalized root distance and quality mismatch.
* `meter_weight`, `groove_weight`, `role_weight`, `head_weight`: categorical
  mismatch distances (0 or 1).
* `boundary_weight`: absolute boundary-level difference divided by 3, capped at 1.
* `max_cost`: maximum weighted cost for each incoming/outgoing transition.

All weights default to 1; the default limit is 7. Setting a weight to zero
disables that preference. Raising weights increases the soft penalty and may
also require adjusting `max_cost`. Edges exceeding the limit are excluded,
and realized joins are checked again. Structural legality remains enforced by
the existing transition rules. Impossible constraints fail with a section name,
index, time range, stage, and chained cause. This version does not retry,
rebuild graphs or adjust infeasible marginal constraints automatically.

## Results, budgets, and limitations

`run_method_a` returns `MethodAPlanResult` for single-pass runs and
`SectionWisePlanResult` for section-wise runs. The latter exposes
`section_results` (local graphs, solutions, bridges and paths), `joins`, a
combined `path`, and global `path_edge_diagnostics`. Shared endpoints appear
only once in the combined path. Both result types expose `solutions` for
diagnostic consumers; `SBDiagnostics.from_plan` builds manifest summaries.
Each section retains `start_mass`, `end_mass`, and `endpoint_resolution`,
including requested/effective probabilities, removed or trimmed mass,
feasibility checks, realized probabilities and maximum marginal errors.
These validated reports are also included in the per-section manifest.
The section-wise MAP score is the log probability of the best connected path
under the composed solved kernels. Shared boundary probability is counted once,
not multiplied in again at every join. This is not a joint optimization of the
SB marginal distributions themselves. Aggregate entropy likewise counts each
shared layer once, including its nonzero entropy.

`SBConfig.max_horizon_per_solve` caps transitions per graph/solve. It is checked
before graph allocation and in the SB contract. A 16-beat song can therefore
run as four 4-beat solves with a limit of 4, while a 16-beat monolithic solve
fails. This is an explicit horizon budget, not a measured RAM guarantee:
section results retain their graphs for review. No whole-song graph is built.
Keeping more boundary options can increase graph-construction time even though
the solve count stays fixed. Endpoint distance/legality heuristics are cached
per candidate within each layer, preserving scores while avoiding repeated
calculations for targets reached by several sources. The cache is discarded
at the next layer. `endpoint_top_k`, `k_max` and `d_max` bound the search.

Random streams are keyed by section index/name and purpose. Repeating the same
configuration and input `RNGKey` reproduces boundaries and paths; the parent key
advances once. Changing section order or names intentionally changes streams.

Coverage: `tests/test_section_planning.py` tests four-section execution, context,
costs/tolerances, shared endpoints, budgets, failures and 12-/19-EDO replay;
it also checks positive multi-state bridge mass, downstream start preferences,
weight-sensitive boundary probabilities, downstream-aware MAP selection and
entropy without duplicated shared layers. `tests/test_cli_workflow.py` covers
MIDI and manifest export.
