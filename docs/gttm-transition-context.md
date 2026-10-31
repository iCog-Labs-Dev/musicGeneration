# GTTM transition context in sparse graph scoring

## Context contract

Production GTTM features are registered in `FEATURE_REGISTRY`. Each
`GTTMFeatureSpec` declares one of these requirements:

- `current`: only the candidate/current state is required.
- `previous_current`: the scored transition states are required.
- `previous_current_right`: the transition and its following state are required.

For a scoring call `prev_state=A, next_state=B`, `TransitionWindow.right_state=C`
represents the musical triple `A -> B -> C`. `TransitionWindow.left_state` is
reserved for future features and is not consumed by the current registry.

## Selected tonal metric

Scoring helpers accept a keyword-only `tonal_metric`. The existing seven positional arguments of registered feature functions remain valid. The metric is an executable dependency; `TransitionWindow` continues to describe neighboring states only.

`run_method_a` passes the selected metric to graph construction. The same choice reaches default candidate root proposals, endpoint-distance pruning, initial edge scoring, retained-edge rescoring, raw feature values, and weighted contributions. Application tension diagnostics receive it separately. `Prior.logp_next` keeps its existing signature; surrounding structural-scoring functions carry the metric.

Omitting the metric uses the existing formulas and caches. Custom metrics bypass shared default distance caches and need not be hashable. Nearest-root ties remain deterministic. Cadence scoring still tests root distance against exactly `1`, so changing the metric's scale can change cadence decisions.

See the [component extension guide](extending-components.md) for the metric contract and selection points, and [ADR-007](decisions/ADR-007-pluggable-component-protocols.md) for ownership and compatibility decisions.

## Chosen strategy: bounded two-pass rescoring

Sparse expansion first constructs legal graph support using the existing
pairwise scoring path and applies `k_max` and `d_max`. After every retained layer
is known, production performs a second batched scoring pass:

1. For every retained edge `A -> B`, find retained outgoing edges `B -> C_i`.
2. Create one aligned `PriorQuery(A, B)` and
   `TransitionWindow(right_state=C_i)` per retained successor.
3. Score the aligned query/window batch through
   `calculate_transition_score_breakdowns`.
4. Uniformly average successor-specific raw and weighted contributions into the
   final first-order score for `A -> B`.
5. For a terminal edge with no following layer, use no right context; right-only
   features contribute zero by definition.

Uniform successor averaging is a documented bounded approximation. It asks
whether a retained edge generally admits good retained continuations, without
selecting an optimistic successor or introducing a circular dependency on the
Schrodinger Bridge probabilities.

## Legality and complexity

The second pass changes weights only. It creates no state and no edge, so all
candidate legality decisions and the original graph support are preserved.

With `E` retained edges and retained outdegree bounded by `d_max`, at most
`E * d_max` windowed scoring cases are produced. State and edge support remain
bounded by the existing `k_max` and `d_max` rules.

## Diagnostics

`SparseGraph.edge_diagnostics_by_time[t][i]` aligns one-to-one with
`SparseGraph.edges_by_time[t][i]`. Each record contains raw named feature
values, weighted named contributions, the GTTM score and energy, prior/data
terms, final edge weight, supplied right contexts, and the context strategy.

For every record:

```text
gttm_score = sum(weighted_feature_contributions)
gttm_energy = -gttm_score
final_log_weight = data_contribution + gttm_contribution
```

`MethodAPlanResult.path_edge_diagnostics` exposes the aligned retained-edge
records for the final MAP or sampled path. `SparseGraph.inactive_feature_names`
reports registered features that never had a non-zero raw contribution in the
retained production graph.
