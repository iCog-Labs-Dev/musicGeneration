# Schrödinger Bridge numeric backend

The sparse SB solver in [`aimusic/planning/sb.py`](../aimusic/planning/sb.py) is written only against [`SBNumericBackend`](../aimusic/planning/sb_backend.py). Backend **selection** (`SBConfig.backend_selection` → `select_sb_backend`) happens before the scaling loop. Inject a backend into `solve_sb(..., backend=...)` to swap NumPy, a dense oracle, or a test double without changing solver orchestration.

JAX is **optional** and **not implemented** in this ticket. `SBBackend.JAX` raises the stable `SBBackendError`. Do not import JAX from planning modules.

## Required operations

| Method | Role |
| --- | --- |
| `logsumexp` | 1-D log-sum-exp over finite, unmasked entries |
| `scatter_logsumexp` | gather messages onto edges, add log-kernel weights, scatter log-sum-exp by group |
| `normalize_log_probs` | subtract log-sum-exp so unmasked finite mass is a log-distribution |

`scatter_logsumexp` is the sparse primitive used for both backward (`reduce_by_source`) and forward (`reduce_by_target`) messages:

```text
out[s] = logsumexp_{e : scatter_to[e] = s and edge_mask[e]}
         (log_weights[e] + messages[gather_from[e]])
```

Empty groups are `-inf`. Duplicate edges to the same `(scatter_to, gather_from)` pair must combine with log-sum-exp (the dense oracle uses `logaddexp` on a kernel matrix).

## Array contract (NumPy now, padded JAX later)

| Field | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| log values (`values`, `log_weights`, `messages`, `log_values`) | `(N,)` | `float64` | finite or `-inf` only |
| `gather_from`, `scatter_to` | `(E,)` | `int64` | in-range for the gathered / scattered axis |
| `mask` / `edge_mask` | `(N,)` or `(E,)` | `bool` | `True` = valid; `None` means all valid |
| `scatter_logsumexp` output | `(out_size,)` | `float64` | `-inf` where a group has no valid edges |

`backend_array_contract()` in code mirrors this table, including edge padding (`edge_mask=False`) and future node padding to `k_max`.

**`-inf`:** log-zero. It is excluded from the max used to stabilize log-sum-exp. If every contributing entry is `-inf` or masked, the reduction is `-inf`.

**NaN and `+inf`:** invalid. Backends must raise `SBSolverError`.

**Masking / padding (JAX):** pad edge arrays to a static `E_max`. Trailing (and any inactive) slots set `edge_mask=False` and should store `-inf` weights. Masked slots must not affect log-sum-exp, normalization, or underflow-floor checks. Layer sizes still vary; a padded JAX backend should also pad **states** per layer to `k_max` with a node mask, keeping padded node logs at `-inf`.

**Underflow floor:** applied to relative shifts `value - max(finite unmasked values)`
**after** duplicate `(scatter_to, gather_from)` cells have been merged with
log-sum-exp. A shift below `SBConfig.log_underflow_floor` is `SBSolverError`.

`NumpySBBackend` and `DenseReferenceBackend` both follow this merge-first
policy. A future JAX backend must match it — a two-pass implementation that
checks the floor before folding duplicates will disagree with the current
backends on inputs like:


**Normalization:** `normalize_log_probs` raises `SBSolverError` if unmasked finite support is empty. Masked positions remain `-inf` in the output.

Unexpected backend exceptions (`NotImplementedError`, `RuntimeError`, missing methods) are translated by `invoke_backend` into `SBBackendError` (a `SBSolverError`). Numeric contract violations stay `SBSolverError`.

## Dense reference oracle

`DenseReferenceBackend` rebuilds a dense `(out_size, message_size)` log-kernel for each scatter. Use it only on tiny graphs. Sparse NumPy and this oracle must match on randomized reachable graphs within a tight tolerance (see `tests/test_sb_backend.py`).

## Required before implementing JAX parity

1. **Keep JAX optional.** Guard the import; planning must still import when JAX is absent. Do not add a static `import jax` to `sb.py`.
2. **Implement the three ops** with the table above, including `edge_mask` / node masks. Match NumPy on the existing parity tests (`float64` or documented `float32` tolerances).
3. **JIT-friendly scatter.** Replace Python grouping loops with a padded segmented log-sum-exp (sort-by-group, or `jax.ops.segment_max` + masked exp). Gather must live in the backend, not in hidden NumPy indexing inside the IPF loop if the loop itself is to be JITed.
4. **Static shapes.** Pad edges to `d_max * k_max` (or a documented `E_max`) and nodes to `k_max`. `out_size` and message length must be those padded sizes; masks drop padding.
5. **No new solver math.** `solve_sb` already accepts an injected backend. Add `JaxSBBackend` behind `select_sb_backend(SBBackend.JAX)` only after ops pass the dense-oracle parity tests.
6. **Errors.** Raise `SBSolverError` for numeric contract failures; let `invoke_backend` wrap implementation gaps as `SBBackendError`.
7. **Tests.** Re-run NumPy vs dense vs JAX on the same tiny randomized graphs; add a skip if JAX is not installed. Do not claim parity until those tests pass.

The current `JaxSBBackend` exists only so selection has a named, documented failure mode.
