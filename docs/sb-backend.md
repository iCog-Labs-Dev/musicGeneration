# Schrödinger Bridge numeric backend

The sparse SB solver in [`aimusic/planning/sb.py`](../aimusic/planning/sb.py) is written only against [`SBNumericBackend`](../aimusic/planning/sb_backend.py). Backend **selection** (`SBConfig.backend_selection` → `select_sb_backend`) happens before the scaling loop. Inject a backend into `solve_sb(..., backend=...)` to swap NumPy, a dense oracle, or a test double without changing solver orchestration.

JAX is **optional**. Install CPU support with `pip install -e ".[jax]"`, then
select `SBConfig(backend_selection=SBBackend.JAX)` or inject `JaxSBBackend()`.
Planning imports and NumPy selection never import JAX. An unavailable/broken JAX
installation raises `SBBackendError` with the original exception as its cause.
For accelerator installation, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

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

`NumpySBBackend`, `DenseReferenceBackend`, and `JaxSBBackend` follow this merge-first
policy. A backend implementation that
checks the floor before folding duplicates will disagree with the current
backends on inputs like two duplicate weights of `-11` and a distinct weight
of `0`, with messages `0` and floor `-10.5`. The merged duplicate cell has
weight `-11 + log(2)` (about `-10.307`) and passes; checking each duplicate
before merging would incorrectly reject it.

The current boundary uses host NumPy arrays, including for injected backends.
Inputs are coerced to the documented dtypes; outputs must have the exact
documented dtype and shape. NaN and `+inf` are invalid even in masked slots.
Masked edge indices must still be in range: use a valid dummy index for padding.
Empty reductions return `-inf`; an empty normalization raises `SBSolverError`.
Zero endpoint probability is supported and remains zero in solved marginals.
An eventual device-native or fully JIT-compiled solver requires a separate
array/loop contract; implementing these three operations alone does not provide it.


**Normalization:** `normalize_log_probs` raises `SBSolverError` if unmasked finite support is empty. Masked positions remain `-inf` in the output.

Unexpected backend exceptions (`NotImplementedError`, `RuntimeError`, missing methods) are translated by `invoke_backend` into `SBBackendError` (a `SBSolverError`). Numeric contract violations stay `SBSolverError`.

## Dense reference oracle

`DenseReferenceBackend` rebuilds a dense `(out_size, message_size)` log-kernel for each scatter. Use it only on tiny graphs. Sparse NumPy and this oracle must match on randomized reachable graphs within a tight tolerance (see `tests/test_sb_backend.py`).

## JAX numerical backend and next-stage requirements

The numerical backend uses lazily compiled reductions with scoped x64 enabled.
The caller's global JAX precision setting is preserved. Host validation and
`np.unique` construct sparse cell IDs; JAX merges duplicates, gathers messages,
and performs segmented max/sum reductions. Memory is proportional to edges and
states, with no dense graph matrix. Empty/masked groups stay `-inf`.

The existing solver still runs its Python IPF loop and exchanges NumPy arrays.
Every operation synchronizes its results back to host, also ensuring device
errors are translated before returning. Compilation specializes to array shapes;
changing graph sizes may compile additional kernels. This implementation makes
no acceleration claim, and does not make the solver differentiable or fully JIT-compatible.

Before a device-native solver is implemented:

1. Define a device-array boundary and retain messages on device throughout IPF.
2. Precompute graph topology and pad edges/nodes to documented static sizes,
   carrying masks and valid dummy indices through every operation.
3. Move convergence control and error reporting into a compiled-loop design
   without changing the mathematics or weakening float64/support semantics.
4. Re-run the same parity and failure suite and benchmark against NumPy,
   separating compilation, transfer, and steady-state execution costs.

Run `python -m pytest tests/test_sb.py tests/test_sb_backend.py tests/test_sb_jax.py -q`.
JAX tests skip only when the package is absent; a broken installed runtime fails
the tests. Tests cover shared contract cases, randomized dense/JAX solver parity,
analytic endpoint/marginal checks, conditionals, optional imports, error translation,
and precision-setting preservation. CPU parity does not establish GPU/TPU parity.
The implementation has been validated with JAX/jaxlib 0.4.38 on Windows CPU;
to reproduce that runtime, install `jax==0.4.38` alongside the project extra.

## REQ-12 verification

Run `python -m pytest tests/test_sb.py tests/test_sb_backend.py -q`.
The suite covers primitive reductions, duplicate merging, masks, empty groups,
normalization, malformed results, exception translation, and backend injection.
Seeded feasible tiny graphs compare forward/backward potentials, node/edge
marginals, and bridge conditionals between NumPy and the dense oracle.
Endpoint and flow checks establish feasibility independently of parity.
A complete uniform kernel provides an analytic oracle: joint mass is
`pi0[:, None] * piT[None, :]`, including zero endpoint probabilities.
Incompatible endpoint masses on disconnected components exercise both explicit
non-convergence and the configured solver-error policy.
