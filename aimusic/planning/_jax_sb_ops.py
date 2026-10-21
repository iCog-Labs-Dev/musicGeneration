"""Lazily loaded JAX kernels; no optional imports at module import time."""

import importlib
from functools import lru_cache
from types import SimpleNamespace
from typing import Any


@lru_cache(maxsize=1)
def load_jax_ops() -> Any:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    x64 = getattr(jax, "enable_x64", None)
    if x64 is None:
        x64 = importlib.import_module("jax.experimental").enable_x64

    def grouped(values, groups, size):
        finite = jnp.isfinite(values)
        maxima = jax.ops.segment_max(values, groups, num_segments=size)
        shifts = jnp.where(finite, values - maxima[groups], -jnp.inf)
        sums = jax.ops.segment_sum(jnp.exp(shifts), groups, num_segments=size)
        result = jnp.where(sums > 0, maxima + jnp.log(sums), -jnp.inf)
        minimum = jnp.min(jnp.where(finite, shifts, jnp.inf), initial=jnp.inf)
        return result, minimum

    @jax.jit
    def reduce(values):
        finite = jnp.isfinite(values)
        maximum = jnp.max(values, initial=-jnp.inf)
        shifts = jnp.where(finite, values - maximum, -jnp.inf)
        total = jnp.sum(jnp.exp(shifts))
        result = jnp.where(total > 0, maximum + jnp.log(total), -jnp.inf)
        minimum = jnp.min(jnp.where(finite, shifts, jnp.inf), initial=jnp.inf)
        return result, minimum

    def scatter(weights, messages, cell_ids, gather, targets, out_size):
        # One segment per unique (output, input) pair, followed by one per output.
        # The duplicate-fold reduction deliberately has no underflow check.
        merged, _ = grouped(weights, cell_ids, gather.shape[0])
        return grouped(merged + messages[gather], targets, out_size)

    @jax.jit
    def normalize(values):
        normalizer, minimum = reduce(values)
        return values - normalizer, normalizer, minimum

    return SimpleNamespace(jax=jax, jnp=jnp, x64=x64, reduce=reduce,
                           scatter=jax.jit(scatter, static_argnums=(5,)),
                           normalize=normalize)
