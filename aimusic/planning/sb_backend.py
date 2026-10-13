"""Public numeric backend contract for sparse Schrödinger Bridge.

Solver mathematics in ``aimusic.planning.sb`` consumes only this interface.
Backend selection lives in ``select_sb_backend`` and is not part of the IPF loop.

JAX is an optional future implementation. Do not import JAX from this module.
See ``docs/sb-backend.md`` for padded-array semantics and parity requirements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from aimusic.core.config import SBBackend

_LOG_DTYPE = np.dtype(np.float64)
_INDEX_DTYPE = np.dtype(np.int64)
_MASK_DTYPE = np.dtype(np.bool_)


class SBSolverError(ValueError):
    """Raised when SB solver iterations or backend ops hit invalid numerics."""


class SBBackendError(SBSolverError):
    """Raised when a selected backend is missing, incomplete, or fails its contract."""


@runtime_checkable
class SBNumericBackend(Protocol):
    """Minimal log-space ops required by sparse SB.

    Arrays follow the semantics in ``docs/sb-backend.md``:

    * log values are 1-D ``float64``, finite or ``-inf`` (log-zero)
    * indices are 1-D ``int64``
    * optional boolean masks mark valid entries; ``False`` is excluded
    * empty finite support yields ``-inf`` for reductions and is an error
      for probability normalization
    """

    def logsumexp(
        self,
        values: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "logsumexp",
    ) -> float:
        """Return ``log(sum(exp(values)))`` over finite, unmasked entries."""
        ...

    def scatter_logsumexp(
        self,
        log_weights: np.ndarray,
        gather_from: np.ndarray,
        messages: np.ndarray,
        scatter_to: np.ndarray,
        out_size: int,
        *,
        edge_mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "scatter_logsumexp",
    ) -> np.ndarray:
        """Grouped log-sum-exp after gathering messages onto sparse edges.

        For each output index ``s``::

            out[s] = logsumexp_e  [log_weights[e] + messages[gather_from[e]]]
                     where scatter_to[e] == s and edge_mask[e]

        Groups with no valid edges are ``-inf``.
        """
        ...

    def normalize_log_probs(
        self,
        log_values: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "normalize_log_probs",
    ) -> np.ndarray:
        """Subtract log-sum-exp so unmasked finite mass forms a log-distribution."""
        ...


# Result validators — keep the public error boundary stable.

def _validate_logsumexp_result(
    result: Any, *, context: str = "logsumexp", **kwargs: Any
) -> float:
    if isinstance(result, (bool, np.bool_)):
        raise SBBackendError(f"{context}: logsumexp returned a bool.")
    if not isinstance(result, (int, float, np.floating)):
        raise SBBackendError(
            f"{context}: logsumexp returned {type(result).__name__}, expected float."
        )
    value = float(result)
    if math.isnan(value) or value == float("inf"):
        raise SBSolverError(f"{context}: logsumexp returned invalid value {value!r}.")
    return value


def _validate_scatter_result(
    result: Any,
    *,
    out_size: Optional[int] = None,
    context: str = "scatter_logsumexp",
    **kwargs: Any,
) -> np.ndarray:
    if not isinstance(result, np.ndarray):
        raise SBBackendError(
            f"{context}: scatter_logsumexp returned {type(result).__name__}, "
            "expected np.ndarray."
        )
    if result.ndim != 1:
        raise SBBackendError(
            f"{context}: scatter_logsumexp returned ndim {result.ndim}, expected 1."
        )
    if out_size is not None and result.shape[0] != out_size:
        raise SBBackendError(
            f"{context}: scatter_logsumexp returned shape {result.shape}, "
            f"expected ({out_size},)."
        )
    if result.dtype != _LOG_DTYPE:
        raise SBBackendError(
            f"{context}: scatter_logsumexp returned dtype {result.dtype}, "
            "expected float64."
        )
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise SBSolverError(f"{context}: scatter_logsumexp returned NaN or +inf.")
    return result


def _validate_normalize_result(
    result: Any,
    *,
    log_values: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    context: str = "normalize_log_probs",
    **kwargs: Any,
) -> np.ndarray:
    if not isinstance(result, np.ndarray):
        raise SBBackendError(
            f"{context}: normalize_log_probs returned {type(result).__name__}, "
            "expected np.ndarray."
        )
    if log_values is not None:
        expected_len = np.asarray(log_values).shape[0]
        if result.ndim != 1 or result.shape[0] != expected_len:
            raise SBBackendError(
                f"{context}: normalize_log_probs returned shape {result.shape}, "
                f"expected ({expected_len},)."
            )
    elif result.ndim != 1:
        raise SBBackendError(
            f"{context}: normalize_log_probs returned ndim {result.ndim}, expected 1."
        )
    if result.dtype != _LOG_DTYPE:
        raise SBBackendError(
            f"{context}: normalize_log_probs returned dtype {result.dtype}, "
            "expected float64."
        )
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise SBSolverError(f"{context}: normalize_log_probs returned NaN or +inf.")
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=_MASK_DTYPE)
        if np.any(np.isfinite(result[~mask_arr])):
            raise SBBackendError(
                f"{context}: normalize_log_probs left finite values at masked positions."
            )
    return result


_RESULT_VALIDATORS = {
    "logsumexp": _validate_logsumexp_result,
    "scatter_logsumexp": _validate_scatter_result,
    "normalize_log_probs": _validate_normalize_result,
}


def invoke_backend(
    backend: SBNumericBackend,
    operation: str,
    /,
    **kwargs: Any,
) -> Any:
    """Call a backend op and translate failures / malformed results into stable errors."""

    try:
        method = getattr(backend, operation)
    except AttributeError as exc:
        raise SBBackendError(
            f"SB backend {type(backend).__name__!r} does not implement {operation}."
        ) from exc
    try:
        result = method(**kwargs)
    except (SBSolverError, SBBackendError):
        raise
    except NotImplementedError as exc:
        raise SBBackendError(
            f"SB backend operation {operation} is not implemented."
        ) from exc
    except Exception as exc:
        raise SBBackendError(
            f"SB backend operation {operation} failed: {exc}"
        ) from exc

    validator = _RESULT_VALIDATORS.get(operation)
    if validator is not None:
        result = validator(result, **kwargs)
    return result


def select_sb_backend(selection: SBBackend) -> SBNumericBackend:
    """Resolve a config enum to a backend instance. Not part of solver math."""

    if selection is SBBackend.NUMPY:
        return NumpySBBackend()
    if selection is SBBackend.JAX:
        return JaxSBBackend()
    raise SBBackendError(f"Unsupported SB backend: {selection!r}")


def _require_1d(name: str, values: np.ndarray) -> None:
    if values.ndim != 1:
        raise SBSolverError(f"{name} must be a 1D array.")


def _as_log_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=_LOG_DTYPE)
    _require_1d(name, array)
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
        raise SBSolverError(
            f"{name} contains NaN or +inf; only finite values and -inf are allowed."
        )
    return array


def _as_index_array(
    name: str, values: np.ndarray, *, length: int, upper: int
) -> np.ndarray:
    array = np.asarray(values, dtype=_INDEX_DTYPE)
    _require_1d(name, array)
    if array.size != length:
        raise SBSolverError(f"{name} must have length {length}.")
    if upper <= 0:
        if array.size:
            raise SBSolverError(f"{name} cannot index an empty dimension.")
        return array
    if np.any(array < 0) or np.any(array >= upper):
        raise SBSolverError(f"{name} contains out-of-range indices.")
    return array


def _as_edge_indices(
    name: str,
    values: np.ndarray,
    *,
    edge_count: int,
    dim_size: int,
    dim_name: str,
) -> np.ndarray:
    if dim_size == 0 and edge_count:
        raise SBSolverError(f"{name} cannot index empty {dim_name}.")
    return _as_index_array(name, values, length=edge_count, upper=dim_size)


def _as_mask(
    name: str, mask: Optional[np.ndarray], *, length: int
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    array = np.asarray(mask, dtype=_MASK_DTYPE)
    _require_1d(name, array)
    if array.size != length:
        raise SBSolverError(f"{name} must have length {length}.")
    return array


def logsumexp_1d(
    values: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    underflow_floor: Optional[float] = None,
    context: str = "logsumexp",
) -> float:
    """Reference 1-D log-sum-exp used by NumPy and dense oracles."""

    finite_mask = np.isfinite(values)
    if mask is not None:
        finite_mask = finite_mask & mask
    if not np.any(finite_mask):
        return float("-inf")
    finite_values = values[finite_mask]
    max_value = float(np.max(finite_values))
    shifts = finite_values - max_value
    if underflow_floor is not None:
        min_shift = float(np.min(shifts))
        if min_shift < underflow_floor:
            raise SBSolverError(
                f"{context} encountered shift below underflow floor: "
                f"min_shift={min_shift:.6g}, floor={underflow_floor:.6g}."
            )
    return float(max_value + np.log(np.sum(np.exp(shifts))))


def _python_logsumexp(
    values: np.ndarray,
    *,
    underflow_floor: Optional[float] = None,
    context: str = "logsumexp",
) -> float:
    """Independent pure-Python log-sum-exp used by ``DenseReferenceBackend``.

    Deliberately does *not* call ``logsumexp_1d`` so that a bug in the NumPy
    helper cannot silently pass backend parity.
    """

    arr = np.asarray(values, dtype=_LOG_DTYPE)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return float("-inf")
    finite_vals = arr[finite_mask]
    m = float(np.max(finite_vals))
    shifts = finite_vals - m
    if underflow_floor is not None:
        min_shift = float(np.min(shifts))
        if min_shift < underflow_floor:
            raise SBSolverError(
                f"{context} encountered shift below underflow floor: "
                f"min_shift={min_shift:.6g}, floor={underflow_floor:.6g}."
            )
    acc = 0.0
    for s in shifts:
        acc += math.exp(float(s))
    return float(m + math.log(acc))


def _apply_mask_as_neg_inf(
    values: np.ndarray, mask: Optional[np.ndarray]
) -> np.ndarray:
    if mask is None:
        return values
    masked = values.copy()
    masked[~mask] = float("-inf")
    return masked


def _prepare_scatter_args(
    log_weights: np.ndarray,
    gather_from: np.ndarray,
    messages: np.ndarray,
    scatter_to: np.ndarray,
    out_size: int,
    edge_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not isinstance(out_size, int) or out_size < 0:
        raise SBSolverError("out_size must be a non-negative int.")
    weights = _as_log_array("log_weights", log_weights)
    msgs = _as_log_array("messages", messages)
    gather = _as_edge_indices(
        "gather_from",
        gather_from,
        edge_count=weights.size,
        dim_size=msgs.size,
        dim_name="messages",
    )
    scatter = _as_edge_indices(
        "scatter_to",
        scatter_to,
        edge_count=weights.size,
        dim_size=out_size,
        dim_name="out_size",
    )
    mask = _as_mask("edge_mask", edge_mask, length=weights.size)
    return weights, msgs, gather, scatter, mask


@dataclass(frozen=True)
class NumpySBBackend:
    """Sparse NumPy implementation of ``SBNumericBackend``."""

    def logsumexp(
        self,
        values: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "logsumexp",
    ) -> float:
        log_values = _as_log_array("values", values)
        log_mask = _as_mask("mask", mask, length=log_values.size)
        return logsumexp_1d(
            log_values,
            mask=log_mask,
            underflow_floor=underflow_floor,
            context=context,
        )

    def scatter_logsumexp(
        self,
        log_weights: np.ndarray,
        gather_from: np.ndarray,
        messages: np.ndarray,
        scatter_to: np.ndarray,
        out_size: int,
        *,
        edge_mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "scatter_logsumexp",
    ) -> np.ndarray:
        weights, msgs, gather, scatter, mask = _prepare_scatter_args(
            log_weights, gather_from, messages, scatter_to, out_size, edge_mask
        )

        # Phase 1: merge duplicate (scatter_to, gather_from) cells with logaddexp.
        merged: dict[tuple[int, int], float] = {}
        for idx in range(weights.size):
            if mask is not None and not bool(mask[idx]):
                continue
            key = (int(scatter[idx]), int(gather[idx]))
            w = float(weights[idx])
            if key in merged:
                merged[key] = float(np.logaddexp(merged[key], w))
            else:
                merged[key] = w

        # Phase 2: add gathered messages (constant per cell) and group by scatter_to.
        grouped: list[list[float]] = [[] for _ in range(out_size)]
        for (scatter_idx, gather_idx), w in merged.items():
            grouped[scatter_idx].append(w + float(msgs[gather_idx]))

        # Phase 3: per-group log-sum-exp with the underflow floor on merged cells.
        result = np.full(out_size, float("-inf"), dtype=_LOG_DTYPE)
        for group_idx, group in enumerate(grouped):
            if group:
                result[group_idx] = logsumexp_1d(
                    np.asarray(group, dtype=_LOG_DTYPE),
                    underflow_floor=underflow_floor,
                    context=f"{context}[group={group_idx}]",
                )
        return result

    def normalize_log_probs(
        self,
        log_values: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "normalize_log_probs",
    ) -> np.ndarray:
        values = _as_log_array("log_values", log_values)
        log_mask = _as_mask("mask", mask, length=values.size)
        working = _apply_mask_as_neg_inf(values, log_mask)
        normalizer = logsumexp_1d(
            working,
            underflow_floor=underflow_floor,
            context=context,
        )
        if not np.isfinite(normalizer):
            raise SBSolverError(f"{context} has empty finite support.")
        normalized = working - normalizer
        if log_mask is not None:
            normalized = normalized.copy()
            normalized[~log_mask] = float("-inf")
        return normalized


@dataclass(frozen=True)
class DenseReferenceBackend:
    """Dense log-kernel oracle for tiny graphs.

    Each scatter builds a dense ``(out_size, message_size)`` log-matrix, merging
    duplicate edges with ``logaddexp``, then reduces with an *independent*
    pure-Python log-sum-exp (``_python_logsumexp``). Normalization is also
    implemented directly rather than delegating to ``NumpySBBackend``.

    Intended only as a correctness reference for small randomized graphs.
    """

    def logsumexp(
        self,
        values: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "logsumexp",
    ) -> float:
        arr = _as_log_array("values", values)
        log_mask = _as_mask("mask", mask, length=arr.size)
        if log_mask is not None:
            arr = _apply_mask_as_neg_inf(arr, log_mask)
        return _python_logsumexp(
            arr, underflow_floor=underflow_floor, context=context
        )

    def scatter_logsumexp(
        self,
        log_weights: np.ndarray,
        gather_from: np.ndarray,
        messages: np.ndarray,
        scatter_to: np.ndarray,
        out_size: int,
        *,
        edge_mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "scatter_logsumexp",
    ) -> np.ndarray:
        weights, msgs, gather, scatter, mask = _prepare_scatter_args(
            log_weights, gather_from, messages, scatter_to, out_size, edge_mask
        )

        log_kernel = np.full((out_size, msgs.size), float("-inf"), dtype=_LOG_DTYPE)
        for edge_idx, (src, dst, weight) in enumerate(zip(scatter, gather, weights)):
            if mask is not None and not bool(mask[edge_idx]):
                continue
            log_kernel[int(src), int(dst)] = np.logaddexp(
                log_kernel[int(src), int(dst)],
                float(weight),
            )
        combined = log_kernel + msgs[np.newaxis, :]
        result = np.full(out_size, float("-inf"), dtype=_LOG_DTYPE)
        for row_idx, row in enumerate(combined):
            result[row_idx] = _python_logsumexp(
                row,
                underflow_floor=underflow_floor,
                context=f"{context}[row={row_idx}]",
            )
        return result

    def normalize_log_probs(
        self,
        log_values: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        underflow_floor: Optional[float] = None,
        context: str = "normalize_log_probs",
    ) -> np.ndarray:
        values = _as_log_array("log_values", log_values)
        log_mask = _as_mask("mask", mask, length=values.size)
        working = _apply_mask_as_neg_inf(values, log_mask)
        finite_mask = np.isfinite(working)
        if not np.any(finite_mask):
            raise SBSolverError(f"{context} has empty finite support.")
        finite_vals = working[finite_mask]
        m = float(np.max(finite_vals))
        shifts = finite_vals - m
        if underflow_floor is not None:
            min_shift = float(np.min(shifts))
            if min_shift < underflow_floor:
                raise SBSolverError(
                    f"{context} encountered shift below underflow floor: "
                    f"min_shift={min_shift:.6g}, floor={underflow_floor:.6g}."
                )
        acc = 0.0
        for s in shifts:
            acc += math.exp(float(s))
        normalizer = m + math.log(acc)
        normalized = working - normalizer
        if log_mask is not None:
            normalized = normalized.copy()
            normalized[~log_mask] = float("-inf")
        return normalized


class JaxSBBackend:
    """Optional JAX backend placeholder. Not implemented in this ticket."""

    def __init__(self) -> None:
        raise SBBackendError(
            "SBBackend.JAX is not implemented. Implement padded scatter "
            "log-sum-exp per docs/sb-backend.md before enabling JAX parity."
        )


def backend_array_contract() -> Mapping[str, str]:
    """Stable shape/dtype notes for tests and JAX implementers."""

    return {
        "log_dtype": "float64",
        "index_dtype": "int64",
        "mask_dtype": "bool",
        "log_rank": "1",
        "missing_log_value": "-inf",
        "invalid_log_values": "NaN and +inf",
        "empty_reduction": "-inf",
        "padding": "trailing edges with edge_mask=False; padded log_weights should be -inf",
        "node_padding": "pad states to k_max with node mask; padded logs are -inf",
    }
