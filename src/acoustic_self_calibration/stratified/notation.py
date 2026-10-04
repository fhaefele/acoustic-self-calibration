from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixRankDiagnostics:
    """Scale-aware SVD diagnostics for one named algebraic matrix."""

    shape: tuple[int, int]
    singular_values: np.ndarray
    tolerance: float
    effective_rank: int
    condition_number: float
    expected_rank: int | None = None


def _as_finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def compaction_matrix(count: int) -> np.ndarray:
    """Return C=[-1;I] so C.T @ v subtracts entry 0 from all others."""
    if count < 2:
        raise ValueError("count must be at least 2")
    matrix = np.vstack(
        [
            -np.ones((1, count - 1), dtype=float),
            np.eye(count - 1, dtype=float),
        ]
    )
    matrix.setflags(write=False)
    return matrix


def reference_arrivals_from_ranges(
    ranges_m: np.ndarray,
    *,
    reference_receiver: int = 0,
) -> np.ndarray:
    """Convert absolute receiver-source ranges to meter-valued relative arrivals."""
    ranges = _as_finite_matrix(ranges_m, name="ranges_m")
    if np.any(ranges < 0.0):
        raise ValueError("ranges_m must be non-negative")
    if not 0 <= reference_receiver < ranges.shape[0]:
        raise ValueError("reference_receiver is out of range")
    return ranges - ranges[[reference_receiver], :]


def apply_arrival_gauge(
    relative_arrivals_m: np.ndarray,
    gauge_shifts_m: np.ndarray,
) -> np.ndarray:
    """Apply the event-wise gauge f'_ij=f_ij+q_j."""
    arrivals = _as_finite_matrix(relative_arrivals_m, name="relative_arrivals_m")
    shifts = np.asarray(gauge_shifts_m, dtype=float).reshape(-1)
    if shifts.shape != (arrivals.shape[1],) or not np.all(np.isfinite(shifts)):
        raise ValueError("gauge_shifts_m must be finite with shape (events,)")
    return arrivals + shifts[None, :]


def restore_offsets_from_gauge(
    shifted_offsets_m: np.ndarray,
    gauge_shifts_m: np.ndarray,
) -> np.ndarray:
    """Return original-gauge offsets from o'_j=o_j+q_j."""
    offsets = np.asarray(shifted_offsets_m, dtype=float).reshape(-1)
    shifts = np.asarray(gauge_shifts_m, dtype=float).reshape(-1)
    if offsets.shape != shifts.shape:
        raise ValueError("shifted_offsets_m and gauge_shifts_m must have the same shape")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(shifts)):
        raise ValueError("offsets and gauge shifts must be finite")
    return offsets - shifts


def corrected_ranges_from_offsets(
    relative_arrivals_m: np.ndarray,
    offsets_m: np.ndarray,
    *,
    negative_tolerance_m: float = 0.0,
) -> np.ndarray:
    """Recover d_ij=f_ij-o_j and reject materially negative ranges."""
    arrivals = _as_finite_matrix(relative_arrivals_m, name="relative_arrivals_m")
    offsets = np.asarray(offsets_m, dtype=float).reshape(-1)
    if offsets.shape != (arrivals.shape[1],) or not np.all(np.isfinite(offsets)):
        raise ValueError("offsets_m must be finite with shape (events,)")
    if negative_tolerance_m < 0.0:
        raise ValueError("negative_tolerance_m must be non-negative")

    ranges = arrivals - offsets[None, :]
    if np.min(ranges) < -negative_tolerance_m:
        raise ValueError("offset solution implies materially negative corrected ranges")
    if negative_tolerance_m > 0.0:
        ranges = np.where(ranges < 0.0, 0.0, ranges)
    return ranges


def modified_squared_range_matrix(
    relative_arrivals_m: np.ndarray,
    offsets_m: np.ndarray,
) -> np.ndarray:
    """Return G_ij(o)=f_ij^2-2*f_ij*o_j."""
    arrivals = _as_finite_matrix(relative_arrivals_m, name="relative_arrivals_m")
    offsets = np.asarray(offsets_m, dtype=float).reshape(-1)
    if offsets.shape != (arrivals.shape[1],) or not np.all(np.isfinite(offsets)):
        raise ValueError("offsets_m must be finite with shape (events,)")
    return arrivals * arrivals - 2.0 * arrivals * offsets[None, :]


def compacted_rank_matrix(
    relative_arrivals_m: np.ndarray,
    offsets_m: np.ndarray,
) -> np.ndarray:
    """Return K(o)=C_m.T @ G(o) @ C_n."""
    arrivals = _as_finite_matrix(relative_arrivals_m, name="relative_arrivals_m")
    receiver_compaction = compaction_matrix(arrivals.shape[0])
    event_compaction = compaction_matrix(arrivals.shape[1])
    modified = modified_squared_range_matrix(arrivals, offsets_m)
    return receiver_compaction.T @ modified @ event_compaction


def cross_gram_from_ranges(ranges_m: np.ndarray) -> np.ndarray:
    """Return Q=-0.5*C_m.T@(d**2)@C_n from absolute cross ranges."""
    ranges = _as_finite_matrix(ranges_m, name="ranges_m")
    if np.any(ranges < 0.0):
        raise ValueError("ranges_m must be non-negative")
    receiver_compaction = compaction_matrix(ranges.shape[0])
    event_compaction = compaction_matrix(ranges.shape[1])
    return -0.5 * receiver_compaction.T @ (ranges * ranges) @ event_compaction


def rank_diagnostics(
    matrix: np.ndarray,
    *,
    expected_rank: int | None = None,
    relative_tolerance: float | None = None,
) -> MatrixRankDiagnostics:
    """Return matrix-specific effective-rank and conditioning diagnostics."""
    values = _as_finite_matrix(matrix, name="matrix")
    singular_values = np.linalg.svd(values, compute_uv=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    if relative_tolerance is None:
        relative_tolerance = np.finfo(float).eps * max(values.shape)
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    tolerance = float(relative_tolerance) * leading
    effective_rank = int(np.sum(singular_values > tolerance))

    if singular_values.size == 0 or effective_rank == 0:
        condition_number = float("inf")
    elif effective_rank < min(values.shape):
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    singular_copy = np.array(singular_values, copy=True)
    singular_copy.setflags(write=False)
    return MatrixRankDiagnostics(
        shape=(int(values.shape[0]), int(values.shape[1])),
        singular_values=singular_copy,
        tolerance=tolerance,
        effective_rank=effective_rank,
        condition_number=condition_number,
        expected_rank=expected_rank,
    )
