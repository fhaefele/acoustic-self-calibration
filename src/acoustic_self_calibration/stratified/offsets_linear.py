from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .notation import (
    MatrixRankDiagnostics,
    apply_arrival_gauge,
    compacted_rank_matrix,
    corrected_ranges_from_offsets,
    rank_diagnostics,
    restore_offsets_from_gauge,
)

LinearOffsetStatus = Literal["solved", "weakly_identified"]


@dataclass(frozen=True)
class LinearOffsetDiagnostics:
    """Diagnostics for one linearly solvable TDOA offset anchor."""

    dimension: int
    receiver_count: int
    event_count: int
    gauge_attempt_index: int
    gauge_shifts_m: np.ndarray
    normalized_singular_values: np.ndarray
    normalized_rank: int
    normalized_condition_number: float
    minimum_denominator: float
    linear_residual_rms: float
    compacted_rank: MatrixRankDiagnostics
    compacted_tail_rms: float


@dataclass(frozen=True)
class LinearOffsetSolution:
    """Recovered original-gauge offsets and corrected ranges."""

    status: LinearOffsetStatus
    offsets_m: np.ndarray
    shifted_offsets_m: np.ndarray
    corrected_ranges_m: np.ndarray
    diagnostics: LinearOffsetDiagnostics


def build_linear_offset_system(
    shifted_arrivals_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build [A B b] u = 1 for the Kuang-Astrom linear anchor."""
    arrivals = np.asarray(shifted_arrivals_m, dtype=float)
    if arrivals.ndim != 2 or not np.all(np.isfinite(arrivals)):
        raise ValueError("shifted_arrivals_m must be a finite 2-D matrix")
    _, event_count = arrivals.shape
    if event_count < 2:
        raise ValueError("at least two events are required")

    a = np.column_stack(
        [arrivals[:, event] ** 2 - arrivals[:, 0] ** 2 for event in range(1, event_count)]
    )
    b = np.column_stack([-2.0 * arrivals[:, event] for event in range(1, event_count)])
    c = 2.0 * arrivals[:, 0]
    system = np.column_stack([a, b, c])
    return system, a, b, c


def _measurement_scale(arrivals: np.ndarray) -> float:
    values = np.abs(arrivals[np.abs(arrivals) > np.finfo(float).eps])
    if values.size == 0:
        raise ValueError("relative arrivals are all zero; linear offset anchor is degenerate")
    return float(np.median(values))


def _default_gauge_candidates(
    arrivals: np.ndarray,
    *,
    max_attempts: int,
) -> tuple[np.ndarray, ...]:
    event_count = arrivals.shape[1]
    scale = _measurement_scale(arrivals)
    index = np.arange(event_count, dtype=float)
    base = (
        0.65 + 0.37 * index,
        ((-1.0) ** np.arange(event_count)) * (0.8 + 0.29 * index),
        np.linspace(-1.2, 1.4, event_count) + 0.31,
        np.sin(0.7 + 1.1 * index) + 1.4,
        np.cos(0.2 + 0.9 * index) - 1.3,
        np.sin(0.31 + 0.63 * index) - 1.7,
        np.cos(0.91 + 1.37 * index) + 1.8,
    )
    return tuple(scale * pattern for pattern in base[:max_attempts])


def _solve_one_gauge(
    arrivals: np.ndarray,
    *,
    dimension: int,
    gauge_shifts: np.ndarray,
    gauge_attempt_index: int,
    negative_tolerance_m: float,
    denominator_relative_tolerance: float,
) -> LinearOffsetSolution | None:
    shifted = apply_arrival_gauge(arrivals, gauge_shifts)
    system, _, _, _ = build_linear_offset_system(shifted)
    column_scales = np.linalg.norm(system, axis=0)
    scale_floor = np.finfo(float).eps * max(1.0, float(np.linalg.norm(system)))
    if np.any(column_scales <= scale_floor):
        return None

    normalized = system / column_scales[None, :]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values.size == 0:
        return None
    tolerance = np.finfo(float).eps * max(normalized.shape) * float(singular_values[0])
    rank = int(np.sum(singular_values > tolerance))
    if rank < normalized.shape[1]:
        return None

    normalized_solution, _, _, _ = np.linalg.lstsq(
        normalized,
        np.ones(normalized.shape[0], dtype=float),
        rcond=None,
    )
    u = normalized_solution / column_scales
    event_count = arrivals.shape[1]
    denominators = np.concatenate([u[: event_count - 1], np.array([np.sum(u[: event_count - 1])])])
    denominator_scale = max(float(np.max(np.abs(u))), np.finfo(float).tiny)
    minimum_denominator = float(np.min(np.abs(denominators)))
    if minimum_denominator <= denominator_relative_tolerance * denominator_scale:
        return None

    shifted_offsets = np.empty(event_count, dtype=float)
    shifted_offsets[0] = u[-1] / np.sum(u[: event_count - 1])
    for event in range(1, event_count):
        shifted_offsets[event] = u[(event_count - 1) + (event - 1)] / u[event - 1]
    offsets = restore_offsets_from_gauge(shifted_offsets, gauge_shifts)
    try:
        corrected = corrected_ranges_from_offsets(
            arrivals,
            offsets,
            negative_tolerance_m=negative_tolerance_m,
        )
    except ValueError:
        return None

    residual = system @ u - 1.0
    compacted = compacted_rank_matrix(arrivals, offsets)
    compacted_diagnostics = rank_diagnostics(
        compacted,
        expected_rank=dimension,
    )
    compacted_singular = compacted_diagnostics.singular_values
    discarded = compacted_singular[dimension:]
    compacted_tail_rms = (
        0.0 if discarded.size == 0 else float(np.sqrt(np.mean(discarded * discarded)))
    )
    condition = float(singular_values[0] / singular_values[-1])
    status: LinearOffsetStatus = (
        "solved" if compacted_diagnostics.effective_rank <= dimension else "weakly_identified"
    )

    offsets_copy = np.array(offsets, copy=True)
    shifted_copy = np.array(shifted_offsets, copy=True)
    ranges_copy = np.array(corrected, copy=True)
    gauge_copy = np.array(gauge_shifts, copy=True)
    singular_copy = np.array(singular_values, copy=True)
    for array in (offsets_copy, shifted_copy, ranges_copy, gauge_copy, singular_copy):
        array.setflags(write=False)

    return LinearOffsetSolution(
        status=status,
        offsets_m=offsets_copy,
        shifted_offsets_m=shifted_copy,
        corrected_ranges_m=ranges_copy,
        diagnostics=LinearOffsetDiagnostics(
            dimension=dimension,
            receiver_count=arrivals.shape[0],
            event_count=event_count,
            gauge_attempt_index=gauge_attempt_index,
            gauge_shifts_m=gauge_copy,
            normalized_singular_values=singular_copy,
            normalized_rank=rank,
            normalized_condition_number=condition,
            minimum_denominator=minimum_denominator,
            linear_residual_rms=float(np.sqrt(np.mean(residual * residual))),
            compacted_rank=compacted_diagnostics,
            compacted_tail_rms=compacted_tail_rms,
        ),
    )


def solve_linear_offsets(
    relative_arrivals_m: np.ndarray,
    *,
    dimension: int,
    gauge_shifts_m: np.ndarray | None = None,
    max_gauge_attempts: int = 7,
    denominator_relative_tolerance: float = 1e-10,
    negative_tolerance_relative: float = 1e-9,
) -> LinearOffsetSolution:
    """Solve the linearly solvable m=2K+3, n=K+2 offset cases.

    When no gauge is supplied, several deterministic measurement-scaled gauges are
    tested and the valid candidate with the best normalized system conditioning is
    selected. Gauge choice uses no source truth or held-out data.
    """
    arrivals = np.asarray(relative_arrivals_m, dtype=float)
    if arrivals.ndim != 2 or not np.all(np.isfinite(arrivals)):
        raise ValueError("relative_arrivals_m must be a finite 2-D matrix")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    expected_shape = (2 * dimension + 3, dimension + 2)
    if arrivals.shape != expected_shape:
        raise ValueError(
            "linear offset anchor requires shape "
            f"{expected_shape} for dimension {dimension}, got {arrivals.shape}"
        )
    if max_gauge_attempts < 1:
        raise ValueError("max_gauge_attempts must be positive")
    if denominator_relative_tolerance <= 0.0:
        raise ValueError("denominator_relative_tolerance must be positive")
    if negative_tolerance_relative < 0.0:
        raise ValueError("negative_tolerance_relative must be non-negative")

    measurement_scale = _measurement_scale(arrivals)
    negative_tolerance_m = negative_tolerance_relative * max(
        measurement_scale,
        np.finfo(float).eps,
    )
    if gauge_shifts_m is None:
        gauges = _default_gauge_candidates(
            arrivals,
            max_attempts=max_gauge_attempts,
        )
    else:
        gauge = np.asarray(gauge_shifts_m, dtype=float).reshape(-1)
        if gauge.shape != (arrivals.shape[1],) or not np.all(np.isfinite(gauge)):
            raise ValueError("gauge_shifts_m must be finite with shape (events,)")
        gauges = (gauge,)

    candidates: list[LinearOffsetSolution] = []
    for attempt, gauge in enumerate(gauges):
        solution = _solve_one_gauge(
            arrivals,
            dimension=dimension,
            gauge_shifts=np.asarray(gauge, dtype=float),
            gauge_attempt_index=attempt,
            negative_tolerance_m=negative_tolerance_m,
            denominator_relative_tolerance=denominator_relative_tolerance,
        )
        if solution is not None:
            candidates.append(solution)

    if not candidates:
        raise ValueError(
            "no numerically valid arrival gauge produced a full-rank linear offset system"
        )
    return min(
        candidates,
        key=lambda solution: (
            solution.diagnostics.normalized_condition_number,
            solution.diagnostics.compacted_tail_rms,
            solution.diagnostics.gauge_attempt_index,
        ),
    )
