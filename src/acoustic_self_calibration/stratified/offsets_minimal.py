from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from .generated_7r6s import (
    ALL_MINOR_SPECS,
    EVENT_COUNT,
    GENERIC_COMPLEX_SOLUTION_COUNT,
    PRIMARY_MINOR_SPECS,
    RECEIVER_COUNT,
    TEMPLATE_SHA256,
)
from .notation import compacted_rank_matrix, corrected_ranges_from_offsets, rank_diagnostics

MinimalOffsetStatus = Literal["solved", "degenerate", "failed"]


@dataclass(frozen=True)
class MinimalOffsetRoot:
    offsets_m: np.ndarray
    corrected_ranges_m: np.ndarray
    all_minor_rms: float
    all_minor_max_abs: float
    jacobian_rank: int
    minimum_corrected_range_m: float


@dataclass(frozen=True)
class MinimalOffsetDiagnostics:
    template_sha256: str
    start_count: int
    converged_start_count: int
    duplicate_root_count: int
    verified_root_count: int
    physical_root_count: int
    generic_complex_solution_count: int
    compacted_rank_at_best: int | None
    search_stabilized: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MinimalOffsetResult:
    status: MinimalOffsetStatus
    roots: tuple[MinimalOffsetRoot, ...]
    diagnostics: MinimalOffsetDiagnostics


_PRIMARY_ROW_INDEX = np.asarray([rows for rows, _ in PRIMARY_MINOR_SPECS], dtype=int)
_PRIMARY_COLUMN_INDEX = np.asarray(
    [columns for _, columns in PRIMARY_MINOR_SPECS],
    dtype=int,
)
_ALL_ROW_INDEX = np.asarray([rows for rows, _ in ALL_MINOR_SPECS], dtype=int)
_ALL_COLUMN_INDEX = np.asarray([columns for _, columns in ALL_MINOR_SPECS], dtype=int)


def _measurement_scale(arrivals: np.ndarray) -> float:
    nonzero = np.abs(arrivals[np.abs(arrivals) > np.finfo(float).eps])
    if nonzero.size == 0:
        raise ValueError("relative_arrivals_m contains no nonzero measurements")
    return max(float(np.median(nonzero)), np.finfo(float).tiny)


def _minor_residuals(
    relative_arrivals_m: np.ndarray,
    offsets_m: np.ndarray,
    scale_m: float,
    *,
    primary_only: bool,
) -> np.ndarray:
    matrix = compacted_rank_matrix(relative_arrivals_m, offsets_m)
    row_index = _PRIMARY_ROW_INDEX if primary_only else _ALL_ROW_INDEX
    column_index = _PRIMARY_COLUMN_INDEX if primary_only else _ALL_COLUMN_INDEX
    blocks = matrix[row_index[:, :, None], column_index[:, None, :]]
    return np.linalg.det(blocks) / (scale_m**8)


def _finite_difference_jacobian(
    relative_arrivals_m: np.ndarray,
    offsets_m: np.ndarray,
    scale_m: float,
) -> np.ndarray:
    base = _minor_residuals(
        relative_arrivals_m,
        offsets_m,
        scale_m,
        primary_only=False,
    )
    jacobian = np.empty((len(base), EVENT_COUNT), dtype=float)
    step_scale = np.sqrt(np.finfo(float).eps)
    for column in range(EVENT_COUNT):
        step = step_scale * max(scale_m, abs(float(offsets_m[column])), 1.0)
        shifted = np.array(offsets_m, copy=True)
        shifted[column] += step
        jacobian[:, column] = (
            _minor_residuals(
                relative_arrivals_m,
                shifted,
                scale_m,
                primary_only=False,
            )
            - base
        ) / step
    return jacobian


def _halton_starts(
    arrivals: np.ndarray,
    *,
    start_count: int,
    physical_only: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    scale = _measurement_scale(arrivals)
    minimum = np.min(arrivals, axis=0)
    points = qmc.Halton(d=EVENT_COUNT, scramble=False).random(start_count)
    logarithmic = np.exp(np.log(0.08) + (np.log(30.0) - np.log(0.08)) * points)
    if physical_only:
        starts = minimum[None, :] - scale * logarithmic

        # Physical event offsets are -d(reference, source), so their distance below
        # the per-event lower bound is strongly correlated across a six-event seed.
        # Reserve part of the fixed start budget for common-range starts instead of
        # spending every start on an independent six-dimensional Halton draw.
        correlated_count = min(start_count, max(6, start_count // 3))
        correlated_radii = np.geomspace(0.2, 16.0, correlated_count)
        starts[:correlated_count] = minimum[None, :] - scale * correlated_radii[:, None]

        lower = minimum - 100.0 * scale
        upper = minimum - 1e-10 * scale
        return starts, lower, upper

    signed = 2.0 * points - 1.0
    radii = np.exp(
        np.log(0.1)
        + (np.log(40.0) - np.log(0.1)) * qmc.Halton(d=1, scramble=False).random(start_count)[:, 0]
    )
    starts = minimum[None, :] + scale * signed * radii[:, None]
    return starts, None, None


def solve_offsets_7r6s(
    relative_arrivals_m: np.ndarray,
    *,
    start_count: int = 64,
    physical_only: bool = True,
    residual_tolerance: float = 1e-7,
    duplicate_relative_tolerance: float = 1e-5,
    max_nfev: int = 500,
) -> MinimalOffsetResult:
    """Enumerate verified real 7r/6s offset roots with deterministic starts.

    The generated polynomial template is exact. Numerical root enumeration uses a
    deterministic Halton schedule and verifies every candidate against all 75 rank-four
    minors. search_stabilized is a numerical completeness diagnostic, not a proof that
    no real root exists outside the bounded search region.
    """
    arrivals = np.asarray(relative_arrivals_m, dtype=float)
    if arrivals.shape != (RECEIVER_COUNT, EVENT_COUNT):
        raise ValueError(
            f"7r/6s solver requires shape {(RECEIVER_COUNT, EVENT_COUNT)}, got {arrivals.shape}"
        )
    if not np.all(np.isfinite(arrivals)):
        raise ValueError("relative_arrivals_m must be finite")
    if start_count < 8:
        raise ValueError("start_count must be at least 8")
    if residual_tolerance <= 0.0 or duplicate_relative_tolerance <= 0.0:
        raise ValueError("root tolerances must be positive")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")

    scale = _measurement_scale(arrivals)
    starts, lower, upper = _halton_starts(
        arrivals,
        start_count=start_count,
        physical_only=physical_only,
    )
    roots: list[MinimalOffsetRoot] = []
    converged = 0
    duplicates = 0
    last_new_root_start = -1

    def residual(offsets: np.ndarray) -> np.ndarray:
        return _minor_residuals(
            arrivals,
            offsets,
            scale,
            primary_only=True,
        )

    for start_index, initial in enumerate(starts):
        if lower is not None and upper is not None:
            fit = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                max_nfev=max_nfev,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                x_scale="jac",
            )
        else:
            fit = least_squares(
                residual,
                initial,
                max_nfev=max_nfev,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                x_scale="jac",
            )
        candidate_offsets = np.asarray(fit.x, dtype=float)
        primary_residual = residual(candidate_offsets)
        if not np.all(np.isfinite(candidate_offsets)):
            continue
        if float(np.max(np.abs(primary_residual))) > residual_tolerance:
            continue
        candidate_residual = _minor_residuals(
            arrivals,
            candidate_offsets,
            scale,
            primary_only=False,
        )
        rms = float(np.sqrt(np.mean(candidate_residual * candidate_residual)))
        maximum = float(np.max(np.abs(candidate_residual)))
        if maximum > residual_tolerance:
            continue
        converged += 1

        if any(
            np.linalg.norm(candidate_offsets - existing.offsets_m)
            <= duplicate_relative_tolerance * (1.0 + np.linalg.norm(candidate_offsets))
            for existing in roots
        ):
            duplicates += 1
            continue

        try:
            corrected = corrected_ranges_from_offsets(
                arrivals,
                candidate_offsets,
                negative_tolerance_m=1e-8 * scale,
            )
        except ValueError:
            if physical_only:
                continue
            corrected = arrivals - candidate_offsets[None, :]

        jacobian = _finite_difference_jacobian(arrivals, candidate_offsets, scale)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        tolerance = (
            np.finfo(float).eps * max(jacobian.shape) * float(singular[0]) if singular.size else 0.0
        )
        jacobian_rank = int(np.sum(singular > tolerance))
        offsets_copy = np.array(candidate_offsets, copy=True)
        ranges_copy = np.array(corrected, copy=True)
        offsets_copy.setflags(write=False)
        ranges_copy.setflags(write=False)
        roots.append(
            MinimalOffsetRoot(
                offsets_m=offsets_copy,
                corrected_ranges_m=ranges_copy,
                all_minor_rms=rms,
                all_minor_max_abs=maximum,
                jacobian_rank=jacobian_rank,
                minimum_corrected_range_m=float(np.min(corrected)),
            )
        )
        last_new_root_start = start_index

    roots.sort(key=lambda item: (item.all_minor_rms, -item.minimum_corrected_range_m))
    reasons: list[str] = []
    if any(item.jacobian_rank < EVENT_COUNT for item in roots):
        reasons.append("nonisolated_or_rank_deficient_root")
    if not roots:
        reasons.append("no_verified_real_root")
        status: MinimalOffsetStatus = "failed"
    elif reasons:
        status = "degenerate"
    else:
        status = "solved"

    stabilized_window = max(8, start_count // 4)
    stabilized = (
        bool(roots)
        and last_new_root_start >= 0
        and start_count - 1 - last_new_root_start >= stabilized_window
    )
    if not stabilized:
        reasons.append("root_search_not_stabilized")

    best_rank = None
    if roots:
        best_rank = rank_diagnostics(
            compacted_rank_matrix(arrivals, roots[0].offsets_m),
            expected_rank=3,
        ).effective_rank

    physical_count = sum(item.minimum_corrected_range_m >= -1e-8 * scale for item in roots)
    return MinimalOffsetResult(
        status=status,
        roots=tuple(roots),
        diagnostics=MinimalOffsetDiagnostics(
            template_sha256=TEMPLATE_SHA256,
            start_count=start_count,
            converged_start_count=converged,
            duplicate_root_count=duplicates,
            verified_root_count=len(roots),
            physical_root_count=physical_count,
            generic_complex_solution_count=GENERIC_COMPLEX_SOLUTION_COUNT,
            compacted_rank_at_best=best_rank,
            search_stabilized=stabilized,
            reasons=tuple(reasons),
        ),
    )
