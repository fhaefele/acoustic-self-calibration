from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notation import MatrixRankDiagnostics, rank_diagnostics


@dataclass(frozen=True)
class PlanarIdentifiabilityDiagnostics:
    metric_design: MatrixRankDiagnostics
    quadratic_design: MatrixRankDiagnostics
    continuous_metric_nullity: int
    metric_nullspace: np.ndarray
    conic_nullity: int
    cross_like_degeneracy: bool
    reasons: tuple[str, ...]


def planar_metric_design(receiver_factors: np.ndarray) -> np.ndarray:
    """Return the five-column linear planar metric design for H and b."""
    x = np.asarray(receiver_factors, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("receiver_factors must have shape (receivers, 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("receiver_factors must be finite")
    nonreference = x[1:]
    return np.column_stack(
        [
            nonreference[:, 0] ** 2,
            2.0 * nonreference[:, 0] * nonreference[:, 1],
            nonreference[:, 1] ** 2,
            -2.0 * nonreference[:, 0],
            -2.0 * nonreference[:, 1],
        ]
    )


def planar_quadratic_design(receiver_factors: np.ndarray) -> np.ndarray:
    """Return the affine-invariant six-column conic design."""
    x = np.asarray(receiver_factors, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("receiver_factors must have shape (receivers, 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("receiver_factors must be finite")
    return np.column_stack(
        [
            x[:, 0] ** 2,
            x[:, 0] * x[:, 1],
            x[:, 1] ** 2,
            x[:, 0],
            x[:, 1],
            np.ones(len(x)),
        ]
    )


def diagnose_planar_identifiability(
    receiver_factors: np.ndarray,
) -> PlanarIdentifiabilityDiagnostics:
    metric = planar_metric_design(receiver_factors)
    quadratic = planar_quadratic_design(receiver_factors)
    metric_diag = rank_diagnostics(metric, expected_rank=5)
    quadratic_diag = rank_diagnostics(quadratic, expected_rank=6)
    _, metric_singular, metric_vt = np.linalg.svd(metric, full_matrices=True)
    metric_tolerance = (
        np.finfo(float).eps * max(metric.shape) * float(metric_singular[0])
        if metric_singular.size
        else 0.0
    )
    metric_rank = int(np.sum(metric_singular > metric_tolerance))
    metric_nullspace = metric_vt[metric_rank:].T
    metric_nullity = int(metric_nullspace.shape[1])
    conic_nullity = max(0, 6 - quadratic_diag.effective_rank)
    reasons: list[str] = []
    if metric_nullity:
        reasons.append("receiver_metric_design_rank_deficient")
    if conic_nullity:
        reasons.append("receiver_points_lie_on_nontrivial_conic")
    cross_like = metric_nullity > 0 and conic_nullity > 0
    nullspace_copy = np.array(metric_nullspace, copy=True)
    nullspace_copy.setflags(write=False)
    return PlanarIdentifiabilityDiagnostics(
        metric_design=metric_diag,
        quadratic_design=quadratic_diag,
        continuous_metric_nullity=metric_nullity,
        metric_nullspace=nullspace_copy,
        conic_nullity=conic_nullity,
        cross_like_degeneracy=cross_like,
        reasons=tuple(reasons),
    )
