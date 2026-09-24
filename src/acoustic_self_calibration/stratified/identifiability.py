from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

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
    # Factors inherit rounding amplified by offset recovery and expansion.
    # Machine epsilon alone can turn an exact conic into a spurious full rank.
    relative_tolerance = 1e-10
    metric_diag = rank_diagnostics(metric, expected_rank=5, relative_tolerance=relative_tolerance)
    quadratic_diag = rank_diagnostics(
        quadratic, expected_rank=6, relative_tolerance=relative_tolerance
    )
    _, metric_singular, metric_vt = np.linalg.svd(metric, full_matrices=True)
    metric_tolerance = (
        relative_tolerance * float(metric_singular[0]) if metric_singular.size else 0.0
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


@dataclass(frozen=True)
class PlanarNoiseSensitivity:
    """Local geometry uncertainty after removing source positions and rigid gauge."""

    weakest_microphone_rms_std_m: float
    relative_weakest_std: float
    observable_rank: int
    parameter_count: int
    residual_degrees_of_freedom: int
    reduced_chi_square: float | None
    fit_p_value: float | None
    noise_scale_inflation: float


def planar_noise_sensitivity(
    whitened_jacobian: np.ndarray,
    microphone_positions_2d_m: np.ndarray,
    *,
    whitened_residual: np.ndarray | None = None,
) -> PlanarNoiseSensitivity:
    """Assess an unconstrained planar fit using its noise-whitened TDOA Jacobian.

    Columns must contain all 2N microphone coordinates followed by source
    coordinates. Source motion is nuisance information and is projected out.
    Two translations and one in-plane rotation are also removed. The remaining
    weakest singular mode gives a one-standard-deviation microphone RMS change.
    Residual variance above the declared noise inflates uncertainty. The fit
    p-value assumes independent Gaussian coordinates after whitening. This is a
    local linear diagnostic, not a global uniqueness certificate.
    """
    microphones = np.asarray(microphone_positions_2d_m, dtype=float)
    jacobian = np.asarray(whitened_jacobian, dtype=float)
    if microphones.ndim != 2 or microphones.shape[1] != 2 or len(microphones) < 3:
        raise ValueError("microphone_positions_2d_m must have shape (N >= 3, 2)")
    columns = microphones.size
    if jacobian.ndim != 2 or jacobian.shape[1] < columns:
        raise ValueError("jacobian must start with all planar microphone coordinates")
    if not np.all(np.isfinite(microphones)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("geometry and jacobian must be finite")
    centered = microphones - np.mean(microphones, axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    gauge = np.zeros((columns, 3))
    gauge[0::2, 0] = 1.0
    gauge[1::2, 1] = 1.0
    gauge[0::2, 2] = -centered[:, 1]
    gauge[1::2, 2] = centered[:, 0]
    gauge_u, gauge_s, _ = np.linalg.svd(gauge, full_matrices=True)
    gauge_rank = int(np.sum(gauge_s > 1e-12 * gauge_s[0]))
    reduced = jacobian[:, :columns] @ gauge_u[:, gauge_rank:]
    nuisance = jacobian[:, columns:]
    nuisance_rank = 0
    if nuisance.size:
        nuisance_u, nuisance_s, _ = np.linalg.svd(nuisance, full_matrices=False)
        nuisance_rank = int(np.sum(nuisance_s > 1e-12 * nuisance_s[0]))
        basis = nuisance_u[:, :nuisance_rank]
        reduced -= basis @ (basis.T @ reduced)
    singular = np.linalg.svd(reduced, compute_uv=False)
    parameter_count = columns - gauge_rank
    rank = int(np.sum(singular > 1e-10 * singular[0])) if singular.size else 0
    uncertainty = (
        float(1.0 / (singular[-1] * np.sqrt(len(microphones))))
        if rank == parameter_count
        else float("inf")
    )
    degrees_of_freedom = jacobian.shape[0] - nuisance_rank - rank
    reduced_chi_square = None
    fit_p_value = None
    inflation = 1.0
    if whitened_residual is not None:
        residual = np.asarray(whitened_residual, dtype=float)
        if residual.shape != (jacobian.shape[0],) or not np.all(np.isfinite(residual)):
            raise ValueError("whitened_residual must be finite and match jacobian rows")
        squared_norm = float(residual @ residual)
        if degrees_of_freedom > 0:
            reduced_chi_square = squared_norm / degrees_of_freedom
            fit_p_value = float(chi2.sf(squared_norm, degrees_of_freedom))
            inflation = float(np.sqrt(max(1.0, reduced_chi_square)))
        else:
            inflation = float("inf")
        uncertainty *= inflation
    return PlanarNoiseSensitivity(
        weakest_microphone_rms_std_m=uncertainty,
        relative_weakest_std=uncertainty / radius if radius > 0.0 else float("inf"),
        observable_rank=rank,
        parameter_count=parameter_count,
        residual_degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=reduced_chi_square,
        fit_p_value=fit_p_value,
        noise_scale_inflation=inflation,
    )
