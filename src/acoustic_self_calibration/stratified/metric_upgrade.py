"""Affine-to-Euclidean metric upgrade for the first 3-D TOA backend.

Contract
--------
For rank-three affine factors Q=X@Y.T with x_0=y_0=0, this module uses

    r_i = L @ x_i
    s_j = solve(L.T, b + y_j)
    H = L.T @ L > 0

with symmetric H (six unknown entries) and b (three entries). Receiver equations

    d_i0^2-d_00^2 = x_i.T@H@x_i - 2*x_i.T@b

are linear in the nine metric unknowns. In the 9-receiver case their generic rank is
8, leaving one free parameter. Each source equation

    d_0j^2 = (b+y_j).T@inv(H)@(b+y_j)

becomes a quartic after multiplying by det(H). The implementation enumerates isolated
real roots, rejects singular/indefinite H using the original rational equations, and
validates reconstructed cross distances. It never clips metric eigenvalues or uses a
nonlinear multistart fallback.

Noisy data are handled diagnostically: roots from all source quartics are evaluated
against every corrected range and ranked by reconstruction RMS. The caller supplies an
acceptance tolerance when exact consistency is not expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares
from scipy.stats import qmc

from ..geometry import canonicalize_scene_conditioned
from .factorization import AffineFactorization
from .notation import cross_gram_from_ranges

MetricUpgradeStatus = Literal["solved", "weakly_identified", "degenerate", "failed"]
_METRIC_HALTON_POOL = 256


@dataclass(frozen=True)
class MetricUpgradeCandidate:
    """One positive-definite Euclidean metric branch."""

    free_parameter: float | None
    metric_matrix: np.ndarray
    metric_vector: np.ndarray
    metric_eigenvalues: np.ndarray
    microphone_positions_m: np.ndarray
    source_positions_m: np.ndarray
    receiver_linear_rms_m2: float
    source_equation_rms_m2: float
    corrected_range_rms_m: float
    corrected_range_max_error_m: float


@dataclass(frozen=True)
class MetricUpgradeDiagnostics:
    """Matrix-specific diagnostics for the 3-D metric stage."""

    range_scale_m: float
    receiver_design_singular_values: np.ndarray
    receiver_design_rank: int
    receiver_design_nullity: int
    receiver_design_condition_number: float
    factor_singular_ratio: float
    real_root_count: int
    positive_definite_candidate_count: int
    accepted_candidate_count: int
    acceptance_rms_m: float
    weak_condition_threshold: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MetricUpgradeResult:
    """All physically realizable metric branches plus status/diagnostics."""

    status: MetricUpgradeStatus
    candidates: tuple[MetricUpgradeCandidate, ...]
    diagnostics: MetricUpgradeDiagnostics


def _metric_matrix_and_vector(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.array(
        [
            [theta[0], theta[1], theta[2]],
            [theta[1], theta[3], theta[4]],
            [theta[2], theta[4], theta[5]],
        ],
        dtype=float,
    )
    b = np.array(theta[6:9], dtype=float)
    return h, b


def build_receiver_metric_system(
    receiver_factors: np.ndarray,
    corrected_ranges_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the linear receiver equations for symmetric H and vector b."""
    x = np.asarray(receiver_factors, dtype=float)
    ranges = np.asarray(corrected_ranges_m, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("receiver_factors must have shape (receivers, 3)")
    if ranges.ndim != 2 or ranges.shape[0] != x.shape[0]:
        raise ValueError("corrected_ranges_m must match receiver_factors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(ranges)):
        raise ValueError("metric-system inputs must be finite")

    rows = []
    rhs = []
    d00_squared = float(ranges[0, 0] ** 2)
    for receiver in range(1, x.shape[0]):
        xi = x[receiver]
        rows.append(
            [
                xi[0] ** 2,
                2.0 * xi[0] * xi[1],
                2.0 * xi[0] * xi[2],
                xi[1] ** 2,
                2.0 * xi[1] * xi[2],
                xi[2] ** 2,
                -2.0 * xi[0],
                -2.0 * xi[1],
                -2.0 * xi[2],
            ]
        )
        rhs.append(float(ranges[receiver, 0] ** 2 - d00_squared))
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def _scaled_linear_family(
    design: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    column_scales = np.linalg.norm(design, axis=0)
    floor = np.finfo(float).eps * max(1.0, float(np.linalg.norm(design)))
    if np.any(column_scales <= floor):
        return (
            np.zeros(9),
            np.zeros((9, 9)),
            np.zeros(0),
            0,
            float("inf"),
        )

    normalized = design / column_scales[None, :]
    _, singular_values, vt = np.linalg.svd(normalized, full_matrices=True)
    tolerance = np.finfo(float).eps * max(normalized.shape) * float(singular_values[0])
    rank = int(np.sum(singular_values > tolerance))
    scaled_particular, _, _, _ = np.linalg.lstsq(normalized, rhs, rcond=None)
    particular = scaled_particular / column_scales

    null_scaled = vt[rank:].T
    null_original = null_scaled / column_scales[:, None]
    condition = float("inf") if rank == 0 else float(singular_values[0] / singular_values[rank - 1])
    return particular, null_original, singular_values, rank, condition


def _source_quartic(
    theta0: np.ndarray,
    direction: np.ndarray,
    source_factor: np.ndarray,
    reference_range: float,
) -> Polynomial:
    p = [Polynomial([theta0[index], direction[index]]) for index in range(9)]
    h00, h01, h02, h11, h12, h22, b0, b1, b2 = p

    a00 = h11 * h22 - h12 * h12
    a01 = h02 * h12 - h01 * h22
    a02 = h01 * h12 - h02 * h11
    a11 = h00 * h22 - h02 * h02
    a12 = h01 * h02 - h00 * h12
    a22 = h00 * h11 - h01 * h01
    determinant = h00 * a00 + h01 * a01 + h02 * a02

    vector = [
        b0 + float(source_factor[0]),
        b1 + float(source_factor[1]),
        b2 + float(source_factor[2]),
    ]
    quadratic = (
        vector[0] * a00 * vector[0]
        + 2.0 * vector[0] * a01 * vector[1]
        + 2.0 * vector[0] * a02 * vector[2]
        + vector[1] * a11 * vector[1]
        + 2.0 * vector[1] * a12 * vector[2]
        + vector[2] * a22 * vector[2]
    )
    return float(reference_range**2) * determinant - quadratic


def _trim_polynomial(polynomial: Polynomial) -> Polynomial:
    coefficients = np.asarray(polynomial.coef, dtype=float)
    if coefficients.size <= 1:
        return polynomial
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    keep = coefficients.size
    while keep > 1 and abs(coefficients[keep - 1]) <= 1e-12 * scale:
        keep -= 1
    return Polynomial(coefficients[:keep])


def _real_roots(
    polynomial: Polynomial,
    *,
    imaginary_relative_tolerance: float,
) -> list[float]:
    trimmed = _trim_polynomial(polynomial)
    if len(trimmed.coef) <= 1:
        return []
    roots = []
    for root in trimmed.roots():
        if not np.isfinite(root):
            continue
        if abs(float(np.imag(root))) <= imaginary_relative_tolerance * (
            1.0 + abs(float(np.real(root)))
        ):
            roots.append(float(np.real(root)))
    return roots


def _deduplicate_roots(
    roots: list[float],
    *,
    relative_tolerance: float,
) -> list[float]:
    ordered = sorted(roots)
    unique: list[float] = []
    for root in ordered:
        if not unique or abs(root - unique[-1]) > relative_tolerance * (1.0 + abs(root)):
            unique.append(root)
    return unique


def _evaluate_candidate(
    theta: np.ndarray,
    *,
    free_parameter: float | None,
    receiver_factors: np.ndarray,
    source_factors: np.ndarray,
    normalized_ranges: np.ndarray,
    range_scale_m: float,
    receiver_design: np.ndarray,
    receiver_rhs: np.ndarray,
    positive_definite_relative_tolerance: float,
) -> MetricUpgradeCandidate | None:
    h, b = _metric_matrix_and_vector(theta)
    eigenvalues = np.linalg.eigvalsh(h)
    largest = float(np.max(np.abs(eigenvalues)))
    pd_floor = positive_definite_relative_tolerance * max(largest, 1.0)
    if float(np.min(eigenvalues)) <= pd_floor:
        return None

    try:
        inverse_h = np.linalg.inv(h)
        l_factor = np.linalg.cholesky(h).T
    except np.linalg.LinAlgError:
        return None

    source_squared_residuals = np.array(
        [
            (b + source_factors[event]) @ inverse_h @ (b + source_factors[event])
            - normalized_ranges[0, event] ** 2
            for event in range(source_factors.shape[0])
        ],
        dtype=float,
    )
    microphones_normalized = receiver_factors @ l_factor.T
    sources_normalized = np.linalg.solve(
        l_factor.T,
        (source_factors + b[None, :]).T,
    ).T
    reconstructed = np.linalg.norm(
        microphones_normalized[:, None, :] - sources_normalized[None, :, :],
        axis=2,
    )
    range_error_m = (reconstructed - normalized_ranges) * range_scale_m
    receiver_residual_m2 = (receiver_design @ theta - receiver_rhs) * (range_scale_m**2)
    source_residual_m2 = source_squared_residuals * (range_scale_m**2)

    microphones_m = microphones_normalized * range_scale_m
    sources_m = sources_normalized * range_scale_m
    try:
        canonical_microphones, canonical_sources, _ = canonicalize_scene_conditioned(
            microphones_m,
            sources_m,
        )
    except ValueError:
        return None
    assert canonical_sources is not None

    arrays = [
        h,
        b,
        eigenvalues,
        canonical_microphones,
        canonical_sources,
    ]
    frozen: list[np.ndarray] = []
    for array in arrays:
        copy = np.array(array, copy=True)
        copy.setflags(write=False)
        frozen.append(copy)

    return MetricUpgradeCandidate(
        free_parameter=free_parameter,
        metric_matrix=frozen[0],
        metric_vector=frozen[1],
        metric_eigenvalues=frozen[2],
        microphone_positions_m=frozen[3],
        source_positions_m=frozen[4],
        receiver_linear_rms_m2=float(np.sqrt(np.mean(receiver_residual_m2 * receiver_residual_m2))),
        source_equation_rms_m2=float(np.sqrt(np.mean(source_residual_m2 * source_residual_m2))),
        corrected_range_rms_m=float(np.sqrt(np.mean(range_error_m * range_error_m))),
        corrected_range_max_error_m=float(np.max(np.abs(range_error_m))),
    )


def upgrade_metric_3d(
    factorization: AffineFactorization,
    corrected_ranges_m: np.ndarray,
    *,
    acceptance_rms_m: float | None = None,
    weak_factor_ratio: float = 1e-6,
    weak_condition_threshold: float = 1e8,
    imaginary_relative_tolerance: float = 1e-7,
    root_relative_tolerance: float = 1e-6,
    positive_definite_relative_tolerance: float = 1e-10,
) -> MetricUpgradeResult:
    """Enumerate 3-D metric branches for the 9r/5s-style TOA backend."""
    if factorization.dimension != 3:
        raise ValueError("upgrade_metric_3d requires a rank-three factorization")
    ranges = np.asarray(corrected_ranges_m, dtype=float)
    if ranges.ndim != 2 or not np.all(np.isfinite(ranges)) or np.any(ranges < 0.0):
        raise ValueError("corrected_ranges_m must be finite and non-negative")
    if ranges.shape != (
        factorization.receiver_factors.shape[0],
        factorization.source_factors.shape[0],
    ):
        raise ValueError("corrected_ranges_m shape does not match factorization")
    if ranges.shape[0] < 9 or ranges.shape[1] < 5:
        raise ValueError("3-D metric backend requires at least 9 receivers and 5 events")

    expected_gram = cross_gram_from_ranges(ranges)
    gram_mismatch = float(
        np.sqrt(np.mean((expected_gram - factorization.compacted_cross_gram) ** 2))
    )
    gram_scale = max(1.0, float(np.sqrt(np.mean(expected_gram * expected_gram))))
    if gram_mismatch > 1e-10 * gram_scale:
        raise ValueError("factorization and corrected_ranges_m describe different data")

    positive_ranges = ranges[ranges > 0.0]
    if positive_ranges.size == 0:
        raise ValueError("corrected_ranges_m contains no positive ranges")
    range_scale_m = float(np.median(positive_ranges))
    normalized_ranges = ranges / range_scale_m
    receiver_factors = factorization.receiver_factors / range_scale_m
    source_factors = factorization.source_factors / range_scale_m

    factor_singular = factorization.rank_diagnostics.singular_values
    if factor_singular.size < 3 or factor_singular[0] <= 0.0:
        factor_ratio = 0.0
    else:
        factor_ratio = float(factor_singular[2] / factor_singular[0])

    reasons: list[str] = []
    if factorization.rank_diagnostics.effective_rank < 3:
        reasons.append("affine_factor_rank_below_3")
        return MetricUpgradeResult(
            status="degenerate",
            candidates=(),
            diagnostics=MetricUpgradeDiagnostics(
                range_scale_m=range_scale_m,
                receiver_design_singular_values=np.empty(0),
                receiver_design_rank=0,
                receiver_design_nullity=9,
                receiver_design_condition_number=float("inf"),
                factor_singular_ratio=factor_ratio,
                real_root_count=0,
                positive_definite_candidate_count=0,
                accepted_candidate_count=0,
                acceptance_rms_m=(
                    1e-8 * range_scale_m if acceptance_rms_m is None else float(acceptance_rms_m)
                ),
                weak_condition_threshold=weak_condition_threshold,
                reasons=tuple(reasons),
            ),
        )

    design, rhs = build_receiver_metric_system(
        receiver_factors,
        normalized_ranges,
    )
    particular, nullspace, singular_values, design_rank, condition = _scaled_linear_family(
        design, rhs
    )
    nullity = 9 - design_rank
    if design_rank < 8 or nullity > 1:
        reasons.append("receiver_metric_design_rank_deficient")
        singular_copy = np.array(singular_values, copy=True)
        singular_copy.setflags(write=False)
        return MetricUpgradeResult(
            status="degenerate",
            candidates=(),
            diagnostics=MetricUpgradeDiagnostics(
                range_scale_m=range_scale_m,
                receiver_design_singular_values=singular_copy,
                receiver_design_rank=design_rank,
                receiver_design_nullity=nullity,
                receiver_design_condition_number=condition,
                factor_singular_ratio=factor_ratio,
                real_root_count=0,
                positive_definite_candidate_count=0,
                accepted_candidate_count=0,
                acceptance_rms_m=(
                    1e-8 * range_scale_m if acceptance_rms_m is None else float(acceptance_rms_m)
                ),
                weak_condition_threshold=weak_condition_threshold,
                reasons=tuple(reasons),
            ),
        )

    if acceptance_rms_m is None:
        acceptance_rms_m = 1e-8 * range_scale_m
    if acceptance_rms_m <= 0.0:
        raise ValueError("acceptance_rms_m must be positive")

    root_values: list[float | None] = []
    if nullity == 0:
        root_values.append(None)
    else:
        direction = nullspace[:, 0]
        raw_roots: list[float] = []
        for event in range(source_factors.shape[0]):
            polynomial = _source_quartic(
                particular,
                direction,
                source_factors[event],
                normalized_ranges[0, event],
            )
            raw_roots.extend(
                _real_roots(
                    polynomial,
                    imaginary_relative_tolerance=imaginary_relative_tolerance,
                )
            )
        root_values.extend(
            _deduplicate_roots(
                raw_roots,
                relative_tolerance=root_relative_tolerance,
            )
        )

    candidates: list[MetricUpgradeCandidate] = []
    for root in root_values:
        theta = particular if root is None else particular + float(root) * nullspace[:, 0]
        candidate = _evaluate_candidate(
            theta,
            free_parameter=root,
            receiver_factors=receiver_factors,
            source_factors=source_factors,
            normalized_ranges=normalized_ranges,
            range_scale_m=range_scale_m,
            receiver_design=design,
            receiver_rhs=rhs,
            positive_definite_relative_tolerance=(positive_definite_relative_tolerance),
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.corrected_range_rms_m)
    accepted = [
        candidate for candidate in candidates if candidate.corrected_range_rms_m <= acceptance_rms_m
    ]

    if factor_ratio < weak_factor_ratio:
        reasons.append("weak_affine_rank_separation")
    if condition > weak_condition_threshold:
        reasons.append("ill_conditioned_receiver_metric_design")
    if not candidates:
        reasons.append("no_positive_definite_metric_branch")
        status: MetricUpgradeStatus = "failed"
    elif not accepted:
        reasons.append("no_metric_branch_meets_range_tolerance")
        status = "weakly_identified"
    elif reasons:
        status = "weakly_identified"
    else:
        status = "solved"

    singular_copy = np.array(singular_values, copy=True)
    singular_copy.setflags(write=False)
    return MetricUpgradeResult(
        status=status,
        candidates=tuple(candidates),
        diagnostics=MetricUpgradeDiagnostics(
            range_scale_m=range_scale_m,
            receiver_design_singular_values=singular_copy,
            receiver_design_rank=design_rank,
            receiver_design_nullity=nullity,
            receiver_design_condition_number=condition,
            factor_singular_ratio=factor_ratio,
            real_root_count=len(root_values),
            positive_definite_candidate_count=len(candidates),
            accepted_candidate_count=len(accepted),
            acceptance_rms_m=float(acceptance_rms_m),
            weak_condition_threshold=weak_condition_threshold,
            reasons=tuple(reasons),
        ),
    )


def _metric_nullspace_starts(nullity: int, start_count: int) -> np.ndarray:
    """Prefix-stable multistart free-variable seeds for nullity > 0."""
    if start_count < 1:
        raise ValueError("start_count must be positive")
    if nullity == 0:
        return np.zeros((start_count, 0), dtype=float)
    mixed_count = 5
    points = qmc.Halton(d=nullity, scramble=False).random(mixed_count)
    starts_list: list[np.ndarray] = [np.zeros(nullity, dtype=float)]
    for amplitude in (4.0, 6.0):
        starts_list.extend(amplitude * (point - 0.5) for point in points)
    if len(starts_list) < start_count:
        needed = start_count - len(starts_list)
        pool_count = max(_METRIC_HALTON_POOL, needed)
        pool = qmc.Halton(d=nullity, scramble=False).random(pool_count)
        for index in range(needed):
            starts_list.append(16.0 * (pool[index] - 0.5))
    return np.asarray(starts_list[:start_count], dtype=float)


def upgrade_metric_3d_overdetermined(
    factorization: AffineFactorization,
    corrected_ranges_m: np.ndarray,
    *,
    start_count: int = 32,
    acceptance_rms_m: float | None = None,
    weak_factor_ratio: float = 1e-6,
    weak_condition_threshold: float = 1e8,
    positive_definite_relative_tolerance: float = 1e-10,
) -> MetricUpgradeResult:
    """Solve the 3-D metric when receiver equations leave up to three free variables."""
    if factorization.dimension != 3:
        raise ValueError("rank-three factorization required")
    ranges = np.asarray(corrected_ranges_m, dtype=float)
    if ranges.ndim != 2 or not np.all(np.isfinite(ranges)) or np.any(ranges < 0.0):
        raise ValueError("corrected_ranges_m must be finite and non-negative")
    if ranges.shape != (
        factorization.receiver_factors.shape[0],
        factorization.source_factors.shape[0],
    ):
        raise ValueError("corrected_ranges_m shape does not match factorization")
    if ranges.shape[0] < 7 or ranges.shape[1] < 6:
        raise ValueError("overdetermined 3-D metric backend requires at least 7x6 data")
    if start_count < 1:
        raise ValueError("start_count must be positive")

    positive_ranges = ranges[ranges > 0.0]
    if positive_ranges.size == 0:
        raise ValueError("corrected_ranges_m contains no positive ranges")
    range_scale_m = float(np.median(positive_ranges))
    normalized_ranges = ranges / range_scale_m
    receiver_factors = factorization.receiver_factors / range_scale_m
    source_factors = factorization.source_factors / range_scale_m

    design, rhs = build_receiver_metric_system(receiver_factors, normalized_ranges)
    particular, nullspace, singular_values, rank, condition = _scaled_linear_family(
        design,
        rhs,
    )
    nullity = 9 - rank
    factor_singular = factorization.rank_diagnostics.singular_values
    factor_ratio = (
        0.0
        if factor_singular.size < 3 or factor_singular[0] <= 0.0
        else float(factor_singular[2] / factor_singular[0])
    )
    reasons: list[str] = []
    if rank < 6 or nullity > 3:
        reasons.append("receiver_metric_design_rank_deficient")
        singular_copy = np.array(singular_values, copy=True)
        singular_copy.setflags(write=False)
        return MetricUpgradeResult(
            status="degenerate",
            candidates=(),
            diagnostics=MetricUpgradeDiagnostics(
                range_scale_m=range_scale_m,
                receiver_design_singular_values=singular_copy,
                receiver_design_rank=rank,
                receiver_design_nullity=nullity,
                receiver_design_condition_number=condition,
                factor_singular_ratio=factor_ratio,
                real_root_count=0,
                positive_definite_candidate_count=0,
                accepted_candidate_count=0,
                acceptance_rms_m=(
                    1e-8 * range_scale_m if acceptance_rms_m is None else float(acceptance_rms_m)
                ),
                weak_condition_threshold=weak_condition_threshold,
                reasons=tuple(reasons),
            ),
        )

    if acceptance_rms_m is None:
        acceptance_rms_m = 1e-8 * range_scale_m
    if acceptance_rms_m <= 0.0:
        raise ValueError("acceptance_rms_m must be positive")

    def source_residual(free_variables: np.ndarray) -> np.ndarray:
        theta = particular + nullspace @ free_variables
        h, b = _metric_matrix_and_vector(theta)
        try:
            inverse_h = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            return np.full(source_factors.shape[0], 1e6, dtype=float)
        return np.asarray(
            [
                (b + source_factors[event]) @ inverse_h @ (b + source_factors[event])
                - normalized_ranges[0, event] ** 2
                for event in range(source_factors.shape[0])
            ],
            dtype=float,
        )

    if nullity == 0:
        starts = np.zeros((1, 0), dtype=float)
    else:
        starts = _metric_nullspace_starts(nullity, start_count)

    candidates: list[MetricUpgradeCandidate] = []
    free_solutions: list[np.ndarray] = []
    for initial in starts:
        fit = least_squares(
            source_residual,
            initial,
            max_nfev=1000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            x_scale="jac",
        )
        free = np.asarray(fit.x, dtype=float)
        residual = source_residual(free)
        if not np.all(np.isfinite(residual)) or float(np.max(np.abs(residual))) > 1e6:
            continue
        if any(
            np.linalg.norm(free - existing) <= 1e-6 * (1.0 + np.linalg.norm(free))
            for existing in free_solutions
        ):
            continue
        free_solutions.append(free)
        theta = particular + nullspace @ free
        candidate = _evaluate_candidate(
            theta,
            free_parameter=None,
            receiver_factors=receiver_factors,
            source_factors=source_factors,
            normalized_ranges=normalized_ranges,
            range_scale_m=range_scale_m,
            receiver_design=design,
            receiver_rhs=rhs,
            positive_definite_relative_tolerance=positive_definite_relative_tolerance,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.corrected_range_rms_m)
    accepted = [
        candidate for candidate in candidates if candidate.corrected_range_rms_m <= acceptance_rms_m
    ]
    if factor_ratio < weak_factor_ratio:
        reasons.append("weak_affine_rank_separation")
    if condition > weak_condition_threshold:
        reasons.append("ill_conditioned_receiver_metric_design")
    if not candidates:
        reasons.append("no_positive_definite_metric_branch")
        status: MetricUpgradeStatus = "failed"
    elif not accepted:
        reasons.append("no_metric_branch_meets_range_tolerance")
        status = "weakly_identified"
    elif reasons:
        status = "weakly_identified"
    else:
        status = "solved"

    singular_copy = np.array(singular_values, copy=True)
    singular_copy.setflags(write=False)
    return MetricUpgradeResult(
        status=status,
        candidates=tuple(candidates),
        diagnostics=MetricUpgradeDiagnostics(
            range_scale_m=range_scale_m,
            receiver_design_singular_values=singular_copy,
            receiver_design_rank=rank,
            receiver_design_nullity=nullity,
            receiver_design_condition_number=condition,
            factor_singular_ratio=factor_ratio,
            real_root_count=len(free_solutions),
            positive_definite_candidate_count=len(candidates),
            accepted_candidate_count=len(accepted),
            acceptance_rms_m=float(acceptance_rms_m),
            weak_condition_threshold=weak_condition_threshold,
            reasons=tuple(reasons),
        ),
    )
