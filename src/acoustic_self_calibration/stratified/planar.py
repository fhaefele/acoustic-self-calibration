from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..geometry import canonicalize_scene_conditioned
from .constraints import PlanarAngleConstraint, metric_angle_rad, right_angle_metric_row
from .factorization import AffineFactorization
from .identifiability import (
    PlanarIdentifiabilityDiagnostics,
    diagnose_planar_identifiability,
    planar_metric_design,
)

PlanarMetricStatus = Literal["solved", "weakly_identified", "degenerate", "failed"]


@dataclass(frozen=True)
class PlanarMetricCandidate:
    microphone_positions_m: np.ndarray
    source_projected_positions_m: np.ndarray
    source_unsigned_heights_m: np.ndarray
    source_height_sign_known: np.ndarray
    source_representative_positions_m: np.ndarray
    metric_matrix: np.ndarray
    metric_vector: np.ndarray
    metric_eigenvalues: np.ndarray
    receiver_linear_rms_m2: float
    corrected_range_rms_m: float
    corrected_range_max_error_m: float
    minimum_height_squared_m2: float


@dataclass(frozen=True)
class PlanarMetricDiagnostics:
    identifiability: PlanarIdentifiabilityDiagnostics
    factor_singular_ratio: float
    range_scale_m: float
    negative_height_count: int
    near_zero_height_count: int
    acceptance_rms_m: float
    constraints_applied: tuple[str, ...]
    constraint_residual_rad: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlanarMetricResult:
    status: PlanarMetricStatus
    candidates: tuple[PlanarMetricCandidate, ...]
    diagnostics: PlanarMetricDiagnostics


def _solve_planar_metric_linear(
    receiver_factors: np.ndarray,
    normalized_ranges: np.ndarray,
    *,
    angle_constraint: PlanarAngleConstraint | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    design = planar_metric_design(receiver_factors)
    rhs = normalized_ranges[1:, 0] ** 2 - normalized_ranges[0, 0] ** 2
    if angle_constraint is not None:
        constraint_row = right_angle_metric_row(receiver_factors, angle_constraint)
        design = np.vstack([design, constraint_row])
        rhs = np.concatenate([rhs, np.array([0.0])])
    column_scales = np.linalg.norm(design, axis=0)
    floor = np.finfo(float).eps * max(1.0, float(np.linalg.norm(design)))
    if np.any(column_scales <= floor):
        return np.zeros(5), np.zeros(0), 0, float("inf")
    normalized = design / column_scales[None, :]
    solution_scaled, _, _, _ = np.linalg.lstsq(normalized, rhs, rcond=None)
    solution = solution_scaled / column_scales
    singular = np.linalg.svd(normalized, compute_uv=False)
    tolerance = (
        np.finfo(float).eps * max(normalized.shape) * float(singular[0]) if singular.size else 0.0
    )
    rank = int(np.sum(singular > tolerance))
    condition = float("inf") if rank < 5 else float(singular[0] / singular[4])
    return solution, singular, rank, condition


def _metric_from_parameters(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.array(
        [
            [parameters[0], parameters[1]],
            [parameters[1], parameters[2]],
        ],
        dtype=float,
    )
    b = np.asarray(parameters[3:5], dtype=float)
    return h, b


def upgrade_metric_planar(
    factorization: AffineFactorization,
    corrected_ranges_m: np.ndarray,
    *,
    acceptance_rms_m: float | None = None,
    weak_factor_ratio: float = 1e-7,
    weak_condition_threshold: float = 1e8,
    height_squared_tolerance_relative: float = 1e-9,
    positive_definite_relative_tolerance: float = 1e-10,
    angle_constraint: PlanarAngleConstraint | None = None,
) -> PlanarMetricResult:
    """Recover planar receivers, projected sources, and unsigned source heights."""
    if factorization.dimension != 2:
        raise ValueError("upgrade_metric_planar requires a rank-two factorization")
    ranges = np.asarray(corrected_ranges_m, dtype=float)
    if ranges.ndim != 2 or not np.all(np.isfinite(ranges)) or np.any(ranges < 0.0):
        raise ValueError("corrected_ranges_m must be finite and non-negative")
    if ranges.shape != (
        factorization.receiver_factors.shape[0],
        factorization.source_factors.shape[0],
    ):
        raise ValueError("corrected_ranges_m shape does not match factorization")
    if ranges.shape[0] < 6 or ranges.shape[1] < 4:
        raise ValueError("planar metric recovery requires at least 6 receivers and 4 events")

    positive_ranges = ranges[ranges > 0.0]
    if positive_ranges.size == 0:
        raise ValueError("corrected_ranges_m contains no positive ranges")
    range_scale_m = float(np.median(positive_ranges))
    normalized_ranges = ranges / range_scale_m
    receiver_factors = factorization.receiver_factors / range_scale_m
    source_factors = factorization.source_factors / range_scale_m

    identifiability = diagnose_planar_identifiability(receiver_factors)
    factor_singular = factorization.rank_diagnostics.singular_values
    factor_ratio = (
        0.0
        if factor_singular.size < 2 or factor_singular[0] <= 0.0
        else float(factor_singular[1] / factor_singular[0])
    )
    reasons = list(identifiability.reasons)
    if acceptance_rms_m is None:
        acceptance_rms_m = 1e-8 * range_scale_m
    if acceptance_rms_m <= 0.0:
        raise ValueError("acceptance_rms_m must be positive")

    if identifiability.continuous_metric_nullity > 0 and angle_constraint is None:
        return PlanarMetricResult(
            status="degenerate",
            candidates=(),
            diagnostics=PlanarMetricDiagnostics(
                identifiability=identifiability,
                factor_singular_ratio=factor_ratio,
                range_scale_m=range_scale_m,
                negative_height_count=0,
                near_zero_height_count=0,
                acceptance_rms_m=float(acceptance_rms_m),
                constraints_applied=(() if angle_constraint is None else ("planar_arm_angle",)),
                constraint_residual_rad=None,
                reasons=tuple(reasons),
            ),
        )

    parameters, _, rank, condition = _solve_planar_metric_linear(
        receiver_factors,
        normalized_ranges,
        angle_constraint=angle_constraint,
    )
    if rank < 5:
        if "receiver_metric_design_rank_deficient" not in reasons:
            reasons.append("receiver_metric_design_rank_deficient")
        return PlanarMetricResult(
            status="degenerate",
            candidates=(),
            diagnostics=PlanarMetricDiagnostics(
                identifiability=identifiability,
                factor_singular_ratio=factor_ratio,
                range_scale_m=range_scale_m,
                negative_height_count=0,
                near_zero_height_count=0,
                acceptance_rms_m=float(acceptance_rms_m),
                constraints_applied=(() if angle_constraint is None else ("planar_arm_angle",)),
                constraint_residual_rad=None,
                reasons=tuple(reasons),
            ),
        )

    if angle_constraint is not None and rank == 5:
        reasons = [
            reason
            for reason in reasons
            if reason
            not in {
                "receiver_metric_design_rank_deficient",
                "receiver_points_lie_on_nontrivial_conic",
            }
        ]

    h, b = _metric_from_parameters(parameters)
    eigenvalues = np.linalg.eigvalsh(h)
    pd_floor = positive_definite_relative_tolerance * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )
    if float(np.min(eigenvalues)) <= pd_floor:
        reasons.append("indefinite_planar_metric")
        return PlanarMetricResult(
            status="failed",
            candidates=(),
            diagnostics=PlanarMetricDiagnostics(
                identifiability=identifiability,
                factor_singular_ratio=factor_ratio,
                range_scale_m=range_scale_m,
                negative_height_count=0,
                near_zero_height_count=0,
                acceptance_rms_m=float(acceptance_rms_m),
                constraints_applied=(() if angle_constraint is None else ("planar_arm_angle",)),
                constraint_residual_rad=None,
                reasons=tuple(reasons),
            ),
        )

    try:
        l_factor = np.linalg.cholesky(h).T
    except np.linalg.LinAlgError:
        reasons.append("singular_planar_metric")
        return PlanarMetricResult(
            status="failed",
            candidates=(),
            diagnostics=PlanarMetricDiagnostics(
                identifiability=identifiability,
                factor_singular_ratio=factor_ratio,
                range_scale_m=range_scale_m,
                negative_height_count=0,
                near_zero_height_count=0,
                acceptance_rms_m=float(acceptance_rms_m),
                constraints_applied=(() if angle_constraint is None else ("planar_arm_angle",)),
                constraint_residual_rad=None,
                reasons=tuple(reasons),
            ),
        )

    microphones_2d = receiver_factors @ l_factor.T
    projected_sources = np.linalg.solve(
        l_factor.T,
        (source_factors + b[None, :]).T,
    ).T
    projected_norm_squared = np.sum(projected_sources * projected_sources, axis=1)
    height_squared = normalized_ranges[0] ** 2 - projected_norm_squared
    height_tolerance = height_squared_tolerance_relative * max(
        1.0,
        float(np.max(normalized_ranges[0] ** 2)),
    )
    negative = height_squared < -height_tolerance
    near_zero = np.abs(height_squared) <= height_tolerance
    if np.any(negative):
        reasons.append("negative_source_height_squared")
        return PlanarMetricResult(
            status="failed",
            candidates=(),
            diagnostics=PlanarMetricDiagnostics(
                identifiability=identifiability,
                factor_singular_ratio=factor_ratio,
                range_scale_m=range_scale_m,
                negative_height_count=int(np.sum(negative)),
                near_zero_height_count=int(np.sum(near_zero)),
                acceptance_rms_m=float(acceptance_rms_m),
                constraints_applied=(() if angle_constraint is None else ("planar_arm_angle",)),
                constraint_residual_rad=None,
                reasons=tuple(reasons),
            ),
        )
    height_squared = np.maximum(height_squared, 0.0)
    unsigned_heights = np.sqrt(height_squared)

    microphones_normalized = np.column_stack([microphones_2d, np.zeros(len(microphones_2d))])
    source_representative_normalized = np.column_stack([projected_sources, unsigned_heights])
    reconstructed = np.linalg.norm(
        microphones_normalized[:, None, :] - source_representative_normalized[None, :, :],
        axis=2,
    )
    range_error_m = (reconstructed - normalized_ranges) * range_scale_m
    receiver_design = planar_metric_design(receiver_factors)
    receiver_rhs = normalized_ranges[1:, 0] ** 2 - normalized_ranges[0, 0] ** 2
    receiver_residual_m2 = (receiver_design @ parameters - receiver_rhs) * range_scale_m**2
    constraint_residual_rad = None
    if angle_constraint is not None:
        recovered_angle = metric_angle_rad(
            receiver_factors,
            h,
            angle_constraint,
        )
        constraint_residual_rad = recovered_angle - angle_constraint.angle_rad
        if abs(constraint_residual_rad) > 1e-8:
            reasons.append("planar_angle_constraint_residual")

    microphones_m = microphones_normalized * range_scale_m
    sources_m = source_representative_normalized * range_scale_m
    canonical_microphones, canonical_sources, _ = canonicalize_scene_conditioned(
        microphones_m,
        sources_m,
    )
    assert canonical_sources is not None
    source_projected_m = canonical_sources[:, :2]
    source_unsigned_m = np.abs(canonical_sources[:, 2])
    sign_known = np.zeros(len(source_unsigned_m), dtype=bool)

    if factor_ratio < weak_factor_ratio:
        reasons.append("weak_planar_affine_rank_separation")
    if condition > weak_condition_threshold:
        reasons.append("ill_conditioned_planar_metric_design")
    corrected_rms = float(np.sqrt(np.mean(range_error_m * range_error_m)))
    if corrected_rms > acceptance_rms_m:
        reasons.append("planar_metric_range_tolerance_failed")
    status: PlanarMetricStatus = "solved" if not reasons else "weakly_identified"

    arrays = [
        canonical_microphones,
        source_projected_m,
        source_unsigned_m,
        sign_known,
        canonical_sources,
        h,
        b,
        eigenvalues,
    ]
    frozen: list[np.ndarray] = []
    for array in arrays:
        copy = np.array(array, copy=True)
        copy.setflags(write=False)
        frozen.append(copy)

    candidate = PlanarMetricCandidate(
        microphone_positions_m=frozen[0],
        source_projected_positions_m=frozen[1],
        source_unsigned_heights_m=frozen[2],
        source_height_sign_known=frozen[3],
        source_representative_positions_m=frozen[4],
        metric_matrix=frozen[5],
        metric_vector=frozen[6],
        metric_eigenvalues=frozen[7],
        receiver_linear_rms_m2=float(np.sqrt(np.mean(receiver_residual_m2 * receiver_residual_m2))),
        corrected_range_rms_m=corrected_rms,
        corrected_range_max_error_m=float(np.max(np.abs(range_error_m))),
        minimum_height_squared_m2=float(np.min(height_squared) * range_scale_m**2),
    )
    return PlanarMetricResult(
        status=status,
        candidates=(candidate,),
        diagnostics=PlanarMetricDiagnostics(
            identifiability=identifiability,
            factor_singular_ratio=factor_ratio,
            range_scale_m=range_scale_m,
            negative_height_count=int(np.sum(negative)),
            near_zero_height_count=int(np.sum(near_zero)),
            acceptance_rms_m=float(acceptance_rms_m),
            constraints_applied=(() if angle_constraint is None else ("planar_arm_angle",)),
            constraint_residual_rad=constraint_residual_rad,
            reasons=tuple(reasons),
        ),
    )


@dataclass(frozen=True)
class PlanarSourceLocalizationResult:
    projected_position_m: np.ndarray
    unsigned_height_m: float
    reference_range_m: float
    linear_rank: int
    linear_rms_m2: float
    height_squared_m2: float


@dataclass(frozen=True)
class PlanarReceiverLocalizationResult:
    position_2d_m: np.ndarray
    linear_rank: int
    range_rms_m: float


def localize_planar_source_from_tdoa(
    microphone_positions_m: np.ndarray,
    relative_arrivals_m: np.ndarray,
) -> PlanarSourceLocalizationResult:
    """Localize one source projection/reference range and recover unsigned height."""
    microphones = np.asarray(microphone_positions_m, dtype=float)
    relative = np.asarray(relative_arrivals_m, dtype=float).reshape(-1)
    if microphones.ndim != 2 or microphones.shape[1] != 3:
        raise ValueError("microphone_positions_m must have shape (M, 3)")
    if relative.shape != (microphones.shape[0],):
        raise ValueError("relative_arrivals_m must have one value per microphone")
    if microphones.shape[0] < 4:
        raise ValueError("at least four planar microphones are required")
    if np.max(np.abs(microphones[:, 2])) > 1e-8:
        raise ValueError("microphones must lie in the canonical z=0 plane")

    receiver_2d = microphones[:, :2]
    reference = receiver_2d[0]
    rows = []
    rhs = []
    for index in range(1, len(receiver_2d)):
        point = receiver_2d[index]
        f_i = float(relative[index])
        rows.append(
            [
                2.0 * (point[0] - reference[0]),
                2.0 * (point[1] - reference[1]),
                2.0 * f_i,
            ]
        )
        rhs.append(float(np.dot(point, point) - np.dot(reference, reference) - f_i * f_i))
    design = np.asarray(rows, dtype=float)
    target = np.asarray(rhs, dtype=float)
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    projected = solution[:2]
    reference_range = float(solution[2])
    if reference_range < 0.0:
        raise ValueError("planar source localization implies negative reference range")
    residual = design @ solution - target
    height_squared = reference_range**2 - float(np.sum((projected - reference) ** 2))
    tolerance = 1e-9 * max(1.0, reference_range**2)
    if height_squared < -tolerance:
        raise ValueError("planar source localization implies negative height squared")
    height_squared = max(height_squared, 0.0)
    projected_copy = np.array(projected, copy=True)
    projected_copy.setflags(write=False)
    return PlanarSourceLocalizationResult(
        projected_position_m=projected_copy,
        unsigned_height_m=float(np.sqrt(height_squared)),
        reference_range_m=reference_range,
        linear_rank=int(rank),
        linear_rms_m2=float(np.sqrt(np.mean(residual * residual))),
        height_squared_m2=float(height_squared),
    )


def localize_planar_receiver_from_ranges(
    source_projected_positions_m: np.ndarray,
    source_unsigned_heights_m: np.ndarray,
    ranges_m: np.ndarray,
) -> PlanarReceiverLocalizationResult:
    """Recover one receiver in the canonical plane from unsigned source observables."""
    projected = np.asarray(source_projected_positions_m, dtype=float)
    heights = np.asarray(source_unsigned_heights_m, dtype=float).reshape(-1)
    ranges = np.asarray(ranges_m, dtype=float).reshape(-1)
    if projected.ndim != 2 or projected.shape[1] != 2:
        raise ValueError("source_projected_positions_m must have shape (E, 2)")
    if heights.shape != (len(projected),) or ranges.shape != (len(projected),):
        raise ValueError("height/range arrays must match source count")
    if len(projected) < 3:
        raise ValueError("at least three sources are required for planar receiver localization")
    planar_squared = ranges * ranges - heights * heights
    tolerance = 1e-9 * max(1.0, float(np.max(ranges * ranges)))
    if np.min(planar_squared) < -tolerance:
        raise ValueError("ranges are incompatible with unsigned source heights")
    planar_ranges = np.sqrt(np.maximum(planar_squared, 0.0))

    reference = projected[0]
    reference_range = float(planar_ranges[0])
    design = []
    rhs = []
    for event in range(1, len(projected)):
        source = projected[event]
        design.append(2.0 * (source - reference))
        rhs.append(
            float(
                np.dot(source, source)
                - np.dot(reference, reference)
                - (planar_ranges[event] ** 2 - reference_range**2)
            )
        )
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(rhs, dtype=float)
    receiver, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    predicted = np.sqrt(np.sum((projected - receiver[None, :]) ** 2, axis=1) + heights * heights)
    receiver_copy = np.array(receiver, copy=True)
    receiver_copy.setflags(write=False)
    return PlanarReceiverLocalizationResult(
        position_2d_m=receiver_copy,
        linear_rank=int(rank),
        range_rms_m=float(np.sqrt(np.mean((predicted - ranges) ** 2))),
    )
