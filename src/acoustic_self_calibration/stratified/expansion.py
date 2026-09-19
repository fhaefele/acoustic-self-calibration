from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from .notation import compaction_matrix, corrected_ranges_from_offsets


@dataclass(frozen=True)
class OffsetExpansionResult:
    offsets_m: np.ndarray
    corrected_ranges_m: np.ndarray
    column_space_rank: int
    left_nullity: int
    membership_rms: np.ndarray
    success: np.ndarray


@dataclass(frozen=True)
class SourceLocalizationResult:
    source_position_m: np.ndarray
    reference_range_m: float
    linear_rank: int
    linear_rms_m2: float
    norm_residual_m2: float


@dataclass(frozen=True)
class ReceiverLocalizationResult:
    receiver_position_m: np.ndarray
    linear_rank: int
    range_rms_m: float
    inlier_rms_m: float


def extend_event_offsets(
    seed_relative_arrivals_m: np.ndarray,
    seed_offsets_m: np.ndarray,
    target_relative_arrivals_m: np.ndarray,
    *,
    membership_tolerance: float = 1e-7,
) -> OffsetExpansionResult:
    """Extend event offsets using the receiver-differenced squared-range column space."""
    seed_arrivals = np.asarray(seed_relative_arrivals_m, dtype=float)
    target_arrivals = np.asarray(target_relative_arrivals_m, dtype=float)
    seed_offsets = np.asarray(seed_offsets_m, dtype=float).reshape(-1)
    if seed_arrivals.ndim != 2 or target_arrivals.ndim != 2:
        raise ValueError("arrival matrices must be two-dimensional")
    if seed_arrivals.shape[0] != target_arrivals.shape[0]:
        raise ValueError("seed and target arrivals must use the same receivers")
    if seed_offsets.shape != (seed_arrivals.shape[1],):
        raise ValueError("seed_offsets_m must match seed events")
    if not np.all(np.isfinite(seed_arrivals)) or not np.all(np.isfinite(target_arrivals)):
        raise ValueError("arrival matrices must be finite")
    if not np.all(np.isfinite(seed_offsets)):
        raise ValueError("seed_offsets_m must be finite")
    if membership_tolerance <= 0.0:
        raise ValueError("membership_tolerance must be positive")

    corrected_seed = corrected_ranges_from_offsets(seed_arrivals, seed_offsets)
    receiver_compaction = compaction_matrix(seed_arrivals.shape[0])
    seed_columns = receiver_compaction.T @ (corrected_seed * corrected_seed)
    u, singular_values, _ = np.linalg.svd(seed_columns, full_matrices=True)
    tolerance = (
        np.finfo(float).eps * max(seed_columns.shape) * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    rank = int(np.sum(singular_values > tolerance))
    left_null = u[:, rank:]
    if left_null.shape[1] == 0:
        raise ValueError("seed squared-range columns leave no offset-membership constraint")

    event_count = target_arrivals.shape[1]
    offsets = np.full(event_count, np.nan, dtype=float)
    corrected = np.full_like(target_arrivals, np.nan)
    membership_rms = np.full(event_count, np.inf, dtype=float)
    success = np.zeros(event_count, dtype=bool)

    for event in range(event_count):
        relative = target_arrivals[:, event]
        quadratic = receiver_compaction.T @ (relative * relative)
        linear = receiver_compaction.T @ relative
        coefficient = -2.0 * (left_null.T @ linear)
        rhs = -(left_null.T @ quadratic)
        denominator = float(np.dot(coefficient, coefficient))
        scale = max(
            1.0,
            float(np.linalg.norm(quadratic)),
            float(np.linalg.norm(linear)),
        )
        if denominator <= np.finfo(float).eps * scale * scale:
            continue
        offset = float(np.dot(coefficient, rhs) / denominator)
        residual = left_null.T @ (quadratic - 2.0 * linear * offset)
        rms = float(np.sqrt(np.mean(residual * residual))) / scale
        candidate_ranges = relative - offset
        if rms > membership_tolerance or float(np.min(candidate_ranges)) < -1e-8 * scale:
            continue

        offsets[event] = offset
        corrected[:, event] = np.maximum(candidate_ranges, 0.0)
        membership_rms[event] = rms
        success[event] = True

    offsets.setflags(write=False)
    corrected.setflags(write=False)
    membership_rms.setflags(write=False)
    success.setflags(write=False)
    return OffsetExpansionResult(
        offsets_m=offsets,
        corrected_ranges_m=corrected,
        column_space_rank=rank,
        left_nullity=int(left_null.shape[1]),
        membership_rms=membership_rms,
        success=success,
    )


def localize_source_from_tdoa(
    receiver_positions_m: np.ndarray,
    relative_arrivals_m: np.ndarray,
) -> SourceLocalizationResult:
    """Localize one source and its reference range from known receivers and TDOA."""
    receivers = np.asarray(receiver_positions_m, dtype=float)
    relative = np.asarray(relative_arrivals_m, dtype=float).reshape(-1)
    if receivers.ndim != 2 or receivers.shape[1] != 3:
        raise ValueError("receiver_positions_m must have shape (M, 3)")
    if relative.shape != (receivers.shape[0],):
        raise ValueError("relative_arrivals_m must have one value per receiver")
    if receivers.shape[0] < 5:
        raise ValueError("at least five receivers are required for linear 3-D localization")
    if not np.all(np.isfinite(receivers)) or not np.all(np.isfinite(relative)):
        raise ValueError("localization inputs must be finite")

    reference = receivers[0]
    design = []
    rhs = []
    for receiver in range(1, receivers.shape[0]):
        point = receivers[receiver]
        f_i = float(relative[receiver])
        design.append(
            np.concatenate(
                [
                    2.0 * (point - reference),
                    np.array([2.0 * f_i]),
                ]
            )
        )
        rhs.append(float(np.dot(point, point) - np.dot(reference, reference) - f_i * f_i))
    matrix = np.asarray(design, dtype=float)
    vector = np.asarray(rhs, dtype=float)
    solution, _, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
    source = solution[:3]
    reference_range = float(solution[3])
    residual = matrix @ solution - vector
    norm_residual = reference_range * reference_range - float(np.sum((source - reference) ** 2))
    source_copy = np.array(source, copy=True)
    source_copy.setflags(write=False)
    return SourceLocalizationResult(
        source_position_m=source_copy,
        reference_range_m=reference_range,
        linear_rank=int(rank),
        linear_rms_m2=float(np.sqrt(np.mean(residual * residual))),
        norm_residual_m2=norm_residual,
    )


def localize_receiver_from_ranges(
    source_positions_m: np.ndarray,
    ranges_m: np.ndarray,
    *,
    robust: bool = False,
    huber_scale_m: float = 0.02,
) -> ReceiverLocalizationResult:
    """Trilaterate one receiver from known sources and absolute ranges.

    Robust mode evaluates deterministic four-event minimal subsets using median
    all-data range error. A Huber polish is accepted only if it improves that median.
    Raw RMS still reports gross outliers; inlier_rms_m trims the largest 20 percent.
    """
    sources = np.asarray(source_positions_m, dtype=float)
    ranges = np.asarray(ranges_m, dtype=float).reshape(-1)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("source_positions_m must have shape (E, 3)")
    if ranges.shape != (sources.shape[0],):
        raise ValueError("ranges_m must contain one range per source")
    if sources.shape[0] < 4:
        raise ValueError("at least four sources are required for 3-D receiver localization")
    if not np.all(np.isfinite(sources)) or not np.all(np.isfinite(ranges)):
        raise ValueError("receiver localization inputs must be finite")
    if np.any(ranges < 0.0):
        raise ValueError("ranges_m must be non-negative")
    if robust and huber_scale_m <= 0.0:
        raise ValueError("huber_scale_m must be positive")

    def linear_solution(indices: np.ndarray) -> tuple[np.ndarray, int]:
        local_sources = sources[indices]
        local_ranges = ranges[indices]
        reference = local_sources[0]
        reference_range = float(local_ranges[0])
        matrix = []
        rhs = []
        for event in range(1, len(indices)):
            source = local_sources[event]
            matrix.append(2.0 * (source - reference))
            rhs.append(
                float(
                    np.dot(source, source)
                    - np.dot(reference, reference)
                    - (local_ranges[event] ** 2 - reference_range**2)
                )
            )
        design = np.asarray(matrix, dtype=float)
        vector = np.asarray(rhs, dtype=float)
        receiver, _, rank, _ = np.linalg.lstsq(design, vector, rcond=None)
        return receiver, int(rank)

    all_indices = np.arange(len(sources), dtype=int)
    receiver, rank = linear_solution(all_indices)
    if robust and len(sources) > 4:
        best_receiver = receiver
        best_rank = rank
        best_score = float(
            np.median(np.abs(np.linalg.norm(sources - receiver[None, :], axis=1) - ranges))
        )
        exact_tolerance = 1e-10 * max(1.0, float(np.max(ranges)))
        if best_score > exact_tolerance:
            for subset in combinations(range(len(sources)), 4):
                candidate, candidate_rank = linear_solution(np.asarray(subset, dtype=int))
                if candidate_rank < 3:
                    continue
                residual = np.abs(np.linalg.norm(sources - candidate[None, :], axis=1) - ranges)
                score = float(np.median(residual))
                if score < best_score:
                    best_receiver = candidate
                    best_rank = candidate_rank
                    best_score = score

            polished = least_squares(
                lambda point: np.linalg.norm(sources - point[None, :], axis=1) - ranges,
                best_receiver,
                loss="huber",
                f_scale=huber_scale_m,
                max_nfev=200,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            polished_receiver = np.asarray(polished.x, dtype=float)
            polished_score = float(
                np.median(
                    np.abs(np.linalg.norm(sources - polished_receiver[None, :], axis=1) - ranges)
                )
            )
            if polished_score < best_score:
                receiver = polished_receiver
            else:
                receiver = best_receiver
            rank = best_rank

    residual = np.linalg.norm(sources - receiver[None, :], axis=1) - ranges
    absolute = np.abs(residual)
    keep_count = max(4, int(np.ceil(0.8 * len(residual))))
    keep = np.argsort(absolute)[:keep_count]
    receiver_copy = np.array(receiver, copy=True)
    receiver_copy.setflags(write=False)
    return ReceiverLocalizationResult(
        receiver_position_m=receiver_copy,
        linear_rank=int(rank),
        range_rms_m=float(np.sqrt(np.mean(residual * residual))),
        inlier_rms_m=float(np.sqrt(np.mean(residual[keep] * residual[keep]))),
    )
