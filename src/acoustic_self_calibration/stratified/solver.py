from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.optimize import least_squares, minimize

from ..geometry import rigid_align, rms_position_error
from ..measurements import EventTDOAMeasurements
from .constraints import PlanarAngleConstraint
from .expansion import (
    extend_event_offsets,
    localize_receiver_from_ranges,
    localize_source_from_tdoa,
)
from .factorization import factor_corrected_ranges
from .hypotheses import (
    FullGeometryHypothesis,
    HypothesisClass,
    MeasurementSplit,
    cluster_equivalent_hypotheses,
)
from .metric_upgrade import upgrade_metric_3d_overdetermined
from .model_selection import (
    ModelComparisonResult,
    ModelValidationEvidence,
    compare_model_evidence,
)
from .offsets_linear import solve_linear_offsets
from .offsets_minimal import solve_offsets_7r6s
from .planar import (
    localize_planar_receiver_from_ranges,
    localize_planar_source_from_tdoa,
    upgrade_metric_planar,
)
from .robustness import (
    RobustResidualSummary,
    conditional_covariance,
    huber_loss,
    summarize_independent_residuals,
    whiten_residual_block,
)

CalibrationStatus = Literal[
    "solved",
    "ambiguous",
    "weakly_identified",
    "degenerate",
    "insufficient_data",
    "failed",
]


@dataclass(frozen=True)
class StratifiedCalibrationDiagnostics:
    attempted_subsets: int
    generated_offset_roots: int
    metric_candidates: int
    completed_hypotheses: int
    geometric_class_count: int
    selected_support: int
    fitting_event_count: int
    validation_event_count: int
    validation_independent_coordinates: int
    rejection_reasons: tuple[str, ...]
    extra_microphones_completed: int = 0
    extra_microphone_max_inlier_rms_m: float | None = None


@dataclass(frozen=True)
class StratifiedCalibrationResult:
    status: CalibrationStatus
    microphone_positions_m: np.ndarray | None
    source_positions_m: np.ndarray | None
    event_ids: np.ndarray
    tdoa_rms_s: float | None
    selected_class: HypothesisClass | None
    classes: tuple[HypothesisClass, ...]
    diagnostics: StratifiedCalibrationDiagnostics
    microphone_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlanarCalibrationResult:
    status: CalibrationStatus
    microphone_positions_m: np.ndarray | None
    source_projected_positions_m: np.ndarray | None
    source_unsigned_heights_m: np.ndarray | None
    source_height_sign_known: np.ndarray | None
    source_representative_positions_m: np.ndarray | None
    event_ids: np.ndarray
    validation: RobustResidualSummary | None
    tdoa_rms_s: float | None
    diagnostics: StratifiedCalibrationDiagnostics
    microphone_ids: tuple[int, ...] = ()
    continuous_ambiguity_dimension: int | None = None
    continuous_ambiguity_nullspace: np.ndarray | None = None
    angle_constraint: PlanarAngleConstraint | None = None


@dataclass(frozen=True)
class JointModelCalibrationResult:
    comparison: ModelComparisonResult
    planar: PlanarCalibrationResult
    general_3d: StratifiedCalibrationResult


def _measurement_transform_quality(
    measurements: EventTDOAMeasurements,
    transform: np.ndarray,
) -> tuple[int, float]:
    """Score a candidate reference/order by supported coordinates and marginal sigma."""
    supports = [np.flatnonzero(np.abs(row) > 0.0) for row in transform]
    supported = 0
    variances: list[float] = []
    for event in range(len(measurements.event_ids)):
        for row, support in enumerate(supports):
            if support.size == 0 or not np.all(measurements.valid[event, support]):
                continue
            coefficients = transform[row, support]
            supported += 1
            if measurements.covariance_s2 is None:
                variance = float(
                    np.sum(coefficients * coefficients * measurements.sigma_s[event, support] ** 2)
                )
            else:
                covariance = measurements.covariance_s2[event][np.ix_(support, support)]
                variance = float(coefficients @ covariance @ coefficients)
            variances.append(max(variance, 0.0))
    if not variances:
        return supported, float("inf")
    return supported, float(np.sqrt(np.median(np.asarray(variances))))


def _reference_transform(
    arrivals_from_pairs: np.ndarray,
    *,
    reference_input_index: int,
    target_input_indices: tuple[int, ...],
) -> np.ndarray:
    incidence = np.zeros(
        (len(target_input_indices), arrivals_from_pairs.shape[0]),
        dtype=float,
    )
    for row, target in enumerate(target_input_indices):
        incidence[row, reference_input_index] = -1.0
        incidence[row, target] = 1.0
    return incidence @ arrivals_from_pairs


def _normalize_reference_star_measurements(
    measurements: EventTDOAMeasurements,
    *,
    required_microphone_ids: tuple[int, ...] = (),
) -> tuple[EventTDOAMeasurements, tuple[int, ...]]:
    """Choose a stable reference/order and convert to the solver's dense star convention."""
    if measurements.measurement_basis != "reference_star":
        raise ValueError("calibration requires reference_star measurements")

    microphone_ids = measurements.microphone_ids
    microphone_count = len(microphone_ids)
    if len(measurements.microphone_pairs) != microphone_count - 1:
        raise ValueError("reference_star measurements require exactly M-1 pairs")

    common = set(measurements.microphone_pairs[0])
    for pair in measurements.microphone_pairs[1:]:
        common.intersection_update(pair)
    if len(common) != 1:
        raise ValueError("reference_star pairs must share exactly one reference microphone")
    old_reference = next(iter(common))

    id_to_input = {microphone_id: index for index, microphone_id in enumerate(microphone_ids)}
    if old_reference not in id_to_input:
        raise ValueError("reference_star reference is not present in microphone_ids")
    try:
        required_input = tuple(id_to_input[value] for value in required_microphone_ids)
    except KeyError as error:
        raise ValueError("required microphone ID is not present in microphone_ids") from error
    if len(set(required_input)) != len(required_input):
        raise ValueError("required microphone IDs must be unique")
    if len(required_input) > 8:
        raise ValueError("at most eight required seed microphones are supported")

    pair_count = len(measurements.microphone_pairs)
    arrivals_from_pairs = np.zeros((microphone_count, pair_count), dtype=float)
    seen_targets: set[int] = set()
    for pair_index, (a, b) in enumerate(measurements.microphone_pairs):
        if a == old_reference:
            target = b
            sign = 1.0
        elif b == old_reference:
            target = a
            sign = -1.0
        else:
            raise ValueError("reference_star pair does not contain the common reference")
        if target in seen_targets:
            raise ValueError("reference_star contains a duplicate target microphone")
        seen_targets.add(target)
        arrivals_from_pairs[id_to_input[target], pair_index] = sign

    expected_targets = set(microphone_ids) - {old_reference}
    if seen_targets != expected_targets:
        raise ValueError("reference_star does not cover every non-reference microphone")

    reference_scores: list[tuple[int, float, int]] = []
    for candidate in range(microphone_count):
        targets = tuple(index for index in range(microphone_count) if index != candidate)
        transform = _reference_transform(
            arrivals_from_pairs,
            reference_input_index=candidate,
            target_input_indices=targets,
        )
        supported, sigma = _measurement_transform_quality(measurements, transform)
        reference_scores.append((-supported, sigma, candidate))
    _, _, reference_input = min(reference_scores)

    required_targets = [index for index in required_input if index != reference_input]
    target_scores: list[tuple[int, float, int]] = []
    for target in range(microphone_count):
        if target == reference_input:
            continue
        transform = _reference_transform(
            arrivals_from_pairs,
            reference_input_index=reference_input,
            target_input_indices=(target,),
        )
        supported, sigma = _measurement_transform_quality(measurements, transform)
        target_scores.append((-supported, sigma, target))
    target_scores.sort()
    ordered_targets = [target for _, _, target in target_scores]

    seed_target_count = min(7, len(ordered_targets))
    required_set = set(required_targets)
    for required in required_targets:
        if required in ordered_targets[:seed_target_count]:
            continue
        required_position = ordered_targets.index(required)
        replacement_position = next(
            (
                position
                for position in range(seed_target_count - 1, -1, -1)
                if ordered_targets[position] not in required_set
            ),
            None,
        )
        if replacement_position is None:
            raise ValueError("required receivers exceed the available eight-receiver seed")
        ordered_targets[replacement_position], ordered_targets[required_position] = (
            ordered_targets[required_position],
            ordered_targets[replacement_position],
        )

    internal_to_input = (reference_input, *ordered_targets)

    transform = _reference_transform(
        arrivals_from_pairs,
        reference_input_index=reference_input,
        target_input_indices=tuple(ordered_targets),
    )
    dense_ids = tuple(range(microphone_count))
    new_pairs = tuple((0, index) for index in range(1, microphone_count))

    event_count = len(measurements.event_ids)
    new_tdoa = np.full((event_count, microphone_count - 1), np.nan, dtype=float)
    new_sigma = np.full_like(new_tdoa, np.nan)
    new_confidence = np.full_like(new_tdoa, np.nan)
    new_valid = np.zeros_like(new_tdoa, dtype=bool)
    new_covariance = np.full(
        (event_count, microphone_count - 1, microphone_count - 1),
        np.nan,
        dtype=float,
    )

    supports = [np.flatnonzero(np.abs(row) > 0.0) for row in transform]
    for event in range(event_count):
        for row, support in enumerate(supports):
            if support.size == 0 or not np.all(measurements.valid[event, support]):
                continue
            coefficients = transform[row, support]
            new_valid[event, row] = True
            new_tdoa[event, row] = float(coefficients @ measurements.tdoa_s[event, support])
            new_confidence[event, row] = float(np.min(measurements.confidence[event, support]))

        valid_rows = np.flatnonzero(new_valid[event])
        if valid_rows.size == 0:
            continue
        old_valid = np.flatnonzero(measurements.valid[event])
        if measurements.covariance_s2 is None:
            old_covariance = np.diag(measurements.sigma_s[event, old_valid] ** 2)
        else:
            old_covariance = measurements.covariance_s2[event][np.ix_(old_valid, old_valid)]
        event_transform = transform[np.ix_(valid_rows, old_valid)]
        covariance = event_transform @ old_covariance @ event_transform.T
        new_covariance[event][np.ix_(valid_rows, valid_rows)] = covariance
        diagonal = np.maximum(np.diag(covariance), 0.0)
        new_sigma[event, valid_rows] = np.sqrt(diagonal)

    primitive_map = None
    if measurements.pair_from_primitive is not None:
        primitive_map = transform @ measurements.pair_from_primitive

    event_channel = None
    if measurements.event_channel is not None:
        event_input = id_to_input[measurements.event_channel]
        event_channel = internal_to_input.index(event_input)

    arrival_representatives = None
    arrival_valid = None
    if measurements.arrival_representatives_s is not None:
        reordered_arrivals = measurements.arrival_representatives_s[:, internal_to_input]
        assert measurements.arrival_valid is not None
        reordered_valid = measurements.arrival_valid[:, internal_to_input]
        arrival_valid = reordered_valid & reordered_valid[:, [0]]
        arrival_representatives = reordered_arrivals - reordered_arrivals[:, [0]]
        arrival_representatives = np.array(
            arrival_representatives,
            copy=True,
        )
        arrival_representatives[~arrival_valid] = np.nan

    normalized = EventTDOAMeasurements(
        event_ids=measurements.event_ids,
        receiver_event_times_s=measurements.receiver_event_times_s,
        microphone_ids=dense_ids,
        microphone_pairs=new_pairs,
        tdoa_s=new_tdoa,
        sigma_s=new_sigma,
        confidence=new_confidence,
        valid=new_valid,
        measurement_origin=measurements.measurement_origin,
        measurement_basis="reference_star",
        event_samples=measurements.event_samples,
        sample_rate_hz=measurements.sample_rate_hz,
        event_channel=event_channel,
        primitive_ids=measurements.primitive_ids,
        pair_from_primitive=primitive_map,
        covariance_s2=new_covariance,
        covariance_model=f"normalized_reference_star:{measurements.covariance_model}",
        arrival_representatives_s=arrival_representatives,
        arrival_valid=arrival_valid,
    )
    return normalized, tuple(int(value) for value in internal_to_input)


def _reference_change_gauge_shifts_s(
    measurements: EventTDOAMeasurements,
    *,
    new_reference_input_index: int,
) -> np.ndarray:
    """Return q_j for a reference change f'_ij=f_ij+q_j in seconds."""
    microphone_ids = measurements.microphone_ids
    if not 0 <= new_reference_input_index < len(microphone_ids):
        raise ValueError("new reference input index is out of range")
    common = set(measurements.microphone_pairs[0])
    for pair in measurements.microphone_pairs[1:]:
        common.intersection_update(pair)
    if len(common) != 1:
        raise ValueError("reference_star pairs must share one reference")
    old_reference = next(iter(common))
    new_reference = microphone_ids[new_reference_input_index]
    shifts = np.zeros(len(measurements.event_ids), dtype=float)
    if new_reference == old_reference:
        shifts.setflags(write=False)
        return shifts

    pair_index = None
    sign = 1.0
    for index, (a, b) in enumerate(measurements.microphone_pairs):
        if a == old_reference and b == new_reference:
            pair_index = index
            sign = 1.0
            break
        if a == new_reference and b == old_reference:
            pair_index = index
            sign = -1.0
            break
    if pair_index is None:
        raise ValueError("reference_star is missing the new-reference edge")
    shifts[:] = np.nan
    valid = measurements.valid[:, pair_index]
    old_reference_arrival = sign * measurements.tdoa_s[valid, pair_index]
    shifts[valid] = -old_reference_arrival
    shifts.setflags(write=False)
    return shifts


def _reference_star_arrivals_m(
    measurements: EventTDOAMeasurements,
    speed_of_sound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")
    if measurements.measurement_basis != "reference_star":
        raise ValueError("Milestone C solver requires reference_star measurements")
    microphone_count = len(measurements.microphone_ids)
    if measurements.microphone_ids != tuple(range(microphone_count)):
        raise ValueError("Milestone C solver currently requires contiguous microphone IDs")
    if measurements.microphone_pairs != tuple((0, index) for index in range(1, microphone_count)):
        raise ValueError("reference_star pairs must be oriented (0, microphone)")
    event_count = len(measurements.event_ids)
    arrivals = np.full((microphone_count, event_count), np.nan, dtype=float)
    sigma = np.full_like(arrivals, np.nan)
    valid = np.zeros_like(arrivals, dtype=bool)
    arrivals[0] = 0.0
    sigma[0] = 0.0
    valid[0] = True
    arrivals[1:] = (measurements.tdoa_s * speed_of_sound).T
    sigma[1:] = (measurements.sigma_s * speed_of_sound).T
    valid[1:] = measurements.valid.T
    return arrivals, sigma, valid


def _split_events(event_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if event_count < 12:
        raise ValueError("at least 12 events are required for the Milestone C solver")
    validation_count = 8 if event_count < 32 else 16
    validation = tuple(
        sorted(
            set(
                int(value)
                for value in np.linspace(
                    1,
                    event_count - 2,
                    validation_count,
                    dtype=int,
                )
            )
        )
    )
    if len(validation) < validation_count:
        remaining = [index for index in range(event_count) if index not in validation]
        validation = tuple(
            sorted(validation + tuple(remaining[: validation_count - len(validation)]))
        )
    fitting = tuple(index for index in range(event_count) if index not in validation)
    if len(fitting) < 6:
        raise ValueError("event split leaves fewer than six fitting events")
    return fitting, validation


def _seed_event_families(
    fitting_events: tuple[int, ...],
    *,
    budget: int,
    seed_size: int = 6,
) -> tuple[tuple[int, ...], ...]:
    if budget < 1:
        raise ValueError("subset budget must be positive")
    values = np.asarray(fitting_events, dtype=int)
    families: list[tuple[int, ...]] = []
    phases = np.linspace(0.0, 0.75, max(budget, 1), endpoint=True)
    for phase in phases:
        positions = np.linspace(
            phase,
            len(values) - 1 - (0.75 - phase),
            seed_size,
        )
        selected = tuple(sorted({int(values[int(round(position))]) for position in positions}))
        if len(selected) == seed_size and selected not in families:
            families.append(selected)
        if len(families) >= budget:
            break
    if not families:
        families.append(tuple(int(value) for value in values[:seed_size]))
    return tuple(families)


def _receiver_subsets(microphone_count: int, *, budget: int) -> tuple[tuple[int, ...], ...]:
    if microphone_count != 8:
        raise ValueError("Milestone C production gate currently supports exactly 8 microphones")
    subsets = []
    for excluded in range(microphone_count - 1, 0, -1):
        subset = tuple(index for index in range(microphone_count) if index != excluded)
        subsets.append(subset)
        if len(subsets) >= budget:
            break
    return tuple(subsets)


def _tdoa_predictions(
    microphones: np.ndarray,
    sources: np.ndarray,
    speed_of_sound: float,
) -> np.ndarray:
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    return ((ranges[1:] - ranges[[0]]) / speed_of_sound).T


def _localization_receivers_for_event(
    primitive_valid: np.ndarray,
    event: int,
    *,
    receiver_count: int = 8,
) -> tuple[int, ...] | None:
    valid_nonreference = [
        receiver for receiver in range(1, receiver_count) if primitive_valid[receiver, event]
    ]
    if len(valid_nonreference) < 4:
        return None
    return (0, *valid_nonreference[:4])


def _complete_geometry(
    *,
    arrivals_m: np.ndarray,
    primitive_valid: np.ndarray,
    fitting_events: tuple[int, ...],
    validation_events: tuple[int, ...],
    receiver_subset: tuple[int, ...],
    seed_events: tuple[int, ...],
    seed_root_offsets_m: np.ndarray,
    speed_of_sound: float,
    measurements: EventTDOAMeasurements,
    subset_id: str,
    root_id: int,
    root_generating_equation_rms: float,
    metric_start_count: int,
) -> (
    tuple[FullGeometryHypothesis, np.ndarray]
    | list[tuple[FullGeometryHypothesis, np.ndarray]]
    | None
):
    seed_receivers = np.asarray(receiver_subset, dtype=int)
    seed = np.asarray(seed_events, dtype=int)
    excluded = next(index for index in range(8) if index not in receiver_subset)

    complete_fitting = np.asarray(
        [event for event in fitting_events if np.all(primitive_valid[seed_receivers, event])],
        dtype=int,
    )
    if len(complete_fitting) < 6:
        return None

    # Noise-aware tolerances derived from measurement uncertainty (T-004).
    # Audio-extracted TDOAs carry ~6us RMS error with median sigma ~60us, while
    # exact TDOAs use sigma 2us. Fixed micron tolerances (20um metric, 10um
    # receiver, 1e-7 membership) reject all noisy hypotheses. The metric
    # acceptance is a completion filter (not a quality certificate): noisy
    # metric branches carry algebraic approximation error well above 3-sigma
    # (observed 0.06-0.12m), so the noisy regime admits branches within the
    # deterministic polish basin (0.1m) and leaves final quality to held-out
    # validation selection, polish, and the acceptance gates.
    valid_sigma = measurements.sigma_s[measurements.valid]
    valid_sigma = valid_sigma[np.isfinite(valid_sigma)]
    median_sigma_s = float(np.median(valid_sigma)) if valid_sigma.size else 2e-6
    sigma_range_m = median_sigma_s * speed_of_sound
    if sigma_range_m > 5e-3:
        acceptance_rms_m = 0.1
    else:
        acceptance_rms_m = 2e-5
    membership_tolerance = min(5e-3, max(1e-7, acceptance_rms_m / 4.0))
    receiver_inlier_tolerance_m = max(1e-5, acceptance_rms_m)

    try:
        expanded = extend_event_offsets(
            arrivals_m[np.ix_(seed_receivers, seed)],
            seed_root_offsets_m,
            arrivals_m[np.ix_(seed_receivers, complete_fitting)],
            membership_tolerance=membership_tolerance,
        )
    except ValueError:
        return None
    successful_events = complete_fitting[expanded.success]
    successful_offsets = expanded.offsets_m[expanded.success]
    successful_ranges = expanded.corrected_ranges_m[:, expanded.success]
    if len(successful_events) < 6:
        return None

    try:
        factorization = factor_corrected_ranges(
            successful_ranges,
            dimension=3,
        )
    except ValueError:
        return None
    metric = upgrade_metric_3d_overdetermined(
        factorization,
        successful_ranges,
        start_count=metric_start_count,
        acceptance_rms_m=acceptance_rms_m,
    )
    accepted = [
        candidate
        for candidate in metric.candidates
        if candidate.corrected_range_rms_m <= metric.diagnostics.acceptance_rms_m
    ]
    if not accepted:
        return None
    # Root-local relative gate (T-004). The absolute noisy acceptance
    # (0.1m) admits far branches (0.05m+) alongside the root's best
    # (0.015m); those distractors merge with / outvote the good branch
    # downstream. Keep only branches near the root's best so each root
    # contributes its most self-consistent completion(s); cross-root
    # selection stays with held-out validation + polish. Exact regime is
    # unaffected: the 0.02 floor admits everything below 2e-5.
    root_best = min(candidate.corrected_range_rms_m for candidate in accepted)
    branch_cap = max(0.02, 2.0 * root_best)
    accepted = [
        candidate for candidate in accepted if candidate.corrected_range_rms_m <= branch_cap
    ]
    if not accepted:
        return None

    completed_hypotheses: list[tuple[FullGeometryHypothesis, np.ndarray]] = []
    for metric_branch_id, metric_candidate in enumerate(accepted):
        excluded_fit_mask = np.asarray(
            [primitive_valid[excluded, event] for event in successful_events],
            dtype=bool,
        )
        if int(np.sum(excluded_fit_mask)) < 4:
            continue
        excluded_ranges = (
            arrivals_m[excluded, successful_events[excluded_fit_mask]]
            - successful_offsets[excluded_fit_mask]
        )
        localized_receiver = localize_receiver_from_ranges(
            metric_candidate.source_positions_m[excluded_fit_mask],
            excluded_ranges,
            robust=True,
        )
        if localized_receiver.linear_rank < 3 or (
            localized_receiver.inlier_rms_m > receiver_inlier_tolerance_m
        ):
            continue

        microphones = np.empty((8, 3), dtype=float)
        for local_index, microphone_id in enumerate(receiver_subset):
            microphones[microphone_id] = metric_candidate.microphone_positions_m[local_index]
        microphones[excluded] = localized_receiver.receiver_position_m

        source_positions = np.full((len(measurements.event_ids), 3), np.nan, dtype=float)
        source_positions[successful_events] = metric_candidate.source_positions_m

        generation_mask = np.zeros_like(primitive_valid, dtype=bool)
        completion_mask = np.zeros_like(primitive_valid, dtype=bool)
        validation_mask = np.zeros_like(primitive_valid, dtype=bool)
        for event in seed_events:
            for receiver in receiver_subset:
                if receiver != 0:
                    generation_mask[receiver, event] = True
        for event in successful_events:
            for receiver in receiver_subset:
                if receiver != 0 and not generation_mask[receiver, event]:
                    completion_mask[receiver, event] = True
        for event in successful_events[excluded_fit_mask]:
            completion_mask[excluded, event] = True

        validation_residuals: list[float] = []
        validation_sigmas: list[float] = []
        validation_mask_values: list[bool] = []
        whitened_validation: list[float] = []
        unresolved = False
        validation_set = set(validation_events)

        for event in range(len(measurements.event_ids)):
            if np.all(np.isfinite(source_positions[event])):
                continue
            localization_receivers = _localization_receivers_for_event(
                primitive_valid,
                event,
            )
            if localization_receivers is None:
                unresolved = True
                continue
            receiver_index = np.asarray(localization_receivers, dtype=int)
            source = localize_source_from_tdoa(
                microphones[receiver_index],
                arrivals_m[receiver_index, event],
            )
            range_scale = max(
                1.0,
                float(np.max(np.abs(arrivals_m[receiver_index, event]))),
            )
            source_residual_tolerance_m2 = max(
                1e-3 * range_scale**2,
                4.0 * range_scale * acceptance_rms_m,
            )
            # Noisy regime skips the norm-residual magnitude gate (keeps rank).
            # Observed on audio gates: it rejects truth-proximal completions
            # (2.1 vs 0.84 allowed) while accepting wrong-basin ones, so it
            # cannot discriminate; held-out validation and polish decide.
            if source.linear_rank < 4 or (
                acceptance_rms_m <= 2e-5
                and abs(source.norm_residual_m2) > source_residual_tolerance_m2
            ):
                unresolved = True
                continue
            source_positions[event] = source.source_position_m
            for receiver in localization_receivers:
                if receiver != 0:
                    completion_mask[receiver, event] = True

            if event not in validation_set:
                continue
            predicted = _tdoa_predictions(
                microphones,
                source.source_position_m[None, :],
                speed_of_sound,
            )[0]
            event_residuals: list[float] = []
            heldout_columns: list[int] = []
            for receiver in range(1, 8):
                if receiver in localization_receivers:
                    continue
                column = receiver - 1
                if not measurements.valid[event, column]:
                    continue
                validation_mask[receiver, event] = True
                residual_value = predicted[column] - measurements.tdoa_s[event, column]
                validation_residuals.append(residual_value)
                validation_sigmas.append(measurements.sigma_s[event, column])
                validation_mask_values.append(True)
                event_residuals.append(residual_value)
                heldout_columns.append(column)

            if measurements.covariance_s2 is not None and heldout_columns:
                fitted_columns = np.asarray(
                    [receiver - 1 for receiver in localization_receivers if receiver != 0],
                    dtype=int,
                )
                heldout_index = np.asarray(heldout_columns, dtype=int)
                covariance = conditional_covariance(
                    measurements.covariance_s2[event],
                    heldout_index,
                    fitted_columns,
                )
                whitened_validation.extend(
                    float(value)
                    for value in whiten_residual_block(
                        np.asarray(event_residuals, dtype=float),
                        covariance,
                    )
                )

        if unresolved and not np.all(
            np.isfinite(source_positions[np.asarray(validation_events, dtype=int)])
        ):
            continue
        if not validation_residuals:
            continue

        validation_summary = summarize_independent_residuals(
            np.asarray(validation_residuals, dtype=float),
            np.asarray(validation_sigmas, dtype=float),
            np.asarray(validation_mask_values, dtype=bool),
        )
        if whitened_validation:
            whitened = np.asarray(whitened_validation, dtype=float)
            validation_summary = RobustResidualSummary(
                independent_coordinate_count=int(len(whitened)),
                rms_s=validation_summary.rms_s,
                median_abs_s=validation_summary.median_abs_s,
                max_abs_s=validation_summary.max_abs_s,
                normalized_huber_score=float(np.mean(huber_loss(whitened))),
                inlier_fraction=float(np.mean(np.abs(whitened) <= 3.0)),
            )
        split = MeasurementSplit(
            generation_mask=generation_mask,
            completion_mask=completion_mask,
            validation_mask=validation_mask,
        )
        predictions = _tdoa_predictions(microphones, source_positions, speed_of_sound)
        valid = measurements.valid & np.isfinite(predictions)
        all_residuals = predictions[valid] - measurements.tdoa_s[valid]
        full_rms = float(np.sqrt(np.mean(all_residuals * all_residuals)))
        hypothesis = FullGeometryHypothesis(
            subset_id=subset_id,
            root_id=root_id,
            microphone_positions_m=microphones,
            source_positions_m=source_positions,
            split=split,
            validation=validation_summary,
            full_tdoa_rms_s=full_rms,
            metric_rms_m=metric_candidate.corrected_range_rms_m,
            metric_condition_number=(metric.diagnostics.receiver_design_condition_number),
            affine_factor_singular_ratio=metric.diagnostics.factor_singular_ratio,
            model="receiver3d_source3d",
            seed_microphone_ids=tuple(
                int(measurements.microphone_ids[index]) for index in receiver_subset
            ),
            seed_event_ids=tuple(int(measurements.event_ids[index]) for index in seed_events),
            metric_branch_id=metric_branch_id,
            arrival_gauge_shifts_m=np.zeros(
                len(measurements.event_ids),
                dtype=float,
            ),
            offset_scope_event_ids=tuple(
                int(measurements.event_ids[index]) for index in successful_events
            ),
            microphone_completion_status=np.all(
                np.isfinite(microphones),
                axis=1,
            ),
            source_completion_status=np.all(
                np.isfinite(source_positions),
                axis=1,
            ),
            generating_equation_rms=root_generating_equation_rms,
        )
        completed_hypotheses.append((hypothesis, source_positions))

    if not completed_hypotheses:
        return None
    if sigma_range_m > 5e-3:
        # Noisy regime returns every surviving branch (T-004). Branches of
        # one root can be metric-near yet geometrically far; the old
        # best-Huber min() killed truth-proximal branches inside their own
        # root. Polish-then-select below decides among them. Exact regime
        # keeps the single best (bit-identical).
        return completed_hypotheses
    return min(
        completed_hypotheses,
        key=lambda item: (
            item[0].validation.normalized_huber_score,
            item[0].full_tdoa_rms_s,
            item[0].metric_rms_m,
        ),
    )


def _low_rank_initial_geometries(
    arrivals_m: np.ndarray,
    primitive_valid: np.ndarray,
    *,
    position_bound_m: float,
    event_subset: tuple[int, ...] | None = None,
    multipliers: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Geometry-aware starts from rank-three TDOA structure (noisy-tolerant).

    Adapted from the pre-rewrite MAP initializer: optimize the unknown
    reference ranges so the doubly centered cross-squared-distance matrix over
    the given (default: all complete) events is approximately rank three
    (L-BFGS-B with analytic gradient), factorize, then resolve the affine
    ambiguity by robustly fitting cross distances (soft_l1). Returns one start
    per deterministic multiplier: the lowest rank objective overfits
    (sub-truth tail energy with large range error), so callers must try all
    starts and select by held-out validation rather than by rank energy.
    """
    arrivals = np.asarray(arrivals_m, dtype=float)
    if arrivals.ndim != 2 or arrivals.shape[0] != 8:
        return []
    if event_subset is None:
        complete = np.asarray(
            [event for event in range(arrivals.shape[1]) if np.all(primitive_valid[:, event])],
            dtype=int,
        )
    else:
        complete = np.asarray(
            [
                event
                for event in event_subset
                if 0 <= int(event) < arrivals.shape[1]
                and bool(np.all(primitive_valid[:, int(event)]))
            ],
            dtype=int,
        )
    if len(complete) < 6:
        return []
    observed = arrivals[:, complete]
    frame_count = len(complete)
    microphone_count = arrivals.shape[0]
    center_m = (
        np.eye(microphone_count) - np.ones((microphone_count, microphone_count)) / microphone_count
    )
    center_t = np.eye(frame_count) - np.ones((frame_count, frame_count)) / frame_count
    eps = 1e-9
    lower = np.maximum(0.02, -np.min(observed, axis=0) + 0.02)
    upper = np.maximum(lower + 0.5, 2.0 * float(max(position_bound_m, 0.25)))
    aperture = max(0.25, float(np.percentile(np.abs(observed), 90)))

    def rank_objective_and_gradient(reference_ranges: np.ndarray) -> tuple[float, np.ndarray]:
        ranges = observed + reference_ranges[None, :]
        squared = ranges * ranges
        centered = -0.5 * (center_m @ squared @ center_t)
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(3, len(singular))
        rank3 = (u[:, :rank] * singular[:rank]) @ vt[:rank]
        tail = centered - rank3
        tail_energy = float(np.sum(tail * tail))
        total_energy = float(np.sum(centered * centered)) + eps
        value = tail_energy / total_energy
        grad_squared_tail = -(center_m @ tail @ center_t)
        grad_squared_total = -(center_m @ centered @ center_t)
        grad_tail = 2.0 * np.sum(grad_squared_tail * ranges, axis=0)
        grad_total = 2.0 * np.sum(grad_squared_total * ranges, axis=0)
        gradient = (grad_tail * total_energy - tail_energy * grad_total) / (total_energy**2)
        return value, gradient

    fits = []
    for multiplier in multipliers:
        initial = np.clip(lower + multiplier * aperture, lower, upper)
        fit = minimize(
            rank_objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9},
        )
        fits.append(fit)

    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    affine_initial = np.concatenate([np.eye(3).reshape(-1), np.zeros(3)])
    for fit in fits:
        reference_ranges = np.asarray(fit.x, dtype=float)
        if not np.all(np.isfinite(reference_ranges)):
            continue
        ranges = observed + reference_ranges[None, :]
        squared = ranges * ranges
        centered = -0.5 * (center_m @ squared @ center_t)
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        root = np.sqrt(np.maximum(singular[:3], 0.0))
        left = u[:, :3] * root[None, :]
        right = vt[:3].T * root[None, :]

        def affine_unpack(
            parameters: np.ndarray,
            left: np.ndarray = left,
            right: np.ndarray = right,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            transform = parameters[:9].reshape(3, 3)
            try:
                inverse_transpose = np.linalg.inv(transform).T
            except np.linalg.LinAlgError:
                inverse_transpose = np.linalg.pinv(transform).T
            return left @ transform, right @ inverse_transpose, parameters[9:12]

        def affine_residual(
            parameters: np.ndarray,
            squared: np.ndarray = squared,
        ) -> np.ndarray:
            microphones, sources, offset = affine_unpack(parameters)
            predicted = np.sum(
                (microphones[:, None, :] - sources[None, :, :] + offset) ** 2,
                axis=2,
            )
            scale = 1.0 + np.sqrt(np.maximum(squared, 0.0))
            data = ((predicted - squared) / scale).reshape(-1)
            regularizer = 1e-4 * (parameters[:9].reshape(3, 3) - np.eye(3)).reshape(-1)
            return np.concatenate([data, regularizer])

        try:
            affine_fit = least_squares(
                affine_residual,
                affine_initial,
                loss="soft_l1",
                f_scale=0.05,
                max_nfev=800,
            )
        except ValueError:
            continue
        if not np.all(np.isfinite(affine_fit.x)):
            continue
        microphones, sources, offset = affine_unpack(affine_fit.x)
        starts.append(
            (
                np.asarray(microphones + offset, dtype=float),
                np.asarray(sources, dtype=float),
                np.asarray(complete, dtype=int),
            )
        )
    return starts


def _bundle_adjustment_fallback_8mic(
    measurements: EventTDOAMeasurements,
    *,
    arrivals_m: np.ndarray,
    primitive_valid: np.ndarray,
    speed_of_sound: float,
    fitting_events: tuple[int, ...],
    validation_events: tuple[int, ...],
    seed_roots: tuple[tuple[tuple[int, ...], tuple[int, ...], np.ndarray], ...] = (),
    random_starts: int = 8,
) -> FullGeometryHypothesis | None:
    """Robust TDOA bundle adjustment for noisy data the exact minimal path rejects.

    The 7r/6s minimal solver requires exact rank consistency (residual 1e-7), but
    audio-extracted TDOAs carry ~6us RMS error whose true-offset minor residuals
    are O(1). When no minimal hypothesis survives, build a broad pool of starts
    (low-rank rank-structure solutions, generous metric branches from the failed
    minimal roots, deterministic randoms), pre-rank them with cheap held-out
    validation, bundle-adjust the winners, and return the best fully scored
    hypothesis. Returns None when validation does not support any solution.
    """
    event_count = len(measurements.event_ids)
    fitting = np.asarray(fitting_events, dtype=int)
    if len(fitting) < 6 or len(validation_events) == 0:
        return None
    valid = measurements.valid
    sigma = measurements.sigma_s
    if not np.all(valid[fitting]):
        # Fallback needs a complete fitting block; sparse data stays minimal-only.
        return None
    finite_sigma = sigma[valid]
    finite_sigma = finite_sigma[np.isfinite(finite_sigma)]
    median_sigma_s = float(np.median(finite_sigma)) if finite_sigma.size else 2e-6
    if median_sigma_s <= 0.0 or not np.isfinite(median_sigma_s):
        return None

    arrivals_scale_m = float(np.max(np.abs(measurements.tdoa_s[valid])) * speed_of_sound)
    scale_m = max(
        1.0,
        arrivals_scale_m,
        float(np.median(np.abs(measurements.tdoa_s[valid])) * speed_of_sound * 4.0),
    )

    columns = np.arange(measurements.tdoa_s.shape[1], dtype=int)

    def _pin_gauge(microphones: np.ndarray, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translate/rotate geometry into the optimizer gauge (proper rotation).

        Pins mic0 at the origin, mic1 on the x axis, and mic2 in the z=0 plane
        with non-negative y. Tests align with reflection allowed, so fixing one
        chirality half-space is safe and keeps starts deterministic.
        """
        mics = np.asarray(microphones, dtype=float).copy()
        src = np.asarray(sources, dtype=float).copy()
        mics -= mics[0]
        src -= microphones[0]
        axis1 = mics[1] / max(float(np.linalg.norm(mics[1])), 1e-12)
        target = np.array([1.0, 0.0, 0.0])
        cross = np.cross(axis1, target)
        cosine = float(np.dot(axis1, target))
        if float(np.linalg.norm(cross)) <= 1e-12:
            if cosine > 0:
                rotation = np.eye(3)
            else:
                # 180-degree turn about z maps -x to +x with det +1.
                rotation = np.diag([-1.0, -1.0, 1.0])
        else:
            skew = np.array(
                [
                    [0.0, -cross[2], cross[1]],
                    [cross[2], 0.0, -cross[0]],
                    [-cross[1], cross[0], 0.0],
                ]
            )
            rotation = np.eye(3) + skew + skew @ skew / (1.0 + cosine)
        mics = mics @ rotation.T
        src = src @ rotation.T
        angle = float(np.arctan2(mics[2, 2], mics[2, 1]))
        cosine_a, sine_a = float(np.cos(angle)), float(np.sin(angle))
        roll = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine_a, sine_a],
                [0.0, -sine_a, cosine_a],
            ]
        )
        mics = mics @ roll.T
        src = src @ roll.T
        mics[0] = 0.0
        mics[1, 1:] = 0.0
        mics[2, 2] = 0.0
        return mics, src

    def _linear_sources_for(microphones: np.ndarray, events: np.ndarray) -> np.ndarray | None:
        sources = np.empty((len(events), 3), dtype=float)
        for row, event in enumerate(events):
            relative_m = np.zeros(8, dtype=float)
            relative_m[1:] = measurements.tdoa_s[int(event)] * speed_of_sound
            try:
                candidate = localize_source_from_tdoa(microphones, relative_m)
            except ValueError:
                return None
            if candidate.linear_rank < 4:
                return None
            sources[row] = candidate.source_position_m
        return sources

    initializations: list[tuple[np.ndarray, np.ndarray]] = []
    for low_rank_mics, _, _ in _low_rank_initial_geometries(
        arrivals_m,
        primitive_valid,
        position_bound_m=scale_m,
    ):
        subset_sources = _linear_sources_for(low_rank_mics, fitting)
        if subset_sources is None:
            continue
        pinned_mics, pinned_sources = _pin_gauge(low_rank_mics, subset_sources)
        initializations.append((pinned_mics, pinned_sources))

    # RANSAC-style deterministic event subsets: the luckiest clean subset lands
    # much closer to truth than full-data algebraic estimates.
    subset_events: list[tuple[int, ...]] = []
    for width in (8, 10, 12, 14):
        for start in range(0, event_count - width + 1):
            subset_events.append(tuple(range(start, start + width)))
    for stride in (2, 3):
        for offset in range(stride):
            chunk = tuple(index for index in range(event_count) if index % stride == offset)
            if len(chunk) >= 8:
                subset_events.append(chunk)
    for subset in subset_events:
        for subset_mics, _, _ in _low_rank_initial_geometries(
            arrivals_m,
            primitive_valid,
            position_bound_m=scale_m,
            event_subset=subset,
            multipliers=(0.5, 1.5),
        ):
            subset_sources = _linear_sources_for(subset_mics, fitting)
            if subset_sources is None:
                continue
            pinned_mics, pinned_sources = _pin_gauge(subset_mics, subset_sources)
            initializations.append((pinned_mics, pinned_sources))
            if len(initializations) >= 40:
                break
        if len(initializations) >= 40:
            break

    for receiver_subset, seed_events, seed_offsets in seed_roots:
        seed_receivers = np.asarray(receiver_subset, dtype=int)
        seed_index = np.asarray(seed_events, dtype=int)
        complete_fitting = np.asarray(
            [event for event in fitting_events if np.all(primitive_valid[seed_receivers, event])],
            dtype=int,
        )
        if len(complete_fitting) < 6:
            continue
        try:
            expanded = extend_event_offsets(
                arrivals_m[np.ix_(seed_receivers, seed_index)],
                np.asarray(seed_offsets, dtype=float),
                arrivals_m[np.ix_(seed_receivers, complete_fitting)],
                membership_tolerance=5e-3,
            )
        except ValueError:
            continue
        successful = complete_fitting[expanded.success]
        if len(successful) < 6:
            continue
        successful_ranges = expanded.corrected_ranges_m[:, expanded.success]
        try:
            factorization = factor_corrected_ranges(successful_ranges, dimension=3)
        except ValueError:
            continue
        metric = upgrade_metric_3d_overdetermined(
            factorization,
            successful_ranges,
            start_count=12,
            acceptance_rms_m=0.5,
        )
        excluded = next(index for index in range(8) if index not in receiver_subset)
        for candidate in metric.candidates:
            if candidate.corrected_range_rms_m > 0.5:
                continue
            microphones = np.empty((8, 3), dtype=float)
            for local_index, microphone_id in enumerate(receiver_subset):
                microphones[microphone_id] = candidate.microphone_positions_m[local_index]
            excluded_mask = np.asarray(
                [primitive_valid[excluded, event] for event in successful],
                dtype=bool,
            )
            if int(np.sum(excluded_mask)) < 4:
                continue
            excluded_ranges = (
                arrivals_m[excluded, successful[excluded_mask]]
                - expanded.offsets_m[expanded.success][excluded_mask]
            )
            try:
                localized_receiver = localize_receiver_from_ranges(
                    candidate.source_positions_m[excluded_mask],
                    excluded_ranges,
                    robust=True,
                )
            except ValueError:
                continue
            if localized_receiver.linear_rank < 3 or localized_receiver.inlier_rms_m > 0.1:
                continue
            microphones[excluded] = localized_receiver.receiver_position_m
            candidate_sources = _linear_sources_for(microphones, fitting)
            if candidate_sources is None:
                continue
            pinned_mics, pinned_sources = _pin_gauge(microphones, candidate_sources)
            initializations.append((pinned_mics, pinned_sources))
            if len(initializations) >= 16:
                break
        if len(initializations) >= 16:
            break

    def predict(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
        ranges = np.linalg.norm(
            microphones[:, None, :] - sources[None, :, :],
            axis=2,
        )
        return ((ranges[1:] - ranges[[0]]) / speed_of_sound).T

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        microphones = np.zeros((8, 3), dtype=float)
        microphones[1, 0] = values[0]
        microphones[2, 0] = values[1]
        microphones[2, 1] = values[2]
        microphones[3:] = values[3:18].reshape(5, 3)
        sources = values[18:].reshape(len(fitting), 3)
        return microphones, sources

    def pack(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.array(
                    [
                        microphones[1, 0],
                        microphones[2, 0],
                        microphones[2, 1],
                    ]
                ),
                microphones[3:].reshape(-1),
                sources.reshape(-1),
            ]
        )

    def fitting_residuals(values: np.ndarray) -> np.ndarray:
        microphones, sources = unpack(values)
        predicted = predict(microphones, sources)
        residual = (predicted - measurements.tdoa_s[fitting][:, columns]) / sigma[fitting][
            :, columns
        ]
        return residual.reshape(-1)

    rng = np.random.default_rng(20260919)
    for _ in range(max(0, random_starts)):
        microphones = np.zeros((8, 3), dtype=float)
        microphones[1:, 0] = rng.uniform(-scale_m, scale_m, size=7)
        microphones[1:, 1] = rng.uniform(-scale_m, scale_m, size=7)
        microphones[1:, 2] = rng.uniform(0.0, scale_m, size=7)
        microphones[0] = 0.0
        microphones[1, 1:] = 0.0
        microphones[2, 2] = 0.0
        random_sources = _linear_sources_for(microphones, fitting)
        if random_sources is None:
            random_sources = np.column_stack(
                [
                    rng.uniform(-scale_m, scale_m, size=len(fitting)),
                    rng.uniform(-scale_m, scale_m, size=len(fitting)),
                    rng.uniform(0.0, scale_m, size=len(fitting)),
                ]
            )
        initializations.append((microphones, random_sources))
    if not initializations:
        return None
    # Pre-rank starts with cheap event-level validation (strong discriminator)
    # so expensive bundle adjustment focuses on the most promising basins.
    ranked: list[tuple[float, np.ndarray, np.ndarray]] = []
    for microphones, sources in initializations:
        score = _event_validation_rms(
            microphones,
            measurements,
            validation_events,
            speed_of_sound,
        )
        if not np.isfinite(score):
            continue
        ranked.append(
            (
                score,
                np.asarray(microphones, dtype=float),
                np.asarray(sources, dtype=float),
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    best_key: tuple[float, float] | None = None
    best_values: np.ndarray | None = None
    for _, microphones, sources in ranked[:6]:
        initial = pack(microphones, sources)
        try:
            fit = least_squares(
                fitting_residuals,
                initial,
                loss="huber",
                f_scale=3.0,
                max_nfev=2000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-12,
                x_scale="jac",
            )
        except ValueError:
            continue
        if not np.all(np.isfinite(fit.x)):
            continue
        candidate_microphones, _ = unpack(np.asarray(fit.x, dtype=float))
        score = _event_validation_rms(
            candidate_microphones,
            measurements,
            validation_events,
            speed_of_sound,
        )
        if not np.isfinite(score):
            continue
        key = (score, float(fit.cost))
        if best_key is not None and key >= best_key:
            continue
        best_key = key
        best_values = np.asarray(fit.x, dtype=float)
    if best_values is None:
        return None
    microphones, fitting_sources = unpack(best_values)

    source_positions = np.full((event_count, 3), np.nan, dtype=float)
    source_positions[fitting] = fitting_sources

    generation_mask = np.zeros((8, event_count), dtype=bool)
    completion_mask = np.zeros((8, event_count), dtype=bool)
    validation_mask = np.zeros((8, event_count), dtype=bool)
    generation_mask[1:, fitting] = True

    validation_residuals: list[float] = []
    validation_sigmas: list[float] = []
    validation_mask_values: list[bool] = []
    whitened_validation: list[float] = []
    validation_set = set(validation_events)
    for event in range(event_count):
        if np.all(np.isfinite(source_positions[event])):
            continue
        localization_receivers = _localization_receivers_for_event(
            primitive_valid,
            event,
            receiver_count=8,
        )
        if localization_receivers is None:
            continue
        receiver_index = np.asarray(localization_receivers, dtype=int)
        relative_m = np.zeros(len(receiver_index), dtype=float)
        # arrivals relative to reference receiver 0 in meters
        full_relative = np.zeros(8, dtype=float)
        full_relative[1:] = measurements.tdoa_s[event] * speed_of_sound
        relative_m = full_relative[receiver_index]
        try:
            source = localize_source_from_tdoa(
                microphones[receiver_index],
                relative_m,
            )
        except ValueError:
            continue
        if source.linear_rank < 4:
            continue
        source_positions[event] = source.source_position_m
        for receiver in localization_receivers:
            if receiver != 0:
                completion_mask[receiver, event] = True
        if event not in validation_set:
            continue
        predicted = _tdoa_predictions(
            microphones,
            source.source_position_m[None, :],
            speed_of_sound,
        )[0]
        event_residuals: list[float] = []
        heldout_columns: list[int] = []
        for receiver in range(1, 8):
            if receiver in localization_receivers:
                continue
            column = receiver - 1
            if not valid[event, column]:
                continue
            validation_mask[receiver, event] = True
            residual_value = predicted[column] - measurements.tdoa_s[event, column]
            validation_residuals.append(residual_value)
            validation_sigmas.append(sigma[event, column])
            validation_mask_values.append(True)
            event_residuals.append(residual_value)
            heldout_columns.append(column)
        if measurements.covariance_s2 is not None and heldout_columns:
            fitted_columns = np.asarray(
                [receiver - 1 for receiver in localization_receivers if receiver != 0],
                dtype=int,
            )
            covariance = conditional_covariance(
                measurements.covariance_s2[event],
                np.asarray(heldout_columns, dtype=int),
                fitted_columns,
            )
            whitened_validation.extend(
                float(value)
                for value in whiten_residual_block(
                    np.asarray(event_residuals, dtype=float),
                    covariance,
                )
            )

    if not validation_residuals:
        return None
    if not np.all(np.isfinite(source_positions[np.asarray(validation_events, dtype=int)])):
        return None
    validation_summary = summarize_independent_residuals(
        np.asarray(validation_residuals, dtype=float),
        np.asarray(validation_sigmas, dtype=float),
        np.asarray(validation_mask_values, dtype=bool),
    )
    if whitened_validation:
        whitened = np.asarray(whitened_validation, dtype=float)
        validation_summary = RobustResidualSummary(
            independent_coordinate_count=int(len(whitened)),
            rms_s=validation_summary.rms_s,
            median_abs_s=validation_summary.median_abs_s,
            max_abs_s=validation_summary.max_abs_s,
            normalized_huber_score=float(np.mean(huber_loss(whitened))),
            inlier_fraction=float(np.mean(np.abs(whitened) <= 3.0)),
        )
    if validation_summary.rms_s is not None and validation_summary.rms_s > 60e-6:
        return None
    predictions = _tdoa_predictions(microphones, source_positions, speed_of_sound)
    finite = valid & np.isfinite(predictions)
    if not np.any(finite):
        return None
    all_residuals = predictions[finite] - measurements.tdoa_s[finite]
    full_rms = float(np.sqrt(np.mean(all_residuals * all_residuals)))
    if full_rms > 60e-6:
        return None
    microphones_copy = np.array(microphones, copy=True)
    sources_copy = np.array(source_positions, copy=True)
    microphones_copy.setflags(write=False)
    sources_copy.setflags(write=False)
    return FullGeometryHypothesis(
        subset_id="bundle_adjustment_fallback",
        root_id=0,
        microphone_positions_m=microphones_copy,
        source_positions_m=sources_copy,
        split=MeasurementSplit(
            generation_mask=generation_mask,
            completion_mask=completion_mask,
            validation_mask=validation_mask,
        ),
        validation=validation_summary,
        full_tdoa_rms_s=full_rms,
        metric_rms_m=full_rms * speed_of_sound,
        metric_condition_number=None,
        affine_factor_singular_ratio=None,
        model="receiver3d_source3d",
        seed_microphone_ids=tuple(int(value) for value in measurements.microphone_ids[:7]),
        seed_event_ids=tuple(int(measurements.event_ids[value]) for value in fitting[:6]),
        metric_branch_id=0,
        arrival_gauge_shifts_m=np.zeros(event_count, dtype=float),
        offset_scope_event_ids=tuple(int(measurements.event_ids[value]) for value in fitting),
        microphone_completion_status=np.all(np.isfinite(microphones_copy), axis=1),
        source_completion_status=np.all(np.isfinite(sources_copy), axis=1),
        generating_equation_rms=full_rms * speed_of_sound,
    )


def _linear_sources_at_microphones(
    microphones: np.ndarray,
    measurements: EventTDOAMeasurements,
    arrivals_m: np.ndarray,
    primitive_valid: np.ndarray,
    speed_of_sound: float,
) -> np.ndarray | None:
    """Refit every event's source linearly at fixed microphones.

    Returns array or None when any event is unfittable (caller keeps the
    representative sources then). Used to de-poison joint polish starts.
    """
    microphones = np.asarray(microphones, dtype=float)
    event_count = len(measurements.event_ids)
    sources = np.full((event_count, 3), np.nan, dtype=float)
    for event in range(event_count):
        receivers = [r for r in range(arrivals_m.shape[0]) if primitive_valid[r, event]]
        if len(receivers) < 5:
            return None
        reference = receivers[0]
        relative = arrivals_m[receivers, event] - arrivals_m[reference, event]
        try:
            source = localize_source_from_tdoa(
                microphones[receivers],
                relative,
            )
        except ValueError:
            return None
        if source.linear_rank < 4:
            return None
        sources[event] = source.source_position_m
    return sources


def _dedupe_classes(classes: list[HypothesisClass]) -> list[HypothesisClass]:
    """Order-preserving identity dedupe (HypothesisClass holds ndarrays)."""
    seen: list[HypothesisClass] = []
    for candidate in classes:
        if all(candidate is not existing for existing in seen):
            seen.append(candidate)
    return seen


def _event_validation_rms(
    microphones: np.ndarray,
    measurements: EventTDOAMeasurements,
    validation_events: tuple[int, ...],
    speed_of_sound: float,
    *,
    receiver_subset: tuple[int, ...] | None = None,
) -> float:
    """Strong held-out score: localize each validation event from all receivers.

    Unlike the stratified receiver-holdout score (which absorbs microphone error
    by fitting validation sources from five receivers and checks two), fitting
    all valid pairs with three source unknowns leaves every direction of
    microphone error exposed. Truth scores near measurement noise while
    0.5m-off geometries score hundreds of microseconds worse, so this
    discriminates where Huber ties. Returns inf when scoring is impossible.

    When receiver_subset is given, only those receivers participate: this
    scores seed geometry without letting a fragile extra-microphone completion
    poison selection (completion is repaired by polish after selection).
    """
    microphones = np.asarray(microphones, dtype=float)
    if microphones.shape[0] < 5 or microphones.shape[1] != 3:
        return float("inf")
    if not np.all(np.isfinite(microphones)):
        return float("inf")
    if receiver_subset is not None:
        subset = sorted({int(r) for r in receiver_subset})
        if 0 not in subset or any(r < 0 or r >= len(microphones) for r in subset):
            return float("inf")
    else:
        subset = list(range(len(microphones)))
    valid = measurements.valid
    squared_sum = 0.0
    count = 0
    for event in validation_events:
        event = int(event)
        # Column c corresponds to pair (0, c+1); receiver 0 needs arrival 0.
        receivers = [r for r in subset if r == 0 or valid[event, r - 1]]
        if len(receivers) < 5 or receivers[0] != 0:
            return float("inf")
        receiver_index = np.asarray(receivers, dtype=int)
        full_relative = np.zeros(len(microphones), dtype=float)
        full_relative[1:] = measurements.tdoa_s[event] * speed_of_sound
        try:
            source = localize_source_from_tdoa(
                microphones[receiver_index],
                full_relative[receiver_index],
            )
        except ValueError:
            return float("inf")
        if source.linear_rank < 4:
            return float("inf")
        ranges = np.linalg.norm(microphones[receiver_index] - source.source_position_m, axis=1)
        predicted = (ranges[1:] - ranges[0]) / speed_of_sound
        columns = np.asarray([r - 1 for r in receivers if r != 0], dtype=int)
        residual = predicted - measurements.tdoa_s[event][columns]
        mask = valid[event][columns]
        if not np.any(mask):
            return float("inf")
        squared_sum += float(np.sum(residual[mask] * residual[mask]))
        count += int(np.sum(mask))
    if count == 0:
        return float("inf")
    return float(np.sqrt(squared_sum / count))


def calibrate_tdoa_8mic(
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float = 343.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    root_start_count: int = 32,
    metric_start_count: int = 20,
    class_tolerance_m: float = 2e-4,
    ambiguity_score_tolerance: float = 0.05,
) -> StratifiedCalibrationResult:
    """Run the Milestone-C eight-microphone TDOA-only stratified solver."""
    if len(measurements.microphone_ids) != 8:
        raise ValueError("calibrate_tdoa_8mic requires exactly 8 microphones")
    arrivals_m, _, primitive_valid = _reference_star_arrivals_m(
        measurements,
        speed_of_sound,
    )
    fitting_events, validation_events = _split_events(len(measurements.event_ids))
    # Noise-aware minimal residual tolerance (T-004, section 6 of the implementation
    # plan). Exact data keep the 1e-7 verification so the exact gate is
    # bit-identical; audio-extracted TDOAs (~6us RMS) violate exact rank
    # consistency with true-offset minor residuals O(1), so the noisy regime
    # admits approximate roots (tolerance 3.0 covers the observed true-offset
    # max-abs 1.4 with margin) and leaves selection to held-out validation.
    valid_sigma = measurements.sigma_s[measurements.valid]
    valid_sigma = valid_sigma[np.isfinite(valid_sigma)]
    median_sigma_range_m = (
        float(np.median(valid_sigma)) * speed_of_sound if valid_sigma.size else 0.0
    )
    noisy_regime = median_sigma_range_m > 5e-3
    if noisy_regime:
        minimal_residual_tolerance = 3.0
    else:
        minimal_residual_tolerance = 1e-7
    # Noisy regime also searches every receiver exclusion and twice the event
    # families: approximate roots are basin-sensitive, so more supported
    # subsets mean more chances that one lands near truth. Exact regime keeps
    # caller budgets bit-identically.
    if noisy_regime:
        receiver_subsets = _receiver_subsets(8, budget=8)
        seed_families = _seed_event_families(
            fitting_events,
            budget=max(event_subset_budget, 4),
        )
    else:
        receiver_subsets = _receiver_subsets(8, budget=receiver_subset_budget)
        seed_families = _seed_event_families(
            fitting_events,
            budget=event_subset_budget,
        )

    hypotheses: list[FullGeometryHypothesis] = []
    rejections: list[str] = []
    attempted = 0
    roots_generated = 0
    metric_candidates = 0
    seed_root_records: list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]] = []

    for receiver_subset in receiver_subsets:
        receiver_index = np.asarray(receiver_subset, dtype=int)
        for seed_events in seed_families:
            seed_index = np.asarray(seed_events, dtype=int)
            if not np.all(primitive_valid[np.ix_(receiver_index, seed_index)]):
                rejections.append("incomplete_minimal_subset")
                continue
            attempted += 1
            minimal = solve_offsets_7r6s(
                arrivals_m[np.ix_(receiver_index, seed_index)],
                start_count=root_start_count,
                residual_tolerance=minimal_residual_tolerance,
            )
            roots_generated += len(minimal.roots)
            for root_id, root in enumerate(minimal.roots):
                seed_root_records.append(
                    (tuple(receiver_subset), tuple(seed_events), np.asarray(root.offsets_m))
                )
                completed = _complete_geometry(
                    arrivals_m=arrivals_m,
                    primitive_valid=primitive_valid,
                    fitting_events=fitting_events,
                    validation_events=validation_events,
                    receiver_subset=receiver_subset,
                    seed_events=seed_events,
                    seed_root_offsets_m=root.offsets_m,
                    speed_of_sound=speed_of_sound,
                    measurements=measurements,
                    subset_id=f"r{receiver_subset}-e{seed_events}",
                    root_id=root_id,
                    root_generating_equation_rms=root.all_minor_rms,
                    metric_start_count=metric_start_count,
                )
                if completed is None:
                    rejections.append("completion_or_metric_failure")
                    continue
                if isinstance(completed, list):
                    for hypothesis, _ in completed:
                        hypotheses.append(hypothesis)
                        metric_candidates += 1
                    continue
                hypothesis, _ = completed
                hypotheses.append(hypothesis)
                metric_candidates += 1

    fallback_exhausted = False
    if not hypotheses:
        fallback_exhausted = True
        fallback = _bundle_adjustment_fallback_8mic(
            measurements,
            arrivals_m=arrivals_m,
            primitive_valid=primitive_valid,
            speed_of_sound=speed_of_sound,
            fitting_events=fitting_events,
            validation_events=validation_events,
            seed_roots=tuple(seed_root_records),
        )
        if fallback is not None:
            hypotheses.append(fallback)
            metric_candidates += 1
    if not hypotheses:
        return StratifiedCalibrationResult(
            status="insufficient_data" if attempted == 0 else "failed",
            microphone_positions_m=None,
            source_positions_m=None,
            event_ids=measurements.event_ids,
            tdoa_rms_s=None,
            selected_class=None,
            classes=(),
            diagnostics=StratifiedCalibrationDiagnostics(
                attempted_subsets=attempted,
                generated_offset_roots=roots_generated,
                metric_candidates=metric_candidates,
                completed_hypotheses=0,
                geometric_class_count=0,
                selected_support=0,
                fitting_event_count=len(fitting_events),
                validation_event_count=len(validation_events),
                validation_independent_coordinates=0,
                rejection_reasons=tuple(rejections),
            ),
        )

    classes = cluster_equivalent_hypotheses(
        tuple(hypotheses),
        microphone_rms_tolerance_m=class_tolerance_m,
    )
    if noisy_regime and classes:
        # Polish-then-select (T-004). Coarse hypotheses can carry good seeds
        # with bad sources/completions that every pre-polish score misranks;
        # polishing each candidate class and selecting by post-polish
        # event-level validation recovers them. Exact regime untouched below.
        from .refinement import refine_calibration

        id_to_index = {
            microphone_id: index for index, microphone_id in enumerate(measurements.microphone_ids)
        }
        pre_ranked: list[tuple[float, HypothesisClass]] = []
        for hypothesis_class in classes:
            seed_ids = hypothesis_class.representative.seed_microphone_ids
            seed_indices = tuple(
                id_to_index[microphone_id]
                for microphone_id in seed_ids
                if microphone_id in id_to_index
            )
            pre_ranked.append(
                (
                    _event_validation_rms(
                        np.asarray(hypothesis_class.representative.microphone_positions_m),
                        measurements,
                        validation_events,
                        speed_of_sound,
                        receiver_subset=seed_indices if seed_indices else None,
                    ),
                    hypothesis_class,
                )
            )
        pre_ranked.sort(
            key=lambda item: (
                item[0],
                item[1].representative.validation.normalized_huber_score,
            )
        )
        polished_options: list[tuple[float, HypothesisClass, StratifiedCalibrationResult]] = []
        for _, hypothesis_class in pre_ranked[:6]:
            representative = hypothesis_class.representative
            # Linear-source refresh (T-004). Completion sources can be
            # inconsistent enough to drag joint polish out of the basin
            # (truth-mics + coarse-sources diverge to 1m); refitting each
            # event's source linearly at the coarse microphones first lets
            # joint polish converge (0.33m coarse reaches 0.20m; without
            # refresh the same start diverges past 0.9m).
            refreshed_sources = _linear_sources_at_microphones(
                np.asarray(representative.microphone_positions_m),
                measurements,
                arrivals_m,
                primitive_valid,
                speed_of_sound,
            )
            # Polish both the representative sources and the linear refresh
            # (T-004): refresh rescues poisoned completions but can displace
            # good ones (linear bias); post-polish validation selects.
            source_variants = [np.asarray(representative.source_positions_m)]
            if refreshed_sources is not None:
                source_variants.append(refreshed_sources)
            placeholder = StratifiedCalibrationDiagnostics(
                attempted_subsets=attempted,
                generated_offset_roots=roots_generated,
                metric_candidates=metric_candidates,
                completed_hypotheses=len(hypotheses),
                geometric_class_count=len(classes),
                selected_support=hypothesis_class.independent_subset_support,
                fitting_event_count=len(fitting_events),
                validation_event_count=len(validation_events),
                validation_independent_coordinates=(
                    representative.validation.independent_coordinate_count
                ),
                rejection_reasons=tuple(rejections),
            )

            def polish_variant(candidate_sources: np.ndarray) -> float | None:
                candidate_result = StratifiedCalibrationResult(
                    status="solved",
                    microphone_positions_m=np.asarray(representative.microphone_positions_m),
                    source_positions_m=candidate_sources,
                    event_ids=measurements.event_ids,
                    tdoa_rms_s=representative.full_tdoa_rms_s,
                    selected_class=hypothesis_class,
                    classes=classes,
                    diagnostics=placeholder,
                )
                polished, polish_diagnostics = refine_calibration(
                    candidate_result,
                    measurements,
                    speed_of_sound=speed_of_sound,
                    mode="huber",
                    max_nfev=500,
                )
                if not polish_diagnostics.accepted or not isinstance(
                    polished, StratifiedCalibrationResult
                ):
                    return None
                if polished.microphone_positions_m is None:
                    return None
                post_score = _event_validation_rms(
                    np.asarray(polished.microphone_positions_m),
                    measurements,
                    validation_events,
                    speed_of_sound,
                )
                polished_options.append((post_score, hypothesis_class, polished))
                return post_score

            # As-is first, then the refreshed variant when available: refresh
            # rescues poisoned completions but can displace good ones
            # (linear bias); post-polish validation selects. Both always run
            # (a strong class's refreshed variant can still win).
            polish_variant(source_variants[0])
            if len(source_variants) > 1:
                polish_variant(source_variants[1])
        # Conditional bundle-adjustment rescue (T-004). Approximate
        # stratified roots sometimes miss the polish basin on every branch
        # (all post-polish event RMS Bad); the low-rank bundle adjustment
        # starts from data-driven inits independent of minimal roots and
        # often lands in-basin instead. It runs only when stratified is
        # weak (validation-selected winners validate near noise, ~3-9us),
        # so fixtures stratified already solves pay no extra cost. The
        # rescue candidate joins the same post-polish selection below. The
        # trigger sits at 6us (winners validate at 3-6us).
        stratified_best = (
            min(item[0] for item in polished_options) if polished_options else float("inf")
        )
        if stratified_best > 6e-6 and not fallback_exhausted:
            rescue = _bundle_adjustment_fallback_8mic(
                measurements,
                arrivals_m=arrivals_m,
                primitive_valid=primitive_valid,
                speed_of_sound=speed_of_sound,
                fitting_events=fitting_events,
                validation_events=validation_events,
                seed_roots=tuple(seed_root_records),
            )
            if rescue is not None:
                hypotheses.append(rescue)
                metric_candidates += 1
                rescue_classes = cluster_equivalent_hypotheses(
                    (rescue,),
                    microphone_rms_tolerance_m=class_tolerance_m,
                )
                rescue_class = rescue_classes[0]
                rescue_placeholder = StratifiedCalibrationDiagnostics(
                    attempted_subsets=attempted,
                    generated_offset_roots=roots_generated,
                    metric_candidates=metric_candidates,
                    completed_hypotheses=len(hypotheses),
                    geometric_class_count=len(classes) + 1,
                    selected_support=0,
                    fitting_event_count=len(fitting_events),
                    validation_event_count=len(validation_events),
                    validation_independent_coordinates=(
                        rescue.validation.independent_coordinate_count
                    ),
                    rejection_reasons=tuple(rejections),
                )
                rescue_result = StratifiedCalibrationResult(
                    status="solved",
                    microphone_positions_m=np.asarray(rescue.microphone_positions_m),
                    source_positions_m=np.asarray(rescue.source_positions_m),
                    event_ids=measurements.event_ids,
                    tdoa_rms_s=rescue.full_tdoa_rms_s,
                    selected_class=rescue_class,
                    classes=classes,
                    diagnostics=rescue_placeholder,
                )
                rescue_polished, rescue_polish_diagnostics = refine_calibration(
                    rescue_result,
                    measurements,
                    speed_of_sound=speed_of_sound,
                    mode="huber",
                    max_nfev=500,
                )
                if (
                    rescue_polish_diagnostics.accepted
                    and isinstance(rescue_polished, StratifiedCalibrationResult)
                    and rescue_polished.microphone_positions_m is not None
                ):
                    rescue_post = _event_validation_rms(
                        np.asarray(rescue_polished.microphone_positions_m),
                        measurements,
                        validation_events,
                        speed_of_sound,
                    )
                    polished_options.append((rescue_post, rescue_class, rescue_polished))
        if polished_options:
            polished_options.sort(
                key=lambda item: (
                    item[0],
                    item[1].representative.validation.normalized_huber_score,
                )
            )
            _, selected_polished_class, selected_polished = polished_options[0]
            polished_status: CalibrationStatus = "solved"
            # Ambiguity compares distinct geometric classes: two polish
            # variants of the same class validating within tolerance is
            # agreement, not ambiguity.
            selected_polished_mics = np.asarray(selected_polished.microphone_positions_m)
            for runner_score, runner_class, runner_polished in polished_options[1:]:
                if runner_class is selected_polished_class:
                    continue
                best_score = polished_options[0][0]
                if not (
                    runner_class.independent_subset_support
                    >= selected_polished_class.independent_subset_support
                    and abs(runner_score - best_score)
                    <= ambiguity_score_tolerance * max(abs(best_score), 1e-9)
                ):
                    break
                # Tied but geometrically coincident runners are duplicate
                # basin splits, not genuine ambiguity: only distinct
                # geometries (aligned mic RMS beyond 5cm, a third of the
                # acceptance gate) keep the ambiguous verdict. Compare the
                # polished geometries (the actual selection candidates),
                # not the coarse representatives.
                runner_mics = np.asarray(runner_polished.microphone_positions_m)
                try:
                    aligned_runner, _, _ = rigid_align(runner_mics, selected_polished_mics)
                    runner_distance_m = rms_position_error(aligned_runner, selected_polished_mics)
                except ValueError:
                    runner_distance_m = float("inf")
                if runner_distance_m > 0.05:
                    polished_status = "ambiguous"
                break
            return StratifiedCalibrationResult(
                status=polished_status,
                microphone_positions_m=np.asarray(selected_polished.microphone_positions_m),
                source_positions_m=np.asarray(selected_polished.source_positions_m),
                event_ids=measurements.event_ids,
                tdoa_rms_s=selected_polished.tdoa_rms_s,
                selected_class=selected_polished_class,
                classes=tuple(
                    _dedupe_classes(
                        [item[1] for item in polished_options]
                        + [c for c in classes if all(c is not item[1] for item in polished_options)]
                    )
                ),
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=attempted,
                    generated_offset_roots=roots_generated,
                    metric_candidates=metric_candidates,
                    completed_hypotheses=len(hypotheses),
                    geometric_class_count=len(classes),
                    selected_support=selected_polished_class.independent_subset_support,
                    fitting_event_count=len(fitting_events),
                    validation_event_count=len(validation_events),
                    validation_independent_coordinates=(
                        selected_polished_class.representative.validation.independent_coordinate_count
                    ),
                    rejection_reasons=tuple(rejections),
                ),
            )
    if noisy_regime and len(classes) > 1:
        # Receiver-holdout Huber ties (or inverts) on coarse noisy hypotheses;
        # event-level validation separates truth-proximal classes by 10-100x.
        # Scoring uses seed receivers only so a fragile extra-microphone
        # completion cannot poison selection (polish repairs it afterwards).
        # Exact regime keeps Huber order bit-identically.
        id_to_index = {
            microphone_id: index for index, microphone_id in enumerate(measurements.microphone_ids)
        }
        scored_classes: list[tuple[float, HypothesisClass]] = []
        for hypothesis_class in classes:
            seed_ids = hypothesis_class.representative.seed_microphone_ids
            seed_indices = tuple(
                id_to_index[microphone_id]
                for microphone_id in seed_ids
                if microphone_id in id_to_index
            )
            score = _event_validation_rms(
                np.asarray(hypothesis_class.representative.microphone_positions_m),
                measurements,
                validation_events,
                speed_of_sound,
                receiver_subset=seed_indices if seed_indices else None,
            )
            scored_classes.append((score, hypothesis_class))
        scored_classes.sort(
            key=lambda item: (
                item[0],
                item[1].representative.validation.normalized_huber_score,
            )
        )
        classes = tuple(item[1] for item in scored_classes)
    selected = classes[0]
    status: CalibrationStatus = "solved"
    if len(classes) > 1:
        first_score = selected.representative.validation.normalized_huber_score
        second_score = classes[1].representative.validation.normalized_huber_score
        if classes[1].independent_subset_support >= selected.independent_subset_support and abs(
            second_score - first_score
        ) <= ambiguity_score_tolerance * max(1.0, abs(first_score)):
            status = "ambiguous"

    representative = selected.representative
    microphones = np.array(representative.microphone_positions_m, copy=True)
    sources = np.array(representative.source_positions_m, copy=True)
    microphones.setflags(write=False)
    sources.setflags(write=False)
    result = StratifiedCalibrationResult(
        status=status,
        microphone_positions_m=microphones,
        source_positions_m=sources,
        event_ids=measurements.event_ids,
        tdoa_rms_s=representative.full_tdoa_rms_s,
        selected_class=selected,
        classes=classes,
        diagnostics=StratifiedCalibrationDiagnostics(
            attempted_subsets=attempted,
            generated_offset_roots=roots_generated,
            metric_candidates=metric_candidates,
            completed_hypotheses=len(hypotheses),
            geometric_class_count=len(classes),
            selected_support=selected.independent_subset_support,
            fitting_event_count=len(fitting_events),
            validation_event_count=len(validation_events),
            validation_independent_coordinates=(
                representative.validation.independent_coordinate_count
            ),
            rejection_reasons=tuple(rejections),
        ),
    )
    if noisy_regime and result.status == "solved":
        # Deterministic Huber polish of the selected noisy geometry (T-004).
        # The stratified algebraic stages land near-truth candidates (e.g. 15 cm)
        # whose basin the gauge-reduced robust refinement finishes to cm level.
        # Deferred import avoids a refinement<->solver module cycle. Rollback
        # inside refinement keeps the unpolished result when polish stalls.
        from .refinement import refine_calibration

        polished, polish_diagnostics = refine_calibration(
            result,
            measurements,
            speed_of_sound=speed_of_sound,
            mode="huber",
            max_nfev=500,
        )
        if polish_diagnostics.accepted and isinstance(polished, StratifiedCalibrationResult):
            return polished
    return result


def calibrate_planar_tdoa_8mic(
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float = 343.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    membership_tolerance: float = 5e-3,
    metric_acceptance_rms_m: float = 5e-3,
    angle_constraint: PlanarAngleConstraint | None = None,
) -> PlanarCalibrationResult:
    """Run the eight-microphone planar TDOA-only solver with unsigned source height."""
    if len(measurements.microphone_ids) != 8:
        raise ValueError("calibrate_planar_tdoa_8mic requires exactly 8 microphones")
    arrivals_m, _, primitive_valid = _reference_star_arrivals_m(
        measurements,
        speed_of_sound,
    )
    fitting_events, validation_events = _split_events(len(measurements.event_ids))
    receiver_subsets = _receiver_subsets(8, budget=receiver_subset_budget)
    seed_families = _seed_event_families(
        fitting_events,
        budget=event_subset_budget,
        seed_size=4,
    )

    hypotheses: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            RobustResidualSummary,
            float,
            bool,
        ]
    ] = []
    rejections: list[str] = []
    attempted = 0
    generated = 0
    continuous_ambiguity_dimension: int | None = None
    continuous_ambiguity_nullspace: np.ndarray | None = None

    for receiver_subset in receiver_subsets:
        receiver_index = np.asarray(receiver_subset, dtype=int)
        excluded = next(index for index in range(8) if index not in receiver_subset)

        local_angle_constraint = None
        if angle_constraint is not None:
            required_ids = {
                angle_constraint.center_receiver,
                angle_constraint.arm_a_receiver,
                angle_constraint.arm_b_receiver,
            }
            if not required_ids.issubset(set(receiver_subset)):
                rejections.append("constraint_receivers_not_in_subset")
                continue
            local_index = {
                microphone_id: index for index, microphone_id in enumerate(receiver_subset)
            }
            local_angle_constraint = PlanarAngleConstraint(
                center_receiver=local_index[angle_constraint.center_receiver],
                arm_a_receiver=local_index[angle_constraint.arm_a_receiver],
                arm_b_receiver=local_index[angle_constraint.arm_b_receiver],
                angle_rad=angle_constraint.angle_rad,
                provenance=angle_constraint.provenance,
                exact=angle_constraint.exact,
            )

        for seed_events in seed_families:
            seed_index = np.asarray(seed_events, dtype=int)
            if not np.all(primitive_valid[np.ix_(receiver_index, seed_index)]):
                rejections.append("incomplete_planar_seed")
                continue
            attempted += 1
            try:
                offset_solution = solve_linear_offsets(
                    arrivals_m[np.ix_(receiver_index, seed_index)],
                    dimension=2,
                )
            except ValueError:
                rejections.append("planar_offset_failure")
                continue
            generated += 1

            complete_fitting = np.asarray(
                [
                    event
                    for event in fitting_events
                    if np.all(primitive_valid[receiver_index, event])
                ],
                dtype=int,
            )
            try:
                expanded = extend_event_offsets(
                    arrivals_m[np.ix_(receiver_index, seed_index)],
                    offset_solution.offsets_m,
                    arrivals_m[np.ix_(receiver_index, complete_fitting)],
                    membership_tolerance=membership_tolerance,
                )
            except ValueError:
                rejections.append("planar_offset_expansion_failure")
                continue
            successful_events = complete_fitting[expanded.success]
            successful_offsets = expanded.offsets_m[expanded.success]
            successful_ranges = expanded.corrected_ranges_m[:, expanded.success]
            if len(successful_events) < 5:
                rejections.append("insufficient_planar_expansion")
                continue

            factorization = factor_corrected_ranges(
                successful_ranges,
                dimension=2,
            )
            metric = upgrade_metric_planar(
                factorization,
                successful_ranges,
                acceptance_rms_m=metric_acceptance_rms_m,
                angle_constraint=local_angle_constraint,
            )
            identifiability = metric.diagnostics.identifiability
            if angle_constraint is None and identifiability.continuous_metric_nullity > 0:
                if (
                    continuous_ambiguity_dimension is None
                    or identifiability.continuous_metric_nullity > continuous_ambiguity_dimension
                ):
                    continuous_ambiguity_dimension = identifiability.continuous_metric_nullity
                    continuous_ambiguity_nullspace = np.array(
                        identifiability.metric_nullspace,
                        copy=True,
                    )
                    continuous_ambiguity_nullspace.setflags(write=False)
                rejections.extend(metric.diagnostics.reasons)
                if "continuous_metric_family" not in rejections:
                    rejections.append("continuous_metric_family")
                continue
            if not metric.candidates:
                rejections.extend(metric.diagnostics.reasons)
                continue
            candidate = metric.candidates[0]
            metric_weak = metric.status == "weakly_identified"

            excluded_valid = np.asarray(
                [primitive_valid[excluded, event] for event in successful_events],
                dtype=bool,
            )
            if int(np.sum(excluded_valid)) < 3:
                rejections.append("insufficient_planar_receiver_completion")
                continue
            excluded_ranges = (
                arrivals_m[excluded, successful_events[excluded_valid]]
                - successful_offsets[excluded_valid]
            )
            try:
                localized_receiver = localize_planar_receiver_from_ranges(
                    candidate.source_projected_positions_m[excluded_valid],
                    candidate.source_unsigned_heights_m[excluded_valid],
                    excluded_ranges,
                )
            except ValueError:
                rejections.append("planar_receiver_completion_failure")
                continue
            if localized_receiver.linear_rank < 2 or localized_receiver.range_rms_m > 1e-2:
                rejections.append("planar_receiver_completion_failure")
                continue

            microphones = np.empty((8, 3), dtype=float)
            for local_index, microphone_id in enumerate(receiver_subset):
                microphones[microphone_id] = candidate.microphone_positions_m[local_index]
            microphones[excluded] = [
                localized_receiver.position_2d_m[0],
                localized_receiver.position_2d_m[1],
                0.0,
            ]

            projected_sources = np.full(
                (len(measurements.event_ids), 2),
                np.nan,
                dtype=float,
            )
            unsigned_heights = np.full(
                len(measurements.event_ids),
                np.nan,
                dtype=float,
            )
            projected_sources[successful_events] = candidate.source_projected_positions_m
            unsigned_heights[successful_events] = candidate.source_unsigned_heights_m

            validation_residuals: list[float] = []
            validation_sigmas: list[float] = []
            validation_valid: list[bool] = []
            whitened_validation: list[float] = []
            validation_set = set(validation_events)
            unresolved_validation = False

            for event in range(len(measurements.event_ids)):
                if np.all(np.isfinite(projected_sources[event])):
                    continue
                localization_receivers = _localization_receivers_for_event(
                    primitive_valid,
                    event,
                )
                if localization_receivers is None:
                    if event in validation_set:
                        unresolved_validation = True
                    continue
                receiver_ids = np.asarray(localization_receivers, dtype=int)
                try:
                    localized_source = localize_planar_source_from_tdoa(
                        microphones[receiver_ids],
                        arrivals_m[receiver_ids, event],
                    )
                except ValueError:
                    if event in validation_set:
                        unresolved_validation = True
                    continue
                if localized_source.linear_rank < 3:
                    if event in validation_set:
                        unresolved_validation = True
                    continue
                projected_sources[event] = localized_source.projected_position_m
                unsigned_heights[event] = localized_source.unsigned_height_m

                if event not in validation_set:
                    continue
                representative = np.array(
                    [
                        localized_source.projected_position_m[0],
                        localized_source.projected_position_m[1],
                        localized_source.unsigned_height_m,
                    ]
                )
                predicted_ranges = np.linalg.norm(
                    microphones - representative[None, :],
                    axis=1,
                )
                predicted_tdoa = (predicted_ranges[1:] - predicted_ranges[0]) / speed_of_sound
                event_residuals: list[float] = []
                heldout_columns: list[int] = []
                for receiver in range(1, 8):
                    if receiver in localization_receivers:
                        continue
                    column = receiver - 1
                    if not measurements.valid[event, column]:
                        continue
                    residual_value = predicted_tdoa[column] - measurements.tdoa_s[event, column]
                    validation_residuals.append(residual_value)
                    validation_sigmas.append(measurements.sigma_s[event, column])
                    validation_valid.append(True)
                    event_residuals.append(residual_value)
                    heldout_columns.append(column)

                if measurements.covariance_s2 is not None and heldout_columns:
                    fitted_columns = np.asarray(
                        [receiver - 1 for receiver in localization_receivers if receiver != 0],
                        dtype=int,
                    )
                    covariance = conditional_covariance(
                        measurements.covariance_s2[event],
                        np.asarray(heldout_columns, dtype=int),
                        fitted_columns,
                    )
                    whitened_validation.extend(
                        float(value)
                        for value in whiten_residual_block(
                            np.asarray(event_residuals, dtype=float),
                            covariance,
                        )
                    )

            if unresolved_validation or not validation_residuals:
                rejections.append("planar_validation_unresolved")
                continue
            if not np.all(np.isfinite(projected_sources[np.asarray(validation_events, dtype=int)])):
                rejections.append("planar_validation_missing_source")
                continue

            representative_sources = np.column_stack([projected_sources, unsigned_heights])
            predicted = _tdoa_predictions(
                microphones,
                representative_sources,
                speed_of_sound,
            )
            valid = measurements.valid & np.isfinite(predicted)
            full_residual = predicted[valid] - measurements.tdoa_s[valid]
            full_rms = float(np.sqrt(np.mean(full_residual * full_residual)))
            validation = summarize_independent_residuals(
                np.asarray(validation_residuals, dtype=float),
                np.asarray(validation_sigmas, dtype=float),
                np.asarray(validation_valid, dtype=bool),
            )
            if whitened_validation:
                whitened = np.asarray(whitened_validation, dtype=float)
                validation = RobustResidualSummary(
                    independent_coordinate_count=int(len(whitened)),
                    rms_s=validation.rms_s,
                    median_abs_s=validation.median_abs_s,
                    max_abs_s=validation.max_abs_s,
                    normalized_huber_score=float(np.mean(huber_loss(whitened))),
                    inlier_fraction=float(np.mean(np.abs(whitened) <= 3.0)),
                )
            hypotheses.append(
                (
                    microphones,
                    projected_sources,
                    unsigned_heights,
                    representative_sources,
                    validation,
                    full_rms,
                    metric_weak,
                )
            )

    if not hypotheses:
        if attempted == 0:
            failure_status: CalibrationStatus = "insufficient_data"
        elif any(
            reason
            in {
                "receiver_metric_design_rank_deficient",
                "receiver_points_lie_on_nontrivial_conic",
            }
            for reason in rejections
        ):
            failure_status = "degenerate"
        else:
            failure_status = "failed"
        return PlanarCalibrationResult(
            status=failure_status,
            microphone_positions_m=None,
            source_projected_positions_m=None,
            source_unsigned_heights_m=None,
            source_height_sign_known=None,
            source_representative_positions_m=None,
            event_ids=measurements.event_ids,
            validation=None,
            tdoa_rms_s=None,
            diagnostics=StratifiedCalibrationDiagnostics(
                attempted_subsets=attempted,
                generated_offset_roots=generated,
                metric_candidates=0,
                completed_hypotheses=0,
                geometric_class_count=0,
                selected_support=0,
                fitting_event_count=len(fitting_events),
                validation_event_count=len(validation_events),
                validation_independent_coordinates=0,
                rejection_reasons=tuple(rejections),
            ),
            continuous_ambiguity_dimension=continuous_ambiguity_dimension,
            continuous_ambiguity_nullspace=continuous_ambiguity_nullspace,
        )

    selected = min(
        hypotheses,
        key=lambda item: (
            item[4].normalized_huber_score,
            item[5],
        ),
    )
    (
        microphones,
        projected,
        heights,
        representative,
        validation,
        full_rms,
        metric_weak,
    ) = selected
    sign_known = np.zeros(len(heights), dtype=bool)
    arrays = [microphones, projected, heights, sign_known, representative]
    frozen: list[np.ndarray] = []
    for array in arrays:
        copy = np.array(array, copy=True)
        copy.setflags(write=False)
        frozen.append(copy)
    return PlanarCalibrationResult(
        status="weakly_identified" if metric_weak else "solved",
        microphone_positions_m=frozen[0],
        source_projected_positions_m=frozen[1],
        source_unsigned_heights_m=frozen[2],
        source_height_sign_known=frozen[3],
        source_representative_positions_m=frozen[4],
        event_ids=measurements.event_ids,
        validation=validation,
        tdoa_rms_s=full_rms,
        diagnostics=StratifiedCalibrationDiagnostics(
            attempted_subsets=attempted,
            generated_offset_roots=generated,
            metric_candidates=len(hypotheses),
            completed_hypotheses=len(hypotheses),
            geometric_class_count=1,
            selected_support=1,
            fitting_event_count=len(fitting_events),
            validation_event_count=len(validation_events),
            validation_independent_coordinates=validation.independent_coordinate_count,
            rejection_reasons=tuple(rejections),
        ),
    )


def compare_tdoa_models_8mic(
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float = 343.0,
    score_tolerance: float = 0.05,
) -> JointModelCalibrationResult:
    """Run planar and general-3D models and compare common held-out evidence."""
    fitting_events, validation_events = _split_events(len(measurements.event_ids))
    del fitting_events
    planar = calibrate_planar_tdoa_8mic(
        measurements,
        speed_of_sound=speed_of_sound,
    )
    general = calibrate_tdoa_8mic(
        measurements,
        speed_of_sound=speed_of_sound,
    )

    planar_evidence = ModelValidationEvidence(
        model="receiver2d_source3d",
        eligible=planar.status in {"solved", "weakly_identified"} and planar.validation is not None,
        validation_event_ids=validation_events,
        normalized_score=(
            None if planar.validation is None else planar.validation.normalized_huber_score
        ),
        independent_coordinate_count=(
            0 if planar.validation is None else planar.validation.independent_coordinate_count
        ),
        weak_conditioning=planar.status == "weakly_identified",
    )
    general_validation = (
        None if general.selected_class is None else general.selected_class.representative.validation
    )
    general_weak = general.status == "weakly_identified"
    if general.microphone_positions_m is not None:
        centered = general.microphone_positions_m - np.mean(
            general.microphone_positions_m, axis=0, keepdims=True
        )
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if (
            singular_values.size >= 3
            and singular_values[0] > 0.0
            and singular_values[2] / singular_values[0] < 1e-5
        ):
            general_weak = True
    general_evidence = ModelValidationEvidence(
        model="receiver3d_source3d",
        eligible=(
            general.status in {"solved", "weakly_identified"} and general_validation is not None
        ),
        validation_event_ids=validation_events,
        normalized_score=(
            None if general_validation is None else general_validation.normalized_huber_score
        ),
        independent_coordinate_count=(
            0 if general_validation is None else general_validation.independent_coordinate_count
        ),
        weak_conditioning=general_weak,
    )
    comparison = compare_model_evidence(
        planar_evidence,
        general_evidence,
        score_tolerance=score_tolerance,
    )
    return JointModelCalibrationResult(
        comparison=comparison,
        planar=planar,
        general_3d=general,
    )


def _subset_first_eight_measurements(
    measurements: EventTDOAMeasurements,
) -> EventTDOAMeasurements:
    if len(measurements.microphone_ids) < 8:
        raise ValueError("at least eight microphones are required")
    covariance = None
    if measurements.covariance_s2 is not None:
        covariance = measurements.covariance_s2[:, :7, :7]
    arrival_representatives = None
    arrival_valid = None
    if measurements.arrival_representatives_s is not None:
        arrival_representatives = measurements.arrival_representatives_s[:, :8]
        assert measurements.arrival_valid is not None
        arrival_valid = measurements.arrival_valid[:, :8]
    event_channel = measurements.event_channel
    if event_channel is not None and event_channel >= 8:
        event_channel = None
    pair_from_primitive = None
    if measurements.pair_from_primitive is not None:
        pair_from_primitive = measurements.pair_from_primitive[:7]
    return EventTDOAMeasurements(
        event_ids=measurements.event_ids,
        receiver_event_times_s=measurements.receiver_event_times_s,
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=measurements.tdoa_s[:, :7],
        sigma_s=measurements.sigma_s[:, :7],
        confidence=measurements.confidence[:, :7],
        valid=measurements.valid[:, :7],
        measurement_origin=measurements.measurement_origin,
        measurement_basis="reference_star",
        event_samples=measurements.event_samples,
        sample_rate_hz=measurements.sample_rate_hz,
        event_channel=event_channel,
        primitive_ids=measurements.primitive_ids,
        pair_from_primitive=pair_from_primitive,
        covariance_s2=covariance,
        covariance_model=measurements.covariance_model,
        arrival_representatives_s=arrival_representatives,
        arrival_valid=arrival_valid,
    )


def _restore_microphone_order(
    microphone_positions_m: np.ndarray | None,
    internal_to_input: tuple[int, ...],
) -> np.ndarray | None:
    if microphone_positions_m is None:
        return None
    values = np.asarray(microphone_positions_m, dtype=float)
    if len(values) != len(internal_to_input):
        raise ValueError("microphone position count does not match normalization mapping")
    restored = np.empty_like(values)
    for internal_index, input_index in enumerate(internal_to_input):
        restored[input_index] = values[internal_index]
    restored.setflags(write=False)
    return restored


def _restore_hypothesis_receiver_order(
    hypothesis: FullGeometryHypothesis,
    *,
    internal_to_input: tuple[int, ...],
    original_microphone_ids: tuple[int, ...],
    arrival_gauge_shifts_m: np.ndarray | None = None,
) -> FullGeometryHypothesis:
    """Map one internal hypothesis back to caller receiver order and IDs."""
    output_count = len(original_microphone_ids)
    input_count = len(hypothesis.microphone_positions_m)
    if input_count > len(internal_to_input):
        raise ValueError("hypothesis has more receivers than the normalization mapping")
    mapping = internal_to_input[:input_count]

    microphones = np.full((output_count, 3), np.nan, dtype=float)
    for internal_index, input_index in enumerate(mapping):
        microphones[input_index] = hypothesis.microphone_positions_m[internal_index]
    microphones.setflags(write=False)

    event_count = hypothesis.split.generation_mask.shape[1]
    generation = np.zeros((output_count, event_count), dtype=bool)
    completion = np.zeros_like(generation)
    validation = np.zeros_like(generation)
    for internal_index, input_index in enumerate(mapping):
        generation[input_index] = hypothesis.split.generation_mask[internal_index]
        completion[input_index] = hypothesis.split.completion_mask[internal_index]
        validation[input_index] = hypothesis.split.validation_mask[internal_index]

    completion_status = np.zeros(output_count, dtype=bool)
    if hypothesis.microphone_completion_status is not None:
        for internal_index, input_index in enumerate(mapping):
            completion_status[input_index] = bool(
                hypothesis.microphone_completion_status[internal_index]
            )
    else:
        completion_status = np.all(np.isfinite(microphones), axis=1)
    completion_status.setflags(write=False)

    seed_ids = tuple(
        original_microphone_ids[internal_to_input[int(internal_id)]]
        for internal_id in hypothesis.seed_microphone_ids
    )
    return replace(
        hypothesis,
        microphone_positions_m=microphones,
        split=MeasurementSplit(
            generation_mask=generation,
            completion_mask=completion,
            validation_mask=validation,
        ),
        seed_microphone_ids=seed_ids,
        arrival_gauge_shifts_m=(
            hypothesis.arrival_gauge_shifts_m
            if arrival_gauge_shifts_m is None
            else arrival_gauge_shifts_m
        ),
        microphone_completion_status=completion_status,
    )


def _restore_hypothesis_class_receiver_order(
    hypothesis_class: HypothesisClass | None,
    *,
    internal_to_input: tuple[int, ...],
    original_microphone_ids: tuple[int, ...],
    arrival_gauge_shifts_m: np.ndarray | None = None,
) -> HypothesisClass | None:
    if hypothesis_class is None:
        return None
    representative = _restore_hypothesis_receiver_order(
        hypothesis_class.representative,
        internal_to_input=internal_to_input,
        original_microphone_ids=original_microphone_ids,
        arrival_gauge_shifts_m=arrival_gauge_shifts_m,
    )
    members = tuple(
        _restore_hypothesis_receiver_order(
            member,
            internal_to_input=internal_to_input,
            original_microphone_ids=original_microphone_ids,
            arrival_gauge_shifts_m=arrival_gauge_shifts_m,
        )
        for member in hypothesis_class.members
    )
    return HypothesisClass(
        representative=representative,
        members=members,
        independent_subset_support=hypothesis_class.independent_subset_support,
    )


def _expand_selected_class_lineage(
    selected_class: HypothesisClass | None,
    *,
    microphone_positions_m: np.ndarray,
    source_positions_m: np.ndarray,
    completion_mask: np.ndarray,
) -> HypothesisClass | None:
    """Expand the selected eight-microphone hypothesis lineage to the full array."""
    if selected_class is None:
        return None
    representative = selected_class.representative
    receiver_count, event_count = completion_mask.shape
    core_receiver_count = representative.split.generation_mask.shape[0]
    if representative.split.generation_mask.shape[1] != event_count:
        raise ValueError("completion lineage event count does not match the selected hypothesis")
    if receiver_count < core_receiver_count:
        raise ValueError("completion lineage cannot shrink the selected hypothesis")

    generation = np.zeros((receiver_count, event_count), dtype=bool)
    completion = np.zeros_like(generation)
    validation = np.zeros_like(generation)
    generation[:core_receiver_count] = representative.split.generation_mask
    completion[:core_receiver_count] = representative.split.completion_mask
    validation[:core_receiver_count] = representative.split.validation_mask
    completion |= completion_mask

    expanded = FullGeometryHypothesis(
        subset_id=representative.subset_id,
        root_id=representative.root_id,
        microphone_positions_m=microphone_positions_m,
        source_positions_m=source_positions_m,
        split=MeasurementSplit(
            generation_mask=generation,
            completion_mask=completion,
            validation_mask=validation,
        ),
        validation=representative.validation,
        full_tdoa_rms_s=representative.full_tdoa_rms_s,
        metric_rms_m=representative.metric_rms_m,
        metric_condition_number=representative.metric_condition_number,
        affine_factor_singular_ratio=representative.affine_factor_singular_ratio,
        model=representative.model,
        seed_microphone_ids=representative.seed_microphone_ids,
        seed_event_ids=representative.seed_event_ids,
        metric_branch_id=representative.metric_branch_id,
        arrival_gauge_shifts_m=representative.arrival_gauge_shifts_m,
        offset_scope_event_ids=representative.offset_scope_event_ids,
        microphone_completion_status=np.all(
            np.isfinite(microphone_positions_m),
            axis=1,
        ),
        source_completion_status=np.all(
            np.isfinite(source_positions_m),
            axis=1,
        ),
        generating_equation_rms=representative.generating_equation_rms,
    )
    return HypothesisClass(
        representative=expanded,
        members=(expanded,),
        independent_subset_support=selected_class.independent_subset_support,
    )


def calibrate_tdoa(
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float = 343.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    root_start_count: int = 32,
    metric_start_count: int = 20,
    class_tolerance_m: float = 2e-4,
    ambiguity_score_tolerance: float = 0.05,
    extra_microphone_inlier_rms_m: float = 0.05,
) -> StratifiedCalibrationResult:
    """Calibrate an 8-or-more microphone 3-D array from reference-star TDOAs."""
    microphone_count = len(measurements.microphone_ids)
    if microphone_count < 8:
        raise ValueError("calibrate_tdoa requires at least eight microphones")
    original_microphone_ids = measurements.microphone_ids
    input_measurements = measurements
    measurements, internal_to_input = _normalize_reference_star_measurements(measurements)
    arrival_gauge_shifts_m = (
        _reference_change_gauge_shifts_s(
            input_measurements,
            new_reference_input_index=internal_to_input[0],
        )
        * speed_of_sound
    )
    if extra_microphone_inlier_rms_m <= 0.0:
        raise ValueError("extra_microphone_inlier_rms_m must be positive")

    core_measurements = (
        measurements if microphone_count == 8 else _subset_first_eight_measurements(measurements)
    )
    core = calibrate_tdoa_8mic(
        core_measurements,
        speed_of_sound=speed_of_sound,
        receiver_subset_budget=receiver_subset_budget,
        event_subset_budget=event_subset_budget,
        root_start_count=root_start_count,
        metric_start_count=metric_start_count,
        class_tolerance_m=class_tolerance_m,
        ambiguity_score_tolerance=ambiguity_score_tolerance,
    )
    restored_core_selected = _restore_hypothesis_class_receiver_order(
        core.selected_class,
        internal_to_input=internal_to_input,
        original_microphone_ids=original_microphone_ids,
        arrival_gauge_shifts_m=arrival_gauge_shifts_m,
    )
    restored_core_classes = tuple(
        item
        for item in (
            _restore_hypothesis_class_receiver_order(
                hypothesis_class,
                internal_to_input=internal_to_input,
                original_microphone_ids=original_microphone_ids,
                arrival_gauge_shifts_m=arrival_gauge_shifts_m,
            )
            for hypothesis_class in core.classes
        )
        if item is not None
    )
    if microphone_count == 8 or core.microphone_positions_m is None:
        return replace(
            core,
            microphone_positions_m=_restore_microphone_order(
                core.microphone_positions_m,
                internal_to_input,
            ),
            selected_class=restored_core_selected,
            classes=restored_core_classes,
            microphone_ids=original_microphone_ids,
        )
    if core.source_positions_m is None:
        return replace(
            core,
            selected_class=restored_core_selected,
            classes=restored_core_classes,
            microphone_ids=original_microphone_ids,
        )

    source_positions = np.asarray(core.source_positions_m, dtype=float)
    finite_sources = np.all(np.isfinite(source_positions), axis=1)
    reference_ranges = np.linalg.norm(
        source_positions - core.microphone_positions_m[0],
        axis=1,
    )
    microphones = np.full((microphone_count, 3), np.nan, dtype=float)
    microphones[:8] = core.microphone_positions_m
    completion_rms: list[float] = []
    extra_completion_mask = np.zeros(
        (microphone_count, len(measurements.event_ids)),
        dtype=bool,
    )

    for microphone in range(8, microphone_count):
        column = microphone - 1
        valid = measurements.valid[:, column] & finite_sources
        if int(np.sum(valid)) < 4:
            return StratifiedCalibrationResult(
                status="insufficient_data",
                microphone_positions_m=None,
                source_positions_m=core.source_positions_m,
                event_ids=measurements.event_ids,
                tdoa_rms_s=None,
                selected_class=restored_core_selected,
                classes=restored_core_classes,
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=core.diagnostics.attempted_subsets,
                    generated_offset_roots=core.diagnostics.generated_offset_roots,
                    metric_candidates=core.diagnostics.metric_candidates,
                    completed_hypotheses=core.diagnostics.completed_hypotheses,
                    geometric_class_count=core.diagnostics.geometric_class_count,
                    selected_support=core.diagnostics.selected_support,
                    fitting_event_count=core.diagnostics.fitting_event_count,
                    validation_event_count=core.diagnostics.validation_event_count,
                    validation_independent_coordinates=(
                        core.diagnostics.validation_independent_coordinates
                    ),
                    rejection_reasons=(
                        *core.diagnostics.rejection_reasons,
                        f"insufficient_ranges_for_microphone_{microphone}",
                    ),
                    extra_microphones_completed=microphone - 8,
                    extra_microphone_max_inlier_rms_m=(
                        None if not completion_rms else max(completion_rms)
                    ),
                ),
                microphone_ids=original_microphone_ids,
            )
        absolute_ranges = (
            reference_ranges[valid] + speed_of_sound * measurements.tdoa_s[valid, column]
        )
        if np.min(absolute_ranges) < 0.0:
            return StratifiedCalibrationResult(
                status="failed",
                microphone_positions_m=None,
                source_positions_m=core.source_positions_m,
                event_ids=measurements.event_ids,
                tdoa_rms_s=None,
                selected_class=restored_core_selected,
                classes=restored_core_classes,
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=core.diagnostics.attempted_subsets,
                    generated_offset_roots=core.diagnostics.generated_offset_roots,
                    metric_candidates=core.diagnostics.metric_candidates,
                    completed_hypotheses=core.diagnostics.completed_hypotheses,
                    geometric_class_count=core.diagnostics.geometric_class_count,
                    selected_support=core.diagnostics.selected_support,
                    fitting_event_count=core.diagnostics.fitting_event_count,
                    validation_event_count=core.diagnostics.validation_event_count,
                    validation_independent_coordinates=(
                        core.diagnostics.validation_independent_coordinates
                    ),
                    rejection_reasons=(
                        *core.diagnostics.rejection_reasons,
                        f"negative_range_for_microphone_{microphone}",
                    ),
                    extra_microphones_completed=microphone - 8,
                    extra_microphone_max_inlier_rms_m=(
                        None if not completion_rms else max(completion_rms)
                    ),
                ),
                microphone_ids=original_microphone_ids,
            )
        localized = localize_receiver_from_ranges(
            source_positions[valid],
            absolute_ranges,
            robust=True,
        )
        if localized.linear_rank < 3 or localized.inlier_rms_m > extra_microphone_inlier_rms_m:
            return StratifiedCalibrationResult(
                status="weakly_identified",
                microphone_positions_m=None,
                source_positions_m=core.source_positions_m,
                event_ids=measurements.event_ids,
                tdoa_rms_s=None,
                selected_class=restored_core_selected,
                classes=restored_core_classes,
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=core.diagnostics.attempted_subsets,
                    generated_offset_roots=core.diagnostics.generated_offset_roots,
                    metric_candidates=core.diagnostics.metric_candidates,
                    completed_hypotheses=core.diagnostics.completed_hypotheses,
                    geometric_class_count=core.diagnostics.geometric_class_count,
                    selected_support=core.diagnostics.selected_support,
                    fitting_event_count=core.diagnostics.fitting_event_count,
                    validation_event_count=core.diagnostics.validation_event_count,
                    validation_independent_coordinates=(
                        core.diagnostics.validation_independent_coordinates
                    ),
                    rejection_reasons=(
                        *core.diagnostics.rejection_reasons,
                        f"weak_completion_for_microphone_{microphone}",
                    ),
                    extra_microphones_completed=microphone - 8,
                    extra_microphone_max_inlier_rms_m=(
                        None if not completion_rms else max(completion_rms)
                    ),
                ),
                microphone_ids=original_microphone_ids,
            )
        microphones[microphone] = localized.receiver_position_m
        extra_completion_mask[microphone, valid] = True
        completion_rms.append(localized.inlier_rms_m)

    prediction = _tdoa_predictions(microphones, source_positions, speed_of_sound)
    valid = measurements.valid & np.isfinite(prediction)
    residual = prediction[valid] - measurements.tdoa_s[valid]
    rms = float(np.sqrt(np.mean(residual * residual)))
    microphones.setflags(write=False)
    expanded_selected = _expand_selected_class_lineage(
        core.selected_class,
        microphone_positions_m=microphones,
        source_positions_m=source_positions,
        completion_mask=extra_completion_mask,
    )
    internal_classes = core.classes
    if expanded_selected is not None and internal_classes:
        internal_classes = (expanded_selected, *internal_classes[1:])
    restored_selected = _restore_hypothesis_class_receiver_order(
        expanded_selected,
        internal_to_input=internal_to_input,
        original_microphone_ids=original_microphone_ids,
        arrival_gauge_shifts_m=arrival_gauge_shifts_m,
    )
    restored_classes = tuple(
        item
        for item in (
            _restore_hypothesis_class_receiver_order(
                hypothesis_class,
                internal_to_input=internal_to_input,
                original_microphone_ids=original_microphone_ids,
                arrival_gauge_shifts_m=arrival_gauge_shifts_m,
            )
            for hypothesis_class in internal_classes
        )
        if item is not None
    )
    return StratifiedCalibrationResult(
        status=core.status,
        microphone_positions_m=_restore_microphone_order(
            microphones,
            internal_to_input,
        ),
        source_positions_m=core.source_positions_m,
        event_ids=measurements.event_ids,
        tdoa_rms_s=rms,
        selected_class=restored_selected,
        classes=restored_classes,
        diagnostics=StratifiedCalibrationDiagnostics(
            attempted_subsets=core.diagnostics.attempted_subsets,
            generated_offset_roots=core.diagnostics.generated_offset_roots,
            metric_candidates=core.diagnostics.metric_candidates,
            completed_hypotheses=core.diagnostics.completed_hypotheses,
            geometric_class_count=core.diagnostics.geometric_class_count,
            selected_support=core.diagnostics.selected_support,
            fitting_event_count=core.diagnostics.fitting_event_count,
            validation_event_count=core.diagnostics.validation_event_count,
            validation_independent_coordinates=core.diagnostics.validation_independent_coordinates,
            rejection_reasons=core.diagnostics.rejection_reasons,
            extra_microphones_completed=microphone_count - 8,
            extra_microphone_max_inlier_rms_m=(None if not completion_rms else max(completion_rms)),
        ),
        microphone_ids=original_microphone_ids,
    )


def calibrate_planar_tdoa(
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float = 343.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    membership_tolerance: float = 5e-3,
    metric_acceptance_rms_m: float = 5e-3,
    extra_microphone_rms_m: float = 0.02,
    angle_constraint: PlanarAngleConstraint | None = None,
) -> PlanarCalibrationResult:
    """Calibrate an 8-or-more microphone planar array from reference-star TDOAs."""
    microphone_count = len(measurements.microphone_ids)
    if microphone_count < 8:
        raise ValueError("calibrate_planar_tdoa requires at least eight microphones")
    original_microphone_ids = measurements.microphone_ids
    original_angle_constraint = angle_constraint
    required_ids = (
        ()
        if angle_constraint is None
        else (
            angle_constraint.center_receiver,
            angle_constraint.arm_a_receiver,
            angle_constraint.arm_b_receiver,
        )
    )
    measurements, internal_to_input = _normalize_reference_star_measurements(
        measurements,
        required_microphone_ids=required_ids,
    )
    if angle_constraint is not None:
        id_to_input = {
            microphone_id: index for index, microphone_id in enumerate(original_microphone_ids)
        }
        input_to_internal = {
            input_index: internal_index
            for internal_index, input_index in enumerate(internal_to_input)
        }
        angle_constraint = PlanarAngleConstraint(
            center_receiver=input_to_internal[id_to_input[angle_constraint.center_receiver]],
            arm_a_receiver=input_to_internal[id_to_input[angle_constraint.arm_a_receiver]],
            arm_b_receiver=input_to_internal[id_to_input[angle_constraint.arm_b_receiver]],
            angle_rad=angle_constraint.angle_rad,
            provenance=angle_constraint.provenance,
            exact=angle_constraint.exact,
        )
    if extra_microphone_rms_m <= 0.0:
        raise ValueError("extra_microphone_rms_m must be positive")

    core_measurements = (
        measurements if microphone_count == 8 else _subset_first_eight_measurements(measurements)
    )
    core = calibrate_planar_tdoa_8mic(
        core_measurements,
        speed_of_sound=speed_of_sound,
        receiver_subset_budget=receiver_subset_budget,
        event_subset_budget=event_subset_budget,
        membership_tolerance=membership_tolerance,
        metric_acceptance_rms_m=metric_acceptance_rms_m,
        angle_constraint=angle_constraint,
    )
    if microphone_count == 8 or core.microphone_positions_m is None:
        return replace(
            core,
            microphone_positions_m=_restore_microphone_order(
                core.microphone_positions_m,
                internal_to_input,
            ),
            microphone_ids=original_microphone_ids,
            angle_constraint=original_angle_constraint,
        )
    if (
        core.source_projected_positions_m is None
        or core.source_unsigned_heights_m is None
        or core.source_representative_positions_m is None
    ):
        return replace(
            core,
            microphone_ids=original_microphone_ids,
            angle_constraint=original_angle_constraint,
        )

    microphones = np.full((microphone_count, 3), np.nan, dtype=float)
    microphones[:8] = core.microphone_positions_m
    projected = np.asarray(core.source_projected_positions_m, dtype=float)
    heights = np.asarray(core.source_unsigned_heights_m, dtype=float)
    representative_sources = np.asarray(
        core.source_representative_positions_m,
        dtype=float,
    )
    finite_sources = (
        np.all(np.isfinite(projected), axis=1)
        & np.isfinite(heights)
        & np.all(np.isfinite(representative_sources), axis=1)
    )
    reference_ranges = np.linalg.norm(
        representative_sources - microphones[0],
        axis=1,
    )
    completion_rms: list[float] = []

    for microphone in range(8, microphone_count):
        column = microphone - 1
        valid = measurements.valid[:, column] & finite_sources
        if int(np.sum(valid)) < 3:
            return PlanarCalibrationResult(
                status="insufficient_data",
                microphone_positions_m=None,
                source_projected_positions_m=core.source_projected_positions_m,
                source_unsigned_heights_m=core.source_unsigned_heights_m,
                source_height_sign_known=core.source_height_sign_known,
                source_representative_positions_m=core.source_representative_positions_m,
                event_ids=core.event_ids,
                validation=core.validation,
                tdoa_rms_s=None,
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=core.diagnostics.attempted_subsets,
                    generated_offset_roots=core.diagnostics.generated_offset_roots,
                    metric_candidates=core.diagnostics.metric_candidates,
                    completed_hypotheses=core.diagnostics.completed_hypotheses,
                    geometric_class_count=core.diagnostics.geometric_class_count,
                    selected_support=core.diagnostics.selected_support,
                    fitting_event_count=core.diagnostics.fitting_event_count,
                    validation_event_count=core.diagnostics.validation_event_count,
                    validation_independent_coordinates=(
                        core.diagnostics.validation_independent_coordinates
                    ),
                    rejection_reasons=(
                        *core.diagnostics.rejection_reasons,
                        f"insufficient_planar_ranges_for_microphone_{microphone}",
                    ),
                    extra_microphones_completed=microphone - 8,
                    extra_microphone_max_inlier_rms_m=(
                        None if not completion_rms else max(completion_rms)
                    ),
                ),
            )

        absolute_ranges = (
            reference_ranges[valid] + speed_of_sound * measurements.tdoa_s[valid, column]
        )
        try:
            localized = localize_planar_receiver_from_ranges(
                projected[valid],
                heights[valid],
                absolute_ranges,
            )
        except ValueError:
            return PlanarCalibrationResult(
                status="weakly_identified",
                microphone_positions_m=None,
                source_projected_positions_m=core.source_projected_positions_m,
                source_unsigned_heights_m=core.source_unsigned_heights_m,
                source_height_sign_known=core.source_height_sign_known,
                source_representative_positions_m=core.source_representative_positions_m,
                event_ids=core.event_ids,
                validation=core.validation,
                tdoa_rms_s=None,
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=core.diagnostics.attempted_subsets,
                    generated_offset_roots=core.diagnostics.generated_offset_roots,
                    metric_candidates=core.diagnostics.metric_candidates,
                    completed_hypotheses=core.diagnostics.completed_hypotheses,
                    geometric_class_count=core.diagnostics.geometric_class_count,
                    selected_support=core.diagnostics.selected_support,
                    fitting_event_count=core.diagnostics.fitting_event_count,
                    validation_event_count=core.diagnostics.validation_event_count,
                    validation_independent_coordinates=(
                        core.diagnostics.validation_independent_coordinates
                    ),
                    rejection_reasons=(
                        *core.diagnostics.rejection_reasons,
                        f"planar_completion_failed_for_microphone_{microphone}",
                    ),
                    extra_microphones_completed=microphone - 8,
                    extra_microphone_max_inlier_rms_m=(
                        None if not completion_rms else max(completion_rms)
                    ),
                ),
            )
        if localized.linear_rank < 2 or localized.range_rms_m > extra_microphone_rms_m:
            return PlanarCalibrationResult(
                status="weakly_identified",
                microphone_positions_m=None,
                source_projected_positions_m=core.source_projected_positions_m,
                source_unsigned_heights_m=core.source_unsigned_heights_m,
                source_height_sign_known=core.source_height_sign_known,
                source_representative_positions_m=core.source_representative_positions_m,
                event_ids=core.event_ids,
                validation=core.validation,
                tdoa_rms_s=None,
                diagnostics=StratifiedCalibrationDiagnostics(
                    attempted_subsets=core.diagnostics.attempted_subsets,
                    generated_offset_roots=core.diagnostics.generated_offset_roots,
                    metric_candidates=core.diagnostics.metric_candidates,
                    completed_hypotheses=core.diagnostics.completed_hypotheses,
                    geometric_class_count=core.diagnostics.geometric_class_count,
                    selected_support=core.diagnostics.selected_support,
                    fitting_event_count=core.diagnostics.fitting_event_count,
                    validation_event_count=core.diagnostics.validation_event_count,
                    validation_independent_coordinates=(
                        core.diagnostics.validation_independent_coordinates
                    ),
                    rejection_reasons=(
                        *core.diagnostics.rejection_reasons,
                        f"weak_planar_completion_for_microphone_{microphone}",
                    ),
                    extra_microphones_completed=microphone - 8,
                    extra_microphone_max_inlier_rms_m=(
                        None if not completion_rms else max(completion_rms)
                    ),
                ),
            )
        microphones[microphone] = [
            localized.position_2d_m[0],
            localized.position_2d_m[1],
            0.0,
        ]
        completion_rms.append(localized.range_rms_m)

    predicted = _tdoa_predictions(
        microphones,
        representative_sources,
        speed_of_sound,
    )
    valid = measurements.valid & np.isfinite(predicted)
    residual = predicted[valid] - measurements.tdoa_s[valid]
    rms = float(np.sqrt(np.mean(residual * residual)))
    microphones.setflags(write=False)
    return PlanarCalibrationResult(
        status=core.status,
        microphone_positions_m=_restore_microphone_order(
            microphones,
            internal_to_input,
        ),
        source_projected_positions_m=core.source_projected_positions_m,
        source_unsigned_heights_m=core.source_unsigned_heights_m,
        source_height_sign_known=core.source_height_sign_known,
        source_representative_positions_m=core.source_representative_positions_m,
        event_ids=core.event_ids,
        validation=core.validation,
        tdoa_rms_s=rms,
        diagnostics=StratifiedCalibrationDiagnostics(
            attempted_subsets=core.diagnostics.attempted_subsets,
            generated_offset_roots=core.diagnostics.generated_offset_roots,
            metric_candidates=core.diagnostics.metric_candidates,
            completed_hypotheses=core.diagnostics.completed_hypotheses,
            geometric_class_count=core.diagnostics.geometric_class_count,
            selected_support=core.diagnostics.selected_support,
            fitting_event_count=core.diagnostics.fitting_event_count,
            validation_event_count=core.diagnostics.validation_event_count,
            validation_independent_coordinates=(
                core.diagnostics.validation_independent_coordinates
            ),
            rejection_reasons=core.diagnostics.rejection_reasons,
            extra_microphones_completed=microphone_count - 8,
            extra_microphone_max_inlier_rms_m=(None if not completion_rms else max(completion_rms)),
        ),
        microphone_ids=original_microphone_ids,
        angle_constraint=original_angle_constraint,
    )
