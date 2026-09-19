from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

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
) -> tuple[FullGeometryHypothesis, np.ndarray] | None:
    seed_receivers = np.asarray(receiver_subset, dtype=int)
    seed = np.asarray(seed_events, dtype=int)
    excluded = next(index for index in range(8) if index not in receiver_subset)

    complete_fitting = np.asarray(
        [event for event in fitting_events if np.all(primitive_valid[seed_receivers, event])],
        dtype=int,
    )
    if len(complete_fitting) < 6:
        return None

    expanded = extend_event_offsets(
        arrivals_m[np.ix_(seed_receivers, seed)],
        seed_root_offsets_m,
        arrivals_m[np.ix_(seed_receivers, complete_fitting)],
    )
    successful_events = complete_fitting[expanded.success]
    successful_offsets = expanded.offsets_m[expanded.success]
    successful_ranges = expanded.corrected_ranges_m[:, expanded.success]
    if len(successful_events) < 6:
        return None

    factorization = factor_corrected_ranges(
        successful_ranges,
        dimension=3,
    )
    metric = upgrade_metric_3d_overdetermined(
        factorization,
        successful_ranges,
        start_count=metric_start_count,
        acceptance_rms_m=2e-5,
    )
    accepted = [
        candidate
        for candidate in metric.candidates
        if candidate.corrected_range_rms_m <= metric.diagnostics.acceptance_rms_m
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
        if localized_receiver.linear_rank < 3 or localized_receiver.inlier_rms_m > 1e-5:
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
            if source.linear_rank < 4 or abs(source.norm_residual_m2) > 1e-3 * range_scale**2:
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
    return min(
        completed_hypotheses,
        key=lambda item: (
            item[0].validation.normalized_huber_score,
            item[0].full_tdoa_rms_s,
            item[0].metric_rms_m,
        ),
    )


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
            )
            roots_generated += len(minimal.roots)
            for root_id, root in enumerate(minimal.roots):
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
                hypothesis, _ = completed
                hypotheses.append(hypothesis)
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
    return StratifiedCalibrationResult(
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
                    or identifiability.continuous_metric_nullity
                    > continuous_ambiguity_dimension
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
