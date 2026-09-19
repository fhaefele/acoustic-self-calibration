import numpy as np
import pytest

from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene
from acoustic_self_calibration.stratified.solver import calibrate_tdoa_8mic


def _measurements(event_count: int, *, sigma_s: float = 2e-6):
    scene = make_random_3d_pulse_scene(8, event_count=event_count)
    ranges = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.source_positions_at_events_m[None, :, :],
        axis=2,
    )
    tdoa = ((ranges[1:] - ranges[[0]]) / 343.0).T
    shape = tdoa.shape
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(event_count),
        receiver_event_times_s=scene.event_times_s,
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=tdoa,
        sigma_s=np.full(shape, sigma_s),
        confidence=np.ones(shape),
        valid=np.ones(shape, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    return scene, measurements


@pytest.mark.parametrize("event_count", [20, 40])
def test_8mic_tdoa_only_gate(event_count: int) -> None:
    scene, measurements = _measurements(event_count)
    result = calibrate_tdoa_8mic(
        measurements,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=32,
        metric_start_count=16,
    )

    assert result.status == "solved"
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    aligned_sources = apply_rigid(
        result.source_positions_m,
        rotation,
        translation,
    )
    assert rms_position_error(aligned_microphones, scene.microphone_positions_m) < 0.15
    assert rms_position_error(aligned_sources, scene.source_positions_at_events_m) < 0.18
    assert result.tdoa_rms_s is not None
    assert result.tdoa_rms_s < 60e-6
    assert result.diagnostics.selected_support >= 2
    assert result.diagnostics.validation_independent_coordinates > 0
    assert result.selected_class is not None
    hypothesis = result.selected_class.representative
    assert hypothesis.model == "receiver3d_source3d"
    assert len(hypothesis.seed_microphone_ids) == 7
    assert len(hypothesis.seed_event_ids) == 6
    assert hypothesis.metric_branch_id is not None
    assert hypothesis.arrival_gauge_shifts_m is not None
    assert hypothesis.arrival_gauge_shifts_m.shape == (event_count,)
    assert hypothesis.offset_scope_event_ids
    assert hypothesis.microphone_completion_status is not None
    assert np.all(hypothesis.microphone_completion_status)
    assert hypothesis.source_completion_status is not None
    assert np.all(hypothesis.source_completion_status)
    assert hypothesis.generating_equation_rms is not None


def test_sparse_missing_seed_measurement_is_skipped_not_zero_filled() -> None:
    _, measurements = _measurements(20)
    valid = np.array(measurements.valid, copy=True)
    tdoa = np.array(measurements.tdoa_s, copy=True)
    sigma = np.array(measurements.sigma_s, copy=True)
    confidence = np.array(measurements.confidence, copy=True)
    valid[0, 6] = False
    tdoa[0, 6] = np.nan
    sigma[0, 6] = np.nan
    confidence[0, 6] = np.nan
    degraded = EventTDOAMeasurements(
        event_ids=measurements.event_ids,
        receiver_event_times_s=measurements.receiver_event_times_s,
        microphone_ids=measurements.microphone_ids,
        microphone_pairs=measurements.microphone_pairs,
        tdoa_s=tdoa,
        sigma_s=sigma,
        confidence=confidence,
        valid=valid,
        measurement_origin=measurements.measurement_origin,
        measurement_basis=measurements.measurement_basis,
    )
    result = calibrate_tdoa_8mic(
        degraded,
        receiver_subset_budget=3,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )
    assert result.status in {"solved", "ambiguous"}
    assert result.diagnostics.attempted_subsets >= 1


def test_validation_outlier_is_robustly_scored_without_refitting_geometry() -> None:
    _, measurements = _measurements(20)
    baseline = calibrate_tdoa_8mic(
        measurements,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )
    tdoa = np.array(measurements.tdoa_s, copy=True)
    tdoa[-2, -1] += 400e-6
    contaminated = EventTDOAMeasurements(
        event_ids=measurements.event_ids,
        receiver_event_times_s=measurements.receiver_event_times_s,
        microphone_ids=measurements.microphone_ids,
        microphone_pairs=measurements.microphone_pairs,
        tdoa_s=tdoa,
        sigma_s=measurements.sigma_s,
        confidence=measurements.confidence,
        valid=measurements.valid,
        measurement_origin=measurements.measurement_origin,
        measurement_basis=measurements.measurement_basis,
    )
    result = calibrate_tdoa_8mic(
        contaminated,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )
    assert baseline.microphone_positions_m is not None
    assert result.microphone_positions_m is not None
    aligned, _, _ = rigid_align(
        result.microphone_positions_m,
        baseline.microphone_positions_m,
    )
    assert rms_position_error(aligned, baseline.microphone_positions_m) < 1e-4
    assert result.selected_class is not None
    assert result.selected_class.representative.validation.inlier_fraction < 1.0


def test_fitting_outlier_does_not_destroy_all_hypotheses() -> None:
    scene, measurements = _measurements(20)
    tdoa = np.array(measurements.tdoa_s, copy=True)
    tdoa[0, 6] += 1.5e-3
    contaminated = EventTDOAMeasurements(
        event_ids=measurements.event_ids,
        receiver_event_times_s=measurements.receiver_event_times_s,
        microphone_ids=measurements.microphone_ids,
        microphone_pairs=measurements.microphone_pairs,
        tdoa_s=tdoa,
        sigma_s=measurements.sigma_s,
        confidence=measurements.confidence,
        valid=measurements.valid,
        measurement_origin=measurements.measurement_origin,
        measurement_basis=measurements.measurement_basis,
    )
    result = calibrate_tdoa_8mic(
        contaminated,
        receiver_subset_budget=3,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )
    assert result.status in {"solved", "ambiguous", "weakly_identified"}
    assert result.microphone_positions_m is not None
    aligned, _, _ = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    assert rms_position_error(aligned, scene.microphone_positions_m) < 0.15
    assert result.diagnostics.completed_hypotheses >= 1
