import numpy as np
import pytest

from acoustic_self_calibration.evaluation import (
    evaluate_against_ground_truth,
    evaluate_scenes,
)
from acoustic_self_calibration.events import EventDetection
from acoustic_self_calibration.ground_truth import GroundTruth
from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.pipeline import AudioCalibrationResult
from acoustic_self_calibration.stratified.solver import (
    StratifiedCalibrationDiagnostics,
    StratifiedCalibrationResult,
)


def _result_and_truth() -> tuple[AudioCalibrationResult, GroundTruth]:
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.1],
            [0.1, 1.0, 0.0],
            [0.2, 0.3, 1.1],
            [0.8, 0.5, 0.3],
            [-0.4, 0.7, 0.9],
            [0.6, -0.5, 1.0],
            [-0.7, -0.3, 0.4],
        ]
    )
    event_count = 12
    sample_rate = 48_000
    samples = 1000 + np.arange(event_count) * 4000
    times = samples / sample_rate
    source = np.column_stack(
        [
            np.linspace(1.5, 0.2, event_count),
            np.linspace(0.0, 1.0, event_count),
            np.linspace(0.8, 1.1, event_count),
        ]
    )

    angle = np.deg2rad(32.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([2.0, -0.7, 0.4])
    estimated_mics = (microphones - translation) @ rotation.T
    estimated_source = (source - translation) @ rotation.T

    diagnostics = StratifiedCalibrationDiagnostics(
        attempted_subsets=2,
        generated_offset_roots=2,
        metric_candidates=1,
        completed_hypotheses=1,
        geometric_class_count=1,
        selected_support=1,
        fitting_event_count=8,
        validation_event_count=4,
        validation_independent_coordinates=12,
        rejection_reasons=(),
    )
    calibration = StratifiedCalibrationResult(
        status="solved",
        microphone_positions_m=estimated_mics,
        source_positions_m=estimated_source,
        event_ids=np.arange(event_count),
        tdoa_rms_s=1e-6,
        selected_class=None,
        classes=(),
        diagnostics=diagnostics,
    )
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(event_count),
        receiver_event_times_s=times,
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=np.zeros((event_count, 7)),
        sigma_s=np.full((event_count, 7), 2e-6),
        confidence=np.ones((event_count, 7)),
        valid=np.ones((event_count, 7), dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    detection = EventDetection(
        event_samples=samples,
        receiver_event_times_s=times,
        event_channel=0,
        prominence=np.ones(event_count),
    )
    result = AudioCalibrationResult(
        calibration=calibration,
        measurements=measurements,
        detection=detection,
        speed_of_sound_mps=343.0,
        model="general_3d",
        temporal_tracking_enabled=True,
        refinement_mode="none",
    )
    truth = GroundTruth(
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=source,
        metadata={},
    )
    return result, truth


def test_evaluation_uses_microphone_fitted_transform_for_entire_scene() -> None:
    result, truth = _result_and_truth()
    evaluation = evaluate_against_ground_truth(result, truth)
    assert evaluation.microphone_rms_error_m < 1e-12
    assert evaluation.source_rms_error_m < 1e-12
    assert evaluation.microphone_uncertainty_diagnostic is None
    assert evaluation.source_uncertainty_diagnostic is None


def test_evaluation_clips_source_to_reference_time_overlap() -> None:
    result, truth = _result_and_truth()
    partial = GroundTruth(
        microphone_positions_m=truth.microphone_positions_m,
        source_times_s=truth.source_times_s[3:9],
        source_positions_m=truth.source_positions_m[3:9],
        metadata={},
    )
    evaluation = evaluate_against_ground_truth(result, partial)
    assert np.array_equal(
        evaluation.source_estimate_indices,
        np.arange(3, 9),
    )
    assert evaluation.source_rms_error_m < 1e-12


def test_evaluation_rejects_unsolved_result() -> None:
    result, truth = _result_and_truth()
    calibration = StratifiedCalibrationResult(
        status="failed",
        microphone_positions_m=None,
        source_positions_m=None,
        event_ids=result.calibration.event_ids,
        tdoa_rms_s=None,
        selected_class=None,
        classes=(),
        diagnostics=result.calibration.diagnostics,
    )
    failed = AudioCalibrationResult(
        calibration=calibration,
        measurements=result.measurements,
        detection=result.detection,
        speed_of_sound_mps=343.0,
        model="general_3d",
        temporal_tracking_enabled=True,
        refinement_mode="none",
    )
    with pytest.raises(ValueError, match="no solved geometry"):
        evaluate_against_ground_truth(failed, truth)


def test_evaluate_scenes_preserves_overlap_semantics() -> None:
    result, truth = _result_and_truth()
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
    estimate = GroundTruth(
        microphone_positions_m=result.microphone_positions_m,
        source_times_s=result.event_times_s,
        source_positions_m=result.source_positions_m,
        metadata={},
        scene_role="estimate",
    )
    evaluation = evaluate_scenes(estimate, truth)
    assert len(evaluation.source_estimate_indices) == len(result.event_times_s)
