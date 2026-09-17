from dataclasses import replace

import numpy as np
import pytest

from acoustic_self_calibration.bayesian import BayesianCalibrationResult
from acoustic_self_calibration.evaluation import evaluate_against_ground_truth
from acoustic_self_calibration.ground_truth import GroundTruth
from acoustic_self_calibration.pipeline import AudioCalibrationResult


def _result_and_truth() -> tuple[AudioCalibrationResult, GroundTruth]:
    ground_truth_mics = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.1], [0.1, 1.0, 0.0], [0.2, 0.3, 1.1]]
    )
    gt_times = np.array([0.0, 0.5, 1.0, 1.5])
    gt_source = np.array([[1.5, 0.0, 0.8], [1.3, 0.4, 0.9], [0.8, 0.8, 1.0], [0.2, 1.0, 1.1]])

    angle = np.deg2rad(32.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([2.0, -0.7, 0.4])
    estimated_mics = (ground_truth_mics - translation) @ rotation.T
    estimated_source = (gt_source - translation) @ rotation.T

    calibration = BayesianCalibrationResult(
        microphone_positions=estimated_mics,
        source_positions=estimated_source,
        speed_of_sound=343.0,
        clock_offsets_s=np.zeros(4),
        clock_drifts=np.zeros(4),
        microphone_position_std_m=np.full((4, 3), 0.02),
        source_position_std_m=np.full((4, 3), 0.03),
        speed_of_sound_std=None,
        clock_offset_std_s=None,
        clock_drift_std=None,
        rms_tdoa_residual_s=1e-6,
        normalized_data_rms=0.5,
        negative_log_posterior=2.0,
        success=True,
        message="ok",
        nfev=10,
    )
    result = AudioCalibrationResult(
        calibration=calibration,
        frame_times_s=gt_times.copy(),
        tdoa_s=np.zeros((4, 3)),
        tdoa_sigma_s=np.full((4, 3), 2e-6),
        confidence=np.ones((4, 3)),
        microphone_pairs=((0, 1), (0, 2), (0, 3)),
    )
    truth = GroundTruth(
        microphone_positions_m=ground_truth_mics,
        source_times_s=gt_times,
        source_positions_m=gt_source,
        metadata={},
    )
    return result, truth


def test_evaluation_uses_one_microphone_fitted_transform_for_entire_scene() -> None:
    result, truth = _result_and_truth()
    evaluation = evaluate_against_ground_truth(result, truth)
    assert evaluation.microphone_rms_error_m < 1e-12
    assert evaluation.source_rms_error_m < 1e-12
    assert np.allclose(evaluation.aligned_microphone_positions_m, truth.microphone_positions_m)
    assert np.allclose(evaluation.aligned_source_positions_m, truth.source_positions_m)
    assert np.array_equal(evaluation.source_estimate_indices, np.arange(4))
    assert evaluation.microphone_uncertainty_diagnostic is not None
    assert evaluation.source_uncertainty_diagnostic is not None


def test_evaluation_interpolates_source_ground_truth_to_estimate_times() -> None:
    result, truth = _result_and_truth()
    source_std = result.calibration.source_position_std_m
    assert source_std is not None
    result = AudioCalibrationResult(
        calibration=replace(
            result.calibration,
            source_positions=result.calibration.source_positions[[0, 2]],
            source_position_std_m=source_std[[0, 2]],
        ),
        frame_times_s=np.array([0.0, 1.0]),
        tdoa_s=np.zeros((2, 3)),
        tdoa_sigma_s=np.full((2, 3), 2e-6),
        confidence=np.ones((2, 3)),
        microphone_pairs=result.microphone_pairs,
    )
    evaluation = evaluate_against_ground_truth(result, truth)
    assert np.allclose(
        evaluation.ground_truth_source_at_estimate_times_m,
        truth.source_positions_m[[0, 2]],
    )


def test_evaluation_clips_source_to_reference_time_overlap() -> None:
    result, truth = _result_and_truth()
    partial_truth = GroundTruth(
        microphone_positions_m=truth.microphone_positions_m,
        source_times_s=np.array([0.5, 1.0]),
        source_positions_m=truth.source_positions_m[[1, 2]],
        metadata={},
    )
    evaluation = evaluate_against_ground_truth(result, partial_truth)

    assert evaluation.microphone_rms_error_m < 1e-12
    assert evaluation.source_rms_error_m < 1e-12
    assert np.array_equal(evaluation.source_estimate_indices, np.array([1, 2]))
    assert np.allclose(
        evaluation.aligned_source_positions_m,
        truth.source_positions_m[[1, 2]],
    )
    assert np.allclose(
        evaluation.ground_truth_source_at_estimate_times_m,
        truth.source_positions_m[[1, 2]],
    )
    assert evaluation.source_uncertainty_diagnostic is not None


def test_evaluation_rejects_non_overlapping_source_time_ranges() -> None:
    result, truth = _result_and_truth()
    non_overlapping_truth = GroundTruth(
        microphone_positions_m=truth.microphone_positions_m,
        source_times_s=np.array([2.0, 2.5]),
        source_positions_m=truth.source_positions_m[[0, 1]],
        metadata={},
    )
    with pytest.raises(ValueError, match="do not overlap"):
        evaluate_against_ground_truth(result, non_overlapping_truth)
