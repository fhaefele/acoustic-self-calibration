import json
from pathlib import Path

import numpy as np

from acoustic_self_calibration.bayesian import BayesianCalibrationResult
from acoustic_self_calibration.export import write_calibration_outputs
from acoustic_self_calibration.ground_truth import GroundTruth, validate_ground_truth_json
from acoustic_self_calibration.pipeline import AudioCalibrationResult


def _result() -> AudioCalibrationResult:
    calibration = BayesianCalibrationResult(
        microphone_positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.3, 1.2]]
        ),
        source_positions=np.array([[0.2, 0.1, 0.8], [0.4, 0.2, 0.9]]),
        speed_of_sound=343.0,
        clock_offsets_s=np.zeros(4),
        clock_drifts=np.zeros(4),
        microphone_position_std_m=np.full((4, 3), 0.01),
        source_position_std_m=np.full((2, 3), 0.02),
        speed_of_sound_std=0.5,
        clock_offset_std_s=np.full(4, 1e-6),
        clock_drift_std=np.full(4, 1e-7),
        rms_tdoa_residual_s=2e-6,
        normalized_data_rms=0.8,
        negative_log_posterior=12.0,
        success=True,
        message="ok",
        nfev=42,
    )
    return AudioCalibrationResult(
        calibration=calibration,
        frame_times_s=np.array([0.1, 0.2]),
        tdoa_s=np.zeros((2, 3)),
        tdoa_sigma_s=np.full((2, 3), 2e-6),
        confidence=np.ones((2, 3)),
        microphone_pairs=((0, 1), (0, 2), (0, 3)),
    )


def test_write_calibration_outputs_writes_only_json_and_png(tmp_path: Path) -> None:
    paths = write_calibration_outputs(
        _result(),
        tmp_path / "calibration",
        input_wav_path="recording.wav",
        settings={"frame_size": 1024},
    )

    assert paths.json.exists()
    assert paths.figure.exists()
    assert paths.figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "calibration.json",
        "calibration.png",
    ]

    document = json.loads(paths.json.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["scene_role"] == "estimate"
    assert document["input"]["wav_path"] == "recording.wav"
    assert document["settings"]["frame_size"] == 1024
    assert len(document["scene"]["microphones"]["positions_m"]) == 4
    assert len(document["scene"]["source"]["std_m"]) == 2
    assert "evaluation" not in document


def test_result_json_is_directly_valid_as_reference_input(tmp_path: Path) -> None:
    result = _result()
    first_paths = write_calibration_outputs(result, tmp_path / "first")

    reference = validate_ground_truth_json(first_paths.json)
    assert reference.scene_role == "estimate"
    assert np.allclose(reference.microphone_positions_m, result.calibration.microphone_positions)
    assert np.allclose(reference.source_positions_m, result.calibration.source_positions)

    second_paths = write_calibration_outputs(
        result,
        tmp_path / "second",
        ground_truth=first_paths.json,
    )
    document = json.loads(second_paths.json.read_text(encoding="utf-8"))
    assert document["reference"]["scene_role"] == "estimate"
    assert document["evaluation"]["alignment"]["fitted_from"] == "microphones_only"
    assert document["evaluation"]["microphones"]["rms_error_m"] < 1e-12
    assert document["evaluation"]["source"]["rms_error_m"] < 1e-12


def test_write_calibration_outputs_adds_ground_truth_evaluation(tmp_path: Path) -> None:
    result = _result()
    ground_truth = GroundTruth(
        microphone_positions_m=result.calibration.microphone_positions.copy(),
        source_times_s=result.frame_times_s.copy(),
        source_positions_m=result.calibration.source_positions.copy(),
        metadata={"name": "exact"},
    )
    paths = write_calibration_outputs(
        result,
        tmp_path / "with_gt.json",
        ground_truth=ground_truth,
    )
    document = json.loads(paths.json.read_text(encoding="utf-8"))
    assert document["reference"]["metadata"]["name"] == "exact"
    assert document["reference"]["scene_role"] == "ground_truth"
    assert document["evaluation"]["alignment"]["fitted_from"] == "microphones_only"
    assert document["evaluation"]["microphones"]["rms_error_m"] < 1e-12
    assert document["evaluation"]["source"]["rms_error_m"] < 1e-12


def test_write_calibration_outputs_uses_json_null_for_missing_uncertainty(tmp_path: Path) -> None:
    result = _result()
    calibration = result.calibration
    without_uncertainty = BayesianCalibrationResult(
        microphone_positions=calibration.microphone_positions,
        source_positions=calibration.source_positions,
        speed_of_sound=calibration.speed_of_sound,
        clock_offsets_s=calibration.clock_offsets_s,
        clock_drifts=calibration.clock_drifts,
        microphone_position_std_m=None,
        source_position_std_m=None,
        speed_of_sound_std=None,
        clock_offset_std_s=None,
        clock_drift_std=None,
        rms_tdoa_residual_s=calibration.rms_tdoa_residual_s,
        normalized_data_rms=calibration.normalized_data_rms,
        negative_log_posterior=calibration.negative_log_posterior,
        success=True,
        message="ok",
        nfev=1,
    )
    result = AudioCalibrationResult(
        calibration=without_uncertainty,
        frame_times_s=result.frame_times_s,
        tdoa_s=result.tdoa_s,
        tdoa_sigma_s=result.tdoa_sigma_s,
        confidence=result.confidence,
        microphone_pairs=result.microphone_pairs,
    )
    paths = write_calibration_outputs(result, tmp_path / "without_uncertainty")
    document = json.loads(paths.json.read_text(encoding="utf-8"))
    assert document["scene"]["microphones"]["std_m"] is None
    assert document["scene"]["source"]["std_m"] is None
