import csv
import json
from pathlib import Path

import numpy as np

from acoustic_self_calibration.bayesian import BayesianCalibrationResult
from acoustic_self_calibration.export import export_calibration
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


def test_export_calibration_writes_all_formats(tmp_path: Path) -> None:
    paths = export_calibration(_result(), tmp_path / "calibration")

    assert paths.npz.exists()
    assert paths.json.exists()
    assert paths.microphones_csv.exists()
    assert paths.trajectory_csv.exists()

    with np.load(paths.npz) as archive:
        assert archive["microphone_positions_m"].shape == (4, 3)
        assert archive["source_positions_m"].shape == (2, 3)
        assert archive["microphone_position_std_m"].shape == (4, 3)
        assert archive["source_position_std_m"].shape == (2, 3)

    metadata = json.loads(paths.json.read_text(encoding="utf-8"))
    assert metadata["success"] is True
    assert len(metadata["source_position_std_m"]) == 2

    with paths.microphones_csv.open(newline="", encoding="utf-8") as handle:
        mic_rows = list(csv.DictReader(handle))
    assert len(mic_rows) == 4
    assert float(mic_rows[1]["std_x_m"]) == 0.01

    with paths.trajectory_csv.open(newline="", encoding="utf-8") as handle:
        trajectory_rows = list(csv.DictReader(handle))
    assert len(trajectory_rows) == 2
    assert float(trajectory_rows[0]["std_z_m"]) == 0.02


def test_export_calibration_represents_missing_uncertainty(tmp_path: Path) -> None:
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
    paths = export_calibration(result, tmp_path / "without_uncertainty.npz")
    metadata = json.loads(paths.json.read_text(encoding="utf-8"))
    assert metadata["microphone_position_std_m"] is None
    assert metadata["source_position_std_m"] is None
    with np.load(paths.npz) as archive:
        assert archive["source_position_std_m"].shape == (0, 3)
