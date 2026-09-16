from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import AudioCalibrationResult


@dataclass(frozen=True)
class CalibrationExportPaths:
    """Files written by :func:`export_calibration`."""

    npz: Path
    json: Path
    microphones_csv: Path
    trajectory_csv: Path


def _optional_array(value: np.ndarray | None, columns: int | None = None) -> np.ndarray:
    if value is not None:
        return np.asarray(value)
    if columns is None:
        return np.empty(0, dtype=float)
    return np.empty((0, columns), dtype=float)


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def export_calibration(
    result: AudioCalibrationResult,
    output_prefix: str | Path,
) -> CalibrationExportPaths:
    """Export one calibration to NPZ, JSON, microphone CSV, and trajectory CSV."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix.suffix:
        prefix = prefix.with_suffix("")

    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    microphones_csv = prefix.with_name(f"{prefix.name}_microphones.csv")
    trajectory_csv = prefix.with_name(f"{prefix.name}_trajectory.csv")

    calibration = result.calibration
    np.savez_compressed(
        npz_path,
        frame_times_s=result.frame_times_s,
        microphone_positions_m=calibration.microphone_positions,
        source_positions_m=calibration.source_positions,
        microphone_position_std_m=_optional_array(calibration.microphone_position_std_m, 3),
        source_position_std_m=_optional_array(calibration.source_position_std_m, 3),
        speed_of_sound_mps=np.array(calibration.speed_of_sound),
        speed_of_sound_std_mps=np.array(_optional_float(calibration.speed_of_sound_std)),
        clock_offsets_s=calibration.clock_offsets_s,
        clock_offset_std_s=_optional_array(calibration.clock_offset_std_s),
        clock_drifts=calibration.clock_drifts,
        clock_drift_std=_optional_array(calibration.clock_drift_std),
        tdoa_s=result.tdoa_s,
        tdoa_sigma_s=result.tdoa_sigma_s,
        confidence=result.confidence,
        microphone_pairs=np.asarray(result.microphone_pairs, dtype=np.int64),
        rms_tdoa_residual_s=np.array(calibration.rms_tdoa_residual_s),
        normalized_data_rms=np.array(calibration.normalized_data_rms),
        negative_log_posterior=np.array(calibration.negative_log_posterior),
        success=np.array(calibration.success),
    )

    metadata = {
        "success": calibration.success,
        "message": calibration.message,
        "nfev": calibration.nfev,
        "rms_tdoa_residual_s": calibration.rms_tdoa_residual_s,
        "normalized_data_rms": calibration.normalized_data_rms,
        "negative_log_posterior": calibration.negative_log_posterior,
        "speed_of_sound_mps": calibration.speed_of_sound,
        "speed_of_sound_std_mps": calibration.speed_of_sound_std,
        "clock_offsets_s": calibration.clock_offsets_s.tolist(),
        "clock_offset_std_s": (
            None
            if calibration.clock_offset_std_s is None
            else calibration.clock_offset_std_s.tolist()
        ),
        "clock_drifts": calibration.clock_drifts.tolist(),
        "clock_drift_std": (
            None if calibration.clock_drift_std is None else calibration.clock_drift_std.tolist()
        ),
        "microphone_positions_m": calibration.microphone_positions.tolist(),
        "microphone_position_std_m": (
            None
            if calibration.microphone_position_std_m is None
            else calibration.microphone_position_std_m.tolist()
        ),
        "frame_times_s": result.frame_times_s.tolist(),
        "source_positions_m": calibration.source_positions.tolist(),
        "source_position_std_m": (
            None
            if calibration.source_position_std_m is None
            else calibration.source_position_std_m.tolist()
        ),
        "microphone_pairs": [list(pair) for pair in result.microphone_pairs],
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    mic_std = calibration.microphone_position_std_m
    with microphones_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["microphone", "x_m", "y_m", "z_m", "std_x_m", "std_y_m", "std_z_m"])
        for index, position in enumerate(calibration.microphone_positions):
            uncertainty = ("", "", "") if mic_std is None else tuple(mic_std[index])
            writer.writerow([index, *position, *uncertainty])

    source_std = calibration.source_position_std_m
    with trajectory_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "x_m", "y_m", "z_m", "std_x_m", "std_y_m", "std_z_m"])
        for index, (time_s, position) in enumerate(
            zip(result.frame_times_s, calibration.source_positions, strict=True)
        ):
            uncertainty = ("", "", "") if source_std is None else tuple(source_std[index])
            writer.writerow([time_s, *position, *uncertainty])

    return CalibrationExportPaths(
        npz=npz_path,
        json=json_path,
        microphones_csv=microphones_csv,
        trajectory_csv=trajectory_csv,
    )
