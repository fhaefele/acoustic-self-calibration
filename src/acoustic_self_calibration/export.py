from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import evaluate_against_ground_truth, evaluate_scenes, evaluation_to_dict
from .ground_truth import GroundTruth, ground_truth_to_dict, load_ground_truth_json
from .pipeline import AudioCalibrationResult
from .visualization import plot_calibration_comparison, plot_scene_comparison


@dataclass(frozen=True)
class CalibrationOutputPaths:
    """JSON result and multi-panel figure written for one operation."""

    json: Path
    figure: Path


def _optional_list(value: Any) -> Any:
    return None if value is None else value.tolist()


def _output_paths(output_prefix: str | Path) -> CalibrationOutputPaths:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return CalibrationOutputPaths(
        json=prefix.with_suffix(".json"),
        figure=prefix.with_suffix(".png"),
    )


def calibration_result_to_dict(
    result: AudioCalibrationResult,
    *,
    input_wav_path: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    ground_truth: GroundTruth | None = None,
) -> dict[str, Any]:
    """Build a result JSON whose ``scene`` can be reused as reference input."""
    calibration = result.calibration
    document: dict[str, Any] = {
        "schema_version": 1,
        "scene_role": "estimate",
        "scene": {
            "microphones": {
                "positions_m": calibration.microphone_positions.tolist(),
                "std_m": _optional_list(calibration.microphone_position_std_m),
            },
            "source": {
                "times_s": result.frame_times_s.tolist(),
                "positions_m": calibration.source_positions.tolist(),
                "std_m": _optional_list(calibration.source_position_std_m),
            },
        },
        "input": {
            "wav_path": None if input_wav_path is None else str(input_wav_path),
        },
        "settings": {} if settings is None else settings,
        "calibration": {
            "speed_of_sound_mps": calibration.speed_of_sound,
            "speed_of_sound_std_mps": calibration.speed_of_sound_std,
            "clock_offsets_s": calibration.clock_offsets_s.tolist(),
            "clock_offset_std_s": _optional_list(calibration.clock_offset_std_s),
            "clock_drifts": calibration.clock_drifts.tolist(),
            "clock_drift_std": _optional_list(calibration.clock_drift_std),
        },
        "measurements": {
            "microphone_pairs": [list(pair) for pair in result.microphone_pairs],
            "tdoa_s": result.tdoa_s.tolist(),
            "tdoa_sigma_s": result.tdoa_sigma_s.tolist(),
            "confidence": result.confidence.tolist(),
        },
        "diagnostics": {
            "success": calibration.success,
            "message": calibration.message,
            "nfev": calibration.nfev,
            "rms_tdoa_residual_s": calibration.rms_tdoa_residual_s,
            "normalized_data_rms": calibration.normalized_data_rms,
            "negative_log_posterior": calibration.negative_log_posterior,
        },
    }
    if ground_truth is not None:
        evaluation = evaluate_against_ground_truth(result, ground_truth)
        document["reference"] = ground_truth_to_dict(ground_truth)
        document["evaluation"] = evaluation_to_dict(evaluation)
    json.dumps(document, allow_nan=False)
    return document


def write_calibration_outputs(
    result: AudioCalibrationResult,
    output_prefix: str | Path,
    *,
    input_wav_path: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    ground_truth: GroundTruth | str | Path | None = None,
) -> CalibrationOutputPaths:
    """Write the JSON numerical result and one 2x2 PNG visualization."""
    paths = _output_paths(output_prefix)
    if isinstance(ground_truth, (str, Path)):
        resolved_ground_truth = load_ground_truth_json(ground_truth)
    else:
        resolved_ground_truth = ground_truth

    evaluation = (
        None
        if resolved_ground_truth is None
        else evaluate_against_ground_truth(result, resolved_ground_truth)
    )
    document = calibration_result_to_dict(
        result,
        input_wav_path=input_wav_path,
        settings=settings,
        ground_truth=resolved_ground_truth,
    )
    paths.json.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_calibration_comparison(
        result,
        paths.figure,
        ground_truth=resolved_ground_truth,
        evaluation=evaluation,
    )
    return paths


def write_scene_comparison_outputs(
    estimate: GroundTruth,
    reference: GroundTruth,
    output_prefix: str | Path,
    *,
    estimate_path: str | Path | None = None,
    reference_path: str | Path | None = None,
) -> CalibrationOutputPaths:
    """Compare two canonical scenes without recalibrating audio."""
    paths = _output_paths(output_prefix)
    evaluation = evaluate_scenes(estimate, reference)
    document = ground_truth_to_dict(estimate)
    document["comparison"] = {
        "estimate_path": None if estimate_path is None else str(estimate_path),
        "reference_path": None if reference_path is None else str(reference_path),
    }
    document["reference"] = ground_truth_to_dict(reference)
    document["evaluation"] = evaluation_to_dict(evaluation)
    paths.json.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_scene_comparison(
        estimate,
        reference,
        paths.figure,
        evaluation=evaluation,
    )
    return paths
