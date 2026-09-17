from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import apply_rigid, rigid_align
from .ground_truth import GroundTruth
from .pipeline import AudioCalibrationResult


@dataclass(frozen=True)
class GroundTruthEvaluation:
    """Aligned estimate and quantitative comparison against ground truth."""

    aligned_microphone_positions_m: np.ndarray
    aligned_source_positions_m: np.ndarray
    ground_truth_source_at_estimate_times_m: np.ndarray
    microphone_error_m: np.ndarray
    source_error_m: np.ndarray
    rotation: np.ndarray
    translation_m: np.ndarray
    reflection_applied: bool
    microphone_rms_error_m: float
    microphone_mean_error_m: float
    microphone_max_error_m: float
    source_rms_error_m: float
    source_mean_error_m: float
    source_max_error_m: float
    microphone_uncertainty_diagnostic: dict[str, float] | None
    source_uncertainty_diagnostic: dict[str, float] | None


def _summary(error_m: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(error_m, dtype=float)
    return (
        float(np.sqrt(np.mean(values * values))),
        float(np.mean(values)),
        float(np.max(values)),
    )


def _interpolate_source_ground_truth(ground_truth: GroundTruth, times_s: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    tolerance = 1e-9
    if times[0] < ground_truth.source_times_s[0] - tolerance:
        raise ValueError("estimated source times start before ground-truth source coverage")
    if times[-1] > ground_truth.source_times_s[-1] + tolerance:
        raise ValueError("estimated source times extend beyond ground-truth source coverage")
    return np.column_stack(
        [
            np.interp(
                times,
                ground_truth.source_times_s,
                ground_truth.source_positions_m[:, dimension],
            )
            for dimension in range(3)
        ]
    )


def _uncertainty_diagnostic(
    error_m: np.ndarray,
    position_std_m: np.ndarray | None,
) -> dict[str, float] | None:
    if position_std_m is None:
        return None
    std = np.asarray(position_std_m, dtype=float)
    radial_std = np.sqrt(np.sum(std * std, axis=1))
    valid = np.isfinite(radial_std) & (radial_std > 0.0)
    if not np.any(valid):
        return None
    ratio = np.asarray(error_m, dtype=float)[valid] / radial_std[valid]
    return {
        "mean_error_over_rss_std": float(np.mean(ratio)),
        "median_error_over_rss_std": float(np.median(ratio)),
        "fraction_within_1x_rss_std": float(np.mean(ratio <= 1.0)),
        "fraction_within_2x_rss_std": float(np.mean(ratio <= 2.0)),
        "fraction_within_3x_rss_std": float(np.mean(ratio <= 3.0)),
    }


def evaluate_against_ground_truth(
    result: AudioCalibrationResult,
    ground_truth: GroundTruth,
) -> GroundTruthEvaluation:
    """Align the recovered scene using microphones and compare it against ground truth.

    A single orthogonal transform is fitted from estimated microphones to ground-truth
    microphones. The same transform is then applied to every recovered source state.
    The source trajectory is never independently aligned.
    """
    calibration = result.calibration
    if calibration.microphone_positions.shape != ground_truth.microphone_positions_m.shape:
        raise ValueError(
            "ground-truth microphone count must exactly match the calibrated microphone count"
        )

    aligned_microphones, rotation, translation = rigid_align(
        calibration.microphone_positions,
        ground_truth.microphone_positions_m,
        allow_reflection=True,
    )
    aligned_source = apply_rigid(calibration.source_positions, rotation, translation)
    source_ground_truth = _interpolate_source_ground_truth(ground_truth, result.frame_times_s)

    microphone_error = np.linalg.norm(
        aligned_microphones - ground_truth.microphone_positions_m,
        axis=1,
    )
    source_error = np.linalg.norm(aligned_source - source_ground_truth, axis=1)

    mic_rms, mic_mean, mic_max = _summary(microphone_error)
    src_rms, src_mean, src_max = _summary(source_error)

    return GroundTruthEvaluation(
        aligned_microphone_positions_m=aligned_microphones,
        aligned_source_positions_m=aligned_source,
        ground_truth_source_at_estimate_times_m=source_ground_truth,
        microphone_error_m=microphone_error,
        source_error_m=source_error,
        rotation=rotation,
        translation_m=translation,
        reflection_applied=bool(np.linalg.det(rotation) < 0.0),
        microphone_rms_error_m=mic_rms,
        microphone_mean_error_m=mic_mean,
        microphone_max_error_m=mic_max,
        source_rms_error_m=src_rms,
        source_mean_error_m=src_mean,
        source_max_error_m=src_max,
        microphone_uncertainty_diagnostic=_uncertainty_diagnostic(
            microphone_error,
            calibration.microphone_position_std_m,
        ),
        source_uncertainty_diagnostic=_uncertainty_diagnostic(
            source_error,
            calibration.source_position_std_m,
        ),
    )


def evaluation_to_dict(evaluation: GroundTruthEvaluation) -> dict[str, Any]:
    """Convert evaluation results into JSON-serializable data."""
    return {
        "alignment": {
            "rotation": evaluation.rotation.tolist(),
            "translation_m": evaluation.translation_m.tolist(),
            "reflection_applied": evaluation.reflection_applied,
            "fitted_from": "microphones_only",
        },
        "microphones": {
            "aligned_estimate_m": evaluation.aligned_microphone_positions_m.tolist(),
            "error_m": evaluation.microphone_error_m.tolist(),
            "rms_error_m": evaluation.microphone_rms_error_m,
            "mean_error_m": evaluation.microphone_mean_error_m,
            "max_error_m": evaluation.microphone_max_error_m,
            "uncertainty_diagnostic": evaluation.microphone_uncertainty_diagnostic,
        },
        "source": {
            "aligned_estimate_m": evaluation.aligned_source_positions_m.tolist(),
            "ground_truth_at_estimate_times_m": evaluation.ground_truth_source_at_estimate_times_m.tolist(),
            "error_m": evaluation.source_error_m.tolist(),
            "rms_error_m": evaluation.source_rms_error_m,
            "mean_error_m": evaluation.source_mean_error_m,
            "max_error_m": evaluation.source_max_error_m,
            "uncertainty_diagnostic": evaluation.source_uncertainty_diagnostic,
        },
    }
