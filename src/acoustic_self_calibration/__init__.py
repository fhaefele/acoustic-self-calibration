"""Bayesian 3-D acoustic self-calibration for a moving broadband source."""

from .bayesian import (
    BayesianCalibrationResult,
    DistancePrior,
    calibrate_bayesian,
    tdoa_sigma_from_confidence,
)
from .evaluation import GroundTruthEvaluation, evaluate_against_ground_truth, evaluate_scenes
from .export import (
    CalibrationOutputPaths,
    calibration_result_to_dict,
    write_calibration_outputs,
    write_scene_comparison_outputs,
)
from .ground_truth import (
    GroundTruth,
    ground_truth_from_dict,
    ground_truth_to_dict,
    load_ground_truth_json,
    make_ground_truth_dict,
    validate_ground_truth_json,
    write_ground_truth_json,
)
from .pipeline import AudioCalibrationResult, calibrate_audio
from .simulation import apply_channel_clock_model, random_microphone_array, render_moving_source
from .tdoa import estimate_pairwise_tdoa_matrix, gcc_phat, make_microphone_pairs
from .visualization import plot_calibration_comparison, plot_scene_comparison
from .wav import calibrate_wav, read_multichannel_wav

__all__ = [
    "AudioCalibrationResult",
    "BayesianCalibrationResult",
    "CalibrationOutputPaths",
    "DistancePrior",
    "GroundTruth",
    "GroundTruthEvaluation",
    "apply_channel_clock_model",
    "calibrate_audio",
    "calibrate_bayesian",
    "calibrate_wav",
    "calibration_result_to_dict",
    "estimate_pairwise_tdoa_matrix",
    "evaluate_against_ground_truth",
    "evaluate_scenes",
    "gcc_phat",
    "ground_truth_from_dict",
    "ground_truth_to_dict",
    "load_ground_truth_json",
    "make_ground_truth_dict",
    "make_microphone_pairs",
    "plot_calibration_comparison",
    "plot_scene_comparison",
    "random_microphone_array",
    "read_multichannel_wav",
    "render_moving_source",
    "tdoa_sigma_from_confidence",
    "validate_ground_truth_json",
    "write_calibration_outputs",
    "write_ground_truth_json",
    "write_scene_comparison_outputs",
]
