"""Bayesian 3-D acoustic self-calibration for a moving broadband source."""

from .bayesian import (
    BayesianCalibrationResult,
    DistancePrior,
    calibrate_bayesian,
    tdoa_sigma_from_confidence,
)
from .export import CalibrationExportPaths, export_calibration
from .pipeline import AudioCalibrationResult, calibrate_audio
from .simulation import apply_channel_clock_model, random_microphone_array, render_moving_source
from .tdoa import estimate_pairwise_tdoa_matrix, gcc_phat, make_microphone_pairs
from .wav import calibrate_wav, read_multichannel_wav

__all__ = [
    "AudioCalibrationResult",
    "BayesianCalibrationResult",
    "CalibrationExportPaths",
    "DistancePrior",
    "apply_channel_clock_model",
    "calibrate_audio",
    "calibrate_bayesian",
    "calibrate_wav",
    "estimate_pairwise_tdoa_matrix",
    "export_calibration",
    "gcc_phat",
    "make_microphone_pairs",
    "random_microphone_array",
    "read_multichannel_wav",
    "render_moving_source",
    "tdoa_sigma_from_confidence",
]
