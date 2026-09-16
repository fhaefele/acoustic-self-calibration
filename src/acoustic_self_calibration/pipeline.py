from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bayesian import (
    BayesianCalibrationResult,
    DistancePrior,
    calibrate_bayesian,
    tdoa_sigma_from_confidence,
)
from .tdoa import estimate_pairwise_tdoa_matrix, make_microphone_pairs


@dataclass(frozen=True)
class AudioCalibrationResult:
    """End-to-end result plus the TDOA measurements used by the MAP solver."""

    calibration: BayesianCalibrationResult
    frame_times_s: np.ndarray
    tdoa_s: np.ndarray
    tdoa_sigma_s: np.ndarray
    confidence: np.ndarray
    microphone_pairs: tuple[tuple[int, int], ...]


def calibrate_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 1024,
    hop_size: int = 4096,
    max_tau_s: float | None = 0.03,
    gcc_interp: int = 16,
    pair_mode: str = "redundant",
    reference_count: int = 2,
    microphone_pairs: Sequence[tuple[int, int]] | None = None,
    speed_of_sound: float = 343.0,
    motion_velocity_change_sigma_mps: float | None = 3.0,
    likelihood: str = "cauchy",
    estimate_clock_offsets: bool = False,
    estimate_clock_drifts: bool = False,
    estimate_speed_of_sound: bool = False,
    distance_priors: Sequence[DistancePrior] = (),
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    max_nfev: int = 4000,
    compute_laplace_uncertainty: bool = True,
) -> AudioCalibrationResult:
    """Self-calibrate directly from synchronized multichannel broadband audio.

    No source waveform or emission timestamps are required. The source must provide
    enough broadband content for GCC-PHAT and must move through a non-degenerate 3-D
    trajectory during the recording.

    The frontend estimates redundant pairwise TDOAs frame by frame. Relative GCC
    peak confidence is converted to heteroscedastic timing uncertainty, and those
    measurements feed the Bayesian/MAP factor graph.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 2:
        raise ValueError("audio must have shape (samples, microphones)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    microphone_count = x.shape[1]
    if microphone_count < 4:
        raise ValueError("At least four microphones are required for 3-D calibration")

    if microphone_pairs is None:
        pairs = tuple(
            make_microphone_pairs(
                microphone_count,
                mode=pair_mode,
                reference_count=reference_count,
            )
        )
    else:
        pairs = tuple((int(a), int(b)) for a, b in microphone_pairs)

    frame_times, tdoa, confidence, pairs = estimate_pairwise_tdoa_matrix(
        x,
        sample_rate,
        microphone_pairs=pairs,
        frame_size=frame_size,
        hop_size=hop_size,
        max_tau=max_tau_s,
        interp=gcc_interp,
    )
    sigma = tdoa_sigma_from_confidence(
        confidence,
        sample_rate,
        best_sigma_samples=best_sigma_samples,
        worst_sigma_samples=worst_sigma_samples,
    )

    calibration = calibrate_bayesian(
        tdoa,
        frame_times,
        microphone_count,
        tdoa_sigma_s=sigma,
        microphone_pairs=pairs,
        speed_of_sound=speed_of_sound,
        estimate_speed_of_sound=estimate_speed_of_sound,
        estimate_clock_offsets=estimate_clock_offsets,
        estimate_clock_drifts=estimate_clock_drifts,
        distance_priors=distance_priors,
        motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
        likelihood=likelihood,
        max_nfev=max_nfev,
        compute_laplace_uncertainty=compute_laplace_uncertainty,
    )

    return AudioCalibrationResult(
        calibration=calibration,
        frame_times_s=frame_times,
        tdoa_s=tdoa,
        tdoa_sigma_s=sigma,
        confidence=confidence,
        microphone_pairs=pairs,
    )
