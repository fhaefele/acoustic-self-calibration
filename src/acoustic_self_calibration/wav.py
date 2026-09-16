from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .bayesian import DistancePrior
from .pipeline import AudioCalibrationResult, calibrate_audio


def _audio_to_float64(audio: np.ndarray) -> np.ndarray:
    """Convert scipy WAV data to finite floating-point samples near [-1, 1]."""
    values = np.asarray(audio)
    if np.issubdtype(values.dtype, np.floating):
        converted = values.astype(np.float64, copy=False)
    elif np.issubdtype(values.dtype, np.signedinteger):
        info = np.iinfo(values.dtype)
        scale = float(max(abs(int(info.min)), int(info.max)))
        converted = values.astype(np.float64) / scale
    elif np.issubdtype(values.dtype, np.unsignedinteger):
        info = np.iinfo(values.dtype)
        midpoint = 0.5 * (float(info.max) + 1.0)
        converted = (values.astype(np.float64) - midpoint) / midpoint
    else:
        raise ValueError(f"Unsupported WAV sample dtype: {values.dtype}")

    if not np.all(np.isfinite(converted)):
        raise ValueError("WAV contains non-finite samples")
    return converted


def read_multichannel_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a multichannel WAV file and normalize integer PCM to float64.

    SciPy represents 24-bit PCM as left-justified int32. Dividing by the int32
    full-scale value therefore also normalizes conventional 24-bit PCM correctly.
    """
    sample_rate, audio = wavfile.read(Path(path))
    samples = _audio_to_float64(np.asarray(audio))
    if samples.ndim != 2:
        raise ValueError("WAV must be multichannel with shape (samples, microphones)")
    if samples.shape[1] < 4:
        raise ValueError("At least four WAV channels are required for 3-D calibration")
    return int(sample_rate), samples


def calibrate_wav(
    path: str | Path,
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
    """Calibrate microphone geometry and moving-source trajectory from a WAV file."""
    sample_rate, audio = read_multichannel_wav(path)
    return calibrate_audio(
        audio,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        max_tau_s=max_tau_s,
        gcc_interp=gcc_interp,
        pair_mode=pair_mode,
        reference_count=reference_count,
        microphone_pairs=microphone_pairs,
        speed_of_sound=speed_of_sound,
        motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
        likelihood=likelihood,
        estimate_clock_offsets=estimate_clock_offsets,
        estimate_clock_drifts=estimate_clock_drifts,
        estimate_speed_of_sound=estimate_speed_of_sound,
        distance_priors=distance_priors,
        best_sigma_samples=best_sigma_samples,
        worst_sigma_samples=worst_sigma_samples,
        max_nfev=max_nfev,
        compute_laplace_uncertainty=compute_laplace_uncertainty,
    )
