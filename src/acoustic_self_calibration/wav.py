from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .pipeline import AudioCalibrationResult, AudioModel, RefinementMode, calibrate_audio
from .stratified.constraints import PlanarAngleConstraint


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
    """Read a multichannel WAV file and normalize integer PCM to float64."""
    sample_rate, audio = wavfile.read(Path(path))
    samples = _audio_to_float64(np.asarray(audio))
    if samples.ndim != 2:
        raise ValueError("WAV must be multichannel with shape (samples, microphones)")
    if samples.shape[1] < 4:
        raise ValueError("At least four WAV channels are required")
    return int(sample_rate), samples


def calibrate_wav(
    path: str | Path,
    *,
    event_channel: int | None = None,
    event_smooth_s: float = 0.0003,
    event_min_gap_s: float = 0.003,
    event_relative_prominence: float = 0.003,
    max_tau_s: float = 0.01,
    tdoa_envelope_smooth_s: float = 0.00008,
    tdoa_template_s: float = 0.0018,
    tdoa_candidate_count: int = 8,
    max_tdoa_rate: float = 0.05,
    tdoa_track_weight: float = 0.4,
    use_temporal_tracking: bool = True,
    speed_of_sound: float = 343.0,
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    root_start_count: int = 32,
    metric_start_count: int = 20,
    extra_microphone_inlier_rms_m: float = 0.05,
    planar_membership_tolerance: float = 5e-3,
    planar_metric_acceptance_rms_m: float = 5e-3,
    planar_extra_microphone_rms_m: float = 0.02,
    angle_constraint: PlanarAngleConstraint | None = None,
    model: AudioModel = "general_3d",
    refinement: RefinementMode = "none",
    refinement_max_nfev: int = 200,
    refinement_improvement_tolerance: float = 1e-10,
) -> AudioCalibrationResult:
    """Calibrate microphone geometry and event sources from a multichannel WAV file."""
    sample_rate, audio = read_multichannel_wav(path)
    return calibrate_audio(
        audio,
        sample_rate,
        event_channel=event_channel,
        event_smooth_s=event_smooth_s,
        event_min_gap_s=event_min_gap_s,
        event_relative_prominence=event_relative_prominence,
        max_tau_s=max_tau_s,
        tdoa_envelope_smooth_s=tdoa_envelope_smooth_s,
        tdoa_template_s=tdoa_template_s,
        tdoa_candidate_count=tdoa_candidate_count,
        max_tdoa_rate=max_tdoa_rate,
        tdoa_track_weight=tdoa_track_weight,
        use_temporal_tracking=use_temporal_tracking,
        speed_of_sound=speed_of_sound,
        best_sigma_samples=best_sigma_samples,
        worst_sigma_samples=worst_sigma_samples,
        receiver_subset_budget=receiver_subset_budget,
        event_subset_budget=event_subset_budget,
        root_start_count=root_start_count,
        metric_start_count=metric_start_count,
        extra_microphone_inlier_rms_m=extra_microphone_inlier_rms_m,
        planar_membership_tolerance=planar_membership_tolerance,
        planar_metric_acceptance_rms_m=planar_metric_acceptance_rms_m,
        planar_extra_microphone_rms_m=planar_extra_microphone_rms_m,
        angle_constraint=angle_constraint,
        model=model,
        refinement=refinement,
        refinement_max_nfev=refinement_max_nfev,
        refinement_improvement_tolerance=refinement_improvement_tolerance,
    )
