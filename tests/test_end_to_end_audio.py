from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration import calibrate_audio, calibrate_wav
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.simulation import (
    broadband_pulse_train,
    random_microphone_array,
    render_moving_source,
)


def _pulsed_scene(microphone_count: int):
    rng = np.random.default_rng(100 + microphone_count)
    sample_rate = 48_000
    duration = 5.0
    microphones = random_microphone_array(
        microphone_count,
        bounds=((-1.2, 1.2), (-1.2, 1.2), (0.0, 1.6)),
        rng=rng,
    )

    trajectory_times = np.linspace(0.0, duration, 31)
    phase = np.linspace(0.0, 2.0 * np.pi, len(trajectory_times))
    trajectory_positions = np.column_stack(
        [
            2.0 * np.cos(0.75 * phase),
            1.6 * np.sin(0.75 * phase),
            1.2 + 0.5 * np.sin(0.55 * phase + 0.2),
        ]
    )
    event_times = np.linspace(0.4, 4.4, 20)
    source_signal = broadband_pulse_train(
        event_times,
        sample_rate,
        duration,
        pulse_duration_s=0.0015,
        rng=rng,
    )
    source_forward = microphones.mean(axis=0)[None, :] - trajectory_positions
    source_forward /= np.linalg.norm(source_forward, axis=1, keepdims=True)
    audio = render_moving_source(
        source_signal,
        sample_rate,
        microphones,
        trajectory_times,
        trajectory_positions,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=1e-5,
        rng=rng,
    )
    source_at_events = np.column_stack(
        [
            np.interp(event_times, trajectory_times, trajectory_positions[:, dimension])
            for dimension in range(3)
        ]
    )
    return sample_rate, microphones, event_times, source_at_events, audio


@pytest.mark.parametrize("microphone_count", [8, 16, 24])
def test_event_pipeline_self_calibrates_rendered_pulsed_source(microphone_count: int) -> None:
    sample_rate, microphones, event_times, true_source, audio = _pulsed_scene(microphone_count)

    result = calibrate_audio(
        audio,
        sample_rate,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        pair_mode="reference",
        motion_velocity_change_sigma_mps=5.0,
        likelihood="cauchy",
        max_nfev=400,
        compute_laplace_uncertainty=False,
    )

    aligned_microphones, rotation, translation = rigid_align(
        result.calibration.microphone_positions,
        microphones,
    )
    aligned_source = apply_rigid(result.calibration.source_positions, rotation, translation)

    assert result.calibration.success
    assert len(result.event_times_s) == len(event_times)
    assert result.detected_event_count == len(event_times)
    assert result.tdoa_sigma_s.shape == result.tdoa_s.shape
    assert np.all(result.tdoa_sigma_s > 0.0)
    assert rms_position_error(aligned_microphones, microphones) < 0.15
    assert rms_position_error(aligned_source, true_source) < 0.18
    assert result.calibration.rms_tdoa_residual_s < 60e-6


def test_pcm16_pulsed_wav_pipeline_returns_joint_position_uncertainty(tmp_path: Path) -> None:
    sample_rate, microphones, event_times, true_source, audio = _pulsed_scene(8)
    path = tmp_path / "moving_pulses.wav"
    scaled = audio / max(float(np.max(np.abs(audio))), 1e-12)
    wavfile.write(path, sample_rate, np.round(0.95 * scaled * 32767.0).astype(np.int16))

    result = calibrate_wav(
        path,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        pair_mode="reference",
        motion_velocity_change_sigma_mps=5.0,
        likelihood="cauchy",
        max_nfev=400,
        compute_laplace_uncertainty=True,
    )

    aligned_microphones, rotation, translation = rigid_align(
        result.calibration.microphone_positions,
        microphones,
    )
    aligned_source = apply_rigid(result.calibration.source_positions, rotation, translation)

    assert result.calibration.success
    assert len(result.event_times_s) == len(event_times)
    assert rms_position_error(aligned_microphones, microphones) < 0.18
    assert rms_position_error(aligned_source, true_source) < 0.22
    assert result.calibration.microphone_position_std_m is not None
    assert result.calibration.source_position_std_m is not None
    assert (
        result.calibration.source_position_std_m.shape == result.calibration.source_positions.shape
    )
    assert np.isfinite(result.calibration.source_position_std_m).all()
