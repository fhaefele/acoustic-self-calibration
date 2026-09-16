import numpy as np
import pytest

from acoustic_self_calibration import calibrate_audio
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.simulation import random_microphone_array, render_moving_source


def _broadband_signal(samples: int, rng: np.random.Generator) -> np.ndarray:
    signal = rng.normal(size=samples)
    signal = np.concatenate([[0.0], np.diff(signal)])
    return signal / np.max(np.abs(signal))


def _cardioid_scene(microphone_count: int):
    rng = np.random.default_rng(30 + microphone_count)
    sample_rate = 16_000
    duration = 8.0
    microphones = random_microphone_array(
        microphone_count,
        bounds=((-1.5, 1.5), (-1.5, 1.5), (0.0, 2.2)),
        rng=rng,
    )
    key_times = np.linspace(0.0, duration, 21)
    phase = np.linspace(0.0, 2.0 * np.pi, len(key_times))
    source_positions = np.column_stack(
        [
            2.0 * np.cos(phase),
            1.5 * np.sin(phase),
            1.2 + 0.7 * np.sin(0.8 * phase + 0.2),
        ]
    )
    source_forward = np.array([0.0, 0.0, 1.1])[None, :] - source_positions
    source_forward /= np.linalg.norm(source_forward, axis=1, keepdims=True)
    audio = render_moving_source(
        _broadband_signal(int(sample_rate * duration), rng),
        sample_rate,
        microphones,
        key_times,
        source_positions,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=1e-5,
        rng=rng,
    )
    return sample_rate, microphones, key_times, source_positions, audio


@pytest.mark.parametrize("microphone_count", [8, 16, 24])
def test_general_audio_pipeline_self_calibrates_rendered_cardioid_source(microphone_count):
    sample_rate, microphones, key_times, source_positions, audio = _cardioid_scene(microphone_count)

    result = calibrate_audio(
        audio,
        sample_rate,
        frame_size=512,
        hop_size=8192,
        max_tau_s=0.025,
        gcc_interp=16,
        pair_mode="redundant",
        reference_count=2,
        motion_velocity_change_sigma_mps=3.0,
        likelihood="cauchy",
        max_nfev=300,
        compute_laplace_uncertainty=False,
    )

    true_source = np.column_stack(
        [
            np.interp(result.frame_times_s, key_times, source_positions[:, dimension])
            for dimension in range(3)
        ]
    )
    aligned_microphones, rotation, translation = rigid_align(
        result.calibration.microphone_positions,
        microphones,
    )
    aligned_source = apply_rigid(result.calibration.source_positions, rotation, translation)

    assert result.calibration.success
    assert result.tdoa_sigma_s.shape == result.tdoa_s.shape
    assert np.all(result.tdoa_sigma_s > 0.0)
    assert rms_position_error(aligned_microphones, microphones) < 0.08
    assert rms_position_error(aligned_source, true_source) < 0.08
    assert result.calibration.rms_tdoa_residual_s < 30e-6
