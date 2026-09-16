"""Minimal end-to-end self-calibration from arbitrary broadband audio."""

from __future__ import annotations

import numpy as np

from acoustic_self_calibration import calibrate_audio, random_microphone_array, render_moving_source
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error


def broadband_signal(samples: int, rng: np.random.Generator) -> np.ndarray:
    signal = rng.normal(size=samples)
    signal = np.concatenate([[0.0], np.diff(signal)])
    return signal / np.max(np.abs(signal))


def main() -> None:
    rng = np.random.default_rng(46)
    sample_rate = 16_000
    duration = 8.0
    microphone_count = 16

    microphones_true = random_microphone_array(
        microphone_count,
        bounds=((-1.5, 1.5), (-1.5, 1.5), (0.0, 2.2)),
        rng=rng,
    )
    key_times = np.linspace(0.0, duration, 21)
    phase = np.linspace(0.0, 2.0 * np.pi, len(key_times))
    source_keys = np.column_stack(
        [
            2.0 * np.cos(phase),
            1.5 * np.sin(phase),
            1.2 + 0.7 * np.sin(0.8 * phase + 0.2),
        ]
    )
    source_forward = np.array([0.0, 0.0, 1.1])[None, :] - source_keys
    source_forward /= np.linalg.norm(source_forward, axis=1, keepdims=True)

    audio = render_moving_source(
        broadband_signal(int(sample_rate * duration), rng),
        sample_rate,
        microphones_true,
        key_times,
        source_keys,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=1e-5,
        rng=rng,
    )

    result = calibrate_audio(
        audio,
        sample_rate,
        frame_size=512,
        hop_size=8192,
        max_tau_s=0.025,
        gcc_interp=16,
        reference_count=2,
        motion_velocity_change_sigma_mps=3.0,
        max_nfev=300,
    )

    source_true = np.column_stack(
        [
            np.interp(result.frame_times_s, key_times, source_keys[:, dimension])
            for dimension in range(3)
        ]
    )
    aligned_microphones, rotation, translation = rigid_align(
        result.calibration.microphone_positions,
        microphones_true,
    )
    aligned_source = apply_rigid(result.calibration.source_positions, rotation, translation)

    print(f"success: {result.calibration.success}")
    print(f"TDOA residual RMS: {result.calibration.rms_tdoa_residual_s * 1e6:.2f} us")
    print(
        "microphone RMS position error: "
        f"{rms_position_error(aligned_microphones, microphones_true):.4f} m"
    )
    print(f"source RMS position error: {rms_position_error(aligned_source, source_true):.4f} m")


if __name__ == "__main__":
    main()
