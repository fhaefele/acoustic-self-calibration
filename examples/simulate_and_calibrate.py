from __future__ import annotations

import numpy as np

from acoustic_self_calibration import calibrate_audio
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.simulation import (
    broadband_pulse_train,
    random_microphone_array,
    render_moving_source,
)


def main() -> None:
    rng = np.random.default_rng(21)
    sample_rate = 48_000
    duration = 5.0
    microphones = random_microphone_array(8, rng=rng)

    trajectory_times = np.linspace(0.0, duration, 31)
    phase = np.linspace(0.0, 2.0 * np.pi, len(trajectory_times))
    trajectory = np.column_stack(
        [
            2.2 * np.cos(0.7 * phase),
            1.7 * np.sin(0.7 * phase),
            1.3 + 0.5 * np.sin(0.45 * phase + 0.3),
        ]
    )
    event_times = np.linspace(0.4, 4.4, 20)
    signal = broadband_pulse_train(event_times, sample_rate, duration, rng=rng)
    forward = microphones.mean(axis=0)[None, :] - trajectory
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    audio = render_moving_source(
        signal,
        sample_rate,
        microphones,
        trajectory_times,
        trajectory,
        source_forward=forward,
        radiation_pattern="cardioid",
        noise_std=1e-5,
        rng=rng,
    )

    result = calibrate_audio(
        audio,
        sample_rate,
        event_min_gap_s=0.05,
        max_tau_s=0.025,
        tdoa_template_s=0.002,
        pair_mode="reference",
        compute_laplace_uncertainty=False,
    )

    true_source = np.column_stack(
        [
            np.interp(event_times, trajectory_times, trajectory[:, dimension])
            for dimension in range(3)
        ]
    )
    aligned_mics, rotation, translation = rigid_align(
        result.calibration.microphone_positions,
        microphones,
    )
    aligned_source = apply_rigid(result.calibration.source_positions, rotation, translation)

    print(f"events: {len(result.event_times_s)}")
    print(f"microphone RMS: {rms_position_error(aligned_mics, microphones):.4f} m")
    print(f"source RMS: {rms_position_error(aligned_source, true_source):.4f} m")


if __name__ == "__main__":
    main()
