from __future__ import annotations

import json
import time

import numpy as np

from acoustic_self_calibration import calibrate_audio
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.simulation import (
    broadband_pulse_train,
    random_microphone_array,
    render_moving_source,
)


def run_case(microphone_count: int) -> dict[str, float | int | bool]:
    rng = np.random.default_rng(200 + microphone_count)
    sample_rate = 48_000
    duration = 5.0
    microphones = random_microphone_array(
        microphone_count,
        bounds=((-1.2, 1.2), (-1.2, 1.2), (0.0, 1.6)),
        rng=rng,
    )
    trajectory_times = np.linspace(0.0, duration, 31)
    phase = np.linspace(0.0, 2.0 * np.pi, len(trajectory_times))
    trajectory = np.column_stack(
        [
            2.0 * np.cos(0.75 * phase),
            1.6 * np.sin(0.75 * phase),
            1.2 + 0.5 * np.sin(0.55 * phase + 0.2),
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

    started = time.perf_counter()
    result = calibrate_audio(
        audio,
        sample_rate,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        pair_mode="reference",
        compute_laplace_uncertainty=False,
        max_nfev=400,
    )
    elapsed = time.perf_counter() - started

    source_true = np.column_stack(
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
    return {
        "microphones": microphone_count,
        "events": len(result.event_times_s),
        "success": result.calibration.success,
        "microphone_rms_m": rms_position_error(aligned_mics, microphones),
        "source_rms_m": rms_position_error(aligned_source, source_true),
        "tdoa_rms_us": 1e6 * result.calibration.rms_tdoa_residual_s,
        "elapsed_s": elapsed,
    }


def main() -> None:
    print(json.dumps([run_case(count) for count in (8, 16, 24)], indent=2))


if __name__ == "__main__":
    main()
