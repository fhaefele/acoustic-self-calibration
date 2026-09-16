"""Reproducible 8/16/24-microphone validation of the public audio API."""

from __future__ import annotations

import json
import time

import numpy as np

from acoustic_self_calibration import calibrate_audio, random_microphone_array, render_moving_source
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error


def broadband_signal(samples: int, rng: np.random.Generator) -> np.ndarray:
    signal = rng.normal(size=samples)
    signal = np.concatenate([[0.0], np.diff(signal)])
    return signal / np.max(np.abs(signal))


def validate(microphone_count: int) -> dict[str, float | int | bool]:
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
        microphones,
        key_times,
        source_keys,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=1e-5,
        rng=rng,
    )

    started = time.perf_counter()
    result = calibrate_audio(
        audio,
        sample_rate,
        frame_size=512,
        hop_size=8192,
        max_tau_s=0.025,
        gcc_interp=16,
        reference_count=2,
        motion_velocity_change_sigma_mps=3.0,
        likelihood="cauchy",
        max_nfev=300,
        compute_laplace_uncertainty=False,
    )
    solve_seconds = time.perf_counter() - started

    true_source = np.column_stack(
        [
            np.interp(result.frame_times_s, key_times, source_keys[:, dimension])
            for dimension in range(3)
        ]
    )
    aligned_microphones, rotation, translation = rigid_align(
        result.calibration.microphone_positions,
        microphones,
    )
    aligned_source = apply_rigid(result.calibration.source_positions, rotation, translation)

    return {
        "microphones": microphone_count,
        "pairs": len(result.microphone_pairs),
        "frames": len(result.frame_times_s),
        "success": result.calibration.success,
        "nfev": result.calibration.nfev,
        "solve_seconds": round(solve_seconds, 4),
        "microphone_rms_error_m": round(rms_position_error(aligned_microphones, microphones), 6),
        "source_rms_error_m": round(rms_position_error(aligned_source, true_source), 6),
        "tdoa_residual_rms_us": round(result.calibration.rms_tdoa_residual_s * 1e6, 4),
    }


def main() -> None:
    print(json.dumps([validate(count) for count in (8, 16, 24)], indent=2))


if __name__ == "__main__":
    main()
