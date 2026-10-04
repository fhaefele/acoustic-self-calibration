"""Minimal end-to-end calibration of a deterministic pulse scene."""

from __future__ import annotations

from acoustic_self_calibration import calibrate_audio
from acoustic_self_calibration.geometry import (
    apply_rigid,
    rigid_align,
    rms_position_error,
)
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene


def main() -> None:
    scene = make_random_3d_pulse_scene(16, event_count=20)
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        model="general_3d",
    )
    if result.microphone_positions_m is None or result.source_positions_m is None:
        raise SystemExit(f"calibration did not produce geometry: {result.status}")

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    aligned_sources = apply_rigid(
        result.source_positions_m,
        rotation,
        translation,
    )

    print(f"status: {result.status}")
    print(
        "microphone RMS position error: "
        f"{rms_position_error(aligned_microphones, scene.microphone_positions_m):.4f} m"
    )
    print(
        "source RMS position error: "
        f"{rms_position_error(aligned_sources, scene.source_positions_at_events_m):.4f} m"
    )
    if result.rms_tdoa_residual_s is not None:
        print(f"TDOA residual RMS: {1e6 * result.rms_tdoa_residual_s:.2f} us")


if __name__ == "__main__":
    main()
