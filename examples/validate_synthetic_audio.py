"""Reproducible 8/12/16/24 microphone stratified audio validation."""

from __future__ import annotations

import json
import time

from acoustic_self_calibration import calibrate_audio
from acoustic_self_calibration.geometry import (
    apply_rigid,
    rigid_align,
    rms_position_error,
)
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene


def validate(microphone_count: int, event_count: int) -> dict[str, float | int | str]:
    scene = make_random_3d_pulse_scene(
        microphone_count,
        event_count=event_count,
    )
    started = time.perf_counter()
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        model="general_3d",
    )
    solve_seconds = time.perf_counter() - started

    if result.microphone_positions_m is None or result.source_positions_m is None:
        return {
            "microphones": microphone_count,
            "events": event_count,
            "status": result.status,
            "solve_seconds": round(solve_seconds, 4),
        }

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    aligned_source = apply_rigid(
        result.source_positions_m,
        rotation,
        translation,
    )
    return {
        "microphones": microphone_count,
        "events": event_count,
        "status": result.status,
        "solve_seconds": round(solve_seconds, 4),
        "microphone_rms_error_m": round(
            rms_position_error(aligned_microphones, scene.microphone_positions_m),
            6,
        ),
        "source_rms_error_m": round(
            rms_position_error(
                aligned_source,
                scene.source_positions_at_events_m,
            ),
            6,
        ),
        "tdoa_residual_rms_us": round(
            1e6 * (result.rms_tdoa_residual_s or 0.0),
            4,
        ),
    }


def main() -> None:
    rows = [
        validate(microphone_count, event_count)
        for microphone_count in (8, 12, 16, 24)
        for event_count in (20, 40)
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
