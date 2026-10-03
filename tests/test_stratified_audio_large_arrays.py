import numpy as np
import pytest

from acoustic_self_calibration.geometry import (
    apply_rigid,
    rigid_align,
    rms_position_error,
)
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    make_random_3d_pulse_scene,
)


@pytest.mark.parametrize("microphone_count", [12, 16, 24])
@pytest.mark.parametrize("event_count", [20, 40])
def test_large_array_audio_gates(
    microphone_count: int,
    event_count: int,
) -> None:
    scene = make_random_3d_pulse_scene(
        microphone_count,
        event_count=event_count,
    )
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
        extra_microphone_inlier_rms_m=0.05,
    )

    assert result.status == "solved"
    assert result.detected_event_count == event_count
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
    assert result.calibration.diagnostics.extra_microphones_completed == microphone_count - 8

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    aligned_sources = apply_rigid(
        result.source_positions_m,
        rotation,
        translation,
    )
    assert (
        rms_position_error(
            aligned_microphones,
            scene.microphone_positions_m,
        )
        < 0.15
    )
    assert (
        rms_position_error(
            aligned_sources,
            scene.source_positions_at_events_m,
        )
        < 0.18
    )
    assert result.rms_tdoa_residual_s is not None
    assert result.rms_tdoa_residual_s < 60e-6

    expected = exact_reference_tdoas(
        scene.microphone_positions_m,
        scene.source_positions_at_events_m,
    )
    assert np.sqrt(np.mean((result.tdoa_s - expected) ** 2)) < 60e-6
