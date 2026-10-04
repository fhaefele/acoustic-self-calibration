import numpy as np

from acoustic_self_calibration import calibrate_audio
from acoustic_self_calibration.geometry import (
    apply_rigid,
    rigid_align,
    rms_position_error,
)
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene


def test_public_audio_entry_point_uses_event_driven_stratified_backend() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
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
    )

    assert result.status == "solved"
    assert result.model == "general_3d"
    assert result.detected_event_count == 20
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
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
    assert np.all(result.tdoa_sigma_s > 0.0)
