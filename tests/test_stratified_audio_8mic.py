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


@pytest.mark.parametrize("event_count", [20, 40])
def test_eight_microphone_audio_gate(event_count: int) -> None:
    scene = make_random_3d_pulse_scene(8, event_count=event_count)
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

    assert result.status == "solved", result.calibration.diagnostics
    assert result.detected_event_count == event_count
    assert len(result.event_times_s) == event_count
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
    assert result.rms_tdoa_residual_s is not None
    assert result.rms_tdoa_residual_s < 60e-6

    exact_tdoa = exact_reference_tdoas(
        scene.microphone_positions_m,
        scene.source_positions_at_events_m,
    )
    assert np.sqrt(np.mean((result.tdoa_s - exact_tdoa) ** 2)) < 60e-6
    assert np.all(result.tdoa_sigma_s > 0.0)
    assert result.measurements.covariance_s2 is not None


def test_tracker_disabled_audio_diagnostic_still_emits_valid_measurements() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        use_temporal_tracking=False,
        receiver_subset_budget=1,
        event_subset_budget=1,
        root_start_count=16,
        metric_start_count=8,
    )
    assert result.measurements.measurement_basis == "reference_star"
    assert np.all(result.measurements.valid)
    assert not result.temporal_tracking_enabled
