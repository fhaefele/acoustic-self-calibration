import numpy as np
import pytest

from acoustic_self_calibration.events import detect_transient_events
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene


@pytest.mark.parametrize("microphone_count", [8, 12, 16, 24])
@pytest.mark.parametrize("event_count", [20, 40])
def test_detects_all_frozen_random_3d_events(
    microphone_count: int,
    event_count: int,
) -> None:
    scene = make_random_3d_pulse_scene(
        microphone_count,
        event_count=event_count,
    )
    detection = detect_transient_events(
        scene.audio,
        scene.sample_rate_hz,
        min_gap_s=0.05,
        relative_prominence=0.003,
    )

    assert len(detection.event_samples) == event_count
    assert 0 <= detection.event_channel < microphone_count
    assert np.all(np.diff(detection.receiver_event_times_s) > 0.0)
    assert np.max(np.abs(detection.receiver_event_times_s - scene.event_times_s)) < 0.02


def test_event_detection_is_invariant_to_global_audio_scaling() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
    first = detect_transient_events(
        scene.audio,
        scene.sample_rate_hz,
        min_gap_s=0.05,
    )
    second = detect_transient_events(
        0.013 * scene.audio,
        scene.sample_rate_hz,
        min_gap_s=0.05,
    )

    assert first.event_channel == second.event_channel
    assert np.array_equal(first.event_samples, second.event_samples)


def test_explicit_event_channel_is_respected() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
    detection = detect_transient_events(
        scene.audio,
        scene.sample_rate_hz,
        event_channel=0,
        min_gap_s=0.05,
    )
    assert detection.event_channel == 0
    assert len(detection.event_samples) == 20
