import numpy as np

from acoustic_self_calibration.events import detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.simulation import broadband_pulse_train


def _shift(signal: np.ndarray, samples: int) -> np.ndarray:
    out = np.zeros_like(signal)
    if samples >= 0:
        out[samples:] = signal[: len(signal) - samples]
    else:
        out[:samples] = signal[-samples:]
    return out


def test_detects_pulses_and_recovers_cycle_consistent_delays() -> None:
    rng = np.random.default_rng(7)
    sample_rate = 48_000
    event_times = np.array([0.20, 0.43, 0.67, 0.92, 1.18, 1.45, 1.73])
    source = broadband_pulse_train(
        event_times,
        sample_rate,
        2.0,
        pulse_duration_s=0.0015,
        rng=rng,
    )

    delays = np.array([21, 0, 37, 12])
    gains = np.array([0.8, 1.2, 0.9, 0.7])
    audio = np.column_stack(
        [gains[index] * _shift(source, int(delay)) for index, delay in enumerate(delays)]
    )
    audio += rng.normal(scale=1e-4, size=audio.shape)

    detection = detect_transient_events(
        audio,
        sample_rate,
        min_gap_s=0.05,
        relative_prominence=0.01,
    )
    assert detection.event_channel == 1
    assert len(detection.event_samples) == len(event_times)

    pairs = ((0, 1), (0, 2), (0, 3))
    measurements = estimate_event_tdoas(
        audio,
        sample_rate,
        detection.event_samples,
        detection.event_channel,
        microphone_pairs=pairs,
        max_tau_s=0.003,
        template_s=0.002,
    )

    expected = np.array(
        [
            (delays[1] - delays[0]) / sample_rate,
            (delays[2] - delays[0]) / sample_rate,
            (delays[3] - delays[0]) / sample_rate,
        ]
    )
    assert np.max(np.abs(measurements.tdoa_s - expected[None, :])) <= 1.0 / sample_rate
    assert np.min(measurements.confidence) > 0.8

    delays_s = measurements.arrival_delays_s
    assert np.allclose(
        measurements.tdoa_s[:, 2],
        delays_s[:, 3] - delays_s[:, 0],
    )
