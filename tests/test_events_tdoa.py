import numpy as np

from acoustic_self_calibration.events import (
    EventDetection,
    estimate_event_tdoa_measurements,
)
from acoustic_self_calibration.simulation import broadband_pulse_train


def _shift(signal: np.ndarray, samples: int) -> np.ndarray:
    out = np.zeros_like(signal)
    if samples >= 0:
        out[samples:] = signal[: len(signal) - samples]
    else:
        out[:samples] = signal[-samples:]
    return out


def _fixture():
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
    event_samples = np.round(event_times * sample_rate).astype(int)
    detection = EventDetection(
        event_samples=event_samples,
        receiver_event_times_s=event_samples / sample_rate,
        event_channel=1,
        prominence=np.ones(len(event_samples)),
    )
    return sample_rate, delays, audio, detection


def test_event_tdoas_emit_reference_star_measurement_contract() -> None:
    sample_rate, delays, audio, detection = _fixture()
    measurements = estimate_event_tdoa_measurements(
        audio,
        sample_rate,
        detection,
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
    assert measurements.microphone_pairs == ((0, 1), (0, 2), (0, 3))
    assert measurements.measurement_basis == "reference_star"
    assert measurements.measurement_origin == "derived_arrivals"
    assert np.max(np.abs(measurements.tdoa_s - expected[None, :])) <= 1.0 / sample_rate
    assert np.all(measurements.sigma_s > 0.0)
    assert measurements.covariance_s2 is not None
    assert measurements.arrival_representatives_s is not None


def test_tracker_disabled_diagnostic_uses_same_contract() -> None:
    sample_rate, _, audio, detection = _fixture()
    tracked = estimate_event_tdoa_measurements(
        audio,
        sample_rate,
        detection,
        max_tau_s=0.003,
        template_s=0.002,
        use_temporal_tracking=True,
    )
    independent = estimate_event_tdoa_measurements(
        audio,
        sample_rate,
        detection,
        max_tau_s=0.003,
        template_s=0.002,
        use_temporal_tracking=False,
    )

    assert tracked.tdoa_s.shape == independent.tdoa_s.shape
    assert tracked.valid.shape == independent.valid.shape
    assert tracked.microphone_pairs == independent.microphone_pairs
    assert np.all(tracked.valid)
    assert np.all(independent.valid)
