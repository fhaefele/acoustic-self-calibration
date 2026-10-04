import numpy as np

from acoustic_self_calibration.chirps import chirp_train
from acoustic_self_calibration.events import (
    detect_transient_events,
    estimate_event_tdoa_measurements,
)
from acoustic_self_calibration.simulation import render_moving_source


def test_recorded_window_uncertainty_is_scale_invariant_and_noise_sensitive():
    fs = 48000
    times = np.linspace(0.1, 1.9, 20)
    pulse = chirp_train(times, fs, 2.0)
    m = np.array([[float(i) * 0.1, 0, 0] for i in range(8)])
    source = np.array([[0.2, 2.0, 0], [0.2, 2.0, 0]])
    clean = render_moving_source(pulse, fs, m, np.array([0, 2]), source, radiation_pattern="omni")
    detection = detect_transient_events(clean, fs, event_channel=0, min_events=12)
    exact = (np.linalg.norm(m - source[0], axis=1) - np.linalg.norm(m[0] - source[0])) / 343
    estimates = []
    for noise in [0.0, 0.003]:
        audio = clean + np.random.default_rng(127).normal(scale=noise, size=clean.shape)
        measurements = estimate_event_tdoa_measurements(
            audio, fs, detection, timing_uncertainty="waveform"
        )
        estimates.append(measurements)
        assert np.isfinite(measurements.sigma_s).all()
        assert np.sqrt(np.mean((measurements.tdoa_s - exact[1:]) ** 2)) < 4e-6
    assert np.median(estimates[1].sigma_s) >= np.median(estimates[0].sigma_s)
    scaled = estimate_event_tdoa_measurements(
        clean * 0.031, fs, detection, timing_uncertainty="waveform"
    )
    np.testing.assert_allclose(scaled.sigma_s, estimates[0].sigma_s, rtol=1e-5)
    np.testing.assert_allclose(scaled.tdoa_s, estimates[0].tdoa_s, atol=1e-12)
