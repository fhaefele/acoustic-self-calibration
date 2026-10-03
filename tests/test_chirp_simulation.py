import numpy as np
import pytest
from scipy.signal import correlate

from acoustic_self_calibration.chirps import ChirpSpecification, chirp_train
from acoustic_self_calibration.events import detect_transient_events
from acoustic_self_calibration.simulation import render_moving_source


def test_chirp_train_detects_varied_and_rapid_calls():
    times = np.r_[0.1, 0.1 + np.cumsum(np.geomspace(0.04, 0.005, 19))]
    signal = chirp_train(times, 48_000, 1, seed=12)
    detected = detect_transient_events(np.tile(signal[:, None], (1, 8)), 48_000)
    assert len(detected.event_samples) == len(times)
    assert np.max(np.abs(detected.receiver_event_times_s - times)) < 0.0003
    np.testing.assert_array_equal(signal, chirp_train(times, 48_000, 1, seed=12))


def test_static_chirp_renderer_has_analytic_propagation_delay():
    fs = 48_000
    pulse = chirp_train(
        np.array([0.1]), fs, 0.25, specification=ChirpSpecification(vary_calls=False)
    )
    mics = np.array([[0, 0, 0], [1, 0, 0]])
    source = np.array([[0, 2, 0], [0, 2, 0]])
    audio = render_moving_source(
        pulse, fs, mics, np.array([0, 0.25]), source, radiation_pattern="omni"
    )
    expected = (np.sqrt(5) - 2) / 343 * fs
    curve = correlate(audio[:, 1], audio[:, 0], mode="full")
    k = int(np.argmax(curve))
    fraction = 0.5 * (curve[k - 1] - curve[k + 1]) / (curve[k - 1] - 2 * curve[k] + curve[k + 1])
    assert abs(k - (len(pulse) - 1) + fraction - expected) < 0.15


def test_chirps_cover_ultrasonic_band_and_reject_aliasing():
    spec = ChirpSpecification(90_000, 30_000, 0.001, "logarithmic")
    signal = chirp_train(np.array([0.02]), 375_000, 0.05, specification=spec)
    assert np.isfinite(signal).all() and np.max(np.abs(signal)) > 0.5
    with pytest.raises(ValueError, match="Nyquist"):
        chirp_train(np.array([0.02]), 48_000, 0.05, specification=spec)
