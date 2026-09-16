import numpy as np

from acoustic_self_calibration.simulation import render_moving_source


def test_renderer_shape_and_finite_values():
    fs = 8_000
    n = 8_000
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    mics = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 0.2],
            [0.3, -0.8, 1.2],
            [1.2, 0.4, 0.7],
        ]
    )
    tt = np.array([0.0, 0.5, 1.0])
    ss = np.array([[-1.0, 0.0, 1.0], [0.0, 0.5, 1.2], [1.0, 1.0, 1.4]])
    audio = render_moving_source(signal, fs, mics, tt, ss, radiation_pattern="cardioid")
    assert audio.shape == (n, len(mics))
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_cardioid_radiates_more_forward_than_backward():
    from acoustic_self_calibration.radiation import radiation_gain

    gains = radiation_gain(np.array([1.0, 0.0, -1.0]), "cardioid")
    assert gains[0] > gains[1] > gains[2]
    assert gains[0] == 1.0
    assert gains[2] == 0.0
