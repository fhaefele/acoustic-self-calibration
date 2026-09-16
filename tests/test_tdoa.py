import numpy as np

from acoustic_self_calibration.tdoa import gcc_phat


def test_gcc_phat_delay_sign_and_value():
    fs = 48_000
    rng = np.random.default_rng(0)
    ref = rng.normal(size=4096)
    delay_samples = 23
    sig = np.concatenate([np.zeros(delay_samples), ref[:-delay_samples]])
    tau, _ = gcc_phat(sig, ref, fs, max_tau=0.01, interp=8)
    assert abs(tau - delay_samples / fs) < 1.0 / fs
