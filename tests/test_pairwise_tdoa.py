import numpy as np

from acoustic_self_calibration.tdoa import (
    despike_tdoa_tracks,
    estimate_pairwise_tdoa_matrix,
    make_microphone_pairs,
)


def test_pair_graph_contains_reference_star():
    pairs = make_microphone_pairs(12, mode="redundant", reference_count=3)
    for j in range(1, 12):
        assert (0, j) in pairs
    assert len(pairs) > 11


def test_pairwise_tdoa_sign():
    fs = 16_000
    rng = np.random.default_rng(0)
    ref = rng.normal(size=4096)
    delay = 17
    late = np.concatenate([np.zeros(delay), ref[:-delay]])
    audio = np.column_stack([ref, late])
    _, tau, _, pairs = estimate_pairwise_tdoa_matrix(
        audio,
        fs,
        microphone_pairs=[(0, 1)],
        frame_size=4096,
        hop_size=4096,
        interp=8,
        max_tau=0.01,
        window="rect",
    )
    assert pairs == ((0, 1),)
    assert abs(tau[0, 0] - delay / fs) < 1 / fs


def test_despike_marks_large_jump():
    fs = 48_000
    tau = np.linspace(-1e-3, 1e-3, 21)[:, None]
    tau[10, 0] += 3e-3
    clean, mask = despike_tdoa_tracks(tau, fs, kernel_size=5, threshold_samples=10)
    assert mask[10, 0]
    assert abs(clean[10, 0] - 0.0) < 2e-4
