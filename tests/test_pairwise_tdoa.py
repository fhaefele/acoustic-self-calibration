import numpy as np

from acoustic_self_calibration.events import EventTDOAMeasurements
from acoustic_self_calibration.pipeline import _spanning_tree_measurements
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


def test_cycle_dependent_event_tdoas_reduce_to_spanning_tree() -> None:
    pairs = ((0, 1), (0, 2), (1, 2), (2, 3), (0, 3))
    tdoa = np.arange(20, dtype=float).reshape(4, 5) * 1e-6
    sigma = np.full_like(tdoa, 2e-6)
    measurements = EventTDOAMeasurements(
        event_samples=np.arange(4),
        event_times_s=np.arange(4, dtype=float),
        event_channel=0,
        arrival_delays_s=np.zeros((4, 4)),
        arrival_confidence=np.ones((4, 4)),
        tdoa_s=tdoa,
        confidence=np.ones_like(tdoa),
        microphone_pairs=pairs,
    )

    independent, independent_sigma = _spanning_tree_measurements(
        measurements,
        sigma,
        4,
    )

    assert independent.microphone_pairs == ((0, 1), (0, 2), (2, 3))
    assert np.array_equal(independent.tdoa_s, tdoa[:, [0, 1, 3]])
    assert np.array_equal(independent_sigma, sigma[:, [0, 1, 3]])
