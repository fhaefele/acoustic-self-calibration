import numpy as np
import pytest

from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.notation import cross_gram_from_ranges


def _scene(
    receiver_count: int,
    event_count: int,
    dimension: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    receivers = rng.normal(size=(receiver_count, dimension))
    sources = rng.normal(size=(event_count, dimension))
    return np.linalg.norm(receivers[:, None, :] - sources[None, :, :], axis=2)


def test_exact_3d_ranges_factor_to_rank_three() -> None:
    ranges = _scene(9, 5, 3, seed=101)
    factorization = factor_corrected_ranges(ranges, dimension=3)

    expected = cross_gram_from_ranges(ranges)
    reconstructed = factorization.receiver_factors[1:] @ factorization.source_factors[1:].T
    assert factorization.receiver_factors.shape == (9, 3)
    assert factorization.source_factors.shape == (5, 3)
    assert factorization.rank_diagnostics.effective_rank == 3
    assert factorization.rank_diagnostics.expected_rank == 3
    assert factorization.discarded_singular_rms < 1e-12
    assert factorization.relative_discarded_energy < 1e-24
    assert factorization.reconstruction_rms < 1e-12
    assert np.allclose(reconstructed, expected, atol=1e-12, rtol=1e-12)
    assert np.allclose(factorization.receiver_factors[0], 0.0)
    assert np.allclose(factorization.source_factors[0], 0.0)


def test_exact_2d_ranges_factor_to_rank_two() -> None:
    ranges = _scene(7, 4, 2, seed=102)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    assert factorization.rank_diagnostics.effective_rank == 2
    assert factorization.reconstruction_rms < 1e-12


def test_noisy_ranges_expose_discarded_tail() -> None:
    rng = np.random.default_rng(103)
    ranges = _scene(12, 8, 3, seed=104)
    noisy = np.maximum(ranges + rng.normal(scale=2e-3, size=ranges.shape), 0.0)
    factorization = factor_corrected_ranges(noisy, dimension=3)

    assert factorization.discarded_singular_rms > 0.0
    assert factorization.relative_discarded_energy > 0.0
    assert factorization.reconstruction_rms > 0.0
    assert np.all(np.isfinite(factorization.rank_diagnostics.singular_values))


def test_factorization_rejects_negative_ranges() -> None:
    ranges = _scene(9, 5, 3, seed=105)
    ranges[1, 2] = -1e-3
    with pytest.raises(ValueError, match="non-negative"):
        factor_corrected_ranges(ranges, dimension=3)
