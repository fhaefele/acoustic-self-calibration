import numpy as np
import pytest

from acoustic_self_calibration.stratified.notation import (
    apply_arrival_gauge,
    compacted_rank_matrix,
    compaction_matrix,
    corrected_ranges_from_offsets,
    cross_gram_from_ranges,
    modified_squared_range_matrix,
    rank_diagnostics,
    reference_arrivals_from_ranges,
    restore_offsets_from_gauge,
)


def _exact_scene(
    *,
    receiver_count: int = 9,
    event_count: int = 5,
    dimension: int = 3,
    seed: int = 17,
):
    rng = np.random.default_rng(seed)
    receivers = rng.normal(size=(receiver_count, dimension))
    sources = rng.normal(size=(event_count, dimension))
    ranges = np.linalg.norm(
        receivers[:, None, :] - sources[None, :, :],
        axis=2,
    )
    return receivers, sources, ranges


def test_compaction_matrix_subtracts_first_entry() -> None:
    values = np.array([3.5, -1.0, 2.0, 7.0])
    compacted = compaction_matrix(4).T @ values
    assert np.allclose(compacted, values[1:] - values[0])


def test_corrected_squared_range_construction_matches_direct_distances() -> None:
    _, _, ranges = _exact_scene()
    arrivals = reference_arrivals_from_ranges(ranges)
    offsets = -ranges[0]

    modified = modified_squared_range_matrix(arrivals, offsets)
    compacted = compacted_rank_matrix(arrivals, offsets)
    direct = (
        compaction_matrix(ranges.shape[0]).T
        @ (ranges * ranges)
        @ compaction_matrix(ranges.shape[1])
    )

    assert np.allclose(compacted, direct, atol=1e-12, rtol=1e-12)
    expected_modified = arrivals * arrivals - 2.0 * arrivals * offsets[None, :]
    assert np.allclose(modified, expected_modified)


def test_cross_gram_matches_centered_coordinate_inner_products() -> None:
    receivers, sources, ranges = _exact_scene()
    q = cross_gram_from_ranges(ranges)
    expected = (receivers[1:] - receivers[0]) @ (sources[1:] - sources[0]).T
    assert np.allclose(q, expected, atol=1e-12, rtol=1e-12)
    diagnostics = rank_diagnostics(q, expected_rank=3)
    assert diagnostics.effective_rank == 3
    assert diagnostics.expected_rank == 3


def test_arrival_gauge_preserves_corrected_ranges() -> None:
    _, _, ranges = _exact_scene()
    arrivals = reference_arrivals_from_ranges(ranges)
    offsets = -ranges[0]
    shifts = np.array([0.7, -1.1, 0.4, 1.6, -0.8])

    shifted_arrivals = apply_arrival_gauge(arrivals, shifts)
    shifted_offsets = offsets + shifts
    restored_offsets = restore_offsets_from_gauge(shifted_offsets, shifts)

    original_ranges = corrected_ranges_from_offsets(arrivals, offsets)
    shifted_ranges = corrected_ranges_from_offsets(shifted_arrivals, shifted_offsets)
    assert np.allclose(restored_offsets, offsets, atol=1e-15, rtol=0.0)
    assert np.allclose(shifted_ranges, original_ranges, atol=1e-15, rtol=0.0)


def test_reference_change_preserves_cross_gram() -> None:
    _, _, ranges = _exact_scene()
    q0 = cross_gram_from_ranges(ranges)

    permutation = np.array([4, 1, 2, 3, 0, 5, 6, 7, 8])
    permuted_ranges = ranges[permutation]
    arrivals = reference_arrivals_from_ranges(permuted_ranges)
    corrected = corrected_ranges_from_offsets(arrivals, -permuted_ranges[0])
    q4 = cross_gram_from_ranges(corrected)

    expected = cross_gram_from_ranges(permuted_ranges)
    assert np.allclose(q4, expected)
    assert np.linalg.matrix_rank(q0) == np.linalg.matrix_rank(q4) == 3


def test_negative_corrected_range_is_rejected() -> None:
    arrivals = np.zeros((3, 2))
    with pytest.raises(ValueError, match="negative"):
        corrected_ranges_from_offsets(arrivals, np.array([1.0, -1.0]))


def test_planar_cross_gram_has_rank_two() -> None:
    rng = np.random.default_rng(44)
    receivers = np.column_stack([rng.normal(size=7), rng.normal(size=7)])
    sources = rng.normal(size=(4, 2))
    ranges = np.linalg.norm(receivers[:, None, :] - sources[None, :, :], axis=2)
    diagnostics = rank_diagnostics(cross_gram_from_ranges(ranges), expected_rank=2)
    assert diagnostics.effective_rank == 2
