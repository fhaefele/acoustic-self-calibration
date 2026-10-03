import numpy as np
import pytest

from acoustic_self_calibration.stratified.notation import reference_arrivals_from_ranges
from acoustic_self_calibration.stratified.offsets_linear import (
    build_linear_offset_system,
    solve_linear_offsets,
)


def _ranges(
    receiver_count: int,
    event_count: int,
    dimension: int,
    *,
    seed: int,
    scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    receivers = scale * rng.normal(size=(receiver_count, dimension))
    sources = scale * rng.normal(size=(event_count, dimension))
    return np.linalg.norm(receivers[:, None, :] - sources[None, :, :], axis=2)


def test_literal_zero_reference_9r5s_system_is_rank_deficient() -> None:
    rng = np.random.default_rng(319)
    receivers = rng.normal(size=(9, 3))
    sources = rng.normal(size=(5, 3))
    ranges = np.linalg.norm(receivers[:, None, :] - sources[None, :, :], axis=2)
    arrivals = reference_arrivals_from_ranges(ranges)

    literal, _, _, _ = build_linear_offset_system(arrivals)
    assert np.linalg.matrix_rank(literal) == 8

    gauge = np.array([0.7, -1.1, 0.4, 1.6, -0.8])
    solution = solve_linear_offsets(
        arrivals,
        dimension=3,
        gauge_shifts_m=gauge,
    )
    assert solution.diagnostics.normalized_rank == 9
    assert np.max(np.abs(solution.shifted_offsets_m - (-ranges[0] + gauge))) < 1e-10
    assert np.sqrt(np.mean((solution.corrected_ranges_m - ranges) ** 2)) < 1e-8


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
@pytest.mark.parametrize("seed", [3, 19, 47])
def test_9r5s_auto_gauge_is_scale_robust(scale: float, seed: int) -> None:
    ranges = _ranges(9, 5, 3, seed=seed, scale=scale)
    arrivals = reference_arrivals_from_ranges(ranges)
    solution = solve_linear_offsets(arrivals, dimension=3)

    tolerance = 2e-10 * max(1.0, scale)
    assert np.max(np.abs(solution.offsets_m + ranges[0])) < tolerance
    assert np.max(np.abs(solution.corrected_ranges_m - ranges)) < tolerance
    assert solution.diagnostics.normalized_rank == 9
    assert np.isfinite(solution.diagnostics.normalized_condition_number)


def test_7r4s_all_2d_anchor_recovers_exact_offsets() -> None:
    ranges = _ranges(7, 4, 2, seed=72)
    arrivals = reference_arrivals_from_ranges(ranges)
    solution = solve_linear_offsets(arrivals, dimension=2)

    assert np.max(np.abs(solution.offsets_m + ranges[0])) < 1e-10
    assert np.sqrt(np.mean((solution.corrected_ranges_m - ranges) ** 2)) < 1e-10
    assert solution.diagnostics.compacted_rank.expected_rank == 2


def test_7r4s_mixed_planar_receiver_3d_source_anchor() -> None:
    rng = np.random.default_rng(81)
    receivers = np.column_stack(
        [
            rng.normal(size=7),
            np.zeros(7),
            rng.normal(size=7),
        ]
    )
    sources = rng.normal(size=(4, 3))
    ranges = np.linalg.norm(receivers[:, None, :] - sources[None, :, :], axis=2)
    arrivals = reference_arrivals_from_ranges(ranges)

    solution = solve_linear_offsets(arrivals, dimension=2)
    assert np.max(np.abs(solution.offsets_m + ranges[0])) < 1e-10
    assert np.sqrt(np.mean((solution.corrected_ranges_m - ranges) ** 2)) < 1e-10


def test_wrong_linear_anchor_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires shape"):
        solve_linear_offsets(np.ones((8, 5)), dimension=3)
