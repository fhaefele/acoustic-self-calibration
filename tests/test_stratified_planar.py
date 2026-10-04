import numpy as np
import pytest

from acoustic_self_calibration.simulation import (
    myotis_cross_microphones,
    myotis_cross_source_positions,
    nondegenerate_planar_microphones,
    nondegenerate_planar_sources,
)
from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.identifiability import (
    diagnose_planar_identifiability,
)
from acoustic_self_calibration.stratified.planar import (
    localize_planar_receiver_from_ranges,
    squared_range_noise_tolerance,
    upgrade_metric_planar,
)


def test_planar_receiver_preserves_noisy_squared_ranges() -> None:
    projected = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    heights = np.full(4, 2.0)
    # A common squared-range perturbation cancels in the linear equations.
    # Clipping the negative first planar square would bias the receiver.
    ranges = np.sqrt(np.sum(projected**2, axis=1) + heights**2 - 0.01)
    with pytest.raises(ValueError, match="incompatible"):
        localize_planar_receiver_from_ranges(projected, heights, ranges)
    tolerance = squared_range_noise_tolerance(float(np.max(ranges)), 0.001)
    result = localize_planar_receiver_from_ranges(
        projected, heights, ranges, range_tolerance_m2=tolerance
    )
    np.testing.assert_allclose(result.position_2d_m, 0.0, atol=1e-14)
    assert result.range_rms_m < 0.003
    assert squared_range_noise_tolerance(3.0, 0.001) == pytest.approx(0.036036)


@pytest.mark.parametrize("bad_range", [-1.0, np.nan, np.inf])
def test_planar_receiver_rejects_invalid_ranges(bad_range: float) -> None:
    projected = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="ranges must be finite and nonnegative"):
        localize_planar_receiver_from_ranges(projected, np.ones(3), np.array([bad_range, 2.0, 2.0]))


def _ranges(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )


def _pairwise(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def test_exact_nondegenerate_planar_metric_recovers_observables() -> None:
    microphones = nondegenerate_planar_microphones()
    times = np.linspace(0.2, 2.0, 12)
    sources = nondegenerate_planar_sources(times)
    ranges = _ranges(microphones, sources)

    factorization = factor_corrected_ranges(ranges, dimension=2)
    result = upgrade_metric_planar(factorization, ranges)

    assert result.status == "solved"
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.corrected_range_rms_m < 1e-10
    assert candidate.receiver_linear_rms_m2 < 1e-10
    assert not np.any(candidate.source_height_sign_known)
    assert np.allclose(
        _pairwise(candidate.microphone_positions_m),
        _pairwise(microphones),
        atol=1e-9,
        rtol=1e-9,
    )
    assert np.allclose(
        np.sort(candidate.source_unsigned_heights_m),
        np.sort(np.abs(sources[:, 1])),
        atol=1e-8,
        rtol=1e-8,
    )


def test_planar_independent_source_sign_flips_have_same_observables() -> None:
    microphones = nondegenerate_planar_microphones()
    times = np.linspace(0.2, 2.0, 10)
    sources = nondegenerate_planar_sources(times)
    ranges = _ranges(microphones, sources)
    baseline = upgrade_metric_planar(
        factor_corrected_ranges(ranges, dimension=2),
        ranges,
    ).candidates[0]

    reflected = sources.copy()
    reflected[[1, 4, 8], 1] *= -1.0
    reflected_ranges = _ranges(microphones, reflected)
    changed = upgrade_metric_planar(
        factor_corrected_ranges(reflected_ranges, dimension=2),
        reflected_ranges,
    ).candidates[0]

    assert np.allclose(
        baseline.source_unsigned_heights_m,
        changed.source_unsigned_heights_m,
        atol=1e-9,
        rtol=1e-9,
    )
    assert not np.any(changed.source_height_sign_known)


def test_noisy_nondegenerate_planar_metric_degrades_smoothly() -> None:
    rng = np.random.default_rng(401)
    microphones = nondegenerate_planar_microphones()
    sources = nondegenerate_planar_sources(np.linspace(0.2, 2.0, 20))
    ranges = _ranges(microphones, sources)
    noisy = np.maximum(ranges + rng.normal(scale=2e-4, size=ranges.shape), 1e-8)
    result = upgrade_metric_planar(
        factor_corrected_ranges(noisy, dimension=2),
        noisy,
        acceptance_rms_m=2e-3,
    )

    assert result.status in {"solved", "weakly_identified"}
    assert result.candidates
    candidate = result.candidates[0]
    assert candidate.corrected_range_rms_m < 2e-3
    assert candidate.corrected_range_max_error_m < 1e-2
    assert result.diagnostics.identifiability.continuous_metric_nullity == 0


def test_exact_myotis_cross_is_reported_continuously_degenerate() -> None:
    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)

    diagnostics = diagnose_planar_identifiability(factorization.receiver_factors)
    result = upgrade_metric_planar(factorization, ranges)

    assert diagnostics.continuous_metric_nullity == 1
    assert diagnostics.metric_nullspace.shape == (5, 1)
    assert diagnostics.conic_nullity >= 1
    assert diagnostics.cross_like_degeneracy
    assert result.status == "degenerate"
    assert not result.candidates
    assert "receiver_metric_design_rank_deficient" in result.diagnostics.reasons


def test_nondegenerate_planar_metric_has_no_continuous_null_direction() -> None:
    microphones = nondegenerate_planar_microphones()
    sources = nondegenerate_planar_sources(np.linspace(0.2, 2.0, 12))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    diagnostics = diagnose_planar_identifiability(factorization.receiver_factors)

    assert diagnostics.continuous_metric_nullity == 0
    assert diagnostics.metric_nullspace.shape == (5, 0)
    assert diagnostics.conic_nullity == 0


def test_planar_event_seeds_ignore_validation_observations() -> None:
    from acoustic_self_calibration.stratified.solver import (
        _planar_seed_event_families,
        _split_events,
    )

    rng = np.random.default_rng(74)
    arrivals = rng.normal(size=(8, 40))
    valid = np.ones_like(arrivals, dtype=bool)
    fitting, validation = _split_events(40)
    expected = _planar_seed_event_families(arrivals, valid, fitting, budget=8)
    arrivals[:, validation] = 1e9 * rng.normal(size=(8, len(validation)))
    assert _planar_seed_event_families(arrivals, valid, fitting, budget=8) == expected
    assert len(expected) == 8
    assert all(len(set(family)) == 4 for family in expected)
    assert all(set(family).issubset(fitting) for family in expected)


def test_planar_event_seed_families_are_prefix_stable() -> None:
    from acoustic_self_calibration.stratified.solver import (
        _planar_seed_event_families,
        _split_events,
    )

    rng = np.random.default_rng(91)
    arrivals = rng.normal(size=(8, 40))
    valid = np.ones_like(arrivals, dtype=bool)
    fitting, _ = _split_events(40)
    budgets = (1, 2, 3, 4, 8, 16)
    families = [
        _planar_seed_event_families(arrivals, valid, fitting, budget=budget) for budget in budgets
    ]
    for budget, family in zip(budgets, families, strict=True):
        assert 1 <= len(family) <= budget
        assert all(set(seed).issubset(fitting) for seed in family)
    for small, large in zip(families, families[1:], strict=False):
        assert large[: len(small)] == small


@pytest.mark.parametrize(
    "layout,count,seed",
    [
        ("star", 8, 1),
        ("star", 8, 2),
        ("cross", 8, 2),
        ("cross", 12, 0),
    ],
)
def test_noisy_planar_recovery_respects_identifiability(layout, count, seed) -> None:
    from acoustic_self_calibration import calibrate_planar_tdoa, reference_star_from_arrivals
    from acoustic_self_calibration.geometry import rigid_align, rms_position_error
    from acoustic_self_calibration.simulation import make_planar_benchmark_pulse_scene

    scene = make_planar_benchmark_pulse_scene(
        count,
        array_span_m=2.0,
        source_distance_range_m=(1.0, 3.0),
        layout=layout,
        event_count=20,
        seed=seed,
        sample_rate_hz=8000,
    )
    arrivals = _ranges(scene.microphone_positions_m, scene.source_positions_at_events_m).T / 343.0
    sigma = 2e-6
    rng = np.random.default_rng(np.random.SeedSequence([seed, 92741]))
    arrivals += rng.normal(0.0, sigma, arrivals.shape)
    measurements = reference_star_from_arrivals(
        arrivals - arrivals[:, [0]],
        np.full_like(arrivals, sigma),
        receiver_event_times_s=scene.event_times_s + arrivals[:, 0],
        reference_microphone=0,
    )
    result = calibrate_planar_tdoa(measurements)
    if layout == "cross":
        assert result.status != "solved"
        return
    assert result.status == "solved"
    assert result.microphone_positions_m is not None
    aligned, _, _ = rigid_align(result.microphone_positions_m, scene.microphone_positions_m)
    assert rms_position_error(aligned, scene.microphone_positions_m) < 0.02
