import numpy as np

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
from acoustic_self_calibration.stratified.planar import upgrade_metric_planar


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
