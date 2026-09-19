import numpy as np
import pytest

from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.metric_upgrade import (
    build_receiver_metric_system,
    upgrade_metric_3d,
)


def _scene(*, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    microphones = scale * rng.normal(size=(9, 3))
    sources = scale * rng.normal(size=(5, 3))
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    return microphones, sources, ranges


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("seed", [120, 141, 177])
def test_exact_9r5s_metric_upgrade_recovers_scene(scale: float, seed: int) -> None:
    microphones, sources, ranges = _scene(seed=seed, scale=scale)
    factorization = factor_corrected_ranges(ranges, dimension=3)
    result = upgrade_metric_3d(factorization, ranges)

    assert result.status == "solved"
    assert result.diagnostics.receiver_design_rank == 8
    assert result.diagnostics.receiver_design_nullity == 1
    assert result.diagnostics.accepted_candidate_count >= 1
    candidate = result.candidates[0]
    assert candidate.corrected_range_rms_m < 1e-8 * scale

    aligned_mics, rotation, translation = rigid_align(
        candidate.microphone_positions_m,
        microphones,
    )
    aligned_sources = apply_rigid(
        candidate.source_positions_m,
        rotation,
        translation,
    )
    assert rms_position_error(aligned_mics, microphones) < 1e-8 * scale
    assert rms_position_error(aligned_sources, sources) < 1e-8 * scale


def test_receiver_metric_design_has_expected_9r5s_rank() -> None:
    _, _, ranges = _scene(seed=201)
    factorization = factor_corrected_ranges(ranges, dimension=3)
    scale = np.median(ranges[ranges > 0.0])
    design, rhs = build_receiver_metric_system(
        factorization.receiver_factors / scale,
        ranges / scale,
    )
    assert design.shape == (8, 9)
    assert rhs.shape == (8,)
    assert np.linalg.matrix_rank(design) == 8


def test_noisy_metric_upgrade_returns_ranked_diagnostic_candidates() -> None:
    rng = np.random.default_rng(202)
    _, _, ranges = _scene(seed=203)
    noisy = np.maximum(ranges + rng.normal(scale=1e-4, size=ranges.shape), 1e-8)
    factorization = factor_corrected_ranges(noisy, dimension=3)
    result = upgrade_metric_3d(
        factorization,
        noisy,
        acceptance_rms_m=1e-3,
    )

    assert result.status in {"solved", "weakly_identified"}
    assert len(result.candidates) >= 1
    assert result.diagnostics.real_root_count >= 1
    assert result.diagnostics.positive_definite_candidate_count >= 1
    rms_values = [candidate.corrected_range_rms_m for candidate in result.candidates]
    assert rms_values == sorted(rms_values)
    assert rms_values[0] < 1e-3


def test_exact_planar_receiver_geometry_is_reported_degenerate_for_3d_metric() -> None:
    rng = np.random.default_rng(204)
    microphones = np.column_stack([rng.normal(size=9), rng.normal(size=9), np.zeros(9)])
    sources = rng.normal(size=(5, 3))
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    factorization = factor_corrected_ranges(ranges, dimension=3)
    result = upgrade_metric_3d(factorization, ranges)

    assert result.status == "degenerate"
    assert "affine_factor_rank_below_3" in result.diagnostics.reasons
    assert not result.candidates


def test_near_planar_geometry_exposes_weak_affine_rank_separation() -> None:
    rng = np.random.default_rng(205)
    microphones = rng.normal(size=(9, 3))
    microphones[:, 2] *= 1e-7
    sources = rng.normal(size=(5, 3))
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    factorization = factor_corrected_ranges(ranges, dimension=3)
    result = upgrade_metric_3d(
        factorization,
        ranges,
        weak_factor_ratio=1e-5,
    )

    assert result.status in {"weakly_identified", "degenerate"}
    if result.status == "weakly_identified":
        assert "weak_affine_rank_separation" in result.diagnostics.reasons


def test_exact_metric_upgrade_keeps_nonphysical_branches_out_of_acceptance() -> None:
    _, _, ranges = _scene(seed=120)
    factorization = factor_corrected_ranges(ranges, dimension=3)
    result = upgrade_metric_3d(factorization, ranges)

    assert result.diagnostics.real_root_count >= 1
    assert result.diagnostics.positive_definite_candidate_count >= 1
    assert result.diagnostics.accepted_candidate_count >= 1
    assert (
        result.diagnostics.accepted_candidate_count
        <= result.diagnostics.positive_definite_candidate_count
        <= result.diagnostics.real_root_count
    )
    accepted = [
        candidate
        for candidate in result.candidates
        if candidate.corrected_range_rms_m <= result.diagnostics.acceptance_rms_m
    ]
    assert accepted
    assert accepted[0].corrected_range_rms_m < 1e-10
