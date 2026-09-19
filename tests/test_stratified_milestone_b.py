import numpy as np

from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.metric_upgrade import upgrade_metric_3d
from acoustic_self_calibration.stratified.notation import reference_arrivals_from_ranges
from acoustic_self_calibration.stratified.offsets_linear import solve_linear_offsets


def test_exact_9r5s_tdoa_to_euclidean_scene_round_trip() -> None:
    rng = np.random.default_rng(310)
    microphones = rng.normal(size=(9, 3))
    sources = rng.normal(size=(5, 3))
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    arrivals = reference_arrivals_from_ranges(ranges)

    offsets = solve_linear_offsets(arrivals, dimension=3)
    factorization = factor_corrected_ranges(
        offsets.corrected_ranges_m,
        dimension=3,
    )
    metric = upgrade_metric_3d(
        factorization,
        offsets.corrected_ranges_m,
    )

    assert offsets.status == "solved"
    assert metric.status == "solved"
    assert metric.diagnostics.accepted_candidate_count >= 1
    candidate = metric.candidates[0]

    aligned_microphones, rotation, translation = rigid_align(
        candidate.microphone_positions_m,
        microphones,
    )
    aligned_sources = apply_rigid(
        candidate.source_positions_m,
        rotation,
        translation,
    )
    assert rms_position_error(aligned_microphones, microphones) < 1e-8
    assert rms_position_error(aligned_sources, sources) < 1e-8


def test_noisy_linear_anchor_reports_conditioning_without_overclaiming_noise() -> None:
    rng = np.random.default_rng(311)
    microphones = rng.normal(size=(9, 3))
    sources = rng.normal(size=(5, 3))
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    arrivals = reference_arrivals_from_ranges(ranges)
    noisy = arrivals + rng.normal(scale=2e-5, size=arrivals.shape)
    noisy[0] = 0.0

    offsets = solve_linear_offsets(noisy, dimension=3)
    assert offsets.status == "solved"
    assert offsets.diagnostics.compacted_rank.effective_rank == 3
    assert np.isfinite(offsets.diagnostics.normalized_condition_number)
    assert offsets.diagnostics.minimum_denominator > 0.0
    assert np.all(np.isfinite(offsets.corrected_ranges_m))


def test_metric_upgrade_rejects_mismatched_factorization_and_ranges() -> None:
    rng = np.random.default_rng(312)
    microphones = rng.normal(size=(9, 3))
    sources = rng.normal(size=(5, 3))
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    factorization = factor_corrected_ranges(ranges, dimension=3)
    changed = ranges.copy()
    changed[1, 1] += 0.05

    try:
        upgrade_metric_3d(factorization, changed)
    except ValueError as error:
        assert "different data" in str(error)
    else:
        raise AssertionError("mismatched factorization/ranges must be rejected")
