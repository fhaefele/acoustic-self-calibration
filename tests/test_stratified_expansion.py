from dataclasses import replace

import numpy as np
import pytest

from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene
from acoustic_self_calibration.stratified import solver
from acoustic_self_calibration.stratified.expansion import (
    extend_event_offsets,
    localize_receiver_from_ranges,
    localize_source_from_tdoa,
)
from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.metric_upgrade import (
    upgrade_metric_3d_overdetermined,
)
from acoustic_self_calibration.stratified.offsets_minimal import solve_offsets_7r6s


def _scene(event_count: int = 20):
    scene = make_random_3d_pulse_scene(8, event_count=event_count)
    microphones = scene.microphone_positions_m
    sources = scene.source_positions_at_events_m
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    arrivals = ranges - ranges[[0], :]
    return microphones, sources, ranges, arrivals


def test_offset_expansion_recovers_all_events_from_7r6s_seed() -> None:
    _, _, ranges, arrivals = _scene(20)
    seed_events = np.linspace(0, 19, 6, dtype=int)
    minimal = solve_offsets_7r6s(arrivals[:7, seed_events], start_count=48)
    physical = min(
        minimal.roots,
        key=lambda root: np.sqrt(np.mean((root.corrected_ranges_m - ranges[:7, seed_events]) ** 2)),
    )
    expanded = extend_event_offsets(
        arrivals[:7, seed_events],
        physical.offsets_m,
        arrivals[:7],
    )

    assert expanded.column_space_rank == 4
    assert expanded.left_nullity == 2
    assert np.all(expanded.success)
    assert np.max(np.abs(expanded.offsets_m + ranges[0])) < 1e-8
    assert np.sqrt(np.mean((expanded.corrected_ranges_m - ranges[:7]) ** 2)) < 1e-8


def test_overdetermined_metric_upgrade_recovers_7_receiver_scene() -> None:
    microphones, sources, ranges, arrivals = _scene(20)
    seed_events = np.linspace(0, 19, 6, dtype=int)
    minimal = solve_offsets_7r6s(arrivals[:7, seed_events], start_count=48)
    physical = min(
        minimal.roots,
        key=lambda root: np.sqrt(np.mean((root.corrected_ranges_m - ranges[:7, seed_events]) ** 2)),
    )
    expanded = extend_event_offsets(
        arrivals[:7, seed_events],
        physical.offsets_m,
        arrivals[:7],
    )
    factorization = factor_corrected_ranges(
        expanded.corrected_ranges_m,
        dimension=3,
    )
    metric = upgrade_metric_3d_overdetermined(
        factorization,
        expanded.corrected_ranges_m,
        start_count=24,
    )
    assert metric.candidates
    assert metric.candidates[0].corrected_range_rms_m < 1e-8


def test_source_localization_from_five_receivers_is_exact() -> None:
    microphones, sources, _, arrivals = _scene(20)
    event = 7
    result = localize_source_from_tdoa(
        microphones[:5],
        arrivals[:5, event],
    )
    assert result.linear_rank == 4
    assert np.linalg.norm(result.source_position_m - sources[event]) < 1e-10
    assert abs(result.norm_residual_m2) < 1e-10


def test_offset_expansion_preserves_model_nullspace_under_roundoff() -> None:
    _, _, ranges, arrivals = _scene(20)
    seed_events = np.linspace(0, 19, 6, dtype=int)
    rng = np.random.default_rng(830)
    perturbed = arrivals[:7, seed_events] + rng.normal(scale=1e-10, size=(7, 6))
    expanded = extend_event_offsets(
        perturbed,
        -ranges[0, seed_events],
        arrivals[:7],
        dimension=3,
    )
    assert expanded.column_space_rank == 4
    assert expanded.left_nullity == 2
    assert np.all(expanded.success)
    np.testing.assert_allclose(expanded.offsets_m, -ranges[0], atol=1e-7)


def test_expansion_rejects_full_rank_seed_without_membership_constraint() -> None:
    rng = np.random.default_rng(0)
    seed_arrivals = rng.uniform(1.0, 9.0, size=(7, 8))
    seed_offsets = np.full(8, -5.0)
    with pytest.raises(ValueError, match="offset-membership constraint"):
        extend_event_offsets(seed_arrivals, seed_offsets, seed_arrivals)
    # An explicit dimension caps the rank and restores a left nullspace.
    expanded = extend_event_offsets(
        seed_arrivals,
        seed_offsets,
        seed_arrivals,
        dimension=3,
    )
    assert expanded.left_nullity >= 2


@pytest.mark.parametrize("negative_excluded_range", [False, True])
def test_completion_handles_relocalized_seed_and_impossible_range(
    monkeypatch: pytest.MonkeyPatch, negative_excluded_range: bool
) -> None:
    _, _, ranges, arrivals = _scene(20)
    fitting, validation = solver._split_events(20)
    seed = solver._seed_event_families(fitting, budget=1)[0]
    original = solver.extend_event_offsets

    def reject_first_seed(*args, **kwargs):
        expanded = original(*args, **kwargs)
        success = expanded.success.copy()
        success[fitting.index(seed[0])] = False
        return replace(expanded, success=success)

    monkeypatch.setattr(solver, "extend_event_offsets", reject_first_seed)
    if negative_excluded_range:
        arrivals[7, fitting[-1]] = -ranges[0, fitting[-1]] - 1.0
    tdoa = arrivals[1:].T / 343.0
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(20),
        receiver_event_times_s=np.arange(20, dtype=float),
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 2e-6),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    completed = solver._complete_geometry(
        arrivals_m=arrivals,
        primitive_valid=np.ones_like(arrivals, dtype=bool),
        fitting_events=fitting,
        validation_events=validation,
        receiver_subset=tuple(range(7)),
        seed_events=seed,
        seed_root_offsets_m=-ranges[0, np.asarray(seed)],
        speed_of_sound=343.0,
        measurements=measurements,
        subset_id="seed_relocalization",
        root_id=0,
        root_generating_equation_rms=0.0,
        metric_start_count=12,
    )
    if negative_excluded_range:
        assert completed is None
        return
    assert completed is not None
    hypothesis = completed[0][0] if isinstance(completed, list) else completed[0]
    assert np.all(hypothesis.split.generation_mask[1:7, seed[0]])
    assert not np.any(hypothesis.split.generation_mask & hypothesis.split.completion_mask)


def test_receiver_trilateration_from_known_sources_is_exact() -> None:
    microphones, sources, ranges, _ = _scene(20)
    result = localize_receiver_from_ranges(sources[:8], ranges[7, :8])
    assert result.linear_rank == 3
    assert np.linalg.norm(result.receiver_position_m - microphones[7]) < 1e-10
    assert result.range_rms_m < 1e-10


def test_robust_receiver_localization_trims_one_gross_range_outlier() -> None:
    microphones, sources, ranges, _ = _scene(20)
    corrupted = np.array(ranges[7, :12], copy=True)
    corrupted[0] += 0.5
    result = localize_receiver_from_ranges(
        sources[:12],
        corrupted,
        robust=True,
    )
    assert result.linear_rank == 3
    assert result.range_rms_m > 0.05
    assert result.inlier_rms_m < 1e-8
    assert np.linalg.norm(result.receiver_position_m - microphones[7]) < 1e-8


def test_room_millimetre_timing_noise_reaches_refined_geometry() -> None:
    from acoustic_self_calibration.geometry import rigid_align, rms_position_error
    from acoustic_self_calibration.measurements import reference_star_from_arrivals
    from acoustic_self_calibration.simulation import make_room_benchmark_pulse_scene

    scene = make_room_benchmark_pulse_scene(8, event_count=20, seed=0, sample_rate_hz=8000)
    arrivals = (
        np.linalg.norm(
            scene.source_positions_at_events_m[:, None, :]
            - scene.microphone_positions_m[None, :, :],
            axis=2,
        )
        / 343.0
    )
    rng = np.random.default_rng(np.random.SeedSequence([0, 92741]))
    arrivals += rng.normal(0.0, 2e-6, arrivals.shape)
    measurements = reference_star_from_arrivals(
        arrivals - arrivals[:, [0]],
        np.full_like(arrivals, 2e-6),
        receiver_event_times_s=scene.event_times_s + arrivals[:, 0],
        reference_microphone=0,
    )
    result = solver.calibrate_tdoa(measurements)
    assert result.status == "solved"
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
    assert result.tdoa_rms_s is not None and result.tdoa_rms_s < 6e-6
    aligned, _, _ = rigid_align(result.microphone_positions_m, scene.microphone_positions_m)
    assert rms_position_error(aligned, scene.microphone_positions_m) < 0.02

    mismatched = replace(result, source_positions_m=result.source_positions_m + 0.5)
    rejected = solver._check_general_fit_uncertainty(mismatched, measurements, 343.0)
    assert rejected.status == "weakly_identified"
    assert "residual_exceeds_timing_uncertainty" in rejected.diagnostics.rejection_reasons
