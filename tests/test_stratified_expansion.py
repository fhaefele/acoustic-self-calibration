import numpy as np

from acoustic_self_calibration.simulation import make_random_3d_pulse_scene
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
