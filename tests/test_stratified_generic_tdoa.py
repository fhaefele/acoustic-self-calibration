import numpy as np
import pytest

from acoustic_self_calibration.measurements import (
    EventTDOAMeasurements,
    reference_star_from_arrivals,
)
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene
from acoustic_self_calibration.stratified.solver import (
    _normalize_reference_star_measurements,
    _seed_event_families,
    calibrate_tdoa,
)


def _measurements(microphone_count: int, event_count: int):
    scene = make_random_3d_pulse_scene(
        microphone_count,
        event_count=event_count,
    )
    ranges = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.source_positions_at_events_m[None, :, :],
        axis=2,
    )
    tdoa = ((ranges[1:] - ranges[[0]]) / 343.0).T
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(event_count),
        receiver_event_times_s=scene.event_times_s,
        microphone_ids=tuple(range(microphone_count)),
        microphone_pairs=tuple((0, index) for index in range(1, microphone_count)),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 2e-6),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    return scene, measurements


@pytest.mark.parametrize("microphone_count", [8, 12, 16, 24])
@pytest.mark.parametrize("event_count", [20, 40])
def test_generic_tdoa_solver_expands_beyond_eight_microphones(
    microphone_count: int,
    event_count: int,
) -> None:
    scene, measurements = _measurements(microphone_count, event_count)
    result = calibrate_tdoa(
        measurements,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )

    assert result.status == "solved", result.diagnostics
    assert result.microphone_positions_m is not None
    assert result.microphone_ids == measurements.microphone_ids
    assert result.source_positions_m is not None
    estimated_distances = np.linalg.norm(
        result.microphone_positions_m[:, None, :] - result.microphone_positions_m[None, :, :],
        axis=2,
    )
    true_distances = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.microphone_positions_m[None, :, :],
        axis=2,
    )
    assert np.sqrt(np.mean((estimated_distances - true_distances) ** 2)) < 1e-5
    assert result.tdoa_rms_s is not None
    assert result.tdoa_rms_s < 1e-8
    assert result.diagnostics.extra_microphones_completed == microphone_count - 8
    assert result.selected_class is not None
    assert result.selected_class.representative.microphone_positions_m.shape == (
        microphone_count,
        3,
    )
    assert np.allclose(
        result.selected_class.representative.microphone_positions_m,
        result.microphone_positions_m,
    )
    split = result.selected_class.representative.split
    assert split.completion_mask.shape == (microphone_count, event_count)
    if microphone_count > 8:
        assert np.any(split.completion_mask[8:])


def test_generic_solver_accepts_nonzero_reference_and_noncontiguous_ids() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
    ranges = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.source_positions_at_events_m[None, :, :],
        axis=2,
    )
    microphone_ids = (101, 7, 42, 300, 9, 55, 88, 12)
    measurements = reference_star_from_arrivals(
        (ranges / 343.0).T,
        np.full((20, 8), 1e-6),
        receiver_event_times_s=scene.event_times_s,
        reference_microphone=42,
        microphone_ids=microphone_ids,
    )

    result = calibrate_tdoa(
        measurements,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )

    assert result.status == "solved"
    assert result.microphone_positions_m is not None
    assert result.microphone_ids == microphone_ids
    assert result.selected_class is not None
    hypothesis = result.selected_class.representative
    assert set(hypothesis.seed_microphone_ids).issubset(set(microphone_ids))
    assert hypothesis.microphone_positions_m.shape == (8, 3)
    assert np.allclose(
        hypothesis.microphone_positions_m,
        result.microphone_positions_m,
    )
    assert hypothesis.split.generation_mask.shape == (8, 20)
    assert hypothesis.arrival_gauge_shifts_m is not None
    assert np.any(np.abs(hypothesis.arrival_gauge_shifts_m) > 0.0)
    for hypothesis_class in result.classes:
        assert hypothesis_class.representative.arrival_gauge_shifts_m is not None
        assert np.allclose(
            hypothesis_class.representative.arrival_gauge_shifts_m,
            hypothesis.arrival_gauge_shifts_m,
            equal_nan=True,
        )
        for member in hypothesis_class.members:
            assert member.arrival_gauge_shifts_m is not None
            assert np.allclose(
                member.arrival_gauge_shifts_m,
                hypothesis.arrival_gauge_shifts_m,
                equal_nan=True,
            )
    estimated_distances = np.linalg.norm(
        result.microphone_positions_m[:, None, :] - result.microphone_positions_m[None, :, :],
        axis=2,
    )
    true_distances = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.microphone_positions_m[None, :, :],
        axis=2,
    )
    assert np.sqrt(np.mean((estimated_distances - true_distances) ** 2)) < 1e-5


def test_normalization_avoids_noisy_reference_and_fixed_first_eight() -> None:
    scene = make_random_3d_pulse_scene(12, event_count=20)
    ranges = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.source_positions_at_events_m[None, :, :],
        axis=2,
    )
    arrival_sigma = np.full((20, 12), 1e-6)
    arrival_sigma[:, 0] = 20e-6
    arrival_sigma[:, 1] = 15e-6
    measurements = reference_star_from_arrivals(
        (ranges / 343.0).T,
        arrival_sigma,
        receiver_event_times_s=scene.event_times_s,
        reference_microphone=0,
    )

    normalized, internal_to_input = _normalize_reference_star_measurements(measurements)

    assert internal_to_input[0] not in {0, 1}
    assert 1 not in internal_to_input[:8]
    assert normalized.microphone_pairs == tuple((0, index) for index in range(1, 12))
    assert normalized.covariance_s2 is not None
    covariance = normalized.covariance_s2[0]
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.any(np.abs(off_diagonal) > 0.0)
    assert normalized.arrival_representatives_s is not None
    assert np.allclose(normalized.arrival_representatives_s[:, 0], 0.0)


def test_required_constraint_receivers_are_kept_in_seed_order() -> None:
    scene = make_random_3d_pulse_scene(12, event_count=20)
    ranges = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.source_positions_at_events_m[None, :, :],
        axis=2,
    )
    arrival_sigma = np.full((20, 12), 1e-6)
    arrival_sigma[:, 10] = 50e-6
    microphone_ids = tuple(100 + index for index in range(12))
    measurements = reference_star_from_arrivals(
        (ranges / 343.0).T,
        arrival_sigma,
        receiver_event_times_s=scene.event_times_s,
        reference_microphone=100,
        microphone_ids=microphone_ids,
    )

    _, internal_to_input = _normalize_reference_star_measurements(
        measurements,
        required_microphone_ids=(110,),
    )

    assert 10 in internal_to_input[:8]


def test_seed_event_families_are_prefix_stable() -> None:
    fitting = tuple(index for index in range(40) if index not in (5, 17, 30, 33))
    budgets = (1, 2, 3, 4, 8, 16)
    families = [_seed_event_families(fitting, budget=budget) for budget in budgets]
    for budget, family in zip(budgets, families, strict=True):
        assert 1 <= len(family) <= budget
        assert all(set(seed).issubset(fitting) for seed in family)
        assert all(len(set(seed)) == 6 for seed in family)
    for small, large in zip(families, families[1:], strict=False):
        assert large[: len(small)] == small
