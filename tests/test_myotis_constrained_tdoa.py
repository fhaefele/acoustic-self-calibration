import numpy as np

from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    myotis_cross_microphones,
    myotis_cross_source_positions,
)
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint
from acoustic_self_calibration.stratified.solver import calibrate_planar_tdoa


def _measurements() -> tuple[np.ndarray, np.ndarray, EventTDOAMeasurements]:
    microphones = myotis_cross_microphones()
    times = np.linspace(0.25, 2.05, 18)
    sources = myotis_cross_source_positions(times)
    tdoa = exact_reference_tdoas(microphones, sources)
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(len(times)),
        receiver_event_times_s=times,
        microphone_ids=tuple(range(len(microphones))),
        microphone_pairs=tuple((0, index) for index in range(1, len(microphones))),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 1e-7),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    return microphones, sources, measurements


def test_constrained_12mic_cross_recovers_metric_and_extra_receivers() -> None:
    microphones, _, measurements = _measurements()
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="explicit_fixture_constraint",
    )
    result = calibrate_planar_tdoa(
        measurements,
        receiver_subset_budget=3,
        event_subset_budget=2,
        angle_constraint=constraint,
        metric_acceptance_rms_m=1e-7,
        extra_microphone_rms_m=1e-7,
    )

    assert result.status == "solved", result.diagnostics
    assert result.microphone_positions_m is not None
    assert result.source_unsigned_heights_m is not None
    assert result.source_height_sign_known is not None
    assert not np.any(result.source_height_sign_known)
    estimated_distances = np.linalg.norm(
        result.microphone_positions_m[:, None, :] - result.microphone_positions_m[None, :, :],
        axis=2,
    )
    true_distances = np.linalg.norm(
        microphones[:, None, :] - microphones[None, :, :],
        axis=2,
    )
    assert np.allclose(estimated_distances, true_distances, atol=1e-6, rtol=1e-6)
    assert result.diagnostics.extra_microphones_completed == 4
    assert result.tdoa_rms_s is not None
    assert result.tdoa_rms_s < 1e-8


def test_removing_cross_constraint_restores_degenerate_outcome() -> None:
    _, _, measurements = _measurements()
    result = calibrate_planar_tdoa(
        measurements,
        receiver_subset_budget=3,
        event_subset_budget=2,
    )

    assert result.status == "degenerate"
    assert result.microphone_positions_m is None
    assert result.continuous_ambiguity_dimension == 1
    assert result.continuous_ambiguity_nullspace is not None
    assert result.continuous_ambiguity_nullspace.shape == (5, 1)
    assert any(
        reason
        in {
            "receiver_metric_design_rank_deficient",
            "receiver_points_lie_on_nontrivial_conic",
        }
        for reason in result.diagnostics.rejection_reasons
    )


def test_constraint_receiver_outside_initial_seed_order_is_supported() -> None:
    microphones = myotis_cross_microphones()
    times = np.linspace(0.25, 2.05, 18)
    sources = myotis_cross_source_positions(times)
    permutation = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 7, 11])
    permuted_microphones = microphones[permutation]
    tdoa = exact_reference_tdoas(permuted_microphones, sources)
    microphone_ids = tuple(int(value) for value in permutation)
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(len(times)),
        receiver_event_times_s=times,
        microphone_ids=microphone_ids,
        microphone_pairs=tuple(
            (microphone_ids[0], microphone_id) for microphone_id in microphone_ids[1:]
        ),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 1e-7),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="explicit_fixture_constraint",
    )

    result = calibrate_planar_tdoa(
        measurements,
        receiver_subset_budget=3,
        event_subset_budget=2,
        angle_constraint=constraint,
        metric_acceptance_rms_m=1e-7,
        extra_microphone_rms_m=1e-7,
    )

    assert result.status == "solved", result.diagnostics
    assert result.microphone_ids == microphone_ids
    assert result.microphone_positions_m is not None
    estimated_distances = np.linalg.norm(
        result.microphone_positions_m[:, None, :] - result.microphone_positions_m[None, :, :],
        axis=2,
    )
    true_distances = np.linalg.norm(
        permuted_microphones[:, None, :] - permuted_microphones[None, :, :],
        axis=2,
    )
    assert np.allclose(estimated_distances, true_distances, atol=1e-6, rtol=1e-6)
