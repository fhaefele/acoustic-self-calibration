import numpy as np
import pytest

from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    myotis_cross_microphones,
    myotis_cross_source_positions,
    nondegenerate_planar_microphones,
    nondegenerate_planar_sources,
)
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint
from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.identifiability import (
    diagnose_planar_identifiability,
)
from acoustic_self_calibration.stratified.planar import upgrade_metric_planar
from acoustic_self_calibration.stratified.solver import compare_tdoa_models_8mic


def _ranges(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )


def _measurements(
    microphones: np.ndarray,
    sources: np.ndarray,
    times: np.ndarray,
) -> EventTDOAMeasurements:
    tdoa = exact_reference_tdoas(microphones, sources)
    return EventTDOAMeasurements(
        event_ids=np.arange(len(times)),
        receiver_event_times_s=times,
        microphone_ids=tuple(range(len(microphones))),
        microphone_pairs=tuple((0, index) for index in range(1, len(microphones))),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 2e-6),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )


def test_near_cross_reports_poor_metric_conditioning_before_exact_degeneracy() -> None:
    microphones = myotis_cross_microphones().copy()
    microphones[10, 0] += 2e-4
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    diagnostics = diagnose_planar_identifiability(factorization.receiver_factors)

    assert diagnostics.continuous_metric_nullity == 0
    assert diagnostics.metric_design.effective_rank == 5
    assert diagnostics.metric_design.condition_number > 1e3


def test_slightly_nonplanar_receivers_do_not_get_forced_into_planar_model() -> None:
    microphones = nondegenerate_planar_microphones().copy()
    microphones[:, 1] = np.linspace(-0.02, 0.02, len(microphones))
    times = np.linspace(0.2, 2.0, 20)
    sources = nondegenerate_planar_sources(times)
    measurements = _measurements(microphones, sources, times)

    result = compare_tdoa_models_8mic(measurements)
    assert result.comparison.status in {"receiver3d_source3d", "ambiguous"}
    assert result.comparison.status != "receiver2d_source3d"


def test_cross_constraint_ablation_restores_continuous_ambiguity() -> None:
    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="explicit_test_constraint",
    )

    constrained = upgrade_metric_planar(
        factorization,
        ranges,
        angle_constraint=constraint,
    )
    blind = upgrade_metric_planar(factorization, ranges)

    assert constrained.status == "solved"
    assert constrained.candidates
    assert constrained.diagnostics.identifiability.continuous_metric_nullity == 1
    assert constrained.diagnostics.constraints_applied == ("planar_arm_angle",)
    assert blind.status == "degenerate"
    assert not blind.candidates
    assert blind.diagnostics.identifiability.continuous_metric_nullity == 1


@pytest.mark.parametrize("event_count", [18, 40])
def test_cross_null_direction_is_independent_of_event_count(event_count: int) -> None:
    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, event_count))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    diagnostics = diagnose_planar_identifiability(factorization.receiver_factors)

    assert diagnostics.continuous_metric_nullity == 1
    assert diagnostics.metric_nullspace.shape == (5, 1)
