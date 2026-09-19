import numpy as np

from acoustic_self_calibration.simulation import (
    myotis_cross_microphones,
    myotis_cross_source_positions,
)
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint
from acoustic_self_calibration.stratified.factorization import factor_corrected_ranges
from acoustic_self_calibration.stratified.planar import upgrade_metric_planar


def _ranges(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )


def test_explicit_right_angle_constraint_resolves_cross_metric_family() -> None:
    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="synthetic_fixture_explicit_test_input",
    )

    unconstrained = upgrade_metric_planar(factorization, ranges)
    constrained = upgrade_metric_planar(
        factorization,
        ranges,
        angle_constraint=constraint,
    )

    assert unconstrained.status == "degenerate"
    assert not unconstrained.candidates
    assert constrained.status == "solved"
    assert constrained.candidates
    candidate = constrained.candidates[0]
    assert candidate.corrected_range_rms_m < 1e-9
    assert constrained.diagnostics.constraints_applied == ("planar_arm_angle",)
    assert constrained.diagnostics.constraint_residual_rad is not None
    assert abs(constrained.diagnostics.constraint_residual_rad) < 1e-9
    assert not np.any(candidate.source_height_sign_known)


def test_only_exact_right_angle_constraint_is_supported_in_milestone_d() -> None:
    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    ranges = _ranges(microphones, sources)
    factorization = factor_corrected_ranges(ranges, dimension=2)
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.deg2rad(80.0),
        provenance="test",
    )
    try:
        upgrade_metric_planar(
            factorization,
            ranges,
            angle_constraint=constraint,
        )
    except ValueError as error:
        assert "right angles" in str(error)
    else:
        raise AssertionError("unsupported non-right-angle constraint must be rejected")
