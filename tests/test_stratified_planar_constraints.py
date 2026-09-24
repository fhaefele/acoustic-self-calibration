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


def test_noisy_ranges_cannot_relax_explicit_exact_angle() -> None:
    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    ranges = _ranges(microphones, sources)
    ranges += np.random.default_rng(582).normal(scale=1e-4, size=ranges.shape)
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="explicit_noisy_range_test",
    )
    result = upgrade_metric_planar(
        factor_corrected_ranges(ranges, dimension=2),
        ranges,
        angle_constraint=constraint,
        acceptance_rms_m=1e-3,
    )
    assert result.candidates
    assert result.diagnostics.constraint_residual_rad is not None
    assert abs(result.diagnostics.constraint_residual_rad) < 1e-10


def test_noise_sensitivity_detects_cross_and_scales_with_arrival_noise() -> None:
    from acoustic_self_calibration.measurements import reference_star_from_arrivals
    from acoustic_self_calibration.simulation import planar_benchmark_microphones
    from acoustic_self_calibration.stratified.refinement import estimate_planar_noise_sensitivity

    rng = np.random.default_rng(187)
    sources = rng.uniform([-1.0, 1.0, -1.0], [1.0, 3.0, 1.0], size=(40, 3))
    for layout in ("cross", "star"):
        microphones = planar_benchmark_microphones(8, array_span_m=2.0, layout=layout)
        arrivals = np.linalg.norm(sources[:, None] - microphones[None, :], axis=2) / 343.0
        estimates = []
        for sigma in (1e-6, 2e-6):
            measurements = reference_star_from_arrivals(
                arrivals - arrivals[:, [0]],
                np.full_like(arrivals, sigma),
                receiver_event_times_s=np.arange(40, dtype=float),
                reference_microphone=0,
            )
            estimates.append(estimate_planar_noise_sensitivity(microphones, sources, measurements))
        if layout == "cross":
            assert estimates[0].observable_rank < estimates[0].parameter_count
            assert np.isinf(estimates[0].weakest_microphone_rms_std_m)
        else:
            assert estimates[0].observable_rank == estimates[0].parameter_count
            assert estimates[0].relative_weakest_std < 0.1
            assert np.isclose(
                estimates[1].weakest_microphone_rms_std_m,
                2.0 * estimates[0].weakest_microphone_rms_std_m,
            )


def test_noise_sensitivity_retains_weak_near_cross_mode() -> None:
    from acoustic_self_calibration.measurements import reference_star_from_arrivals
    from acoustic_self_calibration.simulation import planar_benchmark_microphones
    from acoustic_self_calibration.stratified.refinement import estimate_planar_noise_sensitivity

    rng = np.random.default_rng(781)
    microphones = planar_benchmark_microphones(12, array_span_m=2.0, layout="cross")
    microphones[:, (0, 2)] += rng.normal(0.0, 1e-4, (12, 2))
    sources = rng.uniform([-1.0, 1.0, -1.0], [1.0, 3.0, 1.0], size=(40, 3))
    arrivals = np.linalg.norm(sources[:, None] - microphones[None, :], axis=2) / 343.0
    measurements = reference_star_from_arrivals(
        arrivals - arrivals[:, [0]],
        np.full_like(arrivals, 2e-6),
        receiver_event_times_s=np.arange(40, dtype=float),
        reference_microphone=0,
    )
    sensitivity = estimate_planar_noise_sensitivity(microphones, sources, measurements)
    assert sensitivity.observable_rank == sensitivity.parameter_count
    assert sensitivity.relative_weakest_std > 0.1


def test_solved_cross_is_downgraded_but_known_angle_is_not_assessed() -> None:
    from dataclasses import replace

    from acoustic_self_calibration.measurements import reference_star_from_arrivals
    from acoustic_self_calibration.stratified.solver import (
        PlanarCalibrationResult,
        StratifiedCalibrationDiagnostics,
        _assess_planar_noise_sensitivity,
    )

    microphones = myotis_cross_microphones()
    sources = myotis_cross_source_positions(np.linspace(0.25, 2.05, 18))
    arrivals = _ranges(microphones, sources).T / 343.0
    measurements = reference_star_from_arrivals(
        arrivals - arrivals[:, [0]],
        np.full_like(arrivals, 2e-6),
        receiver_event_times_s=np.arange(len(sources), dtype=float),
        reference_microphone=0,
    )
    result = PlanarCalibrationResult(
        status="solved",
        microphone_positions_m=microphones,
        source_projected_positions_m=None,
        source_unsigned_heights_m=None,
        source_height_sign_known=None,
        source_representative_positions_m=sources,
        event_ids=measurements.event_ids,
        validation=None,
        tdoa_rms_s=0.0,
        diagnostics=StratifiedCalibrationDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, ()),
    )
    assessed = _assess_planar_noise_sensitivity(result, measurements, 343.0)
    assert assessed.status == "weakly_identified"
    assert "planar_geometry_noise_sensitive" in assessed.diagnostics.rejection_reasons
    constrained = replace(
        result,
        angle_constraint=PlanarAngleConstraint(
            center_receiver=3,
            arm_a_receiver=0,
            arm_b_receiver=7,
            angle_rad=np.pi / 2.0,
            provenance="explicit_test_input",
        ),
    )
    assert _assess_planar_noise_sensitivity(constrained, measurements, 343.0) is constrained


def test_noise_sensitivity_inflates_uncertainty_when_fit_exceeds_declared_noise() -> None:
    from dataclasses import replace

    from acoustic_self_calibration.measurements import reference_star_from_arrivals
    from acoustic_self_calibration.simulation import planar_benchmark_microphones
    from acoustic_self_calibration.stratified.refinement import estimate_planar_noise_sensitivity

    rng = np.random.default_rng(187)
    microphones = planar_benchmark_microphones(8, array_span_m=2.0, layout="star")
    sources = rng.uniform([-1.0, 1.0, -1.0], [1.0, 3.0, 1.0], size=(40, 3))
    arrivals = np.linalg.norm(sources[:, None] - microphones[None, :], axis=2) / 343.0
    measurements = reference_star_from_arrivals(
        arrivals - arrivals[:, [0]],
        np.full_like(arrivals, 2e-6),
        receiver_event_times_s=np.arange(40, dtype=float),
        reference_microphone=0,
    )
    exact = estimate_planar_noise_sensitivity(microphones, sources, measurements)
    poor_measurements = replace(
        measurements,
        tdoa_s=measurements.tdoa_s + rng.normal(0.0, 20e-6, measurements.tdoa_s.shape),
    )
    poor = estimate_planar_noise_sensitivity(microphones, sources, poor_measurements)
    assert exact.residual_degrees_of_freedom == 40 * 7 - 40 * 3 - (2 * 8 - 3)
    assert exact.fit_p_value == 1.0
    assert exact.noise_scale_inflation == 1.0
    assert poor.fit_p_value is not None and poor.fit_p_value < 0.001
    assert poor.noise_scale_inflation > 2.0
    assert np.isclose(
        poor.weakest_microphone_rms_std_m,
        exact.weakest_microphone_rms_std_m * poor.noise_scale_inflation,
    )
