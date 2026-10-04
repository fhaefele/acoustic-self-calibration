from dataclasses import replace

import numpy as np
import pytest

from acoustic_self_calibration.geometry import canonicalize_scene_conditioned
from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    make_random_3d_pulse_scene,
    myotis_cross_microphones,
    myotis_cross_source_positions,
    nondegenerate_planar_microphones,
    nondegenerate_planar_sources,
)
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint
from acoustic_self_calibration.stratified.refinement import (
    _MeasurementObjective,
    _objective,
    refine_calibration,
)
from acoustic_self_calibration.stratified.solver import (
    PlanarCalibrationResult,
    StratifiedCalibrationDiagnostics,
    StratifiedCalibrationResult,
)


def _diagnostics() -> StratifiedCalibrationDiagnostics:
    return StratifiedCalibrationDiagnostics(
        attempted_subsets=1,
        generated_offset_roots=1,
        metric_candidates=1,
        completed_hypotheses=1,
        geometric_class_count=1,
        selected_support=1,
        fitting_event_count=12,
        validation_event_count=8,
        validation_independent_coordinates=8,
        rejection_reasons=(),
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


@pytest.mark.parametrize("dimension", [2, 3])
def test_weighted_geometry_jacobian_matches_finite_differences(dimension: int) -> None:
    rng = np.random.default_rng(433)
    microphones = rng.normal(size=(6, 3))
    if dimension == 2:
        microphones[:, 2] = 0.0
    sources = rng.normal(size=(5, 3)) + [0.0, 0.0, 3.0]
    measurements = _measurements(microphones, sources, np.arange(5, dtype=float))
    valid = measurements.valid.copy()
    valid[1, 2] = False
    covariance = np.broadcast_to(2e-12 * (np.eye(5) + np.ones((5, 5))), (5, 5, 5))
    measurements = replace(
        measurements,
        valid=valid,
        tdoa_s=np.where(valid, measurements.tdoa_s, np.nan),
        sigma_s=np.where(valid, measurements.sigma_s, np.nan),
        confidence=np.where(valid, measurements.confidence, np.nan),
        covariance_s2=covariance,
        covariance_model="full",
    )
    objective = _MeasurementObjective(measurements, 343.0)
    free = np.arange(dimension, microphones.shape[0] * dimension)
    analytic = objective.jacobian(microphones, sources, free, microphone_dimension=dimension)
    numerical = np.empty_like(analytic)
    step = 1e-6
    for column in range(len(free) + sources.size):
        mic_plus, mic_minus = microphones.copy(), microphones.copy()
        src_plus, src_minus = sources.copy(), sources.copy()
        if column < len(free):
            receiver, axis = divmod(int(free[column]), dimension)
            mic_plus[receiver, axis] += step
            mic_minus[receiver, axis] -= step
        else:
            src_plus.flat[column - len(free)] += step
            src_minus.flat[column - len(free)] -= step
        numerical[:, column] = (
            objective.residual(mic_plus, src_plus) - objective.residual(mic_minus, src_minus)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=1e-6)


def test_wls_refinement_improves_nearby_3d_geometry() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
    measurements = _measurements(
        scene.microphone_positions_m,
        scene.source_positions_at_events_m,
        scene.event_times_s,
    )
    rng = np.random.default_rng(91)
    initial_microphones = scene.microphone_positions_m + rng.normal(
        scale=0.008, size=scene.microphone_positions_m.shape
    )
    initial_sources = scene.source_positions_at_events_m + rng.normal(
        scale=0.008, size=scene.source_positions_at_events_m.shape
    )
    calibration = StratifiedCalibrationResult(
        status="solved",
        microphone_positions_m=initial_microphones,
        source_positions_m=initial_sources,
        event_ids=measurements.event_ids,
        tdoa_rms_s=None,
        selected_class=None,
        classes=(),
        diagnostics=_diagnostics(),
        microphone_ids=measurements.microphone_ids,
    )

    refined, diagnostics = refine_calibration(
        calibration,
        measurements,
        speed_of_sound=343.0,
        mode="wls",
        max_nfev=100,
        improvement_tolerance=0.0,
    )

    assert diagnostics.attempted
    assert diagnostics.accepted
    assert diagnostics.max_nfev == 100
    assert diagnostics.improvement_tolerance == 0.0
    assert diagnostics.initial_tdoa_rms_s is not None
    assert diagnostics.final_tdoa_rms_s is not None
    assert diagnostics.final_tdoa_rms_s < diagnostics.initial_tdoa_rms_s
    assert refined.tdoa_rms_s == diagnostics.final_tdoa_rms_s


def test_planar_refinement_preserves_unsigned_height_contract() -> None:
    microphones = nondegenerate_planar_microphones()
    times = np.linspace(0.2, 2.0, 20)
    sources = nondegenerate_planar_sources(times)
    canonical_microphones, canonical_sources, _ = canonicalize_scene_conditioned(
        microphones,
        sources,
    )
    assert canonical_sources is not None
    measurements = _measurements(canonical_microphones, canonical_sources, times)
    rng = np.random.default_rng(92)
    initial_microphones = np.array(canonical_microphones, copy=True)
    initial_microphones[:, :2] += rng.normal(
        scale=0.004,
        size=(len(initial_microphones), 2),
    )
    initial_microphones[:, 2] = 0.0
    representative = canonical_sources + rng.normal(
        scale=0.004,
        size=canonical_sources.shape,
    )
    representative[:, 2] = np.abs(representative[:, 2])
    calibration = PlanarCalibrationResult(
        status="solved",
        microphone_positions_m=initial_microphones,
        source_projected_positions_m=representative[:, :2],
        source_unsigned_heights_m=np.abs(representative[:, 2]),
        source_height_sign_known=np.zeros(len(times), dtype=bool),
        source_representative_positions_m=representative,
        event_ids=measurements.event_ids,
        validation=None,
        tdoa_rms_s=None,
        diagnostics=_diagnostics(),
        microphone_ids=measurements.microphone_ids,
    )

    refined, diagnostics = refine_calibration(
        calibration,
        measurements,
        speed_of_sound=343.0,
        mode="huber",
        max_nfev=100,
        improvement_tolerance=0.0,
    )

    assert diagnostics.attempted
    assert diagnostics.accepted
    assert isinstance(refined, PlanarCalibrationResult)
    assert refined.source_unsigned_heights_m is not None
    assert np.all(refined.source_unsigned_heights_m >= 0.0)
    assert refined.source_height_sign_known is not None
    assert not np.any(refined.source_height_sign_known)
    assert diagnostics.final_tdoa_rms_s is not None
    assert diagnostics.initial_tdoa_rms_s is not None
    assert diagnostics.final_tdoa_rms_s < diagnostics.initial_tdoa_rms_s


def test_huber_objective_matches_robust_loss_not_squared_loss() -> None:
    residual = np.array([0.25, 10.0])
    assert _objective(residual, mode="huber") < _objective(residual, mode="wls")


def test_planar_refinement_keeps_explicit_angle_anchors_exact() -> None:
    microphones = myotis_cross_microphones()
    times = np.linspace(0.25, 2.05, 18)
    sources = myotis_cross_source_positions(times)
    canonical_microphones, canonical_sources, _ = canonicalize_scene_conditioned(
        microphones,
        sources,
    )
    assert canonical_sources is not None
    measurements = _measurements(canonical_microphones, canonical_sources, times)
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="explicit_refinement_test",
    )
    rng = np.random.default_rng(93)
    initial_microphones = np.array(canonical_microphones, copy=True)
    free = [index for index in range(len(canonical_microphones)) if index not in {0, 3, 7}]
    initial_microphones[np.ix_(free, [0, 1])] += rng.normal(
        scale=0.002,
        size=(len(free), 2),
    )
    representative = np.array(canonical_sources, copy=True)
    representative += rng.normal(scale=0.002, size=representative.shape)
    representative[:, 2] = np.abs(representative[:, 2])

    calibration = PlanarCalibrationResult(
        status="solved",
        microphone_positions_m=initial_microphones,
        source_projected_positions_m=representative[:, :2],
        source_unsigned_heights_m=np.abs(representative[:, 2]),
        source_height_sign_known=np.zeros(len(times), dtype=bool),
        source_representative_positions_m=representative,
        event_ids=measurements.event_ids,
        validation=None,
        tdoa_rms_s=None,
        diagnostics=_diagnostics(),
        microphone_ids=measurements.microphone_ids,
        angle_constraint=constraint,
    )
    anchors_before = np.array(initial_microphones[[3, 0, 7]], copy=True)

    refined, diagnostics = refine_calibration(
        calibration,
        measurements,
        speed_of_sound=343.0,
        mode="wls",
        max_nfev=100,
        improvement_tolerance=0.0,
    )

    assert diagnostics.attempted
    assert diagnostics.accepted
    assert isinstance(refined, PlanarCalibrationResult)
    assert refined.microphone_positions_m is not None
    assert np.allclose(
        refined.microphone_positions_m[[3, 0, 7]],
        anchors_before,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("sigma_s", [2e-6, 40e-6])
@pytest.mark.parametrize("status", ["solved", "ambiguous"])
def test_expanded_geometry_joint_fit_preserves_status(
    monkeypatch: pytest.MonkeyPatch, sigma_s: float, status
) -> None:
    from acoustic_self_calibration.stratified import refinement, solver

    rng = np.random.default_rng(831)
    microphones = rng.uniform(-2.0, 2.0, (12, 3))
    sources = rng.uniform(-3.0, 3.0, (30, 3))
    measurements = _measurements(microphones, sources, np.arange(30, dtype=float))
    measurements = replace(measurements, sigma_s=np.full_like(measurements.tdoa_s, sigma_s))
    coarse_microphones = microphones[:8] + rng.normal(0.0, 0.005, (8, 3))
    coarse_sources = sources + rng.normal(0.0, 0.005, sources.shape)
    core = StratifiedCalibrationResult(
        status=status,
        microphone_positions_m=coarse_microphones,
        source_positions_m=coarse_sources,
        event_ids=measurements.event_ids,
        tdoa_rms_s=1e-4,
        selected_class=None,
        classes=(),
        diagnostics=_diagnostics(),
    )
    monkeypatch.setattr(solver, "calibrate_tdoa_8mic", lambda *args, **kwargs: core)
    monkeypatch.setattr(
        solver,
        "_normalize_reference_star_measurements",
        lambda value: (value, tuple(range(12))),
    )
    initial_rms = []
    actual_refine = refinement.refine_calibration

    def capture_refinement(calibration, values, **kwargs):
        initial_rms.append(calibration.tdoa_rms_s)
        assert values is measurements
        return actual_refine(calibration, values, **kwargs)

    monkeypatch.setattr(refinement, "refine_calibration", capture_refinement)
    result = solver.calibrate_tdoa(measurements)
    assert result.status == status
    assert result.microphone_positions_m is not None
    assert result.tdoa_rms_s is not None
    assert result.diagnostics.extra_microphones_completed == 4
    if sigma_s > 5e-3 / 343.0 and status == "solved":
        assert len(initial_rms) == 1
        assert result.tdoa_rms_s < initial_rms[0] * 0.1
        assert np.linalg.norm(result.microphone_positions_m[:8] - coarse_microphones) > 1e-4
    else:
        assert len(initial_rms) == int(sigma_s > 5e-3 / 343.0)
        assert np.array_equal(result.microphone_positions_m[:8], coarse_microphones)


@pytest.mark.parametrize("first_support", [1, 3])
def test_tied_geometry_search_checks_beyond_ineligible_or_duplicate_runner(
    first_support: int,
) -> None:
    from acoustic_self_calibration.stratified.solver import _has_distinct_tied_geometry

    microphones = np.random.default_rng(89).normal(size=(8, 3))
    assert _has_distinct_tied_geometry(
        microphones,
        1e-5,
        3,
        [(1.01e-5, first_support, microphones.copy()), (1.02e-5, 3, 2 * microphones)],
        score_tolerance=0.05,
    )
    assert not _has_distinct_tied_geometry(
        microphones,
        1e-5,
        3,
        [(1.01e-5, first_support, microphones.copy()), (1.06e-5, 3, 2 * microphones)],
        score_tolerance=0.05,
    )
