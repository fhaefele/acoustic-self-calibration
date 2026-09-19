import numpy as np

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
from acoustic_self_calibration.stratified.refinement import _objective, refine_calibration
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
