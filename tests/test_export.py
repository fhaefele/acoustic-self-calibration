import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from acoustic_self_calibration.events import EventDetection
from acoustic_self_calibration.export import (
    calibration_result_to_dict,
    write_calibration_outputs,
)
from acoustic_self_calibration.ground_truth import (
    GroundTruth,
    validate_ground_truth_json,
)
from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.pipeline import AudioCalibrationResult
from acoustic_self_calibration.stratified.hypotheses import (
    FullGeometryHypothesis,
    HypothesisClass,
    MeasurementSplit,
)
from acoustic_self_calibration.stratified.refinement import RefinementDiagnostics
from acoustic_self_calibration.stratified.robustness import RobustResidualSummary
from acoustic_self_calibration.stratified.solver import (
    CalibrationStatus,
    PlanarCalibrationResult,
    StratifiedCalibrationDiagnostics,
    StratifiedCalibrationResult,
)


def _audio_result(status: CalibrationStatus = "solved") -> AudioCalibrationResult:
    event_count = 12
    sample_rate = 48_000
    event_samples = 1000 + np.arange(event_count) * 4000
    event_times = event_samples / sample_rate
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 1.0, 0.1],
            [0.1, 0.3, 1.2],
            [1.1, 0.7, 0.4],
            [-0.5, 0.8, 0.6],
            [0.7, -0.6, 1.0],
            [-0.8, -0.4, 0.3],
        ]
    )
    source = np.column_stack(
        [
            np.linspace(1.4, -0.5, event_count),
            np.linspace(-0.4, 1.2, event_count),
            0.8 + 0.2 * np.sin(np.linspace(0.0, 2.0, event_count)),
        ]
    )
    shape = (event_count, 7)
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(event_count),
        receiver_event_times_s=event_times,
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=np.zeros(shape),
        sigma_s=np.full(shape, 2e-6),
        confidence=np.ones(shape),
        valid=np.ones(shape, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    detection = EventDetection(
        event_samples=event_samples,
        receiver_event_times_s=event_times,
        event_channel=0,
        prominence=np.ones(event_count),
    )

    generation = np.zeros((8, event_count), dtype=bool)
    completion = np.zeros_like(generation)
    validation = np.zeros_like(generation)
    generation[1:7, :4] = True
    completion[1:5, 4:8] = True
    validation[5:8, 8:12] = True
    split = MeasurementSplit(generation, completion, validation)
    summary = RobustResidualSummary(
        independent_coordinate_count=12,
        rms_s=3e-6,
        median_abs_s=2e-6,
        max_abs_s=8e-6,
        normalized_huber_score=0.7,
        inlier_fraction=0.92,
    )
    hypothesis = FullGeometryHypothesis(
        subset_id="r0-6-e0-5",
        root_id=1,
        microphone_positions_m=microphones,
        source_positions_m=source,
        split=split,
        validation=summary,
        full_tdoa_rms_s=4e-6,
        metric_rms_m=2e-4,
        metric_condition_number=42.0,
        affine_factor_singular_ratio=0.12,
        model="receiver3d_source3d",
        seed_microphone_ids=(0, 1, 2, 3, 4, 5, 6),
        seed_event_ids=(0, 1, 2, 3, 4, 5),
        metric_branch_id=2,
        arrival_gauge_shifts_m=np.zeros(event_count),
        offset_scope_event_ids=tuple(range(8)),
        microphone_completion_status=np.ones(8, dtype=bool),
        source_completion_status=np.ones(event_count, dtype=bool),
        generating_equation_rms=1e-12,
    )
    class_one = HypothesisClass(
        representative=hypothesis,
        members=(hypothesis,),
        independent_subset_support=2,
    )
    classes = (class_one,)
    if status == "ambiguous":
        classes = (class_one, class_one)

    diagnostics = StratifiedCalibrationDiagnostics(
        attempted_subsets=4,
        generated_offset_roots=7,
        metric_candidates=3,
        completed_hypotheses=2,
        geometric_class_count=len(classes),
        selected_support=2,
        fitting_event_count=8,
        validation_event_count=4,
        validation_independent_coordinates=12,
        rejection_reasons=("one_rejected_root",),
    )
    calibration = StratifiedCalibrationResult(
        status=status,
        microphone_positions_m=None if status == "degenerate" else microphones,
        source_positions_m=None if status == "degenerate" else source,
        event_ids=np.arange(event_count),
        tdoa_rms_s=None if status == "degenerate" else 4e-6,
        selected_class=None if status == "degenerate" else class_one,
        classes=() if status == "degenerate" else classes,
        diagnostics=diagnostics,
    )
    return AudioCalibrationResult(
        calibration=calibration,
        measurements=measurements,
        detection=detection,
        speed_of_sound_mps=343.0,
        model="general_3d",
        temporal_tracking_enabled=True,
        refinement_mode="none",
    )


def test_stratified_json_contains_diagnostics_and_lineage() -> None:
    result = _audio_result()
    event_count = len(result.event_times_s)
    document = calibration_result_to_dict(result)

    assert document["calibration"]["backend"] == "stratified_tdoa"
    assert document["calibration"]["status"] == "solved"
    assert document["calibration"]["refinement"]["applied"] is False
    assert document["measurements"]["measurement_basis"] == "reference_star"
    selected = document["diagnostics"]["selected_hypothesis"]
    assert selected["conditioning"]["metric_condition_number"] == 42.0
    assert selected["model"] == "receiver3d_source3d"
    assert selected["seed_microphone_ids"] == [0, 1, 2, 3, 4, 5, 6]
    assert selected["seed_event_ids"] == [0, 1, 2, 3, 4, 5]
    assert selected["metric_branch_id"] == 2
    assert selected["offset_scope_event_ids"] == list(range(8))
    assert selected["state_completion"]["microphones"] == [True] * 8
    assert selected["state_completion"]["sources"] == [True] * event_count
    assert selected["generating_equation_rms"] == 1e-12
    assert selected["lineage"]["generation_count"] > 0
    assert selected["lineage"]["completion_count"] > 0
    assert selected["lineage"]["validation_count"] > 0
    assert selected["validation"]["inlier_fraction"] == 0.92
    json.dumps(document, allow_nan=False)


def test_ambiguous_and_degenerate_statuses_serialize_without_fake_geometry() -> None:
    ambiguous = calibration_result_to_dict(_audio_result("ambiguous"))
    degenerate = calibration_result_to_dict(_audio_result("degenerate"))

    assert ambiguous["calibration"]["status"] == "ambiguous"
    assert ambiguous["diagnostics"]["ambiguity"]["class_count"] == 2
    assert degenerate["calibration"]["status"] == "degenerate"
    assert degenerate["scene"]["microphones"]["positions_m"] is None
    assert degenerate["scene"]["source"]["positions_m"] is None
    assert degenerate["diagnostics"]["selected_hypothesis"] is None
    json.dumps(ambiguous, allow_nan=False)
    json.dumps(degenerate, allow_nan=False)


def test_write_outputs_and_reuse_solved_json_as_reference(tmp_path: Path) -> None:
    result = _audio_result()
    paths = write_calibration_outputs(
        result,
        tmp_path / "calibration",
        input_wav_path="recording.wav",
        settings={"event_min_gap_s": 0.05},
    )

    assert paths.json.exists()
    assert paths.figure.exists()
    assert paths.figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    reference = validate_ground_truth_json(paths.json)
    assert reference.scene_role == "estimate"
    assert result.microphone_positions_m is not None
    assert np.allclose(reference.microphone_positions_m, result.microphone_positions_m)


def test_write_outputs_adds_ground_truth_evaluation(tmp_path: Path) -> None:
    result = _audio_result()
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
    truth = GroundTruth(
        microphone_positions_m=result.microphone_positions_m.copy(),
        source_times_s=result.event_times_s.copy(),
        source_positions_m=result.source_positions_m.copy(),
        metadata={"name": "exact"},
    )
    paths = write_calibration_outputs(
        result,
        tmp_path / "with_gt",
        ground_truth=truth,
    )
    document = json.loads(paths.json.read_text(encoding="utf-8"))
    assert document["evaluation"]["microphones"]["rms_error_m"] < 1e-12
    assert document["evaluation"]["source"]["rms_error_m"] < 1e-12


def test_planar_json_exports_unsigned_height_semantics() -> None:
    base = _audio_result()
    event_count = len(base.event_times_s)
    projected = np.column_stack(
        [
            np.linspace(-0.5, 0.5, event_count),
            np.linspace(0.2, 0.8, event_count),
        ]
    )
    heights = np.linspace(0.6, 1.2, event_count)
    representative = np.column_stack([projected, heights])
    planar = PlanarCalibrationResult(
        status="solved",
        microphone_positions_m=base.microphone_positions_m,
        source_projected_positions_m=projected,
        source_unsigned_heights_m=heights,
        source_height_sign_known=np.zeros(event_count, dtype=bool),
        source_representative_positions_m=representative,
        event_ids=base.measurements.event_ids,
        validation=None,
        tdoa_rms_s=5e-6,
        diagnostics=base.calibration.diagnostics,
    )
    result = AudioCalibrationResult(
        calibration=planar,
        measurements=base.measurements,
        detection=base.detection,
        speed_of_sound_mps=343.0,
        model="receiver2d_source3d",
        temporal_tracking_enabled=True,
        refinement_mode="none",
    )
    document = calibration_result_to_dict(result)

    source = document["scene"]["source"]
    assert source["representation"] == "positive_plane_normal_representative"
    assert source["projected_positions_m"] == projected.tolist()
    assert source["unsigned_height_m"] == heights.tolist()
    assert source["height_sign_known"] == [False] * event_count
    assert document["calibration"]["model"] == "receiver2d_source3d"


def test_planar_continuous_ambiguity_is_exported() -> None:
    base = _audio_result()
    nullspace = np.arange(5, dtype=float).reshape(5, 1)
    planar = PlanarCalibrationResult(
        status="degenerate",
        microphone_positions_m=None,
        source_projected_positions_m=None,
        source_unsigned_heights_m=None,
        source_height_sign_known=None,
        source_representative_positions_m=None,
        event_ids=base.measurements.event_ids,
        validation=None,
        tdoa_rms_s=None,
        diagnostics=base.calibration.diagnostics,
        continuous_ambiguity_dimension=1,
        continuous_ambiguity_nullspace=nullspace,
    )
    result = AudioCalibrationResult(
        calibration=planar,
        measurements=base.measurements,
        detection=base.detection,
        speed_of_sound_mps=343.0,
        model="receiver2d_source3d",
        temporal_tracking_enabled=True,
        refinement_mode="none",
    )

    document = calibration_result_to_dict(result)
    ambiguity = document["diagnostics"]["ambiguity"]
    assert ambiguity["continuous_family_dimension"] == 1
    assert ambiguity["continuous_family_null_directions"] == nullspace.tolist()


def test_refinement_json_preserves_pre_and_post_state() -> None:
    base = _audio_result()
    pre = base.calibration
    assert isinstance(pre, StratifiedCalibrationResult)
    assert pre.microphone_positions_m is not None
    refined_microphones = np.array(pre.microphone_positions_m, copy=True)
    refined_microphones[1, 0] += 1e-4
    refined = replace(
        pre,
        microphone_positions_m=refined_microphones,
        tdoa_rms_s=3e-6,
    )
    diagnostics = RefinementDiagnostics(
        mode="wls",
        attempted=True,
        accepted=True,
        initial_objective=10.0,
        final_objective=7.0,
        initial_tdoa_rms_s=4e-6,
        final_tdoa_rms_s=3e-6,
        independent_coordinate_count=24,
        nfev=12,
        termination="ftol reached",
        reason="accepted",
        max_nfev=80,
        improvement_tolerance=1e-8,
    )
    result = replace(
        base,
        calibration=refined,
        refinement_mode="wls",
        pre_refinement_calibration=pre,
        refinement_diagnostics=diagnostics,
    )

    document = calibration_result_to_dict(result)
    payload = document["calibration"]["refinement"]
    assert payload["mode"] == "wls"
    assert payload["attempted"] is True
    assert payload["applied"] is True
    assert payload["initial_objective"] == 10.0
    assert payload["final_objective"] == 7.0
    assert payload["max_nfev"] == 80
    assert payload["improvement_tolerance"] == 1e-8
    assert payload["termination"] == "ftol reached"
    assert payload["frozen_validation_score"] == 0.7
    assert payload["pre_refinement_scene"]["microphones"]["positions_m"] == (
        pre.microphone_positions_m.tolist()
    )
    assert payload["post_refinement_scene"]["microphones"]["positions_m"] == (
        refined_microphones.tolist()
    )
