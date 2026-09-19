from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import evaluate_against_ground_truth, evaluate_scenes, evaluation_to_dict
from .ground_truth import GroundTruth, ground_truth_to_dict, load_ground_truth_json
from .pipeline import AudioCalibrationResult
from .stratified.solver import PlanarCalibrationResult
from .visualization import plot_calibration_comparison, plot_scene_comparison


@dataclass(frozen=True)
class CalibrationOutputPaths:
    """JSON result and multi-panel figure written for one solved operation."""

    json: Path
    figure: Path


def _output_paths(output_prefix: str | Path) -> CalibrationOutputPaths:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return CalibrationOutputPaths(
        json=prefix.with_suffix(".json"),
        figure=prefix.with_suffix(".png"),
    )


def _json_array(value: np.ndarray | None) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        objects = array.astype(object)
        objects[~np.isfinite(array)] = None
        return objects.tolist()
    return array.tolist()


def _calibration_scene_payload(
    calibration: Any,
) -> dict[str, Any] | None:
    """Serialize calibration coordinates for refinement rollback/audit."""
    microphones = getattr(calibration, "microphone_positions_m", None)
    if microphones is None:
        return None
    if isinstance(calibration, PlanarCalibrationResult):
        source = {
            "positions_m": _json_array(calibration.source_representative_positions_m),
            "projected_positions_m": _json_array(calibration.source_projected_positions_m),
            "unsigned_height_m": _json_array(calibration.source_unsigned_heights_m),
            "height_sign_known": _json_array(calibration.source_height_sign_known),
        }
    else:
        source = {
            "positions_m": _json_array(getattr(calibration, "source_positions_m", None)),
        }
    return {
        "microphones": {"positions_m": _json_array(microphones)},
        "source": source,
    }


def _frozen_validation_score(calibration: Any) -> float | None:
    if isinstance(calibration, PlanarCalibrationResult):
        return (
            None
            if calibration.validation is None
            else calibration.validation.normalized_huber_score
        )
    selected_class = getattr(calibration, "selected_class", None)
    if selected_class is None:
        return None
    return selected_class.representative.validation.normalized_huber_score


def _refinement_dict(
    result: AudioCalibrationResult,
) -> dict[str, Any]:
    diagnostics = result.refinement_diagnostics
    pre = result.pre_refinement_calibration
    return {
        "mode": result.refinement_mode,
        "attempted": False if diagnostics is None else diagnostics.attempted,
        "applied": False if diagnostics is None else diagnostics.accepted,
        "reason": None if diagnostics is None else diagnostics.reason,
        "initial_objective": (None if diagnostics is None else diagnostics.initial_objective),
        "final_objective": (None if diagnostics is None else diagnostics.final_objective),
        "initial_tdoa_rms_s": (None if diagnostics is None else diagnostics.initial_tdoa_rms_s),
        "final_tdoa_rms_s": (None if diagnostics is None else diagnostics.final_tdoa_rms_s),
        "independent_coordinate_count": (
            0 if diagnostics is None else diagnostics.independent_coordinate_count
        ),
        "nfev": 0 if diagnostics is None else diagnostics.nfev,
        "max_nfev": 0 if diagnostics is None else diagnostics.max_nfev,
        "improvement_tolerance": (
            None if diagnostics is None else diagnostics.improvement_tolerance
        ),
        "termination": None if diagnostics is None else diagnostics.termination,
        "frozen_validation_score": _frozen_validation_score(
            pre if pre is not None else result.calibration
        ),
        "pre_refinement_scene": _calibration_scene_payload(pre),
        "post_refinement_scene": (
            None
            if diagnostics is None or not diagnostics.attempted
            else _calibration_scene_payload(result.calibration)
        ),
    }


def _selected_hypothesis_dict(result: AudioCalibrationResult) -> dict[str, Any] | None:
    if isinstance(result.calibration, PlanarCalibrationResult):
        return None
    selected_class = result.calibration.selected_class
    if selected_class is None:
        return None
    hypothesis = selected_class.representative
    split = hypothesis.split
    return {
        "subset_id": hypothesis.subset_id,
        "root_id": hypothesis.root_id,
        "model": hypothesis.model,
        "seed_microphone_ids": list(hypothesis.seed_microphone_ids),
        "seed_event_ids": list(hypothesis.seed_event_ids),
        "metric_branch_id": hypothesis.metric_branch_id,
        "arrival_gauge_shifts_m": _json_array(hypothesis.arrival_gauge_shifts_m),
        "offset_scope_event_ids": list(hypothesis.offset_scope_event_ids),
        "state_completion": {
            "microphones": _json_array(hypothesis.microphone_completion_status),
            "sources": _json_array(hypothesis.source_completion_status),
        },
        "generating_equation_rms": hypothesis.generating_equation_rms,
        "class_support": selected_class.independent_subset_support,
        "class_member_count": len(selected_class.members),
        "full_tdoa_rms_s": hypothesis.full_tdoa_rms_s,
        "metric_rms_m": hypothesis.metric_rms_m,
        "conditioning": {
            "metric_condition_number": hypothesis.metric_condition_number,
            "affine_factor_singular_ratio": hypothesis.affine_factor_singular_ratio,
        },
        "validation": {
            "independent_coordinate_count": (hypothesis.validation.independent_coordinate_count),
            "rms_s": hypothesis.validation.rms_s,
            "median_abs_s": hypothesis.validation.median_abs_s,
            "max_abs_s": hypothesis.validation.max_abs_s,
            "normalized_huber_score": hypothesis.validation.normalized_huber_score,
            "inlier_fraction": hypothesis.validation.inlier_fraction,
        },
        "lineage": {
            "generation_count": int(np.sum(split.generation_mask)),
            "completion_count": int(np.sum(split.completion_mask)),
            "validation_count": int(np.sum(split.validation_mask)),
        },
    }


def calibration_result_to_dict(
    result: AudioCalibrationResult,
    *,
    input_wav_path: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    ground_truth: GroundTruth | None = None,
) -> dict[str, Any]:
    """Build deterministic JSON for solved, ambiguous, or failed stratified results."""
    calibration = result.calibration
    source_scene: dict[str, Any] = {
        "times_s": _json_array(result.event_times_s),
        "positions_m": _json_array(result.source_positions_m),
        "std_m": None,
    }
    if isinstance(result.calibration, PlanarCalibrationResult):
        source_scene.update(
            {
                "representation": "positive_plane_normal_representative",
                "projected_positions_m": _json_array(result.source_projected_positions_m),
                "unsigned_height_m": _json_array(result.source_unsigned_heights_m),
                "height_sign_known": _json_array(result.source_height_sign_known),
            }
        )
    scene = {
        "microphones": {
            "positions_m": _json_array(result.microphone_positions_m),
            "std_m": None,
        },
        "source": source_scene,
    }
    diagnostics = calibration.diagnostics
    selected = _selected_hypothesis_dict(result)
    classes = () if isinstance(calibration, PlanarCalibrationResult) else calibration.classes
    document: dict[str, Any] = {
        "schema_version": 1,
        "scene_role": "estimate",
        "scene": scene,
        "input": {
            "wav_path": None if input_wav_path is None else str(input_wav_path),
        },
        "settings": {} if settings is None else settings,
        "calibration": {
            "backend": "stratified_tdoa",
            "model": result.model,
            "status": calibration.status,
            "speed_of_sound_mps": result.speed_of_sound_mps,
            "rms_tdoa_residual_s": calibration.tdoa_rms_s,
            "temporal_tracking_enabled": result.temporal_tracking_enabled,
            "refinement": _refinement_dict(result),
        },
        "measurements": {
            "event_ids": _json_array(result.measurements.event_ids),
            "receiver_event_times_s": _json_array(result.event_times_s),
            "event_samples": _json_array(result.event_samples),
            "event_channel": result.event_channel,
            "detected_event_count": result.detected_event_count,
            "microphone_pairs": [list(pair) for pair in result.microphone_pairs],
            "measurement_basis": result.measurements.measurement_basis,
            "measurement_origin": result.measurements.measurement_origin,
            "covariance_model": result.measurements.covariance_model,
            "tdoa_s": _json_array(result.tdoa_s),
            "tdoa_sigma_s": _json_array(result.tdoa_sigma_s),
            "confidence": _json_array(result.confidence),
            "valid": _json_array(result.measurements.valid),
            "valid_count": int(np.sum(result.measurements.valid)),
            "total_count": int(result.measurements.valid.size),
        },
        "diagnostics": {
            "attempted_subsets": diagnostics.attempted_subsets,
            "generated_offset_roots": diagnostics.generated_offset_roots,
            "metric_candidates": diagnostics.metric_candidates,
            "completed_hypotheses": diagnostics.completed_hypotheses,
            "geometric_class_count": diagnostics.geometric_class_count,
            "selected_support": diagnostics.selected_support,
            "fitting_event_count": diagnostics.fitting_event_count,
            "validation_event_count": diagnostics.validation_event_count,
            "validation_independent_coordinates": (diagnostics.validation_independent_coordinates),
            "extra_microphones_completed": diagnostics.extra_microphones_completed,
            "extra_microphone_max_inlier_rms_m": (diagnostics.extra_microphone_max_inlier_rms_m),
            "rejection_reasons": list(diagnostics.rejection_reasons),
            "selected_hypothesis": selected,
            "ambiguity": {
                "class_count": (
                    diagnostics.geometric_class_count
                    if isinstance(calibration, PlanarCalibrationResult)
                    else len(classes)
                ),
                "competing_class_support": [item.independent_subset_support for item in classes],
                "continuous_family_dimension": (
                    calibration.continuous_ambiguity_dimension
                    if isinstance(calibration, PlanarCalibrationResult)
                    else None
                ),
                "continuous_family_null_directions": (
                    _json_array(calibration.continuous_ambiguity_nullspace)
                    if isinstance(calibration, PlanarCalibrationResult)
                    else None
                ),
            },
        },
    }
    if ground_truth is not None and result.microphone_positions_m is not None:
        document["reference"] = ground_truth_to_dict(ground_truth)
        if isinstance(calibration, PlanarCalibrationResult):
            from .myotis import evaluate_planar_myotis_observables

            document["evaluation"] = evaluate_planar_myotis_observables(
                calibration,
                result.event_times_s,
                ground_truth,
            )
        else:
            evaluation = evaluate_against_ground_truth(result, ground_truth)
            document["evaluation"] = evaluation_to_dict(evaluation)
    json.dumps(document, allow_nan=False)
    return document


def write_calibration_outputs(
    result: AudioCalibrationResult,
    output_prefix: str | Path,
    *,
    input_wav_path: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    ground_truth: GroundTruth | str | Path | None = None,
) -> CalibrationOutputPaths:
    """Write JSON and a scene figure for a solved stratified result."""
    if result.microphone_positions_m is None or result.source_positions_m is None:
        raise ValueError("cannot plot calibration outputs without solved geometry")
    paths = _output_paths(output_prefix)
    if isinstance(ground_truth, (str, Path)):
        resolved_ground_truth = load_ground_truth_json(ground_truth)
    else:
        resolved_ground_truth = ground_truth

    evaluation = (
        None
        if resolved_ground_truth is None or isinstance(result.calibration, PlanarCalibrationResult)
        else evaluate_against_ground_truth(result, resolved_ground_truth)
    )
    document = calibration_result_to_dict(
        result,
        input_wav_path=input_wav_path,
        settings=settings,
        ground_truth=resolved_ground_truth,
    )
    paths.json.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_calibration_comparison(
        result,
        paths.figure,
        ground_truth=(
            None
            if isinstance(result.calibration, PlanarCalibrationResult)
            else resolved_ground_truth
        ),
        evaluation=evaluation,
    )
    return paths


def write_scene_comparison_outputs(
    estimate: GroundTruth,
    reference: GroundTruth,
    output_prefix: str | Path,
    *,
    estimate_path: str | Path | None = None,
    reference_path: str | Path | None = None,
) -> CalibrationOutputPaths:
    """Compare two canonical scenes without recalibrating audio."""
    paths = _output_paths(output_prefix)
    evaluation = evaluate_scenes(estimate, reference)
    document = ground_truth_to_dict(estimate)
    document["comparison"] = {
        "estimate_path": None if estimate_path is None else str(estimate_path),
        "reference_path": None if reference_path is None else str(reference_path),
    }
    document["reference"] = ground_truth_to_dict(reference)
    document["evaluation"] = evaluation_to_dict(evaluation)
    paths.json.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_scene_comparison(
        estimate,
        reference,
        paths.figure,
        evaluation=evaluation,
    )
    return paths
