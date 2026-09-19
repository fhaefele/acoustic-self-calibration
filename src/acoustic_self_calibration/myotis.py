from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import evaluate_against_ground_truth, evaluation_to_dict
from .geometry import apply_rigid, rigid_align
from .ground_truth import GroundTruth, load_ground_truth_json
from .pipeline import AudioCalibrationResult, calibrate_audio
from .stratified.constraints import PlanarAngleConstraint
from .stratified.identifiability import diagnose_planar_identifiability
from .stratified.solver import PlanarCalibrationResult, calibrate_planar_tdoa
from .wav import read_multichannel_wav

AUDIO_ENV = "ASC_MYOTIS_AUDIO"
REFERENCE_ENV = "ASC_MYOTIS_REFERENCE"
OUTPUT_ENV = "ASC_MYOTIS_OUTPUT"
REVISION_ENV = "ASC_SOFTWARE_REVISION"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_real_myotis_validation(
    *,
    audio_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    speed_of_sound_mps: float = 343.0,
    channel_mapping: Sequence[int] | None = None,
    constraints: Mapping[str, Any] | None = None,
    solver_seed: int = 0,
    software_revision: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate real-Myotis inputs and return a reproducible pre-run manifest.

    Missing paths return not_run. Valid inputs return ready with hashes, channel mapping,
    timestamp convention, constraints, seed, and software revision.
    """
    env = os.environ if environment is None else environment
    audio_value = audio_path if audio_path is not None else env.get(AUDIO_ENV)
    reference_value = reference_path if reference_path is not None else env.get(REFERENCE_ENV)
    revision = (
        software_revision if software_revision is not None else env.get(REVISION_ENV, "unknown")
    )

    missing = []
    if audio_value is None:
        missing.append(AUDIO_ENV)
    if reference_value is None:
        missing.append(REFERENCE_ENV)
    if missing:
        return {
            "status": "not_run",
            "reason": "missing_input_paths",
            "missing": missing,
            "timestamp_convention": (
                "audio event times are receiver-side landmarks; reference source times "
                "are used only over their documented overlap"
            ),
        }

    assert audio_value is not None
    assert reference_value is not None
    audio = Path(audio_value).expanduser().resolve()
    reference = Path(reference_value).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Myotis audio file not found: {audio}")
    if not reference.is_file():
        raise FileNotFoundError(f"Myotis reference file not found: {reference}")
    if speed_of_sound_mps <= 0:
        raise ValueError("speed_of_sound_mps must be positive")

    sample_rate_hz, samples = read_multichannel_wav(audio)
    ground_truth = load_ground_truth_json(reference)
    channel_count = int(samples.shape[1])
    reference_microphone_count = int(ground_truth.microphone_positions_m.shape[0])

    if channel_mapping is None:
        if channel_count != reference_microphone_count:
            raise ValueError(
                "audio channel count and reference microphone count differ; "
                "provide an explicit channel_mapping"
            )
        mapping = tuple(range(channel_count))
    else:
        mapping = tuple(int(value) for value in channel_mapping)
        if len(mapping) != channel_count:
            raise ValueError("channel_mapping must contain one reference index per audio channel")
        if len(set(mapping)) != len(mapping):
            raise ValueError("channel_mapping entries must be unique")
        if any(value < 0 or value >= reference_microphone_count for value in mapping):
            raise ValueError("channel_mapping contains an out-of-range reference microphone index")

    constraint_payload = {} if constraints is None else dict(constraints)
    json.dumps(constraint_payload, allow_nan=False)
    config = {
        "speed_of_sound_mps": float(speed_of_sound_mps),
        "channel_mapping": list(mapping),
        "constraints": constraint_payload,
        "solver_seed": int(solver_seed),
        "timestamp_convention": (
            "receiver-side event landmarks; reference source trajectory is evaluation-only"
        ),
    }

    return {
        "status": "ready",
        "reason": None,
        "audio": {
            "path": str(audio),
            "sha256": _sha256_file(audio),
            "sample_rate_hz": sample_rate_hz,
            "sample_count": int(samples.shape[0]),
            "channel_count": channel_count,
            "duration_s": float(samples.shape[0] / sample_rate_hz),
        },
        "reference": {
            "path": str(reference),
            "sha256": _sha256_file(reference),
            "microphone_count": reference_microphone_count,
            "source_sample_count": int(len(ground_truth.source_times_s)),
            "source_time_start_s": float(ground_truth.source_times_s[0]),
            "source_time_stop_s": float(ground_truth.source_times_s[-1]),
            "scene_role": ground_truth.scene_role,
        },
        "configuration": config,
        "configuration_sha256": _canonical_config_hash(config),
        "software_revision": revision,
    }


def write_real_myotis_manifest(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write one real-Myotis pre-run/run manifest as stable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _finite_float_or_none(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def analyze_myotis_reference_geometry(
    reference: GroundTruth | np.ndarray,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Analyze receiver geometry only; no audio or source trajectory is consulted."""
    microphones = (
        np.asarray(reference.microphone_positions_m, dtype=float)
        if isinstance(reference, GroundTruth)
        else np.asarray(reference, dtype=float)
    )
    if microphones.ndim != 2 or microphones.shape[1] != 3:
        raise ValueError("reference microphone positions must have shape (M, 3)")
    if len(microphones) < 6:
        raise ValueError("at least six microphones are required for conditioning analysis")
    if not np.all(np.isfinite(microphones)):
        raise ValueError("reference microphone positions must be finite")
    if relative_rank_tolerance <= 0.0:
        raise ValueError("relative_rank_tolerance must be positive")

    centered = microphones - np.mean(microphones, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = relative_rank_tolerance * max(leading, np.finfo(float).tiny)
    affine_rank = int(np.sum(singular_values > tolerance))
    aperture_m = float(
        np.max(
            np.linalg.norm(
                microphones[:, None, :] - microphones[None, :, :],
                axis=2,
            )
        )
    )

    planar_basis = vt[:2].T
    planar_coordinates = centered @ planar_basis
    reconstructed_planar = planar_coordinates @ planar_basis.T
    plane_residual = centered - reconstructed_planar
    plane_rms_m = float(np.sqrt(np.mean(np.sum(plane_residual * plane_residual, axis=1))))
    planar = diagnose_planar_identifiability(planar_coordinates)

    singular_list = [float(value) for value in singular_values]
    s2_over_s1 = (
        None
        if len(singular_values) < 2 or singular_values[0] <= 0.0
        else float(singular_values[1] / singular_values[0])
    )
    s3_over_s2 = (
        None
        if len(singular_values) < 3 or singular_values[1] <= 0.0
        else float(singular_values[2] / singular_values[1])
    )
    s3_over_s1 = (
        None
        if len(singular_values) < 3 or singular_values[0] <= 0.0
        else float(singular_values[2] / singular_values[0])
    )

    identifiable: list[str] = ["receiver pairwise distances under reference geometry"]
    ambiguous: list[str] = []
    if affine_rank <= 2:
        identifiable.append(
            "receiver coordinates within the best-fit plane up to planar rigid gauge"
        )
        ambiguous.append(
            "per-event source sign across the receiver plane is not identifiable from "
            "receiver-source ranges/TDOAs alone"
        )
    if planar.continuous_metric_nullity == 0:
        identifiable.append("generic planar Euclidean metric is locally determined")
    else:
        ambiguous.append(
            "planar metric has a continuous nullspace of dimension "
            f"{planar.continuous_metric_nullity}"
        )
    if planar.cross_like_degeneracy:
        ambiguous.append(
            "cross/conic receiver geometry permits a continuous non-rigid calibration family"
        )

    nullspace = planar.metric_nullspace.tolist()
    report = {
        "receiver_count": int(len(microphones)),
        "aperture_m": aperture_m,
        "centroid_m": np.mean(microphones, axis=0).tolist(),
        "affine": {
            "rank": affine_rank,
            "rank_tolerance": tolerance,
            "singular_values": singular_list,
            "s2_over_s1": s2_over_s1,
            "s3_over_s2": s3_over_s2,
            "s3_over_s1": s3_over_s1,
        },
        "best_fit_plane": {
            "basis": planar_basis.tolist(),
            "rms_normal_residual_m": plane_rms_m,
            "relative_rms_normal_residual": (
                None if aperture_m <= 0.0 else plane_rms_m / aperture_m
            ),
        },
        "planar_conditioning": {
            "metric_design_rank": planar.metric_design.effective_rank,
            "metric_design_condition_number": _finite_float_or_none(
                planar.metric_design.condition_number
            ),
            "metric_nullity": planar.continuous_metric_nullity,
            "metric_nullspace": nullspace,
            "quadratic_design_rank": planar.quadratic_design.effective_rank,
            "quadratic_design_condition_number": _finite_float_or_none(
                planar.quadratic_design.condition_number
            ),
            "conic_nullity": planar.conic_nullity,
            "cross_like_degeneracy": planar.cross_like_degeneracy,
            "reasons": list(planar.reasons),
        },
        "identifiable_quantities": identifiable,
        "ambiguities": ambiguous,
    }
    json.dumps(report, allow_nan=False)
    return report


def _mapped_reference(
    ground_truth: GroundTruth,
    channel_mapping: Sequence[int],
) -> GroundTruth:
    mapping = np.asarray(tuple(channel_mapping), dtype=int)
    return GroundTruth(
        microphone_positions_m=ground_truth.microphone_positions_m[mapping],
        source_times_s=ground_truth.source_times_s,
        source_positions_m=ground_truth.source_positions_m,
        metadata=dict(ground_truth.metadata),
        scene_role=ground_truth.scene_role,
    )


def _audio_result_summary(result: AudioCalibrationResult) -> dict[str, Any]:
    diagnostics = result.calibration.diagnostics
    selected = (
        None
        if isinstance(result.calibration, PlanarCalibrationResult)
        else result.calibration.selected_class
    )
    validation = None
    conditioning = None
    lineage = None
    if selected is not None:
        hypothesis = selected.representative
        validation = {
            "independent_coordinate_count": hypothesis.validation.independent_coordinate_count,
            "rms_s": hypothesis.validation.rms_s,
            "median_abs_s": hypothesis.validation.median_abs_s,
            "max_abs_s": hypothesis.validation.max_abs_s,
            "normalized_huber_score": hypothesis.validation.normalized_huber_score,
            "inlier_fraction": hypothesis.validation.inlier_fraction,
        }
        conditioning = {
            "metric_condition_number": hypothesis.metric_condition_number,
            "affine_factor_singular_ratio": hypothesis.affine_factor_singular_ratio,
        }
        lineage = {
            "generation_count": int(np.sum(hypothesis.split.generation_mask)),
            "completion_count": int(np.sum(hypothesis.split.completion_mask)),
            "validation_count": int(np.sum(hypothesis.split.validation_mask)),
        }

    return {
        "status": result.status,
        "model": result.model,
        "rms_tdoa_residual_s": result.rms_tdoa_residual_s,
        "detected_event_count": result.detected_event_count,
        "used_event_count": int(len(result.measurements.event_ids)),
        "valid_measurement_count": int(np.sum(result.measurements.valid)),
        "total_measurement_count": int(result.measurements.valid.size),
        "event_channel": result.event_channel,
        "temporal_tracking_enabled": result.temporal_tracking_enabled,
        "geometric_class_count": diagnostics.geometric_class_count,
        "selected_support": diagnostics.selected_support,
        "attempted_subsets": diagnostics.attempted_subsets,
        "generated_offset_roots": diagnostics.generated_offset_roots,
        "completed_hypotheses": diagnostics.completed_hypotheses,
        "rejection_reasons": list(diagnostics.rejection_reasons),
        "extra_microphones_completed": diagnostics.extra_microphones_completed,
        "extra_microphone_max_inlier_rms_m": (diagnostics.extra_microphone_max_inlier_rms_m),
        "validation": validation,
        "conditioning": conditioning,
        "lineage": lineage,
        "refinement": {"applied": False},
    }


def run_real_myotis_calibration(
    *,
    audio_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    speed_of_sound_mps: float = 343.0,
    channel_mapping: Sequence[int] | None = None,
    solver_seed: int = 0,
    software_revision: str | None = None,
    environment: Mapping[str, str] | None = None,
    event_channel: int | None = None,
    event_smooth_s: float = 0.0003,
    event_min_gap_s: float = 0.05,
    event_relative_prominence: float = 0.003,
    max_tau_s: float = 0.012,
    tdoa_envelope_smooth_s: float = 0.00008,
    tdoa_template_s: float = 0.0018,
    tdoa_candidate_count: int = 8,
    max_tdoa_rate: float = 0.05,
    tdoa_track_weight: float = 0.4,
    use_temporal_tracking: bool = True,
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    root_start_count: int = 32,
    metric_start_count: int = 20,
    extra_microphone_inlier_rms_m: float = 0.05,
) -> dict[str, Any]:
    """Run blind real-Myotis calibration and evaluate against reference afterward.

    Reference microphone/source coordinates are never passed into event extraction or
    geometry calibration. They are used only for the separate conditioning report and
    post-run evaluation.
    """
    preflight = prepare_real_myotis_validation(
        audio_path=audio_path,
        reference_path=reference_path,
        speed_of_sound_mps=speed_of_sound_mps,
        channel_mapping=channel_mapping,
        constraints={},
        solver_seed=solver_seed,
        software_revision=software_revision,
        environment=environment,
    )
    if preflight["status"] == "not_run" and preflight.get("reason") == "missing_input_paths":
        return {
            **preflight,
            "mode": "blind",
            "reference_used_for_solver": False,
            "geometry_conditioning": None,
            "run": None,
            "evaluation": None,
        }

    audio = Path(preflight["audio"]["path"])
    reference_path_resolved = Path(preflight["reference"]["path"])
    sample_rate_hz, samples = read_multichannel_wav(audio)
    reference = load_ground_truth_json(reference_path_resolved)
    mapping = tuple(int(value) for value in preflight["configuration"]["channel_mapping"])
    mapped_reference = _mapped_reference(reference, mapping)
    geometry_conditioning = analyze_myotis_reference_geometry(mapped_reference)

    run_configuration = {
        "speed_of_sound_mps": float(speed_of_sound_mps),
        "event_channel": event_channel,
        "event_smooth_s": float(event_smooth_s),
        "event_min_gap_s": float(event_min_gap_s),
        "event_relative_prominence": float(event_relative_prominence),
        "max_tau_s": float(max_tau_s),
        "tdoa_envelope_smooth_s": float(tdoa_envelope_smooth_s),
        "tdoa_template_s": float(tdoa_template_s),
        "tdoa_candidate_count": int(tdoa_candidate_count),
        "max_tdoa_rate": float(max_tdoa_rate),
        "tdoa_track_weight": float(tdoa_track_weight),
        "use_temporal_tracking": bool(use_temporal_tracking),
        "best_sigma_samples": float(best_sigma_samples),
        "worst_sigma_samples": float(worst_sigma_samples),
        "receiver_subset_budget": int(receiver_subset_budget),
        "event_subset_budget": int(event_subset_budget),
        "root_start_count": int(root_start_count),
        "metric_start_count": int(metric_start_count),
        "extra_microphone_inlier_rms_m": float(extra_microphone_inlier_rms_m),
        "solver_seed": int(solver_seed),
    }

    try:
        result = calibrate_audio(
            samples,
            sample_rate_hz,
            event_channel=event_channel,
            event_smooth_s=event_smooth_s,
            event_min_gap_s=event_min_gap_s,
            event_relative_prominence=event_relative_prominence,
            max_tau_s=max_tau_s,
            tdoa_envelope_smooth_s=tdoa_envelope_smooth_s,
            tdoa_template_s=tdoa_template_s,
            tdoa_candidate_count=tdoa_candidate_count,
            max_tdoa_rate=max_tdoa_rate,
            tdoa_track_weight=tdoa_track_weight,
            use_temporal_tracking=use_temporal_tracking,
            speed_of_sound=speed_of_sound_mps,
            best_sigma_samples=best_sigma_samples,
            worst_sigma_samples=worst_sigma_samples,
            receiver_subset_budget=receiver_subset_budget,
            event_subset_budget=event_subset_budget,
            root_start_count=root_start_count,
            metric_start_count=metric_start_count,
            extra_microphone_inlier_rms_m=extra_microphone_inlier_rms_m,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        report = {
            **preflight,
            "status": "failed",
            "reason": "calibration_exception",
            "mode": "blind",
            "reference_used_for_solver": False,
            "geometry_conditioning": geometry_conditioning,
            "run_configuration": run_configuration,
            "run_configuration_sha256": _canonical_config_hash(run_configuration),
            "run": {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error),
            },
            "evaluation": None,
        }
        json.dumps(report, allow_nan=False)
        return report

    evaluation = None
    if result.microphone_positions_m is not None and result.source_positions_m is not None:
        try:
            evaluated = evaluate_against_ground_truth(result, mapped_reference)
        except ValueError as error:
            evaluation = {
                "status": "unavailable",
                "reason": str(error),
            }
        else:
            evaluation = {
                "status": "evaluated",
                "source_overlap_count": int(len(evaluated.source_estimate_indices)),
                "source_total_estimate_count": int(len(result.event_times_s)),
                **evaluation_to_dict(evaluated),
            }

    report = {
        **preflight,
        "status": result.status,
        "reason": None,
        "mode": "blind",
        "reference_used_for_solver": False,
        "geometry_conditioning": geometry_conditioning,
        "run_configuration": run_configuration,
        "run_configuration_sha256": _canonical_config_hash(run_configuration),
        "run": _audio_result_summary(result),
        "evaluation": evaluation,
    }
    json.dumps(report, allow_nan=False)
    return report


def _interpolate_reference_source(
    reference: GroundTruth,
    event_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(event_times_s, dtype=float)
    tolerance = 1e-9
    keep = np.flatnonzero(
        (times >= reference.source_times_s[0] - tolerance)
        & (times <= reference.source_times_s[-1] + tolerance)
    )
    if keep.size == 0:
        raise ValueError("estimated and reference source time ranges do not overlap")
    source = np.column_stack(
        [
            np.interp(
                times[keep],
                reference.source_times_s,
                reference.source_positions_m[:, dimension],
            )
            for dimension in range(3)
        ]
    )
    return keep, source


def evaluate_planar_myotis_observables(
    result: PlanarCalibrationResult,
    event_times_s: np.ndarray,
    reference: GroundTruth,
    *,
    source_half_space_sign: int | None = None,
) -> dict[str, Any]:
    if result.microphone_positions_m is None:
        raise ValueError("planar calibration has no microphone geometry")
    if result.source_representative_positions_m is None:
        raise ValueError("planar calibration has no source observables")
    if result.source_unsigned_heights_m is None:
        raise ValueError("planar calibration has no unsigned source heights")
    if source_half_space_sign not in (None, -1, 1):
        raise ValueError("source_half_space_sign must be -1, +1, or absent")

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        reference.microphone_positions_m,
        allow_reflection=True,
    )
    microphone_error = np.linalg.norm(
        aligned_microphones - reference.microphone_positions_m,
        axis=1,
    )
    mic_rms = float(np.sqrt(np.mean(microphone_error * microphone_error)))

    keep, reference_source = _interpolate_reference_source(
        reference,
        event_times_s,
    )
    estimated_representative = apply_rigid(
        result.source_representative_positions_m[keep],
        rotation,
        translation,
    )

    reference_center = np.mean(reference.microphone_positions_m, axis=0)
    centered_receivers = reference.microphone_positions_m - reference_center
    _, _, vt = np.linalg.svd(centered_receivers, full_matrices=False)
    plane_basis = vt[:2].T
    plane_normal = vt[2]
    projected_estimate = (estimated_representative - reference_center) @ plane_basis
    projected_reference = (reference_source - reference_center) @ plane_basis
    projected_error = np.linalg.norm(
        projected_estimate - projected_reference,
        axis=1,
    )

    estimate_normal = (estimated_representative - reference_center) @ plane_normal
    reference_normal = (reference_source - reference_center) @ plane_normal
    unsigned_height_error = np.abs(estimate_normal) - np.abs(reference_normal)

    signed_source_rms_m = None
    if source_half_space_sign is not None:
        signed_estimate = (
            reference_center[None, :]
            + projected_estimate @ plane_basis.T
            + (source_half_space_sign * np.abs(estimate_normal))[:, None] * plane_normal[None, :]
        )
        signed_error = np.linalg.norm(
            signed_estimate - reference_source,
            axis=1,
        )
        signed_source_rms_m = float(np.sqrt(np.mean(signed_error * signed_error)))

    return {
        "alignment": {
            "fitted_from": "microphones_only",
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "reflection_applied": bool(np.linalg.det(rotation) < 0.0),
            "reference_plane_normal": plane_normal.tolist(),
            "half_space_sign_convention": (
                "sign of source coordinate along reference_plane_normal"
            ),
        },
        "microphones": {
            "rms_error_m": mic_rms,
            "mean_error_m": float(np.mean(microphone_error)),
            "max_error_m": float(np.max(microphone_error)),
        },
        "source_observables": {
            "overlap_count": int(len(keep)),
            "estimate_indices": keep.tolist(),
            "projected_rms_error_m": float(np.sqrt(np.mean(projected_error * projected_error))),
            "projected_mean_error_m": float(np.mean(projected_error)),
            "unsigned_height_rms_error_m": float(
                np.sqrt(np.mean(unsigned_height_error * unsigned_height_error))
            ),
            "unsigned_height_mean_abs_error_m": float(np.mean(np.abs(unsigned_height_error))),
            "height_sign_known_count": int(
                np.sum(result.source_height_sign_known)
                if result.source_height_sign_known is not None
                else 0
            ),
            "signed_source_rms_m": signed_source_rms_m,
            "source_half_space_sign": source_half_space_sign,
        },
    }


def _constraint_to_dict(constraint: PlanarAngleConstraint) -> dict[str, Any]:
    return {
        "type": "planar_arm_angle",
        "center_receiver": constraint.center_receiver,
        "arm_a_receiver": constraint.arm_a_receiver,
        "arm_b_receiver": constraint.arm_b_receiver,
        "value": constraint.angle_rad,
        "units": "rad",
        "exact": constraint.exact,
        "provenance": constraint.provenance,
    }


def _planar_result_summary(result: PlanarCalibrationResult) -> dict[str, Any]:
    validation = None
    if result.validation is not None:
        validation = {
            "independent_coordinate_count": result.validation.independent_coordinate_count,
            "rms_s": result.validation.rms_s,
            "median_abs_s": result.validation.median_abs_s,
            "max_abs_s": result.validation.max_abs_s,
            "normalized_huber_score": result.validation.normalized_huber_score,
            "inlier_fraction": result.validation.inlier_fraction,
        }
    return {
        "status": result.status,
        "model": "receiver2d_source3d",
        "rms_tdoa_residual_s": result.tdoa_rms_s,
        "source_height_sign_known_count": (
            None
            if result.source_height_sign_known is None
            else int(np.sum(result.source_height_sign_known))
        ),
        "attempted_subsets": result.diagnostics.attempted_subsets,
        "completed_hypotheses": result.diagnostics.completed_hypotheses,
        "extra_microphones_completed": result.diagnostics.extra_microphones_completed,
        "extra_microphone_max_inlier_rms_m": (result.diagnostics.extra_microphone_max_inlier_rms_m),
        "rejection_reasons": list(result.diagnostics.rejection_reasons),
        "validation": validation,
    }


def run_constrained_real_myotis_evaluation(
    *,
    angle_constraint: PlanarAngleConstraint | None,
    audio_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    speed_of_sound_mps: float = 343.0,
    channel_mapping: Sequence[int] | None = None,
    source_half_space_sign: int | None = None,
    solver_seed: int = 0,
    software_revision: str | None = None,
    environment: Mapping[str, str] | None = None,
    event_channel: int | None = None,
    event_smooth_s: float = 0.0003,
    event_min_gap_s: float = 0.05,
    event_relative_prominence: float = 0.003,
    max_tau_s: float = 0.012,
    tdoa_envelope_smooth_s: float = 0.00008,
    tdoa_template_s: float = 0.0018,
    tdoa_candidate_count: int = 8,
    max_tdoa_rate: float = 0.05,
    tdoa_track_weight: float = 0.4,
    use_temporal_tracking: bool = True,
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    receiver_subset_budget: int = 3,
    event_subset_budget: int = 2,
    root_start_count: int = 32,
    metric_start_count: int = 20,
    extra_microphone_inlier_rms_m: float = 0.05,
    planar_membership_tolerance: float = 5e-3,
    planar_metric_acceptance_rms_m: float = 5e-3,
    planar_extra_microphone_rms_m: float = 0.02,
) -> dict[str, Any]:
    """Run blind and optional constrained Myotis geometry on the same measurements."""
    if source_half_space_sign not in (None, -1, 1):
        raise ValueError("source_half_space_sign must be -1, +1, or absent")
    constraint_payload = {} if angle_constraint is None else _constraint_to_dict(angle_constraint)
    preflight = prepare_real_myotis_validation(
        audio_path=audio_path,
        reference_path=reference_path,
        speed_of_sound_mps=speed_of_sound_mps,
        channel_mapping=channel_mapping,
        constraints=constraint_payload,
        solver_seed=solver_seed,
        software_revision=software_revision,
        environment=environment,
    )
    if preflight["status"] == "not_run" and preflight.get("reason") == "missing_input_paths":
        return {
            **preflight,
            "mode": "blind_and_constrained",
            "blind": None,
            "constrained": {
                "status": "not_run",
                "reason": "missing_input_paths",
            },
            "constraint_ablation": None,
        }
    if angle_constraint is None:
        blind = run_real_myotis_calibration(
            audio_path=preflight["audio"]["path"],
            reference_path=preflight["reference"]["path"],
            speed_of_sound_mps=speed_of_sound_mps,
            channel_mapping=preflight["configuration"]["channel_mapping"],
            solver_seed=solver_seed,
            software_revision=software_revision,
            event_channel=event_channel,
            event_smooth_s=event_smooth_s,
            event_min_gap_s=event_min_gap_s,
            event_relative_prominence=event_relative_prominence,
            max_tau_s=max_tau_s,
            tdoa_envelope_smooth_s=tdoa_envelope_smooth_s,
            tdoa_template_s=tdoa_template_s,
            tdoa_candidate_count=tdoa_candidate_count,
            max_tdoa_rate=max_tdoa_rate,
            tdoa_track_weight=tdoa_track_weight,
            use_temporal_tracking=use_temporal_tracking,
            best_sigma_samples=best_sigma_samples,
            worst_sigma_samples=worst_sigma_samples,
            receiver_subset_budget=receiver_subset_budget,
            event_subset_budget=event_subset_budget,
            root_start_count=root_start_count,
            metric_start_count=metric_start_count,
            extra_microphone_inlier_rms_m=extra_microphone_inlier_rms_m,
        )
        return {
            **preflight,
            "status": "not_run",
            "reason": "explicit_constraint_not_supplied",
            "mode": "blind_and_constrained",
            "blind": blind,
            "constrained": {
                "status": "not_run",
                "reason": "explicit_constraint_not_supplied",
            },
            "constraint_ablation": None,
        }

    audio = Path(preflight["audio"]["path"])
    reference_path_resolved = Path(preflight["reference"]["path"])
    sample_rate_hz, samples = read_multichannel_wav(audio)
    reference = load_ground_truth_json(reference_path_resolved)
    mapping = tuple(int(value) for value in preflight["configuration"]["channel_mapping"])
    mapped_reference = _mapped_reference(reference, mapping)
    geometry_conditioning = analyze_myotis_reference_geometry(mapped_reference)

    run_configuration = {
        "speed_of_sound_mps": float(speed_of_sound_mps),
        "event_channel": event_channel,
        "event_smooth_s": float(event_smooth_s),
        "event_min_gap_s": float(event_min_gap_s),
        "event_relative_prominence": float(event_relative_prominence),
        "max_tau_s": float(max_tau_s),
        "tdoa_envelope_smooth_s": float(tdoa_envelope_smooth_s),
        "tdoa_template_s": float(tdoa_template_s),
        "tdoa_candidate_count": int(tdoa_candidate_count),
        "max_tdoa_rate": float(max_tdoa_rate),
        "tdoa_track_weight": float(tdoa_track_weight),
        "use_temporal_tracking": bool(use_temporal_tracking),
        "best_sigma_samples": float(best_sigma_samples),
        "worst_sigma_samples": float(worst_sigma_samples),
        "receiver_subset_budget": int(receiver_subset_budget),
        "event_subset_budget": int(event_subset_budget),
        "root_start_count": int(root_start_count),
        "metric_start_count": int(metric_start_count),
        "extra_microphone_inlier_rms_m": float(extra_microphone_inlier_rms_m),
        "planar_membership_tolerance": float(planar_membership_tolerance),
        "planar_metric_acceptance_rms_m": float(planar_metric_acceptance_rms_m),
        "planar_extra_microphone_rms_m": float(planar_extra_microphone_rms_m),
        "solver_seed": int(solver_seed),
        "constraint": constraint_payload,
        "source_half_space_sign": source_half_space_sign,
    }

    try:
        blind_result = calibrate_audio(
            samples,
            sample_rate_hz,
            event_channel=event_channel,
            event_smooth_s=event_smooth_s,
            event_min_gap_s=event_min_gap_s,
            event_relative_prominence=event_relative_prominence,
            max_tau_s=max_tau_s,
            tdoa_envelope_smooth_s=tdoa_envelope_smooth_s,
            tdoa_template_s=tdoa_template_s,
            tdoa_candidate_count=tdoa_candidate_count,
            max_tdoa_rate=max_tdoa_rate,
            tdoa_track_weight=tdoa_track_weight,
            use_temporal_tracking=use_temporal_tracking,
            speed_of_sound=speed_of_sound_mps,
            best_sigma_samples=best_sigma_samples,
            worst_sigma_samples=worst_sigma_samples,
            receiver_subset_budget=receiver_subset_budget,
            event_subset_budget=event_subset_budget,
            root_start_count=root_start_count,
            metric_start_count=metric_start_count,
            extra_microphone_inlier_rms_m=extra_microphone_inlier_rms_m,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            **preflight,
            "status": "failed",
            "reason": "frontend_or_blind_calibration_exception",
            "mode": "blind_and_constrained",
            "geometry_conditioning": geometry_conditioning,
            "run_configuration": run_configuration,
            "run_configuration_sha256": _canonical_config_hash(run_configuration),
            "blind": {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error),
            },
            "constrained": {"status": "not_run", "reason": "frontend_failed"},
            "constraint_ablation": None,
        }

    blind_evaluation = None
    if (
        blind_result.microphone_positions_m is not None
        and blind_result.source_positions_m is not None
    ):
        try:
            blind_eval = evaluate_against_ground_truth(blind_result, mapped_reference)
        except ValueError as error:
            blind_evaluation = {"status": "unavailable", "reason": str(error)}
        else:
            blind_evaluation = {
                "status": "evaluated",
                **evaluation_to_dict(blind_eval),
            }

    constrained = calibrate_planar_tdoa(
        blind_result.measurements,
        speed_of_sound=speed_of_sound_mps,
        receiver_subset_budget=receiver_subset_budget,
        event_subset_budget=event_subset_budget,
        membership_tolerance=planar_membership_tolerance,
        metric_acceptance_rms_m=planar_metric_acceptance_rms_m,
        extra_microphone_rms_m=planar_extra_microphone_rms_m,
        angle_constraint=angle_constraint,
    )
    unconstrained = calibrate_planar_tdoa(
        blind_result.measurements,
        speed_of_sound=speed_of_sound_mps,
        receiver_subset_budget=receiver_subset_budget,
        event_subset_budget=event_subset_budget,
        membership_tolerance=planar_membership_tolerance,
        metric_acceptance_rms_m=planar_metric_acceptance_rms_m,
        extra_microphone_rms_m=planar_extra_microphone_rms_m,
        angle_constraint=None,
    )

    constrained_evaluation = None
    if constrained.microphone_positions_m is not None:
        try:
            constrained_evaluation = evaluate_planar_myotis_observables(
                constrained,
                blind_result.event_times_s,
                mapped_reference,
                source_half_space_sign=source_half_space_sign,
            )
        except ValueError as error:
            constrained_evaluation = {
                "status": "unavailable",
                "reason": str(error),
            }
        else:
            constrained_evaluation["status"] = "evaluated"

    report = {
        **preflight,
        "status": constrained.status,
        "reason": None,
        "mode": "blind_and_constrained",
        "geometry_conditioning": geometry_conditioning,
        "constraint": constraint_payload,
        "run_configuration": run_configuration,
        "run_configuration_sha256": _canonical_config_hash(run_configuration),
        "reference_used_for_solver": False,
        "blind": {
            "run": _audio_result_summary(blind_result),
            "evaluation": blind_evaluation,
        },
        "constrained": {
            "run": _planar_result_summary(constrained),
            "evaluation": constrained_evaluation,
            "source_half_space_sign": source_half_space_sign,
        },
        "constraint_ablation": {
            "without_constraint_status": unconstrained.status,
            "without_constraint_rejection_reasons": list(
                unconstrained.diagnostics.rejection_reasons
            ),
            "ambiguity_restored": unconstrained.status
            in {"ambiguous", "degenerate", "weakly_identified", "failed"},
        },
    }
    json.dumps(report, allow_nan=False)
    return report
