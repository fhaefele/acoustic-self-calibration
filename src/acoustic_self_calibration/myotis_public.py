"""Public WAV recovery first, reference-only evaluation after the result is frozen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .array_configuration import ArrayConfiguration
from .export import calibration_result_to_dict
from .geometry import apply_rigid, rigid_align
from .ground_truth import load_ground_truth_json
from .stratified.solver import calibrate_planar_tdoa
from .wav import calibrate_wav


def calibrate_myotis_recording(audio_path, configuration: ArrayConfiguration, **options):
    """No reference input exists at this calibration boundary."""
    return calibrate_wav(
        audio_path,
        array_configuration=configuration,
        max_tau_s=0.012,
        receiver_subset_budget=5,
        **options,
    )


def evaluate_myotis_recording(result, reference_path):
    reference = load_ground_truth_json(reference_path)
    if result.microphone_positions_m is None or result.source_positions_m is None:
        return dict(status="unavailable", reason="geometry_unavailable", success=False)
    aligned, rotation, translation = rigid_align(
        result.microphone_positions_m, reference.microphone_positions_m
    )
    sources = apply_rigid(result.source_positions_m, rotation, translation)
    center = reference.microphone_positions_m.mean(axis=0)
    _, _, vt = np.linalg.svd(reference.microphone_positions_m - center)
    normal = vt[-1]
    finite = np.isfinite(sources).all(axis=1) & np.isfinite(result.emission_times_s)
    reflected = False
    if (
        np.any(finite)
        and np.mean((sources[finite] - center) @ normal)
        * np.mean((reference.source_positions_m - center) @ normal)
        < 0
    ):
        # A single scene reflection resolves output-frame handedness. No event
        # is reflected independently, and nothing feeds back into calibration.
        sources -= 2 * ((sources - center) @ normal)[:, None] * normal
        reflected = True
    keep = (
        finite
        & (result.emission_times_s >= reference.source_times_s[0])
        & (result.emission_times_s <= reference.source_times_s[-1])
    )
    target = np.column_stack(
        [
            np.interp(
                result.emission_times_s[keep],
                reference.source_times_s,
                reference.source_positions_m[:, axis],
            )
            for axis in range(3)
        ]
    )
    mic_rms = float(
        np.sqrt(np.mean(np.sum((aligned - reference.microphone_positions_m) ** 2, axis=1)))
    )
    source_rms = (
        float(np.sqrt(np.mean(np.sum((sources[keep] - target) ** 2, axis=1))))
        if np.any(keep)
        else None
    )
    validation = result.calibration.validation
    heldout = None if validation is None else validation.rms_s
    success = (
        result.status == "solved"
        and len(aligned) == 12
        and source_rms is not None
        and mic_rms < 0.2
        and source_rms < 0.3
        and result.tdoa_rms_s is not None
        and result.tdoa_rms_s < 45e-6
        and heldout is not None
        and heldout < 45e-6
        and int(np.sum(finite)) == result.detected_event_count
        and int(np.sum(keep)) >= 40
    )
    return dict(
        status="evaluated",
        microphone_rms_error_m=mic_rms,
        source_rms_error_m=source_rms,
        detected_events=result.detected_event_count,
        localized_events=int(np.sum(finite)),
        reference_overlap_count=int(np.sum(keep)),
        reference_source_count=len(reference.source_times_s),
        complete_detected_event_coverage=bool(int(np.sum(finite)) == result.detected_event_count),
        unresolved_event_ids=result.measurements.event_ids[~finite].tolist(),
        global_normal_reflection=reflected,
        alignment_fitted_from="microphones_only",
        source_time_basis="inferred_emission_feature",
        heldout_tdoa_rms_s=heldout,
        fitted_tdoa_rms_s=result.tdoa_rms_s,
        success=bool(success),
    )


def public_myotis_report(audio_path, reference_path, configuration, **options):
    # Calibration and ablation consume only the audio and declared construction.
    result = calibrate_myotis_recording(audio_path, configuration, **options)
    frozen = calibration_result_to_dict(result)
    ablation = calibrate_planar_tdoa(result.measurements, receiver_subset_budget=5)
    evaluation = evaluate_myotis_recording(result, reference_path)
    audio_hash = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
    provenance_path = Path(audio_path).with_name("provenance.json")
    provenance: dict[str, Any] = (
        json.loads(provenance_path.read_text())
        if provenance_path.is_file()
        else {"acquisition_provenance": "unverified", "reason": "No adjacent provenance record"}
    )
    recorded_hashes = {entry.get("sha256") for entry in provenance.get("files", {}).values()}
    if recorded_hashes and audio_hash not in recorded_hashes:
        provenance = {
            "acquisition_provenance": "unverified",
            "reason": "Adjacent provenance does not match the audio hash",
        }
    return {
        "schema_version": 1,
        "reference_used_for_solver": False,
        "audio_sha256": audio_hash,
        "configuration": configuration.to_dict(),
        "options": options,
        "provenance": provenance,
        "recovery": frozen,
        "evaluation": evaluation,
        "ablation": {
            "status": ablation.status,
            "continuous_metric_family_dimension": ablation.continuous_ambiguity_dimension,
            "diagnostics": list(ablation.diagnostics.rejection_reasons),
        },
        "accepted": evaluation["success"],
    }
