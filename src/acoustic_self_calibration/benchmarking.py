"""Evaluation-only benchmark metrics; nothing here selects solver geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import apply_rigid, rigid_align
from .simulation import SyntheticPulseScene


def scene_metrics(
    calibration: Any,
    scene: SyntheticPulseScene,
    emission_times_s: np.ndarray,
) -> dict[str, Any]:
    """Evaluate all available states, with explicit one-to-one event coverage."""
    microphones = calibration.microphone_positions_m
    sources = getattr(calibration, "source_positions_m", None)
    if sources is None:
        sources = getattr(calibration, "source_representative_positions_m", None)
    row: dict[str, Any] = {
        "expected_events": len(scene.event_times_s),
        "matched_events": 0,
        "unresolved_events": len(scene.event_times_s),
        "source_rms_error_m": None,
        "source_errors_m": [],
        "source_max_error_m": None,
        "source_in_plane_rms_error_m": None,
        "source_normal_rms_error_m": None,
        "microphone_rms_error_m": None,
        "microphone_max_error_m": None,
        "standard_success": False,
    }
    if microphones is None or not np.isfinite(microphones).all() or sources is None:
        return row
    ids = calibration.microphone_ids or tuple(range(len(microphones)))
    target = scene.microphone_positions_m[list(ids)]
    aligned, rotation, translation = rigid_align(microphones, target)
    mic_errors = np.linalg.norm(aligned - target, axis=1)
    row.update(
        microphone_rms_error_m=float(np.sqrt(np.mean(mic_errors**2))),
        microphone_max_error_m=float(np.max(mic_errors)),
    )
    sources = apply_rigid(np.asarray(sources), rotation, translation)
    # One global reflection of the scene is gauge for planar microphones.
    # The occupied reference side belongs to evaluator truth, never the solver.
    _, singular, vt = np.linalg.svd(target - target.mean(axis=0), full_matrices=False)
    is_planar = singular[-1] < 1e-8 * max(singular[0], 1.0)
    if is_planar:
        normal = vt[-1]
        center = target.mean(axis=0)
        reference_side = np.sign(np.mean((scene.source_positions_at_events_m - center) @ normal))
        finite = np.isfinite(sources).all(axis=1)
        if np.any(finite) and np.mean((sources[finite] - center) @ normal) * reference_side < 0:
            sources -= 2 * ((sources - center) @ normal)[:, None] * normal
    times = np.asarray(emission_times_s)
    valid = np.isfinite(sources).all(axis=1) & np.isfinite(times)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return row
    distances = np.abs(times[valid, None] - scene.event_times_s[None, :])
    estimated, truth = linear_sum_assignment(distances)
    tolerance = min(0.02, 0.45 * float(np.min(np.diff(scene.event_times_s))))
    keep = distances[estimated, truth] < tolerance
    estimated, truth = valid_indices[estimated[keep]], truth[keep]
    error_vectors = sources[estimated] - scene.source_positions_at_events_m[truth]
    errors = np.linalg.norm(error_vectors, axis=1)
    row.update(
        matched_events=len(errors),
        unresolved_events=len(scene.event_times_s) - len(errors),
        source_errors_m=errors.tolist(),
        matched_event_ids=truth.tolist(),
    )
    if len(errors):
        row["source_rms_error_m"] = float(np.sqrt(np.mean(errors**2)))
        row["source_max_error_m"] = float(np.max(errors))
        if is_planar:
            normal_errors = error_vectors @ vt[-1]
            row["source_normal_rms_error_m"] = float(np.sqrt(np.mean(normal_errors**2)))
            row["source_in_plane_rms_error_m"] = float(
                np.sqrt(np.mean(np.maximum(errors**2 - normal_errors**2, 0.0)))
            )
    row["standard_success"] = bool(
        calibration.status == "solved"
        and len(errors) == len(scene.event_times_s)
        and float(np.sqrt(np.mean(mic_errors**2))) < 0.05
        and row["source_rms_error_m"] is not None
        and row["source_rms_error_m"] < 0.10
        and calibration.tdoa_rms_s is not None
        and calibration.tdoa_rms_s < 60e-6
    )
    return row
