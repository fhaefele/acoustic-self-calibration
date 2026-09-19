from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from ..geometry import select_coordinate_gauge
from ..measurements import EventTDOAMeasurements
from .robustness import huber_loss, whiten_residual_block
from .solver import PlanarCalibrationResult, StratifiedCalibrationResult

RefinementKind = Literal["wls", "huber"]


@dataclass(frozen=True)
class RefinementDiagnostics:
    mode: RefinementKind
    attempted: bool
    accepted: bool
    initial_objective: float | None
    final_objective: float | None
    initial_tdoa_rms_s: float | None
    final_tdoa_rms_s: float | None
    independent_coordinate_count: int
    nfev: int
    termination: str
    reason: str
    max_nfev: int = 0
    improvement_tolerance: float = 0.0


def _predict_tdoa(
    microphones: np.ndarray,
    sources: np.ndarray,
    measurements: EventTDOAMeasurements,
    speed_of_sound: float,
) -> np.ndarray:
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    id_to_index = {
        microphone_id: index for index, microphone_id in enumerate(measurements.microphone_ids)
    }
    predicted = np.empty_like(measurements.tdoa_s)
    for pair_index, (a, b) in enumerate(measurements.microphone_pairs):
        predicted[:, pair_index] = (
            ranges[id_to_index[b]] - ranges[id_to_index[a]]
        ) / speed_of_sound
    return predicted


def _normalized_residuals(
    microphones: np.ndarray,
    sources: np.ndarray,
    measurements: EventTDOAMeasurements,
    speed_of_sound: float,
) -> np.ndarray:
    predicted = _predict_tdoa(
        microphones,
        sources,
        measurements,
        speed_of_sound,
    )
    blocks: list[np.ndarray] = []
    for event in range(len(measurements.event_ids)):
        indices = np.flatnonzero(measurements.valid[event])
        if indices.size == 0:
            continue
        residual = predicted[event, indices] - measurements.tdoa_s[event, indices]
        if measurements.covariance_s2 is None:
            blocks.append(residual / measurements.sigma_s[event, indices])
        else:
            covariance = measurements.covariance_s2[event][np.ix_(indices, indices)]
            blocks.append(whiten_residual_block(residual, covariance))
    if not blocks:
        raise ValueError("refinement requires at least one valid TDOA coordinate")
    return np.concatenate(blocks)


def _raw_tdoa_rms(
    microphones: np.ndarray,
    sources: np.ndarray,
    measurements: EventTDOAMeasurements,
    speed_of_sound: float,
) -> float:
    predicted = _predict_tdoa(
        microphones,
        sources,
        measurements,
        speed_of_sound,
    )
    residual = predicted[measurements.valid] - measurements.tdoa_s[measurements.valid]
    return float(np.sqrt(np.mean(residual * residual)))


def _objective(
    values: np.ndarray,
    *,
    mode: RefinementKind,
) -> float:
    if mode == "wls":
        return 0.5 * float(values @ values)
    return float(np.sum(huber_loss(values, delta=1.5)))


def _inverse_gauge(
    points: np.ndarray,
    origin_m: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    return np.asarray(points, dtype=float) @ basis.T + origin_m


def _diagnostics(
    *,
    mode: RefinementKind,
    attempted: bool,
    accepted: bool,
    initial_residual: np.ndarray | None,
    final_residual: np.ndarray | None,
    initial_rms: float | None,
    final_rms: float | None,
    nfev: int,
    termination: str,
    reason: str,
    max_nfev: int = 0,
    improvement_tolerance: float = 0.0,
) -> RefinementDiagnostics:
    return RefinementDiagnostics(
        mode=mode,
        attempted=attempted,
        accepted=accepted,
        initial_objective=(
            None if initial_residual is None else _objective(initial_residual, mode=mode)
        ),
        final_objective=(None if final_residual is None else _objective(final_residual, mode=mode)),
        initial_tdoa_rms_s=initial_rms,
        final_tdoa_rms_s=final_rms,
        independent_coordinate_count=(0 if final_residual is None else int(len(final_residual))),
        nfev=int(nfev),
        termination=termination,
        reason=reason,
        max_nfev=int(max_nfev),
        improvement_tolerance=float(improvement_tolerance),
    )


def _accept_refinement(
    initial_residual: np.ndarray,
    final_residual: np.ndarray,
    *,
    mode: RefinementKind,
    improvement_tolerance: float,
) -> bool:
    initial = _objective(initial_residual, mode=mode)
    final = _objective(final_residual, mode=mode)
    required = improvement_tolerance * max(1.0, initial)
    return np.isfinite(final) and final <= initial - required


def _refine_3d(
    calibration: StratifiedCalibrationResult,
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float,
    mode: RefinementKind,
    max_nfev: int,
    improvement_tolerance: float,
) -> tuple[StratifiedCalibrationResult, RefinementDiagnostics]:
    if calibration.microphone_positions_m is None or calibration.source_positions_m is None:
        return calibration, _diagnostics(
            mode=mode,
            attempted=False,
            accepted=False,
            initial_residual=None,
            final_residual=None,
            initial_rms=None,
            final_rms=None,
            nfev=0,
            termination="not_started",
            reason="geometry_unavailable",
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )
    microphones = np.asarray(calibration.microphone_positions_m, dtype=float)
    sources = np.asarray(calibration.source_positions_m, dtype=float)
    if not np.all(np.isfinite(microphones)) or not np.all(np.isfinite(sources)):
        return calibration, _diagnostics(
            mode=mode,
            attempted=False,
            accepted=False,
            initial_residual=None,
            final_residual=None,
            initial_rms=None,
            final_rms=None,
            nfev=0,
            termination="not_started",
            reason="partial_geometry",
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )

    gauge = select_coordinate_gauge(microphones)
    canonical_mics = (microphones - gauge.origin_m) @ gauge.basis
    canonical_sources = (sources - gauge.origin_m) @ gauge.basis
    fixed = np.zeros_like(canonical_mics, dtype=bool)
    fixed[gauge.origin_index, :] = True
    fixed[gauge.x_axis_index, 1:] = True
    fixed[gauge.plane_index, 2] = True
    free_microphone_indices = np.flatnonzero(~fixed.reshape(-1))
    source_offset = len(free_microphone_indices)

    initial_vector = np.concatenate(
        [
            canonical_mics.reshape(-1)[free_microphone_indices],
            canonical_sources.reshape(-1),
        ]
    )

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mics = np.array(canonical_mics, copy=True)
        flat = mics.reshape(-1)
        flat[free_microphone_indices] = vector[:source_offset]
        refined_sources = vector[source_offset:].reshape(canonical_sources.shape)
        return mics, refined_sources

    def residual(vector: np.ndarray) -> np.ndarray:
        mics, source_states = unpack(vector)
        return _normalized_residuals(
            mics,
            source_states,
            measurements,
            speed_of_sound,
        )

    initial_residual = residual(initial_vector)
    initial_rms = _raw_tdoa_rms(
        canonical_mics,
        canonical_sources,
        measurements,
        speed_of_sound,
    )
    optimized = least_squares(
        residual,
        initial_vector,
        loss="linear" if mode == "wls" else "huber",
        f_scale=1.5,
        max_nfev=max_nfev,
    )
    refined_mics_c, refined_sources_c = unpack(optimized.x)
    final_residual = residual(optimized.x)
    final_rms = _raw_tdoa_rms(
        refined_mics_c,
        refined_sources_c,
        measurements,
        speed_of_sound,
    )
    accepted = _accept_refinement(
        initial_residual,
        final_residual,
        mode=mode,
        improvement_tolerance=improvement_tolerance,
    )
    diagnostics = _diagnostics(
        mode=mode,
        attempted=True,
        accepted=accepted,
        initial_residual=initial_residual,
        final_residual=final_residual,
        initial_rms=initial_rms,
        final_rms=final_rms,
        nfev=optimized.nfev,
        termination=str(optimized.message),
        reason=(
            "accepted"
            if accepted and optimized.success
            else "accepted_at_budget"
            if accepted
            else "insufficient_improvement"
        ),
        max_nfev=max_nfev,
        improvement_tolerance=improvement_tolerance,
    )
    if not accepted:
        return calibration, diagnostics

    refined_mics = _inverse_gauge(
        refined_mics_c,
        gauge.origin_m,
        gauge.basis,
    )
    refined_sources = _inverse_gauge(
        refined_sources_c,
        gauge.origin_m,
        gauge.basis,
    )
    refined_mics.setflags(write=False)
    refined_sources.setflags(write=False)
    return replace(
        calibration,
        microphone_positions_m=refined_mics,
        source_positions_m=refined_sources,
        tdoa_rms_s=final_rms,
    ), diagnostics


def _refine_planar(
    calibration: PlanarCalibrationResult,
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float,
    mode: RefinementKind,
    max_nfev: int,
    improvement_tolerance: float,
) -> tuple[PlanarCalibrationResult, RefinementDiagnostics]:
    if (
        calibration.microphone_positions_m is None
        or calibration.source_representative_positions_m is None
    ):
        return calibration, _diagnostics(
            mode=mode,
            attempted=False,
            accepted=False,
            initial_residual=None,
            final_residual=None,
            initial_rms=None,
            final_rms=None,
            nfev=0,
            termination="not_started",
            reason="geometry_unavailable",
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )
    microphones = np.asarray(calibration.microphone_positions_m, dtype=float)
    sources = np.asarray(calibration.source_representative_positions_m, dtype=float)
    if not np.all(np.isfinite(microphones)) or not np.all(np.isfinite(sources)):
        return calibration, _diagnostics(
            mode=mode,
            attempted=False,
            accepted=False,
            initial_residual=None,
            final_residual=None,
            initial_rms=None,
            final_rms=None,
            nfev=0,
            termination="not_started",
            reason="partial_geometry",
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )

    gauge = select_coordinate_gauge(microphones)
    if gauge.affine_rank != 2:
        return calibration, _diagnostics(
            mode=mode,
            attempted=False,
            accepted=False,
            initial_residual=None,
            final_residual=None,
            initial_rms=None,
            final_rms=None,
            nfev=0,
            termination="not_started",
            reason="receiver_geometry_not_planar",
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )
    basis = np.array(gauge.basis, copy=True)
    canonical_sources = (sources - gauge.origin_m) @ basis
    if float(np.median(canonical_sources[:, 2])) < 0.0:
        basis[:, 2] *= -1.0
        canonical_sources = (sources - gauge.origin_m) @ basis
    canonical_mics = (microphones - gauge.origin_m) @ basis
    mic_xy = canonical_mics[:, :2]
    source_states = np.column_stack(
        [
            canonical_sources[:, :2],
            np.abs(canonical_sources[:, 2]),
        ]
    )

    fixed = np.zeros_like(mic_xy, dtype=bool)
    fixed[gauge.origin_index, :] = True
    fixed[gauge.x_axis_index, 1] = True
    if calibration.angle_constraint is not None:
        id_to_index = {
            microphone_id: index for index, microphone_id in enumerate(calibration.microphone_ids)
        }
        for receiver_id in (
            calibration.angle_constraint.center_receiver,
            calibration.angle_constraint.arm_a_receiver,
            calibration.angle_constraint.arm_b_receiver,
        ):
            fixed[id_to_index[receiver_id], :] = True

    free_microphone_indices = np.flatnonzero(~fixed.reshape(-1))
    source_offset = len(free_microphone_indices)
    initial_vector = np.concatenate(
        [
            mic_xy.reshape(-1)[free_microphone_indices],
            source_states.reshape(-1),
        ]
    )
    lower = np.full_like(initial_vector, -np.inf)
    upper = np.full_like(initial_vector, np.inf)
    lower[source_offset + 2 :: 3] = 0.0

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        refined_xy = np.array(mic_xy, copy=True)
        flat = refined_xy.reshape(-1)
        flat[free_microphone_indices] = vector[:source_offset]
        refined_sources = vector[source_offset:].reshape(source_states.shape)
        refined_mics = np.column_stack([refined_xy, np.zeros(len(refined_xy))])
        return refined_mics, refined_sources

    def residual(vector: np.ndarray) -> np.ndarray:
        mics, source_values = unpack(vector)
        return _normalized_residuals(
            mics,
            source_values,
            measurements,
            speed_of_sound,
        )

    initial_residual = residual(initial_vector)
    initial_mics_c, initial_sources_c = unpack(initial_vector)
    initial_rms = _raw_tdoa_rms(
        initial_mics_c,
        initial_sources_c,
        measurements,
        speed_of_sound,
    )
    optimized = least_squares(
        residual,
        initial_vector,
        bounds=(lower, upper),
        loss="linear" if mode == "wls" else "huber",
        f_scale=1.5,
        max_nfev=max_nfev,
    )
    refined_mics_c, refined_sources_c = unpack(optimized.x)
    final_residual = residual(optimized.x)
    final_rms = _raw_tdoa_rms(
        refined_mics_c,
        refined_sources_c,
        measurements,
        speed_of_sound,
    )
    accepted = _accept_refinement(
        initial_residual,
        final_residual,
        mode=mode,
        improvement_tolerance=improvement_tolerance,
    )
    diagnostics = _diagnostics(
        mode=mode,
        attempted=True,
        accepted=accepted,
        initial_residual=initial_residual,
        final_residual=final_residual,
        initial_rms=initial_rms,
        final_rms=final_rms,
        nfev=optimized.nfev,
        termination=str(optimized.message),
        reason=(
            "accepted"
            if accepted and optimized.success
            else "accepted_at_budget"
            if accepted
            else "insufficient_improvement"
        ),
        max_nfev=max_nfev,
        improvement_tolerance=improvement_tolerance,
    )
    if not accepted:
        return calibration, diagnostics

    refined_mics = _inverse_gauge(
        refined_mics_c,
        gauge.origin_m,
        basis,
    )
    refined_sources = _inverse_gauge(
        refined_sources_c,
        gauge.origin_m,
        basis,
    )
    refined_mics.setflags(write=False)
    refined_sources.setflags(write=False)
    projected = np.array(refined_sources[:, :2], copy=True)
    heights = np.abs(refined_sources[:, 2])
    projected.setflags(write=False)
    heights.setflags(write=False)
    return replace(
        calibration,
        microphone_positions_m=refined_mics,
        source_projected_positions_m=projected,
        source_unsigned_heights_m=heights,
        source_representative_positions_m=refined_sources,
        tdoa_rms_s=final_rms,
    ), diagnostics


def refine_calibration(
    calibration: StratifiedCalibrationResult | PlanarCalibrationResult,
    measurements: EventTDOAMeasurements,
    *,
    speed_of_sound: float,
    mode: RefinementKind,
    max_nfev: int = 200,
    improvement_tolerance: float = 1e-10,
) -> tuple[
    StratifiedCalibrationResult | PlanarCalibrationResult,
    RefinementDiagnostics,
]:
    """Refine one already-selected solved geometry using TDOA measurements only."""
    if mode not in {"wls", "huber"}:
        raise ValueError("refinement mode must be 'wls' or 'huber'")
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    if improvement_tolerance < 0.0:
        raise ValueError("improvement_tolerance must be non-negative")
    if calibration.status != "solved":
        return calibration, _diagnostics(
            mode=mode,
            attempted=False,
            accepted=False,
            initial_residual=None,
            final_residual=None,
            initial_rms=calibration.tdoa_rms_s,
            final_rms=calibration.tdoa_rms_s,
            nfev=0,
            termination="not_started",
            reason="calibration_status_not_solved",
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )
    if isinstance(calibration, PlanarCalibrationResult):
        return _refine_planar(
            calibration,
            measurements,
            speed_of_sound=speed_of_sound,
            mode=mode,
            max_nfev=max_nfev,
            improvement_tolerance=improvement_tolerance,
        )
    return _refine_3d(
        calibration,
        measurements,
        speed_of_sound=speed_of_sound,
        mode=mode,
        max_nfev=max_nfev,
        improvement_tolerance=improvement_tolerance,
    )
