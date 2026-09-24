from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .events import (
    EventDetection,
    detect_transient_events,
    estimate_event_tdoa_measurements,
)
from .measurements import EventTDOAMeasurements
from .stratified.constraints import PlanarAngleConstraint
from .stratified.refinement import RefinementDiagnostics, refine_calibration
from .stratified.solver import (
    PlanarCalibrationResult,
    StratifiedCalibrationResult,
    calibrate_planar_tdoa,
    calibrate_tdoa,
)

AudioModel = Literal["general_3d", "receiver2d_source3d"]
RefinementMode = Literal["none", "wls", "huber"]


@dataclass(frozen=True)
class AudioCalibrationResult:
    """Event-driven stratified calibration plus the measurements that produced it."""

    calibration: StratifiedCalibrationResult | PlanarCalibrationResult
    measurements: EventTDOAMeasurements
    detection: EventDetection
    speed_of_sound_mps: float
    model: AudioModel
    temporal_tracking_enabled: bool
    refinement_mode: RefinementMode
    pre_refinement_calibration: StratifiedCalibrationResult | PlanarCalibrationResult | None = None
    refinement_diagnostics: RefinementDiagnostics | None = None

    @property
    def event_times_s(self) -> np.ndarray:
        return self.measurements.receiver_event_times_s

    @property
    def frame_times_s(self) -> np.ndarray:
        """Compatibility alias; source states are events rather than analysis frames."""
        return self.event_times_s

    @property
    def event_samples(self) -> np.ndarray | None:
        return self.measurements.event_samples

    @property
    def event_channel(self) -> int | None:
        return self.measurements.event_channel

    @property
    def detected_event_count(self) -> int:
        return len(self.detection.event_samples)

    @property
    def tdoa_s(self) -> np.ndarray:
        return self.measurements.tdoa_s

    @property
    def tdoa_sigma_s(self) -> np.ndarray:
        return self.measurements.sigma_s

    @property
    def confidence(self) -> np.ndarray:
        return self.measurements.confidence

    @property
    def microphone_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.measurements.microphone_pairs

    @property
    def microphone_positions_m(self) -> np.ndarray | None:
        return self.calibration.microphone_positions_m

    @property
    def source_positions_m(self) -> np.ndarray | None:
        if isinstance(self.calibration, PlanarCalibrationResult):
            return self.calibration.source_representative_positions_m
        return self.calibration.source_positions_m

    @property
    def source_projected_positions_m(self) -> np.ndarray | None:
        if isinstance(self.calibration, PlanarCalibrationResult):
            return self.calibration.source_projected_positions_m
        return None

    @property
    def source_unsigned_heights_m(self) -> np.ndarray | None:
        if isinstance(self.calibration, PlanarCalibrationResult):
            return self.calibration.source_unsigned_heights_m
        return None

    @property
    def source_height_sign_known(self) -> np.ndarray | None:
        if isinstance(self.calibration, PlanarCalibrationResult):
            return self.calibration.source_height_sign_known
        return None

    @property
    def status(self) -> str:
        return self.calibration.status

    @property
    def rms_tdoa_residual_s(self) -> float | None:
        return self.calibration.tdoa_rms_s

    @property
    def tdoa_rms_s(self) -> float | None:
        """Compatibility alias for the stratified TDOA residual diagnostic."""
        return self.calibration.tdoa_rms_s


def calibrate_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    event_channel: int | None = None,
    event_smooth_s: float = 0.0003,
    event_min_gap_s: float = 0.003,
    event_relative_prominence: float = 0.003,
    max_tau_s: float = 0.01,
    tdoa_envelope_smooth_s: float = 0.00008,
    tdoa_template_s: float = 0.0018,
    tdoa_candidate_count: int = 8,
    max_tdoa_rate: float = 0.05,
    tdoa_track_weight: float = 0.4,
    use_temporal_tracking: bool = True,
    speed_of_sound: float = 343.0,
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
    angle_constraint: PlanarAngleConstraint | None = None,
    model: AudioModel = "general_3d",
    refinement: RefinementMode = "none",
    refinement_max_nfev: int = 200,
    refinement_improvement_tolerance: float = 1e-10,
) -> AudioCalibrationResult:
    """Self-calibrate synchronized microphones from discrete broadband events.

    The frontend uses no source waveform, emission timestamps, geometry prior, or source
    motion prior. Optional temporal lag tracking is only an event-association heuristic
    and can be disabled diagnostically. Geometry is solved from the immutable
    reference-star TDOA contract by the stratified backend.
    """
    values = np.asarray(audio, dtype=float)
    if values.ndim != 2:
        raise ValueError("audio must have shape (samples, microphones)")
    if values.shape[1] < 8:
        raise ValueError("the current stratified audio backends require at least 8 microphones")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("audio contains non-finite samples")
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")
    if model not in {"general_3d", "receiver2d_source3d"}:
        raise ValueError("model must be 'general_3d' or 'receiver2d_source3d'")
    if refinement not in {"none", "wls", "huber"}:
        raise ValueError("refinement must be 'none', 'wls', or 'huber'")
    if refinement_max_nfev < 1:
        raise ValueError("refinement_max_nfev must be positive")
    if refinement_improvement_tolerance < 0.0:
        raise ValueError("refinement_improvement_tolerance must be non-negative")

    detection = detect_transient_events(
        values,
        sample_rate,
        event_channel=event_channel,
        smooth_s=event_smooth_s,
        min_gap_s=event_min_gap_s,
        relative_prominence=event_relative_prominence,
        min_events=12,
    )
    measurements = estimate_event_tdoa_measurements(
        values,
        sample_rate,
        detection,
        max_tau_s=max_tau_s,
        envelope_smooth_s=tdoa_envelope_smooth_s,
        template_s=tdoa_template_s,
        candidate_count=tdoa_candidate_count,
        max_tdoa_rate=max_tdoa_rate,
        transition_weight=tdoa_track_weight,
        use_temporal_tracking=use_temporal_tracking,
        best_sigma_samples=best_sigma_samples,
        worst_sigma_samples=worst_sigma_samples,
    )
    if model == "general_3d":
        calibration: StratifiedCalibrationResult | PlanarCalibrationResult = calibrate_tdoa(
            measurements,
            speed_of_sound=speed_of_sound,
            receiver_subset_budget=receiver_subset_budget,
            event_subset_budget=event_subset_budget,
            root_start_count=root_start_count,
            metric_start_count=metric_start_count,
            extra_microphone_inlier_rms_m=extra_microphone_inlier_rms_m,
        )
    else:
        calibration = calibrate_planar_tdoa(
            measurements,
            speed_of_sound=speed_of_sound,
            receiver_subset_budget=receiver_subset_budget,
            event_subset_budget=event_subset_budget,
            membership_tolerance=planar_membership_tolerance,
            metric_acceptance_rms_m=planar_metric_acceptance_rms_m,
            extra_microphone_rms_m=planar_extra_microphone_rms_m,
            angle_constraint=angle_constraint,
        )
    pre_refinement = None
    refinement_diagnostics = None
    if refinement != "none":
        pre_refinement = calibration
        calibration, refinement_diagnostics = refine_calibration(
            calibration,
            measurements,
            speed_of_sound=speed_of_sound,
            mode=refinement,
            max_nfev=refinement_max_nfev,
            improvement_tolerance=refinement_improvement_tolerance,
        )

    return AudioCalibrationResult(
        calibration=calibration,
        measurements=measurements,
        detection=detection,
        speed_of_sound_mps=float(speed_of_sound),
        model=model,
        temporal_tracking_enabled=bool(use_temporal_tracking),
        refinement_mode=refinement,
        pre_refinement_calibration=pre_refinement,
        refinement_diagnostics=refinement_diagnostics,
    )
