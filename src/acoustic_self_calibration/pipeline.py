from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .array_configuration import ArrayConfiguration
from .events import (
    EventDetection,
    detect_transient_events,
    estimate_event_tdoa_measurements,
)
from .measurements import EventTDOAMeasurements
from .output_frame import MicrophoneFrame, select_microphone_frame, transform_calibration
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
    array_configuration: ArrayConfiguration | None = None
    coordinate_frame: MicrophoneFrame | None = None
    timing_uncertainty_model: Literal["confidence", "waveform"] = "confidence"

    @property
    def event_times_s(self) -> np.ndarray:
        return self.measurements.receiver_event_times_s

    @property
    def frame_times_s(self) -> np.ndarray:
        """Compatibility alias; source states are events rather than analysis frames."""
        return self.event_times_s

    @property
    def emission_times_s(self) -> np.ndarray:
        """Inferred emission-feature times; unresolved events remain NaN."""
        times = np.full(len(self.event_times_s), np.nan)
        if self.source_positions_m is not None and self.microphone_positions_m is not None:
            channel = self.event_channel or 0
            times = (
                self.event_times_s
                - np.linalg.norm(
                    self.source_positions_m - self.microphone_positions_m[channel], axis=1
                )
                / self.speed_of_sound_mps
            )
        return times

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
    microphone_ids: tuple[int, ...] | None = None,
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
    timing_uncertainty: Literal["confidence", "waveform"] = "confidence",
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
    array_configuration: ArrayConfiguration | None = None,
    source_region: Literal["same_side"] | None = None,
    output_origin_microphone_id: int | None = None,
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
    ids = tuple(range(values.shape[1])) if microphone_ids is None else tuple(microphone_ids)
    if len(ids) != values.shape[1] or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
        raise ValueError("microphone_ids must be distinct non-negative IDs matching audio channels")
    if output_origin_microphone_id is not None and output_origin_microphone_id not in ids:
        raise ValueError("output origin microphone ID is absent")
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")
    if array_configuration is not None:
        array_configuration.validate_ids(ids)
        if model != "general_3d" and model != array_configuration.model:
            raise ValueError("model conflicts with array configuration")
        model = array_configuration.model
        supplied_angle = array_configuration.angle_constraint()
        if angle_constraint is not None and angle_constraint != supplied_angle:
            raise ValueError("angle_constraint conflicts with array configuration")
        angle_constraint = supplied_angle
        source_region = None if model == "general_3d" else "same_side"
    if model not in {"general_3d", "receiver2d_source3d"}:
        raise ValueError("model must be 'general_3d' or 'receiver2d_source3d'")
    if source_region not in (None, "same_side"):
        raise ValueError("source_region must be same_side or absent")
    if source_region is not None and model != "receiver2d_source3d":
        raise ValueError("same_side requires the planar model")
    if angle_constraint is not None and model != "receiver2d_source3d":
        raise ValueError("angle_constraint requires the planar model")
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
        timing_uncertainty=timing_uncertainty,
    )
    if microphone_ids is not None:
        measurements = replace(
            measurements,
            microphone_ids=ids,
            microphone_pairs=tuple((ids[a], ids[b]) for a, b in measurements.microphone_pairs),
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
            source_half_space_sign=1 if source_region == "same_side" else None,
        )
    if model == "general_3d" and timing_uncertainty == "waveform":
        spatial_std = calibration.diagnostics.spatial_relative_std
        if spatial_std is not None:
            weak = calibration.status == "solved" and spatial_std > 0.05
            calibration = replace(
                calibration,
                status="weakly_identified" if weak else calibration.status,
                diagnostics=replace(
                    calibration.diagnostics,
                    spatial_relative_std_limit=0.05,
                    rejection_reasons=(
                        *calibration.diagnostics.rejection_reasons,
                        *(("spatial_recorded_waveform_noise_sensitive",) if weak else ()),
                    ),
                ),
            )
    pre_refinement = None
    refinement_diagnostics = None
    structured = array_configuration is not None and array_configuration.name in {
        "cross",
        "t",
        "grid",
    }
    if refinement != "none" and not structured:
        pre_refinement = calibration
        calibration, refinement_diagnostics = refine_calibration(
            calibration,
            measurements,
            speed_of_sound=speed_of_sound,
            mode=refinement,
            max_nfev=refinement_max_nfev,
            improvement_tolerance=refinement_improvement_tolerance,
        )

    if structured and array_configuration is not None:
        from .stratified.structured import apply_array_configuration

        assert isinstance(calibration, PlanarCalibrationResult)
        if (
            calibration.microphone_positions_m is not None
            and array_configuration.metric_ambiguity_reason() is None
        ):
            pre_refinement = calibration
        calibration = apply_array_configuration(
            calibration,
            measurements,
            array_configuration,
            speed_of_sound=speed_of_sound,
            mode="huber" if refinement == "huber" else "wls",
            max_nfev=refinement_max_nfev,
        )
    if source_region == "same_side":
        from .stratified.solver import _apply_source_half_space_prior

        assert isinstance(calibration, PlanarCalibrationResult)
        calibration = _apply_source_half_space_prior(calibration, 1)

    sources_for_coverage = (
        calibration.source_representative_positions_m
        if isinstance(calibration, PlanarCalibrationResult)
        else calibration.source_positions_m
    )
    if calibration.status == "solved" and (
        sources_for_coverage is None
        or not np.isfinite(sources_for_coverage).all()
        or len(measurements.event_ids) != len(detection.event_samples)
    ):
        calibration = replace(
            calibration,
            status="weakly_identified",
            diagnostics=replace(
                calibration.diagnostics,
                rejection_reasons=(
                    *calibration.diagnostics.rejection_reasons,
                    "incomplete_source_coverage",
                ),
            ),
        )
    frame = None
    if (
        calibration.microphone_positions_m is not None
        and np.isfinite(calibration.microphone_positions_m).all()
    ):
        sources = (
            calibration.source_representative_positions_m
            if isinstance(calibration, PlanarCalibrationResult)
            else calibration.source_positions_m
        )
        frame = select_microphone_frame(
            calibration.microphone_positions_m,
            measurements.microphone_ids,
            origin_id=output_origin_microphone_id,
            common_side_sources=sources if source_region else None,
        )
        calibration = transform_calibration(calibration, frame)
        pre_refinement = transform_calibration(pre_refinement, frame)
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
        array_configuration=array_configuration,
        coordinate_frame=frame,
        timing_uncertainty_model=timing_uncertainty,
    )
