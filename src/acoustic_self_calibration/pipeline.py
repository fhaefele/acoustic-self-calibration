from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bayesian import (
    BayesianCalibrationResult,
    DistancePrior,
    calibrate_bayesian,
    tdoa_sigma_from_confidence,
)
from .events import EventTDOAMeasurements, detect_transient_events, estimate_event_tdoas
from .initialization import event_initial_scene_candidates
from .tdoa import make_microphone_pairs


@dataclass(frozen=True)
class AudioCalibrationResult:
    """End-to-end event-driven calibration result and its acoustic measurements."""

    calibration: BayesianCalibrationResult
    event_times_s: np.ndarray
    event_samples: np.ndarray
    event_channel: int
    detected_event_count: int
    tdoa_s: np.ndarray
    tdoa_sigma_s: np.ndarray
    confidence: np.ndarray
    arrival_delays_s: np.ndarray
    arrival_confidence: np.ndarray
    microphone_pairs: tuple[tuple[int, int], ...]

    @property
    def frame_times_s(self) -> np.ndarray:
        """Compatibility alias; source states are now transient events, not frames."""
        return self.event_times_s


def _solve_from_event_multistarts(
    measurements: EventTDOAMeasurements,
    microphone_count: int,
    *,
    sigma: np.ndarray,
    speed_of_sound: float,
    motion_velocity_change_sigma_mps: float | None,
    likelihood: str,
    estimate_clock_offsets: bool,
    estimate_clock_drifts: bool,
    estimate_speed_of_sound: bool,
    distance_priors: Sequence[DistancePrior],
    max_nfev: int,
    compute_laplace_uncertainty: bool,
) -> BayesianCalibrationResult:
    scales = (0.55, 0.75, 1.0) if microphone_count <= 12 else (0.75,)
    candidates = event_initial_scene_candidates(
        measurements.arrival_delays_s,
        measurements.tdoa_s,
        sigma,
        measurements.microphone_pairs,
        speed_of_sound=speed_of_sound,
        scale_candidates=scales,
    )

    preview_budget = min(max_nfev, 250)
    previews: list[BayesianCalibrationResult] = []
    for microphones0, sources0 in candidates:
        try:
            preview = calibrate_bayesian(
                measurements.tdoa_s,
                measurements.event_times_s,
                microphone_count,
                tdoa_sigma_s=sigma,
                microphone_pairs=measurements.microphone_pairs,
                speed_of_sound=speed_of_sound,
                estimate_speed_of_sound=estimate_speed_of_sound,
                estimate_clock_offsets=estimate_clock_offsets,
                estimate_clock_drifts=estimate_clock_drifts,
                distance_priors=distance_priors,
                motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
                initial_microphones=microphones0,
                initial_sources=sources0,
                likelihood=likelihood,
                max_nfev=preview_budget,
                compute_laplace_uncertainty=False,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(preview.negative_log_posterior):
            previews.append(preview)

    if not previews:
        return calibrate_bayesian(
            measurements.tdoa_s,
            measurements.event_times_s,
            microphone_count,
            tdoa_sigma_s=sigma,
            microphone_pairs=measurements.microphone_pairs,
            speed_of_sound=speed_of_sound,
            estimate_speed_of_sound=estimate_speed_of_sound,
            estimate_clock_offsets=estimate_clock_offsets,
            estimate_clock_drifts=estimate_clock_drifts,
            distance_priors=distance_priors,
            motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
            likelihood=likelihood,
            max_nfev=max_nfev,
            compute_laplace_uncertainty=compute_laplace_uncertainty,
        )

    best = min(previews, key=lambda result: result.negative_log_posterior)
    return calibrate_bayesian(
        measurements.tdoa_s,
        measurements.event_times_s,
        microphone_count,
        tdoa_sigma_s=sigma,
        microphone_pairs=measurements.microphone_pairs,
        speed_of_sound=speed_of_sound,
        estimate_speed_of_sound=estimate_speed_of_sound,
        estimate_clock_offsets=estimate_clock_offsets,
        estimate_clock_drifts=estimate_clock_drifts,
        distance_priors=distance_priors,
        motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
        initial_microphones=best.microphone_positions,
        initial_sources=best.source_positions,
        initial_clock_offsets_s=best.clock_offsets_s if estimate_clock_offsets else None,
        initial_clock_drifts=best.clock_drifts if estimate_clock_drifts else None,
        likelihood=likelihood,
        max_nfev=max_nfev,
        compute_laplace_uncertainty=compute_laplace_uncertainty,
    )


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
    pair_mode: str = "redundant",
    reference_count: int = 2,
    microphone_pairs: Sequence[tuple[int, int]] | None = None,
    speed_of_sound: float = 343.0,
    motion_velocity_change_sigma_mps: float | None = 5.0,
    likelihood: str = "cauchy",
    estimate_clock_offsets: bool = False,
    estimate_clock_drifts: bool = False,
    estimate_speed_of_sound: bool = False,
    distance_priors: Sequence[DistancePrior] = (),
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    max_nfev: int = 4000,
    compute_laplace_uncertainty: bool = True,
) -> AudioCalibrationResult:
    """Self-calibrate from discrete broadband/transient emissions in synchronized audio.

    The frontend detects acoustic events and associates their arrivals across the
    microphones. Geometry initialization uses TDOA-derived microphone baseline
    lower bounds and several scale hypotheses, followed by robust joint MAP
    refinement. One 3-D source state is solved per detected event.
    """
    values = np.asarray(audio, dtype=float)
    if values.ndim != 2:
        raise ValueError("audio must have shape (samples, microphones)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    microphone_count = values.shape[1]
    if microphone_count < 4:
        raise ValueError("At least four microphones are required for 3-D calibration")

    if microphone_pairs is None:
        pairs = tuple(
            make_microphone_pairs(
                microphone_count,
                mode=pair_mode,
                reference_count=reference_count,
            )
        )
    else:
        pairs = tuple((int(a), int(b)) for a, b in microphone_pairs)

    detection = detect_transient_events(
        values,
        sample_rate,
        event_channel=event_channel,
        smooth_s=event_smooth_s,
        min_gap_s=event_min_gap_s,
        relative_prominence=event_relative_prominence,
    )
    measurements = estimate_event_tdoas(
        values,
        sample_rate,
        detection.event_samples,
        detection.event_channel,
        microphone_pairs=pairs,
        max_tau_s=max_tau_s,
        envelope_smooth_s=tdoa_envelope_smooth_s,
        template_s=tdoa_template_s,
        candidate_count=tdoa_candidate_count,
        max_tdoa_rate=max_tdoa_rate,
        transition_weight=tdoa_track_weight,
    )
    sigma = tdoa_sigma_from_confidence(
        measurements.confidence,
        sample_rate,
        best_sigma_samples=best_sigma_samples,
        worst_sigma_samples=worst_sigma_samples,
    )

    calibration = _solve_from_event_multistarts(
        measurements,
        microphone_count,
        sigma=sigma,
        speed_of_sound=speed_of_sound,
        motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
        likelihood=likelihood,
        estimate_clock_offsets=estimate_clock_offsets,
        estimate_clock_drifts=estimate_clock_drifts,
        estimate_speed_of_sound=estimate_speed_of_sound,
        distance_priors=distance_priors,
        max_nfev=max_nfev,
        compute_laplace_uncertainty=compute_laplace_uncertainty,
    )

    return AudioCalibrationResult(
        calibration=calibration,
        event_times_s=measurements.event_times_s,
        event_samples=measurements.event_samples,
        event_channel=measurements.event_channel,
        detected_event_count=len(detection.event_samples),
        tdoa_s=measurements.tdoa_s,
        tdoa_sigma_s=sigma,
        confidence=measurements.confidence,
        arrival_delays_s=measurements.arrival_delays_s,
        arrival_confidence=measurements.arrival_confidence,
        microphone_pairs=measurements.microphone_pairs,
    )
