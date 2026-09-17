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
from .initialization import (
    _fit_global_reference_ranges,
    _metric_upgrade_scene,
    _range_bounds,
    event_initial_scene_candidates,
    low_rank_initial_scene_hypotheses,
)
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


def _spanning_tree_measurements(
    measurements: EventTDOAMeasurements,
    sigma: np.ndarray,
    microphone_count: int,
) -> tuple[EventTDOAMeasurements, np.ndarray]:
    """Keep one independent TDOA edge per microphone-tree connection.

    Event pairwise TDOAs are derived from the same per-channel arrival delays, so
    cycles in the requested graph are exactly linearly dependent and their errors
    are correlated. Feeding every edge to a diagonal-noise MAP model overcounts
    those arrivals and can pull the geometry into a lower-residual but incorrect
    basin. A deterministic spanning tree preserves all microphones without
    double-counting cycle constraints. Default pair order yields the mic-0 star.
    """
    values = np.asarray(sigma, dtype=float)
    if values.shape != measurements.tdoa_s.shape:
        raise ValueError("sigma must match event TDOA measurements")

    parent = list(range(microphone_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    selected: list[int] = []
    for column, (a, b) in enumerate(measurements.microphone_pairs):
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        selected.append(column)
        if len(selected) == microphone_count - 1:
            break

    if len(selected) != microphone_count - 1:
        raise ValueError("microphone_pairs must connect every microphone")

    indices = np.asarray(selected, dtype=int)
    independent = EventTDOAMeasurements(
        event_samples=measurements.event_samples,
        event_times_s=measurements.event_times_s,
        event_channel=measurements.event_channel,
        arrival_delays_s=measurements.arrival_delays_s,
        arrival_confidence=measurements.arrival_confidence,
        tdoa_s=measurements.tdoa_s[:, indices],
        confidence=measurements.confidence[:, indices],
        microphone_pairs=tuple(measurements.microphone_pairs[index] for index in selected),
    )
    return independent, values[:, indices]


def _reference_star_range_differences(
    measurements: EventTDOAMeasurements,
    microphone_count: int,
    speed_of_sound: float,
) -> np.ndarray:
    """Extract mic-0 range differences for low-rank structure-from-sound initialization."""
    star = np.empty((len(measurements.event_times_s), microphone_count - 1), dtype=float)
    for microphone in range(1, microphone_count):
        for column, (a, b) in enumerate(measurements.microphone_pairs):
            if (a, b) == (0, microphone):
                star[:, microphone - 1] = measurements.tdoa_s[:, column]
                break
            if (a, b) == (microphone, 0):
                star[:, microphone - 1] = -measurements.tdoa_s[:, column]
                break
        else:
            raise ValueError(
                f"low-rank initialization requires a TDOA edge between mic 0 and mic {microphone}"
            )
    return float(speed_of_sound) * star


def _global_low_rank_initial_scene(
    observed_range_differences: np.ndarray,
    position_bound_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the pre-hypothesis global rank-three initialization basin.

    The Åström-style subset generator adds useful alternatives, but its compacted
    rank score is not directly comparable with the full doubly-centered rank
    objective. Selecting all candidates in one pool can therefore discard the
    strongest global solution before metric upgrade. Keep that global basin as an
    explicit hypothesis and let the MAP previews compete it against subset and
    baseline starts.
    """
    observed = np.asarray(observed_range_differences, dtype=float)
    event_count = observed.shape[0]
    differences = np.vstack([np.zeros(event_count), observed.T])
    lower, upper, aperture = _range_bounds(differences, position_bound_m)

    fits: list[tuple[float, np.ndarray]] = []
    for multiplier in (0.25, 0.75, 1.5, 3.0):
        initial = np.clip(lower + multiplier * aperture, lower, upper)
        score, reference_ranges = _fit_global_reference_ranges(
            differences,
            initial,
            lower,
            upper,
        )
        if np.isfinite(score):
            fits.append((score, reference_ranges))
    if not fits:
        raise ValueError("global low-rank initialization did not produce a finite range fit")
    _, reference_ranges = min(fits, key=lambda item: item[0])
    return _metric_upgrade_scene(differences, reference_ranges)


def _preview_calibration(
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
    initial_microphones: np.ndarray | None = None,
    initial_sources: np.ndarray | None = None,
) -> BayesianCalibrationResult:
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
        initial_microphones=initial_microphones,
        initial_sources=initial_sources,
        likelihood=likelihood,
        max_nfev=max_nfev,
        compute_laplace_uncertainty=False,
    )


def _hypothesis_selection_key(
    result: BayesianCalibrationResult,
) -> tuple[float, float]:
    """Rank converged basins by full posterior, then data RMS.

    Geometry and source positions can trade off while explaining the same TDOAs,
    so raw TDOA RMS alone is not a reliable self-calibration discriminator. The
    posterior includes the source-motion and metric priors that distinguish those
    basins; normalized data RMS is used only as a deterministic tie-break.
    """
    return result.negative_log_posterior, result.normalized_data_rms


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
    solver_measurements, solver_sigma = _spanning_tree_measurements(
        measurements,
        sigma,
        microphone_count,
    )
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Retain the strongest full-data rank-three solution as its own family. This
    # was the useful basin before subset hypotheses were added and must not be
    # eliminated by a compacted-rank surrogate score.
    star: np.ndarray | None = None
    try:
        star = _reference_star_range_differences(
            measurements,
            microphone_count,
            speed_of_sound,
        )
        microphones0, sources0 = _global_low_rank_initial_scene(star, 30.0)
        candidates.append(("global_low_rank", microphones0, sources0))
    except (ValueError, np.linalg.LinAlgError):
        pass

    # Add the Åström-style subset -> compaction -> extension -> full-rank-refinement
    # hypotheses. They compete with, rather than replace, the global basin.
    if star is not None:
        try:
            low_rank_count = 4 if microphone_count <= 12 else 2
            subset_count = 8 if microphone_count <= 12 else 4
            candidates.extend(
                ("subset_low_rank", microphones0, sources0)
                for microphones0, sources0 in low_rank_initial_scene_hypotheses(
                    star,
                    30.0,
                    max_scene_hypotheses=low_rank_count,
                    subset_hypotheses=subset_count,
                )
            )
        except (ValueError, np.linalg.LinAlgError):
            pass

    # The maximum observed pairwise range difference is only a lower bound on the
    # microphone baseline. Sparse source coverage can leave a substantial scale
    # gap, especially for planar/one-sided trajectories, so keep a wider scale set
    # near the minimum microphone count instead of asking one MDS scale to stand in
    # for the whole independent initialization family.
    scales = (0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.1) if microphone_count <= 12 else (0.9, 1.2)
    try:
        candidates.extend(
            ("baseline", microphones0, sources0)
            for microphones0, sources0 in event_initial_scene_candidates(
                measurements.arrival_delays_s,
                solver_measurements.tdoa_s,
                solver_sigma,
                solver_measurements.microphone_pairs,
                speed_of_sound=speed_of_sound,
                scale_candidates=scales,
            )
        )
    except (ValueError, np.linalg.LinAlgError):
        pass

    preview_budget = min(max_nfev, 250)
    previews: list[tuple[str, BayesianCalibrationResult]] = []
    preview_likelihood = "gaussian" if likelihood == "cauchy" else likelihood
    for family, microphones0, sources0 in candidates:
        try:
            preview = _preview_calibration(
                solver_measurements,
                microphone_count,
                sigma=solver_sigma,
                speed_of_sound=speed_of_sound,
                motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
                likelihood=preview_likelihood,
                estimate_clock_offsets=estimate_clock_offsets,
                estimate_clock_drifts=estimate_clock_drifts,
                estimate_speed_of_sound=estimate_speed_of_sound,
                distance_priors=distance_priors,
                max_nfev=preview_budget,
                initial_microphones=microphones0,
                initial_sources=sources0,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(preview.negative_log_posterior) and np.isfinite(preview.normalized_data_rms):
            previews.append((family, preview))

    if not previews:
        return calibrate_bayesian(
            solver_measurements.tdoa_s,
            solver_measurements.event_times_s,
            microphone_count,
            tdoa_sigma_s=solver_sigma,
            microphone_pairs=solver_measurements.microphone_pairs,
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

    # Gaussian continuation keeps large residuals active while the geometry basin
    # settles. Preserve the low-rank construction families, then carry several
    # independently scaled MDS basins through the requested robust MAP solve.
    previews.sort(key=lambda item: _hypothesis_selection_key(item[1]))
    seeds: list[BayesianCalibrationResult] = []
    for family in ("global_low_rank", "subset_low_rank"):
        for preview_family, preview in previews:
            if preview_family == family:
                seeds.append(preview)
                break

    baseline_previews = [
        preview for preview_family, preview in previews if preview_family == "baseline"
    ]
    baseline_keep = 3 if microphone_count <= 12 else 1
    seeds.extend(baseline_previews[:baseline_keep])

    finalist_target = 5 if microphone_count <= 12 else 2
    for _, preview in previews:
        if any(preview is seed for seed in seeds):
            continue
        seeds.append(preview)
        if len(seeds) >= finalist_target:
            break
    seeds = seeds[:finalist_target]

    finalists: list[BayesianCalibrationResult] = []
    for seed in seeds:
        finalist = calibrate_bayesian(
            solver_measurements.tdoa_s,
            solver_measurements.event_times_s,
            microphone_count,
            tdoa_sigma_s=solver_sigma,
            microphone_pairs=solver_measurements.microphone_pairs,
            speed_of_sound=speed_of_sound,
            estimate_speed_of_sound=estimate_speed_of_sound,
            estimate_clock_offsets=estimate_clock_offsets,
            estimate_clock_drifts=estimate_clock_drifts,
            distance_priors=distance_priors,
            motion_velocity_change_sigma_mps=motion_velocity_change_sigma_mps,
            initial_microphones=seed.microphone_positions,
            initial_sources=seed.source_positions,
            initial_clock_offsets_s=seed.clock_offsets_s if estimate_clock_offsets else None,
            initial_clock_drifts=seed.clock_drifts if estimate_clock_drifts else None,
            likelihood=likelihood,
            max_nfev=max_nfev,
            compute_laplace_uncertainty=compute_laplace_uncertainty,
        )
        if np.isfinite(finalist.negative_log_posterior) and np.isfinite(
            finalist.normalized_data_rms
        ):
            finalists.append(finalist)

    if finalists:
        return min(finalists, key=_hypothesis_selection_key)
    return previews[0][1]


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
    microphones. Geometry initialization generates several rank-constrained
    structure-from-sound hypotheses plus TDOA-baseline hypotheses, then robustly
    refines the strongest basins with the joint Bayesian/MAP geometry model. One
    3-D source state is solved per detected event.
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
