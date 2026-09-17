from __future__ import annotations

import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian, tdoa_sigma_from_confidence
from acoustic_self_calibration.events import EventTDOAMeasurements, detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.initialization import event_initial_scene_candidates, low_rank_initial_scene_hypotheses
from acoustic_self_calibration.pipeline import (
    _global_low_rank_initial_scene,
    _hypothesis_selection_key,
    _preview_calibration,
    _reference_star_range_differences,
)
from acoustic_self_calibration.tdoa import make_microphone_pairs
from tests.test_end_to_end_audio import _myotis_layout_pulsed_scene


def best_complete_star(
    measurements: EventTDOAMeasurements,
    sigma: np.ndarray,
    microphone_count: int,
) -> tuple[EventTDOAMeasurements, np.ndarray, int]:
    values = np.asarray(sigma, dtype=float)
    candidates: list[tuple[float, int, list[int]]] = []
    for reference in range(microphone_count):
        columns: list[int] = []
        for microphone in range(microphone_count):
            if microphone == reference:
                continue
            for column, (a, b) in enumerate(measurements.microphone_pairs):
                if {a, b} == {reference, microphone}:
                    columns.append(column)
                    break
            else:
                columns.clear()
                break
        if len(columns) == microphone_count - 1:
            candidates.append((float(np.mean(values[:, columns] ** 2)), reference, columns))
    if not candidates:
        raise RuntimeError("no complete reference star")
    score, reference, selected = min(candidates)
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
    print(
        f"selected_reference={reference} star_rms_sigma_us={1e6 * np.sqrt(score):.9f} "
        f"pairs={independent.microphone_pairs}",
        flush=True,
    )
    return independent, values[:, indices], reference


def errors(result, true_microphones, true_sources):
    aligned, rotation, translation = rigid_align(result.microphone_positions, true_microphones)
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)
    return (
        rms_position_error(aligned, true_microphones),
        rms_position_error(aligned_sources, true_sources),
    )


def main() -> None:
    sample_rate, true_microphones, _, true_sources, audio = _myotis_layout_pulsed_scene()
    microphone_count = len(true_microphones)
    pairs = tuple(make_microphone_pairs(microphone_count, mode="redundant", reference_count=2))
    detection = detect_transient_events(audio, sample_rate, min_gap_s=0.05)
    measurements = estimate_event_tdoas(
        audio,
        sample_rate,
        detection.event_samples,
        detection.event_channel,
        microphone_pairs=pairs,
        max_tau_s=0.012,
        template_s=0.0018,
    )
    sigma = tdoa_sigma_from_confidence(
        measurements.confidence,
        sample_rate,
        best_sigma_samples=0.35,
        worst_sigma_samples=4.0,
    )
    preview_measurements, preview_sigma, _ = best_complete_star(
        measurements, sigma, microphone_count
    )

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    star = _reference_star_range_differences(measurements, microphone_count, 343.0)
    microphones0, sources0 = _global_low_rank_initial_scene(star, 30.0)
    candidates.append(("global", microphones0, sources0))
    for index, (microphones0, sources0) in enumerate(
        low_rank_initial_scene_hypotheses(
            star,
            30.0,
            max_scene_hypotheses=4,
            subset_hypotheses=8,
        )
    ):
        candidates.append((f"subset-{index}", microphones0, sources0))

    scales = (0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.1)
    baseline = event_initial_scene_candidates(
        measurements.arrival_delays_s,
        measurements.tdoa_s,
        sigma,
        measurements.microphone_pairs,
        speed_of_sound=343.0,
        scale_candidates=scales,
    )
    for scale, (microphones0, sources0) in zip(scales, baseline, strict=True):
        candidates.append((f"baseline-{scale:g}", microphones0, sources0))

    previews: list[tuple[str, object]] = []
    for label, microphones0, sources0 in candidates:
        preview = _preview_calibration(
            preview_measurements,
            microphone_count,
            sigma=preview_sigma,
            speed_of_sound=343.0,
            motion_velocity_change_sigma_mps=8.0,
            likelihood="gaussian",
            estimate_clock_offsets=False,
            estimate_clock_drifts=False,
            estimate_speed_of_sound=False,
            distance_priors=(),
            max_nfev=250,
            initial_microphones=microphones0,
            initial_sources=sources0,
        )
        mic, src = errors(preview, true_microphones, true_sources)
        print(
            f"PREVIEW {label} post={preview.negative_log_posterior:.9f} "
            f"nrms={preview.normalized_data_rms:.9f} mic={mic:.9f} src={src:.9f} "
            f"success={preview.success}",
            flush=True,
        )
        previews.append((label, preview))

    previews.sort(key=lambda item: _hypothesis_selection_key(item[1]))
    seeds: list[tuple[str, object]] = []
    for family_prefix in ("global", "subset-"):
        for label, preview in previews:
            if label == family_prefix or label.startswith(family_prefix):
                seeds.append((label, preview))
                break
    baseline_previews = [item for item in previews if item[0].startswith("baseline-")]
    seeds.extend(baseline_previews[:3])
    for item in previews:
        if any(item[1] is seed[1] for seed in seeds):
            continue
        seeds.append(item)
        if len(seeds) >= 5:
            break
    seeds = seeds[:5]
    print("SEEDS " + ", ".join(label for label, _ in seeds), flush=True)

    finalists = []
    for label, seed in seeds:
        final = calibrate_bayesian(
            preview_measurements.tdoa_s,
            preview_measurements.event_times_s,
            microphone_count,
            tdoa_sigma_s=preview_sigma,
            microphone_pairs=preview_measurements.microphone_pairs,
            speed_of_sound=343.0,
            motion_velocity_change_sigma_mps=8.0,
            initial_microphones=seed.microphone_positions,
            initial_sources=seed.source_positions,
            likelihood="cauchy",
            max_nfev=500,
            compute_laplace_uncertainty=False,
        )
        mic, src = errors(final, true_microphones, true_sources)
        print(
            f"FINAL {label} post={final.negative_log_posterior:.9f} "
            f"nrms={final.normalized_data_rms:.9f} mic={mic:.9f} src={src:.9f} "
            f"tdoa_us={1e6 * final.rms_tdoa_residual_s:.6f} success={final.success}",
            flush=True,
        )
        finalists.append((label, final))
    best_label, best = min(finalists, key=lambda item: _hypothesis_selection_key(item[1]))
    mic, src = errors(best, true_microphones, true_sources)
    print(
        f"BEST {best_label} post={best.negative_log_posterior:.9f} mic={mic:.9f} src={src:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
