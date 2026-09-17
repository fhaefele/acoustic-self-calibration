from __future__ import annotations

import sys

import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian, tdoa_sigma_from_confidence
from acoustic_self_calibration.events import detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.initialization import (
    _deterministic_hypothesis_subsets,
    _extend_subset_offsets,
    _fit_global_reference_ranges,
    _fit_subset_offsets,
    _metric_upgrade_scene,
    _range_bounds,
)
from acoustic_self_calibration.pipeline import _reference_star_range_differences
from acoustic_self_calibration.tdoa import make_microphone_pairs


def load_scene():
    namespace: dict[str, object] = {"__name__": "diagnostic_test_helpers"}
    source = open("tests/test_end_to_end_audio.py", encoding="utf-8").read()
    exec(compile(source, "tests/test_end_to_end_audio.py", "exec"), namespace)
    return namespace["_myotis_layout_pulsed_scene"]()


def all_unique_range_candidates(observed: np.ndarray):
    event_count = observed.shape[0]
    differences = np.vstack([np.zeros(event_count), observed.T])
    lower, upper, aperture = _range_bounds(differences, 30.0)
    candidates: list[tuple[str, float, np.ndarray]] = []

    for multiplier in (0.25, 0.75, 1.5, 3.0):
        initial = np.clip(lower + multiplier * aperture, lower, upper)
        score, ranges = _fit_global_reference_ranges(differences, initial, lower, upper)
        if np.isfinite(score):
            candidates.append((f"global-{multiplier:g}", score, ranges))

    for subset_index, (microphone_indices, event_indices) in enumerate(
        _deterministic_hypothesis_subsets(differences, max_hypotheses=8)
    ):
        subset = differences[np.ix_(microphone_indices, event_indices)]
        subset_lower, subset_upper, subset_aperture = _range_bounds(subset, 30.0)
        for multiplier in (0.5, 1.5):
            subset_initial = np.clip(
                subset_lower + multiplier * subset_aperture,
                subset_lower,
                subset_upper,
            )
            subset_ranges = _fit_subset_offsets(
                subset, subset_lower, subset_upper, subset_initial
            )
            extended = _extend_subset_offsets(
                differences,
                microphone_indices,
                event_indices,
                subset_ranges,
                lower,
                upper,
            )
            score, refined = _fit_global_reference_ranges(
                differences,
                extended,
                lower,
                upper,
                maxiter=220,
            )
            if np.isfinite(score):
                candidates.append(
                    (f"subset-{subset_index}-m{multiplier:g}", score, refined)
                )

    candidates.sort(key=lambda item: item[1])
    unique: list[tuple[str, float, np.ndarray]] = []
    for label, score, ranges in candidates:
        if any(
            np.linalg.norm(ranges - existing[2])
            <= 1e-3 * max(1.0, np.linalg.norm(existing[2]))
            for existing in unique
        ):
            continue
        unique.append((label, score, ranges))
    return differences, unique


def errors(result, true_microphones, true_sources):
    aligned, rotation, translation = rigid_align(
        result.microphone_positions, true_microphones
    )
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)
    return (
        rms_position_error(aligned, true_microphones),
        rms_position_error(aligned_sources, true_sources),
    )


def main() -> None:
    index = int(sys.argv[1])
    sample_rate, true_microphones, _, true_sources, audio = load_scene()
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

    observed = _reference_star_range_differences(measurements, microphone_count, 343.0)
    differences, unique = all_unique_range_candidates(observed)
    print(f"unique_count={len(unique)}", flush=True)
    if index >= len(unique):
        print(f"SKIP index={index}", flush=True)
        return

    label, algebraic_score, reference_ranges = unique[index]
    microphones0, sources0 = _metric_upgrade_scene(differences, reference_ranges)
    latent_ranges = differences + reference_ranges[None, :]
    euclidean_ranges = np.linalg.norm(
        microphones0[:, None, :] - sources0[None, :, :], axis=2
    )
    metric_rms = float(np.sqrt(np.mean((euclidean_ranges - latent_ranges) ** 2)))

    reference = int(measurements.event_channel)
    selected: list[int] = []
    for microphone in range(microphone_count):
        if microphone == reference:
            continue
        for column, (a, b) in enumerate(pairs):
            if {a, b} == {reference, microphone}:
                selected.append(column)
                break
        else:
            selected.clear()
            break
    if len(selected) != microphone_count - 1:
        raise RuntimeError("event channel does not have a complete reference star")
    indices = np.asarray(selected, dtype=int)
    star_tdoa = measurements.tdoa_s[:, indices]
    star_sigma = sigma[:, indices]
    star_pairs = tuple(pairs[column] for column in selected)

    preview = calibrate_bayesian(
        star_tdoa,
        measurements.event_times_s,
        microphone_count,
        tdoa_sigma_s=star_sigma,
        microphone_pairs=star_pairs,
        speed_of_sound=343.0,
        motion_velocity_change_sigma_mps=8.0,
        initial_microphones=microphones0,
        initial_sources=sources0,
        likelihood="gaussian",
        max_nfev=250,
        compute_laplace_uncertainty=False,
    )
    final = calibrate_bayesian(
        star_tdoa,
        measurements.event_times_s,
        microphone_count,
        tdoa_sigma_s=star_sigma,
        microphone_pairs=star_pairs,
        speed_of_sound=343.0,
        motion_velocity_change_sigma_mps=8.0,
        initial_microphones=preview.microphone_positions,
        initial_sources=preview.source_positions,
        likelihood="cauchy",
        max_nfev=500,
        compute_laplace_uncertainty=False,
    )
    p_mic, p_src = errors(preview, true_microphones, true_sources)
    f_mic, f_src = errors(final, true_microphones, true_sources)
    print(
        f"RESULT index={index} label={label} algebraic={algebraic_score:.12g} "
        f"metric_rms={metric_rms:.9f} preview_post={preview.negative_log_posterior:.9f} "
        f"preview_mic={p_mic:.9f} preview_src={p_src:.9f} "
        f"final_post={final.negative_log_posterior:.9f} final_nrms={final.normalized_data_rms:.9f} "
        f"final_mic={f_mic:.9f} final_src={f_src:.9f} "
        f"tdoa_us={1e6 * final.rms_tdoa_residual_s:.6f} success={final.success}",
        flush=True,
    )


if __name__ == "__main__":
    main()
