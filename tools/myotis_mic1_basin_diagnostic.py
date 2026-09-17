from __future__ import annotations

import argparse

import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian, tdoa_sigma_from_confidence
from acoustic_self_calibration.events import detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.initialization import (
    event_initial_scene_candidates,
    low_rank_initial_scene_hypotheses,
)
from acoustic_self_calibration.pipeline import (
    _global_low_rank_initial_scene,
    _reference_star_range_differences,
)
from acoustic_self_calibration.simulation import broadband_pulse_train, render_moving_source
from acoustic_self_calibration.tdoa import make_microphone_pairs


def scene():
    rng = np.random.default_rng(812)
    sample_rate = 96_000
    duration = 2.4
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [1.2, 0.0, 0.8],
            [1.2, 0.0, 0.4],
            [1.2, 0.0, -0.4],
            [1.2, 0.0, -0.8],
            [1.2, 0.0, -1.2],
        ],
        dtype=float,
    )
    trajectory_times = np.linspace(0.0, duration, 31)
    progress = trajectory_times / duration
    trajectory_positions = np.column_stack(
        [
            1.03 * (1.0 - progress) ** 1.25,
            2.70 - 1.90 * progress + 0.08 * np.sin(2.0 * np.pi * progress),
            -0.25 + 0.23 * progress + 0.02 * np.sin(3.0 * np.pi * progress),
        ]
    )
    event_times = np.linspace(0.25, 2.05, 18)
    source_signal = broadband_pulse_train(
        event_times,
        sample_rate,
        duration,
        pulse_duration_s=0.0012,
        rng=rng,
    )
    source_forward = microphones.mean(axis=0)[None, :] - trajectory_positions
    source_forward /= np.linalg.norm(source_forward, axis=1, keepdims=True)
    audio = render_moving_source(
        source_signal,
        sample_rate,
        microphones,
        trajectory_times,
        trajectory_positions,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=8e-6,
        rng=rng,
    )
    source_at_events = np.column_stack(
        [
            np.interp(event_times, trajectory_times, trajectory_positions[:, dimension])
            for dimension in range(3)
        ]
    )
    return sample_rate, microphones, source_at_events, audio


def errors(result, true_microphones, true_sources):
    aligned, rotation, translation = rigid_align(result.microphone_positions, true_microphones)
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)
    return (
        rms_position_error(aligned, true_microphones),
        rms_position_error(aligned_sources, true_sources),
    )


def star_view(measurements, sigma, reference: int):
    selected: list[int] = []
    for microphone in range(12):
        if microphone == reference:
            continue
        for column, (a, b) in enumerate(measurements.microphone_pairs):
            if {a, b} == {reference, microphone}:
                selected.append(column)
                break
        else:
            raise RuntimeError(f"missing star edge {reference}-{microphone}")
    indices = np.asarray(selected, dtype=int)
    return (
        measurements.tdoa_s[:, indices],
        sigma[:, indices],
        tuple(measurements.microphone_pairs[index] for index in selected),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("global", "subset", "mds"))
    parser.add_argument("value", type=float, nargs="?", default=0.0)
    args = parser.parse_args()

    sample_rate, true_microphones, true_sources, audio = scene()
    pairs = tuple(make_microphone_pairs(12, mode="redundant", reference_count=2))
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
    star_tdoa, star_sigma, star_pairs = star_view(measurements, sigma, 1)

    if args.kind == "global":
        star = _reference_star_range_differences(measurements, 12, 343.0)
        microphones0, sources0 = _global_low_rank_initial_scene(star, 30.0)
        label = "global"
    elif args.kind == "subset":
        star = _reference_star_range_differences(measurements, 12, 343.0)
        hypotheses = low_rank_initial_scene_hypotheses(
            star,
            30.0,
            max_scene_hypotheses=4,
            subset_hypotheses=8,
        )
        index = int(args.value)
        microphones0, sources0 = hypotheses[index]
        label = f"subset-{index}"
    else:
        scale = float(args.value)
        microphones0, sources0 = event_initial_scene_candidates(
            measurements.arrival_delays_s,
            measurements.tdoa_s,
            sigma,
            pairs,
            speed_of_sound=343.0,
            scale_candidates=(scale,),
        )[0]
        label = f"mds-{scale:g}"

    preview = calibrate_bayesian(
        star_tdoa,
        measurements.event_times_s,
        12,
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
        12,
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
    preview_mic, preview_src = errors(preview, true_microphones, true_sources)
    final_mic, final_src = errors(final, true_microphones, true_sources)
    print(
        f"{label} preview_mic={preview_mic:.9f} preview_src={preview_src:.9f} "
        f"preview_post={preview.negative_log_posterior:.9f} "
        f"preview_nrms={preview.normalized_data_rms:.9f} "
        f"final_mic={final_mic:.9f} final_src={final_src:.9f} "
        f"final_post={final.negative_log_posterior:.9f} "
        f"final_nrms={final.normalized_data_rms:.9f} "
        f"final_tdoa_us={1e6 * final.rms_tdoa_residual_s:.6f} success={final.success}",
        flush=True,
    )


if __name__ == "__main__":
    main()
