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
from acoustic_self_calibration.simulation import (
    broadband_pulse_train,
    random_microphone_array,
    render_moving_source,
)
from acoustic_self_calibration.tdoa import make_microphone_pairs


def pulsed_scene() -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(108)
    sample_rate = 48_000
    duration = 5.0
    microphones = random_microphone_array(
        8,
        bounds=((-1.2, 1.2), (-1.2, 1.2), (0.0, 1.6)),
        rng=rng,
    )
    trajectory_times = np.linspace(0.0, duration, 31)
    phase = np.linspace(0.0, 2.0 * np.pi, len(trajectory_times))
    trajectory_positions = np.column_stack(
        [
            2.0 * np.cos(0.75 * phase),
            1.6 * np.sin(0.75 * phase),
            1.2 + 0.5 * np.sin(0.55 * phase + 0.2),
        ]
    )
    event_times = np.linspace(0.4, 4.4, 20)
    source_signal = broadband_pulse_train(
        event_times,
        sample_rate,
        duration,
        pulse_duration_s=0.0015,
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
        noise_std=1e-5,
        rng=rng,
    )
    source_at_events = np.column_stack(
        [
            np.interp(event_times, trajectory_times, trajectory_positions[:, dimension])
            for dimension in range(3)
        ]
    )
    return sample_rate, microphones, source_at_events, audio


def scene_errors(
    microphones: np.ndarray,
    sources: np.ndarray,
    true_microphones: np.ndarray,
    true_sources: np.ndarray,
) -> tuple[float, float]:
    aligned, rotation, translation = rigid_align(microphones, true_microphones)
    aligned_sources = apply_rigid(sources, rotation, translation)
    return (
        rms_position_error(aligned, true_microphones),
        rms_position_error(aligned_sources, true_sources),
    )


def raw_tdoa_rms(
    microphones: np.ndarray,
    sources: np.ndarray,
    observed: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> float:
    predicted = np.empty_like(observed)
    for column, (a, b) in enumerate(pairs):
        distance_a = np.linalg.norm(sources - microphones[a], axis=1)
        distance_b = np.linalg.norm(sources - microphones[b], axis=1)
        predicted[:, column] = (distance_b - distance_a) / 343.0
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def report(
    label: str,
    microphones: np.ndarray,
    sources: np.ndarray,
    true_microphones: np.ndarray,
    true_sources: np.ndarray,
    observed: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    posterior: float | None = None,
    normalized_rms: float | None = None,
) -> None:
    microphone_rms, source_rms = scene_errors(
        microphones,
        sources,
        true_microphones,
        true_sources,
    )
    fields = [
        label,
        f"mic_rms={microphone_rms:.9f}",
        f"source_rms={source_rms:.9f}",
        f"tdoa_rms_us={1e6 * raw_tdoa_rms(microphones, sources, observed, pairs):.6f}",
    ]
    if posterior is not None:
        fields.append(f"posterior={posterior:.9f}")
    if normalized_rms is not None:
        fields.append(f"normalized_rms={normalized_rms:.9f}")
    print(" ".join(fields), flush=True)


def solve(
    microphones: np.ndarray,
    sources: np.ndarray,
    observed: np.ndarray,
    event_times: np.ndarray,
    sigma: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    max_nfev: int,
):
    return calibrate_bayesian(
        observed,
        event_times,
        8,
        tdoa_sigma_s=sigma,
        microphone_pairs=pairs,
        speed_of_sound=343.0,
        motion_velocity_change_sigma_mps=5.0,
        initial_microphones=microphones,
        initial_sources=sources,
        likelihood="cauchy",
        max_nfev=max_nfev,
        compute_laplace_uncertainty=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("global", "subset", "baseline"))
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()

    sample_rate, true_microphones, true_sources, audio = pulsed_scene()
    pairs = tuple(make_microphone_pairs(8, mode="redundant", reference_count=2))
    detection = detect_transient_events(audio, sample_rate, min_gap_s=0.05)
    measurements = estimate_event_tdoas(
        audio,
        sample_rate,
        detection.event_samples,
        detection.event_channel,
        microphone_pairs=pairs,
        max_tau_s=0.02,
        template_s=0.002,
    )
    sigma = tdoa_sigma_from_confidence(
        measurements.confidence,
        sample_rate,
        best_sigma_samples=0.35,
        worst_sigma_samples=4.0,
    )

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    if args.kind == "global":
        star = _reference_star_range_differences(measurements, 8, 343.0)
        microphones, sources = _global_low_rank_initial_scene(star, 30.0)
        candidates.append(("global", microphones, sources))
    elif args.kind == "subset":
        star = _reference_star_range_differences(measurements, 8, 343.0)
        for index, (microphones, sources) in enumerate(
            low_rank_initial_scene_hypotheses(
                star,
                30.0,
                max_scene_hypotheses=4,
                subset_hypotheses=8,
            )
        ):
            candidates.append((f"subset_h{index}", microphones, sources))
    else:
        if args.scale is None:
            raise SystemExit("--scale is required for baseline")
        microphones, sources = event_initial_scene_candidates(
            measurements.arrival_delays_s,
            measurements.tdoa_s,
            sigma,
            pairs,
            speed_of_sound=343.0,
            scale_candidates=(args.scale,),
        )[0]
        candidates.append((f"baseline_{args.scale:g}", microphones, sources))

    for name, microphones, sources in candidates:
        report(
            f"INIT {name}",
            microphones,
            sources,
            true_microphones,
            true_sources,
            measurements.tdoa_s,
            pairs,
        )
        preview = solve(
            microphones,
            sources,
            measurements.tdoa_s,
            measurements.event_times_s,
            sigma,
            pairs,
            250,
        )
        report(
            f"PREVIEW {name}",
            preview.microphone_positions,
            preview.source_positions,
            true_microphones,
            true_sources,
            measurements.tdoa_s,
            pairs,
            preview.negative_log_posterior,
            preview.normalized_data_rms,
        )

        if args.kind == "baseline":
            final_legacy = solve(
                preview.microphone_positions,
                preview.source_positions,
                measurements.tdoa_s,
                measurements.event_times_s,
                sigma,
                pairs,
                400,
            )
            report(
                f"FINAL_LEGACY {name}",
                final_legacy.microphone_positions,
                final_legacy.source_positions,
                true_microphones,
                true_sources,
                measurements.tdoa_s,
                pairs,
                final_legacy.negative_log_posterior,
                final_legacy.normalized_data_rms,
            )

            redundancy_scale = np.sqrt(len(pairs) / 7.0)
            final_current = solve(
                preview.microphone_positions,
                preview.source_positions,
                measurements.tdoa_s,
                measurements.event_times_s,
                sigma * redundancy_scale,
                pairs,
                400,
            )
            report(
                f"FINAL_CURRENT_WEIGHTING {name}",
                final_current.microphone_positions,
                final_current.source_positions,
                true_microphones,
                true_sources,
                measurements.tdoa_s,
                pairs,
                final_current.negative_log_posterior,
                final_current.normalized_data_rms,
            )


if __name__ == "__main__":
    main()
