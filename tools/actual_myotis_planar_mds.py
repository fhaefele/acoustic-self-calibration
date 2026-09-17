from __future__ import annotations

import sys

import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian, tdoa_sigma_from_confidence
from acoustic_self_calibration.events import detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.initialization import _classical_mds, _localize_sources
from acoustic_self_calibration.tdoa import make_microphone_pairs


def load_scene():
    namespace: dict[str, object] = {"__name__": "diagnostic_test_helpers"}
    source = open("tests/test_end_to_end_audio.py", encoding="utf-8").read()
    exec(compile(source, "tests/test_end_to_end_audio.py", "exec"), namespace)
    return namespace["_myotis_layout_pulsed_scene"]()


def errors(result, true_microphones, true_sources):
    aligned, rotation, translation = rigid_align(result.microphone_positions, true_microphones)
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)
    return (
        rms_position_error(aligned, true_microphones),
        rms_position_error(aligned_sources, true_sources),
    )


def main() -> None:
    scale = float(sys.argv[1])
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

    arrival = measurements.arrival_delays_s
    baselines = np.zeros((microphone_count, microphone_count), dtype=float)
    for a in range(microphone_count):
        for b in range(a + 1, microphone_count):
            lower_bound = 343.0 * float(np.max(np.abs(arrival[:, b] - arrival[:, a])))
            baselines[a, b] = lower_bound
            baselines[b, a] = lower_bound
    microphones = _classical_mds(baselines) * scale
    center = microphones.mean(axis=0)
    centered = microphones - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    plane = vt[:2]
    microphones_planar = center + (centered @ plane.T) @ plane
    sources0 = _localize_sources(
        microphones_planar,
        measurements.tdoa_s,
        sigma,
        pairs,
        343.0,
        30.0,
    )

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
            raise RuntimeError("event reference has no complete star")
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
        initial_microphones=microphones_planar,
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
    init_mic, init_src = errors(
        type("R", (), {"microphone_positions": microphones_planar, "source_positions": sources0})(),
        true_microphones,
        true_sources,
    )
    preview_mic, preview_src = errors(preview, true_microphones, true_sources)
    final_mic, final_src = errors(final, true_microphones, true_sources)
    print(
        f"scale={scale:g} mds_singular={singular.tolist()} init_mic={init_mic:.9f} init_src={init_src:.9f} "
        f"preview_post={preview.negative_log_posterior:.9f} preview_mic={preview_mic:.9f} preview_src={preview_src:.9f} "
        f"final_post={final.negative_log_posterior:.9f} final_nrms={final.normalized_data_rms:.9f} "
        f"final_mic={final_mic:.9f} final_src={final_src:.9f} success={final.success}",
        flush=True,
    )


if __name__ == "__main__":
    main()
