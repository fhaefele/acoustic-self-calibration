from __future__ import annotations

import sys

import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian, tdoa_sigma_from_confidence
from acoustic_self_calibration.events import detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.initialization import event_initial_scene_candidates
from acoustic_self_calibration.tdoa import make_microphone_pairs
from tests.test_end_to_end_audio import _myotis_layout_pulsed_scene


scale = float(sys.argv[1])
sample_rate, true_microphones, _, true_sources, audio = _myotis_layout_pulsed_scene()
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
selected: list[int] = []
for microphone in range(12):
    if microphone == 1:
        continue
    for column, (a, b) in enumerate(pairs):
        if {a, b} == {1, microphone}:
            selected.append(column)
            break
indices = np.asarray(selected, dtype=int)
star_tdoa = measurements.tdoa_s[:, indices]
star_sigma = sigma[:, indices]
star_pairs = tuple(pairs[index] for index in selected)
initial_microphones, initial_sources = event_initial_scene_candidates(
    measurements.arrival_delays_s,
    measurements.tdoa_s,
    sigma,
    pairs,
    speed_of_sound=343.0,
    scale_candidates=(scale,),
)[0]
preview = calibrate_bayesian(
    star_tdoa,
    measurements.event_times_s,
    12,
    tdoa_sigma_s=star_sigma,
    microphone_pairs=star_pairs,
    speed_of_sound=343.0,
    motion_velocity_change_sigma_mps=8.0,
    initial_microphones=initial_microphones,
    initial_sources=initial_sources,
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
aligned_microphones, rotation, translation = rigid_align(
    final.microphone_positions, true_microphones
)
aligned_sources = apply_rigid(final.source_positions, rotation, translation)
print(
    f"scale={scale:g} preview_post={preview.negative_log_posterior:.9f} "
    f"final_post={final.negative_log_posterior:.9f} "
    f"mic={rms_position_error(aligned_microphones, true_microphones):.9f} "
    f"src={rms_position_error(aligned_sources, true_sources):.9f} "
    f"nrms={final.normalized_data_rms:.9f} "
    f"tdoa_us={1e6 * final.rms_tdoa_residual_s:.6f} success={final.success}",
    flush=True,
)
