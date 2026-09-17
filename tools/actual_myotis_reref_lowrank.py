from __future__ import annotations

import sys

import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian, tdoa_sigma_from_confidence
from acoustic_self_calibration.events import detect_transient_events, estimate_event_tdoas
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.initialization import low_rank_initial_scene_hypotheses
from acoustic_self_calibration.pipeline import _global_low_rank_initial_scene
from acoustic_self_calibration.tdoa import make_microphone_pairs
from tests.test_end_to_end_audio import _myotis_layout_pulsed_scene


def rereferenced_differences(measurements, reference: int, microphone_count: int) -> tuple[np.ndarray, list[int]]:
    order = [reference] + [m for m in range(microphone_count) if m != reference]
    observed = np.empty((len(measurements.event_times_s), microphone_count - 1), dtype=float)
    for out_col, microphone in enumerate(order[1:]):
        for column, (a, b) in enumerate(measurements.microphone_pairs):
            if (a, b) == (reference, microphone):
                observed[:, out_col] = measurements.tdoa_s[:, column]
                break
            if (a, b) == (microphone, reference):
                observed[:, out_col] = -measurements.tdoa_s[:, column]
                break
        else:
            raise RuntimeError(f"missing edge {reference}-{microphone}")
    return 343.0 * observed, order


def unpermute(microphones_reordered: np.ndarray, order: list[int]) -> np.ndarray:
    microphones = np.empty_like(microphones_reordered)
    for reordered_index, original_index in enumerate(order):
        microphones[original_index] = microphones_reordered[reordered_index]
    return microphones


kind = sys.argv[1]
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
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
reference = 1
observed, order = rereferenced_differences(measurements, reference, 12)
if kind == "global":
    microphones_reordered, sources0 = _global_low_rank_initial_scene(observed, 30.0)
    label = "reref-global"
else:
    hypotheses = low_rank_initial_scene_hypotheses(
        observed,
        30.0,
        max_scene_hypotheses=4,
        subset_hypotheses=8,
    )
    microphones_reordered, sources0 = hypotheses[index]
    label = f"reref-subset-{index}"
microphones0 = unpermute(microphones_reordered, order)
selected: list[int] = []
for microphone in range(12):
    if microphone == reference:
        continue
    for column, (a, b) in enumerate(pairs):
        if {a, b} == {reference, microphone}:
            selected.append(column)
            break
indices = np.asarray(selected, dtype=int)
star_tdoa = measurements.tdoa_s[:, indices]
star_sigma = sigma[:, indices]
star_pairs = tuple(pairs[i] for i in selected)
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
def errors(result):
    aligned, rotation, translation = rigid_align(result.microphone_positions, true_microphones)
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)
    return rms_position_error(aligned, true_microphones), rms_position_error(aligned_sources, true_sources)
pre_mic, pre_src = errors(preview)
fin_mic, fin_src = errors(final)
print(
    f"{label} preview_post={preview.negative_log_posterior:.9f} preview_mic={pre_mic:.9f} "
    f"preview_src={pre_src:.9f} final_post={final.negative_log_posterior:.9f} "
    f"final_mic={fin_mic:.9f} final_src={fin_src:.9f} final_nrms={final.normalized_data_rms:.9f} "
    f"tdoa_us={1e6 * final.rms_tdoa_residual_s:.6f} success={final.success}",
    flush=True,
)
