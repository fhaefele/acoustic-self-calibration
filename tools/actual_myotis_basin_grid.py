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
preview_motion = float(sys.argv[2])
sample_rate, true_microphones, _, true_sources, audio = _myotis_layout_pulsed_scene()
pairs = tuple(make_microphone_pairs(12, mode="redundant", reference_count=2))
detection = detect_transient_events(audio, sample_rate, min_gap_s=0.05)
measurements = estimate_event_tdoas(audio, sample_rate, detection.event_samples, detection.event_channel,
    microphone_pairs=pairs, max_tau_s=0.012, template_s=0.0018)
sigma = tdoa_sigma_from_confidence(measurements.confidence, sample_rate,
    best_sigma_samples=0.35, worst_sigma_samples=4.0)
reference = 1
selected=[]
for microphone in range(12):
    if microphone == reference: continue
    for column,(a,b) in enumerate(pairs):
        if {a,b} == {reference,microphone}:
            selected.append(column); break
idx=np.asarray(selected,dtype=int)
star_tdoa=measurements.tdoa_s[:,idx]
star_sigma=sigma[:,idx]
star_pairs=tuple(pairs[i] for i in selected)
m0,s0=event_initial_scene_candidates(measurements.arrival_delays_s, measurements.tdoa_s, sigma, pairs,
    speed_of_sound=343.0, scale_candidates=(scale,))[0]
preview=calibrate_bayesian(star_tdoa, measurements.event_times_s, 12,
    tdoa_sigma_s=star_sigma, microphone_pairs=star_pairs, speed_of_sound=343.0,
    motion_velocity_change_sigma_mps=preview_motion, initial_microphones=m0, initial_sources=s0,
    likelihood="gaussian", max_nfev=500, compute_laplace_uncertainty=False)
final=calibrate_bayesian(star_tdoa, measurements.event_times_s, 12,
    tdoa_sigma_s=star_sigma, microphone_pairs=star_pairs, speed_of_sound=343.0,
    motion_velocity_change_sigma_mps=8.0, initial_microphones=preview.microphone_positions,
    initial_sources=preview.source_positions, likelihood="cauchy", max_nfev=500,
    compute_laplace_uncertainty=False)
def errs(r):
    am,R,t=rigid_align(r.microphone_positions,true_microphones)
    asrc=apply_rigid(r.source_positions,R,t)
    return rms_position_error(am,true_microphones),rms_position_error(asrc,true_sources)
pm,ps=errs(preview); fm,fs=errs(final)
print(f"scale={scale:g} pmotion={preview_motion:g} ppost={preview.negative_log_posterior:.9f} pmic={pm:.9f} psrc={ps:.9f} fpost={final.negative_log_posterior:.9f} fmic={fm:.9f} fsrc={fs:.9f} fnrms={final.normalized_data_rms:.9f} success={final.success}", flush=True)
