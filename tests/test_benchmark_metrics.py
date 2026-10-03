from types import SimpleNamespace

import numpy as np

from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.chirps import make_chirp_scene


def test_source_error_and_missing_events_cannot_be_hidden_by_good_microphones():
    scene = make_chirp_scene(sample_rate_hz=48_000)
    calibration = SimpleNamespace(
        microphone_positions_m=scene.microphone_positions_m,
        microphone_ids=tuple(range(8)),
        source_positions_m=scene.source_positions_at_events_m.copy(),
        status="solved",
        tdoa_rms_s=1e-7,
    )
    assert scene_metrics(calibration, scene, scene.event_times_s)["standard_success"]
    calibration.source_positions_m[3] += 2
    row = scene_metrics(calibration, scene, scene.event_times_s)
    assert row["matched_events"] == 20 and not row["standard_success"]
    calibration.source_positions_m[3] = np.nan
    row = scene_metrics(calibration, scene, scene.event_times_s)
    assert row["unresolved_events"] == 1 and not row["standard_success"]


def test_planar_evaluation_allows_only_one_global_normal_reflection():
    scene = make_chirp_scene()
    reflected = scene.source_positions_at_events_m.copy()
    reflected[:, 1] *= -1
    calibration = SimpleNamespace(
        microphone_positions_m=scene.microphone_positions_m,
        microphone_ids=tuple(range(8)),
        source_positions_m=reflected,
        status="solved",
        tdoa_rms_s=0,
    )
    assert scene_metrics(calibration, scene, scene.event_times_s)["source_rms_error_m"] < 1e-10
    reflected[0, 1] *= -1
    assert not scene_metrics(calibration, scene, scene.event_times_s)["standard_success"]
