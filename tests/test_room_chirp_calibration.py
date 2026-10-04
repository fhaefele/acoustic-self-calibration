import numpy as np
import pytest
from scipy.io import wavfile
from scipy.spatial import ConvexHull

from acoustic_self_calibration.array_configuration import ArrayConfiguration
from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.chirps import make_chirp_scene
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.simulation import room_benchmark_floor_polygon
from acoustic_self_calibration.wav import calibrate_wav


@pytest.mark.parametrize("layout", ["rectangular", "irregular"])
def test_direct_wall_chirps_recover_sources_outside_mic_hull(layout, tmp_path):
    scene = make_chirp_scene(geometry="room", layout=layout, seed=19)
    floor = room_benchmark_floor_polygon(layout=layout)
    physical_hull = ConvexHull(floor)
    segments = scene.trajectory_positions_m
    # A convex room contains the whole linear segments when their endpoints
    # satisfy every wall, floor and ceiling half-space.
    assert (
        np.max(segments[:, :2] @ physical_hull.equations[:, :2].T + physical_hull.equations[:, 2])
        < 0
    )
    assert np.min(segments[:, 2]) > 0 and np.max(segments[:, 2]) < 3
    hull = ConvexHull(scene.microphone_positions_m)
    inside = (
        np.max(
            scene.source_positions_at_events_m @ hull.equations[:, :3].T + hull.equations[:, 3],
            axis=1,
        )
        <= 1e-9
    )
    assert np.any(inside) and np.any(~inside)
    config = ArrayConfiguration("room")
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        array_configuration=config,
        max_tau_s=0.03,
        tdoa_template_s=0.002,
    )
    row = scene_metrics(result.calibration, scene, result.emission_times_s)
    assert result.status in {"solved", "weakly_identified"}
    assert row["microphone_rms_error_m"] < 0.05
    assert row["source_rms_error_m"] < 0.1
    assert row["matched_events"] == 20
    errors = np.array(row["source_errors_m"])
    assert np.sqrt(np.mean(errors[~inside] ** 2)) < 0.1
    path = tmp_path / "room.wav"
    normalized = scene.audio / np.max(abs(scene.audio))
    wavfile.write(path, scene.sample_rate_hz, np.round(normalized * 0.95 * 32767).astype(np.int16))
    pcm = calibrate_wav(path, array_configuration=config, max_tau_s=0.03, tdoa_template_s=0.002)
    pcm_metrics = scene_metrics(pcm.calibration, scene, pcm.emission_times_s)
    assert pcm_metrics["microphone_rms_error_m"] < 0.05
    assert pcm_metrics["source_rms_error_m"] < 0.1
    assert pcm_metrics["matched_events"] == 20
    # This tests hull-independent geometry accuracy. The separate acceptance
    # runner additionally requires solved status; weak results do not pass it.


def test_low_noise_room_chirps_do_not_receive_exact_seed_tolerance():
    # Development failure: both float and PCM rejected every algebraic seed
    # at 40 dB despite accurate arrival estimates. No truth enters calibration.
    scene = make_chirp_scene(12, geometry="room", layout="rectangular", seed=12)
    audio = scene.audio.copy()
    rng = np.random.default_rng(12)
    for channel in range(12):
        values = audio[:, channel]
        active = abs(values) > 0.01 * np.max(abs(values))
        audio[:, channel] += rng.normal(
            scale=np.sqrt(np.mean(values[active] ** 2) / 1e4), size=len(audio)
        )
    result = calibrate_audio(
        audio,
        scene.sample_rate_hz,
        array_configuration=ArrayConfiguration("room"),
        max_tau_s=0.03,
        tdoa_template_s=0.002,
        timing_uncertainty="waveform",
    )
    row = scene_metrics(result.calibration, scene, result.emission_times_s)
    assert row["standard_success"], (result.status, row, result.calibration.diagnostics)
