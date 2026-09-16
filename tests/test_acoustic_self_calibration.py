from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from acoustic_self_calibration.calibration import (
    calibrate_from_arrival_times,
    calibrate_from_audio,
    calibrate_from_distances,
    canonicalize_geometry,
    detect_arrival_times,
)
from acoustic_self_calibration.synthetic import (
    export_synthetic_dataset,
    generate_synthetic_recording,
    generate_synthetic_scene,
    load_synthetic_dataset,
)


def test_generate_synthetic_scene_defaults_to_twelve_microphones() -> None:
    scene = generate_synthetic_scene()

    assert scene.microphone_positions.shape == (12, 3)
    assert scene.source_positions.shape == (8, 3)
    assert len(scene.emission_times) == 8
    assert scene.pulse.ndim == 1


def test_export_and_load_synthetic_dataset(tmp_path: Path) -> None:
    wav_path, metadata_path = export_synthetic_dataset(tmp_path, seed=3)
    recording = load_synthetic_dataset(wav_path, metadata_path)

    assert wav_path.exists()
    assert metadata_path.exists()
    assert recording.audio.ndim == 2
    assert recording.audio.shape[1] == 12
    assert recording.scene.microphone_positions.shape == (12, 3)


def test_load_synthetic_dataset_rejects_channel_mismatch(tmp_path: Path) -> None:
    wav_path, metadata_path = export_synthetic_dataset(tmp_path, seed=3)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["microphone_positions"] = metadata["microphone_positions"][:-1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="channel count"):
        load_synthetic_dataset(wav_path, metadata_path)


def test_export_default_stem_matches_microphone_count(tmp_path: Path) -> None:
    wav_path, metadata_path = export_synthetic_dataset(tmp_path, num_mics=6, seed=4)

    assert wav_path.name == "synthetic_6_mic.wav"
    assert metadata_path.name == "synthetic_6_mic.json"


def test_generate_synthetic_recording_handles_early_arrivals() -> None:
    scene = generate_synthetic_scene()
    scene = replace(scene, emission_times=scene.emission_times.copy())
    scene.emission_times[0] = 0.0

    recording = generate_synthetic_recording(scene=scene)

    assert recording.audio.shape[1] == 12
    assert np.max(np.abs(recording.audio)) > 0.0


def _true_arrival_times(recording: object) -> np.ndarray:
    scene = recording.scene
    distances = np.linalg.norm(
        scene.source_positions[:, None, :] - scene.microphone_positions[None, :, :],
        axis=2,
    )
    return scene.emission_times[:, None] + distances / scene.speed_of_sound


def test_detect_arrivals_does_not_need_emission_times_and_handles_inverted_channel() -> None:
    recording = generate_synthetic_recording(seed=7)
    audio = recording.audio.copy()
    audio[:, 0] *= -1.0

    detected = detect_arrival_times(audio, recording.scene.pulse, recording.scene.sample_rate, num_events=8)
    truth = _true_arrival_times(recording)
    sample_errors = (detected - truth) * recording.scene.sample_rate

    assert np.sqrt(np.mean(sample_errors**2)) < 0.5


def test_arrival_detection_has_no_hidden_three_meter_limit() -> None:
    scene = generate_synthetic_scene(seed=2)
    scene = replace(scene, microphone_positions=4.0 * scene.microphone_positions)
    recording = generate_synthetic_recording(scene=scene)
    truth = _true_arrival_times(recording)
    distances = (truth - scene.emission_times[:, None]) * scene.speed_of_sound

    assert np.max(distances) > 5.0
    detected = detect_arrival_times(recording.audio, scene.pulse, scene.sample_rate, num_events=8)
    sample_errors = (detected - truth) * scene.sample_rate
    assert np.sqrt(np.mean(sample_errors**2)) < 0.5


def test_tdoa_calibration_recovers_geometry_without_emission_times() -> None:
    recording = generate_synthetic_recording()
    result = calibrate_from_audio(
        recording.audio,
        recording.scene.pulse,
        recording.scene.sample_rate,
        speed_of_sound=recording.scene.speed_of_sound,
        num_events=8,
    )
    truth_mics, truth_sources = canonicalize_geometry(
        recording.scene.microphone_positions,
        recording.scene.source_positions,
    )

    assert result.success
    assert result.jacobian_rank == result.parameter_count
    assert result.residual_rms < 0.02
    assert np.sqrt(np.mean((result.microphone_positions - truth_mics) ** 2)) < 0.2
    assert np.sqrt(np.mean((result.source_positions - truth_sources) ** 2)) < 0.2


def test_calibrate_from_arrival_times_is_invariant_to_unknown_emission_time() -> None:
    recording = generate_synthetic_recording(seed=4)
    arrivals = _true_arrival_times(recording)
    offsets = np.linspace(0.2, 1.1, len(arrivals))[:, None]
    result = calibrate_from_arrival_times(arrivals + offsets, speed_of_sound=recording.scene.speed_of_sound)
    truth_mics, truth_sources = canonicalize_geometry(
        recording.scene.microphone_positions,
        recording.scene.source_positions,
    )

    assert result.residual_rms < 1e-6
    assert np.sqrt(np.mean((result.microphone_positions - truth_mics) ** 2)) < 1e-4
    assert np.sqrt(np.mean((result.source_positions - truth_sources) ** 2)) < 1e-4


def test_calibrate_from_distances_rejects_underdetermined_four_by_four_problem() -> None:
    distances = np.ones((4, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="underdetermined"):
        calibrate_from_distances(distances)


def test_tdoa_calibration_rejects_underdetermined_geometry() -> None:
    arrivals = np.ones((8, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="underdetermined"):
        calibrate_from_arrival_times(arrivals)


def test_calibrate_from_distances_requires_at_least_four_microphones() -> None:
    distances = np.ones((8, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="At least four microphones"):
        calibrate_from_distances(distances)


def test_calibrate_from_distances_requires_at_least_four_source_positions() -> None:
    distances = np.ones((3, 12), dtype=np.float64)

    with pytest.raises(ValueError, match="At least four source positions"):
        calibrate_from_distances(distances)


def test_canonicalize_geometry_rejects_collinear_anchor_microphones() -> None:
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    sources = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)

    with pytest.raises(ValueError, match="must not be collinear"):
        canonicalize_geometry(microphones, sources)
