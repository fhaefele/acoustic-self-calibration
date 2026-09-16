from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from acoustic_self_calibration.calibration import (
    calibrate_from_dataset,
    calibrate_from_distances,
    canonicalize_geometry,
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


def test_calibration_recovers_geometry_from_generated_dataset(tmp_path: Path) -> None:
    recording = generate_synthetic_recording()
    wav_path, metadata_path = export_synthetic_dataset(tmp_path, recording=recording)
    result = calibrate_from_dataset(wav_path, metadata_path)
    truth_mics, truth_sources = canonicalize_geometry(
        recording.scene.microphone_positions,
        recording.scene.source_positions,
    )

    assert result.success
    assert result.residual_rms < 0.05
    assert np.sqrt(np.mean((result.microphone_positions - truth_mics) ** 2)) < 0.35
    assert np.sqrt(np.mean((result.source_positions - truth_sources) ** 2)) < 0.35


def test_calibration_recovers_geometry_from_checked_in_fixture() -> None:
    fixture_dir = Path(__file__).parent / "data"
    wav_path = fixture_dir / "synthetic_12_mic_fixture.wav"
    metadata_path = fixture_dir / "synthetic_12_mic_fixture.json"
    recording = load_synthetic_dataset(wav_path, metadata_path)
    result = calibrate_from_dataset(wav_path, metadata_path)
    truth_mics, truth_sources = canonicalize_geometry(
        recording.scene.microphone_positions,
        recording.scene.source_positions,
    )

    assert result.success
    assert result.residual_rms < 0.05
    assert np.sqrt(np.mean((result.microphone_positions - truth_mics) ** 2)) < 0.35
    assert np.sqrt(np.mean((result.source_positions - truth_sources) ** 2)) < 0.35


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
