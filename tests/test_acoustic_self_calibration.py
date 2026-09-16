from __future__ import annotations

from pathlib import Path

import numpy as np

from acoustic_self_calibration.calibration import calibrate_from_dataset, canonicalize_geometry
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


def test_export_default_stem_matches_microphone_count(tmp_path: Path) -> None:
    wav_path, metadata_path = export_synthetic_dataset(tmp_path, num_mics=6, seed=4)

    assert wav_path.name == "synthetic_6_mic.wav"
    assert metadata_path.name == "synthetic_6_mic.json"


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
