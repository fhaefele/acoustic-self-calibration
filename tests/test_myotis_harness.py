import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration.ground_truth import write_ground_truth_json
from acoustic_self_calibration.myotis import (
    prepare_real_myotis_validation,
    write_real_myotis_manifest,
)


def _write_inputs(tmp_path, *, channels: int = 4, reference_microphones: int = 4):
    audio_path = tmp_path / "myotis.wav"
    sample_rate = 48_000
    samples = np.arange(4_800 * channels, dtype=np.int16).reshape(4_800, channels)
    wavfile.write(audio_path, sample_rate, samples)

    reference_path = tmp_path / "reference.json"
    microphones = np.column_stack(
        [
            np.linspace(0.0, 1.0, reference_microphones),
            np.zeros(reference_microphones),
            np.linspace(0.2, 0.8, reference_microphones),
        ]
    )
    write_ground_truth_json(
        reference_path,
        microphone_positions_m=microphones,
        source_times_s=np.array([0.0, 0.05, 0.1]),
        source_positions_m=np.array([[1.0, 1.0, 0.3], [0.9, 1.1, 0.4], [0.8, 1.2, 0.5]]),
        metadata={"fixture": True},
    )
    return audio_path, reference_path


def test_missing_real_myotis_inputs_report_not_run() -> None:
    report = prepare_real_myotis_validation(environment={})
    assert report["status"] == "not_run"
    assert report["reason"] == "missing_input_paths"
    assert report["missing"] == ["ASC_MYOTIS_AUDIO", "ASC_MYOTIS_REFERENCE"]


def test_real_myotis_manifest_records_hashes_and_configuration(tmp_path) -> None:
    audio_path, reference_path = _write_inputs(tmp_path)
    report = prepare_real_myotis_validation(
        audio_path=audio_path,
        reference_path=reference_path,
        speed_of_sound_mps=342.5,
        solver_seed=17,
        software_revision="test-revision",
    )

    assert report["status"] == "ready"
    assert report["reason"] is None
    assert report["audio"]["sample_rate_hz"] == 48_000
    assert report["audio"]["channel_count"] == 4
    assert report["audio"]["sample_count"] == 4_800
    assert report["audio"]["duration_s"] == pytest.approx(0.1)
    assert report["reference"]["microphone_count"] == 4
    assert report["reference"]["source_sample_count"] == 3
    assert report["configuration"]["channel_mapping"] == [0, 1, 2, 3]
    assert report["configuration"]["speed_of_sound_mps"] == 342.5
    assert report["configuration"]["solver_seed"] == 17
    assert report["software_revision"] == "test-revision"

    expected_audio_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    expected_reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert report["audio"]["sha256"] == expected_audio_hash
    assert report["reference"]["sha256"] == expected_reference_hash
    assert len(report["configuration_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_real_myotis_manifest_requires_explicit_mapping_for_count_mismatch(tmp_path) -> None:
    audio_path, reference_path = _write_inputs(tmp_path, channels=4, reference_microphones=5)
    with pytest.raises(ValueError, match="provide an explicit channel_mapping"):
        prepare_real_myotis_validation(
            audio_path=audio_path,
            reference_path=reference_path,
        )

    report = prepare_real_myotis_validation(
        audio_path=audio_path,
        reference_path=reference_path,
        channel_mapping=(0, 1, 2, 4),
    )
    assert report["configuration"]["channel_mapping"] == [0, 1, 2, 4]


def test_real_myotis_manifest_environment_paths(tmp_path) -> None:
    audio_path, reference_path = _write_inputs(tmp_path)
    report = prepare_real_myotis_validation(
        environment={
            "ASC_MYOTIS_AUDIO": str(audio_path),
            "ASC_MYOTIS_REFERENCE": str(reference_path),
            "ASC_SOFTWARE_REVISION": "env-revision",
        }
    )
    assert report["audio"]["path"] == str(audio_path.resolve())
    assert report["reference"]["path"] == str(reference_path.resolve())
    assert report["software_revision"] == "env-revision"


def test_real_myotis_manifest_writer_round_trips_json(tmp_path) -> None:
    audio_path, reference_path = _write_inputs(tmp_path)
    report = prepare_real_myotis_validation(
        audio_path=audio_path,
        reference_path=reference_path,
    )
    output_path = tmp_path / "manifest.json"
    returned = write_real_myotis_manifest(output_path, report)

    assert returned == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
