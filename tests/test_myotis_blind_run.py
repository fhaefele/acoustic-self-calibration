import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from acoustic_self_calibration.ground_truth import write_ground_truth_json
from acoustic_self_calibration.myotis import (
    run_real_myotis_calibration,
    write_real_myotis_manifest,
)
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene


def _write_blind_fixture(tmp_path: Path):
    scene = make_random_3d_pulse_scene(8, event_count=20)
    audio_path = tmp_path / "fixture.wav"
    scaled = scene.audio / max(float(np.max(np.abs(scene.audio))), 1e-12)
    wavfile.write(
        audio_path,
        scene.sample_rate_hz,
        np.round(0.95 * scaled * 32767.0).astype(np.int16),
    )
    reference_path = tmp_path / "reference.json"
    write_ground_truth_json(
        reference_path,
        microphone_positions_m=scene.microphone_positions_m,
        source_times_s=scene.trajectory_times_s,
        source_positions_m=scene.trajectory_positions_m,
        metadata={"fixture_role": "harness_only"},
    )
    return scene, audio_path, reference_path


def test_blind_runner_reports_missing_inputs_without_substitution() -> None:
    report = run_real_myotis_calibration(environment={})
    assert report["status"] == "not_run"
    assert report["reason"] == "missing_input_paths"
    assert report["mode"] == "blind"
    assert report["reference_used_for_solver"] is False
    assert report["run"] is None
    assert report["evaluation"] is None


def test_blind_runner_calibrates_before_reference_evaluation(tmp_path: Path) -> None:
    _, audio_path, reference_path = _write_blind_fixture(tmp_path)
    report = run_real_myotis_calibration(
        audio_path=audio_path,
        reference_path=reference_path,
        software_revision="test-revision",
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
    )

    assert report["status"] == "solved"
    assert report["mode"] == "blind"
    assert report["reference_used_for_solver"] is False
    assert report["software_revision"] == "test-revision"
    assert report["geometry_conditioning"]["receiver_count"] == 8
    assert report["run"]["status"] == "solved"
    assert report["run"]["detected_event_count"] == 20
    assert report["run"]["used_event_count"] == 20
    assert report["run"]["valid_measurement_count"] > 0
    assert report["run"]["refinement"]["applied"] is False
    assert report["evaluation"]["status"] == "evaluated"
    assert report["evaluation"]["source_overlap_count"] == 20
    assert report["evaluation"]["microphones"]["rms_error_m"] < 0.18
    assert report["evaluation"]["source"]["rms_error_m"] < 0.22
    assert len(report["audio"]["sha256"]) == 64
    assert len(report["reference"]["sha256"]) == 64
    assert len(report["run_configuration_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_blind_runner_manifest_is_stable_json(tmp_path: Path) -> None:
    _, audio_path, reference_path = _write_blind_fixture(tmp_path)
    report = run_real_myotis_calibration(
        audio_path=audio_path,
        reference_path=reference_path,
        receiver_subset_budget=1,
        event_subset_budget=1,
        root_start_count=16,
        metric_start_count=8,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
    )
    output = tmp_path / "blind-run.json"
    write_real_myotis_manifest(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
