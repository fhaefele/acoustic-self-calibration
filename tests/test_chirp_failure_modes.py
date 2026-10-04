import json
from dataclasses import replace

import numpy as np
from scipy.io import wavfile

from acoustic_self_calibration.chirps import make_chirp_scene
from acoustic_self_calibration.cli import main
from acoustic_self_calibration.export import write_calibration_outputs
from acoustic_self_calibration.pipeline import calibrate_audio


def test_cli_insufficient_recording_always_writes_diagnostics(tmp_path):
    wav = tmp_path / "quiet.wav"
    wavfile.write(wav, 48000, np.zeros((2000, 8), dtype=np.int16))
    prefix = tmp_path / "failed"
    assert main(["calibrate", str(wav), "-o", str(prefix)]) == 2
    document = json.loads(prefix.with_suffix(".json").read_text())
    assert document["calibration"]["status"] == "insufficient_data"
    assert document["visualization"]["written"] is False
    assert not prefix.with_suffix(".png").exists()


def test_absent_geometry_and_partial_sources_keep_diagnostic_ids(tmp_path):
    scene = make_chirp_scene()
    result = calibrate_audio(scene.audio, scene.sample_rate_hz, model="receiver2d_source3d")
    calibration = replace(
        result.calibration,
        status="degenerate",
        microphone_positions_m=None,
        source_representative_positions_m=None,
    )
    absent = replace(result, calibration=calibration)
    paths = write_calibration_outputs(absent, tmp_path / "degenerate")
    document = json.loads(paths.json.read_text())
    assert document["calibration"]["status"] == "degenerate"
    assert len(document["measurements"]["coverage"]["unresolved_event_ids"]) == 20
    assert document["visualization"]["written"] is False
    assert not paths.figure.exists()
