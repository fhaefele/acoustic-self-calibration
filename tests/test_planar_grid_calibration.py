import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration.array_configuration import ArrayConfiguration
from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.chirps import make_array_chirp_scene
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.wav import calibrate_wav


@pytest.mark.parametrize("equal", [False, True])
def test_grid_chirp_unknown_free_or_equal_independent_pitches(equal, tmp_path):
    slots = tuple((3 * r + c, r, c) for r in range(3) for c in range(3))
    xs = [-1.0, 0.0, 1.0] if equal else [-1.0, -0.15, 1.0]
    zs = [-0.8, 0.0, 0.8] if equal else [-1.0, 0.25, 1.0]
    m = np.array([[xs[c], 0, zs[r]] for i, r, c in slots])
    configuration = ArrayConfiguration(
        "grid",
        provenance="fixture declared topology",
        grid_slots=slots,
        equal_row_spacing=equal,
        equal_column_spacing=equal,
    )
    scene = make_array_chirp_scene(m)
    result = calibrate_audio(scene.audio, scene.sample_rate_hz, array_configuration=configuration)
    row = scene_metrics(result.calibration, scene, result.emission_times_s)
    assert row["standard_success"], (row, result.calibration.diagnostics)
    wav = tmp_path / "grid.wav"
    wavfile.write(wav, scene.sample_rate_hz, scene.audio)
    restored = calibrate_wav(wav, array_configuration=configuration)
    assert restored.array_configuration == configuration
    assert restored.microphone_positions_m is not None
    assert result.microphone_positions_m is not None
    np.testing.assert_allclose(restored.microphone_positions_m, result.microphone_positions_m)


def test_two_row_grid_does_not_claim_unique_row_separation():
    slots = tuple((4 * r + c, r, c) for r in range(2) for c in range(4))
    m = np.array([[-1.0, -0.3, 0.2, 1.0][c : c + 1] + [0.0, float(r)] for i, r, c in slots])
    scene = make_array_chirp_scene(m)
    config = ArrayConfiguration("grid", provenance="two-row fixture", grid_slots=slots)
    result = calibrate_audio(scene.audio, scene.sample_rate_hz, array_configuration=config)
    assert result.status in {"degenerate", "ambiguous"}, result.calibration.diagnostics
