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


def test_two_row_grid_has_continuous_ambiguity_even_with_equal_pitches():
    from acoustic_self_calibration.array_configuration import ArrayConfiguration
    from acoustic_self_calibration.chirps import make_array_chirp_scene
    from acoustic_self_calibration.pipeline import calibrate_audio
    from acoustic_self_calibration.stratified.solver import PlanarCalibrationResult

    m = np.array([[x, y, 0.0] for y in [0.0, 1.0] for x in [-1.0, -0.3, 0.4, 1.1]])
    sources = np.column_stack(
        [np.linspace(-1, 1, 20), np.linspace(0.5, 1.5, 20), np.linspace(2, 4, 20)]
    )
    ranges = np.linalg.norm(sources[:, None] - m[None], axis=2)
    for scale in [0.8, 1.2]:
        alternative_mics = m.copy()
        alternative_mics[:, 1] *= scale
        alternative_sources = sources.copy()
        alternative_sources[:, 1] = (sources[:, 1] + 0.5 * (scale**2 - 1)) / scale
        alternative_sources[:, 2] = np.sqrt(
            np.sum(sources**2, axis=1) - np.sum(alternative_sources[:, :2] ** 2, axis=1)
        )
        np.testing.assert_allclose(
            np.linalg.norm(alternative_sources[:, None] - alternative_mics[None], axis=2),
            ranges,
            atol=1e-10,
        )
    config = ArrayConfiguration(
        "grid",
        provenance="declared two-row construction",
        grid_slots=tuple((i, i // 4, i % 4) for i in range(8)),
    )
    scene = make_array_chirp_scene(m, seed=10)
    result = calibrate_audio(scene.audio, scene.sample_rate_hz, array_configuration=config)
    assert result.status == "degenerate"
    assert result.microphone_positions_m is None
    assert isinstance(result.calibration, PlanarCalibrationResult)
    assert result.calibration.continuous_ambiguity_dimension == 1
    for equal in [False, True]:
        configured = ArrayConfiguration(
            "grid",
            provenance="two equal or free rows",
            grid_slots=config.grid_slots,
            equal_row_spacing=equal,
            equal_column_spacing=equal,
        )
        assert configured.metric_ambiguity_reason() == "two_parallel_lines_metric_family"
