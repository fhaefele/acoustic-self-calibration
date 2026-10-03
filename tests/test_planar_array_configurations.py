import numpy as np
import pytest

from acoustic_self_calibration.array_configuration import ArrayConfiguration
from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.chirps import make_array_chirp_scene
from acoustic_self_calibration.export import calibration_result_to_dict
from acoustic_self_calibration.pipeline import calibrate_audio


def test_configuration_validates_membership_and_does_not_default_angle():
    config = ArrayConfiguration("cross", provenance="fixture", arms=((0, 1, 2, 3), (3, 4, 5, 6, 7)))
    assert config.angle_constraint() is None
    config.validate_ids(tuple(range(8)))
    with pytest.raises(ValueError, match="exactly"):
        config.validate_ids(tuple(range(9)))
    with pytest.raises(ValueError, match="unique"):
        ArrayConfiguration("grid", provenance="fixture", grid_slots=((0, 0, 0), (1, 0, 0)))
    with pytest.raises(ValueError, match="provenance"):
        ArrayConfiguration("cross", arms=((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="unknown construction"):
        ArrayConfiguration.from_dict({"name": "room", "pitch_m": 0.4})


@pytest.mark.parametrize("degrees,name", [(90, "cross"), (60, "cross"), (75, "t")])
def test_directed_two_line_chirps_recover_unknown_spacings(degrees, name):
    # Unequal arm coordinates; the T has all second-arm receivers on one ray.
    angle = np.deg2rad(degrees)
    coordinates = np.array([-1.0, -0.7, -0.2, 0.0, 0.3, 0.8, 1.0, 0.5])
    m = np.zeros((8, 3))
    m[:5, 0] = coordinates[:5]
    b = np.array([0.25, 0.65, 1.0]) if name == "t" else np.array([-0.8, 0.35, 1.0])
    m[5:, 0] = b * np.cos(angle)
    m[5:, 2] = b * np.sin(angle)
    config = ArrayConfiguration(
        name,
        provenance="declared fixture construction",
        arms=((0, 1, 2, 3, 4), (3, 5, 6, 7)),
        rays=((3, 4), (3, 7)),
        angle_deg=degrees,
    )
    scene = make_array_chirp_scene(m, seed=11)
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        array_configuration=config,
        best_sigma_samples=0.1,
        worst_sigma_samples=1.0,
    )
    row = scene_metrics(result.calibration, scene, result.emission_times_s)
    assert row["standard_success"], (result.status, row, result.calibration.diagnostics)
    document = calibration_result_to_dict(result)
    assert document["calibration"]["array_configuration"]["angle_deg"] == degrees
    assert document["calibration"]["array_configuration"]["metric_spacing"] == "unknown"
    blind = ArrayConfiguration(name, provenance="angle ablation", arms=config.arms)
    ablation = calibrate_audio(scene.audio, scene.sample_rate_hz, array_configuration=blind)
    assert ablation.status == "degenerate"
    assert ablation.microphone_positions_m is None
