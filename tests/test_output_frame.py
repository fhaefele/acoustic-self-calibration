import numpy as np
import pytest

from acoustic_self_calibration.chirps import make_chirp_scene
from acoustic_self_calibration.export import calibration_result_to_dict
from acoustic_self_calibration.output_frame import select_microphone_frame
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.simulation import exact_reference_tdoas


@pytest.mark.parametrize("planar", [False, True])
def test_frame_preserves_geometry_and_ids_under_rigid_motion_and_permutation(planar):
    rng = np.random.default_rng(871)
    m = rng.normal(size=(12, 3))
    if planar:
        m[:, 2] = 0
    s = rng.normal(size=(20, 3))
    s[:, 2] = abs(s[:, 2]) + 1
    ids = tuple(range(12))
    frame = select_microphone_frame(m, ids, origin_id=3, common_side_sources=s if planar else None)
    mc, sc = frame.transform(m), frame.transform(s)
    assert mc is not None and sc is not None
    assert np.linalg.norm(mc[3]) < 1e-12
    np.testing.assert_allclose(
        exact_reference_tdoas(mc, sc), exact_reference_tdoas(m, s), atol=1e-14
    )
    if planar:
        assert np.max(abs(mc[:, 2])) < 1e-12
        assert np.min(sc[:, 2]) > 0
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    permutation = rng.permutation(12)
    moved = (m @ rotation + [7, 8, 9])[permutation]
    permuted_ids = tuple(ids[i] for i in permutation)
    other = select_microphone_frame(
        moved,
        permuted_ids,
        origin_id=3,
        common_side_sources=s @ rotation + [7, 8, 9] if planar else None,
    )
    other_m = other.transform(moved)
    assert other_m is not None
    np.testing.assert_allclose(other_m[np.argsort(permutation)], mc, atol=1e-12)
    assert (other.x_axis_microphone_id, other.plane_microphone_id) == (
        frame.x_axis_microphone_id,
        frame.plane_microphone_id,
    )


def test_public_origin_export_and_source_emission_time_basis():
    scene = make_chirp_scene()
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        model="receiver2d_source3d",
        source_region="same_side",
        output_origin_microphone_id=3,
    )
    assert result.microphone_positions_m is not None and result.source_positions_m is not None
    assert np.linalg.norm(result.microphone_positions_m[3]) < 1e-10
    assert np.max(abs(result.microphone_positions_m[:, 2])) < 1e-10
    assert np.min(result.source_positions_m[:, 2]) > 0
    document = calibration_result_to_dict(result)
    assert document["calibration"]["coordinate_frame"]["origin_microphone_id"] == 3
    assert document["scene"]["source"]["time_basis"] == "emission_feature"
    np.testing.assert_allclose(document["scene"]["source"]["times_s"], result.emission_times_s)
    with pytest.raises(ValueError, match="origin microphone"):
        calibrate_audio(scene.audio, scene.sample_rate_hz, output_origin_microphone_id=19)
