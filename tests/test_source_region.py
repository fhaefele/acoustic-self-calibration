from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration.chirps import make_chirp_scene
from acoustic_self_calibration.cli import build_parser
from acoustic_self_calibration.export import calibration_result_to_dict
from acoustic_self_calibration.geometry import select_coordinate_gauge
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.stratified.solver import _apply_source_half_space_prior
from acoustic_self_calibration.wav import calibrate_wav


def test_common_side_public_audio_wav_and_refinement(tmp_path):
    scene = make_chirp_scene(seed=10)
    options: dict[str, Any] = dict(
        model="receiver2d_source3d", source_region="same_side", refinement="wls"
    )
    audio = calibrate_audio(scene.audio, scene.sample_rate_hz, **options)
    path = tmp_path / "chirps.wav"
    wavfile.write(path, scene.sample_rate_hz, scene.audio)
    wav = calibrate_wav(path, **options)
    assert audio.status == wav.status
    assert audio.status in {"solved", "weakly_identified"}
    assert wav.microphone_positions_m is not None
    assert wav.source_positions_m is not None
    assert audio.microphone_positions_m is not None
    assert audio.source_positions_m is not None
    np.testing.assert_allclose(audio.microphone_positions_m, wav.microphone_positions_m)
    np.testing.assert_allclose(audio.source_positions_m, wav.source_positions_m)
    assert audio.microphone_positions_m is not None
    assert audio.source_positions_m is not None
    assert audio.source_height_sign_known is not None
    assert audio.source_unsigned_heights_m is not None
    assert np.all(audio.source_height_sign_known)
    gauge = select_coordinate_gauge(audio.microphone_positions_m)
    assert np.all((audio.source_positions_m - gauge.origin_m) @ gauge.basis[:, 2] > 0)
    document = calibration_result_to_dict(audio)
    assert document["scene"]["source"]["representation"] == "declared_common_side"
    assert document["scene"]["source"]["height_sign_known"] == [True] * 20
    assert np.all(np.isfinite(audio.emission_times_s))

    from acoustic_self_calibration.stratified.solver import PlanarCalibrationResult

    assert isinstance(audio.calibration, PlanarCalibrationResult)
    sources = audio.source_positions_m.copy()
    heights = audio.source_unsigned_heights_m.copy()
    sources[3] = np.nan
    heights[3] = np.nan
    partial = replace(
        audio.calibration,
        source_representative_positions_m=sources,
        source_unsigned_heights_m=heights,
    )
    partial = _apply_source_half_space_prior(partial, 1)
    assert partial.source_region_constraint_enforced
    assert partial.source_height_sign_known is not None
    assert partial.source_representative_positions_m is not None
    assert not partial.source_height_sign_known[3]
    assert np.sum(partial.source_height_sign_known) == 19
    assert np.isnan(partial.source_representative_positions_m[3]).all()


def test_common_side_incompatible_model_and_cli():
    with pytest.raises(ValueError, match="requires the planar"):
        calibrate_audio(np.zeros((128, 8)), 48000, source_region="same_side")
    args = build_parser().parse_args(["calibrate", "chirps.wav", "--source-region", "same-side"])
    assert args.source_region == "same-side"
