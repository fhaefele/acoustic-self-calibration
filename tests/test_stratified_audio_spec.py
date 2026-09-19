from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration import calibrate_audio, calibrate_wav
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene

_EVENT_COUNTS = (20, 40)
_MICROPHONE_COUNTS = (8, 12, 16, 24)


def _assert_event_result(
    result: object,
    *,
    microphones: np.ndarray,
    true_source: np.ndarray,
    event_count: int,
    microphone_rms_limit_m: float,
    source_rms_limit_m: float,
    tdoa_rms_limit_s: float,
) -> None:
    assert getattr(result, "status") == "solved"
    estimated_microphones = np.asarray(getattr(result, "microphone_positions_m"), dtype=float)
    estimated_source = np.asarray(getattr(result, "source_positions_m"), dtype=float)
    measurements = getattr(result, "measurements")

    assert estimated_microphones.shape == microphones.shape
    assert estimated_source.shape == true_source.shape
    assert len(measurements.event_ids) == event_count
    assert np.array_equal(measurements.event_ids, np.arange(event_count))
    assert np.all(measurements.valid)
    assert np.all(np.isfinite(measurements.tdoa_s[measurements.valid]))
    assert np.all(np.isfinite(measurements.sigma_s[measurements.valid]))
    assert np.all(measurements.sigma_s[measurements.valid] > 0.0)

    aligned_microphones, rotation, translation = rigid_align(
        estimated_microphones,
        microphones,
    )
    aligned_source = apply_rigid(estimated_source, rotation, translation)
    assert rms_position_error(aligned_microphones, microphones) < microphone_rms_limit_m
    assert rms_position_error(aligned_source, true_source) < source_rms_limit_m
    assert float(getattr(result, "tdoa_rms_s")) < tdoa_rms_limit_s


@pytest.mark.parametrize("microphone_count", _MICROPHONE_COUNTS)
@pytest.mark.parametrize("event_count", _EVENT_COUNTS)
def test_stratified_event_audio_acceptance_spec(
    microphone_count: int,
    event_count: int,
) -> None:
    scene = make_random_3d_pulse_scene(microphone_count, event_count=event_count)

    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        speed_of_sound=343.0,
    )
    _assert_event_result(
        result,
        microphones=scene.microphone_positions_m,
        true_source=scene.source_positions_at_events_m,
        event_count=event_count,
        microphone_rms_limit_m=0.15,
        source_rms_limit_m=0.18,
        tdoa_rms_limit_s=60e-6,
    )


@pytest.mark.parametrize("event_count", _EVENT_COUNTS)
def test_stratified_pcm16_wav_acceptance_spec(tmp_path: Path, event_count: int) -> None:
    scene = make_random_3d_pulse_scene(8, event_count=event_count)
    path = tmp_path / f"moving_pulses_{event_count}.wav"
    scaled = scene.audio / max(float(np.max(np.abs(scene.audio))), 1e-12)
    wavfile.write(path, scene.sample_rate_hz, np.round(0.95 * scaled * 32767.0).astype(np.int16))

    result = calibrate_wav(
        path,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        speed_of_sound=343.0,
    )
    _assert_event_result(
        result,
        microphones=scene.microphone_positions_m,
        true_source=scene.source_positions_at_events_m,
        event_count=event_count,
        microphone_rms_limit_m=0.18,
        source_rms_limit_m=0.22,
        tdoa_rms_limit_s=60e-6,
    )
