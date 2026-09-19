from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration.geometry import (
    apply_rigid,
    rigid_align,
    rms_position_error,
)
from acoustic_self_calibration.simulation import make_random_3d_pulse_scene
from acoustic_self_calibration.wav import calibrate_wav, read_multichannel_wav


@pytest.mark.parametrize("event_count", [20, 40])
def test_pcm16_wav_gate(event_count: int, tmp_path: Path) -> None:
    scene = make_random_3d_pulse_scene(8, event_count=event_count)
    path = tmp_path / f"pulses_{event_count}.wav"
    scaled = scene.audio / max(float(np.max(np.abs(scene.audio))), 1e-12)
    wavfile.write(
        path,
        scene.sample_rate_hz,
        np.round(0.95 * scaled * 32767.0).astype(np.int16),
    )

    result = calibrate_wav(
        path,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        receiver_subset_budget=2,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
    )

    assert result.status == "solved"
    assert result.detected_event_count == event_count
    assert result.microphone_positions_m is not None
    assert result.source_positions_m is not None
    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    aligned_sources = apply_rigid(
        result.source_positions_m,
        rotation,
        translation,
    )
    assert (
        rms_position_error(
            aligned_microphones,
            scene.microphone_positions_m,
        )
        < 0.18
    )
    assert (
        rms_position_error(
            aligned_sources,
            scene.source_positions_at_events_m,
        )
        < 0.22
    )


def test_pcm16_reader_normalizes_integer_audio(tmp_path: Path) -> None:
    path = tmp_path / "pcm16.wav"
    samples = np.array(
        [[-32768, 0, 32767, 16384], [0, 1, -1, 2]],
        dtype=np.int16,
    )
    wavfile.write(path, 48_000, samples)
    sample_rate, audio = read_multichannel_wav(path)
    assert sample_rate == 48_000
    assert audio.dtype == np.float64
    assert np.max(audio) <= 1.0
    assert np.min(audio) >= -1.0
