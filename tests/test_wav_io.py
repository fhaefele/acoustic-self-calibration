import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_self_calibration.wav import read_multichannel_wav


def _samples() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 64, endpoint=False)
    mono = 0.75 * np.sin(2.0 * np.pi * 5.0 * t)
    return np.column_stack([mono, 0.7 * mono, -0.5 * mono, 0.3 * mono])


def _write_pcm24(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    integers = np.clip(np.round(samples * (2**23 - 1)), -(2**23), 2**23 - 1).astype(np.int32)
    unsigned = integers.astype(np.int64) & 0xFFFFFF
    packed = np.empty((integers.size, 3), dtype=np.uint8)
    flat = unsigned.reshape(-1)
    packed[:, 0] = flat & 0xFF
    packed[:, 1] = (flat >> 8) & 0xFF
    packed[:, 2] = (flat >> 16) & 0xFF
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(3)
        handle.setframerate(sample_rate)
        handle.writeframes(packed.tobytes())


@pytest.mark.parametrize("encoding", ["pcm16", "pcm24", "pcm32", "float32"])
def test_read_multichannel_wav_normalizes_common_encodings(tmp_path: Path, encoding: str) -> None:
    sample_rate = 48_000
    expected = _samples()
    path = tmp_path / f"input_{encoding}.wav"

    if encoding == "pcm16":
        wavfile.write(path, sample_rate, np.round(expected * 32767.0).astype(np.int16))
        tolerance = 5e-5
    elif encoding == "pcm24":
        _write_pcm24(path, sample_rate, expected)
        tolerance = 2e-6
    elif encoding == "pcm32":
        wavfile.write(path, sample_rate, np.round(expected * (2**31 - 1)).astype(np.int32))
        tolerance = 2e-9
    else:
        wavfile.write(path, sample_rate, expected.astype(np.float32))
        tolerance = 1e-7

    actual_rate, actual = read_multichannel_wav(path)
    assert actual_rate == sample_rate
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < tolerance


def test_read_multichannel_wav_rejects_mono(tmp_path: Path) -> None:
    path = tmp_path / "mono.wav"
    wavfile.write(path, 16_000, np.zeros(100, dtype=np.int16))
    with pytest.raises(ValueError, match="multichannel"):
        read_multichannel_wav(path)


def test_read_multichannel_wav_requires_four_channels(tmp_path: Path) -> None:
    path = tmp_path / "three.wav"
    wavfile.write(path, 16_000, np.zeros((100, 3), dtype=np.int16))
    with pytest.raises(ValueError, match="four"):
        read_multichannel_wav(path)
