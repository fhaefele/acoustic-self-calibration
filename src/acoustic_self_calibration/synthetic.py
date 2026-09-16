from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import chirp

FloatArray = NDArray[np.float64]
PCM_SCALE = float(np.iinfo(np.int16).max)


@dataclass(frozen=True)
class SyntheticScene:
    microphone_positions: FloatArray
    source_positions: FloatArray
    source_orientations: FloatArray
    emission_times: FloatArray
    pulse: FloatArray
    sample_rate: int
    speed_of_sound: float = 343.0


@dataclass(frozen=True)
class SyntheticRecording:
    scene: SyntheticScene
    audio: FloatArray


def _unit_vectors(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _build_default_microphone_positions(num_mics: int, rng: np.random.Generator) -> FloatArray:
    if num_mics < 4:
        raise ValueError("At least four microphones are required for 3D calibration.")

    angles = np.linspace(0.0, 2.0 * np.pi, num_mics, endpoint=False)
    radii = 1.1 + rng.uniform(-0.15, 0.15, size=num_mics)
    heights = np.linspace(-0.35, 0.35, num_mics)
    jitter = rng.uniform(-0.07, 0.07, size=(num_mics, 3))

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = heights
    return np.column_stack((x, y, z)) + jitter


def _build_default_source_positions(num_points: int) -> FloatArray:
    angles = np.linspace(0.2, 2.8 * np.pi, num_points, endpoint=False)
    radii = np.linspace(0.35, 0.65, num_points)
    heights = np.linspace(-0.45, 0.45, num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y, heights))


def _estimate_orientations(source_positions: FloatArray) -> FloatArray:
    deltas = np.diff(source_positions, axis=0, append=source_positions[-1:])
    if len(source_positions) > 1:
        deltas[-1] = source_positions[-1] - source_positions[-2]
    return _unit_vectors(deltas)


def pulse_reference_sample(pulse: FloatArray) -> int:
    return int(np.argmax(np.abs(pulse)))


def make_excitation_pulse(
    sample_rate: int,
    duration: float = 0.008,
    start_frequency: float = 900.0,
    end_frequency: float = 1800.0,
) -> FloatArray:
    sample_count = max(16, int(round(sample_rate * duration)))
    times = np.arange(sample_count, dtype=np.float64) / sample_rate
    window = np.hanning(sample_count)
    return chirp(times, f0=start_frequency, f1=end_frequency, t1=duration, method="linear") * window


def radiation_gain(
    source_orientations: FloatArray,
    source_positions: FloatArray,
    microphone_positions: FloatArray,
    directivity: float = 0.65,
) -> FloatArray:
    rays = microphone_positions[None, :, :] - source_positions[:, None, :]
    ray_directions = _unit_vectors(rays.reshape(-1, 3)).reshape(rays.shape)
    forward = _unit_vectors(source_orientations)
    cosine = np.einsum("sd,smd->sm", forward, ray_directions)
    base_pattern = 0.5 * (1.0 + cosine)
    return (1.0 - directivity) + directivity * np.clip(base_pattern, 0.1, 1.0)


def generate_synthetic_scene(
    num_mics: int = 12,
    num_source_positions: int = 8,
    sample_rate: int = 8_000,
    emission_interval: float = 0.06,
    seed: int = 0,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    microphone_positions = _build_default_microphone_positions(num_mics, rng)
    source_positions = _build_default_source_positions(num_source_positions)
    source_orientations = _estimate_orientations(source_positions)
    emission_times = 0.04 + emission_interval * np.arange(num_source_positions, dtype=np.float64)
    pulse = make_excitation_pulse(sample_rate=sample_rate)
    return SyntheticScene(
        microphone_positions=microphone_positions,
        source_positions=source_positions,
        source_orientations=source_orientations,
        emission_times=emission_times,
        pulse=pulse,
        sample_rate=sample_rate,
    )


def generate_synthetic_recording(
    scene: SyntheticScene | None = None,
    *,
    num_mics: int = 12,
    num_source_positions: int = 8,
    sample_rate: int = 8_000,
    seed: int = 0,
) -> SyntheticRecording:
    scene = scene or generate_synthetic_scene(
        num_mics=num_mics,
        num_source_positions=num_source_positions,
        sample_rate=sample_rate,
        seed=seed,
    )

    distances = np.linalg.norm(
        scene.source_positions[:, None, :] - scene.microphone_positions[None, :, :],
        axis=2,
    )
    gains = radiation_gain(
        source_orientations=scene.source_orientations,
        source_positions=scene.source_positions,
        microphone_positions=scene.microphone_positions,
    ) / np.maximum(distances, 0.25)

    max_arrival_time = np.max(scene.emission_times[:, None] + distances / scene.speed_of_sound)
    sample_count = int(np.ceil((max_arrival_time + 0.04) * scene.sample_rate)) + len(scene.pulse)
    audio = np.zeros((sample_count, scene.microphone_positions.shape[0]), dtype=np.float64)
    pulse_reference = pulse_reference_sample(scene.pulse)

    for source_index, emission_time in enumerate(scene.emission_times):
        for mic_index in range(scene.microphone_positions.shape[0]):
            arrival_time = emission_time + distances[source_index, mic_index] / scene.speed_of_sound
            start = int(round(arrival_time * scene.sample_rate)) - pulse_reference
            stop = start + len(scene.pulse)
            audio[start:stop, mic_index] += gains[source_index, mic_index] * scene.pulse

    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = 0.9 * audio / peak

    return SyntheticRecording(scene=scene, audio=audio)


def export_synthetic_dataset(
    output_dir: str | Path,
    recording: SyntheticRecording | None = None,
    *,
    num_mics: int = 12,
    seed: int = 0,
    stem: str | None = None,
) -> tuple[Path, Path]:
    recording = recording or generate_synthetic_recording(num_mics=num_mics, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = stem or f"synthetic_{recording.scene.microphone_positions.shape[0]}_mic"
    wav_path = output_path / f"{stem}.wav"
    metadata_path = output_path / f"{stem}.json"

    wav_data = np.clip(recording.audio, -1.0, 1.0)
    wavfile.write(wav_path, recording.scene.sample_rate, np.round(wav_data * PCM_SCALE).astype(np.int16))

    metadata = {
        "sample_rate": recording.scene.sample_rate,
        "speed_of_sound": recording.scene.speed_of_sound,
        "microphone_positions": recording.scene.microphone_positions.tolist(),
        "source_positions": recording.scene.source_positions.tolist(),
        "source_orientations": recording.scene.source_orientations.tolist(),
        "emission_times": recording.scene.emission_times.tolist(),
        "pulse": recording.scene.pulse.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return wav_path, metadata_path


def load_synthetic_dataset(wav_path: str | Path, metadata_path: str | Path) -> SyntheticRecording:
    sample_rate, audio = wavfile.read(Path(wav_path))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if sample_rate != metadata["sample_rate"]:
        raise ValueError("Sample rate in WAV file does not match metadata.")

    if np.issubdtype(audio.dtype, np.integer):
        audio_array = np.clip(audio.astype(np.float64) / PCM_SCALE, -1.0, 1.0)
    else:
        audio_array = audio.astype(np.float64)

    scene = SyntheticScene(
        microphone_positions=np.asarray(metadata["microphone_positions"], dtype=np.float64),
        source_positions=np.asarray(metadata["source_positions"], dtype=np.float64),
        source_orientations=np.asarray(metadata["source_orientations"], dtype=np.float64),
        emission_times=np.asarray(metadata["emission_times"], dtype=np.float64),
        pulse=np.asarray(metadata["pulse"], dtype=np.float64),
        sample_rate=int(metadata["sample_rate"]),
        speed_of_sound=float(metadata["speed_of_sound"]),
    )
    return SyntheticRecording(scene=scene, audio=audio_array)
