from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import correlate

from .synthetic import SyntheticRecording, load_synthetic_dataset, pulse_reference_sample

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationResult:
    microphone_positions: FloatArray
    source_positions: FloatArray
    estimated_distances: FloatArray
    residual_rms: float
    success: bool


def _softplus(value: float) -> float:
    return np.log1p(np.exp(-abs(value))) + max(value, 0.0) + 1e-3


def _rotation_matrix(axis: FloatArray, angle: float) -> FloatArray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine],
        ],
        dtype=np.float64,
    )


def canonicalize_geometry(
    microphone_positions: FloatArray,
    source_positions: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    microphones = np.asarray(microphone_positions, dtype=np.float64)
    sources = np.asarray(source_positions, dtype=np.float64)
    if microphones.ndim != 2 or microphones.shape[1] != 3 or microphones.shape[0] < 3:
        raise ValueError("At least three 3D microphone positions are required to canonicalize geometry.")
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("Source positions must be a two-dimensional array with shape (n, 3).")

    origin = microphones[0].copy()
    microphones = microphones - origin
    sources = sources - origin

    mic1 = microphones[1]
    mic1_norm = np.linalg.norm(mic1)
    if mic1_norm == 0.0:
        raise ValueError("Microphone anchors must not coincide.")
    x_axis = np.array([1.0, 0.0, 0.0])
    rotation_axis = np.cross(mic1, x_axis)
    axis_norm = np.linalg.norm(rotation_axis)
    if axis_norm > 1e-12:
        angle = np.arccos(np.clip(np.dot(mic1 / mic1_norm, x_axis), -1.0, 1.0))
        first_rotation = _rotation_matrix(rotation_axis / axis_norm, angle)
        microphones = microphones @ first_rotation.T
        sources = sources @ first_rotation.T
    elif mic1[0] < 0.0:
        half_turn = _rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi)
        microphones = microphones @ half_turn.T
        sources = sources @ half_turn.T

    mic2 = microphones[2]
    if np.hypot(mic2[1], mic2[2]) < 1e-12:
        raise ValueError("The first three microphone anchors must not be collinear.")
    x_rotation = -np.arctan2(mic2[2], mic2[1])
    second_rotation = _rotation_matrix(np.array([1.0, 0.0, 0.0]), x_rotation)
    microphones = microphones @ second_rotation.T
    sources = sources @ second_rotation.T

    if microphones[2, 1] < 0.0:
        flip = _rotation_matrix(np.array([1.0, 0.0, 0.0]), np.pi)
        microphones = microphones @ flip.T
        sources = sources @ flip.T

    microphones[np.abs(microphones) < 1e-12] = 0.0
    sources[np.abs(sources) < 1e-12] = 0.0
    return microphones, sources


def _arrival_samples_from_recording(recording: SyntheticRecording) -> FloatArray:
    pulse = recording.scene.pulse
    pulse_length = len(pulse)
    pulse_reference = pulse_reference_sample(pulse)
    sample_rate = recording.scene.sample_rate
    max_delay_samples = int(np.ceil(3.0 / recording.scene.speed_of_sound * sample_rate))
    arrival_samples = np.zeros(
        (len(recording.scene.emission_times), recording.audio.shape[1]),
        dtype=np.float64,
    )

    for source_index, emission_time in enumerate(recording.scene.emission_times):
        expected_start = int(round(emission_time * sample_rate))
        search_start = max(0, expected_start - pulse_length)
        search_stop = min(recording.audio.shape[0], expected_start + max_delay_samples + pulse_length)
        for mic_index in range(recording.audio.shape[1]):
            segment = recording.audio[search_start:search_stop, mic_index]
            matched = correlate(segment, pulse, mode="valid")
            peak_offset = int(np.argmax(matched))
            arrival_samples[source_index, mic_index] = search_start + peak_offset + pulse_reference
    return arrival_samples


def _initial_parameter_vector(num_mics: int, num_sources: int) -> FloatArray:
    mic_angles = np.linspace(0.0, 2.0 * np.pi, num_mics, endpoint=False)
    mic_radii = np.full(num_mics, 1.0)
    mic_heights = np.linspace(-0.3, 0.3, num_mics)
    mic_guess = np.column_stack((mic_radii * np.cos(mic_angles), mic_radii * np.sin(mic_angles), mic_heights))

    source_angles = np.linspace(0.0, 1.8 * np.pi, num_sources, endpoint=False)
    source_radii = np.linspace(0.3, 0.6, num_sources)
    source_guess = np.column_stack(
        (
            source_radii * np.cos(source_angles),
            source_radii * np.sin(source_angles),
            np.linspace(-0.4, 0.4, num_sources),
        )
    )
    mic_guess, source_guess = canonicalize_geometry(mic_guess, source_guess)

    params = [
        np.log(np.expm1(max(mic_guess[1, 0], 0.1))),
        mic_guess[2, 0],
        np.log(np.expm1(max(mic_guess[2, 1], 0.1))),
    ]
    if num_mics > 3:
        params.extend(mic_guess[3:].reshape(-1).tolist())
    params.extend(source_guess.reshape(-1).tolist())
    return np.asarray(params, dtype=np.float64)


def _unpack_geometry(parameters: FloatArray, num_mics: int, num_sources: int) -> tuple[FloatArray, FloatArray]:
    index = 0
    microphones = np.zeros((num_mics, 3), dtype=np.float64)
    microphones[1] = [_softplus(parameters[index]), 0.0, 0.0]
    index += 1
    microphones[2] = [parameters[index], _softplus(parameters[index + 1]), 0.0]
    index += 2

    if num_mics > 3:
        tail_length = 3 * (num_mics - 3)
        microphones[3:] = parameters[index : index + tail_length].reshape(num_mics - 3, 3)
        index += tail_length

    sources = parameters[index:].reshape(num_sources, 3)
    return microphones, sources


def calibrate_from_distances(
    distances: FloatArray,
    *,
    max_function_evaluations: int = 20_000,
) -> CalibrationResult:
    observed = np.asarray(distances, dtype=np.float64)
    num_sources, num_mics = observed.shape
    if num_mics < 4:
        raise ValueError("At least four microphones are required for 3D calibration.")
    if num_sources < 4:
        raise ValueError("At least four source positions are required for 3D calibration.")

    initial = _initial_parameter_vector(num_mics=num_mics, num_sources=num_sources)

    def objective(parameters: FloatArray) -> FloatArray:
        microphones, sources = _unpack_geometry(parameters, num_mics=num_mics, num_sources=num_sources)
        predicted = np.linalg.norm(sources[:, None, :] - microphones[None, :, :], axis=2)
        return (predicted - observed).reshape(-1)

    optimization = least_squares(
        objective,
        initial,
        method="trf",
        loss="soft_l1",
        max_nfev=max_function_evaluations,
    )

    microphones, sources = _unpack_geometry(optimization.x, num_mics=num_mics, num_sources=num_sources)
    predicted = np.linalg.norm(sources[:, None, :] - microphones[None, :, :], axis=2)
    residual_rms = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    microphones, sources = canonicalize_geometry(microphones, sources)
    predicted = np.linalg.norm(sources[:, None, :] - microphones[None, :, :], axis=2)
    return CalibrationResult(
        microphone_positions=microphones,
        source_positions=sources,
        estimated_distances=predicted,
        residual_rms=residual_rms,
        success=bool(optimization.success),
    )


def calibrate_from_dataset(wav_path: str | Path, metadata_path: str | Path) -> CalibrationResult:
    recording = load_synthetic_dataset(wav_path=wav_path, metadata_path=metadata_path)
    arrival_samples = _arrival_samples_from_recording(recording)
    arrival_times = arrival_samples / recording.scene.sample_rate
    distances = (arrival_times - recording.scene.emission_times[:, None]) * recording.scene.speed_of_sound
    return calibrate_from_distances(distances)
