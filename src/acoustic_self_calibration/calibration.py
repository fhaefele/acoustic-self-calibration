from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import correlate, find_peaks

from .synthetic import load_synthetic_dataset, pulse_reference_sample

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationResult:
    microphone_positions: FloatArray
    source_positions: FloatArray
    estimated_distances: FloatArray
    residual_rms: float
    success: bool
    jacobian_rank: int
    parameter_count: int


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
    microphones = np.asarray(microphone_positions, dtype=np.float64).copy()
    sources = np.asarray(source_positions, dtype=np.float64).copy()
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
        half_turn = _rotation_matrix(np.array([1.0, 0.0, 0.0]), np.pi)
        microphones = microphones @ half_turn.T
        sources = sources @ half_turn.T

    # Distances and TDOAs cannot distinguish a global mirror image. Pick a stable
    # handedness from the first microphone that is off the anchor plane.
    for microphone in microphones[3:]:
        if abs(microphone[2]) > 1e-10:
            if microphone[2] < 0.0:
                microphones[:, 2] *= -1.0
                sources[:, 2] *= -1.0
            break

    microphones[np.abs(microphones) < 1e-12] = 0.0
    sources[np.abs(sources) < 1e-12] = 0.0
    return microphones, sources


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


def _validate_observation_matrix(
    observations: FloatArray,
    *,
    tdoa: bool,
) -> tuple[FloatArray, int, int, int]:
    observed = np.asarray(observations, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("Observations must have shape (num_sources, num_mics).")
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observations must contain only finite values.")

    num_sources, num_mics = observed.shape
    if num_mics < 4:
        raise ValueError("At least four microphones are required for 3D calibration.")
    if num_sources < 4:
        raise ValueError("At least four source positions are required for 3D calibration.")

    parameter_count = 3 * (num_mics + num_sources) - 6
    observation_count = num_sources * (num_mics - 1 if tdoa else num_mics)
    if observation_count < parameter_count:
        raise ValueError(
            "Calibration is underdetermined: "
            f"{observation_count} independent observations for {parameter_count} geometric parameters."
        )
    return observed, num_sources, num_mics, parameter_count


def _calibration_result(
    optimization: object,
    objective: object,
    *,
    num_mics: int,
    num_sources: int,
    parameter_count: int,
) -> CalibrationResult:
    jacobian = np.asarray(optimization.jac, dtype=np.float64)  # type: ignore[attr-defined]
    jacobian_rank = int(np.linalg.matrix_rank(jacobian))
    if jacobian_rank < parameter_count:
        raise ValueError(
            "Calibration is locally rank deficient: "
            f"Jacobian rank {jacobian_rank} < {parameter_count}."
        )

    microphones, sources = _unpack_geometry(optimization.x, num_mics=num_mics, num_sources=num_sources)  # type: ignore[attr-defined]
    residual = np.asarray(objective(optimization.x), dtype=np.float64)  # type: ignore[attr-defined]
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    microphones, sources = canonicalize_geometry(microphones, sources)
    predicted = np.linalg.norm(sources[:, None, :] - microphones[None, :, :], axis=2)
    return CalibrationResult(
        microphone_positions=microphones,
        source_positions=sources,
        estimated_distances=predicted,
        residual_rms=residual_rms,
        success=bool(optimization.success),  # type: ignore[attr-defined]
        jacobian_rank=jacobian_rank,
        parameter_count=parameter_count,
    )


def calibrate_from_distances(
    distances: FloatArray,
    *,
    max_function_evaluations: int = 20_000,
) -> CalibrationResult:
    """Recover geometry when absolute source-to-microphone ranges are known."""
    observed, num_sources, num_mics, parameter_count = _validate_observation_matrix(distances, tdoa=False)
    if np.any(observed < 0.0):
        raise ValueError("Distances must be non-negative.")

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
    return _calibration_result(
        optimization,
        objective,
        num_mics=num_mics,
        num_sources=num_sources,
        parameter_count=parameter_count,
    )


def calibrate_from_arrival_times(
    arrival_times: FloatArray,
    *,
    speed_of_sound: float = 343.0,
    max_function_evaluations: int = 30_000,
) -> CalibrationResult:
    """Self-calibrate geometry from per-event arrival times using TDOA.

    The unknown emission time of each event is removed by differencing every
    microphone against channel 0. The source events therefore do not need known
    emission timestamps.
    """
    arrivals, num_sources, num_mics, parameter_count = _validate_observation_matrix(arrival_times, tdoa=True)
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive.")

    observed_range_differences = (arrivals - arrivals[:, [0]]) * speed_of_sound
    initial = _initial_parameter_vector(num_mics=num_mics, num_sources=num_sources)

    def objective(parameters: FloatArray) -> FloatArray:
        microphones, sources = _unpack_geometry(parameters, num_mics=num_mics, num_sources=num_sources)
        distances = np.linalg.norm(sources[:, None, :] - microphones[None, :, :], axis=2)
        predicted_range_differences = distances - distances[:, [0]]
        return (predicted_range_differences[:, 1:] - observed_range_differences[:, 1:]).reshape(-1)

    optimization = least_squares(
        objective,
        initial,
        method="trf",
        loss="soft_l1",
        max_nfev=max_function_evaluations,
    )
    return _calibration_result(
        optimization,
        objective,
        num_mics=num_mics,
        num_sources=num_sources,
        parameter_count=parameter_count,
    )


def _quadratic_peak_location(values: FloatArray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-15:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index + np.clip(offset, -1.0, 1.0))


def detect_arrival_times(
    audio: FloatArray,
    reference_pulse: FloatArray,
    sample_rate: int,
    *,
    num_events: int | None = None,
    peak_prominence: float = 0.08,
) -> FloatArray:
    """Detect a repeated known pulse in multichannel audio without emission times."""
    audio_array = np.asarray(audio, dtype=np.float64)
    pulse = np.asarray(reference_pulse, dtype=np.float64)
    if audio_array.ndim != 2:
        raise ValueError("Audio must have shape (num_samples, num_mics).")
    if pulse.ndim != 1 or len(pulse) < 3:
        raise ValueError("reference_pulse must be a one-dimensional pulse with at least three samples.")
    if len(pulse) >= audio_array.shape[0]:
        raise ValueError("Audio must be longer than the reference pulse.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if num_events is not None and num_events < 4:
        raise ValueError("At least four source events are required for 3D calibration.")
    if not 0.0 < peak_prominence < 1.0:
        raise ValueError("peak_prominence must be between 0 and 1.")

    matched = np.column_stack(
        [np.abs(correlate(audio_array[:, mic_index], pulse, mode="valid")) for mic_index in range(audio_array.shape[1])]
    )
    channel_maxima = np.max(matched, axis=0)
    normalized = np.divide(
        matched,
        channel_maxima,
        out=np.zeros_like(matched),
        where=channel_maxima > 0.0,
    )
    aggregate = np.mean(normalized, axis=1)
    aggregate_peak = float(np.max(aggregate))
    if aggregate_peak <= 0.0:
        raise ValueError("No reference-pulse events were detected in the audio.")

    minimum_peak_distance = max(len(pulse), int(round(0.01 * sample_rate)))
    peaks, properties = find_peaks(
        aggregate,
        distance=minimum_peak_distance,
        prominence=peak_prominence * aggregate_peak,
        height=0.05 * aggregate_peak,
    )

    if num_events is not None:
        if len(peaks) < num_events:
            raise ValueError(f"Could only detect {len(peaks)} source events; expected {num_events}.")
        strongest = np.argsort(properties["peak_heights"])[-num_events:]
        peaks = np.sort(peaks[strongest])
    else:
        if len(peaks) > 0:
            peak_heights = aggregate[peaks]
            peaks = peaks[peak_heights >= 0.2 * np.max(peak_heights)]

    if len(peaks) < 4:
        raise ValueError(f"At least four source events are required; detected {len(peaks)}.")

    # Search each channel around the data-derived event centers. The window is
    # bounded by half the closest event spacing, rather than by a hard-coded
    # source/microphone distance.
    peak_gaps = np.diff(peaks)
    search_radius = (
        max(len(pulse), int(np.floor(np.min(peak_gaps) / 2.0)) - 1)
        if len(peak_gaps) > 0
        else 2 * len(pulse)
    )
    pulse_reference = pulse_reference_sample(pulse)
    arrival_samples = np.empty((len(peaks), audio_array.shape[1]), dtype=np.float64)

    for event_index, event_center in enumerate(peaks):
        search_start = max(0, int(event_center) - search_radius)
        search_stop = min(matched.shape[0], int(event_center) + search_radius + 1)
        for mic_index in range(audio_array.shape[1]):
            local_match = matched[search_start:search_stop, mic_index]
            local_peak = int(np.argmax(local_match))
            refined_peak = _quadratic_peak_location(local_match, local_peak)
            arrival_samples[event_index, mic_index] = search_start + refined_peak + pulse_reference

    return arrival_samples / sample_rate


def calibrate_from_audio(
    audio: FloatArray,
    reference_pulse: FloatArray,
    sample_rate: int,
    *,
    speed_of_sound: float = 343.0,
    num_events: int | None = None,
    max_function_evaluations: int = 30_000,
) -> CalibrationResult:
    """Self-calibrate a microphone array from multichannel pulse recordings."""
    arrival_times = detect_arrival_times(
        audio,
        reference_pulse,
        sample_rate,
        num_events=num_events,
    )
    return calibrate_from_arrival_times(
        arrival_times,
        speed_of_sound=speed_of_sound,
        max_function_evaluations=max_function_evaluations,
    )


def calibrate_from_dataset(
    wav_path: str | Path,
    metadata_path: str | Path,
    *,
    num_events: int | None = None,
    max_function_evaluations: int = 30_000,
) -> CalibrationResult:
    """Calibrate a synthetic fixture without using its ground-truth geometry or emission times."""
    recording = load_synthetic_dataset(wav_path=wav_path, metadata_path=metadata_path)
    return calibrate_from_audio(
        recording.audio,
        recording.scene.pulse,
        recording.scene.sample_rate,
        speed_of_sound=recording.scene.speed_of_sound,
        num_events=num_events,
        max_function_evaluations=max_function_evaluations,
    )
