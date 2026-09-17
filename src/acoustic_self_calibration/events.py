from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks


@dataclass(frozen=True)
class EventDetection:
    event_samples: np.ndarray
    event_times_s: np.ndarray
    event_channel: int
    prominence: np.ndarray


@dataclass(frozen=True)
class EventTDOAMeasurements:
    event_samples: np.ndarray
    event_times_s: np.ndarray
    event_channel: int
    arrival_delays_s: np.ndarray
    arrival_confidence: np.ndarray
    tdoa_s: np.ndarray
    confidence: np.ndarray
    microphone_pairs: tuple[tuple[int, int], ...]


def _validate_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=float)
    if values.ndim != 2:
        raise ValueError("audio must have shape (samples, microphones)")
    if values.shape[1] < 4:
        raise ValueError("At least four microphones are required for 3-D calibration")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("audio contains non-finite samples")
    return values


def _energy_envelope(audio: np.ndarray, smooth_samples: int) -> np.ndarray:
    return uniform_filter1d(
        audio * audio,
        size=max(1, int(smooth_samples)),
        axis=0,
        mode="nearest",
    )


def _channel_event_candidates(
    energy: np.ndarray,
    *,
    minimum_gap_samples: int,
    relative_prominence: float,
) -> tuple[np.ndarray, np.ndarray]:
    peaks, properties = find_peaks(
        energy,
        distance=max(1, int(minimum_gap_samples)),
        prominence=0.0,
    )
    prominence = np.asarray(properties["prominences"], dtype=float)
    if prominence.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    threshold = float(np.max(prominence)) * float(relative_prominence)
    keep = prominence >= threshold
    return np.asarray(peaks[keep], dtype=int), prominence[keep]


def detect_transient_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    event_channel: int | None = None,
    smooth_s: float = 0.0003,
    min_gap_s: float = 0.003,
    relative_prominence: float = 0.003,
    min_events: int = 4,
) -> EventDetection:
    """Detect discrete broadband/transient emissions on one microphone channel.

    If ``event_channel`` is omitted, every channel is scored and the channel with
    the largest summed prominence of retained transients is selected. The
    prominence threshold is relative to the strongest transient on each channel,
    which makes the detector insensitive to absolute WAV scaling.
    """
    values = _validate_audio(audio, sample_rate)
    if smooth_s <= 0:
        raise ValueError("smooth_s must be positive")
    if min_gap_s <= 0:
        raise ValueError("min_gap_s must be positive")
    if not 0 < relative_prominence <= 1:
        raise ValueError("relative_prominence must lie in (0, 1]")
    if min_events < 4:
        raise ValueError("min_events must be at least 4")
    if event_channel is not None and not 0 <= int(event_channel) < values.shape[1]:
        raise ValueError("event_channel is outside the available microphone channels")

    smooth_samples = max(1, int(round(float(smooth_s) * sample_rate)))
    minimum_gap_samples = max(1, int(round(float(min_gap_s) * sample_rate)))
    envelope = _energy_envelope(values, smooth_samples)

    channels = range(values.shape[1]) if event_channel is None else (int(event_channel),)
    best_channel = -1
    best_samples = np.empty(0, dtype=int)
    best_prominence = np.empty(0, dtype=float)
    best_score = -np.inf
    for channel in channels:
        samples, prominence = _channel_event_candidates(
            envelope[:, channel],
            minimum_gap_samples=minimum_gap_samples,
            relative_prominence=relative_prominence,
        )
        score = float(np.sum(prominence))
        if score > best_score:
            best_channel = int(channel)
            best_samples = samples
            best_prominence = prominence
            best_score = score

    if len(best_samples) < min_events:
        raise ValueError(
            f"detected only {len(best_samples)} transient events; "
            f"at least {min_events} are required"
        )

    return EventDetection(
        event_samples=best_samples,
        event_times_s=best_samples.astype(float) / float(sample_rate),
        event_channel=best_channel,
        prominence=best_prominence,
    )


def _normalized_correlation_curve(
    signal: np.ndarray,
    *,
    center_sample: int,
    reference_channel: int,
    target_channel: int,
    half_template_samples: int,
    max_lag_samples: int,
) -> np.ndarray:
    template_length = 2 * half_template_samples + 1
    start = center_sample - max_lag_samples - half_template_samples
    stop = center_sample + max_lag_samples + half_template_samples + 1
    template = signal[
        center_sample - half_template_samples : center_sample + half_template_samples + 1,
        reference_channel,
    ].astype(float, copy=True)
    target = signal[start:stop, target_channel].astype(float, copy=False)

    template -= float(np.mean(template))
    template_energy = float(np.dot(template, template))
    if template_energy <= 1e-30:
        return np.zeros(2 * max_lag_samples + 1, dtype=float)

    correlation = correlate(target, template, mode="valid", method="fft")
    ones = np.ones(template_length, dtype=float)
    moving_sum = np.convolve(target, ones, mode="valid")
    moving_square_sum = np.convolve(target * target, ones, mode="valid")
    moving_variance = moving_square_sum - moving_sum * moving_sum / float(template_length)
    denominator = np.sqrt(np.maximum(template_energy * moving_variance, 1e-30))
    return np.asarray(correlation / denominator, dtype=float)


def _parabolic_peak_offset(score: np.ndarray, index: int) -> float:
    """Return a quadratic sub-sample correction for one discrete peak."""
    if index <= 0 or index >= len(score) - 1:
        return 0.0
    left = float(score[index - 1])
    center = float(score[index])
    right = float(score[index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) <= 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def _lag_candidates(
    correlation: np.ndarray,
    *,
    max_lag_samples: int,
    candidate_count: int,
    minimum_peak_spacing_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.clip(np.asarray(correlation, dtype=float), 0.0, 1.0)
    peaks, _ = find_peaks(
        score,
        distance=max(1, int(minimum_peak_spacing_samples)),
    )
    global_peak = int(np.argmax(score))
    if peaks.size == 0:
        peaks = np.array([global_peak], dtype=int)
    elif not np.any(peaks == global_peak):
        peaks = np.concatenate([peaks, np.array([global_peak], dtype=int)])
    order = np.argsort(score[peaks])[::-1][:candidate_count]
    selected = peaks[order]
    refined = np.array(
        [
            float(index - max_lag_samples) + _parabolic_peak_offset(score, int(index))
            for index in selected
        ],
        dtype=float,
    )
    return refined, np.clip(score[selected], 1e-4, 1.0)


def _select_smooth_lag_track(
    candidates: Sequence[tuple[np.ndarray, np.ndarray]],
    event_samples: np.ndarray,
    sample_rate: int,
    *,
    max_tdoa_rate: float,
    transition_weight: float,
    transition_floor_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    for event_index, (lags, score) in enumerate(candidates):
        data_cost = -np.log(np.clip(score, 1e-4, 1.0))
        if event_index == 0:
            costs.append(data_cost)
            backpointers.append(np.full(len(lags), -1, dtype=int))
            continue

        previous_lags = candidates[event_index - 1][0]
        dt_s = (event_samples[event_index] - event_samples[event_index - 1]) / float(sample_rate)
        scale_s = transition_floor_s + max_tdoa_rate * dt_s
        lag_change_s = (lags[:, None] - previous_lags[None, :]) / float(sample_rate)
        transition_cost = transition_weight * np.square(lag_change_s / scale_s)
        total = costs[-1][None, :] + transition_cost
        best_previous = np.argmin(total, axis=1)
        costs.append(data_cost + total[np.arange(len(lags)), best_previous])
        backpointers.append(best_previous)

    path = np.empty(len(candidates), dtype=int)
    path[-1] = int(np.argmin(costs[-1]))
    for event_index in range(len(candidates) - 1, 0, -1):
        path[event_index - 1] = backpointers[event_index][path[event_index]]

    lags = np.array(
        [candidates[index][0][path[index]] for index in range(len(candidates))],
        dtype=float,
    )
    confidence = np.array(
        [candidates[index][1][path[index]] for index in range(len(candidates))],
        dtype=float,
    )
    return lags, confidence


def estimate_event_tdoas(
    audio: np.ndarray,
    sample_rate: int,
    event_samples: np.ndarray,
    event_channel: int,
    *,
    microphone_pairs: Sequence[tuple[int, int]],
    max_tau_s: float = 0.01,
    envelope_smooth_s: float = 0.00008,
    template_s: float = 0.0018,
    candidate_count: int = 8,
    max_tdoa_rate: float = 0.05,
    transition_weight: float = 0.4,
) -> EventTDOAMeasurements:
    """Estimate one cycle-consistent TDOA set for every detected transient event.

    For each non-reference channel, multiple normalized envelope-correlation peaks
    are retained and refined to sub-sample lag estimates with local quadratic peak
    interpolation. A dynamic-programming track then chooses a temporally smooth
    delay sequence, suppressing repeated-waveform and neighbouring-call
    ambiguities. Pairwise TDOAs are derived from the selected per-channel arrival
    delays, so TDOA cycle consistency is exact.
    """
    values = _validate_audio(audio, sample_rate)
    events = np.asarray(event_samples, dtype=int).reshape(-1)
    if len(events) < 4 or np.any(np.diff(events) <= 0):
        raise ValueError("event_samples must contain at least 4 strictly increasing samples")
    if not 0 <= int(event_channel) < values.shape[1]:
        raise ValueError("event_channel is outside the available microphone channels")
    if max_tau_s <= 0 or envelope_smooth_s <= 0 or template_s <= 0:
        raise ValueError("max_tau_s, envelope_smooth_s, and template_s must be positive")
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if max_tdoa_rate <= 0 or transition_weight < 0:
        raise ValueError("max_tdoa_rate must be positive and transition_weight non-negative")

    pairs = tuple((int(a), int(b)) for a, b in microphone_pairs)
    if not pairs:
        raise ValueError("microphone_pairs cannot be empty")
    for a, b in pairs:
        if a == b or not (0 <= a < values.shape[1]) or not (0 <= b < values.shape[1]):
            raise ValueError(f"invalid microphone pair {(a, b)}")

    max_lag_samples = max(1, int(round(max_tau_s * sample_rate)))
    half_template_samples = max(4, int(round(0.5 * template_s * sample_rate)))
    margin = max_lag_samples + half_template_samples + 1
    usable = (events >= margin) & (events < len(values) - margin)
    events = events[usable]
    if len(events) < 4:
        raise ValueError(
            "fewer than 4 detected events have enough surrounding audio for TDOA estimation"
        )

    envelope = _energy_envelope(
        values,
        max(1, int(round(envelope_smooth_s * sample_rate))),
    )
    microphone_count = values.shape[1]
    arrival_lags = np.zeros((len(events), microphone_count), dtype=float)
    arrival_confidence = np.ones((len(events), microphone_count), dtype=float)
    minimum_peak_spacing = max(1, int(round(0.0002 * sample_rate)))

    for channel in range(microphone_count):
        if channel == event_channel:
            continue
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        for center in events:
            correlation = _normalized_correlation_curve(
                envelope,
                center_sample=int(center),
                reference_channel=int(event_channel),
                target_channel=channel,
                half_template_samples=half_template_samples,
                max_lag_samples=max_lag_samples,
            )
            candidates.append(
                _lag_candidates(
                    correlation,
                    max_lag_samples=max_lag_samples,
                    candidate_count=candidate_count,
                    minimum_peak_spacing_samples=minimum_peak_spacing,
                )
            )
        selected_lags, selected_confidence = _select_smooth_lag_track(
            candidates,
            events,
            sample_rate,
            max_tdoa_rate=max_tdoa_rate,
            transition_weight=transition_weight,
            transition_floor_s=max(2.0 / sample_rate, 50e-6),
        )
        arrival_lags[:, channel] = selected_lags
        arrival_confidence[:, channel] = selected_confidence

    arrival_delays_s = arrival_lags / float(sample_rate)
    tdoa = np.empty((len(events), len(pairs)), dtype=float)
    confidence = np.empty_like(tdoa)
    for column, (a, b) in enumerate(pairs):
        tdoa[:, column] = arrival_delays_s[:, b] - arrival_delays_s[:, a]
        confidence[:, column] = np.sqrt(arrival_confidence[:, a] * arrival_confidence[:, b])

    return EventTDOAMeasurements(
        event_samples=events,
        event_times_s=events.astype(float) / float(sample_rate),
        event_channel=int(event_channel),
        arrival_delays_s=arrival_delays_s,
        arrival_confidence=arrival_confidence,
        tdoa_s=tdoa,
        confidence=confidence,
        microphone_pairs=pairs,
    )
