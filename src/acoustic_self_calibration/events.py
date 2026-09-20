from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks

from .measurements import EventTDOAMeasurements, reference_star_from_arrivals


@dataclass(frozen=True)
class EventDetection:
    """Receiver-side transient event landmarks on one acquisition channel."""

    event_samples: np.ndarray
    receiver_event_times_s: np.ndarray
    event_channel: int
    prominence: np.ndarray

    def __post_init__(self) -> None:
        samples = np.array(self.event_samples, dtype=np.int64, copy=True).reshape(-1)
        times = np.array(self.receiver_event_times_s, dtype=float, copy=True).reshape(-1)
        prominence = np.array(self.prominence, dtype=float, copy=True).reshape(-1)
        if len(samples) < 4:
            raise ValueError("at least four detected events are required")
        if samples.shape != times.shape or samples.shape != prominence.shape:
            raise ValueError("event detection arrays must have the same shape")
        if np.any(np.diff(samples) <= 0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("detected events must be strictly increasing")
        if np.any(samples < 0):
            raise ValueError("event_samples must be non-negative")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(prominence)):
            raise ValueError("event detection arrays must be finite")
        if np.any(prominence < 0.0):
            raise ValueError("event prominence must be non-negative")
        for array in (samples, times, prominence):
            array.setflags(write=False)
        object.__setattr__(self, "event_samples", samples)
        object.__setattr__(self, "receiver_event_times_s", times)
        object.__setattr__(self, "prominence", prominence)
        object.__setattr__(self, "event_channel", int(self.event_channel))

    @property
    def event_times_s(self) -> np.ndarray:
        """Compatibility alias; these are receiver-side times, not emission times."""
        return self.receiver_event_times_s


def _validate_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=float)
    if values.ndim != 2:
        raise ValueError("audio must have shape (samples, microphones)")
    if values.shape[1] < 4:
        raise ValueError("at least four microphones are required")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("audio contains non-finite samples")
    return values


def energy_envelope(audio: np.ndarray, smooth_samples: int) -> np.ndarray:
    """Return channel-wise smoothed short-time energy."""
    values = np.asarray(audio, dtype=float)
    if values.ndim != 2:
        raise ValueError("audio must have shape (samples, microphones)")
    if smooth_samples < 1:
        raise ValueError("smooth_samples must be positive")
    return uniform_filter1d(
        values * values,
        size=int(smooth_samples),
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
        np.asarray(energy, dtype=float),
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
    """Detect discrete broadband/transient emissions on one receiver channel.

    If event_channel is absent, every channel is scored independently and the channel
    with the largest summed retained-event prominence is selected. Thresholds are
    relative to each channel's strongest event, so integer/floating WAV scaling does
    not change the selected event samples.
    """
    values = _validate_audio(audio, sample_rate)
    if smooth_s <= 0.0:
        raise ValueError("smooth_s must be positive")
    if min_gap_s <= 0.0:
        raise ValueError("min_gap_s must be positive")
    if not 0.0 < relative_prominence <= 1.0:
        raise ValueError("relative_prominence must lie in (0, 1]")
    if min_events < 4:
        raise ValueError("min_events must be at least 4")
    if event_channel is not None and not 0 <= int(event_channel) < values.shape[1]:
        raise ValueError("event_channel is outside the available microphone channels")

    smooth_samples = max(1, int(round(float(smooth_s) * sample_rate)))
    minimum_gap_samples = max(1, int(round(float(min_gap_s) * sample_rate)))
    envelope = energy_envelope(values, smooth_samples)

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
        receiver_event_times_s=best_samples.astype(float) / float(sample_rate),
        event_channel=best_channel,
        prominence=best_prominence,
    )


def arrival_sigma_from_confidence(
    confidence: np.ndarray,
    sample_rate: float,
    *,
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    floor_quantile: float = 0.1,
    ceiling_quantile: float = 0.9,
) -> np.ndarray:
    """Map relative correlation confidence to per-arrival timing uncertainty."""
    values = np.asarray(confidence, dtype=float)
    if values.ndim != 2:
        raise ValueError("confidence must have shape (events, microphones)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("confidence contains no finite values")
    if best_sigma_samples <= 0.0 or worst_sigma_samples < best_sigma_samples:
        raise ValueError("sigma sample limits are invalid")
    lo = float(np.quantile(finite, floor_quantile))
    hi = float(np.quantile(finite, ceiling_quantile))
    quality = (
        np.ones_like(values) if hi <= lo + 1e-15 else np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    )
    sigma_samples = worst_sigma_samples - quality * (worst_sigma_samples - best_sigma_samples)
    return sigma_samples / float(sample_rate)


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
    candidates: list[tuple[np.ndarray, np.ndarray]],
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


def _select_independent_lags(
    candidates: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    lags = np.array([item[0][0] for item in candidates], dtype=float)
    confidence = np.array([item[1][0] for item in candidates], dtype=float)
    return lags, confidence


def estimate_event_tdoa_measurements(
    audio: np.ndarray,
    sample_rate: int,
    detection: EventDetection,
    *,
    max_tau_s: float = 0.01,
    envelope_smooth_s: float = 0.00008,
    template_s: float = 0.0018,
    candidate_count: int = 8,
    max_tdoa_rate: float = 0.05,
    transition_weight: float = 0.4,
    use_temporal_tracking: bool = True,
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
) -> EventTDOAMeasurements:
    """Estimate cycle-consistent event TDOAs in a mic-0 reference-star basis.

    Temporal tracking is an optional association heuristic only. Setting
    use_temporal_tracking=False chooses the strongest correlation peak independently
    for every event/channel and is used as a diagnostic for frontend prior sensitivity.
    """
    values = _validate_audio(audio, sample_rate)
    events = np.asarray(detection.event_samples, dtype=int).reshape(-1)
    event_channel = int(detection.event_channel)
    if not 0 <= event_channel < values.shape[1]:
        raise ValueError("event_channel is outside the available microphone channels")
    if max_tau_s <= 0.0 or envelope_smooth_s <= 0.0 or template_s <= 0.0:
        raise ValueError("max_tau_s, envelope_smooth_s, and template_s must be positive")
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if max_tdoa_rate <= 0.0 or transition_weight < 0.0:
        raise ValueError("max_tdoa_rate must be positive and transition_weight non-negative")

    max_lag_samples = max(1, int(round(max_tau_s * sample_rate)))
    half_template_samples = max(
        4,
        int(round(0.5 * template_s * sample_rate)),
    )
    margin = max_lag_samples + half_template_samples + 1
    usable = (events >= margin) & (events < len(values) - margin)
    events = events[usable]
    if len(events) < 4:
        raise ValueError("fewer than four events have enough surrounding audio for TDOA estimation")

    microphone_count = values.shape[1]
    arrival_lags = np.zeros((len(events), microphone_count), dtype=float)
    arrival_confidence = np.ones(
        (len(events), microphone_count),
        dtype=float,
    )
    minimum_peak_spacing = max(1, int(round(0.0002 * sample_rate)))

    for channel in range(microphone_count):
        if channel == event_channel:
            continue
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        for center in events:
            # Correlate raw waveforms (T-004): envelope correlation limits timing
            # to ~10us RMS, whose exact rank residuals are O(1) and defeat the
            # minimal solver. Raw correlation of the broadband pulse reaches
            # ~4us RMS. Detection still uses the energy envelope.
            correlation = _normalized_correlation_curve(
                values,
                center_sample=int(center),
                reference_channel=event_channel,
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
        if use_temporal_tracking:
            selected_lags, selected_confidence = _select_smooth_lag_track(
                candidates,
                events,
                sample_rate,
                max_tdoa_rate=max_tdoa_rate,
                transition_weight=transition_weight,
                transition_floor_s=max(2.0 / sample_rate, 50e-6),
            )
        else:
            selected_lags, selected_confidence = _select_independent_lags(candidates)
        arrival_lags[:, channel] = selected_lags
        arrival_confidence[:, channel] = selected_confidence

    event_channel_delays_s = arrival_lags / float(sample_rate)
    reference_arrivals_s = event_channel_delays_s - event_channel_delays_s[:, [0]]
    arrival_sigma_s = arrival_sigma_from_confidence(
        arrival_confidence,
        sample_rate,
        best_sigma_samples=best_sigma_samples,
        worst_sigma_samples=worst_sigma_samples,
    )
    event_times_s = events.astype(float) / float(sample_rate)

    return reference_star_from_arrivals(
        reference_arrivals_s,
        arrival_sigma_s,
        receiver_event_times_s=event_times_s,
        reference_microphone=0,
        arrival_confidence=arrival_confidence,
        arrival_valid=np.ones_like(reference_arrivals_s, dtype=bool),
        event_ids=np.arange(len(events), dtype=np.int64),
        event_samples=events,
        sample_rate_hz=sample_rate,
        event_channel=event_channel,
    )
