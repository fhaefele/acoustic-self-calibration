from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import medfilt


def gcc_phat(
    signal: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    max_tau: float | None = None,
    interp: int = 16,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Estimate TDOA with GCC-PHAT.

    Returns `(tau_seconds, peak_value)`. Positive tau means `signal` arrives
    later than `reference`.
    """
    sig = np.asarray(signal, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    n = len(sig) + len(ref)

    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(ref, n=n)
    cross = sig_fft * np.conj(ref_fft)
    cross /= np.maximum(np.abs(cross), eps)

    cc = np.fft.irfft(cross, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(max_shift, int(interp * sample_rate * max_tau))

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    tau = shift / float(interp * sample_rate)
    peak = float(np.max(np.abs(cc)))
    return tau, peak


def ideal_tdoa(
    microphone_positions: np.ndarray,
    source_positions: np.ndarray,
    *,
    speed_of_sound: float = 343.0,
    reference_channel: int = 0,
) -> np.ndarray:
    """Generate exact TDOAs from geometry for solver tests."""
    microphones = np.asarray(microphone_positions, dtype=float)
    sources = np.asarray(source_positions, dtype=float)
    reference_distance = np.linalg.norm(
        sources - microphones[reference_channel],
        axis=1,
    )
    columns = []
    for index in range(len(microphones)):
        if index == reference_channel:
            continue
        distance = np.linalg.norm(sources - microphones[index], axis=1)
        columns.append((distance - reference_distance) / speed_of_sound)
    return np.stack(columns, axis=1)


def make_microphone_pairs(
    mic_count: int,
    *,
    mode: str = "redundant",
    reference_count: int = 3,
) -> list[tuple[int, int]]:
    """Construct the TDOA measurement graph.

    ``reference`` uses only `(0, j)`. ``redundant`` connects the first few
    reference microphones to later channels while preserving the complete mic-0
    star needed by automatic initialization. ``all`` uses every unique pair.
    """
    if mic_count < 2:
        raise ValueError("mic_count must be at least 2")
    mode = mode.lower()
    if mode == "reference":
        return [(0, j) for j in range(1, mic_count)]
    if mode == "all":
        return [(a, b) for a in range(mic_count) for b in range(a + 1, mic_count)]
    if mode == "redundant":
        references = max(1, min(int(reference_count), mic_count - 1))
        pairs: list[tuple[int, int]] = []
        for a in range(references):
            for b in range(a + 1, mic_count):
                pairs.append((a, b))
        return pairs
    raise ValueError("mode must be 'reference', 'redundant', or 'all'")


def estimate_pairwise_tdoa_matrix(
    audio: np.ndarray,
    sample_rate: int,
    *,
    microphone_pairs: Sequence[tuple[int, int]] | None = None,
    frame_size: int = 4096,
    hop_size: int = 2048,
    max_tau: float | None = None,
    interp: int = 16,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Estimate frame-wise GCC-PHAT TDOAs for a microphone-pair graph.

    Column `p` for oriented pair `(a, b)` is `arrival[b] - arrival[a]`.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 2:
        raise ValueError("audio must have shape (samples, channels)")
    sample_count, channel_count = x.shape
    if sample_count < frame_size:
        raise ValueError("audio is shorter than one frame")

    if microphone_pairs is None:
        pairs = tuple(make_microphone_pairs(channel_count, mode="all"))
    else:
        pairs = tuple((int(a), int(b)) for a, b in microphone_pairs)
    if not pairs:
        raise ValueError("microphone_pairs cannot be empty")
    for a, b in pairs:
        if a == b or not (0 <= a < channel_count) or not (0 <= b < channel_count):
            raise ValueError(f"invalid microphone pair {(a, b)}")

    if window == "hann":
        analysis_window = np.hanning(frame_size)
    elif window == "rect":
        analysis_window = np.ones(frame_size)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

    starts = np.arange(0, sample_count - frame_size + 1, hop_size)
    tdoa = np.zeros((len(starts), len(pairs)), dtype=float)
    confidence = np.zeros_like(tdoa)
    for frame_index, start in enumerate(starts):
        frame = x[start : start + frame_size] * analysis_window[:, None]
        for pair_index, (a, b) in enumerate(pairs):
            tdoa[frame_index, pair_index], confidence[frame_index, pair_index] = gcc_phat(
                frame[:, b],
                frame[:, a],
                sample_rate,
                max_tau=max_tau,
                interp=interp,
            )

    frame_times = (starts + 0.5 * frame_size) / float(sample_rate)
    return frame_times, tdoa, confidence, pairs


def despike_tdoa_tracks(
    tdoa_matrix: np.ndarray,
    sample_rate: float,
    *,
    kernel_size: int = 5,
    threshold_samples: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated GCC peak jumps with a temporal median estimate."""
    tau = np.asarray(tdoa_matrix, dtype=float)
    if tau.ndim != 2:
        raise ValueError("tdoa_matrix must have shape (frames, measurements)")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")

    baseline = np.column_stack(
        [medfilt(tau[:, index], kernel_size=kernel_size) for index in range(tau.shape[1])]
    )
    mask = np.abs(tau - baseline) > float(threshold_samples) / float(sample_rate)
    clean = tau.copy()
    clean[mask] = baseline[mask]
    return clean, mask
