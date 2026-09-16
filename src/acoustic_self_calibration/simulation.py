from __future__ import annotations

import numpy as np

from .radiation import radiation_gain


def _interp_vec(times: np.ndarray, key_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.empty((len(times), values.shape[1]), dtype=float)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(times, key_times, values[:, dim])
    return out


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def render_moving_source(
    source_signal: np.ndarray,
    sample_rate: int,
    microphone_positions: np.ndarray,
    trajectory_times: np.ndarray,
    trajectory_positions: np.ndarray,
    *,
    source_forward: np.ndarray | None = None,
    radiation_pattern: str = "omni",
    speed_of_sound: float = 343.0,
    min_distance: float = 0.15,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a moving point source in a free field.

    The renderer solves a few fixed-point iterations for retarded emission time:

        t_receive = t_emit + ||mic - source(t_emit)|| / c

    and samples the source signal at `t_emit`. This captures moving propagation
    delay (and the associated first-order Doppler timing effect) without a room model.

    Returns
    -------
    audio:
        Array with shape (samples, microphones).
    """
    x = np.asarray(source_signal, dtype=float).reshape(-1)
    mics = np.asarray(microphone_positions, dtype=float)
    tt = np.asarray(trajectory_times, dtype=float).reshape(-1)
    ss = np.asarray(trajectory_positions, dtype=float)

    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("microphone_positions must have shape (M, 3)")
    if ss.shape != (len(tt), 3):
        raise ValueError("trajectory_positions must have shape (K, 3)")
    if np.any(np.diff(tt) <= 0):
        raise ValueError("trajectory_times must be strictly increasing")

    n = len(x)
    receive_t = np.arange(n, dtype=float) / float(sample_rate)
    source_t = np.arange(n, dtype=float) / float(sample_rate)
    out = np.zeros((n, len(mics)), dtype=float)

    if source_forward is None:
        # Estimate a forward direction from trajectory velocity.
        vel = np.gradient(ss, tt, axis=0)
        forward_keys = _normalize(vel)
        bad = np.linalg.norm(vel, axis=1) < 1e-8
        forward_keys[bad] = np.array([1.0, 0.0, 0.0])
    else:
        forward_keys = np.asarray(source_forward, dtype=float)
        if forward_keys.shape == (3,):
            forward_keys = np.repeat(forward_keys[None, :], len(tt), axis=0)
        if forward_keys.shape != ss.shape:
            raise ValueError("source_forward must be shape (3,) or (K, 3)")
        forward_keys = _normalize(forward_keys)

    for i, mic in enumerate(mics):
        # Initial emission-time estimate from source position at receive time.
        s_receive = _interp_vec(receive_t, tt, ss)
        dist = np.linalg.norm(mic - s_receive, axis=1)
        emit_t = receive_t - dist / speed_of_sound

        # Refine retarded time using the source location at the previous estimate.
        for _ in range(4):
            s_emit = _interp_vec(emit_t, tt, ss)
            dist = np.linalg.norm(mic - s_emit, axis=1)
            emit_t = receive_t - dist / speed_of_sound

        s_emit = _interp_vec(emit_t, tt, ss)
        ray = mic - s_emit
        dist = np.maximum(np.linalg.norm(ray, axis=1), min_distance)
        direction = ray / dist[:, None]

        fwd = _normalize(_interp_vec(emit_t, tt, forward_keys))
        cos_theta = np.sum(fwd * direction, axis=1)
        gain = radiation_gain(cos_theta, radiation_pattern)

        emitted = np.interp(emit_t, source_t, x, left=0.0, right=0.0)
        out[:, i] = emitted * gain / dist

    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng()
        out += rng.normal(scale=noise_std, size=out.shape)

    return out


def random_microphone_array(
    count: int,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-2.0, 2.0),
        (-2.0, 2.0),
        (0.0, 2.5),
    ),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if count < 4:
        raise ValueError("At least 4 microphones are needed for 3D calibration")
    if rng is None:
        rng = np.random.default_rng()

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(lo, hi, size=(count, 3))


def apply_channel_clock_model(
    audio: np.ndarray,
    sample_rate: int,
    *,
    clock_offsets_s: np.ndarray | None = None,
    clock_drifts: np.ndarray | None = None,
) -> np.ndarray:
    """Apply per-channel timing offset and linear clock drift to rendered audio.

    Positive offset makes a channel appear later. Drift is seconds of timing error
    per second, centered on the recording midpoint, matching `calibrate_bayesian`.
    Channel 0 is usually left at zero and acts as the timing reference.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 2:
        raise ValueError("audio must have shape (samples, channels)")
    n, channels = x.shape
    offsets = np.zeros(channels) if clock_offsets_s is None else np.asarray(clock_offsets_s, float)
    drifts = np.zeros(channels) if clock_drifts is None else np.asarray(clock_drifts, float)
    if offsets.shape != (channels,) or drifts.shape != (channels,):
        raise ValueError("clock offset/drift arrays must have shape (channels,)")

    t = np.arange(n, dtype=float) / float(sample_rate)
    centered = t - float(np.mean(t))
    out = np.empty_like(x)
    for ch in range(channels):
        delay = offsets[ch] + drifts[ch] * centered
        source_t = t - delay
        out[:, ch] = np.interp(source_t, t, x[:, ch], left=0.0, right=0.0)
    return out
