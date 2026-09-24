from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    per second, centered on the recording midpoint. Channel 0 defines the timing gauge.
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


@dataclass(frozen=True)
class CalibrationAcceptanceLimits:
    """Frozen geometry/TDOA limits carried forward from PR #7."""

    microphone_rms_m: float
    source_rms_m: float
    tdoa_rms_s: float


@dataclass(frozen=True)
class SyntheticGeometryContract:
    """Expected observable outcome for one synthetic geometry fixture."""

    name: str
    expected_outcome: str
    constraint_type: str | None = None
    constraint_value: float | None = None
    source_sign_convention_required: bool = False


RANDOM_3D_EVENT_LIMITS = CalibrationAcceptanceLimits(
    microphone_rms_m=0.15,
    source_rms_m=0.18,
    tdoa_rms_s=60e-6,
)
PCM16_EVENT_LIMITS = CalibrationAcceptanceLimits(
    microphone_rms_m=0.18,
    source_rms_m=0.22,
    tdoa_rms_s=60e-6,
)
CONSTRAINED_MYOTIS_CROSS_LIMITS = CalibrationAcceptanceLimits(
    microphone_rms_m=0.20,
    source_rms_m=0.30,
    tdoa_rms_s=45e-6,
)


def nondegenerate_planar_fixture_contract() -> SyntheticGeometryContract:
    """Contract for the positive mixed-dimensional planar fixture."""
    return SyntheticGeometryContract(
        name="nondegenerate_planar",
        expected_outcome="planar_success",
    )


def myotis_cross_fixture_contract() -> SyntheticGeometryContract:
    """Contract for the unconstrained idealized Myotis cross."""
    return SyntheticGeometryContract(
        name="myotis_cross",
        expected_outcome="cross_ambiguity",
    )


def constrained_myotis_cross_fixture_contract() -> SyntheticGeometryContract:
    """Contract for a cross run with an explicit right-angle arm constraint."""
    return SyntheticGeometryContract(
        name="myotis_cross_constrained",
        expected_outcome="constrained_cross_observable_success",
        constraint_type="inter_arm_angle_rad",
        constraint_value=float(np.pi / 2.0),
        source_sign_convention_required=True,
    )


@dataclass(frozen=True)
class SyntheticPulseScene:
    """Deterministic rendered pulse fixture used by stratified calibration tests."""

    sample_rate_hz: int
    duration_s: float
    microphone_positions_m: np.ndarray
    trajectory_times_s: np.ndarray
    trajectory_positions_m: np.ndarray
    event_times_s: np.ndarray
    source_positions_at_events_m: np.ndarray
    audio: np.ndarray


@dataclass(frozen=True)
class EventCountSweepCase:
    """Static bookkeeping for the sample-efficiency diagnostic sweep."""

    microphone_count: int
    event_count: int
    seed: int
    independent_measurement_count: int
    geometry_unknown_count: int
    measurement_excess: int


def broadband_pulse_train(
    event_times_s: np.ndarray,
    sample_rate: int,
    duration_s: float,
    *,
    pulse_duration_s: float = 0.0015,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate the deterministic-style broadband pulse train used by PR #7 fixtures."""
    times = np.asarray(event_times_s, dtype=float).reshape(-1)
    if sample_rate <= 0 or duration_s <= 0 or pulse_duration_s <= 0:
        raise ValueError("sample_rate, duration_s, and pulse_duration_s must be positive")
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("event_times_s must be non-empty and strictly increasing")
    if times[0] < 0 or times[-1] >= duration_s:
        raise ValueError("event_times_s must lie inside the rendered duration")
    if rng is None:
        rng = np.random.default_rng()

    sample_count = int(round(duration_s * sample_rate))
    pulse_samples = max(16, int(round(pulse_duration_s * sample_rate)))
    pulse = rng.normal(size=pulse_samples)
    pulse = np.concatenate([np.zeros(1), np.diff(pulse)])
    pulse *= np.hanning(pulse_samples)
    pulse /= max(float(np.max(np.abs(pulse))), 1e-12)

    signal = np.zeros(sample_count, dtype=float)
    for time_s in times:
        start = int(round(float(time_s) * sample_rate))
        stop = min(sample_count, start + pulse_samples)
        signal[start:stop] += pulse[: stop - start]
    return signal


def _interpolate_positions(
    event_times_s: np.ndarray,
    trajectory_times_s: np.ndarray,
    trajectory_positions_m: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(event_times_s, trajectory_times_s, trajectory_positions_m[:, dimension])
            for dimension in range(3)
        ]
    )


def make_random_3d_pulse_scene(
    microphone_count: int,
    *,
    event_count: int = 20,
) -> SyntheticPulseScene:
    """Reproduce the pinned PR #7 random-3D pulse fixture.

    The 20-event form intentionally preserves seed, geometry, trajectory, waveform,
    renderer settings, and noise from PR #7. The 40-event form changes only the
    linspace event schedule.
    """
    if event_count < 4:
        raise ValueError("event_count must be at least 4")
    rng = np.random.default_rng(100 + microphone_count)
    sample_rate = 48_000
    duration = 5.0
    microphones = random_microphone_array(
        microphone_count,
        bounds=((-1.2, 1.2), (-1.2, 1.2), (0.0, 1.6)),
        rng=rng,
    )

    trajectory_times = np.linspace(0.0, duration, 31)
    phase = np.linspace(0.0, 2.0 * np.pi, len(trajectory_times))
    trajectory_positions = np.column_stack(
        [
            2.0 * np.cos(0.75 * phase),
            1.6 * np.sin(0.75 * phase),
            1.2 + 0.5 * np.sin(0.55 * phase + 0.2),
        ]
    )
    event_times = np.linspace(0.4, 4.4, event_count)
    source_signal = broadband_pulse_train(
        event_times,
        sample_rate,
        duration,
        pulse_duration_s=0.0015,
        rng=rng,
    )
    source_forward = microphones.mean(axis=0)[None, :] - trajectory_positions
    source_forward /= np.linalg.norm(source_forward, axis=1, keepdims=True)
    audio = render_moving_source(
        source_signal,
        sample_rate,
        microphones,
        trajectory_times,
        trajectory_positions,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=1e-5,
        rng=rng,
    )
    return SyntheticPulseScene(
        sample_rate_hz=sample_rate,
        duration_s=duration,
        microphone_positions_m=microphones,
        trajectory_times_s=trajectory_times,
        trajectory_positions_m=trajectory_positions,
        event_times_s=event_times,
        source_positions_at_events_m=_interpolate_positions(
            event_times,
            trajectory_times,
            trajectory_positions,
        ),
        audio=audio,
    )


def myotis_cross_microphones() -> np.ndarray:
    """Return the pinned 12-microphone idealized Myotis cross fixture."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [1.2, 0.0, 0.8],
            [1.2, 0.0, 0.4],
            [1.2, 0.0, -0.4],
            [1.2, 0.0, -0.8],
            [1.2, 0.0, -1.2],
        ],
        dtype=float,
    )


def myotis_cross_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Return the pinned trajectory keys for the idealized Myotis fixture."""
    duration = 2.4
    trajectory_times = np.linspace(0.0, duration, 25)
    progress = trajectory_times / duration
    trajectory_positions = np.column_stack(
        [
            1.03 * (1.0 - progress) ** 1.25,
            2.70 - 1.90 * progress + 0.08 * np.sin(2.0 * np.pi * progress),
            -0.25 + 0.23 * progress + 0.02 * np.sin(3.0 * np.pi * progress),
        ]
    )
    return trajectory_times, trajectory_positions


def myotis_cross_source_positions(event_times_s: np.ndarray) -> np.ndarray:
    """Interpolate the pinned Myotis trajectory at arbitrary event times."""
    event_times = np.asarray(event_times_s, dtype=float).reshape(-1)
    trajectory_times, trajectory_positions = myotis_cross_trajectory()
    if len(event_times) == 0 or event_times[0] < 0 or event_times[-1] > trajectory_times[-1]:
        raise ValueError("event_times_s must lie inside the Myotis fixture duration")
    if np.any(np.diff(event_times) <= 0):
        raise ValueError("event_times_s must be strictly increasing")
    return _interpolate_positions(event_times, trajectory_times, trajectory_positions)


def make_myotis_cross_pulse_scene() -> SyntheticPulseScene:
    """Reproduce the pinned 18-event idealized Myotis audio fixture."""
    rng = np.random.default_rng(812)
    sample_rate = 96_000
    duration = 2.4
    microphones = myotis_cross_microphones()
    trajectory_times, trajectory_positions = myotis_cross_trajectory()
    event_times = np.linspace(0.25, 2.05, 18)
    source_signal = broadband_pulse_train(
        event_times,
        sample_rate,
        duration,
        pulse_duration_s=0.0012,
        rng=rng,
    )
    source_forward = microphones.mean(axis=0)[None, :] - trajectory_positions
    source_forward /= np.linalg.norm(source_forward, axis=1, keepdims=True)
    audio = render_moving_source(
        source_signal,
        sample_rate,
        microphones,
        trajectory_times,
        trajectory_positions,
        source_forward=source_forward,
        radiation_pattern="cardioid",
        noise_std=8e-6,
        rng=rng,
    )
    return SyntheticPulseScene(
        sample_rate_hz=sample_rate,
        duration_s=duration,
        microphone_positions_m=microphones,
        trajectory_times_s=trajectory_times,
        trajectory_positions_m=trajectory_positions,
        event_times_s=event_times,
        source_positions_at_events_m=myotis_cross_source_positions(event_times),
        audio=audio,
    )


def planar_benchmark_microphones(
    microphone_count: int, *, array_span_m: float, layout: Literal["cross", "star"] = "cross"
) -> np.ndarray:
    """Nested planar arrays in the x/z plane, with channel 0 at the center.

    Layout, angles, spacings, and span are generation truth only. The cross
    is an ambiguity fixture when only planarity and source side are known.
    """
    if microphone_count not in (8, 12, 16, 24):
        raise ValueError("microphone_count must be 8, 12, 16, or 24")
    if not np.isfinite(array_span_m) or array_span_m <= 0:
        raise ValueError("array_span_m must be finite and positive")
    if layout == "cross":
        directions = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        )
    elif layout == "star":
        angles = np.arange(6) * np.pi / 3.0
        directions = np.column_stack((np.cos(angles), np.zeros(6), np.sin(angles)))
        directions[:, 2] /= np.max(np.abs(directions[:, 2]))
    else:
        raise ValueError("layout must be cross or star")
    fractions = (1.0, 0.5, 0.25, 0.75, 0.125, 0.375)
    points = [np.zeros(3)]
    for fraction in fractions:
        points.extend(0.5 * array_span_m * fraction * directions)
    return np.asarray(points[:microphone_count])


def make_planar_benchmark_pulse_scene(
    microphone_count: int,
    *,
    array_span_m: float,
    source_distance_range_m: tuple[float, float],
    layout: Literal["cross", "star"] = "cross",
    event_count: int = 40,
    seed: int = 0,
    sample_rate_hz: int = 48_000,
    noise_std: float = 1e-5,
) -> SyntheticPulseScene:
    """Render a planar benchmark with source distance measured normal to the plane.

    Source motion spans x, y, and z and stays in y > 0. Identical seeds keep
    trajectory and emitted pulses fixed across microphone counts. This is a
    direct-path, synchronized-channel fixture, without simulated reflections.
    """
    microphones = planar_benchmark_microphones(
        microphone_count, array_span_m=array_span_m, layout=layout
    )
    near, far = source_distance_range_m
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("source distances must be finite and satisfy 0 < near < far")
    if event_count not in (20, 40):
        raise ValueError("event_count must be 20 or 40")
    if sample_rate_hz <= 0 or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("sample rate must be positive and noise_std finite and nonnegative")
    duration = 5.0
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    times = np.linspace(0.0, duration, 201)
    progress = times / duration
    positions = np.column_stack(
        [
            0.4 * array_span_m * np.sin(2.0 * np.pi * progress + phases[0]),
            near + (far - near) * (0.5 + 0.5 * np.sin(2.0 * np.pi * progress + phases[1])),
            0.4 * array_span_m * np.sin(3.0 * np.pi * progress + phases[2]),
        ]
    )
    return _render_benchmark_pulse_scene(
        microphones, times, positions, event_count, sample_rate_hz, noise_std, rng
    )


def _render_benchmark_pulse_scene(
    microphones: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    event_count: int,
    sample_rate_hz: int,
    noise_std: float,
    rng: np.random.Generator,
) -> SyntheticPulseScene:
    duration = float(times[-1])
    event_times = np.linspace(0.4, 4.4, event_count)
    signal = broadband_pulse_train(event_times, sample_rate_hz, duration, rng=rng)
    audio = render_moving_source(
        signal,
        sample_rate_hz,
        microphones,
        times,
        positions,
        radiation_pattern="omni",
        noise_std=noise_std,
        rng=rng,
    )
    return SyntheticPulseScene(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration,
        microphone_positions_m=microphones,
        trajectory_times_s=times,
        trajectory_positions_m=positions,
        event_times_s=event_times,
        source_positions_at_events_m=_interpolate_positions(event_times, times, positions),
        audio=audio,
    )


def room_benchmark_floor_polygon(
    *,
    room_span_m: tuple[float, float] = (6.0, 5.0),
    layout: Literal["rectangular", "irregular"] = "rectangular",
) -> np.ndarray:
    """Return counterclockwise convex floor vertices, for generation/evaluation only."""
    span = np.asarray(room_span_m, dtype=float)
    if span.shape != (2,) or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("room_span_m must contain two finite positive dimensions")
    if layout == "rectangular":
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elif layout == "irregular":
        vertices = np.array([[0.0, 0.15], [0.65, 0.0], [1.0, 0.3], [0.85, 1.0], [0.15, 0.9]])
    else:
        raise ValueError("layout must be rectangular or irregular")
    return vertices * span


def room_benchmark_microphones(
    microphone_count: int,
    *,
    room_span_m: tuple[float, float] = (6.0, 5.0),
    room_height_m: float = 3.0,
    layout: Literal["rectangular", "irregular"] = "rectangular",
    seed: int = 0,
) -> np.ndarray:
    """Nested wall microphones with unknown spacings and varying mounting heights.

    Each wall receives a channel before any wall receives a second channel.
    The layout and dimensions are generation truth, not solver constraints.
    """
    if microphone_count not in (8, 12, 16, 24):
        raise ValueError("microphone_count must be 8, 12, 16, or 24")
    if not np.isfinite(room_height_m) or room_height_m <= 0:
        raise ValueError("room_height_m must be finite and positive")
    vertices = room_benchmark_floor_polygon(room_span_m=room_span_m, layout=layout)
    rng = np.random.default_rng(seed)
    wall_indices = np.arange(24) % len(vertices)
    fractions = rng.uniform(0.1, 0.9, 24)
    horizontal = (1.0 - fractions[:, None]) * vertices[wall_indices] + fractions[
        :, None
    ] * vertices[(wall_indices + 1) % len(vertices)]
    heights = room_height_m * rng.uniform(0.15, 0.85, 24)
    return np.column_stack((horizontal, heights))[:microphone_count]


def make_room_benchmark_pulse_scene(
    microphone_count: int,
    *,
    room_span_m: tuple[float, float] = (6.0, 5.0),
    room_height_m: float = 3.0,
    layout: Literal["rectangular", "irregular"] = "rectangular",
    event_count: int = 40,
    seed: int = 0,
    sample_rate_hz: int = 48_000,
    noise_std: float = 1e-5,
) -> SyntheticPulseScene:
    """Render direct sound in a room-shaped volume, without wall reflections.

    The smooth 3-D source path and every interpolated segment remain strictly
    inside the convex room. Synchronized channels use omnidirectional sources.
    Matched seeds preserve microphones, trajectory, and pulses across counts.
    """
    microphones = room_benchmark_microphones(
        microphone_count,
        room_span_m=room_span_m,
        room_height_m=room_height_m,
        layout=layout,
        seed=seed,
    )
    if event_count not in (20, 40):
        raise ValueError("event_count must be 20 or 40")
    if sample_rate_hz <= 0 or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("sample rate must be positive and noise_std finite and nonnegative")
    vertices = room_benchmark_floor_polygon(room_span_m=room_span_m, layout=layout)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(vertices) + 1)
    times = np.linspace(0.0, 5.0, 201)
    progress = times / times[-1]
    # Strictly positive convex weights guarantee containment for either room.
    frequencies = np.arange(1, len(vertices) + 1)
    weights = np.exp(
        1.8 * np.sin(2.0 * np.pi * progress[:, None] * frequencies / 2.0 + phases[:-1])
    )
    weights /= weights.sum(axis=1, keepdims=True)
    horizontal = weights @ vertices
    heights = room_height_m * (0.5 + 0.3 * np.sin(3.0 * np.pi * progress + phases[-1]))
    positions = np.column_stack((horizontal, heights))
    return _render_benchmark_pulse_scene(
        microphones, times, positions, event_count, sample_rate_hz, noise_std, rng
    )


def nondegenerate_planar_microphones() -> np.ndarray:
    """Return a deterministic generic planar array that is not a two-line cross."""
    return np.array(
        [
            [-1.10, 0.0, -0.65],
            [-0.72, 0.0, 0.38],
            [-0.25, 0.0, -0.18],
            [0.18, 0.0, 0.72],
            [0.54, 0.0, -0.82],
            [0.92, 0.0, 0.16],
            [1.18, 0.0, 0.91],
            [1.42, 0.0, -0.37],
        ],
        dtype=float,
    )


def nondegenerate_planar_sources(event_times_s: np.ndarray) -> np.ndarray:
    """Return a deterministic 3-D source path for the generic planar fixture."""
    times = np.asarray(event_times_s, dtype=float).reshape(-1)
    if len(times) < 4 or np.any(np.diff(times) <= 0):
        raise ValueError("event_times_s must contain at least 4 strictly increasing values")
    progress = (times - times[0]) / max(float(times[-1] - times[0]), 1e-12)
    return np.column_stack(
        [
            -0.9 + 1.8 * progress,
            1.2 + 0.55 * np.sin(1.1 * np.pi * progress + 0.2),
            0.35 + 0.8 * np.sin(1.7 * np.pi * progress - 0.3),
        ]
    )


def exact_reference_tdoas(
    microphone_positions_m: np.ndarray,
    source_positions_m: np.ndarray,
    *,
    reference_microphone: int = 0,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """Return exact reference-star TDOAs for a static set of source events."""
    microphones = np.asarray(microphone_positions_m, dtype=float)
    sources = np.asarray(source_positions_m, dtype=float)
    if microphones.ndim != 2 or microphones.shape[1] != 3:
        raise ValueError("microphone_positions_m must have shape (M, 3)")
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("source_positions_m must have shape (E, 3)")
    if not 0 <= reference_microphone < len(microphones):
        raise ValueError("reference_microphone is out of range")
    if speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be positive")

    ranges = np.linalg.norm(
        sources[:, None, :] - microphones[None, :, :],
        axis=2,
    )
    reference = ranges[:, [reference_microphone]]
    keep = [index for index in range(len(microphones)) if index != reference_microphone]
    return (ranges[:, keep] - reference) / float(speed_of_sound)


def event_count_sweep_cases(
    microphone_count: int,
    *,
    event_counts: tuple[int, ...] = (8, 12, 20, 30, 40),
) -> tuple[EventCountSweepCase, ...]:
    """Return static bookkeeping rows for the event-count conditioning sweep."""
    if microphone_count < 4:
        raise ValueError("microphone_count must be at least 4")
    if any(count < 4 for count in event_counts):
        raise ValueError("all event counts must be at least 4")
    seed = 100 + microphone_count
    rows = []
    for event_count in event_counts:
        independent = (microphone_count - 1) * event_count
        unknowns = 3 * microphone_count + 3 * event_count - 6
        rows.append(
            EventCountSweepCase(
                microphone_count=microphone_count,
                event_count=event_count,
                seed=seed,
                independent_measurement_count=independent,
                geometry_unknown_count=unknowns,
                measurement_excess=independent - unknowns,
            )
        )
    return tuple(rows)
