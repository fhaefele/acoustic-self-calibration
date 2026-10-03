"""Reproducible chirp scenes. All geometry here is simulator/evaluator truth."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.signal import chirp

from .simulation import (
    SyntheticPulseScene,
    make_planar_benchmark_pulse_scene,
    make_room_benchmark_pulse_scene,
    render_moving_source,
)


@dataclass(frozen=True)
class ChirpSpecification:
    start_hz: float = 18_000.0
    end_hz: float = 4_000.0
    duration_s: float = 0.0015
    sweep: Literal["linear", "logarithmic"] = "linear"
    vary_calls: bool = True

    def validate(self, sample_rate: int) -> None:
        values = [self.start_hz, self.end_hz, self.duration_s]
        if not np.isfinite(values).all() or sample_rate <= 0:
            raise ValueError("chirp parameters and sample rate must be finite and positive")
        if not 0 < min(self.start_hz, self.end_hz) < max(self.start_hz, self.end_hz):
            raise ValueError("chirp frequencies must be positive and distinct")
        # Variation only reduces frequencies, so this is also the varied-call bound.
        if max(self.start_hz, self.end_hz) >= sample_rate / 2:
            raise ValueError("chirp band must lie strictly below Nyquist")
        if self.duration_s * sample_rate < 16:
            raise ValueError("chirp duration must contain at least 16 samples")
        if self.sweep not in ("linear", "logarithmic"):
            raise ValueError("unsupported chirp sweep")


def chirp_train(
    event_times_s: np.ndarray,
    sample_rate: int,
    duration_s: float,
    *,
    specification: ChirpSpecification = ChirpSpecification(),
    seed: int = 0,
) -> np.ndarray:
    """Place chirp envelope centers at event times, with independent call variation."""
    specification.validate(sample_rate)
    events = np.asarray(event_times_s, dtype=float)
    if (
        events.ndim != 1
        or not len(events)
        or not np.isfinite(events).all()
        or np.any(np.diff(events) <= 0)
        or not np.isfinite(duration_s)
        or duration_s <= 0
    ):
        raise ValueError("event times must be finite, nonempty, and strictly increasing")
    rng = np.random.default_rng(seed)
    signal = np.zeros(int(round(duration_s * sample_rate)))
    for index, event in enumerate(events):
        varied = specification.vary_calls
        length = specification.duration_s * (rng.uniform(0.8, 1.2) if varied else 1.0)
        count = max(16, int(round(length * sample_rate)))
        count += count % 2 == 0
        times = np.arange(count) / sample_rate
        band_scale = rng.uniform(0.8, 1.0) if varied else 1.0
        f0, f1 = specification.start_hz * band_scale, specification.end_hz * band_scale
        if varied and index % 2:
            f0, f1 = f1, f0
        pulse = chirp(
            times,
            f0=f0,
            f1=f1,
            t1=times[-1],
            method=specification.sweep,
            phi=float(rng.uniform(0, 360)) if varied else 0,
        ) * np.hanning(count)
        pulse *= rng.uniform(0.7, 1.0) if varied else 1.0
        start = int(round(float(event) * sample_rate)) - count // 2
        if start < 0 or start + count > len(signal):
            raise ValueError("complete chirp envelopes must lie inside the recording")
        signal[start : start + count] += pulse
    return signal


def make_chirp_scene(
    microphone_count: int = 8,
    *,
    geometry: Literal["planar", "room"] = "planar",
    layout: str = "star",
    event_count: int = 20,
    seed: int = 10,
    sample_rate_hz: int = 48_000,
    array_span_m: float = 2.0,
    source_distance_range_m: tuple[float, float] = (1.0, 3.0),
    specification: ChirpSpecification = ChirpSpecification(),
    rapid: bool = False,
    snr_db: float | None = None,
) -> SyntheticPulseScene:
    """Replace legacy pulses with chirps without changing the legacy fixtures."""
    if geometry == "planar":
        if layout not in ("star", "cross"):
            raise ValueError("planar layout must be star or cross")
        scene = make_planar_benchmark_pulse_scene(
            microphone_count,
            array_span_m=array_span_m,
            source_distance_range_m=source_distance_range_m,
            layout=layout,
            event_count=event_count,
            seed=seed,
            sample_rate_hz=sample_rate_hz,
            noise_std=0,
        )
    elif geometry == "room":
        if layout not in ("rectangular", "irregular"):
            raise ValueError("room layout must be rectangular or irregular")
        scene = make_room_benchmark_pulse_scene(
            microphone_count,
            layout=layout,
            event_count=event_count,
            seed=seed,
            sample_rate_hz=sample_rate_hz,
            noise_std=0,
        )
    else:
        raise ValueError("geometry must be planar or room")
    events = scene.event_times_s
    if rapid:
        events = np.r_[0.4, 0.4 + np.cumsum(np.geomspace(0.15, 0.005, event_count - 1))]
    signal = chirp_train(
        events, sample_rate_hz, scene.duration_s, specification=specification, seed=seed
    )
    audio = render_moving_source(
        signal,
        sample_rate_hz,
        scene.microphone_positions_m,
        scene.trajectory_times_s,
        scene.trajectory_positions_m,
        radiation_pattern="omni",
    )
    if snr_db is not None:
        if not np.isfinite(snr_db):
            raise ValueError("snr_db must be finite or absent")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 23871]))
        for channel in range(microphone_count):
            values = audio[:, channel]
            active = np.abs(values) > 0.01 * np.max(np.abs(values))
            power = float(np.mean(values[active] ** 2))
            audio[:, channel] += rng.normal(
                scale=np.sqrt(power / 10 ** (snr_db / 10)), size=len(audio)
            )
    sources = np.column_stack(
        [
            np.interp(events, scene.trajectory_times_s, scene.trajectory_positions_m[:, dim])
            for dim in range(3)
        ]
    )
    return replace(scene, audio=audio, event_times_s=events, source_positions_at_events_m=sources)


def make_array_chirp_scene(
    microphone_positions_m: np.ndarray,
    *,
    seed: int = 10,
    event_count: int = 20,
    sample_rate_hz: int = 48000,
    specification: ChirpSpecification = ChirpSpecification(),
):
    """Render arbitrary planar fixture coordinates; truth stays outside calibration."""
    scene = make_chirp_scene(seed=seed, event_count=event_count, sample_rate_hz=sample_rate_hz)
    microphones = np.asarray(microphone_positions_m, dtype=float)
    if microphones.ndim != 2 or microphones.shape[1] != 3 or len(microphones) < 8:
        raise ValueError("fixture needs at least eight 3D microphones")
    signal = chirp_train(
        scene.event_times_s,
        sample_rate_hz,
        len(scene.audio) / sample_rate_hz,
        specification=specification,
        seed=seed,
    )
    audio = render_moving_source(
        signal,
        sample_rate_hz,
        microphones,
        scene.trajectory_times_s,
        scene.trajectory_positions_m,
        radiation_pattern="omni",
    )
    return replace(scene, microphone_positions_m=microphones, audio=audio)
