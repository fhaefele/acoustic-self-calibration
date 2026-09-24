"""Synthetic calibration sweeps; one JSON result per case, no geometry truth in solver."""

from __future__ import annotations

import argparse
import json
import time
from typing import Literal

import numpy as np

from acoustic_self_calibration import (
    calibrate_audio,
    calibrate_planar_tdoa,
    calibrate_tdoa,
    reference_star_from_arrivals,
)
from acoustic_self_calibration.geometry import (
    apply_rigid,
    rigid_align,
    rms_position_error,
)
from acoustic_self_calibration.simulation import (
    SyntheticPulseScene,
    make_planar_benchmark_pulse_scene,
    make_random_3d_pulse_scene,
    make_room_benchmark_pulse_scene,
)


def validate(microphone_count: int, event_count: int) -> dict[str, float | int | str]:
    scene = make_random_3d_pulse_scene(
        microphone_count,
        event_count=event_count,
    )
    started = time.perf_counter()
    result = calibrate_audio(
        scene.audio,
        scene.sample_rate_hz,
        event_min_gap_s=0.05,
        max_tau_s=0.02,
        tdoa_template_s=0.002,
        model="general_3d",
    )
    solve_seconds = time.perf_counter() - started

    if result.microphone_positions_m is None or result.source_positions_m is None:
        return {
            "microphones": microphone_count,
            "events": event_count,
            "status": result.status,
            "solve_seconds": round(solve_seconds, 4),
        }

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions_m,
        scene.microphone_positions_m,
    )
    aligned_source = apply_rigid(
        result.source_positions_m,
        rotation,
        translation,
    )
    return {
        "microphones": microphone_count,
        "events": event_count,
        "status": result.status,
        "solve_seconds": round(solve_seconds, 4),
        "microphone_rms_error_m": round(
            rms_position_error(aligned_microphones, scene.microphone_positions_m),
            6,
        ),
        "source_rms_error_m": round(
            rms_position_error(
                aligned_source,
                scene.source_positions_at_events_m,
            ),
            6,
        ),
        "tdoa_residual_rms_us": round(
            1e6 * (result.rms_tdoa_residual_s or 0.0),
            4,
        ),
    }


def validate_scene(
    scene: SyntheticPulseScene,
    *,
    planar: bool,
    stage: Literal["exact", "noisy", "audio"],
    timing_noise_us: float = 2.0,
    noise_seed: int = 0,
) -> dict[str, object]:
    """Solve using observations only; use coordinates only for evaluation."""
    started = time.perf_counter()
    if stage in ("exact", "noisy"):
        arrivals = (
            np.linalg.norm(
                scene.source_positions_at_events_m[:, None, :]
                - scene.microphone_positions_m[None, :, :],
                axis=2,
            )
            / 343.0
        )
        receiver_times = scene.event_times_s + arrivals[:, 0]
        sigma_s = 1e-9
        if stage == "noisy":
            if not np.isfinite(timing_noise_us) or timing_noise_us <= 0.0:
                raise ValueError("timing_noise_us must be finite and positive")
            sigma_s = timing_noise_us * 1e-6
            rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 92741]))
            arrivals += rng.normal(0.0, sigma_s, arrivals.shape)
            receiver_times = scene.event_times_s + arrivals[:, 0]
        # A positive numerical sigma is required even for noiseless observations.
        # Remove the unknown emission-time/range offset before the solver boundary.
        measurements = reference_star_from_arrivals(
            arrivals - arrivals[:, [0]],
            np.full_like(arrivals, sigma_s),
            receiver_event_times_s=receiver_times,
            reference_microphone=0,
        )
        # Planar benchmark sources are one-sided; sign is a gauge convention
        # relative to select_coordinate_gauge, not a truth-derived world axis.
        if planar:
            result = calibrate_planar_tdoa(
                measurements, speed_of_sound=343.0, source_half_space_sign=1
            )
        else:
            result = calibrate_tdoa(measurements, speed_of_sound=343.0)
    else:
        audio_result = calibrate_audio(
            scene.audio,
            scene.sample_rate_hz,
            event_min_gap_s=0.05,
            max_tau_s=0.03,
            tdoa_template_s=0.002,
            model="receiver2d_source3d" if planar else "general_3d",
        )
        result = audio_result.calibration
    row: dict[str, object] = {
        "status": result.status,
        "solve_seconds": round(time.perf_counter() - started, 4),
        "reported_events": len(result.event_ids),
        "microphone_rms_error_m": None,
        "tdoa_residual_rms_us": (
            None if result.tdoa_rms_s is None else round(1e6 * result.tdoa_rms_s, 6)
        ),
        "rejection_reasons": list(dict.fromkeys(result.diagnostics.rejection_reasons)),
        "solver_geometry_prior": "planarity" if planar else "none",
        "source_region_constraint_enforced": bool(
            getattr(result, "source_region_constraint_enforced", False)
        ),
        "arrival_sigma_us": timing_noise_us
        if stage == "noisy"
        else (0.001 if stage == "exact" else None),
        "timing_noise_seed": noise_seed if stage == "noisy" else None,
    }
    for name in (
        "planar_microphone_rms_std_m",
        "planar_relative_std",
        "planar_fit_p_value",
        "planar_reduced_chi_square",
    ):
        value = getattr(result.diagnostics, name)
        row[name] = None if value is None else (value if np.isfinite(value) else "unbounded")
    if result.microphone_positions_m is not None:
        ids = result.microphone_ids or tuple(range(len(result.microphone_positions_m)))
        target = scene.microphone_positions_m[list(ids)]
        aligned, _, _ = rigid_align(result.microphone_positions_m, target)
        row["microphone_rms_error_m"] = round(rms_position_error(aligned, target), 6)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("legacy", "planar", "room"), default="legacy")
    parser.add_argument("--stage", choices=("exact", "noisy", "audio"), default="audio")
    parser.add_argument(
        "--microphones", type=int, nargs="+", choices=(8, 12, 16, 24), default=[8, 12, 16, 24]
    )
    parser.add_argument("--events", type=int, nargs="+", choices=(20, 40), default=[20, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--timing-noise-us",
        type=float,
        default=2.0,
        help="per-channel arrival standard deviation for noisy TDOAs",
    )
    parser.add_argument("--layouts", nargs="+")
    parser.add_argument(
        "--array-spans", type=float, nargs="+", choices=(2.0, 4.0), default=[2.0, 4.0]
    )
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")
    if not np.isfinite(args.timing_noise_us) or args.timing_noise_us <= 0.0:
        parser.error("--timing-noise-us must be finite and positive")
    allowed_layouts = {
        "legacy": ("random_3d",),
        "planar": ("cross", "star"),
        "room": ("rectangular", "irregular"),
    }[args.suite]
    layouts = args.layouts or allowed_layouts
    if any(layout not in allowed_layouts for layout in layouts):
        parser.error(f"--layouts for {args.suite} must be among {allowed_layouts}")
    if args.suite == "legacy" and args.seeds != [0]:
        parser.error("legacy fixtures have fixed seeds; use --seeds 0")
    # Emit each completed case immediately so interrupted sweeps retain results.
    for count in args.microphones:
        for events in args.events:
            for seed in args.seeds:
                for layout in layouts:
                    spans = args.array_spans if args.suite == "planar" else [None]
                    for span in spans:
                        metadata = {
                            "suite": args.suite,
                            "stage": args.stage,
                            "microphones": count,
                            "events": events,
                            "seed": seed if args.suite != "legacy" else 100 + count,
                            "layout": layout,
                        }
                        if args.suite == "legacy" and args.stage == "audio":
                            print(json.dumps(metadata | validate(count, events)), flush=True)
                            continue
                        if args.suite == "planar":
                            assert span is not None
                            assert layout in ("cross", "star")
                            distance = (1.0, 3.0 if span == 2.0 else 6.0)
                            metadata.update(array_span_m=span, source_distance_range_m=distance)
                            scene = make_planar_benchmark_pulse_scene(
                                count,
                                array_span_m=span,
                                source_distance_range_m=distance,
                                layout=layout,
                                event_count=events,
                                seed=seed,
                            )
                        elif args.suite == "room":
                            assert layout in ("rectangular", "irregular")
                            scene = make_room_benchmark_pulse_scene(
                                count,
                                layout=layout,
                                event_count=events,
                                seed=seed,
                            )
                            metadata.update(room_span_m=[6.0, 5.0], room_height_m=3.0)
                        else:
                            scene = make_random_3d_pulse_scene(count, event_count=events)
                        print(
                            json.dumps(
                                metadata
                                | validate_scene(
                                    scene,
                                    planar=args.suite == "planar",
                                    stage=args.stage,
                                    timing_noise_us=args.timing_noise_us,
                                    noise_seed=seed,
                                )
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
