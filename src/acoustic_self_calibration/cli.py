from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .bayesian import DistancePrior
from .export import export_calibration
from .wav import calibrate_wav


def _distance_prior(value: str) -> DistancePrior:
    try:
        microphone_a, microphone_b, distance_m, sigma_m = value.split(",")
        return DistancePrior(
            microphone_a=int(microphone_a),
            microphone_b=int(microphone_b),
            distance_m=float(distance_m),
            sigma_m=float(sigma_m),
        )
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "distance priors must be MIC_A,MIC_B,DISTANCE_M,SIGMA_M"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustic-selfcal",
        description="Bayesian 3-D self-calibration from a moving broadband source in a multichannel WAV file.",
    )
    parser.add_argument("wav", type=Path, help="input multichannel WAV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output prefix (default: INPUT_calibration)",
    )
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=4096)
    parser.add_argument("--max-tau-ms", type=float, default=30.0)
    parser.add_argument("--gcc-interp", type=int, default=16)
    parser.add_argument(
        "--pair-mode", choices=("reference", "redundant", "all"), default="redundant"
    )
    parser.add_argument("--reference-count", type=int, default=2)
    parser.add_argument("--likelihood", choices=("cauchy", "gaussian"), default="cauchy")
    parser.add_argument("--speed-of-sound", type=float, default=343.0)
    parser.add_argument("--motion-sigma-mps", type=float, default=3.0)
    parser.add_argument("--estimate-clock-offsets", action="store_true")
    parser.add_argument("--estimate-clock-drifts", action="store_true")
    parser.add_argument("--estimate-speed-of-sound", action="store_true")
    parser.add_argument(
        "--distance-prior",
        action="append",
        default=[],
        type=_distance_prior,
        metavar="MIC_A,MIC_B,DISTANCE_M,SIGMA_M",
        help="known microphone baseline; repeat as needed",
    )
    parser.add_argument("--best-sigma-samples", type=float, default=0.35)
    parser.add_argument("--worst-sigma-samples", type=float, default=4.0)
    parser.add_argument("--max-nfev", type=int, default=4000)
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="skip the Laplace uncertainty calculation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    if output is None:
        output = args.wav.with_suffix("").with_name(f"{args.wav.stem}_calibration")

    result = calibrate_wav(
        args.wav,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        max_tau_s=args.max_tau_ms / 1000.0,
        gcc_interp=args.gcc_interp,
        pair_mode=args.pair_mode,
        reference_count=args.reference_count,
        speed_of_sound=args.speed_of_sound,
        motion_velocity_change_sigma_mps=args.motion_sigma_mps,
        likelihood=args.likelihood,
        estimate_clock_offsets=args.estimate_clock_offsets,
        estimate_clock_drifts=args.estimate_clock_drifts,
        estimate_speed_of_sound=args.estimate_speed_of_sound,
        distance_priors=tuple(args.distance_prior),
        best_sigma_samples=args.best_sigma_samples,
        worst_sigma_samples=args.worst_sigma_samples,
        max_nfev=args.max_nfev,
        compute_laplace_uncertainty=not args.no_uncertainty,
    )
    paths = export_calibration(result, output)
    calibration = result.calibration

    print(f"success: {calibration.success}")
    print(f"microphones: {len(calibration.microphone_positions)}")
    print(f"source states: {len(calibration.source_positions)}")
    print(f"TDOA RMS residual: {1e6 * calibration.rms_tdoa_residual_s:.3f} us")
    print(f"NPZ: {paths.npz}")
    print(f"JSON: {paths.json}")
    print(f"microphones CSV: {paths.microphones_csv}")
    print(f"trajectory CSV: {paths.trajectory_csv}")
    if not calibration.success:
        print(f"optimizer message: {calibration.message}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
