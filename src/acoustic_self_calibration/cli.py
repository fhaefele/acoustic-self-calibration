from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .bayesian import DistancePrior
from .export import write_calibration_outputs, write_scene_comparison_outputs
from .ground_truth import validate_ground_truth_json
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


def _add_solver_options(parser: argparse.ArgumentParser) -> None:
    audio = parser.add_argument_group("audio and TDOA")
    audio.add_argument("--frame-size", type=int, default=1024)
    audio.add_argument("--hop-size", type=int, default=4096)
    audio.add_argument("--max-tau-ms", type=float, default=30.0)
    audio.add_argument("--gcc-interp", type=int, default=16)
    audio.add_argument(
        "--pair-mode",
        choices=("reference", "redundant", "all"),
        default="redundant",
    )
    audio.add_argument("--reference-count", type=int, default=2)

    model = parser.add_argument_group("model and solver")
    model.add_argument("--likelihood", choices=("cauchy", "gaussian"), default="cauchy")
    model.add_argument("--speed-of-sound", type=float, default=343.0)
    model.add_argument("--motion-sigma-mps", type=float, default=3.0)
    model.add_argument("--best-sigma-samples", type=float, default=0.35)
    model.add_argument("--worst-sigma-samples", type=float, default=4.0)
    model.add_argument("--max-nfev", type=int, default=4000)

    clocks = parser.add_argument_group("clocks and metric scale")
    clocks.add_argument("--estimate-clock-offsets", action="store_true")
    clocks.add_argument("--estimate-clock-drifts", action="store_true")
    clocks.add_argument("--estimate-speed-of-sound", action="store_true")
    clocks.add_argument(
        "--distance-prior",
        action="append",
        default=[],
        type=_distance_prior,
        metavar="MIC_A,MIC_B,DISTANCE_M,SIGMA_M",
        help="known microphone baseline; repeat as needed",
    )

    uncertainty = parser.add_argument_group("uncertainty")
    uncertainty.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="skip the Laplace uncertainty calculation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asc",
        description="Bayesian 3-D acoustic self-calibration and scene comparison.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('acoustic-self-calibration')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="calibrate a multichannel WAV recording",
        description="Calibrate a microphone array and moving source from a multichannel WAV file.",
    )
    calibrate.add_argument("wav", type=Path, help="input multichannel WAV file")
    calibrate.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output prefix for RESULT.json and RESULT.png (default: INPUT_calibration)",
    )
    calibrate.add_argument(
        "-r",
        "--reference",
        type=Path,
        help="canonical scene JSON used as the comparison reference",
    )
    _add_solver_options(calibrate)
    calibrate.set_defaults(handler=_run_calibrate)

    check = subparsers.add_parser(
        "check",
        help="validate a canonical scene JSON",
        description="Validate a ground-truth or estimate JSON scene.",
    )
    check.add_argument("json", type=Path, help="canonical scene JSON to validate")
    check.set_defaults(handler=_run_check)

    compare = subparsers.add_parser(
        "compare",
        help="compare two canonical scene JSON files",
        description="Align ESTIMATE to REFERENCE using microphones and compare both scenes.",
    )
    compare.add_argument("estimate", type=Path, help="estimate scene JSON")
    compare.add_argument("reference", type=Path, help="reference scene JSON")
    compare.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output prefix for comparison JSON and PNG",
    )
    compare.set_defaults(handler=_run_compare)
    return parser


def _settings_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "max_tau_s": args.max_tau_ms / 1000.0,
        "gcc_interp": args.gcc_interp,
        "pair_mode": args.pair_mode,
        "reference_count": args.reference_count,
        "likelihood": args.likelihood,
        "speed_of_sound_mps": args.speed_of_sound,
        "motion_velocity_change_sigma_mps": args.motion_sigma_mps,
        "estimate_clock_offsets": args.estimate_clock_offsets,
        "estimate_clock_drifts": args.estimate_clock_drifts,
        "estimate_speed_of_sound": args.estimate_speed_of_sound,
        "distance_priors": [
            {
                "microphone_a": prior.microphone_a,
                "microphone_b": prior.microphone_b,
                "distance_m": prior.distance_m,
                "sigma_m": prior.sigma_m,
            }
            for prior in args.distance_prior
        ],
        "best_sigma_samples": args.best_sigma_samples,
        "worst_sigma_samples": args.worst_sigma_samples,
        "max_nfev": args.max_nfev,
        "compute_laplace_uncertainty": not args.no_uncertainty,
    }


def _run_calibrate(args: argparse.Namespace) -> int:
    output = args.output
    if output is None:
        output = args.wav.with_suffix("").with_name(f"{args.wav.stem}_calibration")

    reference = None if args.reference is None else validate_ground_truth_json(args.reference)
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
    paths = write_calibration_outputs(
        result,
        output,
        input_wav_path=args.wav,
        settings=_settings_dict(args),
        ground_truth=reference,
    )
    calibration = result.calibration
    print(f"success: {calibration.success}")
    print(f"microphones: {len(calibration.microphone_positions)}")
    print(f"source states: {len(calibration.source_positions)}")
    print(f"TDOA RMS residual: {1e6 * calibration.rms_tdoa_residual_s:.3f} us")
    print(f"JSON: {paths.json}")
    print(f"figure: {paths.figure}")
    if not calibration.success:
        print(f"optimizer message: {calibration.message}")
        return 2
    return 0


def _run_check(args: argparse.Namespace) -> int:
    try:
        scene = validate_ground_truth_json(args.json)
    except (OSError, ValueError) as error:
        print(f"invalid scene JSON: {error}", file=sys.stderr)
        return 2

    print("valid canonical scene JSON")
    print(f"role: {scene.scene_role}")
    print(f"microphones: {len(scene.microphone_positions_m)}")
    print(f"source states: {len(scene.source_positions_m)}")
    print(f"source time range: {scene.source_times_s[0]:.6f} .. {scene.source_times_s[-1]:.6f} s")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    try:
        estimate = validate_ground_truth_json(args.estimate)
        reference = validate_ground_truth_json(args.reference)
        output = args.output
        if output is None:
            output = args.estimate.with_suffix("").with_name(
                f"{args.estimate.stem}_vs_{args.reference.stem}"
            )
        paths = write_scene_comparison_outputs(
            estimate,
            reference,
            output,
            estimate_path=args.estimate,
            reference_path=args.reference,
        )
    except (OSError, ValueError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 2

    print(f"JSON: {paths.json}")
    print(f"figure: {paths.figure}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
