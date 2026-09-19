from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .export import write_calibration_outputs, write_scene_comparison_outputs
from .ground_truth import validate_ground_truth_json
from .stratified.constraints import PlanarAngleConstraint
from .wav import calibrate_wav


def _right_angle_receivers(value: str) -> tuple[int, int, int]:
    try:
        center, arm_a, arm_b = (int(item.strip()) for item in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("right-angle must be CENTER,ARM_A,ARM_B") from error
    if len({center, arm_a, arm_b}) != 3 or min(center, arm_a, arm_b) < 0:
        raise argparse.ArgumentTypeError(
            "right-angle receiver IDs must be distinct and non-negative"
        )
    return center, arm_a, arm_b


def _add_solver_options(parser: argparse.ArgumentParser) -> None:
    audio = parser.add_argument_group("event detection and TDOA")
    audio.add_argument("--event-channel", type=int)
    audio.add_argument("--event-smooth-ms", type=float, default=0.3)
    audio.add_argument("--event-min-gap-ms", type=float, default=3.0)
    audio.add_argument("--event-prominence", type=float, default=0.003)
    audio.add_argument("--max-tau-ms", type=float, default=10.0)
    audio.add_argument("--tdoa-template-ms", type=float, default=1.8)
    audio.add_argument("--tdoa-candidates", type=int, default=8)
    audio.add_argument("--max-tdoa-rate", type=float, default=0.05)
    audio.add_argument("--tdoa-track-weight", type=float, default=0.4)
    audio.add_argument("--no-temporal-tracking", action="store_true")

    solver = parser.add_argument_group("stratified solver")
    solver.add_argument(
        "--model",
        choices=("general-3d", "receiver2d-source3d"),
        default="general-3d",
    )
    solver.add_argument("--speed-of-sound", type=float, default=343.0)
    solver.add_argument("--best-sigma-samples", type=float, default=0.35)
    solver.add_argument("--worst-sigma-samples", type=float, default=4.0)
    solver.add_argument("--receiver-subset-budget", type=int, default=3)
    solver.add_argument("--event-subset-budget", type=int, default=2)
    solver.add_argument("--root-start-count", type=int, default=32)
    solver.add_argument("--metric-start-count", type=int, default=20)
    solver.add_argument("--extra-microphone-rms-m", type=float, default=0.05)
    solver.add_argument("--planar-membership-tolerance", type=float, default=5e-3)
    solver.add_argument("--planar-metric-rms-m", type=float, default=5e-3)
    solver.add_argument("--planar-extra-microphone-rms-m", type=float, default=0.02)
    solver.add_argument(
        "--right-angle",
        type=_right_angle_receivers,
        metavar="CENTER,ARM_A,ARM_B",
        help="explicit known 90-degree planar arm constraint",
    )
    solver.add_argument(
        "--constraint-provenance",
        help="required provenance text when --right-angle is supplied",
    )
    solver.add_argument(
        "--refinement",
        choices=("none", "wls", "huber"),
        default="none",
        help="optional post-selection measurement-only nonlinear refinement",
    )
    solver.add_argument("--refinement-max-nfev", type=int, default=200)
    solver.add_argument(
        "--refinement-improvement-tolerance",
        type=float,
        default=1e-10,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asc",
        description="Stratified TDOA acoustic self-calibration and scene comparison.",
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
        description=(
            "Detect transient events and calibrate a synchronized microphone array "
            "with the stratified TDOA backend."
        ),
    )
    calibrate.add_argument("wav", type=Path, help="input multichannel WAV file")
    calibrate.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output prefix for RESULT.json and RESULT.png",
    )
    calibrate.add_argument(
        "-r",
        "--reference",
        type=Path,
        help="canonical scene JSON used only for post-calibration comparison",
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
        "event_channel": args.event_channel,
        "event_smooth_s": args.event_smooth_ms / 1000.0,
        "event_min_gap_s": args.event_min_gap_ms / 1000.0,
        "event_relative_prominence": args.event_prominence,
        "max_tau_s": args.max_tau_ms / 1000.0,
        "tdoa_template_s": args.tdoa_template_ms / 1000.0,
        "tdoa_candidate_count": args.tdoa_candidates,
        "max_tdoa_rate": args.max_tdoa_rate,
        "tdoa_track_weight": args.tdoa_track_weight,
        "use_temporal_tracking": not args.no_temporal_tracking,
        "model": args.model,
        "speed_of_sound_mps": args.speed_of_sound,
        "best_sigma_samples": args.best_sigma_samples,
        "worst_sigma_samples": args.worst_sigma_samples,
        "receiver_subset_budget": args.receiver_subset_budget,
        "event_subset_budget": args.event_subset_budget,
        "root_start_count": args.root_start_count,
        "metric_start_count": args.metric_start_count,
        "extra_microphone_inlier_rms_m": args.extra_microphone_rms_m,
        "planar_membership_tolerance": args.planar_membership_tolerance,
        "planar_metric_acceptance_rms_m": args.planar_metric_rms_m,
        "planar_extra_microphone_rms_m": args.planar_extra_microphone_rms_m,
        "right_angle": (None if args.right_angle is None else list(args.right_angle)),
        "constraint_provenance": args.constraint_provenance,
        "refinement": args.refinement,
        "refinement_max_nfev": args.refinement_max_nfev,
        "refinement_improvement_tolerance": (args.refinement_improvement_tolerance),
    }


def _run_calibrate(args: argparse.Namespace) -> int:
    output = args.output
    if output is None:
        output = args.wav.with_suffix("").with_name(f"{args.wav.stem}_calibration")

    reference = None if args.reference is None else validate_ground_truth_json(args.reference)
    model = "general_3d" if args.model == "general-3d" else "receiver2d_source3d"
    angle_constraint = None
    if args.right_angle is not None:
        if model != "receiver2d_source3d":
            print(
                "--right-angle requires --model receiver2d-source3d",
                file=sys.stderr,
            )
            return 2
        if not args.constraint_provenance:
            print(
                "--constraint-provenance is required with --right-angle",
                file=sys.stderr,
            )
            return 2
        center, arm_a, arm_b = args.right_angle
        angle_constraint = PlanarAngleConstraint(
            center_receiver=center,
            arm_a_receiver=arm_a,
            arm_b_receiver=arm_b,
            angle_rad=np.pi / 2.0,
            provenance=args.constraint_provenance,
        )

    result = calibrate_wav(
        args.wav,
        event_channel=args.event_channel,
        event_smooth_s=args.event_smooth_ms / 1000.0,
        event_min_gap_s=args.event_min_gap_ms / 1000.0,
        event_relative_prominence=args.event_prominence,
        max_tau_s=args.max_tau_ms / 1000.0,
        tdoa_template_s=args.tdoa_template_ms / 1000.0,
        tdoa_candidate_count=args.tdoa_candidates,
        max_tdoa_rate=args.max_tdoa_rate,
        tdoa_track_weight=args.tdoa_track_weight,
        use_temporal_tracking=not args.no_temporal_tracking,
        speed_of_sound=args.speed_of_sound,
        best_sigma_samples=args.best_sigma_samples,
        worst_sigma_samples=args.worst_sigma_samples,
        receiver_subset_budget=args.receiver_subset_budget,
        event_subset_budget=args.event_subset_budget,
        root_start_count=args.root_start_count,
        metric_start_count=args.metric_start_count,
        extra_microphone_inlier_rms_m=args.extra_microphone_rms_m,
        planar_membership_tolerance=args.planar_membership_tolerance,
        planar_metric_acceptance_rms_m=args.planar_metric_rms_m,
        planar_extra_microphone_rms_m=args.planar_extra_microphone_rms_m,
        angle_constraint=angle_constraint,
        model=model,
        refinement=args.refinement,
        refinement_max_nfev=args.refinement_max_nfev,
        refinement_improvement_tolerance=(args.refinement_improvement_tolerance),
    )
    if result.microphone_positions_m is None or result.source_positions_m is None:
        print(f"calibration status: {result.status}", file=sys.stderr)
        return 2

    paths = write_calibration_outputs(
        result,
        output,
        input_wav_path=args.wav,
        settings=_settings_dict(args),
        ground_truth=reference,
    )
    print(f"status: {result.status}")
    print(f"microphones: {len(result.microphone_positions_m)}")
    print(f"source states: {len(result.source_positions_m)}")
    if result.rms_tdoa_residual_s is not None:
        print(f"TDOA RMS residual: {1e6 * result.rms_tdoa_residual_s:.3f} us")
    print(f"JSON: {paths.json}")
    print(f"figure: {paths.figure}")
    return 0 if result.status == "solved" else 2


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
