"""Run explicitly constrained real-Myotis evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from acoustic_self_calibration.myotis import (
    OUTPUT_ENV,
    run_constrained_real_myotis_evaluation,
    write_real_myotis_manifest,
)
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run blind calibration and an explicitly constrained planar rerun on the "
            "same extracted TDOA measurements."
        )
    )
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--speed-of-sound", type=float, default=343.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--channel-mapping",
        type=str,
        help="comma-separated reference microphone indices in audio-channel order",
    )
    parser.add_argument(
        "--right-angle",
        required=True,
        metavar="CENTER,ARM_A,ARM_B",
        help="explicit receiver IDs defining the known 90-degree arm angle",
    )
    parser.add_argument(
        "--constraint-provenance",
        required=True,
        help="human-readable provenance for the supplied physical constraint",
    )
    parser.add_argument(
        "--source-half-space-sign",
        type=int,
        choices=(-1, 1),
        help=(
            "optional sign along the reported reference plane normal; omit to keep "
            "signed source RMS undefined"
        ),
    )
    parser.add_argument("--event-min-gap-ms", type=float, default=50.0)
    parser.add_argument("--max-tau-ms", type=float, default=12.0)
    parser.add_argument("--tdoa-template-ms", type=float, default=1.8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    center, arm_a, arm_b = (int(value.strip()) for value in args.right_angle.split(","))
    mapping = None
    if args.channel_mapping:
        mapping = tuple(int(value.strip()) for value in args.channel_mapping.split(","))

    constraint = PlanarAngleConstraint(
        center_receiver=center,
        arm_a_receiver=arm_a,
        arm_b_receiver=arm_b,
        angle_rad=np.pi / 2.0,
        provenance=args.constraint_provenance,
    )
    report = run_constrained_real_myotis_evaluation(
        angle_constraint=constraint,
        audio_path=args.audio,
        reference_path=args.reference,
        speed_of_sound_mps=args.speed_of_sound,
        channel_mapping=mapping,
        source_half_space_sign=args.source_half_space_sign,
        solver_seed=args.seed,
        event_min_gap_s=args.event_min_gap_ms / 1000.0,
        max_tau_s=args.max_tau_ms / 1000.0,
        tdoa_template_s=args.tdoa_template_ms / 1000.0,
    )
    output = args.output
    if output is None and os.environ.get(OUTPUT_ENV):
        output = Path(os.environ[OUTPUT_ENV])
    if output is not None:
        write_real_myotis_manifest(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
