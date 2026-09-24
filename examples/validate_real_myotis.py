"""Run blind real-Myotis calibration with reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from acoustic_self_calibration.myotis import (
    OUTPUT_ENV,
    run_real_myotis_calibration,
    write_real_myotis_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production stratified audio pipeline blindly, then evaluate "
            "against the supplied reference only after calibration."
        )
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="multichannel WAV; defaults to $ASC_MYOTIS_AUDIO",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="reference scene JSON; defaults to $ASC_MYOTIS_REFERENCE",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="report JSON; defaults to $ASC_MYOTIS_OUTPUT",
    )
    parser.add_argument("--speed-of-sound", type=float, default=343.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--channel-mapping",
        type=str,
        help="comma-separated reference microphone indices in audio-channel order",
    )
    parser.add_argument("--event-min-gap-ms", type=float, default=50.0)
    parser.add_argument("--max-tau-ms", type=float, default=12.0)
    parser.add_argument("--tdoa-template-ms", type=float, default=1.8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    mapping = None
    if args.channel_mapping:
        mapping = tuple(int(value.strip()) for value in args.channel_mapping.split(","))

    report = run_real_myotis_calibration(
        audio_path=args.audio,
        reference_path=args.reference,
        speed_of_sound_mps=args.speed_of_sound,
        channel_mapping=mapping,
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
