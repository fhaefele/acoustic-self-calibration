from __future__ import annotations

import argparse
from pathlib import Path

from acoustic_self_calibration import (
    calibrate_wav,
    load_ground_truth_json,
    write_calibration_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a multichannel WAV with the stratified TDOA solver"
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("calibration"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--model",
        choices=("general_3d", "receiver2d_source3d"),
        default="general_3d",
    )
    parser.add_argument("--event-min-gap-ms", type=float, default=50.0)
    parser.add_argument("--max-tau-ms", type=float, default=20.0)
    args = parser.parse_args()

    result = calibrate_wav(
        args.wav,
        event_min_gap_s=args.event_min_gap_ms / 1000.0,
        max_tau_s=args.max_tau_ms / 1000.0,
        model=args.model,
    )
    if result.microphone_positions_m is None or result.source_positions_m is None:
        raise SystemExit(f"calibration did not produce geometry: {result.status}")

    reference = None if args.reference is None else load_ground_truth_json(args.reference)
    paths = write_calibration_outputs(
        result,
        args.output,
        input_wav_path=args.wav,
        ground_truth=reference,
    )

    print(f"status: {result.status}")
    print("microphone positions [m]:")
    print(result.microphone_positions_m)
    print("source representative positions [m]:")
    print(result.source_positions_m)
    if result.source_unsigned_heights_m is not None:
        print("source unsigned plane-normal heights [m]:")
        print(result.source_unsigned_heights_m)
    print(f"saved JSON: {paths.json}")
    print(f"saved figure: {paths.figure}")


if __name__ == "__main__":
    main()
