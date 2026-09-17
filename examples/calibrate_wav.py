from __future__ import annotations

import argparse
from pathlib import Path

from acoustic_self_calibration import (
    calibrate_wav,
    load_ground_truth_json,
    write_calibration_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a multichannel WAV recording")
    parser.add_argument("wav", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("calibration"))
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()

    result = calibrate_wav(
        args.wav,
        frame_size=1024,
        hop_size=4096,
        pair_mode="redundant",
        reference_count=2,
        likelihood="cauchy",
    )
    ground_truth = None if args.ground_truth is None else load_ground_truth_json(args.ground_truth)
    paths = write_calibration_outputs(
        result,
        args.output,
        input_wav_path=args.wav,
        ground_truth=ground_truth,
    )

    calibration = result.calibration
    print("microphone positions [m]:")
    print(calibration.microphone_positions)
    print("source positions [m]:")
    print(calibration.source_positions)
    print(f"saved JSON: {paths.json}")
    print(f"saved figure: {paths.figure}")


if __name__ == "__main__":
    main()
