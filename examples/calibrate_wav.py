from __future__ import annotations

import argparse
from pathlib import Path

from acoustic_self_calibration import calibrate_wav, export_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a multichannel WAV recording")
    parser.add_argument("wav", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("calibration"))
    args = parser.parse_args()

    result = calibrate_wav(
        args.wav,
        frame_size=1024,
        hop_size=4096,
        pair_mode="redundant",
        reference_count=2,
        likelihood="cauchy",
    )
    paths = export_calibration(result, args.output)

    calibration = result.calibration
    print("microphone positions [m]:")
    print(calibration.microphone_positions)
    print("microphone position std [m]:")
    print(calibration.microphone_position_std_m)
    print("source positions [m]:")
    print(calibration.source_positions)
    print("source position std [m]:")
    print(calibration.source_position_std_m)
    print(f"trajectory times [s]: {result.frame_times_s}")
    print(f"saved: {paths}")


if __name__ == "__main__":
    main()
