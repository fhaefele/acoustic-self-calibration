from __future__ import annotations

import argparse
from pathlib import Path

from acoustic_self_calibration import calibrate_wav, write_calibration_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("calibration"))
    parser.add_argument("--event-channel", type=int)
    args = parser.parse_args()

    result = calibrate_wav(
        args.wav,
        event_channel=args.event_channel,
        pair_mode="reference",
    )
    paths = write_calibration_outputs(result, args.output, input_wav_path=args.wav)

    print(f"success: {result.calibration.success}")
    print(
        f"events: {len(result.event_times_s)} used / {result.detected_event_count} detected "
        f"on channel {result.event_channel}"
    )
    print(f"JSON: {paths.json}")
    print(f"figure: {paths.figure}")


if __name__ == "__main__":
    main()
