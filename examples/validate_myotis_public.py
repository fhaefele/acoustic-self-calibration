"""Run the committed fixture without a reference crossing the solver boundary."""

import argparse
import json
from pathlib import Path

from acoustic_self_calibration.array_configuration import ArrayConfiguration
from acoustic_self_calibration.myotis_public import public_myotis_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=Path("data/myotis/myotis.wav"))
    parser.add_argument("--reference", type=Path, default=Path("data/myotis/myotis_gt.json"))
    parser.add_argument(
        "--array-config", type=Path, default=Path("examples/array_configs/myotis_cross.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/myotis_public.json")
    )
    args = parser.parse_args()
    config = ArrayConfiguration.from_dict(json.loads(args.array_config.read_text()))
    report = public_myotis_report(args.audio, args.reference, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["evaluation"], indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
