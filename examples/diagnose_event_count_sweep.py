"""Print static bookkeeping for the stratified event-count diagnostic sweep."""

from __future__ import annotations

import json

from acoustic_self_calibration.simulation import event_count_sweep_cases


def main() -> None:
    rows = []
    for microphone_count in (8, 12, 16, 24):
        for case in event_count_sweep_cases(microphone_count):
            rows.append(
                {
                    "microphone_count": case.microphone_count,
                    "event_count": case.event_count,
                    "seed": case.seed,
                    "independent_measurement_count": case.independent_measurement_count,
                    "geometry_unknown_count": case.geometry_unknown_count,
                    "measurement_excess": case.measurement_excess,
                    "geometry_error_m": None,
                    "tdoa_error_s": None,
                    "conditioning": None,
                    "hypothesis_survival": None,
                    "runtime_s": None,
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
