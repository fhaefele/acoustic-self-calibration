from pathlib import Path

import numpy as np

from acoustic_self_calibration.array_configuration import ArrayConfiguration
from acoustic_self_calibration.myotis_public import (
    calibrate_myotis_recording,
    evaluate_myotis_recording,
)
from acoustic_self_calibration.stratified.solver import calibrate_planar_tdoa


def test_committed_myotis_public_run_freezes_before_reference(monkeypatch):
    fixture = Path(__file__).resolve().parents[1] / "data/myotis"
    # Missing dedicated fixtures fail this gate; they are never substituted.
    assert (fixture / "myotis.wav").is_file()
    assert (fixture / "myotis_gt.json").is_file()
    config = ArrayConfiguration(
        "cross",
        provenance="fixture declaration; independent evidence unverified",
        arms=((0, 1, 2, 3, 4, 5, 6), (3, 7, 8, 9, 10, 11)),
        rays=((3, 0), (3, 11)),
        angle_deg=90,
    )
    import acoustic_self_calibration.myotis_public as module

    loader = module.load_ground_truth_json

    def forbidden(*args, **kwargs):
        raise AssertionError("calibration cannot read a reference")

    monkeypatch.setattr(module, "load_ground_truth_json", forbidden)
    result = calibrate_myotis_recording(fixture / "myotis.wav", config)
    assert result.microphone_positions_m is not None and len(result.microphone_positions_m) == 12
    assert result.source_positions_m is not None
    frozen_mics = result.microphone_positions_m.copy()
    frozen_sources = result.source_positions_m.copy()
    monkeypatch.setattr(module, "load_ground_truth_json", loader)
    evaluation = evaluate_myotis_recording(result, fixture / "myotis_gt.json")
    assert evaluation["microphone_rms_error_m"] < 0.2
    assert evaluation["source_rms_error_m"] < 0.3
    assert evaluation["localized_events"] >= 40
    assert evaluation["heldout_tdoa_rms_s"] < 45e-6
    assert evaluation["fitted_tdoa_rms_s"] < 45e-6
    assert evaluation["success"] == (result.status == "solved")
    np.testing.assert_array_equal(result.microphone_positions_m, frozen_mics)
    np.testing.assert_array_equal(result.source_positions_m, frozen_sources)
    ablation = calibrate_planar_tdoa(result.measurements, receiver_subset_budget=5)
    assert ablation.status in {"ambiguous", "degenerate", "weakly_identified", "failed"}
