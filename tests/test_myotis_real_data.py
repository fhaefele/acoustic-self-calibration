"""End-to-end calibration on the committed real Myotis recording."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_self_calibration.ground_truth import GroundTruth, load_ground_truth_json
from acoustic_self_calibration.myotis import run_constrained_real_myotis_evaluation
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "myotis"
_AUDIO = _FIXTURE_DIR / "myotis.wav"
_REFERENCE = _FIXTURE_DIR / "myotis_gt.json"

pytestmark = pytest.mark.skipif(
    not (_AUDIO.exists() and _REFERENCE.exists()),
    reason="real Myotis fixture is not checked out",
)


def _half_space_sign(reference: GroundTruth) -> int:
    center = np.mean(reference.microphone_positions_m, axis=0)
    _, _, vt = np.linalg.svd(reference.microphone_positions_m - center, full_matrices=False)
    normal = vt[2]
    along = (reference.source_positions_m - center) @ normal
    return 1 if float(np.mean(along)) >= 0.0 else -1


def test_real_myotis_cross_recovers_geometry_with_explicit_constraint() -> None:
    reference = load_ground_truth_json(_REFERENCE)
    constraint = PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=11,
        angle_rad=np.pi / 2.0,
        provenance="real Myotis cross frame: junction receiver 3, orthogonal arms to 0 and 11",
    )
    report = run_constrained_real_myotis_evaluation(
        angle_constraint=constraint,
        audio_path=_AUDIO,
        reference_path=_REFERENCE,
        source_half_space_sign=_half_space_sign(reference),
    )

    assert report["mode"] == "blind_and_constrained"
    assert report["reference_used_for_solver"] is False

    blind = report["blind"]["run"]
    assert blind["detected_event_count"] >= 40
    # The blind 3-D model cannot complete on the planar cross: never a false solved.
    assert blind["status"] in {"ambiguous", "degenerate", "weakly_identified", "failed"}

    constrained = report["constrained"]
    assert constrained["run"]["status"] in {"solved", "weakly_identified"}
    evaluation = constrained["evaluation"]
    assert evaluation["status"] == "evaluated"
    assert evaluation["microphones"]["rms_error_m"] < 0.20
    tdoa_rms_s = constrained["run"]["rms_tdoa_residual_s"]
    assert tdoa_rms_s is not None
    assert tdoa_rms_s < 45e-6
    # Held-out receiver coordinates of the withheld events, not the fitted ones.
    validation = constrained["run"]["validation"]
    assert validation is not None
    assert validation["rms_s"] < 45e-6

    source = evaluation["source_observables"]
    assert source["signed_source_rms_m"] is not None
    assert source["signed_source_rms_m"] < 0.30
    assert source["projected_rms_error_m"] < 0.30

    ablation = report["constraint_ablation"]
    assert ablation["ambiguity_restored"] is True
    assert ablation["without_constraint_status"] in {
        "ambiguous",
        "degenerate",
        "weakly_identified",
        "failed",
    }
    json.dumps(report, allow_nan=False)
