from pathlib import Path

import numpy as np
from scipy.io import wavfile

from acoustic_self_calibration.ground_truth import GroundTruth, write_ground_truth_json
from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.myotis import (
    evaluate_planar_myotis_observables,
    run_constrained_real_myotis_evaluation,
)
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    make_myotis_cross_pulse_scene,
    myotis_cross_microphones,
    myotis_cross_source_positions,
)
from acoustic_self_calibration.stratified.constraints import PlanarAngleConstraint
from acoustic_self_calibration.stratified.solver import calibrate_planar_tdoa


def _constraint() -> PlanarAngleConstraint:
    return PlanarAngleConstraint(
        center_receiver=3,
        arm_a_receiver=0,
        arm_b_receiver=7,
        angle_rad=np.pi / 2.0,
        provenance="explicit_test_cross_geometry",
    )


def _reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, GroundTruth]:
    microphones = myotis_cross_microphones()
    times = np.linspace(0.25, 2.05, 18)
    sources = myotis_cross_source_positions(times)
    reference = GroundTruth(
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=sources,
        metadata={"fixture": "exact_cross"},
    )
    return microphones, times, sources, reference


def test_planar_observable_evaluation_does_not_invent_height_sign() -> None:
    microphones, times, sources, reference = _reference()
    tdoa = exact_reference_tdoas(microphones, sources)
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(len(times)),
        receiver_event_times_s=times,
        microphone_ids=tuple(range(12)),
        microphone_pairs=tuple((0, index) for index in range(1, 12)),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 1e-7),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    result = calibrate_planar_tdoa(
        measurements,
        receiver_subset_budget=3,
        event_subset_budget=2,
        angle_constraint=_constraint(),
        metric_acceptance_rms_m=1e-7,
        extra_microphone_rms_m=1e-7,
    )
    assert result.status == "solved"

    unsigned = evaluate_planar_myotis_observables(
        result,
        times,
        reference,
    )
    assert unsigned["microphones"]["rms_error_m"] < 1e-6
    assert unsigned["source_observables"]["projected_rms_error_m"] < 1e-6
    assert unsigned["source_observables"]["unsigned_height_rms_error_m"] < 1e-6
    assert unsigned["source_observables"]["signed_source_rms_m"] is None
    assert unsigned["source_observables"]["height_sign_known_count"] == 0

    center = np.mean(microphones, axis=0)
    _, _, vt = np.linalg.svd(microphones - center, full_matrices=False)
    normal = vt[2]
    normal_coordinate = (sources - center) @ normal
    sign = 1 if float(np.mean(normal_coordinate)) >= 0.0 else -1
    signed = evaluate_planar_myotis_observables(
        result,
        times,
        reference,
        source_half_space_sign=sign,
    )
    assert signed["source_observables"]["signed_source_rms_m"] is not None
    assert signed["source_observables"]["signed_source_rms_m"] < 1e-6
    assert signed["source_observables"]["source_half_space_sign"] == sign


def _write_cross_fixture(tmp_path: Path) -> tuple[Path, Path, int]:
    scene = make_myotis_cross_pulse_scene()
    audio_path = tmp_path / "myotis_cross.wav"
    scaled = scene.audio / max(float(np.max(np.abs(scene.audio))), 1e-12)
    wavfile.write(
        audio_path,
        scene.sample_rate_hz,
        np.round(0.95 * scaled * 32767.0).astype(np.int16),
    )
    reference_path = tmp_path / "myotis_cross_reference.json"
    write_ground_truth_json(
        reference_path,
        microphone_positions_m=scene.microphone_positions_m,
        source_times_s=scene.trajectory_times_s,
        source_positions_m=scene.trajectory_positions_m,
        metadata={"fixture": "audio_cross"},
    )
    center = np.mean(scene.microphone_positions_m, axis=0)
    _, _, vt = np.linalg.svd(
        scene.microphone_positions_m - center,
        full_matrices=False,
    )
    source_normal = (scene.trajectory_positions_m - center) @ vt[2]
    sign = 1 if float(np.mean(source_normal)) >= 0.0 else -1
    return audio_path, reference_path, sign


def test_constrained_runner_separates_blind_constraint_and_ablation(
    tmp_path: Path,
) -> None:
    audio_path, reference_path, sign = _write_cross_fixture(tmp_path)
    report = run_constrained_real_myotis_evaluation(
        angle_constraint=_constraint(),
        audio_path=audio_path,
        reference_path=reference_path,
        source_half_space_sign=sign,
        event_min_gap_s=0.05,
        max_tau_s=0.012,
        tdoa_template_s=0.0018,
        receiver_subset_budget=3,
        event_subset_budget=2,
        root_start_count=24,
        metric_start_count=12,
        planar_membership_tolerance=0.03,
        planar_metric_acceptance_rms_m=0.05,
        planar_extra_microphone_rms_m=0.08,
    )

    assert report["mode"] == "blind_and_constrained"
    assert report["reference_used_for_solver"] is False
    assert report["constraint"]["type"] == "planar_arm_angle"
    assert report["constraint"]["provenance"] == "explicit_test_cross_geometry"
    assert report["blind"]["run"]["status"] in {
        "solved",
        "ambiguous",
        "weakly_identified",
        "failed",
    }
    assert report["constrained"]["run"]["status"] in {
        "solved",
        "weakly_identified",
    }
    assert report["constrained"]["evaluation"]["status"] == "evaluated"
    assert report["constrained"]["evaluation"]["microphones"]["rms_error_m"] < 0.20
    signed_rms = report["constrained"]["evaluation"]["source_observables"]["signed_source_rms_m"]
    assert signed_rms is not None
    assert signed_rms < 0.30
    assert report["constraint_ablation"]["ambiguity_restored"] is True
    assert report["constraint_ablation"]["without_constraint_status"] in {
        "ambiguous",
        "degenerate",
        "weakly_identified",
        "failed",
    }
    assert len(report["run_configuration_sha256"]) == 64
