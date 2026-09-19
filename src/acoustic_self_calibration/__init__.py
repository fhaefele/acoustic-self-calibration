"""Stratified TDOA acoustic self-calibration from discrete broadband events."""

from .evaluation import GroundTruthEvaluation, evaluate_against_ground_truth, evaluate_scenes
from .events import (
    EventDetection,
    arrival_sigma_from_confidence,
    detect_transient_events,
    estimate_event_tdoa_measurements,
)
from .export import (
    CalibrationOutputPaths,
    calibration_result_to_dict,
    write_calibration_outputs,
    write_scene_comparison_outputs,
)
from .geometry import CoordinateGauge, canonicalize_scene_conditioned, select_coordinate_gauge
from .ground_truth import (
    GroundTruth,
    ground_truth_from_dict,
    ground_truth_to_dict,
    load_ground_truth_json,
    make_ground_truth_dict,
    validate_ground_truth_json,
    write_ground_truth_json,
)
from .measurements import EventTDOAMeasurements, reference_star_from_arrivals
from .myotis import (
    analyze_myotis_reference_geometry,
    evaluate_planar_myotis_observables,
    prepare_real_myotis_validation,
    run_constrained_real_myotis_evaluation,
    run_real_myotis_calibration,
    write_real_myotis_manifest,
)
from .pipeline import (
    AudioCalibrationResult,
    AudioModel,
    RefinementDiagnostics,
    RefinementMode,
    calibrate_audio,
)
from .simulation import random_microphone_array, render_moving_source
from .stratified import (
    ModelComparisonResult,
    PlanarAngleConstraint,
    PlanarCalibrationResult,
    StratifiedCalibrationDiagnostics,
    StratifiedCalibrationResult,
    calibrate_planar_tdoa,
    calibrate_tdoa,
    compare_tdoa_models_8mic,
)
from .visualization import plot_calibration_comparison, plot_scene_comparison
from .wav import calibrate_wav, read_multichannel_wav

__all__ = [
    "AudioCalibrationResult",
    "AudioModel",
    "compare_tdoa_models_8mic",
    "calibrate_tdoa",
    "calibrate_planar_tdoa",
    "StratifiedCalibrationResult",
    "StratifiedCalibrationDiagnostics",
    "RefinementDiagnostics",
    "RefinementMode",
    "PlanarCalibrationResult",
    "PlanarAngleConstraint",
    "ModelComparisonResult",
    "CalibrationOutputPaths",
    "CoordinateGauge",
    "GroundTruth",
    "GroundTruthEvaluation",
    "EventDetection",
    "EventTDOAMeasurements",
    "analyze_myotis_reference_geometry",
    "evaluate_planar_myotis_observables",
    "run_constrained_real_myotis_evaluation",
    "run_real_myotis_calibration",
    "arrival_sigma_from_confidence",
    "calibrate_audio",
    "calibrate_wav",
    "canonicalize_scene_conditioned",
    "calibration_result_to_dict",
    "detect_transient_events",
    "estimate_event_tdoa_measurements",
    "evaluate_against_ground_truth",
    "evaluate_scenes",
    "ground_truth_from_dict",
    "ground_truth_to_dict",
    "load_ground_truth_json",
    "make_ground_truth_dict",
    "plot_calibration_comparison",
    "plot_scene_comparison",
    "prepare_real_myotis_validation",
    "random_microphone_array",
    "reference_star_from_arrivals",
    "read_multichannel_wav",
    "render_moving_source",
    "select_coordinate_gauge",
    "validate_ground_truth_json",
    "write_calibration_outputs",
    "write_ground_truth_json",
    "write_real_myotis_manifest",
    "write_scene_comparison_outputs",
]
