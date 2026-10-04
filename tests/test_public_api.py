from __future__ import annotations

import importlib.util
import inspect
from typing import Any, cast, get_args

import numpy as np
import pytest

import acoustic_self_calibration as asc
from acoustic_self_calibration.pipeline import RefinementMode, calibrate_audio
from acoustic_self_calibration.stratified.solver import CalibrationStatus


def test_removed_bayesian_backend_is_not_public_or_importable() -> None:
    assert not hasattr(asc, "calibrate_bayesian")
    assert not hasattr(asc, "BayesianCalibrationResult")
    assert not hasattr(asc, "DistancePrior")
    assert not hasattr(asc, "tdoa_sigma_from_confidence")
    assert importlib.util.find_spec("acoustic_self_calibration.bayesian") is None
    assert importlib.util.find_spec("acoustic_self_calibration.initialization") is None


def test_public_api_exports_stratified_geometry_types() -> None:
    assert hasattr(asc, "StratifiedCalibrationResult")
    assert hasattr(asc, "StratifiedCalibrationDiagnostics")
    assert hasattr(asc, "PlanarCalibrationResult")
    assert hasattr(asc, "PlanarAngleConstraint")
    assert hasattr(asc, "ModelComparisonResult")
    assert hasattr(asc, "calibrate_tdoa")
    assert hasattr(asc, "calibrate_planar_tdoa")
    assert hasattr(asc, "compare_tdoa_models_8mic")


def test_audio_api_exposes_model_and_explicit_refinement_mode() -> None:
    parameters = inspect.signature(calibrate_audio).parameters
    assert "model" in parameters
    assert "refinement" in parameters
    assert "receiver_subset_budget" in parameters
    assert "event_subset_budget" in parameters
    assert "angle_constraint" in parameters
    assert "refinement_max_nfev" in parameters
    assert "refinement_improvement_tolerance" in parameters
    assert set(get_args(RefinementMode)) == {"none", "wls", "huber"}

    audio = np.zeros((128, 8), dtype=float)
    with pytest.raises(ValueError, match="model must be"):
        calibrate_audio(audio, 48_000, model=cast(Any, "bayesian"))
    with pytest.raises(ValueError, match="refinement must be"):
        calibrate_audio(audio, 48_000, refinement=cast(Any, "map"))
    with pytest.raises(ValueError, match="refinement_max_nfev"):
        calibrate_audio(audio, 48_000, refinement_max_nfev=0)
    with pytest.raises(ValueError, match="refinement_improvement_tolerance"):
        calibrate_audio(audio, 48_000, refinement_improvement_tolerance=-1.0)


def test_low_level_frame_pair_api_is_not_top_level_release_surface() -> None:
    assert not hasattr(asc, "estimate_pairwise_tdoa_matrix")
    assert not hasattr(asc, "make_microphone_pairs")
    assert not hasattr(asc, "gcc_phat")


def test_public_calibration_status_includes_degenerate() -> None:
    assert "degenerate" in get_args(CalibrationStatus)
