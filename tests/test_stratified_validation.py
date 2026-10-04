import numpy as np
import pytest

from acoustic_self_calibration.stratified.hypotheses import (
    FullGeometryHypothesis,
    MeasurementSplit,
    cluster_equivalent_hypotheses,
    make_holdout_split,
)
from acoustic_self_calibration.stratified.robustness import (
    RobustResidualSummary,
    conditional_covariance,
    summarize_independent_residuals,
    whiten_residual_block,
)


def test_holdout_split_is_disjoint_and_keeps_validation_primitive_data_withheld() -> None:
    split = make_holdout_split(
        receiver_count=8,
        event_count=20,
        seed_receivers=(0, 1, 2, 3, 4, 5, 6),
        seed_events=(0, 2, 4, 6, 8, 10),
        fitting_events=tuple(range(12)),
        validation_events=tuple(range(12, 20)),
        validation_localization_receivers=(0, 1, 2, 3, 4),
    )
    assert not np.any(split.generation_mask & split.completion_mask)
    assert not np.any(split.generation_mask & split.validation_mask)
    assert not np.any(split.completion_mask & split.validation_mask)
    assert np.all(split.validation_mask[5:, 12:20])
    assert not np.any(split.validation_mask[1:5, 12:20])


def test_split_rejects_leakage() -> None:
    generation = np.zeros((4, 5), dtype=bool)
    completion = np.zeros_like(generation)
    validation = np.zeros_like(generation)
    generation[1, 2] = True
    validation[1, 2] = True
    with pytest.raises(ValueError, match="overlap"):
        MeasurementSplit(generation, completion, validation)


def test_huber_summary_is_less_sensitive_to_one_gross_outlier_than_squared_score() -> None:
    residual = np.array([1e-6, -1e-6, 0.5e-6, 80e-6])
    sigma = np.full(4, 2e-6)
    valid = np.ones(4, dtype=bool)
    summary = summarize_independent_residuals(residual, sigma, valid)
    normalized = residual / sigma
    squared = float(np.mean(normalized * normalized))
    assert summary.normalized_huber_score < squared
    assert summary.inlier_fraction == 0.75


def test_covariance_whitening_respects_shared_reference_correlation() -> None:
    residual = np.array([2e-6, -1e-6])
    covariance = np.array([[8e-12, 4e-12], [4e-12, 8e-12]])
    whitened = whiten_residual_block(residual, covariance)
    assert whitened.shape == (2,)
    assert np.all(np.isfinite(whitened))


def _hypothesis(subset: str, shift: float) -> FullGeometryHypothesis:
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.1, 1.0, 0.0],
            [0.2, 0.1, 1.0],
        ]
    )
    microphones = microphones + np.array([shift, -2.0 * shift, 0.5 * shift])
    summary = RobustResidualSummary(4, 1e-6, 1e-6, 1e-6, 0.2, 1.0)
    split = MeasurementSplit(
        np.zeros((4, 2), dtype=bool),
        np.zeros((4, 2), dtype=bool),
        np.zeros((4, 2), dtype=bool),
    )
    return FullGeometryHypothesis(
        subset_id=subset,
        root_id=0,
        microphone_positions_m=microphones,
        source_positions_m=np.zeros((2, 3)),
        split=split,
        validation=summary,
        full_tdoa_rms_s=1e-6,
        metric_rms_m=1e-8,
    )


def test_consensus_counts_distinct_subsets_not_duplicate_roots() -> None:
    first = _hypothesis("subset-a", 0.0)
    duplicate = _hypothesis("subset-a", 1.0)
    independent = _hypothesis("subset-b", -0.7)
    classes = cluster_equivalent_hypotheses(
        (first, duplicate, independent),
        microphone_rms_tolerance_m=1e-8,
    )
    assert len(classes) == 1
    assert len(classes[0].members) == 3
    assert classes[0].independent_subset_support == 2


def test_conditional_covariance_removes_shared_fitted_information() -> None:
    covariance = np.array(
        [
            [8e-12, 4e-12, 4e-12],
            [4e-12, 8e-12, 4e-12],
            [4e-12, 4e-12, 8e-12],
        ]
    )
    conditional = conditional_covariance(
        covariance,
        np.array([2]),
        np.array([0, 1]),
    )
    assert conditional.shape == (1, 1)
    assert 0.0 < conditional[0, 0] < covariance[2, 2]
    whitened = whiten_residual_block(np.array([2e-6]), conditional)
    assert np.all(np.isfinite(whitened))
