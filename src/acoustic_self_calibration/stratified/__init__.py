"""Stratified TDOA geometry primitives.

Milestones B-D build the geometry backend stage by stage. Modules in this package do
not access audio samples and do not use Bayesian/posterior branch selection.
"""

from .constraints import (
    PlanarAngleConstraint,
    metric_angle_rad,
    right_angle_metric_row,
)
from .expansion import (
    OffsetExpansionResult,
    ReceiverLocalizationResult,
    SourceLocalizationResult,
    extend_event_offsets,
    localize_receiver_from_ranges,
    localize_source_from_tdoa,
)
from .factorization import AffineFactorization, factor_corrected_ranges
from .hypotheses import (
    FullGeometryHypothesis,
    HypothesisClass,
    MeasurementSplit,
    cluster_equivalent_hypotheses,
    make_holdout_split,
)
from .identifiability import (
    PlanarIdentifiabilityDiagnostics,
    diagnose_planar_identifiability,
    planar_metric_design,
    planar_quadratic_design,
)
from .metric_upgrade import (
    MetricUpgradeCandidate,
    MetricUpgradeDiagnostics,
    MetricUpgradeResult,
    build_receiver_metric_system,
    upgrade_metric_3d,
    upgrade_metric_3d_overdetermined,
)
from .model_selection import (
    ModelComparisonResult,
    ModelName,
    ModelValidationEvidence,
    compare_model_evidence,
)
from .notation import (
    MatrixRankDiagnostics,
    apply_arrival_gauge,
    compacted_rank_matrix,
    compaction_matrix,
    corrected_ranges_from_offsets,
    cross_gram_from_ranges,
    modified_squared_range_matrix,
    rank_diagnostics,
    reference_arrivals_from_ranges,
    restore_offsets_from_gauge,
)
from .offsets_linear import (
    LinearOffsetDiagnostics,
    LinearOffsetSolution,
    build_linear_offset_system,
    solve_linear_offsets,
)
from .offsets_minimal import (
    MinimalOffsetDiagnostics,
    MinimalOffsetResult,
    MinimalOffsetRoot,
    solve_offsets_7r6s,
)
from .planar import (
    PlanarMetricCandidate,
    PlanarMetricDiagnostics,
    PlanarMetricResult,
    PlanarReceiverLocalizationResult,
    PlanarSourceLocalizationResult,
    localize_planar_receiver_from_ranges,
    localize_planar_source_from_tdoa,
    upgrade_metric_planar,
)
from .robustness import (
    RobustResidualSummary,
    conditional_covariance,
    huber_loss,
    summarize_independent_residuals,
    whiten_residual_block,
)
from .solver import (
    JointModelCalibrationResult,
    PlanarCalibrationResult,
    StratifiedCalibrationDiagnostics,
    StratifiedCalibrationResult,
    calibrate_planar_tdoa,
    calibrate_planar_tdoa_8mic,
    calibrate_tdoa,
    calibrate_tdoa_8mic,
    compare_tdoa_models_8mic,
)

__all__ = [
    "AffineFactorization",
    "FullGeometryHypothesis",
    "HypothesisClass",
    "LinearOffsetDiagnostics",
    "LinearOffsetSolution",
    "JointModelCalibrationResult",
    "MatrixRankDiagnostics",
    "ModelComparisonResult",
    "ModelName",
    "ModelValidationEvidence",
    "MeasurementSplit",
    "MetricUpgradeCandidate",
    "MetricUpgradeDiagnostics",
    "MetricUpgradeResult",
    "MinimalOffsetDiagnostics",
    "MinimalOffsetResult",
    "MinimalOffsetRoot",
    "OffsetExpansionResult",
    "PlanarAngleConstraint",
    "PlanarIdentifiabilityDiagnostics",
    "PlanarMetricCandidate",
    "PlanarMetricDiagnostics",
    "PlanarMetricResult",
    "PlanarReceiverLocalizationResult",
    "PlanarSourceLocalizationResult",
    "ReceiverLocalizationResult",
    "RobustResidualSummary",
    "SourceLocalizationResult",
    "PlanarCalibrationResult",
    "StratifiedCalibrationDiagnostics",
    "StratifiedCalibrationResult",
    "apply_arrival_gauge",
    "build_linear_offset_system",
    "build_receiver_metric_system",
    "calibrate_planar_tdoa",
    "calibrate_planar_tdoa_8mic",
    "calibrate_tdoa",
    "calibrate_tdoa_8mic",
    "cluster_equivalent_hypotheses",
    "conditional_covariance",
    "compare_model_evidence",
    "compare_tdoa_models_8mic",
    "compaction_matrix",
    "compacted_rank_matrix",
    "corrected_ranges_from_offsets",
    "cross_gram_from_ranges",
    "diagnose_planar_identifiability",
    "extend_event_offsets",
    "factor_corrected_ranges",
    "huber_loss",
    "localize_planar_receiver_from_ranges",
    "localize_planar_source_from_tdoa",
    "localize_receiver_from_ranges",
    "localize_source_from_tdoa",
    "make_holdout_split",
    "metric_angle_rad",
    "modified_squared_range_matrix",
    "planar_metric_design",
    "planar_quadratic_design",
    "rank_diagnostics",
    "reference_arrivals_from_ranges",
    "restore_offsets_from_gauge",
    "right_angle_metric_row",
    "solve_linear_offsets",
    "solve_offsets_7r6s",
    "summarize_independent_residuals",
    "upgrade_metric_3d",
    "upgrade_metric_3d_overdetermined",
    "upgrade_metric_planar",
    "whiten_residual_block",
]
