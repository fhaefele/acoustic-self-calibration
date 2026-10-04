from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelName = Literal["receiver2d_source3d", "receiver3d_source3d"]
ModelComparisonStatus = Literal[
    "receiver2d_source3d",
    "receiver3d_source3d",
    "ambiguous",
    "insufficient_data",
]


@dataclass(frozen=True)
class ModelValidationEvidence:
    model: ModelName
    eligible: bool
    validation_event_ids: tuple[int, ...]
    normalized_score: float | None
    independent_coordinate_count: int
    weak_conditioning: bool = False


@dataclass(frozen=True)
class ModelComparisonResult:
    status: ModelComparisonStatus
    preferred_model: ModelName | None
    competing_models: tuple[ModelName, ...]
    score_difference: float | None
    reason: str


def compare_model_evidence(
    planar: ModelValidationEvidence,
    general_3d: ModelValidationEvidence,
    *,
    score_tolerance: float = 0.05,
    minimum_independent_coordinates: int = 1,
) -> ModelComparisonResult:
    """Compare models only on a common held-out event basis."""
    if planar.model != "receiver2d_source3d":
        raise ValueError("planar evidence has the wrong model tag")
    if general_3d.model != "receiver3d_source3d":
        raise ValueError("general_3d evidence has the wrong model tag")
    if planar.validation_event_ids != general_3d.validation_event_ids:
        raise ValueError("models must use the same validation event IDs")
    if score_tolerance < 0.0:
        raise ValueError("score_tolerance must be non-negative")
    if minimum_independent_coordinates < 1:
        raise ValueError("minimum_independent_coordinates must be positive")

    def usable(evidence: ModelValidationEvidence) -> bool:
        return (
            evidence.eligible
            and evidence.normalized_score is not None
            and evidence.independent_coordinate_count >= minimum_independent_coordinates
        )

    planar_ok = usable(planar)
    general_ok = usable(general_3d)
    if not planar_ok and not general_ok:
        return ModelComparisonResult(
            status="insufficient_data",
            preferred_model=None,
            competing_models=(),
            score_difference=None,
            reason="neither_model_has_eligible_held_out_evidence",
        )
    if planar_ok and not general_ok:
        return ModelComparisonResult(
            status="receiver2d_source3d",
            preferred_model="receiver2d_source3d",
            competing_models=("receiver2d_source3d",),
            score_difference=None,
            reason="only_planar_model_is_eligible",
        )
    if general_ok and not planar_ok:
        return ModelComparisonResult(
            status="receiver3d_source3d",
            preferred_model="receiver3d_source3d",
            competing_models=("receiver3d_source3d",),
            score_difference=None,
            reason="only_general_3d_model_is_eligible",
        )

    if planar_ok and general_ok and planar.weak_conditioning != general_3d.weak_conditioning:
        preferred = "receiver3d_source3d" if planar.weak_conditioning else "receiver2d_source3d"
        return ModelComparisonResult(
            status=preferred,
            preferred_model=preferred,
            competing_models=("receiver2d_source3d", "receiver3d_source3d"),
            score_difference=(
                None
                if planar.normalized_score is None or general_3d.normalized_score is None
                else float(planar.normalized_score - general_3d.normalized_score)
            ),
            reason="competing_model_is_weakly_identified",
        )

    assert planar.normalized_score is not None
    assert general_3d.normalized_score is not None
    difference = float(planar.normalized_score - general_3d.normalized_score)
    if abs(difference) <= score_tolerance:
        return ModelComparisonResult(
            status="ambiguous",
            preferred_model=None,
            competing_models=("receiver2d_source3d", "receiver3d_source3d"),
            score_difference=difference,
            reason="held_out_scores_indistinguishable",
        )
    if difference < 0.0:
        return ModelComparisonResult(
            status="receiver2d_source3d",
            preferred_model="receiver2d_source3d",
            competing_models=("receiver2d_source3d", "receiver3d_source3d"),
            score_difference=difference,
            reason="planar_model_has_better_held_out_score",
        )
    return ModelComparisonResult(
        status="receiver3d_source3d",
        preferred_model="receiver3d_source3d",
        competing_models=("receiver2d_source3d", "receiver3d_source3d"),
        score_difference=difference,
        reason="general_3d_model_has_better_held_out_score",
    )
