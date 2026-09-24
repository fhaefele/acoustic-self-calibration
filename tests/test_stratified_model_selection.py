from acoustic_self_calibration.stratified.model_selection import (
    ModelName,
    ModelValidationEvidence,
    compare_model_evidence,
)


def _evidence(
    model: ModelName,
    *,
    score: float | None,
    eligible: bool = True,
    event_ids: tuple[int, ...] = (12, 13, 14, 15),
    count: int = 12,
    weak: bool = False,
) -> ModelValidationEvidence:
    return ModelValidationEvidence(
        model=model,
        eligible=eligible,
        validation_event_ids=event_ids,
        normalized_score=score,
        independent_coordinate_count=count,
        weak_conditioning=weak,
    )


def test_clearly_planar_evidence_selects_planar() -> None:
    result = compare_model_evidence(
        _evidence("receiver2d_source3d", score=0.2),
        _evidence("receiver3d_source3d", score=None, eligible=False),
    )
    assert result.status == "receiver2d_source3d"
    assert result.preferred_model == "receiver2d_source3d"


def test_clearly_3d_evidence_selects_general_3d() -> None:
    result = compare_model_evidence(
        _evidence("receiver2d_source3d", score=2.0),
        _evidence("receiver3d_source3d", score=0.2),
        score_tolerance=0.05,
    )
    assert result.status == "receiver3d_source3d"
    assert result.preferred_model == "receiver3d_source3d"


def test_near_tie_is_reported_as_model_ambiguity() -> None:
    result = compare_model_evidence(
        _evidence("receiver2d_source3d", score=0.22),
        _evidence("receiver3d_source3d", score=0.20),
        score_tolerance=0.05,
    )
    assert result.status == "ambiguous"
    assert result.preferred_model is None
    assert set(result.competing_models) == {
        "receiver2d_source3d",
        "receiver3d_source3d",
    }


def test_comparison_rejects_different_validation_events() -> None:
    try:
        compare_model_evidence(
            _evidence("receiver2d_source3d", score=0.2),
            _evidence(
                "receiver3d_source3d",
                score=0.2,
                event_ids=(13, 14, 15, 16),
            ),
        )
    except ValueError as error:
        assert "same validation event IDs" in str(error)
    else:
        raise AssertionError("mismatched validation sets must be rejected")


def test_strong_planar_model_beats_weak_3d_tie() -> None:
    result = compare_model_evidence(
        _evidence("receiver2d_source3d", score=0.2),
        _evidence("receiver3d_source3d", score=0.2, weak=True),
    )
    assert result.status == "receiver2d_source3d"
    assert result.reason == "competing_model_is_weakly_identified"
