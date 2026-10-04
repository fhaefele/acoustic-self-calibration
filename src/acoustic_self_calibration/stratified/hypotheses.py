from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import rigid_align, rms_position_error
from .robustness import RobustResidualSummary


@dataclass(frozen=True)
class MeasurementSplit:
    """Primitive reference-star lineage masks, shape (receivers, events)."""

    generation_mask: np.ndarray
    completion_mask: np.ndarray
    validation_mask: np.ndarray

    def __post_init__(self) -> None:
        generation = np.array(self.generation_mask, dtype=bool, copy=True)
        completion = np.array(self.completion_mask, dtype=bool, copy=True)
        validation = np.array(self.validation_mask, dtype=bool, copy=True)
        if generation.shape != completion.shape or generation.shape != validation.shape:
            raise ValueError("all split masks must have the same shape")
        if generation.ndim != 2:
            raise ValueError("split masks must have shape (receivers, events)")
        if np.any(generation & completion):
            raise ValueError("generation and completion measurements overlap")
        if np.any(generation & validation):
            raise ValueError("generation and validation measurements overlap")
        if np.any(completion & validation):
            raise ValueError("completion and validation measurements overlap")
        for array in (generation, completion, validation):
            array.setflags(write=False)
        object.__setattr__(self, "generation_mask", generation)
        object.__setattr__(self, "completion_mask", completion)
        object.__setattr__(self, "validation_mask", validation)


@dataclass(frozen=True)
class FullGeometryHypothesis:
    subset_id: str
    root_id: int
    microphone_positions_m: np.ndarray
    source_positions_m: np.ndarray
    split: MeasurementSplit
    validation: RobustResidualSummary
    full_tdoa_rms_s: float
    metric_rms_m: float
    metric_condition_number: float | None = None
    affine_factor_singular_ratio: float | None = None
    model: str = "receiver3d_source3d"
    seed_microphone_ids: tuple[int, ...] = ()
    seed_event_ids: tuple[int, ...] = ()
    metric_branch_id: int | None = None
    arrival_gauge_shifts_m: np.ndarray | None = None
    offset_scope_event_ids: tuple[int, ...] = ()
    microphone_completion_status: np.ndarray | None = None
    source_completion_status: np.ndarray | None = None
    generating_equation_rms: float | None = None


@dataclass(frozen=True)
class HypothesisClass:
    representative: FullGeometryHypothesis
    members: tuple[FullGeometryHypothesis, ...]
    independent_subset_support: int


def make_holdout_split(
    *,
    receiver_count: int,
    event_count: int,
    seed_receivers: tuple[int, ...],
    seed_events: tuple[int, ...],
    fitting_events: tuple[int, ...],
    validation_events: tuple[int, ...],
    validation_localization_receivers: tuple[int, ...],
) -> MeasurementSplit:
    """Construct disjoint primitive masks for generation, completion, and scoring."""
    if 0 not in seed_receivers or 0 not in validation_localization_receivers:
        raise ValueError("reference receiver 0 must be present in seed/localization sets")
    if set(fitting_events) & set(validation_events):
        raise ValueError("fitting_events and validation_events must be disjoint")
    if not set(seed_events).issubset(set(fitting_events)):
        raise ValueError("seed_events must be a subset of fitting_events")

    shape = (receiver_count, event_count)
    generation = np.zeros(shape, dtype=bool)
    completion = np.zeros(shape, dtype=bool)
    validation = np.zeros(shape, dtype=bool)

    nonreference_seed = [receiver for receiver in seed_receivers if receiver != 0]
    for event in seed_events:
        generation[nonreference_seed, event] = True

    for event in fitting_events:
        for receiver in range(1, receiver_count):
            if not generation[receiver, event]:
                completion[receiver, event] = True

    localization_nonreference = [
        receiver for receiver in validation_localization_receivers if receiver != 0
    ]
    held_out_receivers = [
        receiver
        for receiver in range(1, receiver_count)
        if receiver not in validation_localization_receivers
    ]
    for event in validation_events:
        completion[localization_nonreference, event] = True
        validation[held_out_receivers, event] = True

    return MeasurementSplit(
        generation_mask=generation,
        completion_mask=completion,
        validation_mask=validation,
    )


def cluster_equivalent_hypotheses(
    hypotheses: tuple[FullGeometryHypothesis, ...],
    *,
    microphone_rms_tolerance_m: float,
) -> tuple[HypothesisClass, ...]:
    """Cluster by receiver-only rigid equivalence and count distinct subset support."""
    if microphone_rms_tolerance_m <= 0.0:
        raise ValueError("microphone_rms_tolerance_m must be positive")
    classes: list[list[FullGeometryHypothesis]] = []
    for hypothesis in sorted(
        hypotheses,
        key=lambda item: (
            item.validation.normalized_huber_score,
            item.full_tdoa_rms_s,
            item.metric_rms_m,
            item.subset_id,
            item.root_id,
        ),
    ):
        placed = False
        for group in classes:
            representative = group[0]
            aligned, _, _ = rigid_align(
                hypothesis.microphone_positions_m,
                representative.microphone_positions_m,
            )
            if (
                rms_position_error(
                    aligned,
                    representative.microphone_positions_m,
                )
                <= microphone_rms_tolerance_m
            ):
                group.append(hypothesis)
                placed = True
                break
        if not placed:
            classes.append([hypothesis])

    output = []
    for group in classes:
        representative = min(
            group,
            key=lambda item: (
                item.validation.normalized_huber_score,
                item.full_tdoa_rms_s,
                item.metric_rms_m,
            ),
        )
        output.append(
            HypothesisClass(
                representative=representative,
                members=tuple(group),
                independent_subset_support=len({item.subset_id for item in group}),
            )
        )
    output.sort(
        key=lambda item: (
            -item.independent_subset_support,
            item.representative.validation.normalized_huber_score,
            item.representative.full_tdoa_rms_s,
        )
    )
    return tuple(output)
