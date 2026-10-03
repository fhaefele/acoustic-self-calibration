from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarAngleConstraint:
    """Explicit angle between two receiver arms in the planar metric."""

    center_receiver: int
    arm_a_receiver: int
    arm_b_receiver: int
    angle_rad: float
    provenance: str
    exact: bool = True
    arm_b_center_receiver: int | None = None

    def __post_init__(self) -> None:
        ids = (self.center_receiver, self.arm_a_receiver, self.arm_b_receiver)
        if len(set(ids)) != 3 or any(value < 0 for value in ids):
            raise ValueError("angle constraint receiver IDs must be distinct and non-negative")
        if self.arm_b_center_receiver is not None and (
            self.arm_b_center_receiver < 0 or self.arm_b_center_receiver == self.arm_b_receiver
        ):
            raise ValueError("second ray origin must be non-negative and distinct from its end")
        if not 0.0 < self.angle_rad < np.pi:
            raise ValueError("angle_rad must lie in (0, pi)")
        if not self.provenance:
            raise ValueError("constraint provenance cannot be empty")
        if not self.exact:
            raise ValueError("Milestone D supports exact angle constraints only")


def right_angle_metric_row(
    receiver_factors: np.ndarray,
    constraint: PlanarAngleConstraint,
    *,
    angle_tolerance_rad: float = 1e-12,
) -> np.ndarray:
    """Return one linear metric row for an exact 90-degree arm constraint."""
    if abs(constraint.angle_rad - np.pi / 2.0) > angle_tolerance_rad:
        raise ValueError("current planar metric backend supports only exact right angles")
    x = np.asarray(receiver_factors, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("receiver_factors must have shape (receivers, 2)")
    max_id = max(
        constraint.center_receiver,
        constraint.arm_a_receiver,
        constraint.arm_b_receiver,
        constraint.arm_b_center_receiver or 0,
    )
    if max_id >= len(x):
        raise ValueError("angle constraint receiver ID is out of range")
    center = x[constraint.center_receiver]
    arm_a = x[constraint.arm_a_receiver] - center
    arm_b = (
        x[constraint.arm_b_receiver]
        - x[
            constraint.center_receiver
            if constraint.arm_b_center_receiver is None
            else constraint.arm_b_center_receiver
        ]
    )
    if np.linalg.norm(arm_a) <= 1e-12 or np.linalg.norm(arm_b) <= 1e-12:
        raise ValueError("angle constraint arm has zero affine length")
    return np.array(
        [
            arm_a[0] * arm_b[0],
            arm_a[0] * arm_b[1] + arm_a[1] * arm_b[0],
            arm_a[1] * arm_b[1],
            0.0,
            0.0,
        ],
        dtype=float,
    )


def metric_angle_rad(
    receiver_factors: np.ndarray,
    metric_matrix: np.ndarray,
    constraint: PlanarAngleConstraint,
) -> float:
    """Evaluate the physical arm angle under one recovered planar metric."""
    x = np.asarray(receiver_factors, dtype=float)
    h = np.asarray(metric_matrix, dtype=float)
    center = x[constraint.center_receiver]
    arm_a = x[constraint.arm_a_receiver] - center
    arm_b = (
        x[constraint.arm_b_receiver]
        - x[
            constraint.center_receiver
            if constraint.arm_b_center_receiver is None
            else constraint.arm_b_center_receiver
        ]
    )
    dot = float(arm_a @ h @ arm_b)
    norm_a = float(np.sqrt(arm_a @ h @ arm_a))
    norm_b = float(np.sqrt(arm_b @ h @ arm_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("metric gives a non-positive arm length")
    cosine = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
    return float(np.arccos(cosine))


def angle_metric_rows(receiver_factors: np.ndarray, constraint: PlanarAngleConstraint):
    """Linear dot/norm forms for the directed arm angle (including separated arms)."""
    x = np.asarray(receiver_factors, dtype=float)
    u = x[constraint.arm_a_receiver] - x[constraint.center_receiver]
    v = (
        x[constraint.arm_b_receiver]
        - x[
            constraint.center_receiver
            if constraint.arm_b_center_receiver is None
            else constraint.arm_b_center_receiver
        ]
    )
    if abs(np.linalg.det(np.stack([u, v]))) <= 1e-10 * np.linalg.norm(u) * np.linalg.norm(v):
        raise ValueError("angle arms are singular or nearly collinear")

    def row(a, b):
        return np.array([a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[1] * b[1], 0.0, 0.0])

    return row(u, v), row(u, u), row(v, v)
