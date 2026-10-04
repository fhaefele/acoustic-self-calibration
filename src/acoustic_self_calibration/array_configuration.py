"""Declared construction information; never microphone coordinates or metric pitches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .stratified.constraints import PlanarAngleConstraint


@dataclass(frozen=True)
class ArrayConfiguration:
    name: Literal["arbitrary-planar", "cross", "t", "grid", "room"]
    provenance: str = ""
    arms: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    # Directed endpoint pairs remove the 60/120-degree ray ambiguity and do not
    # require a microphone at the intersection.
    rays: tuple[tuple[int, int], tuple[int, int]] | None = None
    angle_deg: float | None = None
    grid_slots: tuple[tuple[int, int, int], ...] = ()
    equal_row_spacing: bool = False
    equal_column_spacing: bool = False

    def __post_init__(self):
        if self.name not in {"arbitrary-planar", "cross", "t", "grid", "room"}:
            raise ValueError("unknown array configuration")
        if self.name in {"cross", "t"}:
            if self.arms is None or any(
                len(set(arm)) != len(arm) or len(arm) < 2 for arm in self.arms
            ):
                raise ValueError("two arms with distinct microphone memberships are required")
            if len(set(self.arms[0]) & set(self.arms[1])) > 1:
                raise ValueError("arms can share only the junction microphone")
            if self.angle_deg is not None:
                if not np.isfinite(self.angle_deg) or not 0.01 < self.angle_deg < 179.99:
                    raise ValueError("angle_deg must define stable non-collinear directed rays")
                if self.rays is None or any(
                    len(set(ray)) != 2 or not set(ray) <= set(arm)
                    for ray, arm in zip(self.rays, self.arms, strict=True)
                ):
                    raise ValueError("each directed ray must select two distinct IDs on its arm")
        elif self.arms is not None or self.rays is not None or self.angle_deg is not None:
            raise ValueError("arm constraints require cross or t")
        if self.name == "grid":
            ids = [slot[0] for slot in self.grid_slots]
            slots = [(slot[1], slot[2]) for slot in self.grid_slots]
            if not slots or len(set(ids)) != len(ids) or len(set(slots)) != len(slots):
                raise ValueError("grid microphone IDs and occupied slots must be unique")
            if len({r for r, _ in slots}) < 2 or len({c for _, c in slots}) < 2:
                raise ValueError("grid needs at least two rows and columns")
        elif self.grid_slots or self.equal_row_spacing or self.equal_column_spacing:
            raise ValueError("grid constraints require grid configuration")
        if self.name in {"cross", "t", "grid"} and not self.provenance.strip():
            raise ValueError("construction provenance is required")

    @property
    def model(self):
        return "general_3d" if self.name == "room" else "receiver2d_source3d"

    def validate_ids(self, microphone_ids: tuple[int, ...]):
        if self.arms is not None:
            declared = set(self.arms[0]) | set(self.arms[1])
        elif self.grid_slots:
            declared = {slot[0] for slot in self.grid_slots}
        else:
            return
        if declared != set(microphone_ids):
            raise ValueError("construction membership must cover exactly the input microphone IDs")

    def metric_ambiguity_reason(self) -> str | None:
        if self.name in {"cross", "t"} and self.angle_deg is None:
            return "unknown_two_arm_angle_metric_family"
        if (
            self.name == "grid"
            and min(
                len({slot[1] for slot in self.grid_slots}),
                len({slot[2] for slot in self.grid_slots}),
            )
            == 2
        ):
            # Two parallel rows/columns lie on a conic. Perpendicularity and
            # equal pitch within either direction do not determine the gap.
            return "two_parallel_lines_metric_family"
        return None

    def angle_constraint(self) -> PlanarAngleConstraint | None:
        if self.rays is not None and self.angle_deg is not None:
            a, b = self.rays
            return PlanarAngleConstraint(
                a[0],
                a[1],
                b[1],
                np.deg2rad(self.angle_deg),
                self.provenance,
                arm_b_center_receiver=b[0],
            )
        if self.name == "grid":
            for center, row, column in self.grid_slots:
                along_x = [i for i, r, c in self.grid_slots if r == row and c > column]
                along_y = [i for i, r, c in self.grid_slots if c == column and r > row]
                if along_x and along_y:
                    return PlanarAngleConstraint(
                        center, min(along_x), min(along_y), np.pi / 2, self.provenance
                    )
            # A disconnected grid without a corner still has perpendicular axes.
            row_pairs = [
                (a[0], b[0])
                for a in self.grid_slots
                for b in self.grid_slots
                if a[1] == b[1] and a[2] < b[2]
            ]
            col_pairs = [
                (a[0], b[0])
                for a in self.grid_slots
                for b in self.grid_slots
                if a[2] == b[2] and a[1] < b[1]
            ]
            if row_pairs and col_pairs:
                a, b = min(row_pairs), min(col_pairs)
                return PlanarAngleConstraint(
                    a[0], a[1], b[1], np.pi / 2, self.provenance, arm_b_center_receiver=b[0]
                )
        return None

    def to_dict(self):
        return {
            **asdict(self),
            "angle_units": "degrees",
            "metric_spacing": "unknown",
            "conditioning": "conditional_on_exact_declared_construction",
        }

    @classmethod
    def from_dict(cls, value):
        allowed = set(cls.__dataclass_fields__)
        if set(value) - allowed:
            raise ValueError(
                "unknown construction fields: " + ", ".join(sorted(set(value) - allowed))
            )
        converted = dict(value)
        for key in ("arms", "rays", "grid_slots"):
            if converted.get(key) is not None:
                converted[key] = tuple(tuple(int(i) for i in row) for row in converted[key])
        return cls(**converted)
