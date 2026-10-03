"""A public frame anchored to a microphone, separate from the timing reference."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .stratified.solver import PlanarCalibrationResult


@dataclass(frozen=True)
class MicrophoneFrame:
    origin_microphone_id: int
    x_axis_microphone_id: int
    plane_microphone_id: int
    input_origin_m: np.ndarray
    input_basis: np.ndarray

    def transform(self, points):
        if points is None:
            return None
        transformed = (np.asarray(points) - self.input_origin_m) @ self.input_basis
        transformed.setflags(write=False)
        return transformed

    def to_dict(self):
        return dict(
            units="metres",
            origin_microphone_id=self.origin_microphone_id,
            x_axis_microphone_id=self.x_axis_microphone_id,
            plane_microphone_id=self.plane_microphone_id,
            convention="origin microphone; farthest baseline; widest plane baseline; ID tie breaks",
            input_origin_m=self.input_origin_m.tolist(),
            input_basis=self.input_basis.tolist(),
            transform="output = (input - input_origin_m) @ input_basis",
            audit_frame="same transform as final scene; audit origin may move during refinement",
        )


def select_microphone_frame(microphones, ids, *, origin_id=None, common_side_sources=None):
    m = np.asarray(microphones, dtype=float)
    if len(ids) != len(m) or len(set(ids)) != len(ids):
        raise ValueError("microphone frame requires unique IDs matching geometry")
    if not np.isfinite(m).all():
        raise ValueError("microphone frame requires finite microphones")
    if origin_id is None:
        origin_id = 0 if 0 in ids else min(ids)
    if origin_id not in ids:
        raise ValueError("output origin microphone ID is absent")
    origin = m[ids.index(origin_id)]
    centered = m - origin

    def widest(distances):
        maximum = float(np.max(distances))
        return min(
            (i for i, d in enumerate(distances) if maximum - d <= 1e-10 * max(maximum, 1.0)),
            key=lambda i: ids[i],
        )

    axis = widest(np.linalg.norm(centered, axis=1))
    ex = centered[axis] / np.linalg.norm(centered[axis])
    perpendicular = centered - np.outer(centered @ ex, ex)
    plane = widest(np.linalg.norm(perpendicular, axis=1))
    if np.linalg.norm(perpendicular[plane]) <= 1e-10 * max(1.0, np.linalg.norm(centered[axis])):
        raise ValueError("microphones lack a stable plane baseline")
    ey = perpendicular[plane] / np.linalg.norm(perpendicular[plane])
    ez = np.cross(ex, ey)
    depth = centered @ ez
    orientation = widest(np.abs(depth))
    if abs(depth[orientation]) > 1e-10 * max(1.0, np.linalg.norm(centered[axis])):
        if depth[orientation] < 0:
            ez *= -1
    elif common_side_sources is not None:
        heights = (np.asarray(common_side_sources) - origin) @ ez
        finite = heights[np.isfinite(heights)]
        if len(finite) and np.median(finite) < 0:
            ez *= -1
    return MicrophoneFrame(
        origin_id, ids[axis], ids[plane], origin.copy(), np.column_stack([ex, ey, ez])
    )


def transform_calibration(calibration, frame):
    if calibration is None:
        return None
    microphones = frame.transform(calibration.microphone_positions_m)
    if isinstance(calibration, PlanarCalibrationResult):
        sources = frame.transform(calibration.source_representative_positions_m)
        return replace(
            calibration,
            microphone_positions_m=microphones,
            source_representative_positions_m=sources,
            source_projected_positions_m=None if sources is None else sources[:, :2],
            source_unsigned_heights_m=None if sources is None else np.abs(sources[:, 2]),
        )

    def transform_class(group):
        def hypothesis(h):
            return replace(
                h,
                microphone_positions_m=frame.transform(h.microphone_positions_m),
                source_positions_m=frame.transform(h.source_positions_m),
            )

        return replace(
            group,
            representative=hypothesis(group.representative),
            members=tuple(hypothesis(h) for h in group.members),
        )

    return replace(
        calibration,
        microphone_positions_m=microphones,
        source_positions_m=frame.transform(calibration.source_positions_m),
        selected_class=None
        if calibration.selected_class is None
        else transform_class(calibration.selected_class),
        classes=tuple(transform_class(group) for group in calibration.classes),
    )
