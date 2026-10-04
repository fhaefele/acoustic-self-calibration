"""Exact reduced microphone coordinates for declared two-line and grid arrays."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from ..array_configuration import ArrayConfiguration
from .identifiability import planar_noise_sensitivity
from .refinement import _MeasurementObjective, _raw_tdoa_rms


def apply_array_configuration(
    calibration, measurements, configuration, *, speed_of_sound, **options
):
    """Apply the same declared-construction semantics to audio and TDOA stages."""
    reason = configuration.metric_ambiguity_reason()
    if reason is not None:
        return replace(
            calibration,
            status="degenerate",
            microphone_positions_m=None,
            source_representative_positions_m=None,
            source_projected_positions_m=None,
            source_unsigned_heights_m=None,
            source_height_sign_known=None,
            continuous_ambiguity_dimension=1,
            diagnostics=replace(
                calibration.diagnostics,
                rejection_reasons=(*calibration.diagnostics.rejection_reasons, reason),
            ),
        )
    return refine_structured(
        calibration, measurements, configuration, speed_of_sound=speed_of_sound, **options
    )


def structured_coordinates(microphones, ids, configuration: ArrayConfiguration):
    """Return origin, basis, constant coordinate map, initial parameters and bounds.

    A projection supplies the optimizer's starting values. Every subsequent iterate
    uses the declared construction; there is no post-fit coordinate snapping.
    """
    index = {i: k for k, i in enumerate(ids)}
    m = np.asarray(microphones)
    if configuration.arms is not None:
        assert configuration.rays is not None and configuration.angle_deg is not None
        arms = [np.array([index[i] for i in arm]) for arm in configuration.arms]
        directions = []
        centers = []
        for membership, ray in zip(arms, configuration.rays, strict=True):
            center = m[membership].mean(axis=0)
            _, _, vt = np.linalg.svd(m[membership] - center)
            direction = vt[0]
            if direction @ (m[index[ray[1]]] - m[index[ray[0]]]) < 0:
                direction = -direction
            centers.append(center)
            directions.append(direction)
        normal = np.cross(*directions)
        normal /= np.linalg.norm(normal)
        ex = directions[0]
        ey = np.cross(normal, ex)
        basis = np.column_stack([ex, ey, normal])
        coefficients = np.linalg.lstsq(
            np.column_stack([directions[0], -directions[1]]), centers[1] - centers[0], rcond=None
        )[0]
        origin = centers[0] + coefficients[0] * directions[0]
        theta = np.deg2rad(configuration.angle_deg)
        target = [np.array([1.0, 0.0]), np.array([np.cos(theta), np.sin(theta)])]
        shared = set(configuration.arms[0]) & set(configuration.arms[1])
        if shared:
            origin = m[index[next(iter(shared))]]
        free_ids = [i for i in ids if i not in shared]
        mapping = np.zeros((2 * len(ids), len(free_ids)))
        initial = []
        for column, i in enumerate(free_ids):
            arm = 0 if i in configuration.arms[0] else 1
            mapping[2 * index[i] : 2 * index[i] + 2, column] = target[arm]
            initial.append((m[index[i]] - origin) @ directions[arm])
        lower = np.full(len(initial), -np.inf)
        ray_map = np.eye(len(initial))
        for start_id, end_id in configuration.rays:
            if end_id in shared:
                column = free_ids.index(start_id)
                ray_map[column, column] = -1.0
                lower[column] = 0.0
            else:
                column = free_ids.index(end_id)
                if start_id not in shared:
                    ray_map[column, free_ids.index(start_id)] = 1.0
                lower[column] = 0.0
        initial = np.linalg.solve(ray_map, np.asarray(initial))
        mapping = mapping @ ray_map
        if np.any(np.asarray(initial)[np.isfinite(lower)] <= 0):
            raise ValueError("declared directed rays contradict recovered geometry")
    else:
        slots = configuration.grid_slots
        rows, columns = sorted({s[1] for s in slots}), sorted({s[2] for s in slots})
        angle = configuration.angle_constraint()
        if angle is None:
            raise ValueError("grid lacks observed row/column baselines")
        ex = m[index[angle.arm_a_receiver]] - m[index[angle.center_receiver]]
        ex /= np.linalg.norm(ex)
        v = (
            m[index[angle.arm_b_receiver]]
            - m[
                index[
                    angle.center_receiver
                    if angle.arm_b_center_receiver is None
                    else angle.arm_b_center_receiver
                ]
            ]
        )
        ey = v - (v @ ex) * ex
        ey /= np.linalg.norm(ey)
        basis = np.column_stack([ex, ey, np.cross(ex, ey)])
        xy = m @ basis[:, :2]
        xs = np.array(
            [np.mean([xy[index[i], 0] for i, r, c in slots if c == column]) for column in columns]
        )
        ys = np.array([np.mean([xy[index[i], 1] for i, r, c in slots if r == row]) for row in rows])
        origin = basis @ np.array([xs[0], ys[0], np.mean(m @ basis[:, 2])])
        x_count = 1 if configuration.equal_column_spacing else len(columns) - 1
        y_count = 1 if configuration.equal_row_spacing else len(rows) - 1
        mapping = np.zeros((2 * len(ids), x_count + y_count))
        for i, r, c in slots:
            cx, ry = columns.index(c), rows.index(r)
            if configuration.equal_column_spacing:
                mapping[2 * index[i], 0] = c - columns[0]
            else:
                mapping[2 * index[i], :cx] = 1
            if configuration.equal_row_spacing:
                mapping[2 * index[i] + 1, x_count] = r - rows[0]
            else:
                mapping[2 * index[i] + 1, x_count : x_count + ry] = 1
        x_initial = (
            [
                np.dot(np.array(columns) - columns[0], xs - xs[0])
                / np.sum((np.array(columns) - columns[0]) ** 2)
            ]
            if configuration.equal_column_spacing
            else np.diff(xs)
        )
        y_initial = (
            [np.dot(np.array(rows) - rows[0], ys - ys[0]) / np.sum((np.array(rows) - rows[0]) ** 2)]
            if configuration.equal_row_spacing
            else np.diff(ys)
        )
        initial = np.concatenate([x_initial, y_initial])
        if np.any(initial <= 0):
            raise ValueError("declared grid ordering contradicts recovered geometry")
        lower = np.zeros(len(initial))
    return origin, basis, mapping, np.asarray(initial), lower


def refine_structured(
    calibration, measurements, configuration, *, speed_of_sound, mode="wls", max_nfev=200
):
    """Fit unknown spacings and per-event sources within the construction exactly."""
    if (
        calibration.microphone_positions_m is None
        or calibration.source_representative_positions_m is None
    ):
        return calibration
    origin, basis, mapping, initial, lower = structured_coordinates(
        calibration.microphone_positions_m, measurements.microphone_ids, configuration
    )
    sources = (calibration.source_representative_positions_m - origin) @ basis
    event_mask = np.isfinite(sources).all(axis=1)
    if np.nanmedian(sources[:, 2]) < 0:
        basis[:, 2] *= -1
        sources[:, 2] *= -1
    sources = np.where(event_mask[:, None], sources, 0.0)
    count = len(initial)
    start = np.concatenate([initial, sources.ravel()])
    low = np.concatenate([lower, np.full(sources.size, -np.inf)])
    low[count + 2 :: 3] = 0.0

    def unpack(p):
        xy = (mapping @ p[:count]).reshape(-1, 2)
        return np.column_stack([xy, np.zeros(len(xy))]), p[count:].reshape(sources.shape)

    objective = _MeasurementObjective(measurements, speed_of_sound, event_mask=event_mask)

    def residual(p):
        return objective.residual(*unpack(p))

    def jacobian(p):
        jac = objective.jacobian(
            *unpack(p), free_microphone_indices=np.arange(mapping.shape[0]), microphone_dimension=2
        )
        return np.column_stack([jac[:, : mapping.shape[0]] @ mapping, jac[:, mapping.shape[0] :]])

    optimized = least_squares(
        residual,
        start,
        jac=jacobian,
        bounds=(low, np.full(len(low), np.inf)),
        loss="huber" if mode == "huber" else "linear",
        max_nfev=max_nfev,
    )
    microphones, refined_sources = unpack(optimized.x)
    sensitivity = planar_noise_sensitivity(
        objective.jacobian(
            microphones, refined_sources, np.arange(mapping.shape[0]), microphone_dimension=2
        ),
        microphones[:, :2],
        whitened_residual=residual(optimized.x),
        coordinate_tangent=mapping,
    )
    heights = np.abs(refined_sources[:, 2])
    heights[~event_mask] = np.nan
    if (
        not np.all(np.isfinite(optimized.x))
        or np.linalg.norm(residual(optimized.x)) > np.linalg.norm(residual(start)) + 1e-8
    ):
        return replace(
            calibration,
            status="failed",
            diagnostics=replace(
                calibration.diagnostics,
                rejection_reasons=(
                    *calibration.diagnostics.rejection_reasons,
                    "construction_fit_failed",
                ),
            ),
        )
    microphones = microphones @ basis.T + origin
    refined_sources = refined_sources @ basis.T + origin
    refined_sources[~event_mask] = np.nan
    rms = _raw_tdoa_rms(
        microphones, refined_sources, measurements, speed_of_sound, event_mask=event_mask
    )
    # Plane projection is expressed in the current output frame; the public
    # coordinate-frame step supplies canonical x/y values afterward.
    return replace(
        calibration,
        microphone_positions_m=microphones,
        source_representative_positions_m=refined_sources,
        source_unsigned_heights_m=heights,
        source_projected_positions_m=refined_sources[:, :2],
        tdoa_rms_s=rms,
        diagnostics=replace(
            calibration.diagnostics,
            planar_microphone_rms_std_m=sensitivity.weakest_microphone_rms_std_m,
            planar_relative_std=sensitivity.relative_weakest_std,
            planar_fit_p_value=sensitivity.fit_p_value,
            planar_reduced_chi_square=sensitivity.reduced_chi_square,
        ),
        status=(
            "degenerate"
            if sensitivity.observable_rank < sensitivity.parameter_count
            else "weakly_identified"
            if sensitivity.relative_weakest_std > 0.05
            or (sensitivity.fit_p_value is not None and sensitivity.fit_p_value < 0.001)
            else calibration.status
        ),
    )
