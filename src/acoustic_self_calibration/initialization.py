from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize

from .geometry import canonicalize_scene


def pack_geometry(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Pack geometry after fixing the six rigid-body gauge degrees of freedom."""
    m = np.asarray(microphones, dtype=float)
    s = np.asarray(sources, dtype=float)
    head = np.array(
        [m[1, 0], m[2, 0], m[2, 1], m[3, 0], m[3, 1], m[3, 2]],
        dtype=float,
    )
    return np.concatenate([head, m[4:].reshape(-1), s.reshape(-1)])


def unpack_geometry(
    parameters: np.ndarray,
    microphone_count: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the canonical 3-D microphone/source state."""
    x = np.asarray(parameters, dtype=float)
    microphones = np.zeros((microphone_count, 3), dtype=float)
    x1, x2, y2, x3, y3, z3 = x[:6]
    microphones[1] = [x1, 0.0, 0.0]
    microphones[2] = [x2, y2, 0.0]
    microphones[3] = [x3, y3, z3]
    cursor = 6
    if microphone_count > 4:
        count = 3 * (microphone_count - 4)
        microphones[4:] = x[cursor : cursor + count].reshape(microphone_count - 4, 3)
        cursor += count
    sources = x[cursor:].reshape(frame_count, 3)
    return microphones, sources


def free_microphone_parameter_count(microphone_count: int) -> int:
    return 6 + 3 * max(0, microphone_count - 4)


def low_rank_initial_scene(
    observed_range_differences: np.ndarray,
    position_bound_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a 3-D scene from range-difference data.

    The unknown source-to-reference ranges are optimized so the doubly centered
    cross-squared-distance matrix is approximately rank three. A rank-3 factorization
    then supplies microphone and source coordinates up to an affine ambiguity, which
    is resolved by fitting the recovered cross distances.

    This gives the nonlinear MAP optimizer a geometry-aware start without assuming
    any microphone coordinates are known.
    """
    observed = np.asarray(observed_range_differences, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed_range_differences must have shape (frames, microphones-1)")
    frame_count, nonreference_count = observed.shape
    microphone_count = nonreference_count + 1
    if microphone_count < 4 or frame_count < 4:
        raise ValueError("At least four microphones and four source frames are required")

    differences = np.vstack([np.zeros(frame_count), observed.T])
    center_m = (
        np.eye(microphone_count) - np.ones((microphone_count, microphone_count)) / microphone_count
    )
    center_t = np.eye(frame_count) - np.ones((frame_count, frame_count)) / frame_count

    eps = 1e-9
    lower = np.maximum(0.02, -np.min(differences, axis=0) + 0.02)
    upper = np.maximum(lower + 0.5, 2.0 * float(position_bound_m))
    aperture = max(0.25, float(np.percentile(np.abs(differences), 90)))

    def rank_objective_and_gradient(reference_ranges: np.ndarray) -> tuple[float, np.ndarray]:
        ranges = differences + reference_ranges[None, :]
        squared = ranges * ranges
        centered = -0.5 * (center_m @ squared @ center_t)
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(3, len(singular))
        rank3 = (u[:, :rank] * singular[:rank]) @ vt[:rank]
        tail = centered - rank3

        tail_energy = float(np.sum(tail * tail))
        total_energy = float(np.sum(centered * centered)) + eps
        value = tail_energy / total_energy

        grad_squared_tail = -(center_m @ tail @ center_t)
        grad_squared_total = -(center_m @ centered @ center_t)
        grad_tail = 2.0 * np.sum(grad_squared_tail * ranges, axis=0)
        grad_total = 2.0 * np.sum(grad_squared_total * ranges, axis=0)
        gradient = (grad_tail * total_energy - tail_energy * grad_total) / (total_energy**2)
        return value, gradient

    fits = []
    for multiplier in (0.25, 0.75, 1.5, 3.0):
        initial = np.clip(lower + multiplier * aperture, lower, upper)
        fit = minimize(
            rank_objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9},
        )
        fits.append(fit)

    reference_ranges = min(fits, key=lambda fit: float(fit.fun)).x
    ranges = differences + reference_ranges[None, :]
    squared = ranges * ranges
    centered = -0.5 * (center_m @ squared @ center_t)

    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    root = np.sqrt(np.maximum(singular[:3], 0.0))
    left = u[:, :3] * root[None, :]
    right = vt[:3].T * root[None, :]

    affine_initial = np.concatenate([np.eye(3).reshape(-1), np.zeros(3)])

    def affine_unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transform = parameters[:9].reshape(3, 3)
        try:
            inverse_transpose = np.linalg.inv(transform).T
        except np.linalg.LinAlgError:
            inverse_transpose = np.linalg.pinv(transform).T
        return left @ transform, right @ inverse_transpose, parameters[9:12]

    def affine_residual(parameters: np.ndarray) -> np.ndarray:
        microphones, sources, offset = affine_unpack(parameters)
        predicted = np.sum(
            (microphones[:, None, :] - sources[None, :, :] + offset) ** 2,
            axis=2,
        )
        scale = 1.0 + np.sqrt(np.maximum(squared, 0.0))
        data = ((predicted - squared) / scale).reshape(-1)
        regularizer = 1e-4 * (parameters[:9].reshape(3, 3) - np.eye(3)).reshape(-1)
        return np.concatenate([data, regularizer])

    affine_fit = least_squares(
        affine_residual,
        affine_initial,
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=1500,
    )
    microphones, sources, offset = affine_unpack(affine_fit.x)
    return canonicalize_scene(microphones + offset, sources)
