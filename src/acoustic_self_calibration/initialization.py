from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize

from .geometry import canonicalize_scene


def pack_geometry(microphones: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Legacy compact geometry packer kept for downstream callers."""
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
    """Legacy compact geometry unpacker kept for downstream callers."""
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
    """Initialize a 3-D scene from range-difference data."""
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


def _classical_mds(distance_m: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=float)
    count = distance.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance * distance) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:3]
    values = np.sqrt(np.maximum(eigenvalues[order], 0.0))
    return eigenvectors[:, order] * values[None, :]


def _localize_sources(
    microphones: np.ndarray,
    tdoa_s: np.ndarray,
    tdoa_sigma_s: np.ndarray,
    microphone_pairs: Sequence[tuple[int, int]],
    speed_of_sound: float,
    position_bound_m: float,
) -> np.ndarray:
    center = microphones.mean(axis=0)
    aperture = max(float(np.max(np.linalg.norm(microphones - center, axis=1))), 0.25)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    starts = [center, *(center + 2.5 * aperture * directions)]

    sources: list[np.ndarray] = []
    for event in range(len(tdoa_s)):
        def residual(source: np.ndarray) -> np.ndarray:
            predicted = np.array(
                [
                    (
                        np.linalg.norm(source - microphones[b])
                        - np.linalg.norm(source - microphones[a])
                    )
                    / speed_of_sound
                    for a, b in microphone_pairs
                ]
            )
            return (predicted - tdoa_s[event]) / tdoa_sigma_s[event]

        event_starts = ([sources[-1]] if sources else []) + starts
        best_cost = np.inf
        best_source = center
        for start in event_starts:
            fit = least_squares(
                residual,
                np.clip(start, -position_bound_m, position_bound_m),
                bounds=(-position_bound_m, position_bound_m),
                max_nfev=100,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            cost = float(np.sum(residual(fit.x) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_source = fit.x
        sources.append(np.asarray(best_source, dtype=float))
    return np.stack(sources)


def event_initial_scene_candidates(
    arrival_delays_s: np.ndarray,
    tdoa_s: np.ndarray,
    tdoa_sigma_s: np.ndarray,
    microphone_pairs: Sequence[tuple[int, int]],
    *,
    speed_of_sound: float,
    position_bound_m: float = 30.0,
    scale_candidates: Sequence[float] = (0.55, 0.75, 1.0),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build coarse event-calibration starts from TDOA baseline lower bounds.

    For a moving source, the maximum observed absolute TDOA for a microphone pair
    is a lower bound on that microphone baseline. Classical MDS on these bounds
    gives a useful coarse array shape without assuming any microphone ordering.
    Several scale hypotheses are retained because finite source coverage generally
    underestimates the true baselines. Source points are then hyperbolically
    localized against each array hypothesis before joint MAP refinement.
    """
    arrival = np.asarray(arrival_delays_s, dtype=float)
    tau = np.asarray(tdoa_s, dtype=float)
    sigma = np.asarray(tdoa_sigma_s, dtype=float)
    if arrival.ndim != 2 or arrival.shape[0] != tau.shape[0]:
        raise ValueError("arrival_delays_s must have shape (events, microphones)")
    microphone_count = arrival.shape[1]
    if microphone_count < 4:
        raise ValueError("At least four microphones are required")

    baselines = np.zeros((microphone_count, microphone_count), dtype=float)
    for a in range(microphone_count):
        for b in range(a + 1, microphone_count):
            lower_bound = speed_of_sound * float(np.max(np.abs(arrival[:, b] - arrival[:, a])))
            baselines[a, b] = lower_bound
            baselines[b, a] = lower_bound
    microphones_base = _classical_mds(baselines)

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for scale in scale_candidates:
        if scale <= 0:
            raise ValueError("scale_candidates must be positive")
        microphones = microphones_base * float(scale)
        sources = _localize_sources(
            microphones,
            tau,
            sigma,
            microphone_pairs,
            speed_of_sound,
            position_bound_m,
        )
        candidates.append((microphones, sources))
    return candidates
