from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

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


def _affine_metric_starts() -> list[np.ndarray]:
    """Deterministic starts for the affine-to-Euclidean metric upgrade."""
    starts = [np.eye(3)]
    starts.extend(np.diag(values) for values in sorted(set(permutations((0.5, 1.0, 4.0)))))
    return starts


def _range_bounds(
    differences: np.ndarray,
    position_bound_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    lower = np.maximum(0.02, -np.min(differences, axis=0) + 0.02)
    upper = np.maximum(lower + 0.5, 2.0 * float(position_bound_m))
    aperture = max(0.25, float(np.percentile(np.abs(differences), 90)))
    return lower, upper, aperture


def _centered_rank_objective_and_gradient(
    differences: np.ndarray,
    center_m: np.ndarray,
    center_t: np.ndarray,
    reference_ranges: np.ndarray,
) -> tuple[float, np.ndarray]:
    eps = 1e-12
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


def _fit_global_reference_ranges(
    differences: np.ndarray,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    maxiter: int = 300,
) -> tuple[float, np.ndarray]:
    microphone_count, event_count = differences.shape
    center_m = (
        np.eye(microphone_count) - np.ones((microphone_count, microphone_count)) / microphone_count
    )
    center_t = np.eye(event_count) - np.ones((event_count, event_count)) / event_count

    def objective(reference_ranges: np.ndarray) -> tuple[float, np.ndarray]:
        return _centered_rank_objective_and_gradient(
            differences,
            center_m,
            center_t,
            reference_ranges,
        )

    fit = minimize(
        objective,
        np.clip(initial, lower, upper),
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9},
    )
    return float(fit.fun), np.asarray(fit.x, dtype=float)


def _reference_compaction(
    differences: np.ndarray,
    reference_ranges: np.ndarray,
    anchor_event: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Åström-style double compaction of squared ranges.

    ``differences`` contains range differences to microphone 0. If ``r_j`` is the
    unknown source-to-mic-0 range, then ``D^2 + 2 D r`` is the squared-distance
    difference to mic 0. Subtracting one event column removes the remaining row
    offset. The resulting matrix has rank at most three for a 3-D Euclidean scene.
    """
    adjusted = differences * differences + 2.0 * differences * reference_ranges[None, :]
    columns = np.array(
        [event for event in range(differences.shape[1]) if event != anchor_event],
        dtype=int,
    )
    compacted = adjusted[1:, columns] - adjusted[1:, [anchor_event]]
    return compacted, columns


def _reference_compaction_score(
    differences: np.ndarray,
    reference_ranges: np.ndarray,
    anchor_event: int,
) -> float:
    compacted, _ = _reference_compaction(differences, reference_ranges, anchor_event)
    singular = np.linalg.svd(compacted, compute_uv=False)
    tail = singular[min(3, len(singular)) :]
    return float(np.dot(tail, tail) / (np.dot(singular, singular) + 1e-12))


def _fit_subset_offsets(
    differences: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Numerically solve a small rank-three offset problem on one minimal-ish subset."""

    def objective(reference_ranges: np.ndarray) -> float:
        return _reference_compaction_score(differences, reference_ranges, 0)

    fit = minimize(
        objective,
        np.clip(initial, lower, upper),
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": 250, "ftol": 1e-13, "gtol": 1e-9},
    )
    return np.asarray(fit.x, dtype=float)


def _extend_subset_offsets(
    differences: np.ndarray,
    microphone_indices: np.ndarray,
    event_indices: np.ndarray,
    subset_ranges: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Extend one subset offset hypothesis to every source event.

    This mirrors the extension step in Åström's TDOA RANSAC code: a rank-three
    basis is estimated from a small subset, then every remaining source offset is
    solved independently by projecting its compacted column onto that basis.
    """
    restricted = differences[microphone_indices]
    anchor_event = int(event_indices[0])
    reference_ranges = np.maximum(lower.copy(), 0.02)
    reference_ranges[event_indices] = subset_ranges

    local = restricted[:, event_indices]
    compacted, _ = _reference_compaction(local, subset_ranges, 0)
    u, singular, _ = np.linalg.svd(compacted, full_matrices=False)
    rank = min(3, int(np.count_nonzero(singular > singular[0] * 1e-10)) if singular.size else 0)
    if rank == 0:
        return reference_ranges
    basis = u[:, :rank]

    anchor_d = restricted[1:, anchor_event]
    anchor_term = anchor_d * anchor_d + 2.0 * anchor_d * reference_ranges[anchor_event]
    known = set(int(index) for index in event_indices)
    for event in range(differences.shape[1]):
        if event in known:
            continue
        d = restricted[1:, event]
        p = d * d - anchor_term
        q = 2.0 * d
        p_orthogonal = p - basis @ (basis.T @ p)
        q_orthogonal = q - basis @ (basis.T @ q)
        denominator = float(np.dot(q_orthogonal, q_orthogonal))
        if denominator <= 1e-14:
            continue
        estimate = -float(np.dot(q_orthogonal, p_orthogonal)) / denominator
        reference_ranges[event] = float(np.clip(estimate, lower[event], upper[event]))
    return reference_ranges


def _deterministic_hypothesis_subsets(
    differences: np.ndarray,
    *,
    max_hypotheses: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    microphone_count, event_count = differences.shape
    if microphone_count < 7 or event_count < 6:
        return []

    rng = np.random.default_rng(0)
    microphone_size = min(7, microphone_count)
    event_size = min(6, event_count)

    mic_sets: list[np.ndarray] = []
    strength_order = np.argsort(np.std(differences[1:], axis=1))[::-1] + 1
    mic_sets.append(np.concatenate([[0], np.sort(strength_order[: microphone_size - 1])]))
    evenly_spaced_mics = np.unique(
        np.round(np.linspace(1, microphone_count - 1, microphone_size - 1)).astype(int)
    )
    if len(evenly_spaced_mics) == microphone_size - 1:
        mic_sets.append(np.concatenate([[0], evenly_spaced_mics]))
    while len(mic_sets) < max_hypotheses:
        selected = np.sort(rng.choice(np.arange(1, microphone_count), microphone_size - 1, replace=False))
        candidate = np.concatenate([[0], selected])
        if not any(np.array_equal(candidate, existing) for existing in mic_sets):
            mic_sets.append(candidate)

    event_sets: list[np.ndarray] = []
    event_sets.append(np.round(np.linspace(0, event_count - 1, event_size)).astype(int))
    while len(event_sets) < max_hypotheses:
        selected = np.sort(rng.choice(np.arange(event_count), event_size, replace=False))
        if not any(np.array_equal(selected, existing) for existing in event_sets):
            event_sets.append(selected)

    return [
        (mic_sets[index % len(mic_sets)], event_sets[(3 * index) % len(event_sets)])
        for index in range(max_hypotheses)
    ]


def _metric_upgrade_scene(
    differences: np.ndarray,
    reference_ranges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ranges = differences + reference_ranges[None, :]
    squared = ranges * ranges
    microphone_count, event_count = differences.shape
    center_m = (
        np.eye(microphone_count) - np.ones((microphone_count, microphone_count)) / microphone_count
    )
    center_t = np.eye(event_count) - np.ones((event_count, event_count)) / event_count
    centered = -0.5 * (center_m @ squared @ center_t)

    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    root = np.sqrt(np.maximum(singular[:3], 0.0))
    left = u[:, :3] * root[None, :]
    right = vt[:3].T * root[None, :]

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

    affine_fits: list[tuple[float, object]] = []
    data_count = squared.size
    for transform0 in _affine_metric_starts():
        affine_initial = np.concatenate([transform0.reshape(-1), np.zeros(3)])
        fit = least_squares(
            affine_residual,
            affine_initial,
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=800,
        )
        residual = affine_residual(fit.x)[:data_count]
        score = float(np.dot(residual, residual))
        if np.isfinite(score):
            affine_fits.append((score, fit))

    if not affine_fits:
        raise ValueError("low-rank metric upgrade failed to produce a finite scene")
    _, affine_fit = min(affine_fits, key=lambda item: item[0])
    microphones, sources, offset = affine_unpack(affine_fit.x)
    return canonicalize_scene(microphones + offset, sources)


def low_rank_initial_scene_hypotheses(
    observed_range_differences: np.ndarray,
    position_bound_m: float,
    *,
    max_scene_hypotheses: int = 4,
    subset_hypotheses: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate several low-rank Euclidean scene hypotheses.

    The construction follows the structure of Åström's robust TDOA solvers:

    1. choose small receiver/source subsets,
    2. solve the unknown per-source reference ranges from a rank-three double
       compaction constraint,
    3. extend each hypothesis to the remaining source events by projection onto
       the recovered rank-three basis,
    4. bundle-refine all reference ranges against the full low-rank constraint,
    5. perform an affine-to-Euclidean metric upgrade.

    We use deterministic numerical subset solves instead of the original generated
    minimal polynomial solvers, which keeps this implementation dependency-free
    while preserving the hypothesis/extension/refinement architecture.
    """
    observed = np.asarray(observed_range_differences, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed_range_differences must have shape (events, microphones-1)")
    event_count, nonreference_count = observed.shape
    microphone_count = nonreference_count + 1
    if microphone_count < 4 or event_count < 4:
        raise ValueError("At least four microphones and four source events are required")
    if max_scene_hypotheses < 1 or subset_hypotheses < 0:
        raise ValueError("hypothesis counts must be positive")

    differences = np.vstack([np.zeros(event_count), observed.T])
    lower, upper, aperture = _range_bounds(differences, position_bound_m)
    range_candidates: list[tuple[float, np.ndarray]] = []

    for multiplier in (0.25, 0.75, 1.5, 3.0):
        initial = np.clip(lower + multiplier * aperture, lower, upper)
        score, reference_ranges = _fit_global_reference_ranges(
            differences,
            initial,
            lower,
            upper,
        )
        if np.isfinite(score):
            range_candidates.append((score, reference_ranges))

    for microphone_indices, event_indices in _deterministic_hypothesis_subsets(
        differences,
        max_hypotheses=subset_hypotheses,
    ):
        subset = differences[np.ix_(microphone_indices, event_indices)]
        subset_lower, subset_upper, subset_aperture = _range_bounds(subset, position_bound_m)
        for multiplier in (0.5, 1.5):
            subset_initial = np.clip(
                subset_lower + multiplier * subset_aperture,
                subset_lower,
                subset_upper,
            )
            subset_ranges = _fit_subset_offsets(
                subset,
                subset_lower,
                subset_upper,
                subset_initial,
            )
            extended = _extend_subset_offsets(
                differences,
                microphone_indices,
                event_indices,
                subset_ranges,
                lower,
                upper,
            )
            score, refined = _fit_global_reference_ranges(
                differences,
                extended,
                lower,
                upper,
                maxiter=220,
            )
            if np.isfinite(score):
                range_candidates.append((score, refined))

    range_candidates.sort(key=lambda item: item[0])
    unique_ranges: list[np.ndarray] = []
    for _, reference_ranges in range_candidates:
        if any(
            np.linalg.norm(reference_ranges - existing)
            <= 1e-3 * max(1.0, np.linalg.norm(existing))
            for existing in unique_ranges
        ):
            continue
        unique_ranges.append(reference_ranges)
        if len(unique_ranges) >= max_scene_hypotheses:
            break

    scenes: list[tuple[np.ndarray, np.ndarray]] = []
    for reference_ranges in unique_ranges:
        try:
            scenes.append(_metric_upgrade_scene(differences, reference_ranges))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not scenes:
        raise ValueError("low-rank hypothesis generation failed to produce a Euclidean scene")
    return scenes


def low_rank_initial_scene(
    observed_range_differences: np.ndarray,
    position_bound_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest low-rank scene hypothesis."""
    return low_rank_initial_scene_hypotheses(
        observed_range_differences,
        position_bound_m,
        max_scene_hypotheses=1,
    )[0]


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
    scale_candidates: Sequence[float] = (1.0, 1.25, 1.6),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build coarse event-calibration starts from TDOA baseline lower bounds."""
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
