from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .geometry import canonicalize_scene
from .initialization import (
    free_microphone_parameter_count,
    low_rank_initial_scene,
    pack_geometry,
    unpack_geometry,
)


@dataclass(frozen=True)
class DistancePrior:
    """Gaussian prior on the distance between two microphones."""

    microphone_a: int
    microphone_b: int
    distance_m: float
    sigma_m: float


@dataclass(frozen=True)
class BayesianCalibrationResult:
    """MAP estimate plus a local Laplace approximation for global parameters."""

    microphone_positions: np.ndarray
    source_positions: np.ndarray
    speed_of_sound: float
    clock_offsets_s: np.ndarray
    clock_drifts: np.ndarray
    microphone_position_std_m: np.ndarray | None
    speed_of_sound_std: float | None
    clock_offset_std_s: np.ndarray | None
    clock_drift_std: np.ndarray | None
    rms_tdoa_residual_s: float
    normalized_data_rms: float
    negative_log_posterior: float
    success: bool
    message: str
    nfev: int


def tdoa_sigma_from_confidence(
    confidence: np.ndarray,
    sample_rate: float,
    *,
    best_sigma_samples: float = 0.35,
    worst_sigma_samples: float = 4.0,
    floor_quantile: float = 0.1,
    ceiling_quantile: float = 0.9,
) -> np.ndarray:
    """Convert relative GCC peak confidence to timing standard deviations.

    GCC peak magnitude is not itself a calibrated probability. This mapping is a
    pragmatic heteroscedastic model and should eventually be replaced by a delay
    error model calibrated on the actual hardware and acoustic environment.
    """
    values = np.asarray(confidence, dtype=float)
    if values.ndim != 2:
        raise ValueError("confidence must have shape (frames, measurements)")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("confidence contains no finite values")
    lo = float(np.quantile(finite, floor_quantile))
    hi = float(np.quantile(finite, ceiling_quantile))
    quality = (
        np.ones_like(values) if hi <= lo + 1e-15 else np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    )
    sigma_samples = worst_sigma_samples - quality * (worst_sigma_samples - best_sigma_samples)
    return sigma_samples / float(sample_rate)


def _mic_parameter_indices(index: int) -> tuple[int, ...]:
    if index == 0:
        return ()
    if index == 1:
        return (0,)
    if index == 2:
        return (1, 2)
    if index == 3:
        return (3, 4, 5)
    start = 6 + 3 * (index - 4)
    return (start, start + 1, start + 2)


def _validate_pairs(
    microphone_pairs: Sequence[tuple[int, int]] | None,
    mic_count: int,
) -> tuple[tuple[int, int], ...]:
    pairs = (
        tuple((0, index) for index in range(1, mic_count))
        if microphone_pairs is None
        else tuple((int(a), int(b)) for a, b in microphone_pairs)
    )
    if not pairs:
        raise ValueError("microphone_pairs cannot be empty")
    if len(set(pairs)) != len(pairs):
        raise ValueError("microphone_pairs contains duplicates")
    for a, b in pairs:
        if a == b or not (0 <= a < mic_count) or not (0 <= b < mic_count):
            raise ValueError(f"invalid microphone pair {(a, b)}")
    return pairs


def _reference_star(
    tau: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    mic_count: int,
) -> np.ndarray:
    """Extract 0->j measurements needed by the low-rank initializer."""
    out = np.empty((tau.shape[0], mic_count - 1), dtype=float)
    for microphone in range(1, mic_count):
        for column, (a, b) in enumerate(pairs):
            if (a, b) == (0, microphone):
                out[:, microphone - 1] = tau[:, column]
                break
            if (a, b) == (microphone, 0):
                out[:, microphone - 1] = -tau[:, column]
                break
        else:
            raise ValueError(
                "Automatic initialization requires a TDOA edge between mic 0 and "
                f"mic {microphone}; supply initial geometry or include the reference star"
            )
    return out


def _metric_scale_is_anchored(priors: Sequence[DistancePrior]) -> bool:
    return any(prior.distance_m > 0 and prior.sigma_m > 0 for prior in priors)


def calibrate_bayesian(
    tdoa_matrix: np.ndarray,
    frame_times_s: np.ndarray,
    mic_count: int,
    *,
    tdoa_sigma_s: float | np.ndarray,
    microphone_pairs: Sequence[tuple[int, int]] | None = None,
    speed_of_sound: float = 343.0,
    estimate_speed_of_sound: bool = False,
    speed_of_sound_prior_mean: float = 343.0,
    speed_of_sound_prior_sigma: float = 4.0,
    estimate_clock_offsets: bool = False,
    clock_offset_prior_sigma_s: float = 250e-6,
    estimate_clock_drifts: bool = False,
    clock_drift_prior_sigma: float = 30e-6,
    distance_priors: Sequence[DistancePrior] = (),
    motion_velocity_change_sigma_mps: float | None = 1.5,
    initial_microphones: np.ndarray | None = None,
    initial_sources: np.ndarray | None = None,
    initial_clock_offsets_s: np.ndarray | None = None,
    initial_clock_drifts: np.ndarray | None = None,
    likelihood: str = "cauchy",
    position_bound_m: float = 30.0,
    max_nfev: int = 4000,
    compute_laplace_uncertainty: bool = True,
) -> BayesianCalibrationResult:
    """Joint Bayesian/MAP calibration of microphone geometry and source motion.

    TDOA column `(a, b)` means `arrival[b] - arrival[a]`. The data model is
    heteroscedastic; optional Gaussian factors model smooth source motion, clock
    offset/drift, sound speed, and known microphone distances. Robust mode uses
    Cauchy IRLS weights only on TDOA data factors so Gaussian priors stay Gaussian.
    """
    tau = np.asarray(tdoa_matrix, dtype=float)
    times = np.asarray(frame_times_s, dtype=float).reshape(-1)
    if tau.ndim != 2:
        raise ValueError("tdoa_matrix must have shape (frames, measurements)")
    frame_count, measurement_count = tau.shape
    pairs = _validate_pairs(microphone_pairs, mic_count)
    if measurement_count != len(pairs):
        raise ValueError("tdoa_matrix columns must match microphone_pairs")
    if len(times) != frame_count or np.any(np.diff(times) <= 0):
        raise ValueError("frame_times_s must match frames and be strictly increasing")
    if frame_count < 4 or mic_count < 4:
        raise ValueError("At least 4 microphones and 4 source frames are required")
    if likelihood not in {"gaussian", "cauchy"}:
        raise ValueError("likelihood must be 'gaussian' or 'cauchy'")
    if estimate_speed_of_sound and not _metric_scale_is_anchored(distance_priors):
        raise ValueError(
            "Estimating speed of sound requires at least one DistancePrior to fix metric scale"
        )

    sigma = np.asarray(tdoa_sigma_s, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full_like(tau, float(sigma))
    if sigma.shape != tau.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("tdoa_sigma_s must be positive, finite, and scalar or TDOA-shaped")

    if initial_microphones is None and initial_sources is None:
        star = _reference_star(tau, pairs, mic_count)
        microphones0, sources0 = low_rank_initial_scene(speed_of_sound * star, position_bound_m)
    elif initial_microphones is not None and initial_sources is not None:
        microphones0, sources0 = canonicalize_scene(initial_microphones, initial_sources)
    else:
        raise ValueError("Provide both initial_microphones and initial_sources, or neither")

    mic_parameter_count = free_microphone_parameter_count(mic_count)
    geometry_count = mic_parameter_count + 3 * frame_count
    cursor = geometry_count
    log_c_index = cursor if estimate_speed_of_sound else None
    cursor += int(estimate_speed_of_sound)
    offset_slice = slice(cursor, cursor + mic_count - 1) if estimate_clock_offsets else None
    cursor += mic_count - 1 if estimate_clock_offsets else 0
    drift_slice = slice(cursor, cursor + mic_count - 1) if estimate_clock_drifts else None
    cursor += mic_count - 1 if estimate_clock_drifts else 0
    parameter_count = cursor

    offsets0 = np.zeros(mic_count, dtype=float)
    if initial_clock_offsets_s is not None:
        candidate = np.asarray(initial_clock_offsets_s, dtype=float)
        if candidate.shape != (mic_count,):
            raise ValueError("initial_clock_offsets_s must have shape (mic_count,)")
        offsets0 = candidate - candidate[0]
    drifts0 = np.zeros(mic_count, dtype=float)
    if initial_clock_drifts is not None:
        candidate = np.asarray(initial_clock_drifts, dtype=float)
        if candidate.shape != (mic_count,):
            raise ValueError("initial_clock_drifts must have shape (mic_count,)")
        drifts0 = candidate - candidate[0]

    initial_parts = [pack_geometry(microphones0, sources0)]
    if estimate_speed_of_sound:
        initial_parts.append(np.array([np.log(speed_of_sound)]))
    if estimate_clock_offsets:
        initial_parts.append(offsets0[1:])
    if estimate_clock_drifts:
        initial_parts.append(drifts0[1:])
    x0 = np.concatenate(initial_parts)

    lower = np.full(parameter_count, -np.inf)
    upper = np.full(parameter_count, np.inf)
    lower[:geometry_count] = -position_bound_m
    upper[:geometry_count] = position_bound_m
    for index in (0, 2, 5):
        lower[index] = 1e-4
    if log_c_index is not None:
        lower[log_c_index] = np.log(250.0)
        upper[log_c_index] = np.log(450.0)

    centered_times = times - float(np.mean(times))
    dt = np.diff(times)

    def decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        microphones, sources = unpack_geometry(x[:geometry_count], mic_count, frame_count)
        c = float(np.exp(x[log_c_index])) if log_c_index is not None else float(speed_of_sound)
        offsets = np.zeros(mic_count)
        if offset_slice is not None:
            offsets[1:] = x[offset_slice]
        drifts = np.zeros(mic_count)
        if drift_slice is not None:
            drifts[1:] = x[drift_slice]
        return microphones, sources, c, offsets, drifts

    def factor_residuals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        microphones, sources, c, offsets, drifts = decode(x)
        prediction = np.empty_like(tau)
        for column, (a, b) in enumerate(pairs):
            da = np.linalg.norm(sources - microphones[a], axis=1)
            db = np.linalg.norm(sources - microphones[b], axis=1)
            prediction[:, column] = (
                (db - da) / c + offsets[b] - offsets[a] + (drifts[b] - drifts[a]) * centered_times
            )
        standardized = (prediction - tau) / sigma

        priors: list[np.ndarray] = []
        if motion_velocity_change_sigma_mps is not None and frame_count >= 3:
            velocity = np.diff(sources, axis=0) / dt[:, None]
            acceleration_like = velocity[1:] - velocity[:-1]
            priors.append((acceleration_like / float(motion_velocity_change_sigma_mps)).reshape(-1))
        if log_c_index is not None:
            priors.append(np.array([(c - speed_of_sound_prior_mean) / speed_of_sound_prior_sigma]))
        if offset_slice is not None:
            priors.append(offsets[1:] / clock_offset_prior_sigma_s)
        if drift_slice is not None:
            priors.append(drifts[1:] / clock_drift_prior_sigma)
        for prior in distance_priors:
            if prior.sigma_m <= 0:
                raise ValueError("DistancePrior sigma_m must be positive")
            distance = np.linalg.norm(
                microphones[prior.microphone_a] - microphones[prior.microphone_b]
            )
            priors.append(np.array([(distance - prior.distance_m) / prior.sigma_m]))
        return standardized, np.concatenate(priors) if priors else np.empty(0)

    data_rows = frame_count * measurement_count
    motion_rows = (
        3 * (frame_count - 2)
        if motion_velocity_change_sigma_mps is not None and frame_count >= 3
        else 0
    )
    prior_rows = (
        motion_rows
        + int(log_c_index is not None)
        + (mic_count - 1 if offset_slice is not None else 0)
        + (mic_count - 1 if drift_slice is not None else 0)
        + len(distance_priors)
    )
    sparsity = lil_matrix((data_rows + prior_rows, parameter_count), dtype=int)
    for frame in range(frame_count):
        source_columns = range(mic_parameter_count + 3 * frame, mic_parameter_count + 3 * frame + 3)
        for pair_index, (a, b) in enumerate(pairs):
            row = frame * measurement_count + pair_index
            sparsity[row, list(source_columns)] = 1
            for microphone in (a, b):
                for column in _mic_parameter_indices(microphone):
                    sparsity[row, column] = 1
            if log_c_index is not None:
                sparsity[row, log_c_index] = 1
            if offset_slice is not None:
                if a:
                    sparsity[row, offset_slice.start + a - 1] = 1
                if b:
                    sparsity[row, offset_slice.start + b - 1] = 1
            if drift_slice is not None:
                if a:
                    sparsity[row, drift_slice.start + a - 1] = 1
                if b:
                    sparsity[row, drift_slice.start + b - 1] = 1

    row = data_rows
    if motion_rows:
        for frame in range(1, frame_count - 1):
            columns = []
            for source_frame in (frame - 1, frame, frame + 1):
                start = mic_parameter_count + 3 * source_frame
                columns.extend(range(start, start + 3))
            sparsity[row : row + 3, columns] = 1
            row += 3
    if log_c_index is not None:
        sparsity[row, log_c_index] = 1
        row += 1
    if offset_slice is not None:
        for column in range(offset_slice.start, offset_slice.stop):
            sparsity[row, column] = 1
            row += 1
    if drift_slice is not None:
        for column in range(drift_slice.start, drift_slice.stop):
            sparsity[row, column] = 1
            row += 1
    for prior in distance_priors:
        for microphone in (prior.microphone_a, prior.microphone_b):
            for column in _mic_parameter_indices(microphone):
                sparsity[row, column] = 1
        row += 1
    sparsity = sparsity.tocsr()

    data_weights = np.ones_like(tau)

    def residual(x: np.ndarray) -> np.ndarray:
        data, priors = factor_residuals(x)
        return np.concatenate([(np.sqrt(data_weights) * data).reshape(-1), priors])

    current = np.clip(x0, lower + 1e-12, upper - 1e-12)
    fit = None
    total_nfev = 0
    outer_iterations = 1 if likelihood == "gaussian" else 6
    for _ in range(outer_iterations):
        solver_kwargs: dict[str, object] = {}
        if parameter_count > 100:
            solver_kwargs["jac_sparsity"] = sparsity
            solver_kwargs["tr_solver"] = "lsmr"
        fit = least_squares(
            residual,
            current,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            **solver_kwargs,
        )
        current = fit.x
        total_nfev += int(fit.nfev)
        if likelihood == "gaussian":
            break
        standardized, _ = factor_residuals(current)
        new_weights = np.clip(1.0 / (1.0 + standardized * standardized), 1e-6, 1.0)
        if float(np.max(np.abs(new_weights - data_weights))) < 1e-3:
            data_weights[:] = new_weights
            break
        data_weights[:] = new_weights

    assert fit is not None
    microphones, sources, c, offsets, drifts = decode(fit.x)
    normalized, priors = factor_residuals(fit.x)
    raw = normalized * sigma

    microphone_std = None
    c_std = None
    offset_std = None
    drift_std = None
    if compute_laplace_uncertainty:
        jacobian = fit.jac.toarray() if hasattr(fit.jac, "toarray") else np.asarray(fit.jac)
        hessian = jacobian.T @ jacobian
        source_indices = np.arange(mic_parameter_count, geometry_count)
        global_indices = np.concatenate(
            [
                np.arange(mic_parameter_count),
                np.arange(geometry_count, parameter_count),
            ]
        )
        hgg = hessian[np.ix_(global_indices, global_indices)]
        if source_indices.size:
            hgq = hessian[np.ix_(global_indices, source_indices)]
            hqq = hessian[np.ix_(source_indices, source_indices)]
            information = hgg - hgq @ np.linalg.pinv(hqq, rcond=1e-10) @ hgq.T
        else:
            information = hgg
        diagonal = np.clip(np.diag(np.linalg.pinv(information, rcond=1e-10)), 0.0, np.inf)
        microphone_std = np.zeros((mic_count, 3))
        for microphone in range(1, mic_count):
            for local_index, parameter_index in enumerate(_mic_parameter_indices(microphone)):
                if microphone == 1:
                    dimension = 0
                elif microphone == 2:
                    dimension = (0, 1)[local_index]
                elif microphone == 3:
                    dimension = (0, 1, 2)[local_index]
                else:
                    dimension = local_index
                microphone_std[microphone, dimension] = np.sqrt(diagonal[parameter_index])
        nuisance_cursor = mic_parameter_count
        if log_c_index is not None:
            c_std = float(c * np.sqrt(diagonal[nuisance_cursor]))
            nuisance_cursor += 1
        if offset_slice is not None:
            offset_std = np.zeros(mic_count)
            offset_std[1:] = np.sqrt(diagonal[nuisance_cursor : nuisance_cursor + mic_count - 1])
            nuisance_cursor += mic_count - 1
        if drift_slice is not None:
            drift_std = np.zeros(mic_count)
            drift_std[1:] = np.sqrt(diagonal[nuisance_cursor : nuisance_cursor + mic_count - 1])

    data_cost = (
        0.5 * np.sum(normalized * normalized)
        if likelihood == "gaussian"
        else 0.5 * np.sum(np.log1p(normalized * normalized))
    )
    return BayesianCalibrationResult(
        microphone_positions=microphones,
        source_positions=sources,
        speed_of_sound=c,
        clock_offsets_s=offsets,
        clock_drifts=drifts,
        microphone_position_std_m=microphone_std,
        speed_of_sound_std=c_std,
        clock_offset_std_s=offset_std,
        clock_drift_std=drift_std,
        rms_tdoa_residual_s=float(np.sqrt(np.mean(raw * raw))),
        normalized_data_rms=float(np.sqrt(np.mean(normalized * normalized))),
        negative_log_posterior=float(data_cost + 0.5 * np.sum(priors * priors)),
        success=bool(fit.success),
        message=str(fit.message),
        nfev=total_nfev,
    )
