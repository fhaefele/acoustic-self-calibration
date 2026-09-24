from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustResidualSummary:
    independent_coordinate_count: int
    rms_s: float
    median_abs_s: float
    max_abs_s: float
    normalized_huber_score: float
    inlier_fraction: float


def huber_loss(values: np.ndarray, *, delta: float = 1.5) -> np.ndarray:
    """Elementwise Huber loss in normalized residual units."""
    z = np.asarray(values, dtype=float)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    absolute = np.abs(z)
    return np.where(
        absolute <= delta,
        0.5 * z * z,
        delta * (absolute - 0.5 * delta),
    )


def summarize_independent_residuals(
    residual_s: np.ndarray,
    sigma_s: np.ndarray,
    valid: np.ndarray,
    *,
    huber_delta: float = 1.5,
    sigma_floor_s: float = 1e-7,
    inlier_sigma: float = 3.0,
) -> RobustResidualSummary:
    """Score independent residual coordinates with capped heteroscedastic weights."""
    residual = np.asarray(residual_s, dtype=float)
    sigma = np.asarray(sigma_s, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if residual.shape != sigma.shape or residual.shape != mask.shape:
        raise ValueError("residual_s, sigma_s, and valid must have the same shape")
    if sigma_floor_s <= 0.0 or inlier_sigma <= 0.0:
        raise ValueError("sigma_floor_s and inlier_sigma must be positive")
    if not np.any(mask):
        raise ValueError("at least one valid residual is required")
    if np.any(~np.isfinite(residual[mask])):
        raise ValueError("valid residuals must be finite")
    if np.any(~np.isfinite(sigma[mask])) or np.any(sigma[mask] <= 0.0):
        raise ValueError("valid sigma values must be finite and positive")

    values = residual[mask]
    effective_sigma = np.maximum(sigma[mask], sigma_floor_s)
    normalized = values / effective_sigma
    absolute = np.abs(values)
    return RobustResidualSummary(
        independent_coordinate_count=int(values.size),
        rms_s=float(np.sqrt(np.mean(values * values))),
        median_abs_s=float(np.median(absolute)),
        max_abs_s=float(np.max(absolute)),
        normalized_huber_score=float(np.mean(huber_loss(normalized, delta=huber_delta))),
        inlier_fraction=float(np.mean(np.abs(normalized) <= inlier_sigma)),
    )


def whiten_residual_block(
    residual_s: np.ndarray,
    covariance_s2: np.ndarray,
    *,
    relative_eigenvalue_floor: float = 1e-10,
) -> np.ndarray:
    """Whiten one correlated residual block on its supported covariance subspace."""
    residual = np.asarray(residual_s, dtype=float).reshape(-1)
    covariance = np.asarray(covariance_s2, dtype=float)
    if covariance.shape != (len(residual), len(residual)):
        raise ValueError("covariance_s2 must be square and match residual_s")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(covariance)):
        raise ValueError("whitening inputs must be finite")
    return (
        covariance_whitener(covariance, relative_eigenvalue_floor=relative_eigenvalue_floor)
        @ residual
    )


def covariance_whitener(
    covariance_s2: np.ndarray,
    *,
    relative_eigenvalue_floor: float = 1e-10,
) -> np.ndarray:
    """Build a reusable transform onto the supported covariance subspace."""
    covariance = np.asarray(covariance_s2, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_s2 must be square")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("whitening inputs must be finite")
    if relative_eigenvalue_floor <= 0.0:
        raise ValueError("relative_eigenvalue_floor must be positive")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(np.max(eigenvalues))
    if maximum <= 0.0:
        raise ValueError("covariance has no positive supported subspace")
    keep = eigenvalues > relative_eigenvalue_floor * maximum
    if not np.any(keep):
        raise ValueError("covariance supported subspace is empty")
    return eigenvectors[:, keep].T / np.sqrt(eigenvalues[keep])[:, None]


def conditional_covariance(
    covariance_s2: np.ndarray,
    target_indices: np.ndarray,
    fitted_indices: np.ndarray,
) -> np.ndarray:
    """Return Cov(target | fitted) using a Schur complement."""
    covariance = np.asarray(covariance_s2, dtype=float)
    target = np.asarray(target_indices, dtype=int).reshape(-1)
    fitted = np.asarray(fitted_indices, dtype=int).reshape(-1)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_s2 must be square")
    if target.size == 0:
        raise ValueError("target_indices cannot be empty")
    if np.any(target < 0) or np.any(target >= covariance.shape[0]):
        raise ValueError("target_indices out of range")
    if np.any(fitted < 0) or np.any(fitted >= covariance.shape[0]):
        raise ValueError("fitted_indices out of range")
    target_block = covariance[np.ix_(target, target)]
    if fitted.size == 0:
        return np.array(target_block, copy=True)
    fit_block = covariance[np.ix_(fitted, fitted)]
    cross = covariance[np.ix_(target, fitted)]
    if (
        not np.all(np.isfinite(target_block))
        or not np.all(np.isfinite(fit_block))
        or not np.all(np.isfinite(cross))
    ):
        raise ValueError("selected covariance blocks must be finite")
    conditional = target_block - cross @ np.linalg.pinv(fit_block) @ cross.T
    return 0.5 * (conditional + conditional.T)
