from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notation import MatrixRankDiagnostics, cross_gram_from_ranges, rank_diagnostics


@dataclass(frozen=True)
class AffineFactorization:
    """Rank-K cross-distance factorization Q=X@Y.T with zero reference rows."""

    dimension: int
    receiver_factors: np.ndarray
    source_factors: np.ndarray
    compacted_cross_gram: np.ndarray
    rank_diagnostics: MatrixRankDiagnostics
    discarded_singular_rms: float
    relative_discarded_energy: float
    reconstruction_rms: float


def factor_corrected_ranges(
    corrected_ranges_m: np.ndarray,
    *,
    dimension: int,
) -> AffineFactorization:
    """Factor corrected receiver-source ranges up to an affine transformation."""
    ranges = np.asarray(corrected_ranges_m, dtype=float)
    if ranges.ndim != 2 or not np.all(np.isfinite(ranges)):
        raise ValueError("corrected_ranges_m must be a finite 2-D matrix")
    if np.any(ranges < 0.0):
        raise ValueError("corrected_ranges_m must be non-negative")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if min(ranges.shape[0] - 1, ranges.shape[1] - 1) < dimension:
        raise ValueError("not enough receivers/events for requested factor dimension")

    cross_gram = cross_gram_from_ranges(ranges)
    u, singular_values, vt = np.linalg.svd(cross_gram, full_matrices=False)
    root = np.sqrt(np.maximum(singular_values[:dimension], 0.0))
    receiver_nonreference = u[:, :dimension] * root[None, :]
    source_nonreference = vt[:dimension].T * root[None, :]

    receiver_factors = np.vstack([np.zeros((1, dimension), dtype=float), receiver_nonreference])
    source_factors = np.vstack([np.zeros((1, dimension), dtype=float), source_nonreference])
    reconstruction = receiver_nonreference @ source_nonreference.T
    reconstruction_rms = float(np.sqrt(np.mean((reconstruction - cross_gram) ** 2)))
    discarded = singular_values[dimension:]
    discarded_rms = 0.0 if discarded.size == 0 else float(np.sqrt(np.mean(discarded * discarded)))
    total_energy = float(np.sum(singular_values * singular_values))
    discarded_energy = float(np.sum(discarded * discarded))
    relative_discarded_energy = 0.0 if total_energy == 0.0 else discarded_energy / total_energy

    rank_info = rank_diagnostics(cross_gram, expected_rank=dimension)
    receiver_copy = np.array(receiver_factors, copy=True)
    source_copy = np.array(source_factors, copy=True)
    gram_copy = np.array(cross_gram, copy=True)
    receiver_copy.setflags(write=False)
    source_copy.setflags(write=False)
    gram_copy.setflags(write=False)

    return AffineFactorization(
        dimension=dimension,
        receiver_factors=receiver_copy,
        source_factors=source_copy,
        compacted_cross_gram=gram_copy,
        rank_diagnostics=rank_info,
        discarded_singular_rms=discarded_rms,
        relative_discarded_energy=relative_discarded_energy,
        reconstruction_rms=reconstruction_rms,
    )
