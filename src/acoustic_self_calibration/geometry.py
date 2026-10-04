from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np


def rigid_align(
    estimate: np.ndarray,
    target: np.ndarray,
    *,
    allow_reflection: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align `estimate` to `target` using Kabsch.

    Returns aligned points, rotation R, translation t such that
    aligned ~= estimate @ R + t.
    """
    x = np.asarray(estimate, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("estimate and target must both have shape (N, 3)")

    xc = x.mean(axis=0)
    yc = y.mean(axis=0)
    x0 = x - xc
    y0 = y - yc

    u, _, vt = np.linalg.svd(x0.T @ y0)
    r = u @ vt
    if not allow_reflection and np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    t = yc - xc @ r
    return x @ r + t, r, t


def apply_rigid(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ rotation + translation


def rms_position_error(estimate: np.ndarray, target: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean(np.sum((estimate - target) ** 2, axis=-1))))


@overload
def canonicalize_scene(
    microphones: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def canonicalize_scene(
    microphones: np.ndarray,
    sources: None = None,
) -> tuple[np.ndarray, None]: ...


def canonicalize_scene(
    microphones: np.ndarray,
    sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Transform a scene into the optimizer's canonical coordinate gauge."""
    m = np.asarray(microphones, dtype=float)
    if m.shape[0] < 4 or m.shape[1] != 3:
        raise ValueError("At least 4 3D microphones are required")

    origin = m[0]
    a = m[1] - origin
    ex = a / np.linalg.norm(a)

    b = m[2] - origin
    b_perp = b - np.dot(b, ex) * ex
    ey = b_perp / np.linalg.norm(b_perp)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    if np.dot(m[3] - origin, ez) < 0:
        ez = -ez

    basis = np.stack([ex, ey, ez], axis=1)

    def tr(points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - origin) @ basis

    mc = tr(m)
    sc = None if sources is None else tr(sources)
    return mc, sc


@dataclass(frozen=True)
class CoordinateGauge:
    """Conditioning-aware Euclidean frame without reordering microphone IDs."""

    origin_index: int
    x_axis_index: int
    plane_index: int
    orientation_index: int | None
    origin_m: np.ndarray
    basis: np.ndarray
    affine_rank: int


def _lexicographic_coordinate_key(point: np.ndarray) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def select_coordinate_gauge(
    microphones: np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
) -> CoordinateGauge:
    """Choose a deterministic, well-conditioned planar/3-D coordinate gauge.

    Selection depends on geometry rather than input order: the farthest pair defines
    the x axis, the point farthest from that line defines the gauge plane, and the point
    farthest from the plane fixes 3-D handedness when available. Original microphone
    ordering is preserved; the selected pivot indices are returned explicitly.
    """
    m = np.asarray(microphones, dtype=float)
    if m.ndim != 2 or m.shape[1] != 3 or len(m) < 4:
        raise ValueError("At least 4 3D microphones are required")
    if not np.all(np.isfinite(m)):
        raise ValueError("microphones contains non-finite values")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")

    pair_candidates: list[tuple[float, tuple[float, ...], int, int]] = []
    for a in range(len(m) - 1):
        for b in range(a + 1, len(m)):
            distance = float(np.linalg.norm(m[b] - m[a]))
            endpoint_keys = sorted(
                (_lexicographic_coordinate_key(m[a]), _lexicographic_coordinate_key(m[b]))
            )
            pair_candidates.append(
                (
                    distance,
                    endpoint_keys[0] + endpoint_keys[1],
                    a,
                    b,
                )
            )
    max_distance = max(candidate[0] for candidate in pair_candidates)
    scale = max(max_distance, 1.0)
    distance_tolerance = relative_tolerance * scale
    farthest_pairs = [
        candidate
        for candidate in pair_candidates
        if max_distance - candidate[0] <= distance_tolerance
    ]
    _, _, a, b = min(farthest_pairs, key=lambda candidate: candidate[1])
    if _lexicographic_coordinate_key(m[b]) < _lexicographic_coordinate_key(m[a]):
        a, b = b, a

    origin = m[a]
    x_vector = m[b] - origin
    x_norm = float(np.linalg.norm(x_vector))
    if x_norm <= distance_tolerance:
        raise ValueError("microphone geometry has no stable baseline")
    ex = x_vector / x_norm

    plane_candidates: list[tuple[float, tuple[float, float, float], int, np.ndarray]] = []
    for index in range(len(m)):
        if index in (a, b):
            continue
        vector = m[index] - origin
        perpendicular = vector - np.dot(vector, ex) * ex
        distance = float(np.linalg.norm(perpendicular))
        plane_candidates.append(
            (distance, _lexicographic_coordinate_key(m[index]), index, perpendicular)
        )
    max_perpendicular = max(candidate[0] for candidate in plane_candidates)
    if max_perpendicular <= distance_tolerance:
        raise ValueError("microphone geometry is collinear; cannot define a plane")
    plane_choices = [
        candidate
        for candidate in plane_candidates
        if max_perpendicular - candidate[0] <= distance_tolerance
    ]
    _, _, plane_index, plane_perpendicular = min(
        plane_choices,
        key=lambda candidate: candidate[1],
    )
    ey = plane_perpendicular / np.linalg.norm(plane_perpendicular)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    signed_plane_distances = (m - origin) @ ez
    abs_plane_distances = np.abs(signed_plane_distances)
    max_plane_distance = float(np.max(abs_plane_distances))
    orientation_index: int | None
    if max_plane_distance <= distance_tolerance:
        orientation_index = None
        dominant = int(np.argmax(np.abs(ez)))
        if ez[dominant] < 0.0:
            ez = -ez
        affine_rank = 2
    else:
        orientation_candidates = np.flatnonzero(
            max_plane_distance - abs_plane_distances <= distance_tolerance
        )
        orientation_index = min(
            (int(index) for index in orientation_candidates),
            key=lambda index: _lexicographic_coordinate_key(m[index]),
        )
        if signed_plane_distances[orientation_index] < 0.0:
            ez = -ez
        affine_rank = 3

    basis = np.stack([ex, ey, ez], axis=1)
    origin_copy = np.array(origin, dtype=float, copy=True)
    basis_copy = np.array(basis, dtype=float, copy=True)
    origin_copy.setflags(write=False)
    basis_copy.setflags(write=False)
    return CoordinateGauge(
        origin_index=a,
        x_axis_index=b,
        plane_index=plane_index,
        orientation_index=orientation_index,
        origin_m=origin_copy,
        basis=basis_copy,
        affine_rank=affine_rank,
    )


@overload
def canonicalize_scene_conditioned(
    microphones: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, CoordinateGauge]: ...


@overload
def canonicalize_scene_conditioned(
    microphones: np.ndarray,
    sources: None = None,
) -> tuple[np.ndarray, None, CoordinateGauge]: ...


def canonicalize_scene_conditioned(
    microphones: np.ndarray,
    sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, CoordinateGauge]:
    """Canonicalize without assuming fixed microphone indices are well conditioned."""
    m = np.asarray(microphones, dtype=float)
    gauge = select_coordinate_gauge(m)

    def transform(points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - gauge.origin_m) @ gauge.basis

    canonical_microphones = transform(m)
    canonical_sources = None if sources is None else transform(sources)
    return canonical_microphones, canonical_sources, gauge
