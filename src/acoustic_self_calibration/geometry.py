from __future__ import annotations

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
