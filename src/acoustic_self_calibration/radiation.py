from __future__ import annotations

import numpy as np


def radiation_gain(cos_theta: np.ndarray, pattern: str = "omni") -> np.ndarray:
    """Return non-negative source radiation gain.

    Parameters
    ----------
    cos_theta:
        Cosine of angle between source forward direction and source->receiver ray.
    pattern:
        "omni", "cardioid", "hypercardioid", or "dipole".
    """
    x = np.clip(np.asarray(cos_theta, dtype=float), -1.0, 1.0)
    pattern = pattern.lower()

    if pattern == "omni":
        return np.ones_like(x)
    if pattern == "cardioid":
        return 0.5 * (1.0 + x)
    if pattern == "hypercardioid":
        return np.abs(0.25 + 0.75 * x)
    if pattern == "dipole":
        return np.abs(x)

    raise ValueError(f"Unknown radiation pattern: {pattern!r}")
