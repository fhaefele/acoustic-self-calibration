"""A single fixed cross deformation preserves every event, not just one call."""

import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_self_calibration.geometry import rigid_align, rms_position_error


@pytest.mark.parametrize("angle_degrees", [60, 75, 105, 120])
def test_all_myotis_calls_admit_one_fixed_alternative_array(angle_degrees):
    path = Path(__file__).resolve().parents[1] / "data/myotis/myotis_gt.json"
    scene = json.loads(path.read_text())["scene"]
    microphones = np.asarray(scene["microphones"]["positions_m"])
    sources = np.asarray(scene["source"]["positions_m"])
    theta = np.deg2rad(angle_degrees)
    transform = np.array([[1.0, np.cos(theta)], [0.0, np.sin(theta)]])
    alternative_mics = microphones.copy()
    alternative_mics[:, [0, 2]] = microphones[:, [0, 2]] @ transform.T
    alternative_sources = sources.copy()
    alternative_sources[:, [0, 2]] = sources[:, [0, 2]] @ np.linalg.inv(transform)
    heights_squared = np.sum(sources**2, axis=1) - np.sum(
        alternative_sources[:, [0, 2]] ** 2, axis=1
    )
    assert np.all(heights_squared > 0)
    alternative_sources[:, 1] = np.sqrt(heights_squared)
    ranges = np.linalg.norm(sources[:, None] - microphones[None], axis=2)
    alternative_ranges = np.linalg.norm(
        alternative_sources[:, None] - alternative_mics[None], axis=2
    )
    assert ranges.shape == (41, 12)
    np.testing.assert_allclose(alternative_ranges, ranges, atol=1e-10, rtol=0)
    np.testing.assert_allclose(alternative_mics[3], 0, atol=1e-12)
    aligned, _, _ = rigid_align(alternative_mics, microphones)
    assert rms_position_error(aligned, microphones) > 0.05
    assert np.all(alternative_sources[:, 1] > 0)
