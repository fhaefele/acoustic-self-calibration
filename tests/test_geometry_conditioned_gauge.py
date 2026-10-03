import numpy as np
import pytest

from acoustic_self_calibration.geometry import (
    canonicalize_scene_conditioned,
    select_coordinate_gauge,
)
from acoustic_self_calibration.simulation import myotis_cross_microphones


def test_conditioned_gauge_handles_cross_with_collinear_leading_microphones() -> None:
    microphones = myotis_cross_microphones()
    canonical, _, gauge = canonicalize_scene_conditioned(microphones)

    assert gauge.affine_rank == 2
    assert gauge.orientation_index is None
    assert np.allclose(canonical[gauge.origin_index], 0.0, atol=1e-12)
    assert abs(canonical[gauge.x_axis_index, 1]) < 1e-12
    assert abs(canonical[gauge.x_axis_index, 2]) < 1e-12
    assert abs(canonical[gauge.plane_index, 2]) < 1e-12
    assert np.max(np.abs(canonical[:, 2])) < 1e-12


def test_conditioned_gauge_is_input_permutation_invariant() -> None:
    microphones = myotis_cross_microphones()
    canonical, _, gauge = canonicalize_scene_conditioned(microphones)

    permutation = np.array([7, 2, 10, 0, 11, 4, 5, 1, 9, 8, 6, 3])
    permuted = microphones[permutation]
    permuted_canonical, _, permuted_gauge = canonicalize_scene_conditioned(permuted)

    restored = np.empty_like(permuted_canonical)
    restored[permutation] = permuted_canonical
    assert np.allclose(restored, canonical, atol=1e-12)
    assert np.allclose(permuted_gauge.origin_m, gauge.origin_m, atol=1e-12)
    assert np.allclose(permuted_gauge.basis, gauge.basis, atol=1e-12)


def test_conditioned_gauge_reports_collinear_geometry() -> None:
    microphones = np.column_stack(
        [
            np.linspace(0.0, 3.0, 5),
            np.zeros(5),
            np.zeros(5),
        ]
    )
    with pytest.raises(ValueError, match="collinear"):
        select_coordinate_gauge(microphones)


def test_conditioned_gauge_detects_full_3d_rank() -> None:
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 1.5, 0.0],
            [0.2, 0.1, 1.2],
            [1.2, 0.7, -0.4],
        ]
    )
    canonical, _, gauge = canonicalize_scene_conditioned(microphones)
    assert gauge.affine_rank == 3
    assert gauge.orientation_index is not None
    assert canonical[gauge.orientation_index, 2] > 0.0
