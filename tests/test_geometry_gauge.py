import numpy as np

from acoustic_self_calibration.bayesian import calibrate_bayesian
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.tdoa import make_microphone_pairs


def test_planar_array_with_collinear_leading_channels_is_supported() -> None:
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [1.2, 0.0, 0.8],
            [1.2, 0.0, 0.4],
            [1.2, 0.0, -0.4],
            [1.2, 0.0, -0.8],
            [1.2, 0.0, -1.2],
        ],
        dtype=float,
    )
    times = np.linspace(0.05, 0.85, 20)
    phase = np.linspace(0.0, 1.0, len(times))
    sources = np.column_stack(
        [
            1.0 - 0.9 * phase,
            2.6 - 1.8 * phase,
            -0.25 + 0.22 * phase,
        ]
    )
    pairs = make_microphone_pairs(len(microphones), mode="redundant", reference_count=2)
    tdoa = np.stack(
        [
            (
                np.linalg.norm(sources - microphones[b], axis=1)
                - np.linalg.norm(sources - microphones[a], axis=1)
            )
            / 343.0
            for a, b in pairs
        ],
        axis=1,
    )

    result = calibrate_bayesian(
        tdoa,
        times,
        len(microphones),
        tdoa_sigma_s=1e-6,
        microphone_pairs=pairs,
        initial_microphones=microphones,
        initial_sources=sources,
        motion_velocity_change_sigma_mps=None,
        likelihood="gaussian",
        max_nfev=100,
        compute_laplace_uncertainty=False,
    )

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions,
        microphones,
    )
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)

    assert result.success
    assert result.rms_tdoa_residual_s < 1e-8
    assert rms_position_error(aligned_microphones, microphones) < 1e-4
    assert rms_position_error(aligned_sources, sources) < 1e-4
