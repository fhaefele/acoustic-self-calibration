import numpy as np
import pytest

from acoustic_self_calibration.bayesian import DistancePrior, calibrate_bayesian
from acoustic_self_calibration.geometry import apply_rigid, rigid_align, rms_position_error
from acoustic_self_calibration.tdoa import ideal_tdoa


def scene(mic_count=8, frames=18, seed=17):
    rng = np.random.default_rng(seed)
    mics = rng.uniform([-1.5, -1.5, 0.0], [1.5, 1.5, 2.3], size=(mic_count, 3))
    t = np.linspace(0.0, 1.7, frames)
    phase = np.linspace(0.0, 1.8 * np.pi, frames)
    src = np.column_stack(
        [1.8 * np.cos(phase), 1.3 * np.sin(phase), 1.2 + 0.65 * np.sin(0.7 * phase)]
    )
    return mics, src, t


def test_bayesian_map_recovers_geometry_with_clock_offsets():
    rng = np.random.default_rng(2)
    mics, src, times = scene()
    c = 343.0
    tau = ideal_tdoa(mics, src, speed_of_sound=c)
    offsets = np.zeros(len(mics))
    offsets[1:] = rng.normal(scale=25e-6, size=len(mics) - 1)
    tau_obs = tau + offsets[1:][None, :] + rng.normal(scale=1.5e-6, size=tau.shape)

    result = calibrate_bayesian(
        tau_obs,
        times,
        len(mics),
        tdoa_sigma_s=2.0e-6,
        initial_microphones=mics + rng.normal(scale=0.03, size=mics.shape),
        initial_sources=src + rng.normal(scale=0.05, size=src.shape),
        estimate_clock_offsets=True,
        clock_offset_prior_sigma_s=100e-6,
        motion_velocity_change_sigma_mps=5.0,
        likelihood="gaussian",
        max_nfev=1500,
    )

    aligned_mics, r, tr = rigid_align(result.microphone_positions, mics)
    aligned_src = apply_rigid(result.source_positions, r, tr)
    assert result.success
    assert rms_position_error(aligned_mics, mics) < 0.08
    assert rms_position_error(aligned_src, src) < 0.12
    assert np.sqrt(np.mean((result.clock_offsets_s[1:] - offsets[1:]) ** 2)) < 15e-6
    assert result.microphone_position_std_m is not None
    assert np.isfinite(result.microphone_position_std_m).all()
    assert result.source_position_std_m is not None
    assert result.source_position_std_m.shape == src.shape
    assert np.isfinite(result.source_position_std_m).all()
    assert np.all(result.source_position_std_m >= 0.0)


def test_sound_speed_requires_metric_anchor():
    mics, src, times = scene()
    tau = ideal_tdoa(mics, src)
    with pytest.raises(ValueError, match="DistancePrior"):
        calibrate_bayesian(
            tau,
            times,
            len(mics),
            tdoa_sigma_s=2e-6,
            estimate_speed_of_sound=True,
            initial_microphones=mics,
            initial_sources=src,
        )


def test_sound_speed_with_known_baseline():
    rng = np.random.default_rng(8)
    mics, src, times = scene(mic_count=10, frames=20, seed=7)
    true_c = 346.2
    tau = ideal_tdoa(mics, src, speed_of_sound=true_c)
    baseline = float(np.linalg.norm(mics[1] - mics[0]))
    result = calibrate_bayesian(
        tau + rng.normal(scale=1e-6, size=tau.shape),
        times,
        len(mics),
        tdoa_sigma_s=1.5e-6,
        speed_of_sound=343.0,
        estimate_speed_of_sound=True,
        speed_of_sound_prior_mean=343.0,
        speed_of_sound_prior_sigma=10.0,
        distance_priors=[DistancePrior(0, 1, baseline, 0.002)],
        initial_microphones=mics + rng.normal(scale=0.02, size=mics.shape),
        initial_sources=src + rng.normal(scale=0.03, size=src.shape),
        motion_velocity_change_sigma_mps=5.0,
        likelihood="gaussian",
        max_nfev=1500,
    )
    assert abs(result.speed_of_sound - true_c) < 2.0
    assert result.speed_of_sound_std is not None
    assert np.isfinite(result.speed_of_sound_std)


@pytest.mark.parametrize("mic_count", [8, 16, 24])
def test_bayesian_supports_requested_array_sizes(mic_count):
    mics, src, times = scene(mic_count=mic_count, frames=10, seed=mic_count)
    tau = ideal_tdoa(mics, src)
    result = calibrate_bayesian(
        tau,
        times,
        mic_count,
        tdoa_sigma_s=1e-6,
        initial_microphones=mics,
        initial_sources=src,
        motion_velocity_change_sigma_mps=None,
        likelihood="gaussian",
        max_nfev=50,
        compute_laplace_uncertainty=False,
    )
    assert result.success
    assert result.rms_tdoa_residual_s < 1e-8
    assert result.source_position_std_m is None


def test_pairwise_measurement_graph_matches_reference_solution():
    from acoustic_self_calibration.tdoa import make_microphone_pairs

    mics, src, times = scene(mic_count=8, frames=12, seed=33)
    pairs = make_microphone_pairs(len(mics), mode="redundant", reference_count=3)
    cols = []
    for a, b in pairs:
        da = np.linalg.norm(src - mics[a], axis=1)
        db = np.linalg.norm(src - mics[b], axis=1)
        cols.append((db - da) / 343.0)
    tau = np.stack(cols, axis=1)
    result = calibrate_bayesian(
        tau,
        times,
        len(mics),
        tdoa_sigma_s=1e-6,
        microphone_pairs=pairs,
        initial_microphones=mics,
        initial_sources=src,
        motion_velocity_change_sigma_mps=None,
        likelihood="gaussian",
        compute_laplace_uncertainty=False,
        max_nfev=50,
    )
    assert result.success
    assert result.rms_tdoa_residual_s < 1e-8


def test_cauchy_likelihood_rejects_sparse_bad_tdoa_peaks():
    from acoustic_self_calibration.tdoa import make_microphone_pairs

    rng = np.random.default_rng(123)
    microphone_count = 12
    frame_count = 24
    speed_of_sound = 343.0
    microphones = rng.uniform(
        [-1.5, -1.5, 0.0],
        [1.5, 1.5, 2.3],
        size=(microphone_count, 3),
    )
    times = np.linspace(0.0, 2.3, frame_count)
    phase = np.linspace(0.0, 1.8 * np.pi, frame_count)
    sources = np.column_stack(
        [
            1.8 * np.cos(phase),
            1.3 * np.sin(phase),
            1.2 + 0.65 * np.sin(0.7 * phase),
        ]
    )
    pairs = make_microphone_pairs(microphone_count, mode="redundant", reference_count=2)
    tdoa = np.stack(
        [
            (
                np.linalg.norm(sources - microphones[b], axis=1)
                - np.linalg.norm(sources - microphones[a], axis=1)
            )
            / speed_of_sound
            for a, b in pairs
        ],
        axis=1,
    )
    tdoa += rng.normal(scale=2e-6, size=tdoa.shape)

    outliers = rng.random(tdoa.shape) < 0.04
    tdoa[outliers] += rng.choice([-1.0, 1.0], size=outliers.sum()) * rng.uniform(
        0.4e-3,
        1.0e-3,
        size=outliers.sum(),
    )

    result = calibrate_bayesian(
        tdoa,
        times,
        microphone_count,
        tdoa_sigma_s=3e-6,
        microphone_pairs=pairs,
        initial_microphones=microphones + rng.normal(scale=0.03, size=microphones.shape),
        initial_sources=sources + rng.normal(scale=0.05, size=sources.shape),
        motion_velocity_change_sigma_mps=5.0,
        likelihood="cauchy",
        max_nfev=1000,
        compute_laplace_uncertainty=False,
    )

    aligned_microphones, rotation, translation = rigid_align(
        result.microphone_positions,
        microphones,
    )
    aligned_sources = apply_rigid(result.source_positions, rotation, translation)

    assert result.success
    assert rms_position_error(aligned_microphones, microphones) < 0.02
    assert rms_position_error(aligned_sources, sources) < 0.03
