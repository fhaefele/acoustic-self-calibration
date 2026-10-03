import numpy as np
import pytest

from acoustic_self_calibration.geometry import rigid_align, rms_position_error
from acoustic_self_calibration.simulation import (
    CONSTRAINED_MYOTIS_CROSS_LIMITS,
    PCM16_EVENT_LIMITS,
    RANDOM_3D_EVENT_LIMITS,
    constrained_myotis_cross_fixture_contract,
    event_count_sweep_cases,
    exact_reference_tdoas,
    make_myotis_cross_pulse_scene,
    make_random_3d_pulse_scene,
    myotis_cross_fixture_contract,
    myotis_cross_microphones,
    myotis_cross_source_positions,
    nondegenerate_planar_fixture_contract,
    nondegenerate_planar_microphones,
    nondegenerate_planar_sources,
)


@pytest.mark.parametrize("microphone_count", [8, 12, 16, 24])
@pytest.mark.parametrize("event_count", [20, 40])
def test_random_3d_pulse_fixture_contract(microphone_count: int, event_count: int) -> None:
    scene = make_random_3d_pulse_scene(microphone_count, event_count=event_count)

    assert scene.sample_rate_hz == 48_000
    assert scene.duration_s == 5.0
    assert scene.microphone_positions_m.shape == (microphone_count, 3)
    assert scene.trajectory_positions_m.shape == (31, 3)
    assert scene.event_times_s.shape == (event_count,)
    assert scene.source_positions_at_events_m.shape == (event_count, 3)
    assert scene.audio.shape == (240_000, microphone_count)
    assert np.allclose(scene.event_times_s, np.linspace(0.4, 4.4, event_count))
    assert np.all(np.isfinite(scene.audio))
    assert np.max(np.abs(scene.audio)) > 0.0


def test_40_event_fixture_respects_pinned_frontend_minimum_gap() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=40)
    assert np.min(np.diff(scene.event_times_s)) > 0.05


def test_event_count_sweep_bookkeeping_matches_plan() -> None:
    rows = event_count_sweep_cases(8)
    assert [row.event_count for row in rows] == [8, 12, 20, 30, 40]
    row20 = next(row for row in rows if row.event_count == 20)
    row40 = next(row for row in rows if row.event_count == 40)
    assert (row20.independent_measurement_count, row20.geometry_unknown_count) == (140, 78)
    assert (row40.independent_measurement_count, row40.geometry_unknown_count) == (280, 138)
    assert all(row.seed == 108 for row in rows)


def _cross_deformation(
    microphones: np.ndarray,
    sources: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered_receivers = microphones[:, [0, 2]] - np.array([1.2, 0.0])
    projected_sources = sources[:, [0, 2]] - np.array([1.2, 0.0])
    heights = sources[:, 1]

    gram = np.array([[1.0, alpha], [alpha, 1.0]])
    transform = np.linalg.cholesky(gram).T
    transformed_receivers = centered_receivers @ transform.T
    transformed_sources = np.linalg.solve(transform.T, projected_sources.T).T
    height_squared = (
        np.sum(projected_sources * projected_sources, axis=1)
        + heights * heights
        - np.sum(transformed_sources * transformed_sources, axis=1)
    )
    assert np.all(height_squared > 0.0)

    mics_out = np.column_stack(
        [
            transformed_receivers[:, 0] + 1.2,
            np.zeros(len(microphones)),
            transformed_receivers[:, 1],
        ]
    )
    sources_out = np.column_stack(
        [
            transformed_sources[:, 0] + 1.2,
            np.sqrt(height_squared),
            transformed_sources[:, 1],
        ]
    )
    return mics_out, sources_out


@pytest.mark.parametrize("event_count", [18, 40])
def test_myotis_cross_has_continuous_distance_preserving_ambiguity(event_count: int) -> None:
    microphones = myotis_cross_microphones()
    event_times = np.linspace(0.25, 2.05, event_count)
    sources = myotis_cross_source_positions(event_times)

    centered = microphones[:, [0, 2]] - np.array([1.2, 0.0])
    conic_design = np.column_stack(
        [
            centered[:, 0] ** 2,
            centered[:, 0] * centered[:, 1],
            centered[:, 1] ** 2,
            centered[:, 0],
            centered[:, 1],
            np.ones(len(centered)),
        ]
    )
    assert np.linalg.matrix_rank(conic_design, tol=1e-12) == 5

    transformed_mics, transformed_sources = _cross_deformation(
        microphones,
        sources,
        alpha=0.6,
    )
    original_ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    transformed_ranges = np.linalg.norm(
        transformed_mics[:, None, :] - transformed_sources[None, :, :],
        axis=2,
    )
    assert np.max(np.abs(original_ranges - transformed_ranges)) < 1e-12

    aligned_mics, _, _ = rigid_align(transformed_mics, microphones)
    assert rms_position_error(aligned_mics, microphones) > 0.20


def test_planar_source_height_sign_is_independently_unobservable() -> None:
    microphones = nondegenerate_planar_microphones()
    times = np.linspace(0.2, 2.0, 10)
    sources = nondegenerate_planar_sources(times)
    baseline = exact_reference_tdoas(microphones, sources)

    for event_index in (0, 3, 8):
        reflected = sources.copy()
        reflected[event_index, 1] *= -1.0
        candidate = exact_reference_tdoas(microphones, reflected)
        assert np.allclose(candidate, baseline, atol=1e-15, rtol=0.0)


def test_nondegenerate_planar_fixture_is_not_the_cross_degeneracy() -> None:
    microphones = nondegenerate_planar_microphones()
    planar = microphones[:, [0, 2]]
    centered = planar - planar.mean(axis=0, keepdims=True)
    conic_design = np.column_stack(
        [
            centered[:, 0] ** 2,
            centered[:, 0] * centered[:, 1],
            centered[:, 1] ** 2,
            centered[:, 0],
            centered[:, 1],
            np.ones(len(centered)),
        ]
    )
    assert np.linalg.matrix_rank(conic_design, tol=1e-12) == 6


def test_myotis_audio_fixture_keeps_original_18_event_schedule() -> None:
    scene = make_myotis_cross_pulse_scene()
    assert scene.sample_rate_hz == 96_000
    assert scene.duration_s == 2.4
    assert scene.microphone_positions_m.shape == (12, 3)
    assert np.allclose(scene.event_times_s, np.linspace(0.25, 2.05, 18))
    assert np.min(np.diff(scene.event_times_s)) > 0.05

    forty_event_spacing = float(np.min(np.diff(np.linspace(0.25, 2.05, 40))))
    assert forty_event_spacing < 0.05


def test_frozen_acceptance_threshold_contracts() -> None:
    assert RANDOM_3D_EVENT_LIMITS.microphone_rms_m == 0.15
    assert RANDOM_3D_EVENT_LIMITS.source_rms_m == 0.18
    assert RANDOM_3D_EVENT_LIMITS.tdoa_rms_s == 60e-6
    assert PCM16_EVENT_LIMITS.microphone_rms_m == 0.18
    assert PCM16_EVENT_LIMITS.source_rms_m == 0.22
    assert CONSTRAINED_MYOTIS_CROSS_LIMITS.microphone_rms_m == 0.20
    assert CONSTRAINED_MYOTIS_CROSS_LIMITS.source_rms_m == 0.30
    assert CONSTRAINED_MYOTIS_CROSS_LIMITS.tdoa_rms_s == 45e-6


def test_planar_cross_and_constrained_cross_contracts_are_distinct() -> None:
    planar = nondegenerate_planar_fixture_contract()
    cross = myotis_cross_fixture_contract()
    constrained = constrained_myotis_cross_fixture_contract()

    assert planar.expected_outcome == "planar_success"
    assert cross.expected_outcome == "cross_ambiguity"
    assert constrained.expected_outcome == "constrained_cross_observable_success"
    assert cross.constraint_type is None
    assert constrained.constraint_type == "inter_arm_angle_rad"
    assert constrained.constraint_value == pytest.approx(np.pi / 2.0)
    assert constrained.source_sign_convention_required
