import numpy as np
import pytest

from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    nondegenerate_planar_microphones,
    nondegenerate_planar_sources,
)
from acoustic_self_calibration.stratified.solver import calibrate_planar_tdoa_8mic


def _measurements(event_count: int, *, noise_s: float = 0.0):
    microphones = nondegenerate_planar_microphones()
    times = np.linspace(0.2, 2.0, event_count)
    sources = nondegenerate_planar_sources(times)
    tdoa = exact_reference_tdoas(microphones, sources)
    if noise_s > 0.0:
        rng = np.random.default_rng(451)
        tdoa = tdoa + rng.normal(scale=noise_s, size=tdoa.shape)
    sigma = np.full_like(tdoa, max(noise_s, 2e-6))
    measurements = EventTDOAMeasurements(
        event_ids=np.arange(event_count),
        receiver_event_times_s=times,
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=tdoa,
        sigma_s=sigma,
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )
    return microphones, sources, measurements


@pytest.mark.parametrize("event_count", [20, 40])
def test_planar_tdoa_gate_exact(event_count: int) -> None:
    microphones, sources, measurements = _measurements(event_count)
    result = calibrate_planar_tdoa_8mic(measurements)

    assert result.status == "solved"
    assert result.microphone_positions_m is not None
    assert result.source_unsigned_heights_m is not None
    assert result.source_height_sign_known is not None
    assert result.tdoa_rms_s is not None
    assert result.tdoa_rms_s < 1e-7
    assert not np.any(result.source_height_sign_known)
    estimated_distances = np.linalg.norm(
        result.microphone_positions_m[:, None, :] - result.microphone_positions_m[None, :, :],
        axis=2,
    )
    true_distances = np.linalg.norm(
        microphones[:, None, :] - microphones[None, :, :],
        axis=2,
    )
    assert np.allclose(estimated_distances, true_distances, atol=1e-5, rtol=1e-5)
    assert np.allclose(
        np.sort(result.source_unsigned_heights_m),
        np.sort(np.abs(sources[:, 1])),
        atol=1e-4,
        rtol=1e-4,
    )


def test_planar_tdoa_gate_seeded_noise() -> None:
    microphones, _, measurements = _measurements(40, noise_s=2e-6)
    result = calibrate_planar_tdoa_8mic(
        measurements,
        membership_tolerance=2e-2,
        metric_acceptance_rms_m=2e-2,
    )

    assert result.status == "solved"
    assert result.microphone_positions_m is not None
    estimated_distances = np.linalg.norm(
        result.microphone_positions_m[:, None, :] - result.microphone_positions_m[None, :, :],
        axis=2,
    )
    true_distances = np.linalg.norm(
        microphones[:, None, :] - microphones[None, :, :],
        axis=2,
    )
    assert np.sqrt(np.mean((estimated_distances - true_distances) ** 2)) < 0.05
    assert result.tdoa_rms_s is not None
    assert result.tdoa_rms_s < 60e-6


def test_planar_half_space_prior_orients_and_reports_signs() -> None:
    from acoustic_self_calibration.geometry import select_coordinate_gauge
    from acoustic_self_calibration.stratified.solver import calibrate_planar_tdoa

    _, _, measurements = _measurements(20)
    blind = calibrate_planar_tdoa(measurements)
    signed = calibrate_planar_tdoa(measurements, source_half_space_sign=1)

    assert blind.source_height_sign_known is not None
    assert not np.any(blind.source_height_sign_known)
    assert not blind.source_region_constraint_enforced

    assert signed.status == "solved", signed.diagnostics
    assert signed.source_region_constraint_enforced
    assert signed.source_half_space_sign == 1
    assert signed.source_height_sign_known is not None
    assert np.all(signed.source_height_sign_known)
    assert signed.microphone_positions_m is not None
    assert signed.source_representative_positions_m is not None

    gauge = select_coordinate_gauge(np.asarray(signed.microphone_positions_m))
    signed_heights = (
        np.asarray(signed.source_representative_positions_m) - gauge.origin_m
    ) @ gauge.basis[:, 2]
    assert np.all(signed_heights > 0.0)

    # Prior is orientation only: mic geometry and unsigned heights unchanged.
    assert np.allclose(
        signed.microphone_positions_m,
        blind.microphone_positions_m,
        atol=1e-9,
        rtol=1e-9,
    )
    assert np.allclose(
        signed.source_unsigned_heights_m,
        blind.source_unsigned_heights_m,
        atol=1e-9,
        rtol=1e-9,
    )
