import numpy as np

from acoustic_self_calibration.measurements import EventTDOAMeasurements
from acoustic_self_calibration.simulation import (
    exact_reference_tdoas,
    make_random_3d_pulse_scene,
    nondegenerate_planar_microphones,
    nondegenerate_planar_sources,
)
from acoustic_self_calibration.stratified.solver import compare_tdoa_models_8mic


def _measurement_object(
    tdoa_s: np.ndarray,
    times_s: np.ndarray,
) -> EventTDOAMeasurements:
    return EventTDOAMeasurements(
        event_ids=np.arange(len(times_s)),
        receiver_event_times_s=times_s,
        microphone_ids=tuple(range(8)),
        microphone_pairs=tuple((0, index) for index in range(1, 8)),
        tdoa_s=tdoa_s,
        sigma_s=np.full_like(tdoa_s, 2e-6),
        confidence=np.ones_like(tdoa_s),
        valid=np.ones_like(tdoa_s, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="reference_star",
    )


def test_model_comparison_prefers_planar_when_3d_is_structurally_unsupported() -> None:
    microphones = nondegenerate_planar_microphones()
    times = np.linspace(0.2, 2.0, 20)
    sources = nondegenerate_planar_sources(times)
    measurements = _measurement_object(
        exact_reference_tdoas(microphones, sources),
        times,
    )

    result = compare_tdoa_models_8mic(measurements)
    assert result.planar.status == "solved"
    assert result.comparison.status == "receiver2d_source3d"
    assert result.comparison.preferred_model == "receiver2d_source3d"


def test_model_comparison_does_not_prefer_planar_for_generic_3d_scene() -> None:
    scene = make_random_3d_pulse_scene(8, event_count=20)
    ranges = np.linalg.norm(
        scene.microphone_positions_m[:, None, :] - scene.source_positions_at_events_m[None, :, :],
        axis=2,
    )
    tdoa = ((ranges[1:] - ranges[[0]]) / 343.0).T
    measurements = _measurement_object(tdoa, scene.event_times_s)

    result = compare_tdoa_models_8mic(measurements)
    assert result.general_3d.status == "solved"
    assert result.comparison.status == "receiver3d_source3d"
    assert result.comparison.preferred_model == "receiver3d_source3d"
