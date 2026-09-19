import numpy as np
import pytest

from acoustic_self_calibration.measurements import (
    EventTDOAMeasurements,
    reference_star_from_arrivals,
)


def _arrival_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrivals = np.array(
        [
            [0.0, 20e-6, -15e-6, 8e-6],
            [0.0, 24e-6, -10e-6, 11e-6],
            [0.0, 29e-6, -7e-6, 15e-6],
        ]
    )
    sigma = np.full_like(arrivals, 2e-6)
    times = np.array([0.2, 0.4, 0.6])
    return arrivals, sigma, times


def test_reference_star_from_arrivals_builds_independent_basis_and_covariance() -> None:
    arrivals, sigma, times = _arrival_fixture()
    measurements = reference_star_from_arrivals(
        arrivals,
        sigma,
        receiver_event_times_s=times,
        reference_microphone=0,
        event_samples=np.array([9600, 19200, 28800]),
        sample_rate_hz=48_000,
        event_channel=0,
    )

    assert measurements.microphone_pairs == ((0, 1), (0, 2), (0, 3))
    assert measurements.measurement_origin == "derived_arrivals"
    assert measurements.measurement_basis == "reference_star"
    assert np.all(measurements.valid)
    assert np.allclose(measurements.tdoa_s, arrivals[:, 1:] - arrivals[:, [0]])
    assert np.allclose(measurements.sigma_s, np.sqrt(8e-12))
    assert all(measurements.event_graph_connected(index) for index in range(3))
    assert [measurements.independent_measurement_rank(index) for index in range(3)] == [3, 3, 3]

    covariance = measurements.covariance_s2
    assert covariance is not None
    expected = np.full((3, 3), 4e-12)
    np.fill_diagonal(expected, 8e-12)
    assert np.allclose(covariance[0], expected)


def test_reference_change_preserves_recoverable_pair_differences() -> None:
    arrivals, sigma, times = _arrival_fixture()
    star0 = reference_star_from_arrivals(
        arrivals,
        sigma,
        receiver_event_times_s=times,
        reference_microphone=0,
    )
    star2 = reference_star_from_arrivals(
        arrivals,
        sigma,
        receiver_event_times_s=times,
        reference_microphone=2,
    )

    def pair_value(measurements: EventTDOAMeasurements, event: int, a: int, b: int) -> float:
        representatives = measurements.arrival_representatives_s
        assert representatives is not None
        ids = measurements.microphone_ids
        return float(representatives[event, ids.index(b)] - representatives[event, ids.index(a)])

    for event in range(3):
        for a in range(4):
            for b in range(4):
                assert pair_value(star0, event, a, b) == pytest.approx(
                    pair_value(star2, event, a, b)
                )


def test_missing_arrival_marks_pairs_invalid_without_zero_filling() -> None:
    arrivals, sigma, times = _arrival_fixture()
    valid = np.ones_like(arrivals, dtype=bool)
    valid[1, 3] = False
    arrivals[1, 3] = np.nan
    sigma[1, 3] = np.nan

    measurements = reference_star_from_arrivals(
        arrivals,
        sigma,
        receiver_event_times_s=times,
        reference_microphone=0,
        arrival_valid=valid,
    )

    pair_index = measurements.microphone_pairs.index((0, 3))
    assert not measurements.valid[1, pair_index]
    assert np.isnan(measurements.tdoa_s[1, pair_index])
    assert np.isnan(measurements.sigma_s[1, pair_index])
    assert np.isnan(measurements.confidence[1, pair_index])
    assert not measurements.event_graph_connected(1)
    assert measurements.independent_measurement_rank(1) == 2


def test_measurement_arrays_are_owned_and_read_only() -> None:
    arrivals, sigma, times = _arrival_fixture()
    measurements = reference_star_from_arrivals(
        arrivals,
        sigma,
        receiver_event_times_s=times,
        reference_microphone=0,
    )
    arrivals[0, 1] = 999.0

    representatives = measurements.arrival_representatives_s
    assert representatives is not None
    assert representatives[0, 1] != 999.0
    with pytest.raises(ValueError):
        measurements.tdoa_s[0, 0] = 0.0
    with pytest.raises(ValueError):
        representatives[0, 0] = 0.0


def test_duplicate_pair_does_not_increase_independent_rank() -> None:
    tdoa = np.array([[1e-6, 2e-6, 2e-6, 1e-6]])
    measurements = EventTDOAMeasurements(
        event_ids=np.array([0]),
        receiver_event_times_s=np.array([0.1]),
        microphone_ids=(0, 1, 2),
        microphone_pairs=((0, 1), (0, 2), (0, 2), (1, 2)),
        tdoa_s=tdoa,
        sigma_s=np.full_like(tdoa, 1e-6),
        confidence=np.ones_like(tdoa),
        valid=np.ones_like(tdoa, dtype=bool),
        measurement_origin="independent_pairs",
        measurement_basis="redundant_pairs",
    )
    assert measurements.independent_measurement_rank(0) == 2


def test_invalid_entries_must_be_nan() -> None:
    with pytest.raises(ValueError, match="invalid tdoa_s/sigma_s"):
        EventTDOAMeasurements(
            event_ids=np.array([0]),
            receiver_event_times_s=np.array([0.1]),
            microphone_ids=(0, 1),
            microphone_pairs=((0, 1),),
            tdoa_s=np.array([[0.0]]),
            sigma_s=np.array([[1e-6]]),
            confidence=np.array([[0.5]]),
            valid=np.array([[False]]),
            measurement_origin="independent_pairs",
            measurement_basis="independent_pairs",
        )


def test_receive_time_semantics_require_increasing_event_landmarks() -> None:
    arrivals = np.zeros((2, 3))
    sigma = np.full_like(arrivals, 1e-6)
    with pytest.raises(ValueError, match="strictly increasing"):
        reference_star_from_arrivals(
            arrivals,
            sigma,
            receiver_event_times_s=np.array([0.2, 0.2]),
            reference_microphone=0,
        )


def test_event_samples_and_receive_times_must_agree() -> None:
    arrivals = np.zeros((2, 3))
    sigma = np.full_like(arrivals, 1e-6)
    with pytest.raises(ValueError, match="event_samples/sample_rate_hz"):
        reference_star_from_arrivals(
            arrivals,
            sigma,
            receiver_event_times_s=np.array([0.2, 0.41]),
            reference_microphone=0,
            event_samples=np.array([9600, 19200]),
            sample_rate_hz=48_000,
        )


def test_reference_star_round_trips_all_redundant_pair_differences() -> None:
    arrivals, sigma, times = _arrival_fixture()
    measurements = reference_star_from_arrivals(
        arrivals,
        sigma,
        receiver_event_times_s=times,
        reference_microphone=0,
    )
    recovered = np.zeros_like(arrivals)
    for column, (_, target) in enumerate(measurements.microphone_pairs):
        recovered[:, target] = measurements.tdoa_s[:, column]

    for a in range(arrivals.shape[1]):
        for b in range(arrivals.shape[1]):
            expected = arrivals[:, b] - arrivals[:, a]
            actual = recovered[:, b] - recovered[:, a]
            assert np.allclose(actual, expected, atol=1e-15, rtol=0.0)
