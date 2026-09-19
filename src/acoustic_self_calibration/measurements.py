from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

MeasurementOrigin = Literal["independent_pairs", "derived_arrivals"]
MeasurementBasis = Literal["reference_star", "independent_pairs", "redundant_pairs"]


def _readonly_copy(value: np.ndarray | Sequence[float], *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _readonly_optional(
    value: np.ndarray | Sequence[float] | None,
    *,
    dtype: Any,
) -> np.ndarray | None:
    if value is None:
        return None
    return _readonly_copy(value, dtype=dtype)


@dataclass(frozen=True)
class EventTDOAMeasurements:
    """Validated immutable boundary between event extraction and geometry.

    receiver_event_times_s are observed receive-time landmarks at the acquisition
    reference channel. They are not source emission times.
    """

    event_ids: np.ndarray
    receiver_event_times_s: np.ndarray
    microphone_ids: tuple[int, ...]
    microphone_pairs: tuple[tuple[int, int], ...]
    tdoa_s: np.ndarray
    sigma_s: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    measurement_origin: MeasurementOrigin
    measurement_basis: MeasurementBasis
    event_samples: np.ndarray | None = None
    sample_rate_hz: int | None = None
    event_channel: int | None = None
    primitive_ids: tuple[str, ...] | None = None
    pair_from_primitive: np.ndarray | None = None
    covariance_s2: np.ndarray | None = None
    covariance_model: str = "diagonal_marginals"
    arrival_representatives_s: np.ndarray | None = None
    arrival_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        event_ids = _readonly_copy(self.event_ids, dtype=np.int64).reshape(-1)
        event_ids.setflags(write=False)
        times = _readonly_copy(self.receiver_event_times_s, dtype=float).reshape(-1)
        times.setflags(write=False)
        tdoa = _readonly_copy(self.tdoa_s, dtype=float)
        sigma = _readonly_copy(self.sigma_s, dtype=float)
        confidence = _readonly_copy(self.confidence, dtype=float)
        valid = _readonly_copy(self.valid, dtype=bool)

        if len(event_ids) == 0:
            raise ValueError("event_ids cannot be empty")
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError("event_ids must be unique")
        if times.shape != event_ids.shape or not np.all(np.isfinite(times)):
            raise ValueError("receiver_event_times_s must be finite with shape (E,)")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("receiver_event_times_s must be strictly increasing")

        microphone_ids = tuple(int(value) for value in self.microphone_ids)
        if len(microphone_ids) < 2 or len(set(microphone_ids)) != len(microphone_ids):
            raise ValueError("microphone_ids must contain at least two unique IDs")
        microphone_id_set = set(microphone_ids)

        pairs = tuple((int(a), int(b)) for a, b in self.microphone_pairs)
        if not pairs:
            raise ValueError("microphone_pairs cannot be empty")
        for pair in pairs:
            a, b = pair
            if a == b or a not in microphone_id_set or b not in microphone_id_set:
                raise ValueError(f"invalid microphone pair {pair}")

        expected_shape = (len(event_ids), len(pairs))
        if tdoa.shape != expected_shape:
            raise ValueError(f"tdoa_s must have shape {expected_shape}")
        if sigma.shape != expected_shape:
            raise ValueError(f"sigma_s must have shape {expected_shape}")
        if confidence.shape != expected_shape:
            raise ValueError(f"confidence must have shape {expected_shape}")
        if valid.shape != expected_shape:
            raise ValueError(f"valid must have shape {expected_shape}")

        if np.any(~np.isfinite(tdoa[valid])):
            raise ValueError("valid tdoa_s entries must be finite")
        if np.any(~np.isfinite(sigma[valid])) or np.any(sigma[valid] <= 0.0):
            raise ValueError("valid sigma_s entries must be finite and positive")
        if np.any(~np.isfinite(confidence[valid])):
            raise ValueError("valid confidence entries must be finite")
        if np.any((confidence[valid] < 0.0) | (confidence[valid] > 1.0)):
            raise ValueError("valid confidence entries must lie in [0, 1]")
        if np.any(np.isfinite(tdoa[~valid])) or np.any(np.isfinite(sigma[~valid])):
            raise ValueError("invalid tdoa_s/sigma_s entries must be NaN")
        if np.any(np.isfinite(confidence[~valid])):
            raise ValueError("invalid confidence entries must be NaN")

        event_samples = _readonly_optional(self.event_samples, dtype=np.int64)
        sample_rate_hz = None if self.sample_rate_hz is None else int(self.sample_rate_hz)
        if self.sample_rate_hz is not None and float(self.sample_rate_hz) != sample_rate_hz:
            raise ValueError("sample_rate_hz must be an integer sample rate")
        if event_samples is not None:
            event_samples = event_samples.reshape(-1)
            event_samples.setflags(write=False)
            if event_samples.shape != event_ids.shape or np.any(event_samples < 0):
                raise ValueError("event_samples must be non-negative with shape (E,)")
            if len(event_samples) > 1 and np.any(np.diff(event_samples) <= 0):
                raise ValueError("event_samples must be strictly increasing")
            if sample_rate_hz is None or sample_rate_hz <= 0:
                raise ValueError("sample_rate_hz must be positive when event_samples are present")
            sample_times = event_samples.astype(float) / float(sample_rate_hz)
            tolerance_s = 0.5 / float(sample_rate_hz) + 1e-12
            if not np.allclose(times, sample_times, atol=tolerance_s, rtol=0.0):
                raise ValueError(
                    "receiver_event_times_s must agree with event_samples/sample_rate_hz"
                )
        elif sample_rate_hz is not None and sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")

        event_channel = None if self.event_channel is None else int(self.event_channel)
        if event_channel is not None and event_channel not in microphone_id_set:
            raise ValueError("event_channel must be one of microphone_ids")

        primitive_ids = None
        pair_from_primitive = _readonly_optional(self.pair_from_primitive, dtype=float)
        if self.primitive_ids is not None:
            primitive_ids = tuple(str(value) for value in self.primitive_ids)
            if len(set(primitive_ids)) != len(primitive_ids):
                raise ValueError("primitive_ids must be unique")
            if pair_from_primitive is None:
                raise ValueError("pair_from_primitive is required when primitive_ids are provided")
            if pair_from_primitive.shape != (len(pairs), len(primitive_ids)):
                raise ValueError("pair_from_primitive has incompatible shape")
            if not np.all(np.isfinite(pair_from_primitive)):
                raise ValueError("pair_from_primitive must be finite")
        elif pair_from_primitive is not None:
            raise ValueError("primitive_ids are required when pair_from_primitive is provided")

        covariance = _readonly_optional(self.covariance_s2, dtype=float)
        if covariance is not None:
            if covariance.shape != (len(event_ids), len(pairs), len(pairs)):
                raise ValueError("covariance_s2 must have shape (E, P, P)")
            for event_index in range(len(event_ids)):
                indices = np.flatnonzero(valid[event_index])
                if indices.size == 0:
                    continue
                block = covariance[event_index][np.ix_(indices, indices)]
                if not np.all(np.isfinite(block)):
                    raise ValueError("valid covariance submatrices must be finite")
                if not np.allclose(block, block.T, atol=1e-15, rtol=1e-10):
                    raise ValueError("valid covariance submatrices must be symmetric")
                eigenvalues = np.linalg.eigvalsh(block)
                tolerance = max(1e-24, float(np.max(np.abs(eigenvalues))) * 1e-10)
                if float(np.min(eigenvalues)) < -tolerance:
                    raise ValueError("valid covariance submatrices must be positive semidefinite")

        arrival_representatives = _readonly_optional(self.arrival_representatives_s, dtype=float)
        arrival_valid = _readonly_optional(self.arrival_valid, dtype=bool)
        if (arrival_representatives is None) != (arrival_valid is None):
            raise ValueError(
                "arrival_representatives_s and arrival_valid must be provided together"
            )
        if arrival_representatives is not None and arrival_valid is not None:
            arrival_shape = (len(event_ids), len(microphone_ids))
            if (
                arrival_representatives.shape != arrival_shape
                or arrival_valid.shape != arrival_shape
            ):
                raise ValueError(f"arrival arrays must have shape {arrival_shape}")
            if np.any(~np.isfinite(arrival_representatives[arrival_valid])):
                raise ValueError("valid arrival representatives must be finite")
            if np.any(np.isfinite(arrival_representatives[~arrival_valid])):
                raise ValueError("invalid arrival representatives must be NaN")

        if self.measurement_origin not in ("independent_pairs", "derived_arrivals"):
            raise ValueError("unsupported measurement_origin")
        if self.measurement_basis not in ("reference_star", "independent_pairs", "redundant_pairs"):
            raise ValueError("unsupported measurement_basis")
        if not self.covariance_model:
            raise ValueError("covariance_model cannot be empty")

        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "receiver_event_times_s", times)
        object.__setattr__(self, "microphone_ids", microphone_ids)
        object.__setattr__(self, "microphone_pairs", pairs)
        object.__setattr__(self, "tdoa_s", tdoa)
        object.__setattr__(self, "sigma_s", sigma)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "event_samples", event_samples)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "event_channel", event_channel)
        object.__setattr__(self, "primitive_ids", primitive_ids)
        object.__setattr__(self, "pair_from_primitive", pair_from_primitive)
        object.__setattr__(self, "covariance_s2", covariance)
        object.__setattr__(self, "arrival_representatives_s", arrival_representatives)
        object.__setattr__(self, "arrival_valid", arrival_valid)

    def pair_incidence_matrix(self) -> np.ndarray:
        """Return oriented pair incidence rows in microphone-ID order."""
        index = {microphone_id: column for column, microphone_id in enumerate(self.microphone_ids)}
        incidence = np.zeros((len(self.microphone_pairs), len(self.microphone_ids)), dtype=float)
        for row, (a, b) in enumerate(self.microphone_pairs):
            incidence[row, index[a]] = -1.0
            incidence[row, index[b]] = 1.0
        incidence.setflags(write=False)
        return incidence

    def independent_measurement_rank(self, event_index: int) -> int:
        """Return the incidence rank of valid pair measurements for one event."""
        if not 0 <= event_index < len(self.event_ids):
            raise IndexError("event_index out of range")
        incidence = self.pair_incidence_matrix()[self.valid[event_index]]
        if incidence.size == 0:
            return 0
        return int(np.linalg.matrix_rank(incidence))

    def event_graph_connected(self, event_index: int) -> bool:
        """Return whether valid pair measurements connect every microphone ID."""
        if not 0 <= event_index < len(self.event_ids):
            raise IndexError("event_index out of range")
        adjacency = {microphone_id: set() for microphone_id in self.microphone_ids}
        for pair_index, (a, b) in enumerate(self.microphone_pairs):
            if self.valid[event_index, pair_index]:
                adjacency[a].add(b)
                adjacency[b].add(a)
        start = self.microphone_ids[0]
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(self.microphone_ids)


def reference_star_from_arrivals(
    arrival_representatives_s: np.ndarray,
    arrival_sigma_s: np.ndarray,
    *,
    receiver_event_times_s: np.ndarray,
    reference_microphone: int,
    microphone_ids: Sequence[int] | None = None,
    arrival_confidence: np.ndarray | None = None,
    arrival_valid: np.ndarray | None = None,
    event_ids: np.ndarray | None = None,
    event_samples: np.ndarray | None = None,
    sample_rate_hz: int | None = None,
    event_channel: int | None = None,
) -> EventTDOAMeasurements:
    """Create an independent reference-star TDOA basis from per-channel arrivals."""
    arrivals = np.asarray(arrival_representatives_s, dtype=float)
    sigma = np.asarray(arrival_sigma_s, dtype=float)
    if arrivals.ndim != 2 or sigma.shape != arrivals.shape:
        raise ValueError("arrival arrays must have the same shape (E, M)")
    event_count, microphone_count = arrivals.shape
    if event_count == 0 or microphone_count < 2:
        raise ValueError("arrival arrays must contain events and at least two microphones")

    ids = tuple(range(microphone_count)) if microphone_ids is None else tuple(microphone_ids)
    if len(ids) != microphone_count or len(set(ids)) != microphone_count:
        raise ValueError("microphone_ids must provide one unique ID per arrival column")
    try:
        reference_index = ids.index(int(reference_microphone))
    except ValueError as error:
        raise ValueError("reference_microphone must be one of microphone_ids") from error

    confidence = (
        np.ones_like(arrivals)
        if arrival_confidence is None
        else np.asarray(arrival_confidence, dtype=float)
    )
    primitive_valid = (
        np.isfinite(arrivals) & np.isfinite(sigma) & (sigma > 0.0)
        if arrival_valid is None
        else np.asarray(arrival_valid, dtype=bool)
    )
    if confidence.shape != arrivals.shape or primitive_valid.shape != arrivals.shape:
        raise ValueError("arrival_confidence and arrival_valid must match arrival arrays")
    if np.any(~np.isfinite(sigma[primitive_valid])) or np.any(sigma[primitive_valid] <= 0.0):
        raise ValueError("valid arrival sigma values must be finite and positive")
    if np.any(~np.isfinite(confidence[primitive_valid])):
        raise ValueError("valid arrival confidence values must be finite")
    if np.any((confidence[primitive_valid] < 0.0) | (confidence[primitive_valid] > 1.0)):
        raise ValueError("valid arrival confidence values must lie in [0, 1]")

    targets = [index for index in range(microphone_count) if index != reference_index]
    pairs = tuple((ids[reference_index], ids[target]) for target in targets)
    pair_count = len(pairs)
    pair_valid = np.zeros((event_count, pair_count), dtype=bool)
    tdoa = np.full((event_count, pair_count), np.nan, dtype=float)
    pair_sigma = np.full_like(tdoa, np.nan)
    pair_confidence = np.full_like(tdoa, np.nan)

    incidence = np.zeros((pair_count, microphone_count), dtype=float)
    for pair_index, target_index in enumerate(targets):
        incidence[pair_index, reference_index] = -1.0
        incidence[pair_index, target_index] = 1.0
        valid = primitive_valid[:, reference_index] & primitive_valid[:, target_index]
        pair_valid[:, pair_index] = valid
        tdoa[valid, pair_index] = arrivals[valid, target_index] - arrivals[valid, reference_index]
        pair_sigma[valid, pair_index] = np.sqrt(
            sigma[valid, reference_index] ** 2 + sigma[valid, target_index] ** 2
        )
        pair_confidence[valid, pair_index] = np.sqrt(
            confidence[valid, reference_index] * confidence[valid, target_index]
        )

    covariance = np.full((event_count, pair_count, pair_count), np.nan, dtype=float)
    for event_index in range(event_count):
        valid_pairs = np.flatnonzero(pair_valid[event_index])
        if valid_pairs.size == 0:
            continue
        variances = np.where(primitive_valid[event_index], sigma[event_index] ** 2, 0.0)
        block_incidence = incidence[valid_pairs]
        block = (block_incidence * variances[None, :]) @ block_incidence.T
        covariance[event_index][np.ix_(valid_pairs, valid_pairs)] = block

    representatives = np.array(arrivals, copy=True)
    representatives[~primitive_valid] = np.nan
    if event_ids is None:
        event_ids = np.arange(event_count, dtype=np.int64)

    return EventTDOAMeasurements(
        event_ids=np.asarray(event_ids),
        receiver_event_times_s=np.asarray(receiver_event_times_s, dtype=float),
        microphone_ids=ids,
        microphone_pairs=pairs,
        tdoa_s=tdoa,
        sigma_s=pair_sigma,
        confidence=pair_confidence,
        valid=pair_valid,
        measurement_origin="derived_arrivals",
        measurement_basis="reference_star",
        event_samples=event_samples,
        sample_rate_hz=sample_rate_hz,
        event_channel=event_channel,
        primitive_ids=tuple(f"arrival:{microphone_id}" for microphone_id in ids),
        pair_from_primitive=incidence,
        covariance_s2=covariance,
        covariance_model="derived_arrival_diagonal",
        arrival_representatives_s=representatives,
        arrival_valid=primitive_valid,
    )
