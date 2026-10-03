import numpy as np

from acoustic_self_calibration.simulation import make_random_3d_pulse_scene
from acoustic_self_calibration.stratified.offsets_minimal import _halton_starts, solve_offsets_7r6s


def _fixture(event_count: int):
    scene = make_random_3d_pulse_scene(8, event_count=event_count)
    seed_events = np.linspace(0, event_count - 1, 6, dtype=int)
    microphones = scene.microphone_positions_m[:7]
    sources = scene.source_positions_at_events_m[seed_events]
    ranges = np.linalg.norm(
        microphones[:, None, :] - sources[None, :, :],
        axis=2,
    )
    arrivals = ranges - ranges[[0], :]
    return arrivals, ranges


def test_7r6s_finds_physical_exact_root_on_20_event_fixture() -> None:
    arrivals, ranges = _fixture(20)
    result = solve_offsets_7r6s(arrivals, start_count=48)

    assert result.roots
    errors = [np.sqrt(np.mean((root.corrected_ranges_m - ranges) ** 2)) for root in result.roots]
    assert min(errors) < 1e-8
    best = result.roots[int(np.argmin(errors))]
    assert best.jacobian_rank == 6
    assert best.all_minor_max_abs < 1e-7
    assert result.diagnostics.template_sha256


def test_7r6s_is_deterministic() -> None:
    arrivals, _ = _fixture(40)
    first = solve_offsets_7r6s(arrivals, start_count=32)
    second = solve_offsets_7r6s(arrivals, start_count=32)

    assert len(first.roots) == len(second.roots)
    for left, right in zip(first.roots, second.roots, strict=True):
        assert np.allclose(left.offsets_m, right.offsets_m, atol=1e-10, rtol=1e-10)


def test_7r6s_rejects_wrong_shape() -> None:
    try:
        solve_offsets_7r6s(np.zeros((8, 6)))
    except ValueError as error:
        assert "requires shape" in str(error)
    else:
        raise AssertionError("wrong solver shape must be rejected")


def test_7r6s_finds_physical_exact_root_on_40_event_fixture() -> None:
    arrivals, ranges = _fixture(40)
    result = solve_offsets_7r6s(arrivals, start_count=64)

    assert result.roots
    errors = [np.sqrt(np.mean((root.corrected_ranges_m - ranges) ** 2)) for root in result.roots]
    assert min(errors) < 1e-8
    best = result.roots[int(np.argmin(errors))]
    assert best.jacobian_rank == 6
    assert best.all_minor_max_abs < 1e-7
    assert result.diagnostics.verified_root_count == len(result.roots)


def test_halton_starts_are_prefix_stable() -> None:
    arrivals, _ = _fixture(20)
    for physical_only in (True, False):
        counts = (8, 24, 33, 64, 100)
        starts = [
            _halton_starts(arrivals, start_count=count, physical_only=physical_only)[0]
            for count in counts
        ]
        for index, count in enumerate(counts):
            assert starts[index].shape[0] == count
        for small, large in zip(starts, starts[1:], strict=False):
            assert np.allclose(small, large[: small.shape[0]], atol=1e-12, rtol=1e-12)
