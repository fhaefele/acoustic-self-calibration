import numpy as np
import pytest

from acoustic_self_calibration.simulation import (
    make_planar_benchmark_pulse_scene,
    make_room_benchmark_pulse_scene,
    planar_benchmark_microphones,
    render_moving_source,
    room_benchmark_floor_polygon,
    room_benchmark_microphones,
)


def test_renderer_shape_and_finite_values():
    fs = 8_000
    n = 8_000
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    mics = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 0.2],
            [0.3, -0.8, 1.2],
            [1.2, 0.4, 0.7],
        ]
    )
    tt = np.array([0.0, 0.5, 1.0])
    ss = np.array([[-1.0, 0.0, 1.0], [0.0, 0.5, 1.2], [1.0, 1.0, 1.4]])
    audio = render_moving_source(signal, fs, mics, tt, ss, radiation_pattern="cardioid")
    assert audio.shape == (n, len(mics))
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_cardioid_radiates_more_forward_than_backward():
    from acoustic_self_calibration.radiation import radiation_gain

    gains = radiation_gain(np.array([1.0, 0.0, -1.0]), "cardioid")
    assert gains[0] > gains[1] > gains[2]
    assert gains[0] == 1.0
    assert gains[2] == 0.0


@pytest.mark.parametrize("layout", ["cross", "star"])
@pytest.mark.parametrize("span,distances", [(2.0, (1.0, 3.0)), (4.0, (1.0, 6.0))])
@pytest.mark.parametrize("count", [8, 12, 16, 24])
def test_planar_benchmark_layout_and_source_region(layout, span, distances, count):
    scene = make_planar_benchmark_pulse_scene(
        count,
        array_span_m=span,
        source_distance_range_m=distances,
        layout=layout,
        sample_rate_hz=8_000,
        seed=19,
    )
    microphones = scene.microphone_positions_m
    assert microphones.shape == (count, 3)
    assert len(np.unique(microphones, axis=0)) == count
    np.testing.assert_allclose(microphones[:, 1], 0.0)
    np.testing.assert_allclose(np.ptp(microphones[:, [0, 2]], axis=0), span)
    x, z = microphones[:, 0], microphones[:, 2]
    conic_design = np.column_stack((x * x, x * z, z * z, x, z, np.ones(count)))
    assert np.linalg.matrix_rank(conic_design) == (5 if layout == "cross" else 6)
    heights = scene.trajectory_positions_m[:, 1]
    assert heights.min() >= distances[0]
    assert heights.max() <= distances[1]
    assert (
        np.linalg.matrix_rank(
            scene.source_positions_at_events_m - scene.source_positions_at_events_m.mean(axis=0)
        )
        == 3
    )
    assert scene.audio.shape == (40_000, count)
    assert np.isfinite(scene.audio).all()
    assert len(scene.event_times_s) == 40


@pytest.mark.parametrize("layout", ["cross", "star"])
def test_planar_benchmark_microphone_counts_are_nested(layout):
    largest = planar_benchmark_microphones(24, array_span_m=4.0, layout=layout)
    for count in (8, 12, 16):
        np.testing.assert_array_equal(
            planar_benchmark_microphones(count, array_span_m=4.0, layout=layout), largest[:count]
        )


@pytest.mark.parametrize("layout", ["rectangular", "irregular"])
@pytest.mark.parametrize("count", [8, 12, 16, 24])
def test_room_benchmark_walls_and_source_containment(layout, count):
    scene = make_room_benchmark_pulse_scene(count, layout=layout, seed=19, sample_rate_hz=8_000)
    vertices = room_benchmark_floor_polygon(layout=layout)
    edges = np.roll(vertices, -1, axis=0) - vertices

    def signed_wall_distances(points):
        delta = points[:, None, :2] - vertices[None, :, :]
        return (
            edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
        ) / np.linalg.norm(edges, axis=1)

    microphones = scene.microphone_positions_m
    wall_distances = signed_wall_distances(microphones)
    assert np.all(wall_distances >= -1e-12)
    np.testing.assert_allclose(wall_distances.min(axis=1), 0.0, atol=1e-12)
    assert len(np.unique(microphones, axis=0)) == count
    assert np.linalg.matrix_rank(microphones - microphones.mean(axis=0)) == 3
    for positions in (scene.trajectory_positions_m, scene.source_positions_at_events_m):
        assert np.all(signed_wall_distances(positions) > 0.0)
        assert np.all((positions[:, 2] > 0.0) & (positions[:, 2] < 3.0))
        assert np.linalg.matrix_rank(positions - positions.mean(axis=0)) == 3
    assert scene.audio.shape == (40_000, count)
    assert np.isfinite(scene.audio).all()
    assert np.max(np.abs(scene.audio)) > 0.01


@pytest.mark.parametrize("layout", ["rectangular", "irregular"])
def test_room_benchmark_repeatability_and_nested_counts(layout):
    largest = room_benchmark_microphones(24, layout=layout, seed=42)
    for count in (8, 12, 16):
        np.testing.assert_array_equal(
            room_benchmark_microphones(count, layout=layout, seed=42), largest[:count]
        )
    first = make_room_benchmark_pulse_scene(8, layout=layout, seed=42, sample_rate_hz=2_000)
    repeat = make_room_benchmark_pulse_scene(8, layout=layout, seed=42, sample_rate_hz=2_000)
    more = make_room_benchmark_pulse_scene(12, layout=layout, seed=42, sample_rate_hz=2_000)
    np.testing.assert_array_equal(first.audio, repeat.audio)
    np.testing.assert_array_equal(
        first.source_positions_at_events_m, more.source_positions_at_events_m
    )
    assert not np.array_equal(largest, room_benchmark_microphones(24, layout=layout, seed=43))
