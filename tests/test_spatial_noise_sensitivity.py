"""Geometric diversity and rigid gauge must affect uncertainty correctly."""

import numpy as np

from acoustic_self_calibration.chirps import make_chirp_scene
from acoustic_self_calibration.measurements import reference_star_from_arrivals
from acoustic_self_calibration.stratified.refinement import estimate_spatial_noise_sensitivity


def measurements(microphones, sources):
    arrivals = np.linalg.norm(sources[:, None] - microphones[None], axis=2) / 343
    return reference_star_from_arrivals(
        arrivals - arrivals[:, [0]],
        np.full_like(arrivals, 1e-6),
        receiver_event_times_s=np.arange(len(sources), dtype=float),
        reference_microphone=0,
    )


def test_spatial_uncertainty_detects_low_motion_and_removes_rigid_gauge():
    scene = make_chirp_scene(geometry="room", layout="irregular", seed=10)
    m, sources = scene.microphone_positions_m, scene.source_positions_at_events_m
    data = measurements(m, sources)
    diverse = estimate_spatial_noise_sensitivity(m, sources, data)
    assert diverse.observable_rank == diverse.parameter_count == 3 * len(m) - 6
    assert diverse.relative_weakest_std < 0.05
    quiet = sources[0] + 0.001 * (sources - sources[0])
    limited = estimate_spatial_noise_sensitivity(m, quiet, measurements(m, quiet))
    assert limited.relative_weakest_std > 0.05
    rotation, _ = np.linalg.qr(np.random.default_rng(572).normal(size=(3, 3)))
    translated = m @ rotation + np.array([71.0, -8.0, 19.0])
    shifted_sources = sources @ rotation + np.array([71.0, -8.0, 19.0])
    transformed = estimate_spatial_noise_sensitivity(translated, shifted_sources, data)
    np.testing.assert_allclose(
        transformed.relative_weakest_std, diverse.relative_weakest_std, rtol=1e-8
    )
    np.testing.assert_allclose(
        transformed.weakest_microphone_rms_std_m, diverse.weakest_microphone_rms_std_m, rtol=1e-8
    )
