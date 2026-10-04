"""Export standalone truth/estimate plots; references are evaluation-only."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from run_chirp_matrix import ROOT, fixture
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.geometry import apply_rigid, rigid_align
from acoustic_self_calibration.pipeline import calibrate_audio


def plot(case, output):
    scene, config = fixture(case)
    options = json.loads((ROOT / "benchmarks/chirp_matrix.json").read_text())["audio_options"]
    result = calibrate_audio(
        scene.audio, scene.sample_rate_hz, array_configuration=config, **options
    )
    row = scene_metrics(result.calibration, scene, result.emission_times_s)
    fig = plt.figure(figsize=(11, 4.8))
    geometry = fig.add_subplot(121, projection="3d")
    errors = fig.add_subplot(122)
    truth_mics = scene.microphone_positions_m
    truth_sources = scene.source_positions_at_events_m
    geometry.scatter(*truth_mics.T, marker="o", facecolors="none", label="true microphones")
    geometry.scatter(*truth_sources.T, s=15, label="true emitted calls")
    if result.microphone_positions_m is not None and result.source_positions_m is not None:
        aligned, rotation, translation = rigid_align(result.microphone_positions_m, truth_mics)
        sources = apply_rigid(result.source_positions_m, rotation, translation)
        center = truth_mics.mean(axis=0)
        _, singular, vt = np.linalg.svd(truth_mics - center)
        if singular[-1] < 1e-8 * max(singular[0], 1):
            normal = vt[-1]
            finite = np.isfinite(sources).all(axis=1)
            if (
                np.any(finite)
                and np.mean((sources[finite] - center) @ normal)
                * np.mean((truth_sources - center) @ normal)
                < 0
            ):
                sources -= 2 * ((sources - center) @ normal)[:, None] * normal
            xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 2), np.linspace(-1.2, 1.2, 2))
            plane = center + xx[..., None] * vt[0] + yy[..., None] * vt[1]
            geometry.plot_surface(
                plane[:, :, 0], plane[:, :, 1], plane[:, :, 2], alpha=0.08, color="gray"
            )
        else:
            low, high = np.zeros(3), np.array([6, 5, 3])
            corners = np.array(
                [
                    [x, y, z]
                    for x in [low[0], high[0]]
                    for y in [low[1], high[1]]
                    for z in [low[2], high[2]]
                ]
            )
            for i, a in enumerate(corners):
                for b in corners[i + 1 :]:
                    if np.sum(a != b) == 1:
                        geometry.plot(*np.stack([a, b]).T, color="gray", alpha=0.3, lw=0.6)
        geometry.scatter(*aligned.T, marker="x", label="estimated microphones")
        finite = np.isfinite(sources).all(axis=1) & np.isfinite(result.emission_times_s)
        geometry.scatter(*sources[finite].T, marker="+", label="estimated calls")
        distance = abs(result.emission_times_s[finite, None] - scene.event_times_s[None, :])
        estimated, truth = linear_sum_assignment(distance)
        keep = distance[estimated, truth] < min(0.02, 0.45 * np.min(np.diff(scene.event_times_s)))
        estimated, truth = np.flatnonzero(finite)[estimated[keep]], truth[keep]
        errors.scatter(
            scene.event_times_s[truth],
            np.linalg.norm(sources[estimated] - truth_sources[truth], axis=1) * 100,
            s=18,
            label="source error",
        )
        missing = np.setdiff1d(np.arange(len(truth_sources)), truth)
        errors.scatter(
            scene.event_times_s[missing],
            np.full(len(missing), 12.0),
            marker="x",
            c="red",
            label="unresolved / unmatched",
        )
    else:
        errors.text(0.5, 0.5, "geometry unavailable", ha="center", transform=errors.transAxes)
    errors.axhline(10, color="gray", ls="--", lw=0.8, label="10 cm target")
    errors.set(xlabel="emission feature time (s)", ylabel="source error (cm)")
    geometry.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    geometry.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.03))
    errors.legend(fontsize=7)
    fig.suptitle(
        f"{case['scenario']}, {case['microphones']} mics, seed {case['seed']}: {result.status}; {row['matched_events']}/{row['expected_events']} calls"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(Path("/tmp") / f"asc-review-{case['scenario']}.png")
    plt.close(fig)
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                case=case,
                metrics=row,
                options=options,
                revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def plot_ambiguity(output):
    reference = json.loads((ROOT / "data/myotis/myotis_gt.json").read_text())["scene"]
    microphones = np.asarray(reference["microphones"]["positions_m"])
    sources = np.asarray(reference["source"]["positions_m"])
    ranges = np.linalg.norm(sources[:, None] - microphones[None], axis=2)
    fig = plt.figure(figsize=(12, 4))
    for index, degrees in enumerate([60, 90, 120], start=1):
        theta = np.deg2rad(degrees)
        transform = np.array([[1, np.cos(theta)], [0, np.sin(theta)]])
        alternative_mics, alternative_sources = microphones.copy(), sources.copy()
        alternative_mics[:, [0, 2]] = microphones[:, [0, 2]] @ transform.T
        alternative_sources[:, [0, 2]] = sources[:, [0, 2]] @ np.linalg.inv(transform)
        alternative_sources[:, 1] = np.sqrt(
            np.sum(sources**2, axis=1) - np.sum(alternative_sources[:, [0, 2]] ** 2, axis=1)
        )
        error = np.max(
            abs(
                np.linalg.norm(alternative_sources[:, None] - alternative_mics[None], axis=2)
                - ranges
            )
        )
        axis = fig.add_subplot(1, 3, index, projection="3d")
        axis.scatter(*alternative_mics.T, marker="x", label="12 fixed microphones")
        axis.scatter(*alternative_sources.T, s=10, label="41 source calls, same side")
        axis.set(
            xlabel="x (m)",
            ylabel="normal (m)",
            zlabel="in-plane (m)",
            title=f"{degrees}°; max range difference {error:.1e} m",
        )
        axis.set(xlim=(-1.6, 1.6), ylim=(0, 3), zlim=(-1.6, 1.6))
        axis.set_box_aspect((3.2, 3, 3.2))
        axis.legend(fontsize=6)
    fig.suptitle("Unknown cross angle: different complete scenes with identical direct ranges")
    fig.tight_layout()
    fig.savefig(output / "cross_ambiguity.svg")
    fig.savefig("/tmp/asc-review-cross-ambiguity.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/figures")
    args = parser.parse_args()
    for scenario, count in [("star", 8), ("cross90", 8), ("grid-free", 9), ("room-rectangular", 8)]:
        case = dict(
            scenario=scenario, microphones=count, events=20, span=2.0, seed=10, stage="float-clean"
        )
        plot(case, args.output / f"{scenario}_{count}.svg")
    plot_ambiguity(args.output)


if __name__ == "__main__":
    main()
