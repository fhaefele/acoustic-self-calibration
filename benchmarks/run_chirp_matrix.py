"""Preregistered chirp acceptance with checkpointing, failures and expected case IDs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.spatial import ConvexHull

from acoustic_self_calibration.array_configuration import ArrayConfiguration
from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.chirps import chirp_train, make_chirp_scene
from acoustic_self_calibration.measurements import reference_star_from_arrivals
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.simulation import render_moving_source
from acoustic_self_calibration.stratified.solver import calibrate_planar_tdoa, calibrate_tdoa
from acoustic_self_calibration.stratified.structured import refine_structured
from acoustic_self_calibration.wav import calibrate_wav

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/chirp_matrix.json"


def case_id(case):
    return "/".join(
        str(case[k]) for k in ("scenario", "microphones", "events", "span", "seed", "stage")
    )


def cases(manifest, suite):
    if suite == "pr":
        subset = manifest["pr_subset"]
        dimensions = [
            subset[k] for k in ("scenarios", "counts", "events", "span_m", "seeds", "stages")
        ]
    else:
        seeds = (
            manifest["development_seeds"]
            if suite == "development"
            else manifest["evaluation_seeds"]
        )
        scenarios = (
            manifest["standard_scenarios"]
            + manifest["negative_scenarios"]
            + manifest["stress_scenarios"]
        )
        dimensions = [
            scenarios,
            sorted(set(manifest["microphone_counts"] + manifest["grid_microphone_counts"])),
            manifest["event_counts"],
            manifest["planar_spans_m"],
            seeds,
            manifest["stages"],
        ]
    output = []
    for scenario, count, events, span, seed, stage in itertools.product(*dimensions):
        if "grid" not in scenario and count not in manifest["microphone_counts"]:
            continue
        if scenario.startswith("room") and span != 2.0:
            continue
        if scenario == "thin-grid" and count != 8:
            continue
        c = dict(
            scenario=scenario, microphones=count, events=events, span=span, seed=seed, stage=stage
        )
        c["case_id"] = case_id(c)
        output.append(c)
    return output


def fixture(case):
    n, seed, span = case["microphones"], case["seed"], case["span"]
    scenario = case["scenario"]
    rng = np.random.default_rng(seed)
    if scenario.startswith("room"):
        layout = "irregular" if scenario == "room-irregular" else "rectangular"
        scene = make_chirp_scene(
            n, geometry="room", layout=layout, seed=seed, event_count=case["events"]
        )
        config = ArrayConfiguration("room")
        if scenario == "room-floor-ceiling":
            positions = scene.trajectory_positions_m.copy()
            positions[:, 2] = 0.1 + 2.8 * (
                0.5 + 0.5 * np.sin(3 * np.pi * scene.trajectory_times_s / scene.duration_s)
            )
            scene = replace(scene, trajectory_positions_m=positions)
    else:
        # Source motion is generated independently of construction membership.
        scene = make_chirp_scene(
            8 if "grid" in scenario else n,
            seed=seed,
            event_count=case["events"],
            array_span_m=span,
            source_distance_range_m=(1.0, 3.0 if span == 2 else 6.0),
            rapid=scenario == "rapid",
        )
        config = ArrayConfiguration("arbitrary-planar")
        if "grid" in scenario:
            rows, cols = (
                {8: (3, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4), 24: (4, 6)}[n]
                if scenario != "thin-grid"
                else (2, 4)
            )
            slots = [(r, c) for r in range(rows) for c in range(cols)][:n]
            equal = scenario == "grid-equal"
            x = (
                np.linspace(-span / 2, span / 2, cols)
                if equal
                else np.r_[0, np.cumsum(rng.uniform(0.5, 1.5, cols - 1))]
            )
            z = (
                np.linspace(-span * 0.4, span * 0.4, rows)
                if equal
                else np.r_[0, np.cumsum(rng.uniform(0.5, 1.5, rows - 1))]
            )
            x = (x - x.mean()) / np.ptp(x) * span
            z = (z - z.mean()) / np.ptp(z) * span * 0.8
            m = np.array([[x[c], 0, z[r]] for r, c in slots])
            config = ArrayConfiguration(
                "grid",
                provenance="synthetic declared grid topology",
                grid_slots=tuple((i, r, c) for i, (r, c) in enumerate(slots)),
                equal_row_spacing=equal,
                equal_column_spacing=equal,
            )
        elif (
            scenario.startswith("cross")
            or scenario.startswith("t")
            or scenario in {"blind-cross", "near-cross"}
        ):
            theta = (
                90 if scenario in {"blind-cross", "near-cross"} else int(scenario.lstrip("crosst"))
            )
            a_count = n // 2 + 1
            a = np.linspace(-span / 2, span / 2, a_count)
            junction = int(np.argmin(abs(a)))
            a -= a[junction]
            b = (
                np.linspace(0.2, span * 0.7, n - a_count)
                if scenario.startswith("t")
                else np.linspace(-span * 0.5, span * 0.5, n - a_count)
            )
            b[np.abs(b) < 0.01] = 0.12
            m = np.zeros((n, 3))
            m[:a_count, 0] = a
            m[a_count:, 0] = b * np.cos(np.deg2rad(theta))
            m[a_count:, 2] = b * np.sin(np.deg2rad(theta))
            if scenario == "near-cross":
                m[:, 2] += rng.normal(scale=0.002, size=n)
            else:
                config = ArrayConfiguration(
                    "t" if scenario.startswith("t") else "cross",
                    provenance="synthetic declared two-line construction",
                    arms=(tuple(range(a_count)), tuple([junction, *range(a_count, n)])),
                    rays=None
                    if scenario == "blind-cross"
                    else ((junction, a_count - 1), (junction, n - 1)),
                    angle_deg=None if scenario == "blind-cross" else theta,
                )
        elif scenario == "generic":
            m = np.column_stack(
                [
                    rng.uniform(-span / 2, span / 2, n),
                    np.zeros(n),
                    rng.uniform(-span / 2, span / 2, n),
                ]
            )
        else:
            m = scene.microphone_positions_m
        scene = replace(scene, microphone_positions_m=m)
    signal = chirp_train(scene.event_times_s, scene.sample_rate_hz, scene.duration_s, seed=seed)
    audio = render_moving_source(
        signal,
        scene.sample_rate_hz,
        scene.microphone_positions_m,
        scene.trajectory_times_s,
        scene.trajectory_positions_m,
        radiation_pattern="omni",
    )
    sources = np.column_stack(
        [
            np.interp(
                scene.event_times_s, scene.trajectory_times_s, scene.trajectory_positions_m[:, k]
            )
            for k in range(3)
        ]
    )
    scene = replace(scene, audio=audio, source_positions_at_events_m=sources)
    if "40db" in case["stage"]:
        for channel in range(n):
            values = audio[:, channel]
            active = abs(values) > 0.01 * np.max(abs(values))
            power = np.mean(values[active] ** 2)
            audio[:, channel] += rng.normal(scale=np.sqrt(power / 1e4), size=len(audio))
    return scene, config


def run_case(case):
    manifest = json.loads(MANIFEST.read_text())
    start = time.perf_counter()
    row = {**case, "standard_success": False}
    try:
        scene, config = fixture(case)
        stage = case["stage"]
        row["configuration"] = config.to_dict()
        if stage in {"exact", "arrival-noise"}:
            arrivals = (
                np.linalg.norm(
                    scene.source_positions_at_events_m[:, None]
                    - scene.microphone_positions_m[None, :],
                    axis=2,
                )
                / 343
            )
            sigma = manifest["arrival_noise_sigma_s"] if stage == "arrival-noise" else 1e-7
            if stage == "arrival-noise":
                arrivals += np.random.default_rng(case["seed"] + 1713).normal(
                    scale=sigma, size=arrivals.shape
                )
            measurements = reference_star_from_arrivals(
                arrivals - arrivals[:, [0]],
                np.full_like(arrivals, sigma),
                receiver_event_times_s=scene.event_times_s,
                reference_microphone=0,
            )
            if config.name == "room":
                calibration = calibrate_tdoa(measurements)
            else:
                calibration = calibrate_planar_tdoa(
                    measurements,
                    angle_constraint=config.angle_constraint(),
                    source_half_space_sign=1,
                )
                if config.name in {"cross", "t", "grid"} and config.angle_constraint() is not None:
                    calibration = refine_structured(
                        calibration, measurements, config, speed_of_sound=343.0
                    )
            times = scene.event_times_s
        else:
            options = dict(manifest["audio_options"], array_configuration=config)
            if stage.startswith("pcm"):
                normalized = scene.audio / max(float(np.max(abs(scene.audio))), 1e-12)
                with tempfile.TemporaryDirectory(prefix="asc-chirp-") as temporary:
                    path = Path(temporary) / "scene.wav"
                    wavfile.write(
                        path,
                        scene.sample_rate_hz,
                        np.round(normalized * 0.95 * 32767).astype(np.int16),
                    )
                    result = calibrate_wav(path, **options)
            else:
                result = calibrate_audio(scene.audio, scene.sample_rate_hz, **options)
            calibration = result.calibration
            times = result.emission_times_s
        row.update(scene_metrics(calibration, scene, times))
        row["status"] = calibration.status
        row["tdoa_rms_s"] = calibration.tdoa_rms_s
        row["rejection_reasons"] = list(calibration.diagnostics.rejection_reasons)
        if config.name == "room":
            hull = ConvexHull(scene.microphone_positions_m)
            inside = (
                np.max(
                    scene.source_positions_at_events_m @ hull.equations[:, :3].T
                    + hull.equations[:, 3],
                    axis=1,
                )
                <= 1e-9
            )
            row["outside_microphone_hull_events"] = int(np.sum(~inside))
        negative = case["scenario"] in manifest["negative_scenarios"]
        stress = case["scenario"] in manifest["stress_scenarios"]
        row["acceptance_class"] = "negative" if negative else "stress" if stress else "standard"
        if stage == "exact" and not negative and not stress:
            row["exact_success"] = bool(
                row["standard_success"]
                and row["microphone_rms_error_m"] < 0.001
                and row["source_rms_error_m"] < 0.001
                and calibration.tdoa_rms_s < 1e-8
            )
        row["accepted"] = (
            calibration.status in {"ambiguous", "degenerate"}
            if negative
            else calibration.status != "solved" or row["standard_success"]
            if stress
            else row.get("exact_success", row["standard_success"])
        )
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
        row.update(
            status="failed", exception_type=type(error).__name__, reason=str(error), accepted=False
        )
    row["elapsed_s"] = time.perf_counter() - start
    row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pr", "development", "evaluation"], default="pr")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stages", nargs="+")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--counts", nargs="+", type=int)
    parser.add_argument("--events", nargs="+", type=int)
    parser.add_argument("--spans", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/chirp_pr.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    expected = cases(manifest, args.suite)
    if args.stages:
        expected = [c for c in expected if c["stage"] in args.stages]
    if args.scenarios:
        expected = [c for c in expected if c["scenario"] in args.scenarios]
    for option, key in (
        ("counts", "microphones"),
        ("events", "events"),
        ("spans", "span"),
        ("seeds", "seed"),
    ):
        selected = getattr(args, option)
        if selected:
            expected = [c for c in expected if c[key] in selected]
    if not expected:
        parser.error("selected filters match no preregistered cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = (
        json.loads(args.output.read_text())["results"]
        if args.resume and args.output.exists()
        else []
    )
    completed = {r["case_id"] for r in results}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=ROOT)
    source_fingerprint = hashlib.sha256(source_diff).hexdigest()
    manifest_fingerprint = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    source_files = sorted((ROOT / "src").rglob("*.py"))
    source_tree_fingerprint = hashlib.sha256(
        b"".join(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() for p in source_files)
    ).hexdigest()
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous.get("source_tree_sha256") != source_tree_fingerprint
            or previous.get("manifest_sha256") != manifest_fingerprint
            or previous.get("expected_case_ids") != [c["case_id"] for c in expected]
        ):
            raise ValueError("cannot resume results from different solver code")
    report = {
        "schema_version": 1,
        "source_diff_sha256": source_fingerprint,
        "source_tree_sha256": source_tree_fingerprint,
        "suite": args.suite,
        "revision": revision,
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "manifest_sha256": manifest_fingerprint,
        "manifest": manifest,
        "expected_case_ids": [c["case_id"] for c in expected],
        "results": results,
    }

    def checkpoint():
        report["results"] = sorted(results, key=lambda r: r["case_id"])
        ids = {r["case_id"] for r in results}
        report["complete"] = ids == {c["case_id"] for c in expected}
        report["passed"] = sum(bool(r["accepted"]) for r in results)
        report["failed"] = sum(not bool(r["accepted"]) for r in results)
        report["accepted"] = report["complete"] and report["failed"] == 0
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c): c for c in expected if c["case_id"] not in completed}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            checkpoint()
            print(
                result["case_id"],
                result["status"],
                "PASS" if result["accepted"] else "FAIL",
                flush=True,
            )
    checkpoint()
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
