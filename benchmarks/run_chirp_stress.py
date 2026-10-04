"""Direct chirp stress characterization, separate from standard acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from run_chirp_matrix import ROOT, fixture

from acoustic_self_calibration.benchmarking import scene_metrics
from acoustic_self_calibration.chirps import chirp_train
from acoustic_self_calibration.pipeline import calibrate_audio
from acoustic_self_calibration.simulation import render_moving_source

MANIFEST = ROOT / "benchmarks/chirp_stress.json"


def cases(manifest, quick):
    output = []
    for seed in manifest["seeds"][:1] if quick else manifest["seeds"]:
        for count in manifest["microphone_counts"][:1] if quick else manifest["microphone_counts"]:
            for family in manifest["families"]:
                interventions = [
                    x
                    for x in manifest["interventions"]
                    if x != "near-planar-room" or family.startswith("room")
                    if x != "near-collinear-planar" or family == "star"
                ]
                variants = [(x, "float-clean", True) for x in interventions]
                variants += [("none", stage, True) for stage in manifest["noise_stages"]]
                variants += [("none", "float-clean", False), ("rapid-calls", "float-clean", False)]
                for intervention, stage, tracker in variants:
                    c = dict(
                        scenario=family,
                        microphones=count,
                        events=20,
                        span=2.0,
                        seed=seed,
                        stage=stage,
                        intervention=intervention,
                        tracking=tracker,
                    )
                    c["case_id"] = "/".join(
                        map(str, [family, count, seed, intervention, stage, tracker])
                    )
                    output.append(c)
    return output


def run_case(case):
    start = time.perf_counter()
    row: dict[str, Any] = dict(case, standard_success=False, accurate_solved=False)
    try:
        scene, configuration = fixture(case)
        intervention = case["intervention"]
        mics = scene.microphone_positions_m.copy()
        trajectory = scene.trajectory_positions_m.copy()
        events = scene.event_times_s.copy()
        if intervention == "low-motion":
            trajectory[:] = trajectory[0] + 0.001 * (trajectory - trajectory[0])
        elif intervention == "near-planar-room":
            mics[:, 2] = np.mean(mics[:, 2]) + 0.002 * (mics[:, 2] - np.mean(mics[:, 2]))
        elif intervention == "near-collinear-planar":
            mics[:, 2] *= 0.002
        elif intervention == "rapid-calls":
            events = np.r_[0.4, 0.4 + np.cumsum(np.geomspace(0.12, 0.004, len(events) - 1))]
        signal = chirp_train(events, scene.sample_rate_hz, scene.duration_s, seed=case["seed"])
        audio = render_moving_source(
            signal,
            scene.sample_rate_hz,
            mics,
            scene.trajectory_times_s,
            trajectory,
            radiation_pattern="omni",
        )
        sources = np.column_stack(
            [np.interp(events, scene.trajectory_times_s, trajectory[:, k]) for k in range(3)]
        )
        rng = np.random.default_rng(case["seed"] + 19637)
        if "db" in case["stage"]:
            snr = float(case["stage"].split("-")[1].removesuffix("db"))
            for channel in range(len(mics)):
                values = audio[:, channel]
                active = abs(values) > 0.01 * np.max(abs(values))
                power = float(np.mean(values[active] ** 2))
                audio[:, channel] += rng.normal(
                    scale=np.sqrt(power / 10 ** (snr / 10)), size=len(audio)
                )
        if intervention == "gain":
            audio *= np.geomspace(0.02, 5, len(mics))[None, :]
        elif intervention == "quiet-channel":
            audio[:, -1] = 0
        elif intervention == "clipping":
            threshold = 0.15 * np.max(abs(audio), axis=0)
            audio = np.clip(audio, -threshold, threshold)
        elif intervention == "missing-call":
            # Lose one emitted call on half the channels; truth retains the call.
            call = len(events) // 2
            for channel in range(len(mics) // 2):
                center = int(
                    round(
                        (events[call] + np.linalg.norm(sources[call] - mics[channel]) / 343)
                        * scene.sample_rate_hz
                    )
                )
                audio[max(0, center - 150) : center + 150, channel] = 0
        scene = replace(
            scene,
            audio=audio,
            microphone_positions_m=mics,
            trajectory_positions_m=trajectory,
            event_times_s=events,
            source_positions_at_events_m=sources,
        )
        options = json.loads((ROOT / "benchmarks/chirp_matrix.json").read_text())["audio_options"]
        options["use_temporal_tracking"] = case["tracking"]
        row["audio_options"] = options
        result = calibrate_audio(
            audio, scene.sample_rate_hz, array_configuration=configuration, **options
        )
        row.update(scene_metrics(result.calibration, scene, result.emission_times_s))
        row.update(
            status=result.status,
            detected_events=result.detected_event_count,
            measurement_events=len(result.measurements.event_ids),
            unresolved_event_ids=result.measurements.event_ids[
                ~np.isfinite(result.source_positions_m).all(axis=1)
            ].tolist()
            if result.source_positions_m is not None
            else result.measurements.event_ids.tolist(),
            rejection_reasons=list(result.calibration.diagnostics.rejection_reasons),
            accurate_solved=bool(row["standard_success"]),
        )
        row["geometry_within_limits"] = bool(
            row["microphone_rms_error_m"] is not None
            and row["source_rms_error_m"] is not None
            and row["microphone_rms_error_m"] < 0.05
            and row["source_rms_error_m"] < 0.1
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        row.update(status="failed", reason=str(error), exception_type=type(error).__name__)
    row["elapsed_s"] = time.perf_counter() - start
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/results/chirp_stress.json"
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    expected = cases(manifest, args.quick)
    report: dict[str, Any] = dict(
        schema_version=1,
        purpose="stress characterization, not standard acceptance",
        manifest=manifest,
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        base_manifest_sha256=hashlib.sha256(
            (ROOT / "benchmarks/chirp_matrix.json").read_bytes()
        ).hexdigest(),
        uv_lock_sha256=hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        source_tree_sha256=hashlib.sha256(
            b"".join(
                str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes()
                for p in sorted((ROOT / "src").rglob("*.py"))
            )
        ).hexdigest(),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        expected_case_ids=[c["case_id"] for c in expected],
        results=[],
        complete=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_case, c) for c in expected]):
            row = future.result()
            report["results"].append(row)
            report["results"].sort(key=lambda r: r["case_id"])
            report["complete"] = len(report["results"]) == len(expected)
            report["accurate_solved"] = sum(r["accurate_solved"] for r in report["results"])
            report["solved_outside_standard_limits"] = sum(
                r["status"] == "solved" and not r["accurate_solved"] for r in report["results"]
            )
            report["solved_geometry_outside_limits"] = sum(
                r["status"] == "solved" and not r.get("geometry_within_limits", False)
                for r in report["results"]
            )
            report["solved_incomplete_reference_coverage"] = sum(
                r["status"] == "solved" and r.get("unresolved_events", 20) > 0
                for r in report["results"]
            )
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(row["case_id"], row["status"], flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
