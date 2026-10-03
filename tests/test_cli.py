import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_self_calibration.cli import build_parser, main
from acoustic_self_calibration.ground_truth import write_ground_truth_json


def _write_scene(path: Path, *, offset: float = 0.0) -> Path:
    microphones = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source = np.array(
        [
            [2.0, 0.0, 1.0],
            [1.8, 0.3, 1.1],
            [1.5, 0.6, 1.2],
        ]
    )
    translation = np.array([offset, -0.5 * offset, 0.25 * offset])
    return write_ground_truth_json(
        path,
        microphone_positions_m=microphones + translation,
        source_times_s=np.array([0.0, 0.5, 1.0]),
        source_positions_m=source + translation,
    )


def test_calibrate_subcommand_parses_stratified_options() -> None:
    args = build_parser().parse_args(
        [
            "calibrate",
            "recording.wav",
            "-o",
            "run01",
            "-r",
            "truth.json",
            "--event-min-gap-ms",
            "50",
            "--root-start-count",
            "48",
            "--no-temporal-tracking",
        ]
    )
    assert args.command == "calibrate"
    assert args.output == Path("run01")
    assert args.reference == Path("truth.json")
    assert args.event_min_gap_ms == pytest.approx(50.0)
    assert args.root_start_count == 48
    assert args.no_temporal_tracking is True


def test_old_bayesian_solver_options_are_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["calibrate", "recording.wav", "--motion-sigma-mps", "3"])


def test_old_flat_cli_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["recording.wav"])


def test_check_valid_scene(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scene_path = _write_scene(tmp_path / "scene.json")
    assert main(["check", str(scene_path)]) == 0
    output = capsys.readouterr().out
    assert "valid canonical scene JSON" in output
    assert "role: ground_truth" in output
    assert "microphones: 4" in output
    assert "source states: 3" in output


def test_check_invalid_scene_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    assert main(["check", str(path)]) == 2
    assert "invalid scene JSON" in capsys.readouterr().err


def test_compare_writes_json_and_png(tmp_path: Path) -> None:
    estimate = _write_scene(tmp_path / "estimate.json", offset=0.4)
    reference = _write_scene(tmp_path / "reference.json")
    prefix = tmp_path / "comparison"

    assert main(["compare", str(estimate), str(reference), "-o", str(prefix)]) == 0
    json_path = prefix.with_suffix(".json")
    png_path = prefix.with_suffix(".png")
    assert json_path.exists()
    assert png_path.exists()
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["comparison"]["estimate_path"] == str(estimate)
    assert document["comparison"]["reference_path"] == str(reference)
    assert document["evaluation"]["microphones"]["rms_error_m"] < 1e-12
    assert document["evaluation"]["source"]["rms_error_m"] < 1e-12


def test_planar_model_and_constraint_options_parse() -> None:
    args = build_parser().parse_args(
        [
            "calibrate",
            "recording.wav",
            "--model",
            "receiver2d-source3d",
            "--right-angle",
            "3,0,7",
            "--constraint-provenance",
            "survey",
            "--refinement",
            "huber",
            "--refinement-max-nfev",
            "75",
            "--refinement-improvement-tolerance",
            "1e-7",
        ]
    )
    assert args.model == "receiver2d-source3d"
    assert args.right_angle == (3, 0, 7)
    assert args.constraint_provenance == "survey"
    assert args.refinement == "huber"
    assert args.refinement_max_nfev == 75
    assert args.refinement_improvement_tolerance == pytest.approx(1e-7)


def test_cli_help_contains_stratified_terms_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["calibrate", "--help"])
    help_text = capsys.readouterr().out.lower()
    assert "stratified" in help_text
    assert "receiver2d-source3d" in help_text
    assert "refinement" in help_text
    assert "wls" in help_text
    assert "huber" in help_text
    assert "refinement-max-nfev" in help_text
    assert "refinement-improvement-tolerance" in help_text
    assert "bayesian" not in help_text
    assert "posterior" not in help_text
    assert "motion prior" not in help_text
    assert "laplace" not in help_text
