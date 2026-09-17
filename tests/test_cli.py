from pathlib import Path

import pytest

from acoustic_self_calibration.cli import build_parser


def test_cli_parses_metric_distance_prior_and_ground_truth() -> None:
    args = build_parser().parse_args(
        [
            "recording.wav",
            "--ground-truth",
            "truth.json",
            "--estimate-speed-of-sound",
            "--distance-prior",
            "0,1,1.234,0.002",
        ]
    )
    assert args.ground_truth == Path("truth.json")
    assert args.estimate_speed_of_sound is True
    assert len(args.distance_prior) == 1
    prior = args.distance_prior[0]
    assert prior.microphone_a == 0
    assert prior.microphone_b == 1
    assert prior.distance_m == pytest.approx(1.234)
    assert prior.sigma_m == pytest.approx(0.002)


def test_cli_rejects_malformed_distance_prior() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["recording.wav", "--distance-prior", "0,1,1.0"])
