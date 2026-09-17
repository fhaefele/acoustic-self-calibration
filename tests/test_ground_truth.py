import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_self_calibration.ground_truth import (
    ground_truth_from_dict,
    load_ground_truth_json,
    make_ground_truth_dict,
    validate_ground_truth_json,
    write_ground_truth_json,
)


def _data():
    microphones = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    times = np.array([0.0, 0.5, 1.0])
    source = np.array([[2.0, 0.0, 1.0], [1.8, 0.3, 1.1], [1.5, 0.6, 1.2]])
    return microphones, times, source


def test_ground_truth_helpers_round_trip_json(tmp_path: Path) -> None:
    microphones, times, source = _data()
    document = make_ground_truth_dict(
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=source,
        metadata={"name": "trial-01"},
    )
    assert document["schema_version"] == 1
    assert document["scene_role"] == "ground_truth"
    assert document["metadata"]["name"] == "trial-01"

    path = write_ground_truth_json(
        tmp_path / "gt.json",
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=source,
        metadata={"name": "trial-01"},
    )
    parsed_json = json.loads(path.read_text(encoding="utf-8"))
    assert parsed_json["scene"]["source"]["times_s"] == [0.0, 0.5, 1.0]

    loaded = validate_ground_truth_json(path)
    assert np.allclose(loaded.microphone_positions_m, microphones)
    assert np.allclose(loaded.source_times_s, times)
    assert np.allclose(loaded.source_positions_m, source)
    assert loaded.metadata["name"] == "trial-01"
    assert loaded.scene_role == "ground_truth"
    assert load_ground_truth_json(path).scene_role == "ground_truth"


def test_ground_truth_parser_accepts_estimate_scene_role() -> None:
    microphones, times, source = _data()
    document = make_ground_truth_dict(
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=source,
    )
    document["scene_role"] = "estimate"
    parsed = ground_truth_from_dict(document)
    assert parsed.scene_role == "estimate"
    assert np.allclose(parsed.microphone_positions_m, microphones)


def test_ground_truth_parser_rejects_bad_schema_version() -> None:
    microphones, times, source = _data()
    document = make_ground_truth_dict(
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=source,
    )
    document["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        ground_truth_from_dict(document)


def test_ground_truth_parser_rejects_old_pre_scene_schema() -> None:
    microphones, times, source = _data()
    old_schema = {
        "schema_version": 1,
        "scene_role": "ground_truth",
        "microphones": {"positions_m": microphones.tolist()},
        "source": {"times_s": times.tolist(), "positions_m": source.tolist()},
    }
    with pytest.raises(ValueError, match="scene.microphones.positions_m"):
        ground_truth_from_dict(old_schema)


def test_ground_truth_requires_strictly_increasing_source_times() -> None:
    microphones, _, source = _data()
    with pytest.raises(ValueError, match="strictly increasing"):
        make_ground_truth_dict(
            microphone_positions_m=microphones,
            source_times_s=np.array([0.0, 0.5, 0.5]),
            source_positions_m=source,
        )
