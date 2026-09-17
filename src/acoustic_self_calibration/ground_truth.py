from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_VALID_SCENE_ROLES = {"ground_truth", "estimate"}


@dataclass(frozen=True)
class GroundTruth:
    """Validated reference scene used for ground-truth evaluation."""

    microphone_positions_m: np.ndarray
    source_times_s: np.ndarray
    source_positions_m: np.ndarray
    metadata: dict[str, Any]
    scene_role: str = "ground_truth"


def _validate_ground_truth_arrays(
    microphone_positions_m: np.ndarray,
    source_times_s: np.ndarray,
    source_positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    microphones = np.asarray(microphone_positions_m, dtype=float)
    times = np.asarray(source_times_s, dtype=float).reshape(-1)
    source = np.asarray(source_positions_m, dtype=float)

    if microphones.ndim != 2 or microphones.shape[1] != 3 or microphones.shape[0] < 4:
        raise ValueError(
            "microphone_positions_m must have shape (microphones, 3) with at least 4 microphones"
        )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_positions_m must have shape (samples, 3)")
    if source.shape[0] != times.shape[0] or times.size < 2:
        raise ValueError(
            "source_times_s and source_positions_m must contain the same number of samples"
        )
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("source_times_s must be strictly increasing")
    if not np.all(np.isfinite(microphones)):
        raise ValueError("microphone_positions_m contains non-finite values")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(source)):
        raise ValueError("source ground truth contains non-finite values")
    return microphones, times, source


def _scene_document(
    *,
    microphone_positions_m: np.ndarray,
    source_times_s: np.ndarray,
    source_positions_m: np.ndarray,
    scene_role: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scene_role not in _VALID_SCENE_ROLES:
        raise ValueError(f"scene_role must be one of {sorted(_VALID_SCENE_ROLES)}")
    microphones, times, source = _validate_ground_truth_arrays(
        microphone_positions_m,
        source_times_s,
        source_positions_m,
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "scene_role": scene_role,
        "scene": {
            "microphones": {"positions_m": microphones.tolist()},
            "source": {
                "times_s": times.tolist(),
                "positions_m": source.tolist(),
            },
        },
    }
    if metadata is not None:
        document["metadata"] = metadata
    json.dumps(document, allow_nan=False)
    return document


def make_ground_truth_dict(
    *,
    microphone_positions_m: np.ndarray,
    source_times_s: np.ndarray,
    source_positions_m: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated ground-truth document using the canonical scene schema."""
    return _scene_document(
        microphone_positions_m=microphone_positions_m,
        source_times_s=source_times_s,
        source_positions_m=source_positions_m,
        scene_role="ground_truth",
        metadata=metadata,
    )


def write_ground_truth_json(
    path: str | Path,
    *,
    microphone_positions_m: np.ndarray,
    source_times_s: np.ndarray,
    source_positions_m: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write validated ground truth using the canonical scene JSON schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = make_ground_truth_dict(
        microphone_positions_m=microphone_positions_m,
        source_times_s=source_times_s,
        source_positions_m=source_positions_m,
        metadata=metadata,
    )
    target.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def ground_truth_from_dict(document: dict[str, Any]) -> GroundTruth:
    """Parse a canonical scene document for use as an evaluation reference.

    Both ``scene_role: ground_truth`` and ``scene_role: estimate`` are accepted.
    This intentionally allows a calibration result JSON to be reused directly as
    the reference for a later run. The previous pre-``scene`` schema is rejected.
    """
    if document.get("schema_version") != 1:
        raise ValueError("scene schema_version must be 1")
    scene_role = document.get("scene_role")
    if scene_role not in _VALID_SCENE_ROLES:
        raise ValueError("scene_role must be 'ground_truth' or 'estimate'")
    try:
        scene = document["scene"]
        microphone_positions = scene["microphones"]["positions_m"]
        source_times = scene["source"]["times_s"]
        source_positions = scene["source"]["positions_m"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "JSON must contain scene.microphones.positions_m and scene.source.times_s/positions_m"
        ) from error

    microphones, times, source = _validate_ground_truth_arrays(
        np.asarray(microphone_positions, dtype=float),
        np.asarray(source_times, dtype=float),
        np.asarray(source_positions, dtype=float),
    )
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    return GroundTruth(
        microphone_positions_m=microphones,
        source_times_s=times,
        source_positions_m=source,
        metadata=dict(metadata),
        scene_role=str(scene_role),
    )


def validate_ground_truth_json(path: str | Path) -> GroundTruth:
    """Validate a JSON scene for use as ground truth/reference and return it.

    A calibration output JSON is valid input because it uses the same canonical
    ``scene`` schema with ``scene_role: estimate``.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("JSON root must be an object")
    return ground_truth_from_dict(document)


def load_ground_truth_json(path: str | Path) -> GroundTruth:
    """Load a validated reference scene from JSON."""
    return validate_ground_truth_json(path)


def ground_truth_to_dict(ground_truth: GroundTruth) -> dict[str, Any]:
    """Convert a validated reference scene back to the canonical JSON shape."""
    return _scene_document(
        microphone_positions_m=ground_truth.microphone_positions_m,
        source_times_s=ground_truth.source_times_s,
        source_positions_m=ground_truth.source_positions_m,
        scene_role=ground_truth.scene_role,
        metadata=ground_truth.metadata or None,
    )
