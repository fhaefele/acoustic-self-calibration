# JSON format

The project uses one canonical JSON scene schema for reference input, calibration output, and standalone comparison input.

Every valid scene document contains the same top-level `scene` object. Calibration and comparison outputs add extra sections, but their `scene` remains valid input for later `asc calibrate -r ...`, `asc check`, or `asc compare` operations.

The previous pre-`scene` JSON layout is intentionally unsupported.

## Canonical scene schema

A minimal ground-truth/reference file is:

```json
{
  "schema_version": 1,
  "scene_role": "ground_truth",
  "scene": {
    "microphones": {
      "positions_m": [
        [0.0, 0.0, 1.2],
        [1.0, 0.0, 1.2],
        [0.0, 1.0, 1.2],
        [0.0, 0.0, 2.0]
      ]
    },
    "source": {
      "times_s": [0.0, 0.1, 0.2],
      "positions_m": [
        [2.0, 0.0, 1.0],
        [1.95, 0.2, 1.02],
        [1.85, 0.4, 1.05]
      ]
    }
  },
  "metadata": {
    "name": "trial_01"
  }
}
```

Required fields:

- `schema_version`: currently `1`.
- `scene_role`: either `"ground_truth"` or `"estimate"`.
- `scene.microphones.positions_m`: shape `(M, 3)`, at least four microphones.
- `scene.source.times_s`: strictly increasing seconds.
- `scene.source.positions_m`: shape `(N, 3)`, same length as `times_s`.
- all coordinates and times must be finite.

`metadata` is optional and must be a JSON object when present.

`scene_role: "ground_truth"` means measured/simulated truth. `scene_role: "estimate"` means the scene came from an estimator. Both roles can be used as references.

## Validation

CLI:

```bash
asc check scene.json
```

Python:

```python
from acoustic_self_calibration import validate_ground_truth_json

scene = validate_ground_truth_json("scene.json")
```

The validator returns a `GroundTruth` scene object and raises `ValueError` for invalid schema, shapes, time ordering, non-finite values, or unsupported roles.

## Ground-truth helpers

```python
from acoustic_self_calibration import make_ground_truth_dict, write_ground_truth_json

payload = make_ground_truth_dict(
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
)

write_ground_truth_json(
    "ground_truth.json",
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
)
```

## Calibration output

Calibration results use the same core scene with `scene_role: "estimate"` and may add per-axis uncertainty:

```json
{
  "schema_version": 1,
  "scene_role": "estimate",
  "scene": {
    "microphones": {
      "positions_m": [[0.0, 0.0, 0.0]],
      "std_m": [[0.0, 0.0, 0.0]]
    },
    "source": {
      "times_s": [0.1],
      "positions_m": [[1.0, 0.0, 1.0]],
      "std_m": [[0.01, 0.01, 0.02]]
    }
  },
  "input": {"wav_path": "recording.wav"},
  "settings": {},
  "calibration": {
    "speed_of_sound_mps": 343.0,
    "speed_of_sound_std_mps": null,
    "clock_offsets_s": [],
    "clock_offset_std_s": null,
    "clock_drifts": [],
    "clock_drift_std": null
  },
  "measurements": {
    "microphone_pairs": [[0, 1]],
    "tdoa_s": [[0.0]],
    "tdoa_sigma_s": [[0.000001]],
    "confidence": [[1.0]]
  },
  "diagnostics": {
    "success": true,
    "message": "...",
    "nfev": 10,
    "rms_tdoa_residual_s": 0.000001,
    "normalized_data_rms": 0.8,
    "negative_log_posterior": 12.0
  }
}
```

If uncertainty calculation is disabled, uncertainty fields are JSON `null`.

A result can be reused directly:

```bash
asc calibrate second.wav -r first_calibration.json -o second_vs_first
```

The first calibration remains `scene_role: "estimate"`, making clear that it is a reference estimate rather than physical truth.

## Evaluation section

When `asc calibrate ... -r REFERENCE` is used, output adds the validated input as `reference` plus an `evaluation` object.

Alignment is fit **only from estimated microphones to reference microphones**. The same transform is applied to the estimated source trajectory; the source is never independently aligned. Reference source positions are linearly interpolated to the estimate timestamps, and reference time coverage must span all estimate times.

The evaluation reports microphone and source RMS / mean / max position errors and, when calibration uncertainties are available, practical uncertainty-vs-error diagnostics.

## Standalone comparison output

`asc compare ESTIMATE REFERENCE -o PREFIX` performs the same geometry comparison without recalibrating audio:

```bash
asc compare run02.json run01.json -o run02_vs_run01
```

The comparison JSON keeps the estimate scene as its canonical top-level `scene`, so the output itself remains valid scene input. It additionally records:

```json
{
  "comparison": {
    "estimate_path": "run02.json",
    "reference_path": "run01.json"
  },
  "reference": {
    "schema_version": 1,
    "scene_role": "estimate",
    "scene": {}
  },
  "evaluation": {}
}
```

The companion PNG contains 3-D, XY, XZ, and YZ panels with estimate and reference overlaid.

## Uncertainty diagnostic

When position standard deviations are available during calibration-time reference evaluation, the practical radial diagnostic is:

```text
rss_std = sqrt(std_x^2 + std_y^2 + std_z^2)
ratio   = Euclidean_position_error / rss_std
```

This is not a formal 3-D Gaussian coverage probability. The solver currently exposes marginal per-axis standard deviations, not complete per-position covariance matrices, and alignment uncertainty is not included.
