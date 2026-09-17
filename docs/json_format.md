# JSON format

The project uses one canonical JSON scene schema for both reference input and calibration output.

The core rule is simple: every valid document contains the same top-level `scene` object. Calibration output adds extra sections such as settings, measurements, diagnostics, uncertainty, and evaluation, but its `scene` remains valid reference input for a later run.

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

`scene_role: "ground_truth"` means the scene is intended as measured/simulated truth. `scene_role: "estimate"` means it came from a calibration result. Both are accepted by `--ground-truth`, which makes it possible to use one calibration result as the reference for another run without converting formats.

## Convenience helpers and validation

Create a correctly formatted GT file with:

```python
from acoustic_self_calibration import write_ground_truth_json

write_ground_truth_json(
    "ground_truth.json",
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
    metadata={"name": "trial_01"},
)
```

Or create the JSON-compatible dictionary:

```python
from acoustic_self_calibration import make_ground_truth_dict

payload = make_ground_truth_dict(
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
)
```

Validate any canonical scene JSON before using it:

```python
from acoustic_self_calibration import validate_ground_truth_json

reference = validate_ground_truth_json("reference.json")
```

The function returns a validated `GroundTruth` object and raises `ValueError` for invalid schema, shapes, time ordering, non-finite values, or unsupported roles.

`load_ground_truth_json(...)` uses the same validation and remains the normal loader used by the CLI.

## Calibration output

A calibration result uses exactly the same core `scene` structure, with `scene_role: "estimate"` and optional per-axis uncertainties added inside the scene:

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
  "input": {
    "wav_path": "recording.wav"
  },
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

If uncertainty calculation is disabled, the `std_m` fields and other uncertainty fields are JSON `null`.

Because the canonical scene is at the same location in every document, a result can be reused directly:

```bash
acoustic-selfcal second.wav \
  --ground-truth first_calibration.json \
  --output second_vs_first
```

In this case the first calibration is a **reference estimate**, not physical truth. Its `scene_role: "estimate"` is preserved in the second result's `reference` section.

## Evaluation section

When `--ground-truth` is supplied, the output JSON adds the validated input document as `reference` plus an `evaluation` object.

The rigid alignment is fit **only from estimated microphones to reference microphones**. The same transform is then applied to the estimated source trajectory. The source is never independently aligned.

Reference source positions are linearly interpolated to the solver's analysis-frame times before source error is calculated. Reference time coverage must span all estimated source times.

```json
{
  "reference": {
    "schema_version": 1,
    "scene_role": "ground_truth",
    "scene": {
      "microphones": {"positions_m": []},
      "source": {"times_s": [], "positions_m": []}
    }
  },
  "evaluation": {
    "alignment": {
      "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "translation_m": [0.0, 0.0, 0.0],
      "reflection_applied": false,
      "fitted_from": "microphones_only"
    },
    "microphones": {
      "aligned_estimate_m": [],
      "error_m": [],
      "rms_error_m": 0.0,
      "mean_error_m": 0.0,
      "max_error_m": 0.0,
      "uncertainty_diagnostic": null
    },
    "source": {
      "aligned_estimate_m": [],
      "ground_truth_at_estimate_times_m": [],
      "error_m": [],
      "rms_error_m": 0.0,
      "mean_error_m": 0.0,
      "max_error_m": 0.0,
      "uncertainty_diagnostic": null
    }
  }
}
```

## Uncertainty diagnostic

When position standard deviations are available, the evaluation reports the practical radial diagnostic

```text
rss_std = sqrt(std_x^2 + std_y^2 + std_z^2)
ratio   = Euclidean_position_error / rss_std
```

and records mean/median ratio plus fractions below `1x`, `2x`, and `3x` RSS standard deviation.

This is not a formal 3-D Gaussian coverage probability. The solver currently exposes marginal per-axis standard deviations, not complete per-position covariance matrices, and alignment uncertainty is not included.

## Visualization output

Every CLI run writes one PNG next to the JSON result. The figure contains four panels:

1. 3-D scene,
2. XY projection,
3. XZ projection,
4. YZ projection.

Without a reference, the figure shows estimated microphones and source trajectory. With a reference, every panel shows both the aligned estimate and reference scene. Estimated microphone positions are connected to their reference counterparts. The 2-D panels also draw sparse 1-sigma uncertainty ellipses when uncertainties are available.
