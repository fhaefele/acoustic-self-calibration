# JSON format

The project uses one canonical scene schema for reference input and solved calibration
output. Calibration JSON adds event measurements and stratified diagnostics.

## Canonical scene

A minimal reference scene is:

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
      "times_s": [0.0, 0.1],
      "positions_m": [
        [2.0, 0.0, 1.0],
        [1.9, 0.2, 1.0]
      ]
    }
  }
}
```

`scene_role` is `ground_truth` or `estimate`. Source times must be strictly
increasing. Coordinates and times must be finite.

## Stratified calibration section

Solved calibration output adds:

```json
{
  "calibration": {
    "backend": "stratified_tdoa",
    "model": "general_3d",
    "status": "solved",
    "speed_of_sound_mps": 343.0,
    "rms_tdoa_residual_s": 0.00001,
    "temporal_tracking_enabled": true,
    "refinement": {
      "mode": "wls",
      "attempted": true,
      "applied": true,
      "reason": "accepted",
      "initial_objective": 12.4,
      "final_objective": 8.1,
      "initial_tdoa_rms_s": 0.000011,
      "final_tdoa_rms_s": 0.000009,
      "independent_coordinate_count": 140,
      "nfev": 18,
      "max_nfev": 200,
      "improvement_tolerance": 1e-10,
      "termination": "ftol termination condition is satisfied",
      "frozen_validation_score": 0.73,
      "pre_refinement_scene": {"microphones": {}, "source": {}},
      "post_refinement_scene": {"microphones": {}, "source": {}}
    }
  }
}
```

Status is one of:

- `solved`
- `ambiguous`
- `weakly_identified`
- `degenerate`
- `insufficient_data`
- `failed`

The release does not export Bayesian posterior covariance. `std_m` fields are `null`.
When refinement mode is `none`, refinement diagnostics and pre/post scenes are null/empty
and `applied` is false. When `wls` or `huber` is requested, the original selected
geometry and frozen held-out validation score remain recorded separately from the accepted
post-selection training fit. `max_nfev` and `improvement_tolerance` record the configured
optimization budget and acceptance threshold.

## Measurements

The measurement section records the immutable event TDOA boundary:

```json
{
  "measurements": {
    "event_ids": [0, 1],
    "receiver_event_times_s": [0.4, 0.6],
    "event_samples": [19200, 28800],
    "event_channel": 3,
    "microphone_pairs": [[0, 1], [0, 2]],
    "measurement_basis": "reference_star",
    "measurement_origin": "derived_arrivals",
    "covariance_model": "shared_reference_arrival_covariance",
    "tdoa_s": [[0.0001, -0.0002]],
    "tdoa_sigma_s": [[0.000002, 0.000003]],
    "confidence": [[0.9, 0.8]],
    "valid": [[true, true]]
  }
}
```

Invalid measurements remain invalid; they are not serialized as fitted zeros.

## Diagnostics

The diagnostics section includes:

- minimal-subset/root counts,
- metric candidate and geometric-class counts,
- selected class support,
- fitting/validation event counts,
- conditioning,
- robust held-out residual summaries,
- generation/completion/validation lineage counts,
- rejected-hypothesis reasons,
- extra-microphone completion diagnostics,
- competing class support.

A selected hypothesis can contain:

```json
{
  "conditioning": {
    "metric_condition_number": 42.0,
    "affine_factor_singular_ratio": 0.12
  },
  "validation": {
    "normalized_huber_score": 0.7,
    "inlier_fraction": 0.92
  },
  "lineage": {
    "generation_count": 36,
    "completion_count": 20,
    "validation_count": 12
  }
}
```

Unscored stages use `null`, not convincing-looking zero residuals.

## Planar source representation

For `model: "receiver2d_source3d"`, `scene.source` also contains:

```json
{
  "representation": "positive_plane_normal_representative",
  "projected_positions_m": [[0.4, -0.2]],
  "unsigned_height_m": [1.1],
  "height_sign_known": [false]
}
```

The ordinary `positions_m` field is a positive-normal representative for visualization
and scene tooling. It is not an independently observed signed 3-D source coordinate.
The observable values are projection, unsigned height, and the sign-known mask.

## Reference evaluation

For general 3-D output, alignment is fit from microphones only. The same rigid transform
is applied to source positions. Reference source coordinates are interpolated only over
the time overlap.

For planar output, calibration-time evaluation reports microphone error and observable
projected-source/unsigned-height errors. It does not choose source-height signs.

## Reusing output

A solved general-3D result can be validated or compared directly:

```bash
asc check result.json
asc compare estimate.json reference.json -o comparison
```

Diagnostic records with `null` geometry are valid run records but are not valid scene
references until geometry exists.

Planar output contains a conventional representative source scene, but consumers must
retain the observable semantics described above.
