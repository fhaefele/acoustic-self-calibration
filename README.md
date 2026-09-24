# acoustic-self-calibration

Stratified **TDOA acoustic self-calibration** for synchronized stationary microphone
arrays and discrete broadband/transient source events.

The production path is:

```text
multichannel audio
-> transient detection
-> event arrival association
-> reference-star TDOAs + timing covariance
-> stratified offset recovery
-> corrected ranges
-> affine factorization
-> Euclidean metric recovery
-> held-out validation + geometric consensus
-> calibrated microphones + event source states
```

The production solver does **not** use source-motion priors, posterior scores, Bayesian/MAP
fallbacks, or source truth to choose geometry.

## Supported geometry models

- `general_3d`: 3-D receivers and 3-D source events.
- `receiver2d_source3d`: planar receivers with 3-D sources.

For the planar model, source output includes in-plane projection, **unsigned** plane-normal
height, and a per-event sign-known mask. Independent height signs are not identifiable
from ranges/TDOAs alone.

Exact cross/two-line receiver layouts can have a continuous non-rigid ambiguity. The
solver reports that ambiguity rather than forcing a right angle. A physical constraint
such as a known arm angle must be supplied explicitly with provenance.

## Requirements

The current production audio path assumes:

- at least 8 synchronized microphone channels,
- known positive speed of sound,
- discrete broadband/transient events,
- enough geometric diversity for the chosen dimensional model.

The tested release surface covers 8, 12, 16, and 24 microphones.

## CLI

```text
asc calibrate WAV
asc check JSON
asc compare ESTIMATE_JSON REFERENCE_JSON
```

General 3-D calibration:

```bash
asc calibrate recording.wav \
  -o results/run01 \
  --model general-3d \
  --event-min-gap-ms 50 \
  --max-tau-ms 20
```

Planar receiver model:

```bash
asc calibrate recording.wav \
  -o results/planar \
  --model receiver2d-source3d
```

Planar model with an explicit right-angle arm constraint:

```bash
asc calibrate recording.wav \
  -o results/planar_constrained \
  --model receiver2d-source3d \
  --right-angle 3,0,7 \
  --constraint-provenance "survey drawing / hardware construction"
```

The implemented solver controls are stratified-only:

```text
--model {general-3d,receiver2d-source3d}
--speed-of-sound FLOAT
--receiver-subset-budget INT
--event-subset-budget INT
--root-start-count INT
--metric-start-count INT
--extra-microphone-rms-m FLOAT
--planar-membership-tolerance FLOAT
--planar-metric-rms-m FLOAT
--planar-extra-microphone-rms-m FLOAT
--right-angle CENTER,ARM_A,ARM_B
--constraint-provenance TEXT
--refinement {none,wls,huber}
--refinement-max-nfev INT
--refinement-improvement-tolerance FLOAT
```

Temporal tracking is an optional frontend association heuristic only; it is not a source
motion prior.

## Python API

```python
from acoustic_self_calibration import calibrate_wav

result = calibrate_wav(
    "recording.wav",
    event_min_gap_s=0.05,
    max_tau_s=0.02,
    model="general_3d",
)

print(result.status)
print(result.microphone_positions_m)
print(result.source_positions_m)
```

For in-memory audio:

```python
from acoustic_self_calibration import calibrate_audio

result = calibrate_audio(audio, sample_rate=48_000)
```

For precomputed reference-star TDOAs:

```python
from acoustic_self_calibration import calibrate_tdoa

result = calibrate_tdoa(measurements)
```

Planar TDOA calibration with an explicit constraint:

```python
from acoustic_self_calibration import PlanarAngleConstraint, calibrate_planar_tdoa

constraint = PlanarAngleConstraint(
    center_receiver=3,
    arm_a_receiver=0,
    arm_b_receiver=7,
    angle_rad=3.141592653589793 / 2,
    provenance="measured hardware geometry",
)

result = calibrate_planar_tdoa(measurements, angle_constraint=constraint)
```

## Result semantics

Calibration status is explicit:

- `solved`
- `ambiguous`
- `weakly_identified`
- `degenerate`
- `insufficient_data`
- `failed`

Diagnostics include subset/root counts, conditioning, geometric-class support, held-out
robust residuals, generation/completion/validation lineage, rejection reasons, and
extra-microphone completion quality.

Optional measurement-only refinement is available with `refinement="wls"` or
`refinement="huber"`. It runs only after a solved stratified geometry has been selected,
does not change the frozen held-out validation score or branch selection, and retains the
pre-refinement geometry for rollback/audit. The evaluation budget and acceptance tolerance are
explicit controls. JSON records those budgets together with pre/post coordinates, objective,
termination, TDOA RMS, and whether the refinement was accepted.

For planar results, the exported 3-D source positions are a conventional positive-normal
representative for visualization. Machine-readable observable fields also contain
projected coordinates, unsigned height, and `height_sign_known`.

## JSON and reference evaluation

`asc calibrate` writes:

```text
RESULT.json
RESULT.png
```

Reference alignment is fit from microphones only. The same rigid transform is then
applied to source states, and reference source coordinates are interpolated only over
overlapping timestamps.

Planar evaluation reports observable projected-source and unsigned-height errors rather
than silently choosing a source-side sign.

See [docs/json_format.md](docs/json_format.md).

## Real Myotis

The repository does not bundle the real Myotis WAV/reference files.

Blind run:

```bash
ASC_MYOTIS_AUDIO=/path/to/myotis.wav \
ASC_MYOTIS_REFERENCE=/path/to/reference.json \
ASC_MYOTIS_OUTPUT=/tmp/myotis-blind.json \
uv run python examples/validate_real_myotis.py
```

Explicitly constrained run:

```bash
uv run python examples/evaluate_constrained_myotis.py \
  --audio /path/to/myotis.wav \
  --reference /path/to/reference.json \
  --right-angle 3,0,7 \
  --constraint-provenance "survey drawing / hardware construction"
```

Reference geometry/source states are evaluation-only in the blind run. Missing files
produce `not_run/missing_input_paths`; synthetic data are never substituted.

## Validation

Hard rendered-audio gates cover 8/12/16/24 microphones at both 20 and 40 events:

- microphone RMS < 0.15 m
- source RMS < 0.18 m
- TDOA RMS < 60 us

PCM16 WAV gates use 0.18 m microphone RMS, 0.22 m source RMS, and 60 us TDOA RMS.

The suite also covers missing/outlier measurements, generated 7r/6s root verification,
planar exact/noisy recovery, source-height sign ambiguity, continuous cross ambiguity,
constraint ablation, dimensional-model comparison, diagnostics JSON, and Myotis workflow
reproducibility.

See [VALIDATION.md](VALIDATION.md).

## Development

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests examples
uv run pytest
uv build
```

## Package layout

```text
src/acoustic_self_calibration/
    events.py
    measurements.py
    pipeline.py
    wav.py
    stratified/
        notation.py
        offsets_linear.py
        offsets_minimal.py
        factorization.py
        metric_upgrade.py
        planar.py
        expansion.py
        constraints.py
        hypotheses.py
        robustness.py
        identifiability.py
        solver.py
    geometry.py
    simulation.py
    evaluation.py
    ground_truth.py
    export.py
    visualization.py
    myotis.py
    cli.py
```

## Scope

This is a research-grade free-field calibration implementation. Reverberation,
asynchronous clocks, direct-path selection, and arbitrary degenerate geometries remain
explicit limitations. Ambiguity and insufficient information are intended outputs, not
conditions to hide with a fallback estimator.

## License

MIT
