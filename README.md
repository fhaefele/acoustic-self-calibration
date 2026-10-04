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

For the planar model, source output includes in-plane projection, unsigned height,
and a per-event sign-known mask. `source_region="same_side"` declares that every
call stays on one side of the array; output places that occupied side at positive z.
Without this declaration, height signs remain unknown.

`ArrayConfiguration` selects arbitrary-planar, cross, T, grid or room construction.
Arbitrary-planar has no shape/angle/spacing input. Cross/T configurations declare
arm membership and optional directed-ray angles in degrees (including 60, 75 and
90); every spacing remains unknown. A grid declares row/column topology, with
optional independently equal row and column spacing of unknown pitch.

Room sources can leave the microphone convex hull. The solver receives no room
boundaries and cannot infer physical floor/ceiling orientation from TDOAs alone.
All current fixtures use synchronized direct sound; echoes are outside this scope.

Exact cross/two-line receiver layouts can have a continuous non-rigid ambiguity. The
solver reports that ambiguity rather than forcing a right angle. A physical constraint
such as a known arm angle must be supplied explicitly with provenance.

## Requirements

The current production audio path assumes:

- at least 8 synchronized microphone channels,
- known positive speed of sound,
- discrete broadband/transient events,
- enough geometric diversity for the chosen dimensional model.

The legacy tests cover 8, 12, 16 and 24 microphones. New grid tests also cover 9 channels.
See [completion progress](docs/COMPLETION_PROGRESS.md) for current acceptance gaps.

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
  --model receiver2d-source3d \
  --source-region same-side \
  --timing-uncertainty waveform
```

Planar model with an explicit right-angle arm constraint:

```bash
asc calibrate recording.wav \
  -o results/planar_constrained \
  --model receiver2d-source3d \
  --right-angle 3,0,7 \
  --constraint-provenance "survey drawing / hardware construction"
```

Construction files contain IDs/topology, degrees and provenance, never measured
coordinates or distances. See [the grid example](examples/array_configs/grid_3x3.json)
and [the conditional Myotis cross](examples/array_configs/myotis_cross.json).

```bash
asc calibrate recording.wav -o results/grid \
  --array-config examples/array_configs/grid_3x3.json \
  --timing-uncertainty waveform --output-origin-microphone 0
```

If WAV channels are reordered, supply stable IDs in channel order with
`--microphone-ids 5,1,7,0,3,6,2,4`; construction files and output origins use those IDs.

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

All public audio/WAV coordinates are in metres relative to microphone 0 (or the
smallest ID if 0 is absent). `output_origin_microphone_id` chooses a different
origin; frame metadata records the baseline convention and transform.

Source times are inferred emission features: receiver feature time minus fitted
propagation delay. Receiver detection times remain separately in measurements.
For same-side planar results, `height_sign_known` records the declared constraint;
without it, positions are an unsigned positive-normal representative.

The default timing uncertainty remains confidence-based for compatibility.
`timing_uncertainty="waveform"` estimates uncertainty from recorded waveform
residuals/slope and exposes correlated waveform mismatch. It does not consume a
known chirp or simulator truth. Weak identification remains a non-success status.

## JSON and reference evaluation

`asc calibrate` writes diagnostic JSON for success and non-success. When geometry exists it also writes:

```text
RESULT.json
RESULT.png
```

Reference alignment is fit from microphones only. The same rigid transform is then
applied to source states, and reference source coordinates are interpolated only over
overlapping timestamps.

Absent geometry is recorded explicitly and visualization is omitted. Unresolved event
IDs and coverage counts remain in JSON; the CLI returns 2 for a non-success result.
Planar evaluation retains observable projection/height semantics.

See [docs/json_format.md](docs/json_format.md).

## Myotis fixture

The repository includes [the WAV/reference fixture](data/myotis) and its
[provenance audit](data/myotis/provenance.json). Independent evidence for its
physical 90-degree construction and acquisition history remains unverified.
Recovery below is conditional on the explicitly declared angle, not a successful
arbitrary-planar solution.

```bash
uv run python examples/validate_myotis_public.py
```

Calibration and angle ablation finish before reference loading. The frozen report
shows microphone RMS 5.13 cm, source RMS 17.18 cm, 41 localized sources from 43
detections, and two unresolved events. Status remains `weakly_identified`; the
strengthened acceptance command returns 2. The reference only overlaps 40 localized
source times and must not be used to choose source signs or tune geometry.

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

New chirp acceptance requires `solved`, complete event coverage, microphone RMS
<5 cm and source RMS <10 cm. The 10-case PR subset passes; the full development
and held-out evaluation matrices are separate required checks. The initial
96-case float/PCM subset had two failures, both corrected by a seed-tolerance fix.
The broader noisy-arrival matrix still contains failures and weak results;
Myotis and full acceptance remain incomplete.

See [VALIDATION.md](VALIDATION.md), [the implementation plan](docs/SELF_CALIBRATION_COMPLETION_PLAN.md),
and [measured progress](docs/COMPLETION_PROGRESS.md).

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

Recovery plots: [room](benchmarks/figures/room-rectangular_8.svg),
[cross](benchmarks/figures/cross90_8.svg),
[grid](benchmarks/figures/grid-free_9.svg), and
[unknown-angle alternatives](benchmarks/figures/cross_ambiguity.svg).
