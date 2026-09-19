# Validation

The production release uses the public event-driven stratified APIs:
`calibrate_audio(...)`, `calibrate_wav(...)`, `calibrate_tdoa(...)`, and
`calibrate_planar_tdoa(...)`.

No production validation path uses the removed Bayesian/MAP backend.

## Random 3-D rendered audio

Deterministic fixtures cover:

```text
microphones: 8, 12, 16, 24
events:      20 and 40
sample rate: 48 kHz
duration:    5.0 s
source:      moving non-planar cardioid pulse source
```

Hard gates:

```text
microphone RMS < 0.15 m
source RMS     < 0.18 m
TDOA RMS       < 60 us
```

The clean benchmark also requires all scheduled events, finite positive timing
uncertainties, and valid reference-star measurements.

Run:

```bash
uv run python examples/validate_synthetic_audio.py
```

## PCM16 WAV regression

The 8-microphone 20- and 40-event fixtures are quantized to PCM16 and passed through
`calibrate_wav(...)`.

WAV gates:

```text
microphone RMS < 0.18 m
source RMS     < 0.22 m
TDOA RMS       < 60 us
```

Decoder tests separately cover PCM16, packed PCM24, PCM32, and float WAV data.

## Measurement and robustness checks

The suite covers:

- transient event timing and scaling invariance,
- tracker-enabled and tracker-disabled event association,
- immutable reference-star TDOA/covariance contracts,
- reference changes and redundant-pair round trips,
- missing measurements and gross TDOA outliers,
- covariance-aware held-out whitening,
- generation/completion/validation leakage rejection,
- independent-subset geometric consensus.

## Stratified geometry checks

Exact/noisy tests cover:

- 9r/5s and 7r/4s linear offset anchors,
- generated 7r/6s minimal rank constraints,
- independent verification against all 75 rank-four minors,
- corrected-range SVD factorization,
- 3-D Euclidean metric recovery,
- event and microphone expansion,
- 8/12/16/24 microphone TDOA-only calibration.

The 7r/6s runtime root search is deterministic but numerical. Its
`search_stabilized` flag is a completeness diagnostic, not a symbolic proof that no
additional isolated root exists outside the search region.

## Planar observables and ambiguity

The nondegenerate planar fixture verifies:

- rank-2 receiver factorization,
- Euclidean planar metric recovery,
- source projections,
- unsigned source plane-normal heights,
- exact invariance under independent source-side sign flips.

The exact Myotis cross is a **degeneracy** fixture. It verifies:

- conic design rank 5,
- one-dimensional planar metric nullspace,
- continuous non-rigid range/TDOA-equivalent deformation,
- no forced unique right-angle solution.

A supplied exact right-angle arm constraint lifts the exhibited metric ambiguity. An
ablation removes the constraint and must restore ambiguity/degeneracy. Source height
sign remains separate and requires an explicit half-space convention for signed error.

## Model comparison

Planar and general-3D models are compared on the same held-out event basis. Tests cover:

- clearly planar data,
- generic 3-D data,
- near-ties returning `ambiguous`,
- near-planar/near-cross structural conditioning.

Training residual alone never forces a model selection.

## JSON and release surface

Tests verify that solved, ambiguous, and degenerate results serialize with:

- backend/model/status,
- subset/root/hypothesis counts,
- conditioning,
- held-out robust residual summaries,
- lineage counts,
- rejection reasons,
- ambiguity support,
- extra-microphone completion,
- refinement mode/attempt/acceptance,
- refinement optimizer budget and improvement tolerance,
- pre/post refinement coordinates and frozen validation score.

The public package no longer exports or contains the old Bayesian solver/initializer.

## Real Myotis workflow

The repository does not bundle the real Myotis WAV/reference files. Therefore no
empirical real-data pass is claimed without supplied external paths.

Blind run:

```bash
ASC_MYOTIS_AUDIO=/path/to/myotis.wav \
ASC_MYOTIS_REFERENCE=/path/to/reference.json \
ASC_MYOTIS_OUTPUT=/tmp/myotis-blind.json \
uv run python examples/validate_real_myotis.py
```

The blind runner records input hashes, channel mapping, timing convention, solver
configuration, receiver-only conditioning, extracted-event/measurement coverage,
stratified diagnostics, and post-hoc evaluation. Reference geometry/source states are
not passed into calibration.

Explicit constrained run:

```bash
uv run python examples/evaluate_constrained_myotis.py \
  --audio /path/to/myotis.wav \
  --reference /path/to/reference.json \
  --right-angle 3,0,7 \
  --constraint-provenance "survey drawing / hardware construction"
```

Blind, constrained, and constraint-ablation outputs remain separate. Planar source
evaluation reports projected-source and unsigned-height error. Signed source RMS is
reported only with an explicit source half-space sign relative to the stored reference
plane normal.

Missing external paths return `not_run/missing_input_paths`; synthetic data are never
substituted for the real run.

## Full quality commands

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests examples
uv run pytest
uv build
```

Feature-branch pushes in this repository do not trigger the full GitHub Actions matrix.
Unless a pull request/manual dispatch has run, local/static checks should not be described
as the Python 3.11-3.14 CI matrix passing.

## Interpretation

These tests validate the implementation under deterministic free-field fixtures and
explicitly modeled ambiguity cases. They are not a prediction of accuracy in a
reverberant room. Multipath, source occlusion, channel-response differences, incorrect
direct-path peaks, synchronization error, weak trajectories, and degenerate receiver
layouts can all reduce identifiability.
