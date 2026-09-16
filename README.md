# acoustic-self-calibration

Bayesian/MAP **3-D acoustic self-calibration** for a **moving broadband sound source** and stationary microphone arrays, with 8–24 microphones as the primary target range.

The package works directly from synchronized multichannel audio. It does **not** require a known calibration pulse or known source emission timestamps.

## What you get

From one multichannel recording, the solver estimates:

- stationary 3-D microphone positions,
- the moving source position at each analysis frame,
- per-axis microphone position uncertainty,
- per-axis source trajectory uncertainty,
- optional relative microphone clock offsets and drift,
- optional speed of sound when a metric microphone-distance anchor is provided,
- TDOA residual and posterior diagnostics.

The returned position uncertainties are local **Laplace posterior standard deviations** from the joint MAP Hessian. Source uncertainty is marginalized over microphone geometry and enabled nuisance parameters; it is not computed by pretending the microphones are exact.

## Quick start

```bash
uv sync --all-groups
uv run pytest
```

### Run without cloning the repository

The package exposes the `acoustic-selfcal` command. Because the repository is private, the shortest one-shot invocation uses Git over SSH and your existing GitHub SSH credentials:

```bash
uvx \
  --from 'git+ssh://git@github.com/fhaefele/acoustic-self-calibration.git@main' \
  acoustic-selfcal \
  /path/to/file.wav \
  --output wavcalib
```

`--output` is an **output prefix**, not a single output filename. The command above writes:

```text
wavcalib.npz
wavcalib.json
wavcalib_microphones.csv
wavcalib_trajectory.csv
```

If the repository becomes public, the equivalent HTTPS form is:

```bash
uvx \
  --from 'git+https://github.com/fhaefele/acoustic-self-calibration.git@main' \
  acoustic-selfcal \
  /path/to/file.wav \
  --output wavcalib
```

### Run from a local checkout

```bash
uv run acoustic-selfcal recording.wav -o results/calibration
```

The command writes:

```text
results/calibration.npz
results/calibration.json
results/calibration_microphones.csv
results/calibration_trajectory.csv
```

`calibration_microphones.csv` contains:

```text
microphone,x_m,y_m,z_m,std_x_m,std_y_m,std_z_m
```

`calibration_trajectory.csv` contains:

```text
time_s,x_m,y_m,z_m,std_x_m,std_y_m,std_z_m
```

The compressed NPZ additionally contains the raw pairwise TDOAs, their estimated timing uncertainties, GCC confidence values, microphone-pair graph, clock terms, speed of sound, diagnostics, and all position/uncertainty arrays.

A typical real-data command is:

```bash
uv run acoustic-selfcal recording.wav \
  --output results/run01 \
  --frame-size 1024 \
  --hop-size 4096 \
  --pair-mode redundant \
  --reference-count 2 \
  --likelihood cauchy
```

For recordings assembled from channels with unknown relative timing, experimental clock terms can be enabled:

```bash
uv run acoustic-selfcal recording.wav \
  --estimate-clock-offsets \
  --estimate-clock-drifts
```

Sound speed cannot be estimated together with unconstrained scene scale from TDOAs alone. To estimate it, provide at least one measured microphone baseline:

```bash
uv run acoustic-selfcal recording.wav \
  --estimate-speed-of-sound \
  --distance-prior 0,1,1.234,0.002
```

That example means microphones 0 and 1 are `1.234 m` apart with a `0.002 m` standard deviation.

### Python API for a WAV file

```python
from acoustic_self_calibration import calibrate_wav, export_calibration

result = calibrate_wav(
    "recording.wav",
    frame_size=1024,
    hop_size=4096,
    pair_mode="redundant",
    reference_count=2,
    likelihood="cauchy",
)

cal = result.calibration

microphone_positions = cal.microphone_positions
microphone_std = cal.microphone_position_std_m

source_times = result.frame_times_s
source_positions = cal.source_positions
source_std = cal.source_position_std_m

paths = export_calibration(result, "results/calibration")
```

The WAV reader accepts common integer PCM and floating-point WAV data. Integer PCM is normalized to floating point before calibration; true 24-bit PCM is supported through SciPy's left-justified `int32` representation.

### Python API for an in-memory array

```python
from acoustic_self_calibration import calibrate_audio

result = calibrate_audio(
    audio,  # shape: (samples, microphones)
    sample_rate=48_000,
    frame_size=1024,
    hop_size=4096,
    pair_mode="redundant",
    reference_count=2,
    likelihood="cauchy",
)
```

Lower-level pairwise-TDOA and MAP APIs are public for research use.

## Pipeline

```text
arbitrary moving broadband source
            |
            v
synchronized multichannel waveform / WAV
            |
            v
redundant pairwise GCC-PHAT TDOAs
            |
            v
confidence -> timing uncertainty
            |
            v
low-rank Euclidean scene initializer
            |
            v
Bayesian factor graph / MAP optimization
  - stationary microphone positions
  - moving source trajectory
  - constant-velocity motion prior
  - robust Cauchy-IRLS or Gaussian TDOA likelihood
  - optional microphone clock offsets
  - optional clock drift (experimental)
  - optional sound-speed estimation with metric anchor
            |
            v
joint Laplace uncertainty
  - microphone position std
  - source trajectory std
  - enabled global nuisance-parameter std
```

## Measurement model

For microphone pair `(a, b)` and source state `t`:

```text
tau_ab,t = (||s_t - m_b|| - ||s_t - m_a||) / c
           + (offset_b - offset_a)
           + (drift_b - drift_a) * (t - mean(t))
           + noise
```

The absolute world coordinate frame is not observable from TDOAs. The optimizer fixes a canonical six-degree-of-freedom rigid-body gauge. A global mirror ambiguity is resolved by a fixed handedness convention. Therefore all positions and their per-axis standard deviations are expressed in that recovered canonical coordinate frame unless you subsequently register the array to external anchors.

## Recording requirements

For useful real-data calibration:

- use at least four microphones; 8–24 is the main tested range,
- all channels should preferably share one sample clock,
- the source should contain enough broadband energy for delay estimation,
- the source must move through a genuinely 3-D, non-degenerate trajectory,
- avoid trajectories that are almost entirely linear or planar when full 3-D geometry is required,
- direct-path-dominant, high-SNR data will perform substantially better than highly reverberant recordings,
- choose `frame_size` and `hop_size` so each analysis frame sees an approximately stationary propagation geometry while still obtaining enough independent source poses.

## Why Bayesian MAP instead of generic MCMC?

With 24 microphones and hundreds of source states, the problem has thousands of continuous variables. A sparse MAP factor graph gives the useful Bayesian structure—heteroscedastic measurement uncertainty, priors, nuisance timing parameters, robust likelihoods, and local posterior covariance—without the cost of generic full-state MCMC.

The reported Laplace uncertainties quantify local curvature of the fitted probabilistic model. They do **not** account for unmodeled room reflections, source/microphone response errors, an incorrectly calibrated GCC confidence-to-sigma mapping, or multimodal posterior structure.

## Synthetic moving-source renderer

The simulator supports:

- arbitrary microphone positions and counts,
- arbitrary 3-D source trajectories,
- retarded emission time for continuous source motion,
- inverse-distance propagation,
- omni, cardioid, hypercardioid, and dipole source radiation patterns,
- additive waveform noise,
- optional per-channel clock offset/drift simulation.

Unlike a discrete pulse-event simulator, propagation delay is solved against the source position at the **emission time**, so source motion affects the rendered waveform continuously.

## Validation

The automated tests cover:

- 8, 16, and 24 microphones,
- rendered moving cardioid broadband audio,
- a PCM16 WAV -> calibration -> joint uncertainty end-to-end path,
- PCM16, true PCM24, PCM32, and float WAV decoding,
- pairwise GCC-PHAT extraction,
- automatic low-rank initialization,
- joint microphone/source MAP recovery,
- marginalized microphone and source-position Laplace uncertainty,
- heteroscedastic timing uncertainty,
- relative clock-offset estimation,
- sound-speed estimation with a known baseline,
- calibration exports to NPZ, JSON, and CSV,
- directivity and moving-source propagation.

Run the reproducible synthetic benchmark:

```bash
uv run python examples/validate_synthetic_audio.py
```

See [`VALIDATION.md`](VALIDATION.md) for the current synthetic free-field benchmark and its limitations.

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
    bayesian.py        sparse MAP factor graph + joint Laplace uncertainty
    initialization.py  low-rank geometry bootstrap + coordinate gauge packing
    pipeline.py        in-memory audio -> TDOA -> MAP API
    wav.py             multichannel WAV loading + calibrate_wav API
    export.py          NPZ / JSON / microphone CSV / trajectory CSV export
    cli.py             acoustic-selfcal command-line interface
    tdoa.py            GCC-PHAT and redundant pair graphs
    simulation.py      continuous moving-source renderer + clock simulation
    radiation.py       source directivity patterns
    geometry.py        canonical gauge, rigid alignment, error metrics
```

## Scope

This is a research-grade free-field baseline, not a claim that reverberant-room calibration is solved. The main real-world extensions are explicit multipath/direct-path selection, measured microphone/channel responses, empirical TDOA uncertainty calibration, stronger outlier rejection, and more mature asynchronous-clock models.

## License

MIT
