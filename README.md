# acoustic-self-calibration

Bayesian/MAP **3-D acoustic self-calibration** for a **moving broadband sound source** and stationary microphone arrays, with 8–24 microphones as the primary target range.

This repository solves the general problem directly from synchronized multichannel audio. It does **not** require a known calibration pulse or known source emission timestamps.

## Pipeline

```text
arbitrary moving broadband source
            |
            v
synchronized multichannel waveform
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
Laplace uncertainty for global parameters
```

## Measurement model

For a microphone pair `(a, b)` at source state `t`:

```text
tau_ab,t = (||s_t - m_b|| - ||s_t - m_a||) / c
           + (offset_b - offset_a)
           + (drift_b - drift_a) * (t - mean(t))
           + noise
```

The absolute world coordinate frame is not observable from TDOAs. The optimizer fixes a canonical 6-DOF rigid-body gauge. A global mirror ambiguity is also resolved by a fixed handedness convention.

## Why Bayesian MAP instead of generic MCMC?

With 24 microphones and hundreds of source states, the problem has thousands of continuous variables. A sparse MAP factor graph gives the useful Bayesian structure—heteroscedastic measurement uncertainty, priors, nuisance timing parameters, robust likelihoods, and local posterior covariance—without the cost of generic full-state MCMC.

## Quick start

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests examples
uv run pytest
uv build
```

End-to-end calibration from audio:

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

mics = result.calibration.microphone_positions
trajectory = result.calibration.source_positions
mic_std = result.calibration.microphone_position_std_m
```

Lower-level TDOA + MAP APIs are also public for research use.

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

## Identifiability

Movement must sufficiently excite 3-D geometry. A nearly linear or planar trajectory can make some dimensions weakly observable even if the Jacobian is technically full rank.

Sound speed and scene scale are jointly scale-ambiguous from TDOAs alone. If `estimate_speed_of_sound=True`, at least one metric microphone-distance prior must be supplied:

```python
from acoustic_self_calibration import DistancePrior, calibrate_bayesian

prior = DistancePrior(0, 1, distance_m=1.234, sigma_m=0.002)
```

The solver refuses sound-speed estimation without a metric anchor.

## Validation

The automated tests cover:

- 8, 16, and 24 microphones,
- rendered moving cardioid broadband audio,
- pairwise GCC-PHAT extraction,
- automatic low-rank initialization,
- joint microphone/source MAP recovery,
- heteroscedastic timing uncertainty,
- relative clock-offset estimation,
- sound-speed estimation with a known baseline,
- Laplace uncertainty,
- directivity and moving-source propagation.

Run the reproducible benchmark:

```bash
uv run python examples/validate_synthetic_audio.py
```

See [`VALIDATION.md`](VALIDATION.md) for the current synthetic free-field benchmark and its limitations.

## Package layout

```text
src/acoustic_self_calibration/
    bayesian.py        sparse MAP factor graph + Laplace uncertainty
    initialization.py  low-rank geometry bootstrap + coordinate gauge packing
    pipeline.py        end-to-end audio -> TDOA -> MAP API
    tdoa.py            GCC-PHAT and redundant pair graphs
    simulation.py      continuous moving-source renderer + clock simulation
    radiation.py       source directivity patterns
    geometry.py        canonical gauge, rigid alignment, error metrics
```

## Scope

This is a research-grade free-field baseline, not a claim that reverberant-room calibration is solved. The main real-world extensions are explicit multipath/direct-path selection, measured microphone/channel responses, empirical TDOA uncertainty calibration, stronger outlier rejection, and more mature asynchronous-clock models.

## License

MIT
