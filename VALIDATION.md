# Synthetic event-driven validation

The primary end-to-end validation now uses **discrete broadband pulses**, not uniformly sampled analysis frames.

Each synthetic case:

1. creates a random 3-D microphone array,
2. creates a smooth moving 3-D source trajectory,
3. generates a sequence of short broadband pulses,
4. renders the moving source through the free-field propagation model with source directivity,
5. adds noise,
6. runs the same automatic transient detection and multi-candidate TDOA association used for WAV input,
7. jointly estimates microphone geometry and one source position per detected event,
8. rigidly aligns the estimated microphones to ground truth for evaluation.

The tests do **not** feed ideal geometric TDOAs to the calibration pipeline.

## Automated coverage

`tests/test_events.py` checks the transient frontend directly:

- automatic event-channel selection,
- event count recovery,
- per-channel delay estimation,
- exact pairwise TDOA cycle consistency.

`tests/test_end_to_end_audio.py` validates full synthetic calibration for:

- 8 microphones,
- 16 microphones,
- 24 microphones,
- moving cardioid source radiation,
- PCM16 WAV round-trip,
- microphone/source Laplace uncertainty.

The current benchmark uses 20 broadband source events over a 5 s moving trajectory at 48 kHz. The event frontend retains multiple envelope-correlation delay candidates and chooses smooth delay tracks before the Bayesian/MAP geometry fit.

Run the automated tests with:

```bash
uv run pytest
```

Run the standalone 8/16/24 benchmark with:

```bash
uv run python examples/validate_synthetic_audio.py
```

## What this validates

The synthetic benchmark exercises the intended source model end to end:

```text
moving pulsed source
        ↓
rendered multichannel audio
        ↓
automatic event detection
        ↓
multi-candidate delay association
        ↓
cycle-consistent TDOAs
        ↓
low-rank initialization
        ↓
Bayesian/MAP microphone + source calibration
```

This specifically guards against the previous failure mode where arbitrary silent or unrelated WAV frames were treated as source observations.

## Limitations

The renderer is still a free-field model. Real rooms can contain strong reflections, occlusion, channel-response differences, source directivity changes, and missed/extra events. The multi-candidate temporal tracker is intended to reduce call-association ambiguity, but it is not yet a full multipath or multi-hypothesis structure-from-sound system.

Real recordings should therefore be checked for:

- sensible detected event count and event channel,
- plausible and temporally smooth per-channel arrival-delay tracks,
- TDOA residuals,
- geometric stability under reasonable event/TDOA settings,
- consistency with any available physical baselines or ground truth.
