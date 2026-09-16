# acoustic-self-calibration
Algorithm for self-calibration of stationary multi-microphone array and moving sound source.

This repository contains a UV-managed Python package for:

- generating artificial multichannel recordings of a moving 3D sound source,
- simulating directional radiation with a configurable starter cardioid pattern,
- exporting generated WAV/JSON test datasets for arbitrary microphone counts with a 12-microphone default,
- detecting repeated reference pulses without known emission timestamps,
- estimating microphone and source geometry from time-difference-of-arrival (TDOA) observations.

## Quick start

```bash
uv sync --dev
uv run pytest
```

## Self-calibration from multichannel audio

The self-calibration path does **not** require known source emission times. It detects a repeated reference pulse in every channel, converts those detections to TDOAs relative to channel 0, and jointly estimates microphone and source positions.

```python
from acoustic_self_calibration import (
    calibrate_from_audio,
    generate_synthetic_recording,
)

recording = generate_synthetic_recording(num_mics=12)
result = calibrate_from_audio(
    recording.audio,
    recording.scene.pulse,
    recording.scene.sample_rate,
    speed_of_sound=recording.scene.speed_of_sound,
    num_events=len(recording.scene.source_positions),
)

print(result.microphone_positions)
print(result.source_positions)
print(result.residual_rms)
```

For checked-in synthetic fixtures, `calibrate_from_dataset()` loads the WAV and reference pulse metadata but does not use the ground-truth microphone/source coordinates or the stored emission times for calibration.

## Known-range solver

`calibrate_from_distances()` remains available when absolute source-to-microphone ranges are already known. This is a different problem from TDOA self-calibration and is kept as a useful lower-level geometry solver.

Both solvers reject obviously underdetermined configurations and verify the local Jacobian rank before returning a calibration result.
