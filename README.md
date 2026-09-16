# acoustic-self-calibration
Algorithm for self-calibration of stationary multi-microphone array and moving sound source.

This repository now contains a minimal UV-managed Python package for:

- generating artificial multichannel recordings of a moving 3D sound source,
- simulating directional radiation with a configurable starter cardioid pattern,
- exporting generated WAV/JSON test datasets for arbitrary microphone counts with a 12-microphone default,
- estimating microphone and source geometry from those synthetic datasets.

## Quick start

```bash
uv sync --dev
uv run pytest
```

## Basic usage

```python
from acoustic_self_calibration import (
    calibrate_from_dataset,
    export_synthetic_dataset,
    generate_synthetic_recording,
)

recording = generate_synthetic_recording(num_mics=12)
wav_path, metadata_path = export_synthetic_dataset("tests/data", recording=recording)
result = calibrate_from_dataset(wav_path, metadata_path)
```
