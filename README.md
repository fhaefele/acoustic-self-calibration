# acoustic-self-calibration

Bayesian/MAP **3-D acoustic self-calibration** for a **moving broadband sound source** and stationary microphone arrays, with 8–24 microphones as the primary tested range.

The package works directly from synchronized multichannel audio. It does **not** require a known calibration pulse or known source emission timestamps.

## What you get

From one multichannel recording, the solver estimates:

- stationary 3-D microphone positions,
- the moving source position at each analysis frame,
- per-axis microphone position uncertainty,
- per-axis source trajectory uncertainty,
- optional relative microphone clock offsets and drift,
- optional speed of sound when a metric microphone-distance anchor is provided,
- TDOA and posterior diagnostics.

Every run writes exactly two output files:

```text
RESULT.json
RESULT.png
```

The JSON is the complete machine-readable result. The PNG is one 2x2 figure containing a 3-D scene plus XY, XZ, and YZ projections.

## Run directly from GitHub

Because this repository is private, the one-shot `uvx` form uses your GitHub SSH credentials:

```bash
uvx \
  --from 'git+ssh://git@github.com/fhaefele/acoustic-self-calibration.git@main' \
  acoustic-selfcal \
  /path/to/file.wav \
  --output wavcalib
```

This produces:

```text
wavcalib.json
wavcalib.png
```

`--output` is an output prefix. Passing `--output wavcalib.json` is also accepted; the suffix is stripped before writing `wavcalib.json` and `wavcalib.png`.

## Ground-truth / reference evaluation

Provide a canonical scene JSON file to evaluate the recovered microphone geometry and source trajectory:

```bash
uvx \
  --from 'git+ssh://git@github.com/fhaefele/acoustic-self-calibration.git@main' \
  acoustic-selfcal \
  recording.wav \
  --ground-truth ground_truth.json \
  --output wavcalib
```

The output files remain exactly the same:

```text
wavcalib.json
wavcalib.png
```

With a reference scene:

- the JSON embeds the reference and adds an `evaluation` section,
- microphone RMS / mean / max errors are reported,
- source trajectory RMS / mean / max errors are reported,
- uncertainty-vs-error diagnostics are reported when posterior stds are available,
- the figure overlays estimated and reference microphones and source trajectory in all four panels.

The evaluation fits **one rigid transform from estimated microphones to reference microphones only**. That exact transform is then applied to the source trajectory. The source is never independently aligned. Reference source positions are interpolated to the acoustic analysis-frame times.

Calibration output uses the same canonical `scene` schema as GT input, so a previous result JSON can be used directly as the reference for a later run:

```bash
uv run acoustic-selfcal second.wav \
  --ground-truth first_calibration.json \
  --output second_vs_first
```

A reused calibration result keeps `scene_role: "estimate"`, making it explicit that it is a reference estimate rather than measured physical truth.

See [`docs/json_format.md`](docs/json_format.md) for the complete unified JSON schema.

## Ground-truth JSON helpers and validation

You do not need to hand-author GT JSON. Use the convenience function:

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

Or build the validated JSON object without writing a file:

```python
from acoustic_self_calibration import make_ground_truth_dict

payload = make_ground_truth_dict(
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
)
```

Validate a GT file or a previous calibration result before using it:

```python
from acoustic_self_calibration import validate_ground_truth_json

reference = validate_ground_truth_json("reference.json")
```

The canonical minimal GT schema is:

```json
{
  "schema_version": 1,
  "scene_role": "ground_truth",
  "scene": {
    "microphones": {
      "positions_m": [[0.0, 0.0, 1.2], [1.0, 0.0, 1.2], [0.0, 1.0, 1.2], [0.0, 0.0, 2.0]]
    },
    "source": {
      "times_s": [0.0, 0.1, 0.2],
      "positions_m": [[2.0, 0.0, 1.0], [1.95, 0.2, 1.02], [1.85, 0.4, 1.05]]
    }
  },
  "metadata": {
    "name": "trial_01"
  }
}
```

The previous pre-`scene` JSON schema is intentionally unsupported.

## Local checkout

```bash
uv sync --all-groups
uv run acoustic-selfcal recording.wav --output results/calibration
```

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

For channels with unknown relative timing:

```bash
uv run acoustic-selfcal recording.wav \
  --estimate-clock-offsets \
  --estimate-clock-drifts
```

Clock drift remains experimental.

Sound speed and scene scale are ambiguous from TDOAs alone. To estimate sound speed, provide at least one known microphone baseline:

```bash
uv run acoustic-selfcal recording.wav \
  --estimate-speed-of-sound \
  --distance-prior 0,1,1.234,0.002
```

This means microphones 0 and 1 are `1.234 m` apart with a `0.002 m` standard deviation.

## Python API

```python
from acoustic_self_calibration import (
    calibrate_wav,
    load_ground_truth_json,
    write_calibration_outputs,
)

result = calibrate_wav(
    "recording.wav",
    frame_size=1024,
    hop_size=4096,
    pair_mode="redundant",
    reference_count=2,
    likelihood="cauchy",
)

reference = load_ground_truth_json("reference.json")
paths = write_calibration_outputs(
    result,
    "results/calibration",
    input_wav_path="recording.wav",
    ground_truth=reference,
)

print(paths.json)
print(paths.figure)
```

For an in-memory array:

```python
from acoustic_self_calibration import calibrate_audio

result = calibrate_audio(
    audio,  # shape: (samples, microphones)
    sample_rate=48_000,
)
```

## Visualization

`RESULT.png` is one figure with:

1. **3-D scene**
2. **XY projection**
3. **XZ projection**
4. **YZ projection**

Every panel contains microphone positions and source trajectory. With a reference, every panel contains both the aligned estimate and reference scene. Microphone correspondence lines make geometry error visible directly. Sparse 1-sigma uncertainty ellipses are shown in the 2-D panels when uncertainty is available.

## Pipeline

```text
moving broadband source
        |
        v
multichannel waveform / WAV
        |
        v
redundant pairwise GCC-PHAT TDOAs
        |
        v
confidence -> heteroscedastic timing uncertainty
        |
        v
low-rank Euclidean scene initializer
        |
        v
Bayesian factor graph / MAP optimization
  - stationary microphone positions
  - moving source trajectory
  - motion prior
  - robust Cauchy-IRLS or Gaussian TDOA likelihood
  - optional clock offsets / drift
  - optional sound-speed estimation with metric anchor
        |
        v
joint Laplace uncertainty
        |
        +--> result JSON
        +--> 3-D + XY/XZ/YZ PNG
        +--> optional reference evaluation
```

## Recording requirements

For useful real-data calibration:

- use at least four microphones; 8–24 is the main tested range,
- channels should preferably share one sample clock,
- the source should contain broadband energy suitable for delay estimation,
- the source must move through a genuinely 3-D, non-degenerate trajectory,
- direct-path-dominant, high-SNR data will work substantially better than highly reverberant data,
- choose frame/hop sizes so geometry is approximately stationary inside a frame while enough independent source poses remain.

The absolute world coordinate frame is not observable from TDOAs. Normal estimates therefore live in the solver's canonical gauge. Reference evaluation removes that arbitrary rigid frame by aligning estimated microphones to reference microphones before reporting geometric error.

## Uncertainty interpretation

The reported position uncertainties are local **Laplace posterior standard deviations** around the MAP solution. Source uncertainty is marginalized over microphone geometry and enabled nuisance parameters.

They do not capture all real-world error sources: unmodeled room reflections, channel-response mismatch, incorrect direct-path peaks, imperfect GCC confidence calibration, or multimodal posterior structure.

## Validation

The automated tests cover:

- 8, 16, and 24 microphones,
- rendered moving cardioid broadband audio,
- PCM16 WAV -> calibration -> joint uncertainty,
- PCM16, true PCM24, PCM32, and float WAV decoding,
- pairwise GCC-PHAT extraction,
- low-rank initialization,
- microphone/source MAP recovery,
- microphone/source Laplace uncertainty,
- relative clock-offset estimation,
- sound-speed estimation with a known baseline,
- JSON-only output,
- canonical scene JSON helpers and validation,
- direct reuse of result JSON as reference input,
- explicit rejection of the old pre-`scene` schema,
- microphone-only rigid alignment for reference evaluation,
- source reference interpolation to estimate times,
- 3-D + XY/XZ/YZ visualization generation.

Run:

```bash
uv run pytest
uv run python examples/validate_synthetic_audio.py
```

See [`VALIDATION.md`](VALIDATION.md) for the synthetic free-field benchmark and limitations.

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
    initialization.py  low-rank geometry bootstrap + gauge packing
    pipeline.py        in-memory audio -> TDOA -> MAP API
    wav.py             WAV loading + calibrate_wav API
    ground_truth.py    scene JSON validation/read/write helpers
    evaluation.py      rigid alignment + reference metrics
    visualization.py   one 3-D/XY/XZ/YZ figure
    export.py          JSON result + PNG writer
    cli.py             acoustic-selfcal CLI
    tdoa.py            GCC-PHAT and redundant pair graphs
    simulation.py      continuous moving-source renderer
    radiation.py       source directivity patterns
    geometry.py        gauge, rigid alignment, error metrics
```

## Scope

This is a research-grade free-field baseline, not a claim that reverberant-room calibration is solved. Important real-world extensions remain explicit multipath/direct-path selection, measured microphone/channel responses, empirical TDOA uncertainty calibration, stronger outlier rejection, and mature asynchronous-clock models.

## License

MIT
