# acoustic-self-calibration

Bayesian/MAP **3-D acoustic self-calibration** for a **moving broadband sound source** and stationary microphone arrays, with 8–24 microphones as the primary tested range.

The package works directly from synchronized multichannel audio. It does **not** require a known calibration pulse or known source emission timestamps.

## What you get

From one multichannel recording, the solver estimates:

- stationary 3-D microphone positions,
- the moving source position at each analysis frame,
- per-axis microphone and source-position uncertainty,
- optional relative microphone clock offsets and drift,
- optional speed of sound when a metric microphone-distance anchor is provided,
- TDOA and posterior diagnostics.

Each calibration writes exactly:

```text
RESULT.json
RESULT.png
```

The JSON is the complete machine-readable result. The PNG is one 2x2 figure containing a 3-D scene plus XY, XZ, and YZ projections.

## CLI

The installed command is `asc`. The CLI is intentionally organized around subcommands:

```text
asc calibrate WAV
asc check JSON
asc compare ESTIMATE_JSON REFERENCE_JSON
```

The previous flat `acoustic-selfcal recording.wav ...` command is intentionally unsupported.

### Calibrate

```bash
asc calibrate recording.wav -o run01
```

With a reference scene:

```bash
asc calibrate recording.wav -o run01 -r reference.json
```

Only output and reference have short flags:

```text
-o, --output PREFIX
-r, --reference JSON
```

Solver options use long names only:

```text
--frame-size INT
--hop-size INT
--max-tau-ms FLOAT
--gcc-interp INT
--pair-mode {reference,redundant,all}
--reference-count INT
--likelihood {cauchy,gaussian}
--speed-of-sound FLOAT
--motion-sigma-mps FLOAT
--estimate-clock-offsets
--estimate-clock-drifts
--estimate-speed-of-sound
--distance-prior MIC_A,MIC_B,DISTANCE_M,SIGMA_M
--best-sigma-samples FLOAT
--worst-sigma-samples FLOAT
--max-nfev INT
--no-uncertainty
```

A typical real-data command is:

```bash
asc calibrate recording.wav \
  -o results/run01 \
  --frame-size 1024 \
  --hop-size 4096 \
  --pair-mode redundant \
  --reference-count 2 \
  --likelihood cauchy
```

Sound speed and scene scale are ambiguous from TDOAs alone. To estimate sound speed, provide at least one known microphone baseline:

```bash
asc calibrate recording.wav \
  --estimate-speed-of-sound \
  --distance-prior 0,1,1.234,0.002
```

### Check a scene JSON

Both physical ground truth and calibration outputs use the same canonical scene schema. Validate either form with:

```bash
asc check reference.json
```

Successful output includes the scene role, microphone count, source-state count, and source time range. Invalid input exits nonzero with a validation error.

The same validation is available from Python:

```python
from acoustic_self_calibration import validate_ground_truth_json

reference = validate_ground_truth_json("reference.json")
```

### Compare two scene JSON files

Compare an estimate against a reference without recalibrating audio:

```bash
asc compare run02.json truth.json -o run02_vs_truth
```

or compare two calibration runs:

```bash
asc compare run02.json run01.json -o run02_vs_run01
```

This writes:

```text
run02_vs_run01.json
run02_vs_run01.png
```

The comparison uses the same microphone-only rigid alignment and source-trajectory interpolation as calibration-time reference evaluation. The first positional file is always the estimate being evaluated; the second is the reference.

### Help and version

```bash
asc --help
asc --version
asc calibrate --help
asc check --help
asc compare --help
```

## Run directly from GitHub

Because this repository is private, a one-shot invocation can use your GitHub SSH credentials:

```bash
uvx \
  --from 'git+ssh://git@github.com/fhaefele/acoustic-self-calibration.git@main' \
  asc calibrate \
  /path/to/file.wav \
  -o wavcalib
```

## Canonical scene JSON

Ground-truth input and calibration output share one schema. The minimal GT form is:

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
  }
}
```

Calibration results use `scene_role: "estimate"` and add uncertainties, measurements, settings, diagnostics, and optionally evaluation. A previous calibration JSON can therefore be used directly with `-r/--reference` or `asc compare`.

The pre-`scene` JSON schema is intentionally unsupported. See [`docs/json_format.md`](docs/json_format.md) for the complete format.

## Ground-truth JSON helpers

```python
from acoustic_self_calibration import make_ground_truth_dict, write_ground_truth_json

payload = make_ground_truth_dict(
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
)

write_ground_truth_json(
    "ground_truth.json",
    microphone_positions_m=microphones,
    source_times_s=times,
    source_positions_m=trajectory,
)
```

## Reference evaluation

The evaluation fits **one rigid transform from estimated microphones to reference microphones only**. That exact transform is applied to the source trajectory; the source is never independently aligned. Reference source positions are interpolated to the acoustic analysis-frame times.

The JSON reports microphone and source RMS / mean / max errors plus uncertainty-vs-error diagnostics when posterior standard deviations are available. The figure overlays the aligned estimate and reference in all four panels.

## Python API

```python
from acoustic_self_calibration import (
    calibrate_wav,
    load_ground_truth_json,
    write_calibration_outputs,
)

result = calibrate_wav("recording.wav")
reference = load_ground_truth_json("reference.json")
paths = write_calibration_outputs(
    result,
    "results/calibration",
    input_wav_path="recording.wav",
    ground_truth=reference,
)
```

For an in-memory array:

```python
from acoustic_self_calibration import calibrate_audio

result = calibrate_audio(audio, sample_rate=48_000)
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
- canonical scene JSON validation,
- calibration-result reuse as reference input,
- `asc calibrate`, `asc check`, and `asc compare`,
- standalone scene comparison JSON+PNG output,
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
    evaluation.py      rigid alignment + scene comparison metrics
    visualization.py   3-D/XY/XZ/YZ figures
    export.py          JSON + PNG writers
    cli.py             asc subcommand CLI
    tdoa.py            GCC-PHAT and redundant pair graphs
    simulation.py      continuous moving-source renderer
    radiation.py       source directivity patterns
    geometry.py        gauge, rigid alignment, error metrics
```

## Scope

This is a research-grade free-field baseline, not a claim that reverberant-room calibration is solved. Important real-world extensions remain explicit multipath/direct-path selection, measured microphone/channel responses, empirical TDOA uncertainty calibration, stronger outlier rejection, and mature asynchronous-clock models.

## License

MIT
