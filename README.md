# acoustic-self-calibration

Bayesian/MAP **3-D acoustic self-calibration** for a moving transient/broadband sound source and a stationary microphone array, with 8–24 microphones as the primary tested range.

> **Project status:** this is a work-in-progress, vibe-coded research project. Expect rough edges, breaking changes, and evolving APIs while the approach is actively developed and validated.

The calibration frontend is **event driven**. It detects discrete acoustic emissions, associates their arrivals across channels, and sends only those source events to the geometry solver. Silence and arbitrary uniformly spaced WAV frames are not treated as source observations.

## Method

For a synchronized multichannel recording the current pipeline is:

1. detect transient/broadband events on a strong microphone channel,
2. retain multiple delay candidates for each event and channel,
3. track a temporally smooth delay sequence to reject neighbouring-call / repeated-waveform ambiguities,
4. convert the selected per-channel arrivals into cycle-consistent pairwise TDOAs,
5. initialize microphone and source geometry from the TDOAs,
6. jointly refine microphone positions and one 3-D source position per event with the Bayesian/MAP solver,
7. optionally compute a local Laplace uncertainty approximation.

This follows the same broad structure-from-sound philosophy as approaches that treat acoustic correspondence / TDOA association as a separate problem before geometric optimization, rather than assuming every arbitrary time frame contains a valid source observation.

The package does **not** require known source positions or known emission timestamps for calibration. Reference JSON is evaluation-only.

## What you get

From one multichannel recording, the solver estimates:

- stationary 3-D microphone positions,
- one moving-source 3-D position per detected acoustic event,
- the detected event times and event-channel sample indices,
- selected per-channel arrival delays and pairwise TDOAs,
- per-axis microphone and source-position uncertainty,
- optional relative microphone clock offsets and drift,
- optional speed of sound when a metric microphone-distance anchor is provided,
- posterior and TDOA diagnostics.

Each calibration writes exactly:

```text
RESULT.json
RESULT.png
```

The JSON is the complete machine-readable result. The PNG is one 2x2 figure containing a 3-D scene plus XY, XZ, and YZ projections.

## CLI

```text
asc calibrate WAV
asc check JSON
asc compare ESTIMATE_JSON REFERENCE_JSON
```

### Calibrate

```bash
asc calibrate recording.wav -o run01
```

With a reference scene for evaluation:

```bash
asc calibrate recording.wav -o run01 -r reference.json
```

Important event/TDOA options:

```text
--event-channel INT          detection channel; default is automatic
--event-smooth-ms FLOAT      event-energy smoothing window
--event-min-gap-ms FLOAT     minimum separation between detected events
--event-prominence FLOAT     relative event-prominence threshold
--max-tau-ms FLOAT           maximum inter-channel delay search
--tdoa-envelope-ms FLOAT     smoothing used for delay association
--tdoa-template-ms FLOAT     event template duration
--tdoa-candidates INT        delay hypotheses retained per event/channel
--max-tdoa-rate FLOAT        temporal delay-change scale
--tdoa-track-weight FLOAT    smooth-track transition weight
--pair-mode {reference,redundant,all}
--reference-count INT
```

Model / solver options:

```text
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

A typical pulsed-source command is simply:

```bash
asc calibrate recording.wav \
  -o results/run01 \
  --pair-mode reference \
  --likelihood cauchy
```

For very rapid pulses, lower the event gap. For example:

```bash
asc calibrate recording.wav \
  -o results/run01 \
  --event-min-gap-ms 2.5 \
  --max-tau-ms 10
```

If automatic event-channel selection is poor, specify a channel with clearly visible direct arrivals:

```bash
asc calibrate recording.wav --event-channel 2 -o run01
```

Sound speed and scene scale are coupled. To estimate sound speed, provide at least one known microphone baseline:

```bash
asc calibrate recording.wav \
  --estimate-speed-of-sound \
  --distance-prior 0,1,1.234,0.002
```

The old `--frame-size`, `--hop-size`, and `--gcc-interp` calibration controls are intentionally removed. Calibration no longer samples the recording on a uniform frame grid.

### Check a scene JSON

```bash
asc check reference.json
```

Both ground truth and calibration outputs use the same canonical scene schema.

### Compare two scene JSON files

```bash
asc compare run02.json truth.json -o run02_vs_truth
```

or:

```bash
asc compare run02.json run01.json -o run02_vs_run01
```

Comparison fits one rigid transform from the microphones only, then applies that same transform to the source trajectory.

## Run directly from GitHub

```bash
uvx \
  --from 'git+https://github.com/fhaefele/acoustic-self-calibration.git@main' \
  asc calibrate \
  /path/to/file.wav \
  -o wavcalib
```

Check:

```bash
uvx \
  --from 'git+https://github.com/fhaefele/acoustic-self-calibration.git@main' \
  asc check scene.json
```

Compare:

```bash
uvx \
  --from 'git+https://github.com/fhaefele/acoustic-self-calibration.git@main' \
  asc compare estimate.json reference.json -o comparison
```

## Canonical scene JSON

Minimal ground truth:

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

Calibration results use `scene_role: "estimate"` and add event measurements, uncertainties, settings, diagnostics, and optionally evaluation.

The source times in an estimate are the detected event times on the selected event channel. A reference only needs to cover the portion of the event sequence for which source ground truth exists; source metrics are evaluated over the overlapping time interval.

See [`docs/json_format.md`](docs/json_format.md) for the complete format.

## Python API

```python
from acoustic_self_calibration import calibrate_wav

result = calibrate_wav("recording.wav")
print(result.event_channel)
print(result.event_times_s)
print(result.calibration.microphone_positions)
print(result.calibration.source_positions)
```

For an in-memory array:

```python
from acoustic_self_calibration import calibrate_audio

result = calibrate_audio(audio, sample_rate=48_000)
```

The transient frontend is also exposed directly:

```python
from acoustic_self_calibration import detect_transient_events, estimate_event_tdoas
```

## Synthetic pulsed source

The simulation utilities can generate discrete broadband emissions and propagate them from a moving source with source directivity:

```python
import numpy as np
from acoustic_self_calibration.simulation import broadband_pulse_train, render_moving_source

sample_rate = 48_000
event_times = np.linspace(0.4, 4.4, 20)
signal = broadband_pulse_train(event_times, sample_rate, 5.0)

audio = render_moving_source(
    signal,
    sample_rate,
    microphones,
    trajectory_times,
    trajectory_positions,
    source_forward=source_forward,
    radiation_pattern="cardioid",
)
```

The automated end-to-end tests use this path; they do not feed ideal TDOAs directly to the geometry solver.

## Recording requirements

For useful real-data calibration:

- use at least four microphones; 8–24 is the main tested range,
- channels should preferably share one sample clock,
- the source should emit discrete transients or short broadband calls that can be associated across microphones,
- event spacing should be large enough that direct arrivals can be distinguished from neighbouring emissions for the chosen maximum TDOA search,
- the source must move through a genuinely 3-D, non-degenerate set of positions,
- direct-path-dominant, high-SNR data works substantially better than strongly reverberant data,
- if a recording is highly directional, choose an event channel on which the direct calls are consistently visible.

The current event association keeps multiple delay candidates and applies temporal smoothness, but it is still a free-field baseline rather than a complete multipath solver.

## Reference evaluation

The evaluation fits **one rigid transform from estimated microphones to reference microphones only**. That exact transform is applied to the source trajectory; the source is never independently aligned. Reference source positions are interpolated to the overlapping detected-event times.

The JSON reports microphone and source RMS / mean / max errors plus uncertainty-vs-error diagnostics when posterior standard deviations are available.

## Uncertainty interpretation

The reported position uncertainties are local **Laplace posterior standard deviations** around the MAP solution. They do not capture every real-world error source, especially multipath ambiguity, channel-response mismatch, event misassociation, or multimodal posterior structure.

## Validation

Automated tests cover:

- event detection from sparse broadband pulse trains,
- multi-hypothesis / smooth-track inter-channel delay association,
- exact cycle consistency of selected TDOAs,
- end-to-end rendered moving cardioid pulse calibration with 8, 16, and 24 microphones,
- PCM16 WAV -> event detection -> calibration -> joint uncertainty,
- WAV decoding,
- low-rank initialization,
- microphone/source MAP recovery,
- clock and sound-speed model tests,
- canonical scene JSON validation and comparison,
- JSON + PNG output generation,
- CLI smoke tests.

Run:

```bash
uv run pytest
uv run python examples/validate_synthetic_audio.py
```

## Development

```bash
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests examples
uv run pytest
uv build
```

## Package layout

```text
src/acoustic_self_calibration/
    events.py          transient detection + multi-candidate TDOA tracking
    pipeline.py        event-driven audio -> MAP calibration API
    bayesian.py        sparse MAP factor graph + joint Laplace uncertainty
    initialization.py  low-rank geometry bootstrap + gauge packing
    wav.py             WAV loading + calibrate_wav API
    ground_truth.py    scene JSON validation/read/write helpers
    evaluation.py      rigid alignment + scene comparison metrics
    visualization.py   3-D/XY/XZ/YZ figures
    export.py          JSON + PNG writers
    cli.py             asc subcommand CLI
    tdoa.py            low-level GCC-PHAT utilities and pair graphs
    simulation.py      pulse generation + moving-source renderer
    radiation.py       source directivity patterns
    geometry.py        gauge, rigid alignment, error metrics
```

## Scope

This is a research-grade free-field baseline. Important real-world extensions include stronger direct-path/multipath reasoning, joint multi-channel event association, measured microphone/channel responses, empirical delay-uncertainty calibration, and more mature asynchronous-clock models.

## License

MIT
