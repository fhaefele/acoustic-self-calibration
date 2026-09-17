# Synthetic audio validation

Validated on 2026-09-16 with the public `calibrate_audio(...)` and `calibrate_wav(...)` APIs.

This benchmark is end-to-end. It does **not** feed ideal geometric TDOAs into the solver. Each run:

1. creates a random 3-D microphone array,
2. creates a continuously moving non-planar source trajectory,
3. points a cardioid radiation pattern roughly toward the array center,
4. renders broadband audio with retarded emission time,
5. adds waveform noise,
6. extracts a redundant pairwise GCC-PHAT graph,
7. converts relative GCC confidence into heteroscedastic timing uncertainty,
8. initializes geometry with the low-rank Euclidean bootstrap,
9. solves the Bayesian/MAP factor graph with a robust Cauchy likelihood,
10. rigidly aligns the recovered scene to truth only for error reporting.

Run:

```bash
uv run python examples/validate_synthetic_audio.py
```

## Results

| Microphones | TDOA pairs | Source frames | Mic RMS error | Source RMS error | TDOA residual RMS | MAP evaluations |
|---:|---:|---:|---:|---:|---:|---:|
| 8  | 13 | 16 | 0.006390 m | 0.019613 m | 6.5637 us | 10 |
| 16 | 29 | 16 | 0.007571 m | 0.020350 m | 9.1339 us | 13 |
| 24 | 45 | 16 | 0.008282 m | 0.013458 m | 8.6676 us | 14 |

All runs reported optimizer success.

## Scenario

- sample rate: 16 kHz,
- duration: 8 s,
- analysis frame: 512 samples,
- analysis hop: 8192 samples,
- GCC interpolation: 16x,
- microphone bounds: x/y ±1.5 m, z 0–2.2 m,
- source path: non-planar elliptical loop,
- radiation: cardioid,
- pair graph: redundant, two reference roots,
- likelihood: Cauchy,
- motion prior: 3 m/s velocity-change sigma,
- waveform noise standard deviation: `1e-5` in normalized pressure units.

## WAV workflow regression

The automated suite writes the 8-microphone rendered scene to a PCM16 multichannel WAV file and runs the public `calibrate_wav(...)` path with Laplace uncertainty enabled. The test checks:

- WAV decoding and integer-to-float normalization,
- full audio -> pairwise TDOA -> MAP calibration,
- microphone RMS position error below 0.10 m,
- source RMS trajectory error below 0.10 m,
- finite microphone-position standard deviations,
- finite source-position standard deviations with the same `(frames, 3)` shape as the recovered trajectory.

Separate decoder tests exercise PCM16, true packed PCM24, PCM32, and float32 WAV files.

## Ground-truth evaluation regression

Ground-truth tests use a scene transformed by a known rigid transform and verify that:

- alignment is fitted from microphones only,
- the same transform recovers the source trajectory without independent source alignment,
- GT source samples are interpolated to estimate times,
- incomplete GT time coverage is rejected,
- exact transformed scenes produce near-zero microphone and source error,
- JSON output embeds the GT and aligned evaluation metrics,
- exactly one JSON result and one PNG figure are emitted,
- the PNG contains the 3-D plus XY/XZ/YZ comparison panels.

The project no longer emits NPZ or CSV result files.

## Laplace uncertainty

The solver partitions the final MAP Hessian into global parameters and source states. Microphone/global marginal covariance is obtained with the source block eliminated through a Schur complement. Source-state marginal variances then use the corresponding block-inverse identity, so source standard deviations include coupling to uncertain microphone geometry and enabled nuisance parameters.

These are local posterior standard deviations in the solver's gauge-fixed coordinate frame. They quantify curvature of the fitted model near the MAP solution; they do not capture multimodal ambiguity or model mismatch such as unmodeled room reflections.

The GT evaluation additionally reports a practical radial uncertainty diagnostic based on Euclidean position error divided by `sqrt(std_x^2 + std_y^2 + std_z^2)`. This is not presented as a formal 3-D Gaussian coverage probability because complete per-position covariance matrices and alignment uncertainty are not currently exported.

## Other automated checks

The test suite also covers:

- Gaussian/MAP recovery with noisy ideal timing data,
- relative channel clock-offset estimation,
- pairwise measurement-graph consistency,
- known-baseline sound-speed estimation,
- rejection of unanchored sound-speed estimation,
- marginalized microphone and source-position Laplace uncertainty,
- JSON GT read/write helpers and schema validation,
- JSON-only result output,
- moving-source rendering and source directivity,
- GCC-PHAT delay sign and magnitude.

## Interpretation

These results validate the implementation under its current free-field synthetic model. They are **not** a prediction of accuracy in a reverberant room. Real recordings add multipath, channel-response differences, temperature/sound-speed changes, synchronization error, source occlusion, and incorrect direct-path peaks.

Clock-drift variables are implemented but remain experimental because slow trajectory changes and clock drift can be weakly separable without longer recordings or stronger priors.
