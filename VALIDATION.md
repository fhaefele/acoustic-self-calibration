# Synthetic audio validation

Validated on 2026-09-16 with the public `calibrate_audio(...)` API.

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

## Other automated checks

The test suite also covers:

- Gaussian/MAP recovery with noisy ideal timing data,
- relative channel clock-offset estimation,
- pairwise measurement-graph consistency,
- known-baseline sound-speed estimation,
- rejection of unanchored sound-speed estimation,
- Laplace microphone-position uncertainty,
- moving-source rendering and source directivity,
- GCC-PHAT delay sign and magnitude.

## Interpretation

These results validate the implementation under its current free-field synthetic model. They are **not** a prediction of accuracy in a reverberant room. Real recordings add multipath, channel-response differences, temperature/sound-speed changes, synchronization error, source occlusion, and incorrect direct-path peaks.

Clock-drift variables are implemented but remain experimental because slow trajectory changes and clock drift can be weakly separable without longer recordings or stronger priors.
