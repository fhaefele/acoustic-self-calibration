# Completion evidence

Implementation plan: [SELF_CALIBRATION_COMPLETION_PLAN.md](SELF_CALIBRATION_COMPLETION_PLAN.md).
Starting revision: `31b64b8`. Each entry records completed checks; an absent entry
does not mean its milestone is complete.

## M0 — Contract and identifiability

The executable contract is `benchmarks/contract.json`. The four-angle counterexample
in `tests/test_cross_identifiability.py` verifies all 492 ranges with one fixed
alternative array per angle and strictly one-sided sources.

The committed Myotis files have a hash/provenance record in `data/myotis/provenance.json`.
The import commit labels the WAV real; the later reference-frame note calls it
rendered. Original acquisition evidence is unavailable in this repository, so
recorded-versus-derived provenance and the independent construction angle are
**unverified**. These labels are not inferred from matching ground truth.

Checks: cross counterexample and existing Myotis geometry/planar metric tests;
Ruff format/lint. The evaluation reference has not been changed.

## M1 — Chirp fixtures and source metrics

`chirps.py` adds windowed linear/logarithmic sweeps, alternating directions,
call-to-call variation, rapid schedules, ultrasonic bands, and signal-relative
noise. Legacy noise-pulse fixtures retain their original renderer and seeds.
`benchmarking.py` reports source errors, one-to-one matching, and coverage;
missing events and wrong sources prevent a success result.

Checks: 5 chirp/benchmark tests passed, including an independent static-path delay
oracle. Ruff and Ty passed. The seed-10 exact planar chirp case recovered all
20 source states and geometry to numerical precision. Audio accuracy remains a
later gate; this exact measurement test is not an audio acceptance claim.

## M2 — Public common-side assumption

Audio, WAV and CLI accept the declared common-side assumption and reject its use
with the spatial model. JSON distinguishes declared orientation from unsigned
representatives. Finite events keep their sign mask when other events are missing;
refinement is followed by the same physical orientation check. Strictly on-plane
events use the existing scale-relative 1e-12 tolerance and reject the strict-side
input. This assumption cannot detect acoustically invisible side crossings.

Checks: 26 API/CLI/export/planar tests passed; Ruff and Ty passed. A clean seed-10
chirp trial achieved 1.3 mm microphone and 8.2 mm source RMS, but was classified
`weakly_identified`, so it does **not** pass the standard acceptance gate. The
noise/conditioning classification remains an M6 investigation.

## M2a — Structured construction implementation

`ArrayConfiguration` is shared by audio/WAV and CLI `--array-config`. It records
membership, directed rays, exact angle, grid labels, independent optional equal
spacing relationships, and provenance. There are no metric pitches or square-cell
assumptions. Separated arm endpoints support arrays without a junction microphone.
Non-right metric recovery enumerates the signed feasible metric roots; noisy
full-rank equations are fitted under an exact nonlinear constraint.

Structured polishing uses reduced line/grid coordinates at every iterate. Unknown
spacings remain free; local uncertainty is evaluated within that parameter space.
Unknown-angle two-arm inputs retain a metric-family diagnosis, and thin two-row
grids do not claim unique row separation. Frozen pre-polish validation is retained.

Checks: 16 focused tests passed, including direct 60/75/90-degree cross/T chirps,
9-channel uneven/equal grid chirps through WAV, and angle/two-row ablations.
All tested recovery scenes met 5/10 cm and complete-event coverage. Ruff/Ty passed.
The broader preregistered channel/event/noise matrix remains an M6/M8 gate;
these focused checks alone do not establish full acceptance.

## M3 — Microphone-relative output frame

The public audio/WAV result defaults to microphone ID 0 as origin (or accepts
`--output-origin-microphone`). Baselines use distance and ID tie breaks, so the
first three channels may be collinear. Declared common-side sources occupy z>0;
planar microphones occupy z=0. JSON includes origin/axis IDs, units and the rigid
transform. Final, selected-candidate and refinement-audit scenes receive that
same transform; an audit origin can differ after refinement and is labeled.

Source-state timestamps now represent inferred emission features rather than
receiver arrivals. Receiver event times remain in the measurements section.
This distinction prevents propagation delay from shifting reference comparisons.

Checks: public-frame and numerical frame tests cover origin tolerance, unchanged
TDOAs, rigid-motion/channel-permutation invariance and exported time semantics.
Export/evaluation fixtures now declare the same emission-time basis explicitly.

## M4 — Direct room fixtures

The seed-19 rectangular and irregular convex-room fixtures are tested without
passing boundaries or microphone hulls to calibration. Wall/floor/ceiling
half-space checks cover trajectory segments; both inside- and outside-microphone-
hull calls are evaluated. Float chirps and PCM16 use the ordinary public room mode.
No general nonconvex-cave or echo capability is claimed. Near-floor/ceiling and
the larger channel/noise matrix remain part of the expanded acceptance run.

M4 status correction: rectangular seed 19 passed the strict gate. Irregular seed
19 recovered all 20 sources with 1.6 cm source RMS, but returned weakly identified;
that case therefore fails strict acceptance. Hull tests check geometry and coverage
independently; the benchmark's `standard_success` still requires `solved`. M4 is
not fully accepted until that status/conditioning issue and the expanded matrix
are resolved.

M4 fixture checks: 2 cases passed in 108.57 s (float and PCM16 geometry/coverage
checks, including both hull categories). Strict status acceptance is evaluated
separately and remains unmet for the irregular case under legacy timing weights.

## M5 — Public Myotis recovery and ablation

`examples/validate_myotis_public.py` freezes a public WAV result and runs an
arbitrary-planar ablation on the same measurements before loading the evaluation
reference. A reference-boundary regression fails if calibration attempts to read
that reference. Dedicated fixture absence fails instead of skipping. The JSON
report records the supplied 90-degree rays (3→0 and 3→11) and the unverified
acquisition/construction provenance.

Committed fixture outcome: microphone RMS 0.0513 m, source RMS 0.1718 m, fitted
TDOA RMS 5.14 us and frozen held-out RMS 36.11 us. There are 43 detected peaks,
41 localized sources, two unresolved IDs (41, 42), and 40 sources in reference
time overlap. Source comparisons use inferred emission features and one global
normal reflection; individual events are never reflected toward truth.

**The strengthened Myotis acceptance is not met:** status is weakly identified.
The acceptance command returns exit code 2 and the archived report sets accepted
false. Removing construction constraints does not produce a uniquely solved
scene. Independent physical construction evidence remains unavailable.

## M6 — Recorded-window timing and source completion

The optional `timing_uncertainty="waveform"` path estimates uncertainty from
recorded waveform residuals and slopes after fitting amplitude/offset. Correlated
residuals and split-window mismatch inflate uncertainty; no clean waveform or
true geometry is used. Confidence weighting remains the legacy default.

The chirp matrix exposed a completion defect: the first five channels of an
8-channel cross can lie on one arm. Completion now retains a well-conditioned
legacy subset, otherwise selects diverse receivers from recovered geometry alone.
Held-out timing values do not influence that selection. Directed construction
rays retain positive unknown lengths throughout structured refinement.

The preregistered PR subset passes all 10 exact/clean-float cases (star, cross90,
8/9-channel grid, rectangular room), requiring complete coverage, `solved`,
5/10 cm audio limits and 1 mm exact limits. The before-fix report retains its two
cross failures. This is a subset result, not full M6 acceptance. The development
and evaluation matrices remain required. Myotis accuracy/coverage and its weak
status are preserved by the completion fix.

Checks: 12 timing/frame/failure/configuration regressions passed; 10 focused
Myotis/configuration/grid/source-side regressions passed after the final subset
fix. Ruff lint/format and Ty checks pass. Exact commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python benchmarks/run_chirp_matrix.py \
  --suite pr --workers 2 --output benchmarks/results/chirp_pr_final.json
uv run pytest tests/test_myotis_public.py tests/test_output_frame.py \
  tests/test_waveform_uncertainty.py tests/test_chirp_failure_modes.py \
  tests/test_planar_array_configurations.py
uv run pytest tests/test_myotis_public.py tests/test_planar_array_configurations.py \
  tests/test_planar_grid_calibration.py tests/test_source_region.py
```

## M7 — Diagnostic output and identity invariance

CLI non-success runs now write JSON even when no geometry is available. They
return exit code 2 and explicitly omit visualization. Export reports detected,
measurement, localized and unresolved event counts/IDs; incomplete source
coverage downgrades a `solved` public result. Microphone IDs can be supplied in
channel order and survive configuration, measurement pairs and output frames.
A public-audio permutation regression checks the same physical frame to 3 mm
for microphones and 5 mm for sources. Event IDs survive edge-window rejection.

The quiet-recording and absent-geometry checks pass. Broader low-SNR, channel
failure and low-diversity characterization remains required for full M7 acceptance.

### M6/M7 expanded diagnosis and corrections

The first full development exact run attempted all 732 expected cases: all 552
standard identifiable recovery cases passed the 1 mm gates; 56 negative cases
reported degeneracy, four two-row grid cases reported numerical failure instead
of degeneracy, and 120 sensitivity/stress cases were characterized separately.
The two-row configuration handler now reports the exhibited one-dimensional
metric family, including with independent equal-pitch declarations. All 12
two-row exact rechecks pass. The original four failures remain archived.

Stress generation exposed hundreds of false detections at low SNR under the
relative-prominence-only detector. Detection now also requires energy above
the recorded median plus eight robust noise deviations. Nine seed/SNR checks
(10–12 × 40/20/10 dB) retain all 20 chirps; amplitude scaling preserves IDs.
The interrupted before-fix stress report retains 18/22 rows and lists four
unattempted cases explicitly. The corrected quick characterization completes
22/22: eight accurate solved recoveries, twelve failed/weak cases, and two
solved recordings missing one reference call each. Those two are coverage
failures, not complete accuracy passes. No echoes were introduced.

A nearly stationary room source previously produced incorrect `solved` geometry.
Spatial Jacobian sensitivity now removes source nuisance coordinates and six
rigid gauges, and records rank, conditional uncertainty and the applied limit.
Recorded-waveform timing uses the 5% relative precision gate. Legacy heuristic
confidence weights use an aperture-scale observability gate: their absolute
sigma calibration does not certify centimetre precision. Both limits are
exported; synthetic 5/10 cm accuracy/coverage gates remain unchanged. The
low-motion stress result now reports weak identification (relative uncertainty
17.7); normal room examples measure 0.006–0.013.

Checks: 50 detection/sensitivity/refinement/export/failure tests passed in
52.44 s; 17 configuration/public API/frame tests passed in 64.71 s. The final
legacy audio regressions and broader noisy/audio matrices remain required.

Final M7 focused run: 29 tests passed in 31.26 s (one WAV metadata warning);
Ruff format/lint and Ty pass. The 10-case reliability PR subset passes again.
The initial 96-case float/PCM development subset completed with 94 passes and
two failures: rectangular room, 12 channels, seed 12, 40 dB (float and PCM).
Those remain real M6 failures; clean versions passed and they are not hidden by
the successful PR subset. The full noisy-arrival matrix remains in progress.
