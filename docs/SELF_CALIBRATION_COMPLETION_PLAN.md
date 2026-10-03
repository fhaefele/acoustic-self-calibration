# Completion plan: acoustic self-calibration from broadband chirps

Status: implementation handoff; user requirements clarified. Unique no-prior
Myotis recovery remains an unmet requirement, as explained in section 2.
Audit date: 2026-10-02.
Audited revision: `00d22f82a4c353f881e46c1c243571fa1f32f2dd`.
Implementation branch: `rewrite/stratified-tdoa-calibration`; existing PR: #8.

## 1. Goal and confirmed inputs

Recover the fixed microphone positions and the moving sound source's positions
at detected broadband chirp events from synchronized multichannel recordings.
Sound speed is known. Emission times, microphone positions, source positions,
array aperture, arm angles, microphone spacings, and source trajectory are not
solver inputs. Do not use reference geometry to initialize, select, orient,
regularize, tune, or stop a solve.

Support these physical setups:

1. Planar microphones, with all source events on the same side of their plane.
2. Nonplanar microphones mounted on room/cave walls, with the source moving
   inside the physical room. The source may leave the microphone convex hull.

Return coordinates in meters in a documented microphone-relative frame, with
one identified microphone at `(0, 0, 0)`. Absolute world position and orientation
do not matter. A global reflection is also unobservable without external
orientation information. Metric scale comes from timing and known sound speed;
evaluation must not fit a scale factor.

No measured angle or distance is allowed, including for the Myotis recording.
The existing explicitly constrained solver can remain available for other uses,
but results using it do not satisfy this task.

Synthetic acceptance examples must use rendered broadband chirps and the public
audio/WAV path. Exact and noisy TDOAs are intermediate diagnostic tests, not a
substitute for end-to-end audio acceptance. Chirp shape may vary between calls;
neither clean emitted templates nor true emission times are available to the
production frontend. One active moving emitter is the initial scope.

### Resolved scope

- A room source may be outside the microphone convex hull while inside the room.
  Do not add a microphone-hull constraint to the solver.
- Treat the floor/ceiling as physical scene description, with no surveyed floor,
  ceiling, gravity direction, or wall coordinates supplied to the solver. This
  follows the no-geometry-input contract; it is not an estimated floor model.
- Direct sound only. Echoes, reflection rendering, reverberation, clock drift,
  multiple simultaneous emitters, and continuous-sound localization are outside
  this completion plan. Room/cave fixtures must have unobstructed direct paths
  for the observations they simulate.
- Agreed initial synthetic success targets: microphone RMS <0.05 m and source
  RMS <0.10 m on identifiable chirp scenes. Retain older tests as regression
  baselines, not as substitutes for these new targets.

Do not silently equate a room boundary with the microphone convex hull. Wall
microphones at limited heights can surround a source horizontally while their
3-D convex hull excludes positions near the floor or ceiling. Without supplied
floor geometry, gravity/floor position is not observable from direct-path TDOAs.

## 2. Verdict and mathematical limit

The branch partially meets the goal. It contains substantial calibration,
uncertainty, ambiguity, event timing, and synthetic validation machinery, but
does not yet provide the complete audio-to-relative-scene contract requested.
Passing its current tests does not establish that contract.

There is also an identifiability limit, not merely missing implementation. The
committed Myotis reference is an exact two-line planar cross. Its microphone and
source geometry cannot be uniquely recovered from timing and a common source
side alone. Fixing a microphone at the origin does not remove this ambiguity.

An intuitive special case: place a mic at the junction of two one-metre arms
and a mic at each arm's tip. Put the source two metres perpendicular to their
plane, directly in front of the junction. The three sound-path lengths are
`2, sqrt(5), sqrt(5)` metres for both a 90-degree arm angle and a 60-degree arm
angle. The tip microphones are respectively `sqrt(2)` and `1` metre apart.
Thus different microphone geometry can give identical arrival times. That
single-position example illustrates the issue; the following construction and
appendix establish it for the full moving Myotis reference path as well.

For a cross in a 2-D plane, let microphone coordinates be `m=(x,z)`, source
projections `u=(x,z)`, and positive source heights `h`. Define

```text
A(theta) = [[1, cos(theta)], [0, sin(theta)]]
m' = A m
u' = inverse(A).T u
h' = sqrt(||u||^2 + h^2 - ||u'||^2)
```

Cross microphones lie on one of the two axes, so `||A m|| = ||m||`.
Also `(A m).T u' = m.T u`, and total source squared distance from the junction
is preserved. Consequently every microphone-to-source range is identical when
the radicands are positive. A continuous range of angles preserves positive
heights and a smooth source path. TDOAs, direct propagation delays, and free-field
omnidirectional attenuation therefore cannot distinguish these scenes.

An audit calculation using `data/myotis/myotis_gt.json` changed the cross angle
from 90 to 60 degrees and obtained:

| Quantity | Result |
| --- | ---: |
| Maximum range difference | 4.45e-16 m |
| Maximum TDOA difference, c=343 m/s | 1.30e-18 s |
| Minimum alternative source distance in front of the plane | 0.78448 m |
| Microphone RMS difference after best rigid alignment | 0.20031 m |
| Junction microphone coordinate in both scenes | (0, 0, 0) |

Reference coordinates are used only to demonstrate this impossibility, never as
production solver inputs. Real placement departures from the ideal cross could
change identifiability; their effect must be supported by measurement evidence
above the noise level, not inferred from a favorable fitted residual.

**Unmet requirement:** unique, accurate recovery of the exact-cross Myotis
geometry under the confirmed input restrictions cannot be guaranteed. Report
the supported ambiguity and recoverable quantities. Do not mark a convenient
member of the family `solved`, choose the most nearly right-angle member, or
claim this limitation is fixed by more starts, more calls, or chirp bandwidth.
If unique Myotis recovery remains mandatory, completion is blocked unless the
available observations change. An additional microphone away from the two cross
arms is a possible acquisition improvement; its coordinates may remain unknown.
Verify the resulting design rank rather than promising that any added channel
automatically suffices.

## 3. Audit of the current branch

| Requirement | Evidence at audited revision | Work needed |
| --- | --- | --- |
| Fixed mics and moving event sources | `pipeline.py`, `stratified/solver.py`, and event audio tests exist | Retain and validate on required scenes |
| Common planar source side | `calibrate_planar_tdoa(source_half_space_sign=...)` and `_apply_source_half_space_prior` exist | Expose through audio/WAV/CLI; preserve through completion/refinement/export |
| Common side for incomplete results | Helper returns unchanged if any representative source is nonfinite | Handle valid events individually and report unresolved signs honestly |
| Room source region | `calibrate_tdoa` has no source-region constraint | Correct for unknown room boundaries; do not impose a mic hull |
| Room fixtures | Sources stay inside rectangular/irregular convex room walls | Retain and add explicit inside-room/outside-mic-hull acceptance cases |
| Chirp fixtures | `broadband_pulse_train` uses differentiated Gaussian noise with a Hann window | Add explicit broadband chirps and call-to-call variation |
| Source accuracy in new scene suites | `validate_scene` reports microphone RMS, timing RMS, and status | Add event-associated source metrics and coverage gates |
| Public coordinate frame | Gauge utilities exist; exported scene uses solver coordinates directly | Establish common output frame, origin ID, axes, and transform metadata |
| Planar JSON semantics | Export always labels positions `positive_plane_normal_representative` | Distinguish unsigned representative from explicitly one-sided coordinates |
| Myotis acceptance | Uses right angle `(3,0,11)`, permits `weakly_identified`, obtains evaluation-side sign from reference | Add no-prior production path and truthful ambiguity acceptance |
| Myotis source-side input | Harness sends sign to evaluation, not `calibrate_planar_tdoa` | Separate solver assumption from evaluation alignment |
| Evaluation timestamps | Measurements expose receiver event times; Myotis reference declares emission times | Audit and specify emission/arrival time alignment before tightening source gates |
| Physical echoes | Renderer explicitly uses direct sound only | Matches clarified scope; no echo implementation is required |
| Documentation | README/VALIDATION say real files are absent; handoff understates implemented low-level half-space support | Replace stale claims with current evidence |
| Fixture provenance | Myotis import calls WAV real; subsequent commit and JSON note refer to a frame used to “render” it | Establish recorded/derived/simulated provenance and reference time/frame history |

Focused audit verification on Python 3.12.14:

```text
pytest tests/test_stratified_planar_solver.py tests/test_simulation.py \
       tests/test_myotis_real_data.py -v --durations=10
35 passed, 1 WAV ancillary-chunk warning, 73.62 s
```

The real fixture test took 67.80 s. Its passing result proves its existing
constrained assertions, not no-prior recovery. Formatting, lint, type checks,
and CLI startup also passed during workspace setup. The existing GitHub matrix
at the audited revision passed Python 3.11–3.14; this audit did not rerun the
complete numerical suite locally.

An additional fixture check with seed 19 found 30/40 event sources outside the
microphone hull for the 8-microphone rectangular room and 22/40 for the irregular
room. Both satisfy the current wall-containment tests. Under the clarified goal,
these are valid scenes and useful regression cases, not invalid fixtures.

## 4. Accuracy, evidence, and failure policy

The user accepted the tighter 5 cm / 10 cm synthetic targets. They are completion
gates for the standard direct-sound chirp matrix, not already demonstrated
capabilities or permission to adjust ground truth. Distinguish low-SNR stress
characterization from that standard matrix as specified in section 6.

| Tier | Microphone RMS | Source RMS | Timing RMS | Purpose |
| --- | ---: | ---: | ---: | --- |
| Identifiable exact TDOA geometry | <1 mm | <1 mm | <0.01 us | Geometry correctness, no frontend noise |
| New standard direct-sound chirp acceptance | <5 cm | <10 cm | <60 us; separately report progress toward 20 us | Required on held-out identifiable scenes |
| Existing rendered-audio regression limits | <15 cm | <18 cm | <60 us | Mandatory preservation of existing gates |
| Existing PCM16 regression limits | <18 cm | <22 cm | <60 us | Mandatory preservation of existing gates |
| Existing constrained Myotis historical limits | <20 cm | <30 cm | <45 us fitted and validation | Historical only; constraint disallowed for this task |

For an identifiable scene, success requires `solved`, complete finite microphone
coordinates, required source-event coverage, source-region compliance where
applicable, and both geometry error limits. Low timing residual alone is not
success. Missing or rejected events must count in coverage denominators; never
report accuracy only on a favorable unacknowledged subset.

For known ambiguous scenes, success of the software means a correctly reported
ambiguity with evidence, not successful recovery of unique true coordinates.
Keep those counts separate in all reports. `weakly_identified` is not an accuracy
pass. Do not lower conditioning thresholds just to change that label.

Fit one translation/orthogonal transform from matching microphone IDs to
evaluate a scene. Apply it to all sources, without rescaling or per-event source
reflections. In planar same-side evaluation, any residual global normal reflection
must use the declared common-side convention. Never choose event signs from
reference source coordinates. Report in-plane and normal-height errors in
addition to full source RMS. Match events explicitly and document timestamp
basis; do not conceal emission/arrival-time errors by arbitrary time shifts.

## 5. Milestones

Implement in coherent commits. Every milestone must include the named behavior,
tests, and evidence before being marked complete. Keep unrelated redesign out of
this work. Use the existing stratified solver and covariance/lineage machinery.

### M0 — Freeze the contract and reproduce the identifiability evidence

Files: this document, `tests/test_myotis_geometry_report.py`,
`tests/test_stratified_planar.py`, `VALIDATION.md`.

1. Encode the confirmed input restrictions and success criteria in benchmark
   manifests. State exactly which physical facts are solver inputs versus
   simulation-only ground truth; no room/floor boundary or microphone hull is
   available to candidate selection.
2. Add a deterministic range-equivalent cross construction to a test helper.
   Exercise multiple angles, one common source side, fixed origin, and moving
   sources. Verify unequal inter-microphone distances and identical ranges/TDOAs.
3. Retain generic planar/star cases that have full metric design rank. Include
   near-cross cases to assess noise sensitivity without asserting global
   uniqueness from one local Jacobian.
4. Audit Myotis file provenance, channel order, sound-speed assumption, reference
   frame change, and emission-time convention. Record file hashes. Do not modify
   reference coordinates to improve a score. If provenance cannot be established,
   label it unverified and identify the missing evidence.

Exit checks: exact ambiguity proof passes at <1e-10 m range discrepancy; the
alternative microphones differ by >0.1 m after rigid alignment; all alternative
sources stay on the same side. The task's no-angle/no-distance policy is explicit
and Myotis unique recovery is recorded as blocked where appropriate.

### M1 — Add realistic chirp fixtures and a complete benchmark report

Files: `simulation.py`, `examples/validate_synthetic_audio.py`,
`tests/test_simulation.py`, `tests/test_events_detection.py`,
`tests/test_events_tdoa.py`; new `tests/test_chirp_simulation.py`.

1. Add a typed chirp specification: sample rate, start/end frequency, sweep law,
   duration, amplitude/envelope, polarity, phase, and deterministic seed. Use a
   windowed linear or logarithmic FM sweep; reject aliased bands. Include up/down
   sweeps and call-to-call bandwidth/duration/amplitude variation. Keep existing
   random-pulse fixtures unchanged as regression data.
2. Initial audible fixture: 48 kHz, 4–18 kHz sweeps, 0.8–2 ms duration. Add a
   high-rate ultrasonic fixture at 375 kHz with a safe 30–90 kHz band. These are
   synthetic settings, not a claim about the actual Myotis spectrum. Measure that
   spectrum before choosing the real-data processing band.
3. Include both 20/40 well-separated calls and an accelerating schedule with
   roughly 5 ms terminal spacing to exercise the failure fixed by the latest
   commit. Model propagation, source motion during a pulse, amplitude decay,
   fractional delays, channel gain changes, noise, and PCM quantization.
4. Keep the emitted signal, event times, geometry, and trajectory behind the
   simulator/evaluator boundary. Production calls receive audio, sample rate,
   sound speed, channel IDs, and permitted model assumptions only.
5. Generate geometry independently from the solver, and check delays using an
   independent analytic/static-position oracle plus a fractional-delay test.
   Never validate an interpolator only against its own inverse.
6. Add source RMS, per-event errors, detected/matched/rejected event counts,
   microphone maximum error, fitted/validation residuals, region compliance,
   configuration hash, seed, revision, and elapsed time to JSONL benchmark rows.
   Null means unavailable; missing timing residuals must not become zero.
7. Specify event matching and emission-versus-arrival timestamp semantics before
   comparing a moving source to a reference trajectory. Retain receiver arrival
   times separately. For a localized event, derive its emission-feature time as
   `t_emit = t_receive - norm(source - event_channel_microphone) / sound_speed`.
   Use the actual detection channel ID and the same waveform timing landmark
   (onset, envelope peak, or correlation reference) on both sides. Account for
   known detector/template delay; do not learn a time shift from source truth.
   Store the time basis in JSON and audit the Myotis reference landmark before
   treating its emission timestamps as interchangeable with detected peaks.
   Source truth may match events for evaluation only; matching cannot feed
   frontend or solver choices. Unlocalized events have unavailable emission
   times, not invented ones. Preserve IDs when checking time ordering.

Exit checks: chirp waveform/fractional-delay tests pass; clean separated and rapid
fixtures have the expected matched calls without merges; all benchmark rows carry
source metrics and coverage. A deliberately corrupted source estimate must fail
the benchmark even when its microphone estimate is accurate.

### M2 — Carry the same-side assumption through the public planar path

Files: `pipeline.py`, `wav.py`, `cli.py`, `stratified/solver.py`,
`stratified/refinement.py`, `export.py`, `evaluation.py`; public API/CLI/export tests.

1. Introduce an explicit common-side option in audio, WAV, and CLI calibration;
   use a simple physical meaning such as `--source-region same-side`. A caller
   need not know the world-plane normal. Choose positive output height as the
   coordinate convention and preserve low-level compatibility as needed.
2. Wire it to the existing planar half-space support. Reject incompatible
   general-3D options. Do not silently infer planarity from reference files.
3. Apply sign orientation to every finite completed event. Keep unresolved events
   unresolved; do not return a whole result unsigned because one source is missing.
   Preserve sign metadata and region checks after every completion/refinement.
4. Define finite-height tolerance and behavior for events on the microphone plane.
   Do not claim audio can detect a true side crossing: independent source
   reflections are acoustically indistinguishable for exactly planar receivers.
   The common side is a declared physical assumption, not a measured fact.
5. Export the assumption and its convention, valid-event mask, unsigned heights,
   and actual signed representative semantics. Replace the unconditional
   `positive_plane_normal_representative` label when inappropriate.

Exit checks: low-level TDOA, `calibrate_audio`, `calibrate_wav`, CLI, and JSON give
consistent one-sided scenes; channel reordering preserves IDs and physics; partial
results retain honest sign masks; refinement preserves the assumption. Blind
cross cases remain ambiguous even with the common-side option.

### M3 — Define and test the public microphone-relative coordinate frame

Files: `geometry.py`, result types in `stratified/solver.py`, `pipeline.py`,
`export.py`, `visualization.py`, `ground_truth.py`; coordinate/export tests.

1. Add an output-origin microphone ID, defaulting to channel ID 0 when present
   and otherwise the smallest valid ID. Report the selected ID. Do not confuse
   this output origin with the internally chosen timing-reference microphone.
2. Translate that microphone to zero. Choose well-separated microphone baselines
   for axes, with ID-based tie breaking. For planar output, put microphones in
   `z=0` and common-side sources in `z>0`. Avoid dependence on the first three
   rows, which can be collinear in Myotis.
3. Transform all scene-related quantities together: microphones, sources,
   projections, plane normals, uncertainty when present, selected hypotheses,
   and pre/post-refinement audit geometry. Document whether raw internal audit
   coordinates use their own frame, rather than mixing frames silently.
4. Record units, origin/axis IDs, frame convention, and transform metadata in
   exported JSON. Keep input channel order and event IDs intact.

Exit checks: selected origin norm <1e-10 m; pairwise distances and predicted TDOAs
are unchanged; arbitrary rigid transforms, channel permutations, and alternative
timing references preserve the physical result. Comparison uses one microphone
alignment with no scale fit and no independent source fitting.

### M4 — Validate room/cave geometry without inventing a boundary prior

Files: `simulation.py`, `examples/validate_synthetic_audio.py`, `pipeline.py`,
`export.py`; new `tests/test_room_chirp_calibration.py` and scene-validity tests.

1. Use the existing general-3D model for wall-mounted microphones. Do not add
   a room-shape inference system or reject source candidates because they lie
   outside the microphone convex hull. The actual room boundary is unknown.
2. Retain rectangular and irregular room fixtures. Generate wall-mounted mics
   at varied heights, with source motion in three dimensions that stays inside
   the physical room and between floor and ceiling. Check whole trajectory
   segments, not just emission points. Add near-floor/near-ceiling source paths
   within the room and clear of microphones.
3. Include both inside-hull and outside-hull source cases, including the seed-19
   eight-channel fixtures identified by this audit. Use the hull only for
   fixture classification/evaluation. Feed neither hull nor room geometry to
   the solver. Require source accuracy in both categories.
4. For any nonconvex cave fixture, validate source containment and source-to-mic
   line of sight for every simulated direct observation. An irregular convex
   room is an initial wall-array test, not evidence covering arbitrary caves.
5. Export the actual assumptions (`general_3d`, synchronized, known sound speed)
   separately from simulation metadata. Do not report a floor or room-boundary
   constraint as enforced. A candidate with otherwise valid acoustic evidence
   must not be downgraded solely for leaving the mic hull.

Exit checks: exact/noisy geometry and rendered chirp/WAV runs recover sources in
both hull categories; the standard matrix meets 5/10 cm limits with complete
coverage. Room coordinates never cross the solver input boundary. Unobservable
floor/world orientation does not change relative-scene accuracy.

### M5 — Make Myotis an honest no-prior end-to-end case

Files: `myotis.py`, Myotis examples, `tests/test_myotis_real_data.py`,
`tests/test_myotis_constrained_evaluation.py`, `export.py`, and data provenance docs.

1. Add a reproducible run using the same public WAV path as synthetic chirps:
   planar model and same-side assumption, no angle/distance input, no reference
   read until the geometry result is frozen. Use the recording's actual sample
   rate. Share frontend/configuration code rather than duplicating a special solve.
2. Separate the current constrained historical test from the no-prior acceptance
   case. Never cite its 20/30 cm thresholds as proof of no-prior recovery.
3. Remove reference-derived source-side selection from the production path.
   Evaluate common-side output using the documented global frame convention;
   do not reflect individual events toward reference positions.
4. Investigate candidate-family/near-degeneracy evidence from extracted timing.
   Report solved microphone/source coverage, unresolved IDs, conditioning,
   validation coverage, and family diagnostics. For the exact-cross fixture,
   return `ambiguous`/`degenerate`, with a representative/family only if meaningful.
5. Add a reference-independence test: changing/removing the evaluation reference
   must not change calibrated coordinates, event associations, or status. Reject
   silent synthetic fallback. The dedicated real-data gate must fail on a missing
   fixture rather than count a skip as successful real-data validation.
6. Preserve a runnable command and JSON/PNG report. If the data cannot support a
   unique scene, explicitly state which requested output remains unavailable and
   why; retain the original user requirement as unmet.

Exit checks: the committed recording is actually processed; no prohibited prior
is present; reference-independence and ambiguity tests pass; source/event coverage
is explicit. Unique real-geometry success is not an exit claim for an unidentifiable
cross. Do not implement arbitrary regularization to make that claim.

### M6 — Tighten accuracy on identifiable chirp scenes

Files: `events.py`, `stratified/solver.py`, `stratified/refinement.py`, benchmark
runner and focused regression tests, only where measured failures justify edits.

1. Establish separate error budgets for detection/association, TDOA extraction,
   geometric recovery with exact/noisy TDOAs, completion, and output evaluation.
   Diagnose the first failing stage before increasing search budgets.
2. Improve sub-sample chirp correlation, bandwidth handling, repeated-call
   association, and timing uncertainty from recorded windows. Exercise tracker
   enabled/disabled; any association smoothness setting must be explicit and must
   not become an undocumented source-motion/geometry prior.
3. Preserve timing covariance under re-referencing. Keep fitting, completion,
   and held-out validation lineage auditable. Retain frozen validation evidence
   through later polishing, and report separately any residual fitted afterward.
4. Audit automatic fallback searches and polishing already in `solver.py`.
   Record their use and budget; do not describe them as absent. Preserve physical
   feasibility, candidate ambiguity, and rollback under every path.
5. Meet 5 cm microphone / 10 cm source RMS on the preregistered standard
   identifiable chirp cases. If targets fail, report stage-specific errors and
   failure rates; fix the cause or mark the milestone incomplete. Keep old gates
   unchanged. Low-SNR stress results cannot substitute for standard acceptance.

Exit checks: no false `solved` result on structural ambiguity fixtures; all
mandatory legacy gates pass; new target attainment and coverage are reported per
scene family and noise level. Do not tune parameters on held-out evaluation seeds.

### M7 — Check reliability and failure reporting for direct chirps

Files: `events.py`, result diagnostics, CLI/export handling, chirp benchmark
manifest; new `tests/test_chirp_failure_modes.py`.

1. Characterize 20 dB and lower SNR, gain variation, clipped/quiet channels,
   missing calls, bad associations, near-collinear/near-planar room arrays, and
   low-diversity source motion. Keep ordinary direct chirp tests noise-relative
   and deterministic. Do not introduce echoes.
2. Define explicit coverage/completeness diagnostics. A partially localized
   recording must not masquerade as a complete scene. Preserve event IDs for
   rejected/unresolved events, uncertainty reasons, and validation denominators.
3. Ensure CLI non-success exit codes and machine-readable diagnostics for
   ambiguous, degenerate, insufficient-data, and failed runs. Where geometry is
   absent, still write a diagnostic JSON result; visualization can be omitted
   with an explicit reason. Avoid unhandled numerical exceptions in normal
   failure cases.
4. Check invariance to channel ordering, event IDs, signal amplitude, and timing
   reference changes. Search-budget increases must preserve earlier deterministic
   starts. Repeated runs with the same inputs/configuration must reproduce status
   and coordinates within documented numerical tolerances.

Exit checks: planted insufficient/degenerate cases fail honestly; stress reports
include geometry errors and unresolved events, not just residuals. All clean
acceptance cases remain accurate. No echo capability is claimed or required.

### M8 — Publish reproducible evidence and finalize the PR

Files: `README.md`, `VALIDATION.md`, `docs/json_format.md`, handoff/resume docs,
CI workflow, benchmark manifests/results, PR description.

1. Replace stale file-availability, source-side, and CI claims. Document normal
   user commands for planar same-side and the selected room mode, without a
   reference file as a solver requirement.
2. Store compact versioned benchmark manifests and results, not temporary logs or
   large generated audio dumps. Include revision, dependency lock, hashes, seed,
   sound speed, sample rate, pulse specification, acoustic conditions, exact
   options, timings, error/coverage metrics, and unsuccessful cases.
3. Add plots showing true/estimated microphones, source positions at matched calls,
   errors over time, source-region boundaries where defined, and ambiguous
   alternatives. Display withheld/missing events rather than silently connecting
   them into an apparently measured continuous trajectory.
4. Keep a fast PR subset and a full reproducible acceptance matrix. Run the
   dedicated Myotis case explicitly; verify it did not skip. Preserve the existing
   Python 3.11–3.14 lint/type/test/build/CLI checks.
5. Rewrite the PR description around verified behavior and explicit limitations.
   Distinguish accurate unique recoveries, correct ambiguity diagnoses, and
   remaining unmet user goals. Do not merge under a claim of complete Myotis
   recovery while its identifiability blocker remains.

Exit checks: a fresh checkout can reproduce the declared reports; all mandatory
checks pass; the acceptance matrix has no unacknowledged missing cases; each
advertised capability maps to an actual test/result.

## 6. Acceptance matrix and execution rules

Milestone dependencies and suggested commit units:

| Milestone | Requires | Primary deliverable/check |
| --- | --- | --- |
| M0 | Confirmed scope in section 1 | Identifiability proof and fixture provenance |
| M1 | M0 | Chirp renderer, independent delay checks, full benchmark metrics |
| M2 | M0 | Same-side input through TDOA/audio/WAV/CLI and partial-result tests |
| M3 | M2 | Shared output frame and round-trip/invariance tests |
| M4 | M1, M3 | Wall-array recovery inside room, including outside-mic-hull sources |
| M5 | M0, M2, M3 | Reference-independent Myotis run and explicit ambiguity evidence |
| M6 | M1–M5 | Measured fixes meeting the new identifiable-scene targets |
| M7 | M6 | Stress characterization and reliable diagnostic outputs |
| M8 | M0–M7 | Fresh-checkout reproduction, CI, documentation, final evidence |

For each milestone, store a compact outcome record with revision, exact command,
exit code, cases attempted/passed/failed/skipped, metric limits, and log/artifact
locations. A code change without its stated acceptance evidence is incomplete.

Use 8/12/16/24 channels and 20/40 events. Include generic planar, three-line star,
exact cross, near-cross, rectangular-wall, and irregular-wall layouts. For planar
fixtures retain 2 m aperture / 1–3 m source distance and 4 m / 1–6 m variants.
For room fixtures start with 6 × 5 × 3 m and vary wall mounting heights. These
dimensions are generation truth only.

Run exact TDOA, 2 us arrival-noise TDOA, rendered floating-point chirps, and PCM16
WAV stages. The standard audio acceptance matrix uses direct sound at clean and
40 dB signal-relative SNR and must meet the 5/10 cm targets. Use 20 dB and lower
as separately labeled stress characterization; report every outcome. Define SNR
using clean per-channel signal power over call-active samples, with that mask
available to the renderer/evaluator only. Do not use an unexplained absolute
noise amplitude. Exact cross cases belong to an ambiguity matrix, not the
recovery denominator. Keep near-cross sensitivity cases separate from generic
full-rank recovery cases and report their accuracy/status honestly.

Preregister development seeds 10–12 and evaluation seeds 30–34 before tuning.
Existing seeds 0–5 remain legacy checks. Use the full channel/event/scene matrix
for exact/noisy TDOAs. Start audio development with 8/12 channels and 20 events;
expand to the full matrix once those pass. Choose a documented stratified subset
for PR CI; run and archive the full evaluation before completion. If evaluation
failures lead to tuning, retain those results and reserve a fresh evaluation set.

For each stage, define expected case IDs and assert that all produced rows match
the manifest. Report runtime and memory, but do not invent an unmeasured runtime
SLA. Preserve output/exit status of long runs and avoid repeatedly restarting
successful expensive suites without a changed risk.

Commands for current baseline checks:

```bash
uv sync --locked --all-groups
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest \
  tests/test_stratified_planar_solver.py tests/test_simulation.py \
  tests/test_myotis_real_data.py -v --durations=10
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests examples
uv build
uv run asc calibrate --help
```

The implementing agent must add exact commands for every new benchmark mode
and its manifest when that mode is implemented. In restricted workspaces, use
writable cache locations such as `UV_CACHE_DIR=/tmp/asc-uv-cache`,
`XDG_CACHE_HOME=/tmp/asc-cache`, and `MPLCONFIGDIR=/tmp/asc-matplotlib`.

## 7. Implementation handoff rules

- Never supply angle, spacing, coordinates, aperture, source truth, or a true
  room hull to the blind solver. Physical shape assumptions must be explicit.
- Never weaken existing tests, change reference geometry, omit failed seeds, or
  relabel uncertain geometry merely to achieve a green summary.
- Establish geometry identifiability before optimizing accuracy or runtime.
- Keep the existing covariance, microphone/event ID, and measurement-lineage
  contracts. Avoid duplicate Myotis-only solvers or fixture-specific branches.
- Commit coherent milestone changes with their checks. Report what passed,
  failed, was skipped, and remains mathematically or empirically unresolved.
- Complete the implementation and provide reviewable evidence before proposing
  PR completion. Its no-prior Myotis uniqueness requirement remains unmet under
  the demonstrated ambiguity; the user has not agreed to count an ambiguity
  result as fulfilling unique recovery. Do not describe the whole goal as met.

## Appendix: reproduce the Myotis identifiability counterexample

Run from the repository root after installing dependencies. This calculation
uses reference geometry only as a mathematical counterexample, not as calibration
input. It does not need the WAV and does not establish the WAV's provenance.

```bash
uv run python - <<'PY'
import json
import numpy as np

with open('data/myotis/myotis_gt.json') as stream:
    scene = json.load(stream)['scene']
m = np.asarray(scene['microphones']['positions_m'], dtype=float)
s = np.asarray(scene['source']['positions_m'], dtype=float)
theta = np.deg2rad(60.0)
A = np.array([[1.0, np.cos(theta)], [0.0, np.sin(theta)]])
m_alt = m.copy()
m_alt[:, [0, 2]] = m[:, [0, 2]] @ A.T
s_alt = s.copy()
s_alt[:, [0, 2]] = s[:, [0, 2]] @ np.linalg.inv(A)
height_squared = np.sum(s * s, axis=1) - np.sum(s_alt[:, [0, 2]] ** 2, axis=1)
assert np.all(height_squared > 0)
s_alt[:, 1] = np.sqrt(height_squared)
r = np.linalg.norm(m[:, None, :] - s[None, :, :], axis=2)
r_alt = np.linalg.norm(m_alt[:, None, :] - s_alt[None, :, :], axis=2)
centered = m_alt - m_alt.mean(axis=0)
target = m - m.mean(axis=0)
U, _, Vt = np.linalg.svd(centered.T @ target)
aligned = centered @ (U @ Vt) + m.mean(axis=0)
error = np.sqrt(np.mean(np.sum((aligned - m) ** 2, axis=1)))
assert np.max(np.abs(r - r_alt)) < 1e-10
assert error > 0.1
assert np.allclose(m_alt[3], 0.0)
print('max range difference (m):', np.max(np.abs(r - r_alt)))
print('max TDOA difference (s):',
      np.max(np.abs((r - r[:1]) - (r_alt - r_alt[:1]))) / 343.0)
print('minimum alternative front distance (m):', s_alt[:, 1].min())
print('microphone RMS after rigid alignment (m):', error)
PY
```
