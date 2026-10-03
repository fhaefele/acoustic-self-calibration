# Acoustic self-calibration handoff

Updated 2026-09-24. This document contains the decisions, PR plan, results, and next steps needed to resume. No GitHub page, session history, `/tmp` log, or other documentation is required. Source and tests are in this repository.

## Current state and user instructions

- Branch `rewrite/stratified-tdoa-calibration`. PR #8 “Rewrite calibration around stratified TDOA geometry” is **open, not draft**; pushed through `9638491`. CI green on Python 3.11–3.14.
- User preferences: very concise updates and commits; prioritize performance, reliability, and predictable failure; avoid duplicate logic; use CLI for git; make coherent commits; do not weaken tests to hide failures. Parallel work is authorized when useful. For long commands, preserve output and use completion notifications rather than repeated polling.

## Why the benchmark changed

The initial request was to inspect the PR, fix its tickets, and make the code work. Early tests found inaccurate geometry with low TDOA residual. The user then defined the physical conditions under which self-calibration should work. Synthetic truth is for evaluation only, never branch selection or hidden solver constraints.

The accepted failure policy is: `solved` only for defensible geometry; `weakly_identified`, `ambiguous`, or `degenerate` for uncertain or non-unique geometry, with a candidate when useful; `failed`/`insufficient_data` when no usable estimate exists. A low residual alone does not prove correct geometry.

### Solver design requirements

Use event TDOAs to recover event range offsets, factor corrected ranges, upgrade the metric, complete extra microphones/events, then validate and optionally refine. Keep general-3D and planar-receiver/3D-source models distinct. Preserve channel/event IDs and separate measurements used for seed generation, completion, and validation; derived microphone pairs are not independent observations. Handle unequal timing uncertainty, missing data, and outliers without pretending a robust loss resolves geometric ambiguity. Any distance, angle, half-space, room, or motion prior must be explicit input and identified in output. Do not use source truth or a posterior score to select a branch. Refinement must start from a valid selected geometry, keep rollback behavior, and clearly distinguish its fitted residual from pre-refinement held-out evidence. The original plan called for a stratified primary path without automatic Bayesian/MAP or multistart fallback; the current solver has extra searches, so audit that gap rather than claim it is resolved.

### Agreed planar examples

- Known to solver: 8, 12, 16, or 24 microphone channels and IDs; synchronized recording; sound speed; microphone planarity; source on one side of the plane.
- Unknown to solver: all microphone coordinates, spacings, arm angles, and physical aperture. Do **not** pass simulated aperture or a right-angle cross constraint to the blind benchmark.
- Generate cross/T-like and three-line star layouts. Use a 2 × 2 m aperture with perpendicular source distance 1–3 m, and a 4 × 4 m aperture with distance 1–6 m. Move the source laterally and in depth. Run 20- and 40-event cases, exact TDOAs first, then noisy TDOAs and rendered audio, across several development and fresh evaluation seeds.
- Crosses are ambiguity tests. A planar cross retains a continuous metric ambiguity even with exact ranges and a known source side: changing its arm angle can be compensated by source coordinates/heights. Extra events do not remove it. An externally supplied arm angle is a separate constrained test.
- Contract gap: `make_planar_benchmark_pulse_scene` generates sources on one side, but `calibrate_planar_tdoa` does not accept/enforce that half-space. Its heights remain unsigned. The benchmark reports `source_region_constraint_enforced: false`. Implement an explicit prior and signed-height semantics before claiming full support for the agreed setup.

### Agreed room examples

- Rectangular and irregular convex rooms, currently 6 × 5 m floor and 3 m high. Place 8/12/16/24 microphones at varied wall positions and heights; move the source in three dimensions strictly inside.
- Room shape, walls, mic positions, and source truth remain hidden from the blind general-3D solver. Current fixtures use synchronized direct sound plus noise. **No wall reflections are simulated yet.** Room containment is not a solver input. Do not claim reflection robustness or a known-wall constraint.
- Fixtures live in `src/acoustic_self_calibration/simulation.py`. `examples/validate_synthetic_audio.py` supports `--suite planar|room`, `--stage exact|noisy|audio`, mic counts, events, seeds, layouts, and planar spans. It streams JSON results with status, runtime, TDOA residual, mic RMS after rigid alignment, and diagnostics.

## Implemented in the working tree

- Event timing: fractional-delay interpolation from filtered local waveform windows, raw-correlation peak selection, narrowband fallback.
- General geometry: dimension-aware offset expansion and optional rank refinement; negative inferred ranges reject their candidate; noise-scaled completion tolerance; approximate branches receive source refresh/Huber refinement; widespread residuals inconsistent with timing uncertainty downgrade status. Noisy expanded arrays get joint all-mic refinement with rollback.
- Planar geometry: fitting-only diverse event seeds plus a temporal seed; corrected squared-range noise allowance; exact explicit right-angle constraint; metric conditioning and local Jacobian sensitivity after projecting out source and rigid-gauge directions. A Gaussian goodness-of-fit check can downgrade `solved`. These are local diagnostics, not a global uniqueness proof.
- Refinement: shared covariance whitening and analytic Jacobian. Ambiguity comparison considers later comparable candidates.
- CI: draft-aware trigger, Python 3.11–3.14 matrix, bounded BLAS threads. Run `35992415084` **success** on all four versions after ready-for-review.

## Measured results and limits

Local results for the current tree (2026-09-24), `OPENBLAS_NUM_THREADS=1`. Accuracy gates unchanged: mic/source RMS 0.15/0.18/0.22 m, TDOA residual 60 µs.

| Run | Result | Limit |
| --- | --- | --- |
| Exact planar seeds 0–2 (96 rows: 4 mic × 2 span × 2 layout × 2 events × 3 seeds) | 48/48 stars `solved`, mic RMS 0.0; 48/48 crosses `degenerate`; 0 false-solved crosses | Full matrix |
| Noisy planar seeds 0–2, 2 µs (96 rows) | 0 false-solved crosses; star 46 `solved` / 2 weak; cross 37 weak / 11 failed; max solved mic RMS 8.1 mm | Full matrix |
| Noisy planar fresh seeds 3–5, 2 µs (pre noise-gate tighten) | 1 false-solved cross (16 mic/20 ev/seed 5/span 2, mic RMS 0.39 m, relative std 0.094) | Found and fixed |
| Noisy planar seeds 3–4 + 5 after `relative_weakest_std` 0.1→0.05 | seed3–4: 32 star `solved`, 0 false; seed5: 12 star `solved`, 0 false; 0 solved >0.15 m | Full fresh recheck |
| Exact room seeds 0–2 (48 rows) | 48/48 `solved`, mic RMS 0.0 | Full matrix |
| Noisy room 2 µs (48 rows) | 48/48 `solved`, max mic RMS 8.2 mm | Full matrix |
| Audio T-004 `test_stratified_audio_{spec,8mic,large_arrays}`, `test_stratified_wav`, `test_end_to_end_audio` | 17 + 6 passed (gates 0.15/0.18/0.22 m, 60 µs) | 42 min wall |
| Planar audio fresh seeds 3–5 (96 rows) | 0 false-solved crosses; 0 solved >0.15 m; star 38 solved / 10 weak; cross 41 weak / 7 failed | Optional confidence run |
| Non-audio pytest (`tests/` minus audio files) | 266 passed, 0 failed | Full local suite |
| GitHub CI PR #8 run `35992415084` | Python 3.11/3.12/3.13/3.14 all success (full suite + ruff + ty + build + CLI) | Remote |
| Ruff check/format, Ty, `uv build`, `asc --help` | Passed | After final edits |

Known limits: no wall-reflection simulation; room containment not a solver input; planar cross unconstrained path is a continuous-ambiguity test (expect weak/degenerate, never false `solved`); planar noise sensitivity is a local diagnostic.

## PR ticket plan, reproduced here

- **T-001, committed:** Ruff format/lint clean.
- **T-002, committed:** deterministic physical 7-receiver/6-source starts; exact 8-mic 20/40-event recovery; prefix-stable Halton/metric/event seed families.
- **T-003, committed:** exact 8/12/16/24-mic completion for 20/40 events with channel IDs and lineage intact.
- **T-004, verified locally:** audio + WAV suites pass with gates 0.15/0.18/0.22 m and 60 µs (17+6 tests). Large-array audio green.
- **T-005, committed:** reject infeasible planar candidates locally; noisy planar regression; 0 false-solved crosses on seeds 0–5 after noise-sensitivity gate 0.05.
- **T-006, verified locally:** constrained Myotis, unconstrained exact cross degeneracy, ID survival, unsigned-height evaluation tests pass.
- **T-007, verified locally:** model-selection and expansion membership-constraint tests pass.
- **T-008, verified locally:** focused suites, Ruff, Ty, CLI smoke, `uv build`; no debug-only artifacts.
- **T-009, done:** draft-aware CI trigger and py 3.11–3.14 matrix verified; run `35992415084` green; PR #8 ready for review (not draft).

## Next sequence when explicitly resumed

1. Merge PR #8 if desired, or open follow-ups for remaining product gaps (wall reflections, room containment prior, signed planar half-space enforcement).
2. Optional: more fresh-seed planar audio if rate confidence is needed beyond seeds 0–5.

Run checks from this repository. Examples:

```bash
uv run --no-sync python examples/validate_synthetic_audio.py --suite planar --stage exact --events 20 --seeds 0 1 2
uv run --no-sync python examples/validate_synthetic_audio.py --suite planar --stage noisy --events 20 --seeds 0 1 2 --timing-noise-us 2
uv run --no-sync python examples/validate_synthetic_audio.py --suite room --stage exact --events 20 --seeds 0 1 2
uv run --no-sync python examples/validate_synthetic_audio.py --suite room --stage noisy --microphones 8 --events 20 --seeds 0 --timing-noise-us 2
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check src tests examples
```

Use `OPENBLAS_NUM_THREADS=1` for solver-heavy runs. Preserve output and exit codes for long checks. Run broader suites only for a concrete remaining risk or required gate.
