# Acoustic self-calibration handoff

Updated 2026-10-04. Work is on `rewrite/stratified-tdoa-calibration`, PR #8.
The user authorized implementing the plan and committing/pushing each milestone.
Merging is not part of this authorization.

## Contract and implementation

The authoritative user contract and milestones are in
[SELF_CALIBRATION_COMPLETION_PLAN.md](SELF_CALIBRATION_COMPLETION_PLAN.md).
[COMPLETION_PROGRESS.md](COMPLETION_PROGRESS.md) records measured outcomes and
unmet gates. Do not infer acceptance from code being implemented or pushed.

Fixed unknown microphones and a moving unknown broadband chirp source are solved
from synchronized direct sound, with known sound speed. Public audio/WAV output
uses a microphone-relative frame and inferred emission-feature times. Same-side
planar input resolves reflection only. Room sources may leave the microphone hull;
room boundaries/floor/ceiling are not supplied to calibration. No echoes are in scope.

Arbitrary-planar mode receives no construction input. Cross/T may declare arm
membership and directed angles; unknown spacings stay free. Grid declares topology
and optional independently equal row/column pitches, both unknown. Exact cross
angles remain non-identifiable without supplied construction evidence, regardless
of the number of moving-source calls. Preserve that ablation.

Implemented M0–M5 include chirp fixtures, angle/grid parameterization, public
same-side handling, microphone frames, convex-room fixtures, and a reference-free
Myotis calibration boundary. M6 adds recorded-window timing uncertainty and
geometry-only selection of source-completion receivers. M7 diagnostic JSON survives
absent geometry, preserves IDs/coverage, and returns non-success exit codes.

## Current evidence and limitations

- The 10-case exact/clean-float PR chirp subset passes. See
  `benchmarks/results/chirp_pr_final.json` and its preserved before-fix report.
- The public Myotis report is deliberately **not accepted**: weak identification,
  41 localized calls from 43 detections, two unresolved IDs, 40 reference-overlap
  calls. It meets the older geometry/timing limits. The supplied 90-degree angle
  has no independently verified physical provenance; acquisition descriptions
  conflict. Never advertise this as verified unique real-recording recovery.
- The broader preregistered matrices and stress outcomes must be read from their
  archived reports. A filtered or incomplete report is not full acceptance.
- Do not claim current Python 3.11–3.14 CI success using the previous September run.
  Local checks and newly triggered remote CI are distinct evidence.

The implementation continuation with concrete milestones/checks is in
[REMAINING_ACCEPTANCE_WORK.md](REMAINING_ACCEPTANCE_WORK.md).

## Reproduction and continuation

Use `uv sync --locked --all-groups`. See `VALIDATION.md` for full commands.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; restricted environments may need
`UV_CACHE_DIR=/tmp/asc-uv-cache`, `XDG_CACHE_HOME=/tmp/asc-cache`, and
`MPLCONFIGDIR=/tmp/asc-matplotlib`.

`benchmarks/run_chirp_matrix.py` supports exact/noisy/float/PCM stages, filters,
parallel worker processes, checkpoints and fingerprint-checked resume. Expected
case IDs and failed cases remain in every report. Development seeds are 10–12;
evaluation seeds 30–34 were reserved before tuning. If evaluation results cause
solver tuning, keep those failures and reserve fresh evaluation seeds.

Diagnose the first failing stage: exact geometry, noisy arrivals, audio measurement,
completion/coverage, or uncertainty. Do not lower acceptance/conditioning thresholds,
change truth or silently drop events. Frozen validation lineage must survive polish.
Timing uncertainty from recorded windows is optional; confidence mode is the legacy
default and legacy audio gates remain mandatory.

Run Ruff format/lint, Ty, the full pytest suite, build and CLI smoke checks before
claiming final readiness. Dedicated Myotis fixtures must fail if absent, not skip.
PR completion remains blocked until required acceptance outcomes are documented;
accuracy with `weakly_identified` is not a solved scene.
