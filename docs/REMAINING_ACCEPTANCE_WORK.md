# Remaining acceptance work

The implementation is pushed in coherent milestones, but the user's complete
recovery goal is not yet accepted. This document extends the authoritative
[completion plan](SELF_CALIBRATION_COMPLETION_PLAN.md); it does not replace or
weaken its limits. Results/provenance remain in
[COMPLETION_PROGRESS.md](COMPLETION_PROGRESS.md).

## Current failure boundary

All 552 standard exact development cases recovered geometry within 1 mm. The
full 732-row noisy-arrival development run has 27 unsuccessful standard cases.
Rechecking those cases with the coverage correction yields 16 weak and 11 failed
results, with no falsely complete `solved` result among those failures. Successful
exact cases are not evidence that noisy initialization/completion is reliable.

The 96-case initial float/PCM development subset passed 94 cases; a continuous
seed-admission tolerance fixed both 12-channel/40 dB room failures in subsequent
rechecks, preserving the original report. The full audio/evaluation matrices are
not demonstrated. The Myotis fixture remains weakly identified despite meeting
older geometry/timing limits, and its physical construction provenance is unverified.

## R0 — Reproduce and isolate every failing stage

Use the preserved full noisy report and `--failed-from` to reproduce all 27 cases
at the recorded options. Do not select only favorable seeds or increase search
budgets first. Record the first failure among offset recovery, corrected-range
expansion, metric feasibility, receiver completion, source completion and final
uncertainty. Include covariance, generating/completion/validation coordinate IDs,
root counts, rejected roots, source rank and fit errors. Work in
`stratified/solver.py`, `offsets_linear.py`, `expansion.py`, `planar.py` and the
benchmark runner. Add stage-specific rejection reasons without replacing unknown
scores with zero.

Check: 27 expected IDs produced once; reference geometry remains evaluator-only;
each unsuccessful case has a first-stage diagnosis. Preserve original reports.

## R1 — Use construction information during noisy initialization

The current cross/T/grid reduced parameterization constrains final fitting, but
the initial stratified metric/completion path mostly receives an angle constraint.
Carry declared arm membership/grid topology into that initial path. Remap IDs
through reference/core/subset permutations; do not require a junction microphone.
Use reduced free arm coordinates or ordered grid gaps and independent unknown
pitches. Keep positive directed-ray lengths and source heights. No numeric gap,
aperture, true coordinate, room boundary or source position may initialize fitting.

For noisy near-conic designs, evaluate covariance-weighted constrained metric
candidates rather than treating noisy numerical rank as physical identifiability.
Enumerate feasible non-right-angle metric branches and retain competing scenes
when independent evidence cannot select one. Keep fitting/validation coordinate
lineage and the existing physical/covariance gates. A declared angle must not skip
local uncertainty or justify post-fit coordinate snapping.

Checks: all exact development cases retain 1 mm / 0.01 us accuracy; unknown-angle
crosses and two-row/column grids remain degenerate; directed 60/75/90-degree cross/T
and unequal/equal grid examples pass. Re-run all 27 failures and then the full
noisy matrix. Do not force rare statistically weak fits to `solved` by loosening
thresholds; unresolved statistical acceptance must remain explicit.

## R2 — Complete sources without hiding missing events

Audit the failed/weak source events using recovered geometry and recorded timing.
Receiver selection must use geometry/coverage, with enough independent baselines,
and must preserve an independent scoring set. Use the shared covariance-aware
objective for weighted per-event localization after a valid geometry exists.
Keep finite same-side sources at positive normal height; retain NaN/unknown for
unresolved events. If a later fit completes an event, record which coordinates
were consumed and do not rewrite the frozen pre-fit held-out evidence.

Checks: no raw/public result remains `solved` with unresolved source events;
coverage denominators retain detected/rejected IDs; all standard cases meet full
coverage and 5/10 cm RMS. Channel relabeling/reordering must preserve emission
timestamps, including IDs outside 0…N−1. Frame selection must use stable IDs,
not the timing reference or a reference-source sign.

## R3 — Establish defensible Myotis recovery

First obtain independent acquisition/frame/channel evidence if it exists. The
fixture's current 90-degree rays (3→0, 3→11) are a supplied declaration with
unverified physical provenance. Do not infer that evidence from reference
positions. Keep calibration and angle ablation frozen before reference loading.

Investigate broadband-call timing bias from recorded windows: duration/bandwidth
variation, intra-call Doppler deformation, segment disagreement and residual
correlation. Compare shorter/multiple-window estimators on independent chirp-delay
fixtures before changing the real-data path. Do not inflate uncertainties merely
to make the goodness-of-fit label pass. Any source motion used for association
must remain an explicit frontend option, never a geometry prior.

Checks: dedicated fixtures fail if absent; arbitrary-planar ablation does not
claim unique recovery; declared-angle recovery requires `solved`, 12 microphones,
complete detected-event localization, at least 40 reference-overlap calls, RMS
<0.20/0.30 m and fitted/frozen-held-out TDOA RMS <45 us. Record all rejected IDs,
feature-time semantics and construction assumptions. If the recording cannot
support that outcome, explain the unsupported output instead of manufacturing it.

## R4 — Finish the preregistered matrices and PR readiness

After development acceptance, expand float/PCM runs to every manifest dimension:
8/12/16/24 channels plus 9-channel grids, 20/40 calls, both planar spans,
all standard configurations, clean/40 dB and seeds 10–12. Keep 20/10 dB and planted
channel/motion failures in separately labeled stress characterization. No echoes.

Evaluation seeds 30–34 remain unused and reserved. Run the full evaluation at a
frozen source/manifest/dependency fingerprint. If results cause further tuning,
preserve those outcomes and preregister new evaluation seeds. Expected case IDs
must cover every required row; filtered reports are explicitly partial.

Run Ruff format/lint, Ty, package build, CLI checks and full Python 3.11–3.14 CI.
The existing 15/18 cm rendered-audio and 18/22 cm PCM gates remain unchanged.
Update plots, progress, handoff and PR #8 with the actual final results. Commit
and push each checked milestone as authorized. Do not merge or claim the PR is
fully ready while strengthened Myotis or full acceptance remains unmet.
