# Stratified TDOA self-calibration rewrite

## Decision and scope

The Bayesian/MAP-centered geometry backend has been replaced by a geometry-first
stratified TDOA solver. Keep event-driven measurement extraction separate from geometry, use an
explicit planar-receiver/3-D-source model, and permit deterministic measurement-only
refinement after a valid geometric hypothesis has been selected.

This document and [the implementation plan](STRATIFIED_IMPLEMENTATION_PLAN.md)
consolidate the opening design commits (`e338eba`, `203ff17`), the subsequent literature
review, and the mathematical review of the PR #7 fixtures. The rewrite described here is
implemented on `rewrite/stratified-tdoa-calibration`; PR #7 remains abandoned, with only
its pinned fixtures/frontend ideas reused. A successful empirical real-Myotis run is not
claimed unless external WAV/reference paths are supplied and recorded by the F-stage
runner.

The first-pass physical model assumes synchronized microphone channels, known speed of
sound, correctly associated discrete emissions, and near-field direct-path range
differences. Microphone positions and event source positions are unknown. Source
emission times cancel in TDOAs; the recovered event range offsets are not arbitrary
per-microphone clock offsets.

Bayesian posterior scoring, trajectory-motion priors, and Laplace uncertainty must not
choose geometric branches. Explicitly supplied physical constraints may be supported in
a separately labeled constrained-calibration mode. No reference microphone coordinates,
source trajectory, cross-arm angle, or source half-space may enter the blind tests.

## What the review changes

The stratified construction remains the primary route because it separates offset
recovery, affine factorization, and metric recovery [R1]. However, a low residual is
not sufficient evidence of correct geometry: distinguish a wrong numerical basin from
weak conditioning and from genuine non-identifiability.

**The exact synthetic Myotis cross is not an identifiable generic planar array.**
Every microphone in the pinned fixture lies on one of two intersecting lines. Centered
in that plane, its coordinates satisfy `x*z = 0`; the six-column quadratic receiver
design has rank five. A continuous deformation changes the angle between the arms while
compensating the sources' projected positions and heights, preserving every range and
therefore every TDOA. The derivation and reproducible numerical specification are in
implementation-plan §2.3. This is our analysis of the fixture, not a reported result
from a paper or the real recording.

**Planar receivers also leave an independent height-sign ambiguity for every event.**
A source at `(u, v, h)` and one at `(u, v, -h)` have identical ranges to the plane.
Without cross-event physical constraints, there are independent sign choices, not just
one global reflection. Report projected positions and unsigned heights, or label any
chosen 3-D representative and declared half-space convention. A known half-space does
not remove the cross-arm deformation above. The mixed-dimensional construction [R3]
and the distance equation motivate this distinction.

Accordingly, success means recovering the observable geometry with honest ambiguity
and conditioning diagnostics. It does not mean forcing every dataset into one unique
signed 3-D scene. Full-rank local Jacobians alone cannot establish global uniqueness.

## Target architecture

```text
multichannel audio
    -> event detection / correspondence / measurement lineage
    -> TDOAs + validity + uncertainty/covariance + timestamp semantics
    -> reference conversion and arrival-gauge handling
    -> stratified offset hypotheses
    -> fit-only offset expansion and corrected ranges
    -> affine factorization and Euclidean metric recovery
    -> fit-only completion of remaining microphones/events
    -> physical validation, degeneracy checks, held-out prediction, consensus
    -> selected geometry or explicit ambiguity / insufficient-data result
    -> optional deterministic measurement-only refinement
```

### 1. Measurement boundary

The geometry solver accepts precomputed measurements and never accesses audio samples.
Retain primitive measurement identities, missing-data masks, reference conventions, and
whether pairs were measured independently or derived from common per-channel arrivals.
Derived redundant pairs do not become independent observations merely by being exported
as additional columns. Use a nonredundant basis with appropriate covariance, or whiten
the supported covariance subspace; never silently multiply evidence by duplicating pairs.

Receiver detection times and emission times are distinct. Preserve the acquisition
channel/sample timestamps and derive emission-time estimates only under a documented
waveform-landmark convention. Reference changes must not silently shift time meaning.
The inherited smooth lag tracker is an optional association heuristic, not proof that
the frontend has no temporal assumptions; test with it disabled.

### 2. Stratified geometry

Implement the linear 9-receiver/5-event 3-D and 7-receiver/4-event 2-D offset cases as
mathematical anchors. A literal zero-reference arrival matrix makes the published
linear system's reference row inconsistent; the implementation must explicitly handle
arrival gauge, conditioning, and offset recovery rather than least-squares fitting
that inconsistency. See implementation-plan §6.2 for the equation and exact test.

For eight-microphone 3-D support, prioritize a generated **7r/6s offset solver**, then
consider 6r/8s as an alternative. R1 reports five versus fourteen algebraic offset
solutions respectively; these are not guaranteed counts of real valid scene geometries.
The smaller root count is an engineering starting point, not a robustness claim.

Metric recovery is a separate mathematical deliverable with explicit unknowns,
constraints, positive-definiteness tests, and noisy-data tolerances. Exploit extra
fit-only events where this simplifies recovery, but do not assume 40 events make every
metric problem linear. Return all validated isolated real roots for supported generic
cases; detect continuous families instead of pretending they are finite root lists.

Implement the receiver-2-D/source-3-D model explicitly using R3. Do not project a generic
3-D answer onto a plane or silently constrain a near-planar real array to be exactly
planar. Use matrix-specific rank and conditioning diagnostics, including the metric
constraints, not only the low-rank factorization.

### 3. Completion, validation, and robustness

A minimal hypothesis does not yet contain every microphone or event. Make expansion
and completion explicit stages, with provenance for measurements used to generate
roots, recover remaining offsets, locate additional states, and validate predictions.
A new event fitted using all its TDOAs is not held out when those same TDOAs are scored.
Reserve receiver measurements within validation events, account for shared measurement
noise, and record the fit/validation masks. Full-data refits are training diagnostics,
not replacements for the frozen validation score.

Use validity checks as eligibility gates. Rank admissible candidates by a specified
normalized robust prediction score, inlier support, and consensus across distinct
subsets; do not add seconds, meters, ranks, and vote counts in an unspecified scalar.
Define minimum support and model-selection tolerances before reference evaluation.
When supported models or non-equivalent branches cannot be distinguished, report that
fact. More subsets cannot eliminate a structural ambiguity.

Heteroscedastic weighting, missing-data handling, and gross-outlier tests belong in the
first usable solver. Retain raw measurements for validation. Sparse/low-rank denoising
is only an optional ablation after its matrix model is justified; arbitrary near-field
raw TDOAs must not be assumed to have the corrected-range matrix's rank.

### 4. Refinement

Refinement starts from a physically valid selected stratified geometry, respects its
Euclidean gauge and dimensional model, and uses measurement residuals only. Preserve
both the unrefined and refined results. Compare weighted least squares with a fixed
robust baseline for contaminated data. No multistart MAP fallback, motion prior, or
reference-based branch choice is permitted. Refinement must not turn an unresolved
continuous family into a claimed unique result by arbitrary numerical regularization.

## Validation strategy

### Random 3-D and WAV regressions

Use **40 broadband pulses as the primary development benchmark and retain the frozen
20-pulse cases as hard regressions**, each for 8, 12, 16, and 24 microphones. Preserve
seeds, geometry, trajectory, recording duration, pulse generation, noise, detection
checks, and the existing limits: microphone RMS `< 0.15 m`, source RMS `< 0.18 m`,
TDOA RMS `< 60 us`. Forty supplements rather than replaces the harder lower-data cases.

The existing random-3-D schedule spans 0.4–4.4 seconds in a five-second recording;
40 pulses are approximately 102.6 ms apart, above the existing 50 ms detection gap.
Check actual detection and association rather than treating this arithmetic as an
end-to-end pass. Add the 8/12/20/30/40-event diagnostic sweep, reporting errors,
conditioning, completion/validation coverage, root survival, ambiguity, and runtime.
The 8/12/30-event points are not new universal success thresholds.

Retain the PCM16 eight-microphone test and its own original geometry thresholds;
remove only out-of-scope posterior-uncertainty assertions. See the implementation plan
for exact fixture provenance and limits.

### Planar and Myotis validation

Separate four cases: nondegenerate planar success tests; the exact cross as a continuous
ambiguity regression; explicitly constrained cross calibration; and real-recording
validation. Retain the cross's original geometry/trajectory as diagnostic data rather
than changing it until an unconstrained solver happens to reproduce the reference.

The synthetic cross originally has 18 events over 0.25–2.05 seconds. Forty pulses in
that interval would be only 46.2 ms apart, below its 50 ms detector gap. Keep the legacy
18-event audio schedule. A 40-event TDOA-only ambiguity check is useful; any separate
40-event audio variant needs an explicit timing/configuration change, not a hidden one.

For constrained cross tests, supply measured geometry information explicitly and record
it in the result; a cross-arm angle or a distance between noncentral microphones on
opposite arms can remove the exhibited deformation. Check remaining observability and
height signs separately. Removing the constraint must restore ambiguity reporting.

The real Myotis recording remains the integration target, not a result already achieved.
Inspect its actual geometry for cross/conic and near-planar conditioning separately.
Reference coordinates are evaluation-only in the blind run. Report microphone-only
rigid alignment without scale fitting, observable planar-source error, signed source
error only under a declared sign convention, time overlap, measurement coverage, and
surviving ambiguity. Do not select branches or tune thresholds using source truth.

## Literature-informed roles

The following are design roles, not claims that these methods have been reproduced on
this repository. R1–R10 resolve to the implementation plan's reference map.

| Approach | Role in this rewrite | Applicability limit |
| --- | --- | --- |
| Kuang/Åström stratification and TOA metric recovery [R1], [R2] | Primary geometry stages and exact anchors. | Generic conditions and metric validity still need testing. |
| Ask/Kuang/Åström mixed dimensions [R3] | Explicit planar/3-D recovery. | Does not supply missing source signs or a cross-arm angle. |
| Vanwynsberghe et al. [R4] | Sparse-outlier tests; optional justified denoising ablation. | Not a guarantee for this blind near-field pipeline. |
| Woźniak/Kowalczyk [R5] | Heteroscedastic weighting and outlier diagnostics. | Distributed-array/DOA assumptions differ. |
| Wang/Chen/Yin constrained TLS [R6] | Optional comparison under reproduced assumptions. | Reference-node/source-estimation structure is not free information. |
| Zhao/Wang/Xu geometric calibration [R7] | Conditioning/confidence analysis ideas. | DOA-based array-node calibration, not a demonstrated blind scalar-microphone TDOA replacement. |
| Scalar/vector sensor self-calibration [R8] | Identifiability and relaxation methodology. | Gain/phase and DOA estimation, not unknown microphone positions. |
| Anchorless far-field direction estimation [R9] | Optional regime-specific comparison. | Bearings do not determine near-field source ranges. |
| Low-rank clock synchronization [R10] | Deferred literature lead only. | Different timing unknowns; verify before implementation use. |

The supplied IEEE document `10545598` remains unidentified from accessible trustworthy
metadata and is not design evidence. Obtain its title/DOI/full text before assigning a
role. No first-pass milestone depends on it or on reproducing the optional baselines.

## Repository policy and sequencing

Preserve orthogonal infrastructure only after auditing its assumptions. In particular,
replace first-three-microphone gauge construction: the cross fixture begins with
collinear microphones. Keep original microphone IDs through pivot/permutation changes,
retain microphone-only evaluation and source-time-overlap behavior, and adapt exports
and CLI results to partial/ambiguous outcomes.

Begin with **Milestone 0: identifiability, conventions, and algebraic feasibility**.
Then implement the measurement contracts, exact linear anchors and metric equations,
7r/6s solver plus completion/validation, planar recovery, audio integration, real Myotis,
and cleanup. Do not delete the old backend until its replacement meets the applicable
gates. Do not preserve it as an automatic fallback that conceals an algebraic failure.

Deferred work remains Bayesian uncertainty after geometry selection, motion-based sign
resolution, asynchronous clocks/drift, sound-speed estimation with metric anchors, and
full reverberation/direct-path generative models. Basic robust measurement handling and
honest degeneracy diagnostics are not deferred.

## Grounding references

[R1]: STRATIFIED_IMPLEMENTATION_PLAN.md#r1
[R2]: STRATIFIED_IMPLEMENTATION_PLAN.md#r2
[R3]: STRATIFIED_IMPLEMENTATION_PLAN.md#r3
[R4]: STRATIFIED_IMPLEMENTATION_PLAN.md#r4
[R5]: STRATIFIED_IMPLEMENTATION_PLAN.md#r5
[R6]: STRATIFIED_IMPLEMENTATION_PLAN.md#r6
[R7]: STRATIFIED_IMPLEMENTATION_PLAN.md#r7
[R8]: STRATIFIED_IMPLEMENTATION_PLAN.md#r8
[R9]: STRATIFIED_IMPLEMENTATION_PLAN.md#r9
[R10]: STRATIFIED_IMPLEMENTATION_PLAN.md#r10

See the [reference map](STRATIFIED_IMPLEMENTATION_PLAN.md#18-reference-map) for paper
links, evidence limits, and pinned repository provenance. Numerical review checks are
specified in implementation-plan §§2.3 and 6.2; they are not production-solver results.
