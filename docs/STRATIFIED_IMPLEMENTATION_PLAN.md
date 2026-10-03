# Detailed implementation plan: stratified TDOA rewrite

Companion to [STRATIFIED_REWRITE.md](STRATIFIED_REWRITE.md). This consolidated
specification incorporates the two opening commits, the literature review, and the
subsequent mathematical/implementation review. It replaces conflicting earlier wording
in place: 40 events supplement the frozen 20-event regressions; the unconstrained exact
Myotis cross is an ambiguity test, not a unique-geometry success test.

This document now serves as the implementation record for the completed rewrite.
Milestones A-G are implemented on `rewrite/stratified-tdoa-calibration`. Mathematical
counterexamples remain distinct from production validation, and no successful empirical
real-Myotis calibration is claimed without externally supplied WAV/reference data.

## 1. Non-negotiable design constraints

1. Use stratified offset recovery, affine factorization, and metric recovery as the
   primary geometry path. No automatic Bayesian/MAP or multistart fallback.
2. Exclude posterior scores, source-motion priors, and source truth from branch choice.
   Statistical measurement weighting and robust residuals remain allowed.
3. Assume synchronized microphone channels and known positive speed of sound. Event
   range offsets are distinct from unknown microphone clock offsets/drifts.
4. Treat general 3-D and receiver-2-D/source-3-D as distinct dimensional models.
5. Preserve random-3D 20-event acceptance tests and add 40-event tests at the same
   thresholds. Keep the real recording separate from synthetic Myotis-shaped data.
6. Report independent planar height signs, continuous metric families, insufficient
   information, and weak conditioning instead of forcing a unique scene.
7. Track primitive measurement lineage and separate generation, completion, and
   validation data. Derived pairs are not independent observations.
8. Permit optional measurement-only refinement only from valid selected geometry;
   preserve the pre-refinement solution and frozen validation score.
9. Include heteroscedastic errors, missing measurements, and gross-outlier handling in
   the first usable solver, without asserting that robust losses cure ambiguity.
10. Any arm angle, microphone distance, half-space, or other physical constraint must be
    explicit input in a labeled constrained run. Blind geometry tests receive none.

## 2. Test specification comes first

### 2.1 Freeze 20 pulses and add the 40-pulse benchmark

Port fixtures from PR #7 at commit `cd02c4990234dd9ef877cb1af149fb4df64574d5`
([F1](#f1)), not an unpinned moving branch. Parameterize `_pulsed_scene` by event count
without changing its 20-event behavior:

```text
microphone counts: 8, 12, 16, 24
hard gates:        20 AND 40 broadband pulses
primary benchmark: 40 pulses
array / source:    existing random 3-D array / moving 3-D trajectory
seed:              100 + microphone_count
audio:             48 kHz, 5.0 s, existing cardioid renderer and noise
schedule:          linspace(0.4, 4.4, event_count)
```

Preserve the geometry, trajectory keys, RNG behavior, pulse waveform/duration, noise,
measurement-quality assertions, and existing frontend settings. Adapt result accessors
and remove only obsolete Bayesian parameters. Both counts must retain:

```text
microphone RMS < 0.15 m
source RMS     < 0.18 m
TDOA RMS       < 60 us
all scheduled events detected and used in the clean benchmark
finite, positive timing uncertainty on valid observations
```

With 40 pulses, spacing is `4.0 / 39 = 0.102564... s`, above the existing 50 ms detector
gap; keep the 20 ms max-lag and 2 ms template settings. Assert actual event correspondence
and extracted TDOA quality, not just the count. Do not pass true pulse times to the
frontend or geometry solver.

The 8/12/20/30/40-event sweep is an additional sample-efficiency diagnostic. Only 20 and
40 inherit the hard audio geometry gates. Record numerical errors, matrix conditioning,
independent measurement count, completion coverage, held-out support, surviving branches,
and runtime. Some configurations may legitimately be unsupported or ambiguous. Keep
geometry seeds fixed; distinguish non-nested rendered schedules from a separate
nested-subsampling TDOA experiment. More events along a weak trajectory do not guarantee
identifiability or independence.

For eight generic 3-D microphones, `(M-1)E` independent arrival-difference coordinates
versus `3M+3E-6` geometry unknowns gives 140/78 at 20 events and 280/138 at 40. This is
our degree-of-freedom count after rigid gauge removal, not a sufficient uniqueness test.

Retain the PCM16 eight-microphone WAV regression at 20 events and add a 40-event variant.
Its original thresholds are microphone RMS `< 0.18 m` and source RMS `< 0.22 m`.
Do not accidentally replace these with the non-WAV limits. Remove/quarantine posterior
uncertainty assertions only; retain I/O, event, and geometry checks.

### 2.2 Separate planar success, cross ambiguity, constrained cross, and real data

Create distinct fixtures with distinct contracts:

- **Nondegenerate planar arrays:** exact and noisy corrected-TOA/TDOA scenes with 3-D
  source motion; require metric recovery and observable source coordinates. Check
  unsigned height separately from signed trajectory error.
- **Exact Myotis cross:** retain the original 12 coordinates, trajectory, seed 812,
  and 18-event audio schedule. Require detection and measurement accuracy, and require
  an explicit continuous-ambiguity diagnostic from the unconstrained geometry solver.
  Do not demand that TDOAs uniquely identify the fixture's right angle.
- **Explicitly constrained cross:** supply an arm angle or cross-arm distance through
  the public constraint contract. Retain the old cross limits (`0.20 m` microphone,
  `0.30 m` source, `45 us` TDOA RMS) for the applicable observable/source-sign metric.
  Signed source RMS requires a declared half-space/orientation convention. Removing
  the metric constraint must restore ambiguity reporting; check remaining freedoms.
- **Real Myotis:** a separate integration run with actual audio and reference data.
  Neither the synthetic cross nor a constrained run establishes a blind real-data pass.

The legacy cross schedule is `linspace(0.25, 2.05, 18)` within 2.4 seconds. Forty pulses
in the same interval are only `1.8 / 39 = 0.046154... s` apart, below its 50 ms detector
gap. Keep the original schedule. Use 40 exact TDOA events for the ambiguity experiment;
a 40-event audio variant must explicitly change/validate duration or detector settings.

Add near-cross and slightly nonplanar sweeps. Report how uncertainty/conditioning and
model support change; do not project them to a plane merely because an initializer
looks planar. Reference geometry may classify benchmark difficulty offline, but the
blind solver must derive its diagnostics from measurements and candidate geometry.

### 2.3 Exact cross counterexample and reproducibility gate

The following is our derivation for F1, separate from the generic results in R3.
Use column vectors in the receiver plane, centered at `(x,z)=(1.2,0)`. Each receiver
`r_i=(x_i,z_i)` satisfies `x_i*z_i=0`. The matrix with rows

```text
[x_i^2, x_i*z_i, z_i^2, x_i, z_i, 1]
```

has rank 5, not 6. For `|alpha| < 1`, choose `A` with

```text
A.T @ A = [[1, alpha], [alpha, 1]]
r_i' = A @ r_i
u_j' = solve(A.T, u_j)
h_j'^2 = ||u_j||^2 + h_j^2 - ||u_j'||^2
```

Here `u_j` is the source projection and `h_j` its normal height. Whenever all new
squared heights are positive, choose either sign of their square roots. Receiver norms
are preserved because `x_i*z_i=0`, receiver/source inner products are preserved by the
inverse-transpose transform, and total source norms are preserved by the height update.
Thus `||r_i'-u_j'||^2+h_j'^2 = ||r_i-u_j||^2+h_j^2` for every pair. This is a continuous
non-rigid ambiguity even with exact ranges; TDOAs and extra events cannot remove it.

Reproduce using F1's `_myotis_microphones`, trajectory interpolation at the 18-event
schedule and at `linspace(0.25, 2.05, 40)`, and `alpha=0.6`. Standalone NumPy checks
produced the following rounded values; use tolerance assertions, not bitwise output:

| Quantity | 18 events | 40 events |
| --- | ---: | ---: |
| Maximum range difference | 4.4e-16 m | 4.4e-16 m |
| Receiver RMS after receiver-only rigid alignment | 0.245775 m | 0.245775 m |
| Source RMS with the more favorable global normal orientation | 0.371004 m | 0.367230 m |
| Minimum transformed squared source height | 0.409881 m^2 | 0.409881 m^2 |

Required exact checks: quadratic rank 5, range difference `< 1e-12 m`, and inequivalent
receiver geometry after rigid alignment. Evaluating both normal orientations using
truth is only a counterexample diagnostic showing that global reflection does not
explain the error. It is not a permitted calibration branch-selection procedure.
Port these mathematical checks into tests during Milestone 0; no test may depend on a
sandbox file from the review conversation.

A supplied distance between noncentral microphones on opposite arms, or the arm angle,
can fix this exhibited one-parameter freedom. A source half-space alone cannot. Never
infer that all remaining ambiguities are gone without testing the constrained system.

### 2.4 Real-Myotis harness and evaluation

Provide a documented integration entry point accepting audio/reference paths or
environment variables. Record input checksums, channel mapping, sample rate, timestamp
conventions, speed of sound, all supplied constraints, solver configuration, seed, and
software revision. If data are unavailable, report `not_run`; do not substitute a
synthetic run or fabricate metrics.

For the blind run, reference geometry is evaluation-only. Inspect real receiver
planarity/conic conditioning in a separate analysis, and compare with the solver's
own diagnostics. Report microphone-only rigid alignment with no scale fitting,
projected-source/unsigned-height error for planar results, signed source RMS only with
a declared sign convention, residual distributions in seconds, event/measurement
coverage, constraint provenance, selected/competing models, and ambiguity reasons.

Preserve evaluation's existing source-time-overlap behavior [F3]. Do not extrapolate
beyond reference coverage, use all reference sources to choose the plane normal, or
hide events lost in extraction, completion, or overlap filtering.

### 2.5 Literature roles and evidence limits

R1/R2 remain the primary geometry references; R3 provides the mixed-dimensional model.
R4 motivates sparse-outlier experiments; R5 motivates heteroscedastic weighting.
These design adaptations are not reproduced performance claims. Optional denoising
must justify the matrix rank/model it uses and always score against original data.

R6 is a constrained-TLS comparison only under its reference-node/source-estimation
assumptions. R7, supplied as ScienceDirect `S0003682X24002597`, is DOA-based array-node
calibration, not an established replacement for blind near-field scalar-microphone
TDOAs; confirm source/anchor assumptions from full text before reproduction. R8,
supplied as IEEE `9918024`, concerns gain/phase calibration jointly with DOA, not unknown
microphone coordinates; reuse its identifiability/relaxation methodology, not its
measurement model. R9 is a far-field comparison, not a source-range estimator here.

R10 is retained only as a deferred synchronization literature lead; its metadata and
technical assumptions must be re-verified before implementation use. The supplied IEEE
`10545598` remains unidentified from accessible reliable metadata. Neither item creates
first-pass requirements. Do not present abstract-level screening as full-text method
verification or claim any alternative resolves the exact cross ambiguity without an
additional measurement/constraint.

## 3. Proposed package layout and reuse audit

```text
src/acoustic_self_calibration/
    events.py                 detection, correspondence, optional lag tracking
    tdoa.py                   delays, primitive measurement and pair-graph utilities
    measurements.py           validated immutable measurement/covariance contracts
    stratified/
        notation.py           signs, reference transforms, units, arrival gauge
        offsets_linear.py     conditioned linear offset cases
        offsets_minimal.py    generated 7r/6s solver; optional alternatives
        expansion.py          fit-only offset/state completion and provenance
        factorization.py      corrected distances -> affine factors
        metric_upgrade.py     explicit 3-D metric constraints and real branches
        planar.py             mixed-dimensional metric/unsigned-height recovery
        constraints.py        explicit physical inputs; no hidden fixture knowledge
        hypotheses.py         subsets, held-out predictions, consensus, model choice
        robustness.py         covariance-aware residuals, outlier diagnostics
        identifiability.py    matrix/Jacobian ranks, weak modes, ambiguity
        refinement.py         optional fixed-model geometric least squares
        solver.py             orchestration, budgets, structured failure results
    pipeline.py               audio -> measurements -> geometry
    wav.py                    WAV loading and wrapper
    geometry.py               stable gauges, alignment and metrics
    simulation.py             existing renderer plus explicit fixture variants
    evaluation.py             reference comparison and observable metrics
    ground_truth.py           scene schema and timestamp conventions
    visualization.py          plots including ambiguous/partial outcomes
    export.py                 scene/diagnostic JSON and existing useful outputs
    cli.py                    public commands
```

Preserve WAV I/O, scene JSON, plotting, and simulation only after auditing assumptions.
`geometry.canonicalize_scene` currently uses the first three microphones; F1's cross
starts with collinear microphones [F2]. Replace this with conditioning-aware pivot
selection or a rank-revealing basis, deterministic tie-breaking, and preservation of
original channel IDs. Test degenerate triplets and permutations. A basis/reflection
choice is gauge fixing, not measured handedness.

Retain microphone-only alignment and time-overlap evaluation, but extend them for
unsigned planar height and partial results. Do not keep old Bayesian internals solely
for compatibility. Delete them only after the replacement gates pass, without leaving
an automatic fallback that masks failure.

## 4. Data contracts

### 4.1 `EventTDOAMeasurements`

The frozen boundary object must validate and own read-only copies of its arrays;
`dataclass(frozen=True)` alone does not make NumPy array contents immutable.

```text
event_ids                         stable IDs, shape (E,)
event_samples                     (E,) integers or absent for TDOA-only inputs
sample_rate_hz                    positive when samples are present
receiver_event_times_s            (E,), observed landmark times, not emission times
event_channel                     acquisition reference channel, or absent
microphone_ids / microphone_pairs original IDs and oriented (a,b) pairs
tdoa_s / sigma_s / confidence      (E,P)
valid                             (E,P) boolean
measurement_origin                independent_pairs | derived_arrivals
primitive_ids / pair_from_primitive mapping used for weighting and split provenance
covariance_model                  stated exact/approximate model and parameters
arrival_representatives_s          optional (E,M) reference-gauged values
arrival_valid                     optional (E,M) boolean
```

Valid entries must be finite; `sigma_s > 0`. Invalid entries must never become zeros in
a fit. Validate dimensions, IDs, pair orientation, covariance symmetry/PSD, and graph
connectivity per event. Return component/insufficient-data diagnostics for disconnected
measurement graphs, not fabricated ranges from zero-filled matrices. Algebraic seeds
require complete connected subproblems; any completion/denoising is labeled and must
not masquerade as observed data.

For arrival-derived pairs, with incidence matrix `B`, use `tau=B*a` and covariance
`Sigma_tau=B*Sigma_a*B.T`. Keep primitive arrival/reference correlations, including
shared reference noise. Either use a full-row-rank difference basis with its covariance
or whiten only the supported covariance subspace. For independently extracted pair
measurements, use their justified covariance model; a diagonal approximation must be
explicit and sensitivity-tested. `sigma_s` remains a useful marginal diagnostic, not
permission to treat all redundant edges as independent.

Tests must cover duplicated edges, orientation reversal, reference changes, missing
pairs, disconnected events, confidence extremes, and different pair-graph choices.
Changing an algebraic representation must not multiply statistical information.

### 4.2 `StratifiedHypothesis` and completion provenance

Record model, seed microphone/event IDs, offset root ID, metric branch ID, arrival-gauge
shifts, offset scope, affine/metric diagnostics, reconstructed coordinates, completion
status for each state, and generating-equation residuals. State arrays cannot imply
full coverage before expansion has succeeded.

Store generation/completion/validation masks at the primitive-measurement level, plus
the split ID, random seed, thresholds, rejected-root reasons, and equivalence-class ID.
Store raw and whitened residual summaries separately; do not combine meters and seconds
inside an undocumented `validation_score`. Failed or unscored stages use null values,
not convincing-looking zero residuals.

### 4.3 `CalibrationResult`, ambiguity, and constraints

The result contains recovered microphone coordinates, observable event source
coordinates, original event IDs/acquisition times, optional estimated emission times,
model, constraints, coverage, diagnostics, and refinement status. Use explicit outcomes
such as `solved`, `ambiguous`, `weakly_identified`, `degenerate`, `insufficient_data`, and
`failed`, with machine-readable reasons. `solved` is only claimed at the declared
observable equivalence level, never as unique signed 3-D recovery by default.

For planar results, export projected source coordinates, unsigned heights, and a
per-event sign-known mask. A conventional positive-height 3-D representative may be
exported for visualization only if marked as such. Distinguish independent height-sign
ambiguity from a global rigid reflection and from continuous array deformation.

Represent a continuous family's dimension/null directions when established; otherwise
report suspected rank deficiency without inventing a finite count or a proof of global
uniqueness. Do not enumerate `2^E` height-sign combinations just to communicate a sign
mask. Posterior uncertainty is optional/deferred; these diagnostics are mandatory.

Physical constraints carry type, microphone/event IDs, value, units, uncertainty or
exact status, and provenance. They may not be populated from hidden fixture/reference
metadata. Removing a supplied constraint must not leave cached constrained geometry.

## 5. Measurement notation, reference, and time conventions

Use receivers `r_i`, sources `s_j`, and range `d_ij=||r_i-s_j||`. Public pair `(a,b)`
means `tdoa[j,(a,b)]=t_bj-t_aj`. With reference receiver 0, define meter-valued arrivals
and offsets by

```text
f_ij = c * (t_ij - t_0j)
f_0j = 0
d_ij = f_ij - o_j
o_j  = -d_0j
```

Offsets can be negative. Positivity applies to corrected ranges, not to offsets.
A known per-event gauge shift preserves geometry:

```text
f'_ij = f_ij + q_j
o'_j  = o_j  + q_j
f'_ij - o'_j = f_ij - o_j
```

Apply this consistently to initializers, offset expansion, diagnostics, and return to
the original reference. Changing the reference microphone, changing arrival gauge,
normalizing physical units, and choosing a coordinate gauge are distinct operations.

For receiver peak time `t_receive`, `t_emit=t_receive-d_reference/c` is valid only when
both refer to the same emitted waveform landmark. Document detector/pulse-landmark
bias; preserve the raw time and any correction separately. A true pulse onset is not
automatically an observed envelope peak. Never fix timing by shifting a trajectory to
minimize error against reference source positions. Test emission/receive conversions
and preserve original acquisition times through algebraic reference changes.

Required exact tests: TOA -> TDOA -> corrected-range round trip; reversed pairs;
reference/gauge transformations; microphone/event permutations; seconds/meters scaling;
and consistency between sample indices, receiver times, and declared emission times.

## 6. Offset recovery and generated-solver feasibility

### 6.1 Corrected squared-range matrices

Use R1/R3 for the rank construction; implement and unit-test the following conventions
against direct synthetic distances rather than a generic low-rank optimization:

```text
G_ij(o) = f_ij^2 - 2*f_ij*o_j
C_m = [-ones(m-1); I_(m-1)]       # m x (m-1)
C_n = [-ones(n-1); I_(n-1)]       # n x (n-1)
K(o) = C_m.T @ G(o) @ C_n
Q = -0.5 * C_m.T @ (d**2) @ C_n
```

The event-only `o_j^2` terms vanish on receiver differencing. `Q_ij` equals
`(r_(i+1)-r_0).T @ (s_(j+1)-s_0)`. Its expected generic rank is 3 for the full 3-D model
and 2 for planar receivers with 3-D sources. Record uncompact, single-difference, and
double-difference ranks by matrix name; never report one unexplained `effective_rank`.
Low rank is necessary but does not by itself establish Euclidean feasibility.

### 6.2 Linear anchors and the zero-reference trap

Start with 9r/5s for K=3 and 7r/4s for K=2 [R1]. Test the all-2-D anchor and the
mixed-dimensional planar use separately; passing one does not establish the other.
For `m=2K+3`, `n=K+2`, the linear construction uses

```text
A[:,j-1] = f[:,j]^2 - f[:,0]^2      (j=1,...,n-1)
B[:,j-1] = -2*f[:,j]
b         = 2*f[:,0]
[A B b] @ u = ones(m)
o_0 = u[-1] / sum(u[:n-1])
o_j = u[(n-1)+(j-1)] / u[j-1]
```

If the frontend's zero-reference `f` is inserted literally, the reference row of the
left matrix is all zero while the right side is one. Do not fit this `0=1` inconsistency
with least squares and call the offsets recovered. Use the arrival-gauge identity in
§5 to construct a numerically suitable equivalent system, or derive and test an
explicit alternative homogeneous formulation.

Production re-gauging must be deterministic, measurement-scaled, independent of source
truth and validation data, and bounded in attempts. Check singular values, denominator
magnitudes, coefficient scaling, and reconstructed original-range equations. A fixed
shift vector that works for one example is not a general recipe; reject/retry a bad
gauge without confusing numerical gauge failure with physical non-identifiability.

Exact reproduction: `rng=default_rng(319)`, then 9x3 normal receivers followed by 5x3
normal sources; form exact distances and subtract receiver 0's row. The literal linear
matrix has rank 8/9. Per-event shifts `[0.7,-1.1,0.4,1.6,-0.8]` yield rank 9 and shifted
offsets within `3.9e-14 m` of `-d[0,:]+q` in the standalone check. Require `< 1e-10 m`
for this specific test and corrected-range RMS `< 1e-8 m` for well-conditioned
meter-scale exact anchors. Add many seeds, scale changes, reference permutations,
small-denominator cases, and noisy inputs; do not demand exact recovery under noise.

### 6.3 Prioritize 7r/6s, with an early generation gate

Implement 7r/6s first for the eight-microphone full-3-D requirement; retain 6r/8s as an
optional alternative. R1's generic algebraic offset counts are 5 and 14, respectively,
not counts of positive-range Euclidean solutions. Do not call 7r/6s the globally minimal
configuration of every self-calibration formulation.

Generate the degree-four determinant constraints from the compacted rank-3 matrix.
Before committing to a production template, specify polynomial variables, monomial
ordering, independent constraints, elimination/action matrix construction, saturation
or extraneous-root handling, gauges/scaling, and all coefficient extraction steps.
Pin generator dependencies, seed, command, license/provenance, and template hash under
`tools/`; generated runtime evaluation remains NumPy/SciPy-only.

Return all finite isolated real offset roots supported by the template, using documented
scale-aware imaginary/duplicate tolerances. Verify each against equations not used in
generation and against original ranges after metric recovery. Account for multiplicity,
near-collisions, roots at infinity, and rank-deficient/positive-dimensional cases.
Distinguish a rejected complex root, a failed metric branch, and an unresolved family.

For noisy overdetermined data, generate from supported subsets and validate with
residual tolerances; there need not be an exact common real zero of every equation.
Independent exact tests across seeds/scales/references and a noise sweep are required.
Failure of solver generation or unstable metric recovery triggers design review, not
an undisclosed optimization fallback. A numerical root count alone is not certification
that all physical solutions have been found.

## 7. Factorization and hypothesis expansion

### 7.1 Corrected ranges to affine factors

For every eligible offset branch, reconstruct ranges only for its covered measurements.
Reject nonfinite or materially negative ranges; define contact/zero-range tolerances
explicitly instead of blindly rejecting numerical roundoff. Construct `Q` as in §6.1
and obtain a rank-K factorization `Q=X @ Y.T` using SVD, with square-root singular-value
splitting as the default normalization. Append zero rows for receiver/source 0.
Keep singular values, discarded residual, dimensions, and factor transforms.

Do not perform ordinary MDS on a fabricated complete receiver/source distance matrix.
Only cross distances are observed. Missing entries cannot be treated as measured zeros.
Under noise, affine factors are an approximation whose metric reconstruction must still
be tested against original observations.

### 7.2 Explicit expansion before full-data scoring

`expansion.py` consumes a seed branch, fitting-only measurements, and split/provenance
metadata. It returns covered offsets/states, unresolved branches, per-state masks, and
failure reasons. A seed solution must not populate a full-data score before this stage.

Two permitted routes, each with its own rank/coverage tests:

1. Extend offsets of additional fitting events using the established compacted column
   space. A new column is affine in its event offset; solve the small membership system
   on observed generating microphones. Reject ill-conditioned or inconsistent extensions.
   Factor/upgrade the enlarged corrected-range subproblem. Extra fit-only data can
   simplify metric recovery, but do not presume it removes every nonlinear unknown.
2. Upgrade the seed first, then localize extra fitting events and microphones against
   the recovered geometry. Use deterministic range/TDOA localization with explicit
   real-branch and rank handling, not free joint multistart geometry optimization.

For known receivers and a new event, a useful independently derived localization
system uses `q=d_0`, `f_i=d_i-d_0`, and unknown source `s`:

```text
2*(r_i-r_0).T@s + 2*f_i*q = ||r_i||^2 - ||r_0||^2 - f_i^2
q^2 = ||s-r_0||^2, q >= 0
```

Solve/validate the linear-plus-norm system on fitting receivers, preserving multiple
solutions or degeneracy where applicable. Added receivers can be trilaterated from
covered sources and corrected ranges; check source-span degeneracy and branch support.
Planar source localization returns projected positions and unsigned heights. Completion
may polish an individual localized state against its fitting measurements, but must not
use withheld values or silently become a joint posterior/multistart fallback.

Test expansion to 8/12/16/24 microphones and 20/40 events, varying order, sparse coverage,
missing receivers/events, and reference. Preserve original IDs and quantify unresolved
states. Geometry spread for sampling must come from measurements or candidate estimates,
never reference coordinates. Only covered, honestly validated states count as support.

## 8. Explicit affine-to-Euclidean metric upgrade

Use R2 as the TOA recovery reference. The following fixes one implementable convention;
validate its derivation independently on exact coordinates and finite differences.

With `Q=X@Y.T`, column vectors `x_i,y_j`, and `x_0=y_0=0`, set `r_0=0` and write

```text
r_i = L @ x_i
s_j = solve(L.T, b + y_j)
H = L.T @ L > 0
```

The three-dimensional unknowns are symmetric `H` (six entries) and `b` (three entries).
Receiver equations are linear in them:

```text
d_i0^2 - d_00^2 = x_i.T @ H @ x_i - 2*x_i.T @ b
```

Source equations, including source 0, supply the remaining constraints:

```text
d_0j^2 = (b+y_j).T @ inv(H) @ (b+y_j)
```

Use the linear receiver design to eliminate known combinations, then specify the
remaining polynomial system for each supported configuration. For generic designs,
9r/5s leaves one linear free parameter and 7r/6s leaves three before source
constraints, assuming full row rank. Verify actual rank rather than assuming it from counts. Clearing an
inverse with determinant/adjugate introduces singular-H candidates; reject them using
the original rational/range equations and positive-definiteness, not clipping eigenvalues.

For each backend, document unknown/free-variable counts, linear pivot strategy,
remaining polynomial degrees, template or univariate solver, extraneous-root checks,
and exact/noisy acceptance tolerances. Forty events may improve constraints without
making this stage automatically linear. General 3-D, all-2-D, and mixed-dimensional
systems need separate specifications.

Recover a suitable `L` by Cholesky and reconstruct both point sets. Canonicalize only
after metric recovery. Validate the full observed corrected distances as well as
original TDOAs. Enumerate valid isolated real metric branches for generic cases; return
rank-deficient/continuous ambiguity when the constraints support a family.

Required tests include exact/noisy ranges, multiple roots, wrong positive-range offset
branches, near-singular H, coordinate scale changes, permutations, mirror equivalence,
and direct reconstruction/Jacobian checks. Keep metric-fit error distinct from the
pre-upgrade algebraic rank score.

## 9. Explicit planar receiver / 3-D source recovery

Set `Dr=2`, `Ds=3`, and compacted rank `Dmin=2` [R3]. Factor `Q` at rank 2, not rank 3.
With the same notation as §8 but two-dimensional `x,y,b` and 2x2 H, use

```text
r_i = (L@x_i, 0)
u_j = solve(L.T, b+y_j)
d_i0^2-d_00^2 = x_i.T@H@x_i - 2*x_i.T@b
h_j^2 = d_0j^2 - (b+y_j).T@inv(H)@(b+y_j)
```

There are five linear metric unknowns (three H entries and two b entries). Generic
receiver geometry must supply rank five to this design; the equivalent augmented
quadratic receiver design has six columns. SVD factor rank two alone is insufficient.
The exact cross has deficient metric design even though it spans a plane. Diagnose
that freedom; a pseudoinverse or ridge penalty must not select the right angle and
claim it was observed.

Reject materially negative `h_j^2`; report near-zero heights and their conditioning.
For positive heights retain an unknown per-event sign unless external information
fixes it. Test single-event sign flips, not just global scene reflection. No motion
smoothness is available to connect signs in the blind model.

Use an in-plane microphone-only rigid alignment and unsigned height comparison for
observable source error. Signed error requires an explicitly declared normal/half-space
convention. Do not choose a global normal or per-event signs from source truth inside
the solver. Any evaluation-only symmetry minimization must be labeled and cannot be
reported as successful physical side recovery.

Test nondegenerate planar exact/noisy cases, the unconstrained cross family, constrained
cross ablations, and near-planar data. Planar refinement must keep receiver coordinates
strictly two-dimensional; never refine them as unconstrained 3-D points. For near-planar
recordings evaluate competing models with the same held-out procedure rather than
assuming exact planarity from the reference or initializer.

## 10. Honest validation, consensus, robustness, and identifiability

### 10.1 Deterministic subsets and genuine validation

Use quality, temporal coverage, reference diversity, measurement-space spread, and
candidate conditioning to select deterministic seeds; add seeded random subsets within
a fixed budget. Never use true geometry for selection. Bound attempts and retain
rejection reasons, including absence of a complete usable algebraic subset.

Define splits before any data-dependent completion or denoising. An initial benchmark
policy to validate is 24 fitting/16 validation events at E=40, and 12/8 at E=20. Select
spread across time, not only adjacent events. These are design defaults, subject to
coverage/rank checks, not promises that every dataset supports them.

Calibrate all microphones using fitting events only. For each validation event, locate
its source using a fixed sufficient receiver subset (start with five well-conditioned
receivers for the general 3-D case); predict its measurements on the other receivers.
Check localization rank/uniqueness using fitting values only. When roots remain, report
multiple predictions or mark the event unresolved; do not choose a localization root
using withheld data and then present the chosen residual as an ordinary independent
prediction. Rotate receiver subsets across predeclared folds as feasible.

A withheld event whose source was fitted from all its measurements is not a held-out
prediction. Nor is a pair held out when the same primitive arrival was used elsewhere
to fit it. Track dependencies and model shared-reference covariance; derived receiver
pairs alone do not create extra independent validation samples. Use conditional/block
residuals when needed, with the covariance model explicitly recorded.

Perturbing validation measurements must not change the pre-scoring candidate geometry,
fit-only gauge/scaling, or completion results. Add this leakage test. Candidate selection
uses a validation set; it is not itself an unbiased final test estimate. Keep independent
seeds/final benchmark runs for reporting. Any later all-data refit has separate residuals.

### 10.2 Defined scoring and model selection

First apply physical validity, coverage, rank, and degeneracy eligibility rules. On an
identical covered validation set for each model, compute covariance-whitened residuals
and a fixed robust normalized score, for example

```text
score = sum(rho(z_k)) / N_independent_validation_coordinates
```

Specify the robust loss and tuning constant, confidence/sigma floors, inlier threshold,
minimum independent support, and allowed missing coverage in configuration before real
reference evaluation. Retain seconds-valued RMS/median/quantiles separately. Range and
metric reconstruction errors are eligibility/diagnostic quantities, not arbitrary
addends in a mixture of incompatible units.

Cluster equivalent candidates using receiver-only rigid alignment without scale changes,
then compare observable source geometry. Count support from distinct subsets/folds,
not repeated roots from one subset; report overlap rather than claiming all votes are
statistically independent. For continuous ambiguity, record a family rather than
letting the cluster tolerance manufacture a finite set of solutions.

Use a predeclared score tolerance and support rule for model preference. Prefer the
simpler model only under that explicit predictive-equivalence policy; retain competing
models when evidence/conditioning is insufficient. Do not choose general 3-D solely
because it has lower training residual or label tiny residual differences significant
without a calibrated error model. Rank and conditioning support decisions but are not
proof of global uniqueness.

### 10.3 Robust measurements and failure behavior

Use heterogeneous uncertainty from the start, with finite weight caps/floors. Compare
weighted least squares with a deterministic Huber or trimmed-consensus baseline on the
same splits; fix its parameters without ground-truth tuning. Test isolated gross errors,
entire degraded events, persistent bad channels/pairs, missing measurements, and
correlated reflected-peak bias. Include cycle-consistent wrong arrivals: cycle closure
alone cannot prove direct-path correctness.

Optional sparse/low-rank denoising from the literature is an ablation, not mandatory
preprocessing. State which matrix and rank model it assumes. Do not assume arbitrary
raw near-field TDOAs have rank 3, fill missing observations as exact data, or score only
denoised values. Preserve originals and measure changes in branch survival and coverage.

### 10.4 Matrix-specific diagnostics and rejection

For each stage record matrix dimensions, singular values, tolerance, effective rank,
condition estimate, expected rank under the chosen model, and relevant null directions.
Include corrected-factor rank, receiver metric/conic design rank, and a gauge-reduced
measurement Jacobian where available. State the scaling used; there is no universal
unexplained `conditioning_score`.

Record attempted subsets, offset roots, real roots, metric branches, completed states,
validation support, and distinct surviving geometric classes. Local Jacobian rank is a
local diagnostic and misses some discrete/global ambiguities. Explicit planar sign
flags remain necessary even with full local rank.

Reject nonfinite/complex roots, impossible ranges, indefinite metrics, unsupported
rank, failed generating equations beyond scaled tolerances, and violated declared
physical constraints. Apply coordinate/range bounds only if configured and disclosed;
never tune a bound to eliminate a valid alternative using reference geometry. Separate
numerical failure from insufficient data, weak identification, and proven ambiguity.

## 11. Optional deterministic refinement

Refine only a physically valid selected geometry under a fixed stable gauge. Use the
original TDOA equations with the covariance/basis treatment in §4, not independent
per-pair divisions when the pairs are correlated. No source-motion residuals.

For 3-D, optimize gauge-reduced receiver/source coordinates. For planar receivers,
optimize two receiver coordinates and projected source coordinates/heights while
retaining unresolved sign metadata. Any exact physical constraints stay exact; noisy
supplied constraints retain their stated measurement uncertainty and provenance.

Keep weighted least squares as a baseline and a fixed robust option for outliers.
Record objective, termination, budgets, pre/post coordinates and residuals. Accept a
refinement only under a stated improvement tolerance without invalidating the model,
physical constraints, or identifiability diagnosis. Preserve rollback. A post-selection
all-data refit is labeled training fit and cannot replace the frozen predictive score.

## 12. Audio frontend and timestamp integration

Port event detection, cross-channel association, and TDOA extraction from F1 without
the Bayesian backend. Retain raw candidate delays/confidence and acquisition timestamps;
produce the validated measurement contract. The inherited dynamic-programming smooth
lag tracker is optional and explicitly labeled as a temporal association assumption.
Test tracker-disabled and alternate-reference operation; do not treat exact cycle
consistency of arrival-derived pairs as independent validation.

Algebraic tests must bypass audio and tracking. In audio tests, separately evaluate
detection, event identity, delay extraction, geometry, and timestamp/trajectory matching.
Include receiver-time versus emission-time tests with the retarded-time renderer. Do
not supply known simulated emission times to the solver or use reference sources to
correct frontend delays. Log actual detected, usable, completed, and evaluated counts.

## 13. Public API, constraints, and CLI

Retain the conceptual interface:

```python
measurements = measure_events(audio, sample_rate, ...)
solution = calibrate_tdoa(measurements, speed_of_sound=343.0, ...)
result = calibrate_audio(audio, sample_rate, ...)
```

Keep useful CLI commands `asc calibrate WAV`, `asc check JSON`, and
`asc compare ESTIMATE_JSON REFERENCE_JSON`. Remove/deprecate old likelihood, motion,
and posterior-uncertainty switches. Expose model choices, subset budget, seed, optional
refinement, and documented physical-constraint input only as needed. Users should not
have to select an internal polynomial template to use the default pipeline.

Export structured ambiguous/partial results rather than raising opaque errors or
claiming success from a small residual. Define CLI exit statuses and JSON compatibility
for solved, ambiguous, insufficient-data, and numerical-failure outcomes. Keep timestamps,
channel IDs, units, covariance approximation, supplied constraints, and representative
source-sign conventions visible in exports and plots.

## 14. Output diagnostics

Replace posterior-centric diagnostics with stage-specific records. Include result
status/reasons, model and competitors, constraint provenance, seed/metric backend,
gauge shifts, per-stage counts and rejection reasons, coverage, fit/validation split
IDs, matrix ranks/tolerances, residuals with units, local weak modes, geometric
classes/family information, source-sign flags, and refinement/rollback status.

The following is an illustrative field sketch, not measured output. Null denotes
unavailable diagnostics; it must not be converted to a successful zero-error result:

```json
{
  "status": "ambiguous",
  "model": "receiver2d_source3d",
  "ambiguity_reasons": ["continuous_array_deformation", "per_event_height_sign"],
  "constraints": [],
  "offset_solver": null,
  "generated_hypotheses": 0,
  "completed_events": 0,
  "validation": {"split_id": null, "independent_coordinates": 0, "score": null},
  "ranks": {
    "compacted_range": {"expected": 2, "observed": null},
    "receiver_quadratic_design": {"columns": 6, "observed": null}
  },
  "tdoa_rms_s": null,
  "corrected_range_rms_m": null,
  "isolated_geometric_class_count": null,
  "refinement": {
    "mode": "none",
    "attempted": false,
    "accepted": false,
    "max_nfev": 0,
    "improvement_tolerance": null
  }
}
```

Include the evidence supporting an actual ambiguity diagnosis in real output; the
illustrative reason strings above are not a license to assume every planar scene is a
cross. Reserve family dimension and proof/counterexample fields for established results.

## 15. Implementation sequence and exit gates

### Milestone 0 — identifiability and mathematical feasibility

Reproduce §§2.3 and 6.2; specify planar sign/evaluation policy, range/reference/time
conventions, and exact constraints for metric recovery. Exercise a small 7r/6s template
prototype on exact random scenes with source-independent root verification. Pin its
reproduction procedure. Approve the model/feasibility specification before substantial
production backend replacement. This gate distinguishes missing information from a
numerical failure and does not imply the prototype is production-ready.

### Milestone A — contracts and fixtures

**Implementation status:** A01–A07 are implemented on
`rewrite/stratified-tdoa-calibration`. The branch workflow does not run on feature-branch
pushes, so full matrix CI remains to be exercised by PR or manual workflow dispatch.

Port the pinned 20-event fixtures, add 40-event variants and the diagnostic sweep,
introduce covariance/lineage/validity and timestamp contracts, fix gauge-pivot assumptions,
and add the real-data harness. Establish distinct planar, cross-ambiguity, and constrained
cross test contracts. Verify reference, duplicate-pair, masking, and time conversions.

### Milestone A completion note

Milestone A is complete on `rewrite/stratified-tdoa-calibration`. The branch now
contains the frozen 20-event fixtures, primary 40-event variants, sweep hooks,
measurement/covariance/timestamp contracts, reference-star basis, distinct planar and
cross fixture contracts, real-Myotis input harness, and conditioning-aware coordinate
gauge support.

The 20/40-event **end-to-end geometry thresholds remain specification gates**, not
claims that the unfinished stratified solver already passes them. Those gates become
executable when Milestones C/E connect the TDOA solver and audio frontend.

### Milestone B — exact linear and metric stages

Implement conditioned/re-gauged 9r/5s and 7r/4s anchors, corrected-range reconstruction,
SVD factors, explicit metric equations, positive-definiteness tests, and exact/noisy
stage diagnostics. Test the mixed-dimensional use separately. No audio dependency.

### Milestone B completion note

Milestone B is complete on `rewrite/stratified-tdoa-calibration`. The branch now
contains explicit paper-notation compaction, deterministic arrival re-gauging, the
9r/5s and 7r/4s linear anchors, corrected-range SVD factorization, and the first
rank-three affine-to-Euclidean metric backend. The metric stage normalizes physical
range scale internally, enumerates isolated real quartic roots, rejects singular or
indefinite metrics using the original equations, reconstructs both receiver/source
coordinates, and reports weak/degenerate outcomes rather than clipping a metric.

Focused tests cover the documented zero-reference trap, the exact `rng=319` gauge
reproduction, multiple seeds/scales, all-2D and mixed-dimensional 7r/4s offsets,
exact and noisy factorization, exact TDOA-to-Euclidean round trip, noisy metric
candidate ranking, multiple algebraic roots, exact planar degeneracy under the 3-D
backend, and near-planar weak-rank separation.

As with Milestone A, feature-branch pushes do not trigger this repository's CI
workflow. Full Ruff/Ty/Python-matrix/build execution remains to be exercised by a PR
or manual workflow dispatch; the branch has not been represented as having passed a
workflow that did not run.

### Milestone C — production 7r/6s, expansion, and prediction

Check in the generated template and reproducible tooling. Test all isolated real roots,
extraneous-root rejection, noise/scale/degenerate cases, fit-only offset/state expansion,
covariance-aware validation, leakage prevention, and independent-subset consensus.
Solve the eight-microphone TDOA-only 20/40-event cases before audio integration. Explicitly
exercise missing measurements and outliers; do not conceal unsupported subsets.

### Milestone C completion note

Milestone C is implemented on `rewrite/stratified-tdoa-calibration`. The branch now
contains a reproducible 7r/6s rank-constraint template, deterministic real-root
enumeration, independent all-minor verification, event/microphone expansion,
lineage-safe held-out validation, covariance-aware robust scoring, geometric-class
consensus, and an eight-microphone TDOA-only solver.

Independent numerical execution of the committed equations on the project fixtures
recovers the exact 20- and 40-event eight-microphone scenes to approximately machine
precision and reproduces the frozen geometry/TDOA limits with wide margin. The tests
also cover sparse missing measurements plus fitting and validation outliers.

Root-search completeness is reported honestly: the generated primary-minor system and
all-minor verifier are exact, but the current runtime root enumeration uses a bounded
deterministic Halton/least-squares schedule. `search_stabilized` is therefore a
numerical completeness diagnostic, not a symbolic/action-matrix certificate that no
additional isolated real root exists outside the search region. No branch is selected
using source truth, and every accepted root is checked against equations not used by
the primary solve.

As in Milestones A/B, feature-branch pushes do not trigger this repository's CI
workflow, and the execution container cannot resolve GitHub for a branch clone. The
full Ruff/Ty/Python 3.11–3.14/build matrix still requires PR or manual workflow
dispatch; this document does not claim that workflow has run.

### Milestone D — planar observables and constrained calibration

Implement rank-2 factorization, linear planar metric design, unsigned heights, continuous
cross diagnostics, explicit physical constraints, and near-planar model comparisons.
Require nondegenerate planar accuracy and unconstrained-cross ambiguity; verify that
removing a constraint restores the unresolved freedom. Test independent source signs.

### Milestone E — audio/WAV integration

Reconnect detection and association, validate uncertainty/lineage/timestamps, and pass
the 8/12/16/24 microphone audio gates at both 20 and 40 events plus PCM16 variants.
Keep the original 18-event cross audio timing. Report pre-refinement geometry and any
completion failures, not just final RMS.

### Milestone E completion note

Milestone E is implemented on `rewrite/stratified-tdoa-calibration`. The production
audio path now detects discrete transients, associates per-channel arrivals, emits the
immutable reference-star measurement contract, and invokes the stratified geometry
backend directly. The public audio/WAV entry points no longer pass motion priors,
posterior likelihood choices, or Laplace-uncertainty controls into calibration.

The rendered-audio specification covers 8/12/16/24 microphones at both frozen 20-event
and primary 40-event counts. Arrays larger than eight use the solved eight-microphone
core and recovered source states to complete additional microphones deterministically
from their measured reference-star ranges. PCM16 WAV gates retain the documented
0.18 m microphone / 0.22 m source acceptance limits.

Temporal delay tracking remains an optional frontend association heuristic only.
Tracker-disabled extraction is separately testable and uses the same measurement
contract. The geometry backend itself receives no event-motion smoothness term.

Result serialization now records deterministic stratified diagnostics rather than
Bayesian posterior/uncertainty fields. Optional post-selection WLS/Huber geometry
refinement is implemented with covariance-aware residuals, frozen validation scores,
rollback state, explicit budgets, and pre/post refinement diagnostics.

As with prior milestones, feature-branch pushes do not trigger this repository's CI
workflow and the execution environment cannot clone the branch from GitHub. The full
Ruff/Ty/Python 3.11–3.14/build matrix therefore still requires a PR or manual workflow
dispatch; this completion note does not claim that workflow has run.

### Milestone F — real Myotis

Run the actual recording with recorded input/configuration hashes; evaluate observable
geometry and conditioning without reference-driven tuning. Distinguish blind,
constrained, ambiguous, and unavailable-data outcomes. Record the run before release;
an unidentifiable dataset requires honest reporting or justified extra acquisition,
not a fabricated unique solution.

### Milestone G — cleanup

Remove obsolete Bayesian geometry/initialization code and misleading options only after
applicable gates pass. Update README, examples, exports, JSON documentation, plotting,
and validation instructions without leaving competing silent backends.

## 16. CI and acceptance policy

Production code commits must keep the existing applicable suite and implemented stage
tests green:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests examples
uv run pytest
uv build
```

Retain the Python 3.11–3.14 matrix. During staging, new not-yet-implemented end-to-end
specifications may be explicitly marked strict expected-failure with an owner, reason,
and removal milestone, or isolated in a documented specification job. Do not disable
previously passing tests, blanket-xfail algebraic errors, or claim pending gates pass.
Remove those temporary markers before release.

Release gates: frozen 20-event and added 40-event random-3D/WAV thresholds; exact/noisy
math and generated-root checks; genuine validation and lineage invariance; robust
missing/outlier behavior; planar observable recovery; explicit cross/sign ambiguities;
constraint ablations; and a recorded real-data integration outcome. Large multi-seed
noise/runtime sweeps may use a separate job, but small deterministic checks belong in
normal CI. Publish success/ambiguity/failure rates and seed lists, not best seeds only.

For documentation-only commits, check links/anchors, fence/example consistency, and
cross-document requirements. Running standalone mathematical examples is not running
the production suite or validating real audio; state that distinction in review notes.

## 17. Deliberately postponed work

Defer posterior/Laplace uncertainty after geometry selection, trajectory-based sign
resolution, microphone clock offsets/drift, sound-speed estimation with metric anchors,
full multipath/direct-path mixture models, and learned uncertainty calibration.

Do not defer basic robust weighting, validity/covariance handling, explicit metric
constraints, or ambiguity diagnostics. Optional literature comparisons do not block the
first release, and none may introduce unavailable anchors, DOAs, or source positions
into the blind benchmark. Any future prior-assisted result must be labeled separately
from what the acoustic measurements alone identify.

## 18. Reference map

Paper roles are scoped in §2.5. Equations and counterexamples labeled as our derivations
are implementation/review analysis, not attributed experimental results. Full-text
reproduction and benchmark evidence remain required before adopting optional methods.

### R1

Yubin Kuang and Kalle Åström, **Stratified Sensor Network Self-Calibration From TDOA
Measurements**, EUSIPCO 2013.
[Paper](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2013/papers/1569743721.pdf).
Relevant anchors: linear-offset Eq. (2), generic offset-solution counts in Fig. 1,
compacted rank construction, and expansion in §3.3. Primary geometry reference.

### R2

Kuang, Burgess, Torstensson, and Åström, **A Complete
Characterization and Solution to the Microphone Position Self-Calibration Problem**,
ICASSP 2013.
[Institutional record](https://portal.research.lu.se/en/publications/a-complete-characterization-and-solution-to-the-microphone-positi/).
TOA metric-recovery reference; §8 fixes the implementation's own explicit convention.

### R3

Erik Ask, Yubin Kuang, and Kalle Åström, **A Unifying Approach to Minimal Problems in
Collinear and Planar TDOA Sensor Network Self-Calibration**, EUSIPCO 2014.
[Paper](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2014/HTML/papers/1569925215.pdf).
Mixed-dimensional rank/recovery and distance-to-subspace treatment. The cross-specific
continuous-deformation proof in §2.3 is our additional fixture analysis.

### R4

Charles Vanwynsberghe et al., **Geometric calibration of very large microphone arrays in
mismatched free field**, JASA 2019.
[DOI](https://doi.org/10.1121/1.5083829).
Sparse-outlier/robustness reference; any matrix-denoising adaptation needs its own model
justification and an original-measurement validation ablation.

### R5

Woźniak and Kowalczyk, **Reverberation-Robust Self-Calibration and
Synchronization of Distributed Microphone Arrays by Mitigating Heteroscedasticity and
Outlier Occurrence in TDoA Measurements**, Sensors 24(1), 114 (2024 issue).
[DOI](https://doi.org/10.3390/s24010114).
Measurement-robustness ideas; not the same scalar-microphone geometry/input assumptions.

### R6

Wang, Chen, and Yin, **A constrained total least squares calibration method for
distributed microphone array**, Applied Acoustics 2018.
[DOI](https://doi.org/10.1016/j.apacoust.2018.05.022).
Optional baseline only when reference-node/source-estimation assumptions are reproduced
and disclosed; verify the full method before implementing the comparison.

### R7

Zhao, Wang, and Xu, **A geometric solution to microphone array position calibration and
its confidence interval analysis**, Applied Acoustics 2024.
[Supplied article](https://www.sciencedirect.com/science/article/abs/pii/S0003682X24002597)
/ [DOI](https://doi.org/10.1016/j.apacoust.2024.110108).
DOA-based geometric calibration/conditioning reference. Abstract-level relevance is not
a full-text verification of its source/anchor assumptions or a blind-TDOA implementation.

### R8

Krishnaprasad Nambur Ramamohan, Sundeep Prabhakar Chepuri, Daniel Fernandez Comesana,
and Geert Leus, **Self-Calibration of Acoustic Scalar and Vector Sensor Arrays**,
IEEE Transactions on Signal Processing, DOI 10.1109/TSP.2022.3214383.
[Supplied IEEE record](https://ieeexplore.ieee.org/abstract/document/9918024)
/ [DOI](https://doi.org/10.1109/TSP.2022.3214383)
/ [authors' preprint](https://arxiv.org/abs/2104.02561).
Joint gain/phase and DOA estimation; methodological reference, not position recovery.

### R9

Cui, Li, and Yu, **A systematic solution to 3D anchorless direction estimation using
TDOA measurements**, Signal Processing 2024.
[DOI](https://doi.org/10.1016/j.sigpro.2023.109363).
Optional far-field comparison; reproduce only after verifying the regime and available
inputs. Do not substitute bearings for near-field event source positions.

### R10

**Low rank properties for synchronizing microphones and sources in Ad-Hoc wireless
acoustic sensor network**, Digital Signal Processing, reported 2026 reference.
[DOI lead](https://doi.org/10.1016/j.dsp.2026.106106).
Retained from the earlier literature review as a deferred lead only. Metadata/full-text
access was not re-established during consolidation; re-verify before technical use.
No first-pass requirement or performance claim relies on this item.

### Unresolved supplied reference

[IEEE document 10545598](https://ieeexplore.ieee.org/abstract/document/10545598) could not
be reliably identified from accessible metadata. Obtain title/DOI/full text before
classification; do not guess the title, assumptions, or relevance.

### F1

Pinned PR #7 fixture and frontend provenance, commit
`cd02c4990234dd9ef877cb1af149fb4df64574d5`:
[test_end_to_end_audio.py](https://github.com/fhaefele/acoustic-self-calibration/blob/cd02c4990234dd9ef877cb1af149fb4df64574d5/tests/test_end_to_end_audio.py),
[events.py](https://github.com/fhaefele/acoustic-self-calibration/blob/cd02c4990234dd9ef877cb1af149fb4df64574d5/src/acoustic_self_calibration/events.py),
[simulation.py](https://github.com/fhaefele/acoustic-self-calibration/blob/cd02c4990234dd9ef877cb1af149fb4df64574d5/src/acoustic_self_calibration/simulation.py).
Source of schedules, original thresholds, cross coordinates, temporal lag tracking,
derived pair measurements, and retarded-time rendering assumptions.

### F2

[geometry.py at 203ff17](https://github.com/fhaefele/acoustic-self-calibration/blob/203ff1712d2ca757248c4c7a8d9bdbc8734e55eb/src/acoustic_self_calibration/geometry.py).
Audit target: first-three-microphone gauge construction and rigid alignment without
scale fitting. Do not reuse the collinearity assumption unchanged.

### F3

[evaluation.py at 203ff17](https://github.com/fhaefele/acoustic-self-calibration/blob/203ff1712d2ca757248c4c7a8d9bdbc8734e55eb/src/acoustic_self_calibration/evaluation.py).
Preserve microphone-only alignment and reference-time-overlap semantics while adding
observable planar metrics and explicit timestamp conventions.

[R1]: #r1
[R2]: #r2
[R3]: #r3
[R4]: #r4
[R5]: #r5
[R6]: #r6
[R7]: #r7
[R8]: #r8
[R9]: #r9
[R10]: #r10

## 19. Scoped implementation tickets

These tickets are intentionally small. Each should fit in one focused pull request,
have one primary deliverable, and be independently reviewable. Do not combine adjacent
tickets merely because they touch the same module. Dependencies name ticket IDs; a
ticket should not grow to absorb unfinished work from its dependencies.

### Foundation and mathematical-contract tickets

#### T00 — Encode the Myotis-cross ambiguity regression

- **Status:** complete — encoded in `tests/test_stratified_fixtures.py` (`275d34e`).
- **Scope:** Add a pure-math test that constructs the ideal two-arm Myotis receiver
  layout and verifies a nontrivial continuous deformation with unchanged
  receiver-source distances/TDOAs.
- **Depends on:** none.
- **Done when:** the test demonstrates measurement equivalence numerically and documents
  that right-angle recovery is not an identifiable acceptance criterion.

#### T01 — Encode planar source-height sign ambiguity

- **Status:** complete — encoded in `tests/test_stratified_fixtures.py` (`275d34e`).
- **Scope:** Add a pure-math test that reflects one source event at a time through a
  planar receiver array and verifies identical distances/TDOAs.
- **Depends on:** none.
- **Done when:** independent per-event sign flips are covered and the expected ambiguity
  is documented in the test.

#### T02 — Implement arrival-gauge conversion helpers

- **Status:** complete — arrival-gauge helpers and exact invariance tests are implemented in the stratified notation layer.

- **Scope:** Add notation helpers for `f'_ij = f_ij + q_j` / `o'_j = o_j + q_j`,
  including conversion back to gauge-invariant corrected ranges.
- **Depends on:** none.
- **Done when:** zero-reference frontend measurements can be re-gauged and round-trip
  tests recover the same corrected ranges.

#### T03 — Replace fixed-index coordinate gauge selection

- **Status:** complete — conditioning-aware non-reordering gauge utilities/tests in `e0ff607`.
- **Scope:** Implement geometry-conditioned gauge selection instead of using fixed first
  microphones; include a planar-specific path.
- **Depends on:** none.
- **Done when:** receiver permutations and collinear first microphones no longer break
  canonicalization, and true degeneracy raises an explicit diagnostic.

#### T04 — Write the concrete metric-upgrade math contract

- **Status:** complete — `metric_upgrade.py` documents the supported 3-D H/b convention, equations, realizability checks, polynomial root handling, and noisy tolerances.

- **Scope:** Add a developer-facing section or module doc specifying unknowns, gauge,
  equations, realizability/positive-definiteness checks, mirror handling, and noisy
  tolerances for the first 3-D metric-upgrade path.
- **Depends on:** none.
- **Done when:** `metric_upgrade.py` can be implemented without unresolved algebraic
  placeholders.

### Milestone A — contracts and fixtures

#### A01 — Port the frozen 20-event random-3D fixtures

- **Status:** complete — frozen 20-event fixtures and thresholds are encoded.
- **Scope:** Bring the PR #7 8/12/16/24-microphone 20-event synthetic fixtures into the
  rewrite branch without changing seeds, scene geometry, or thresholds.
- **Depends on:** none.
- **Done when:** the fixtures exist as specification tests, even if solver-facing tests
  are temporarily marked with a narrowly documented expected-failure.

#### A02 — Add matching 40-event random-3D fixtures

- **Status:** complete — deterministic 40-event variants use the same acceptance limits.
- **Scope:** Add 40-event variants of A01 with otherwise equivalent scene generation and
  the same geometry/TDOA thresholds.
- **Depends on:** A01.
- **Done when:** all four microphone counts have deterministic 40-event fixtures and no
  20-event fixture was altered.

#### A03 — Add the event-count diagnostic sweep

- **Status:** complete — 8/12/20/30/40 diagnostic sweep bookkeeping and output hooks are present.
- **Scope:** Add a non-gating diagnostic helper/test covering 8, 12, 20, 30, and 40
  events for fixed geometry seeds.
- **Depends on:** A01, A02.
- **Done when:** the sweep records event count plus placeholders/hooks for geometry
  error, TDOA error, conditioning, hypothesis survival, and runtime.

#### A04 — Introduce the measurement boundary dataclass

- **Status:** complete — immutable validated measurement/covariance/timestamp contract is implemented.
- **Scope:** Implement `EventTDOAMeasurements` fields for receive-time semantics, TDOAs,
  confidence/sigma, validity mask, pair list, event channel, and measurement-basis tag.
- **Depends on:** none.
- **Done when:** shape/value validation is unit-tested and timestamps are explicitly
  named as receiver times rather than source emission times.

#### A05 — Add an independent reference-star measurement basis

- **Status:** complete — reference-star basis, covariance, rank, masking, and pair round-trip tests are implemented.
- **Scope:** Convert per-channel arrival delays into an `M-1` independent reference-star
  basis per event while retaining redundant pairs only as optional diagnostics.
- **Depends on:** A04.
- **Done when:** reference-star measurements round-trip to redundant pair differences and
  duplicate derived pairs do not increase the counted independent measurement rank.

#### A06 — Add nondegenerate planar and cross fixtures

- **Status:** complete — nondegenerate planar, unconstrained cross, and constrained-cross contracts are distinct.
- **Scope:** Add one generic nondegenerate planar receiver fixture for positive recovery
  tests and preserve the exact Myotis cross as a separate ambiguity fixture.
- **Depends on:** T00, T01.
- **Done when:** tests clearly distinguish `planar_success` from `cross_ambiguity`
  expectations.

#### A07 — Add the real-Myotis integration harness

- **Status:** complete — real-Myotis path/env harness records hashes, mapping, timing/config provenance, and not-run status.
- **Scope:** Add a path/environment-variable driven harness that loads real audio and
  reference geometry and emits evaluation JSON without embedding private/large data.
- **Depends on:** A04.
- **Done when:** missing data produces an explicit skip/unavailable outcome and supplied
  paths produce a reproducible invocation/config record.

### Milestone B — exact linear and metric stages

#### B01 — Implement compacted measurement construction

- **Status:** complete — corrected squared-range construction, compaction operators, cross-Gram construction, and rank diagnostics are implemented.

- **Scope:** Implement the paper-notation matrix construction and differencing/compaction
  operators only.
- **Depends on:** T02, A04.
- **Done when:** hand-built exact examples match direct equation-by-equation reference
  calculations.

#### B02 — Implement the 9r/5s 3-D linear offset anchor

- **Status:** complete — conditioned/re-gauged 9r/5s 3-D linear offset recovery is implemented with the pinned zero-reference regression.

- **Scope:** Implement only the 9-receiver/5-event linear offset solver with explicit
  arrival-gauge handling.
- **Depends on:** B01, T02.
- **Done when:** noiseless random scenes recover gauge-invariant corrected ranges below
  the documented tolerance from zero-reference input.

#### B03 — Implement the 7r/4s 2-D linear offset anchor

- **Status:** complete — 7r/4s 2-D and mixed planar-receiver/3-D-source linear anchors are implemented and tested.

- **Scope:** Implement only the 7-receiver/4-event linear offset solver with the same
  gauge convention as B02.
- **Depends on:** B01, T02.
- **Done when:** exact planar anchor tests pass and reference-receiver changes preserve
  corrected geometry.

#### B04 — Implement corrected-range factorization

- **Status:** complete — corrected-range SVD affine factorization exposes singular values, discarded energy, and reconstruction error.

- **Scope:** Convert accepted offsets to corrected ranges/squared-distance structure and
  produce SVD affine factors plus singular-value diagnostics.
- **Depends on:** B02.
- **Done when:** exact 3-D scenes reconstruct the intended affine rank and expose all
  singular values used by later conditioning checks.

#### B05 — Implement the first exact 3-D metric upgrade

- **Status:** complete — the first exact 3-D 9r/5s affine-to-Euclidean metric upgrade enumerates real polynomial branches, enforces H>0, and validates original ranges.

- **Scope:** Implement the metric-upgrade equations specified in T04 for exact 3-D
  affine factors; no noisy heuristics yet.
- **Depends on:** T04, B04, T03.
- **Done when:** exact scenes recover all isolated real metric branches, reject
  non-realizable branches, and reproduce corrected distances to numerical precision.

#### B06 — Add noisy and degenerate metric-upgrade diagnostics

- **Status:** complete — noisy/near-degenerate diagnostics report factor rank separation, receiver-design rank/conditioning, metric branch counts, and tolerance failures.

- **Scope:** Add condition estimates, residual tolerances, positive-definiteness checks,
  and an explicit degenerate/non-isolated status around B05.
- **Depends on:** B05.
- **Done when:** near-degenerate scenes report poor conditioning/degeneracy instead of a
  plausible-looking unique solution.

### Milestone C — production 7r/6s, expansion, and consensus

#### C01 — Build the 7r/6s solver-generation scaffold

- **Status:** complete — reproducible 7r/6s generator/template scaffold with template hash and byte-for-byte regeneration test (`3fcf52e8`, `28952fd3`).

- **Scope:** Add the symbolic/generation script, deterministic inputs, generated-file
  header, and reproducibility check without yet committing a production template.
- **Depends on:** B01.
- **Done when:** rerunning generation produces byte-stable or semantically identical
  intermediate solver artifacts.

#### C02 — Check in the 7r/6s generated runtime template

- **Status:** complete — NumPy/SciPy runtime 7r/6s real-root enumerator with deterministic Halton starts and generated primary-minor template (`f696f74d`, `f07c6417`).

- **Scope:** Generate and commit the NumPy/SciPy-only 7r/6s runtime solver template.
- **Depends on:** C01.
- **Done when:** runtime code has no symbolic dependency and returns all finite isolated
  candidate roots for exact supported scenes.

#### C03 — Verify roots and detect non-isolated cases

- **Status:** complete — candidates are independently checked against all 75 minors, deduplicated, range-checked, and assigned finite-difference Jacobian-rank diagnostics; exact 20/40-event physical-root regressions are present (`f07c6417`, `ef159716`).

- **Scope:** Evaluate C02 roots against the original determinant/rank equations and add
  explicit root rejection/degeneracy diagnostics.
- **Depends on:** C02.
- **Done when:** extraneous roots are rejected, valid roots survive scale/permutation
  tests, and near-degenerate scenes do not masquerade as finite unique root sets.

#### C04 — Expand a hypothesis to additional events

- **Status:** complete — column-space event-offset expansion recovers additional fitting events with per-event membership diagnostics (`148c1739`).

- **Scope:** Add `stratified/expansion.py` support for recovering one or more additional
  event offsets/source states using only a designated completion receiver subset.
- **Depends on:** B05, C03.
- **Done when:** exact held-back events are recovered and the measurements used for their
  completion are recorded explicitly.

#### C05 — Expand a hypothesis to additional microphones

- **Status:** complete — deterministic source localization and robust receiver trilateration expand seed geometry to additional events/microphones (`148c1739`, `ca366d90`).

- **Scope:** Extend C04 to localize additional receivers from a designated completion
  event subset.
- **Depends on:** C04.
- **Done when:** exact held-back receivers are recovered and completion provenance is
  retained.

#### C06 — Enforce generation/completion/validation lineage

- **Status:** complete — generation/completion/validation primitive masks are explicit, disjoint, and leakage-tested (`eaf34bb6`).

- **Scope:** Introduce `generation_mask`, `completion_mask`, and `validation_mask` and
  reject leakage where a scored state reuses its fit measurements as held-out evidence.
- **Depends on:** C04, C05.
- **Done when:** dedicated leakage tests fail on overlap and pass on genuinely disjoint
  validation measurements.

#### C07 — Add heteroscedastic robust validation

- **Status:** complete — heteroscedastic Huber validation, missing/outlier handling, shared-reference conditional covariance, and supported-subspace whitening are implemented (`eaf34bb6`, `4c933851`, `6496fe0c`).

- **Scope:** Implement one deterministic robust scorer on the independent/whitened
  measurement basis, with validity masks and capped weights.
- **Depends on:** A05, C06.
- **Done when:** clean-data behavior matches weighted least squares closely and seeded
  sparse-outlier/missing-data tests degrade gracefully.

#### C08 — Add geometric-class consensus

- **Status:** complete — rigid-equivalence clustering counts distinct subset support rather than duplicate roots and retains competing classes (`eaf34bb6`).

- **Scope:** Cluster equivalent hypotheses across independent minimal subsets and count
  support per geometric class rather than per polynomial root.
- **Depends on:** C03, C07.
- **Done when:** duplicate roots from one subset do not inflate consensus and competing
  inequivalent classes remain separately visible.

#### C09 — Pass the 8-microphone TDOA-only 20/40-event gate

- **Status:** complete — the eight-microphone TDOA-only orchestrator covers frozen 20-event and primary 40-event gates, fitting/validation splits, eighth-microphone completion, held-out prediction, missing data, and gross outliers (`5f64133b`, `6496fe0c`, `ef6b1646`).

- **Scope:** Wire B/C stages together for precomputed TDOAs only; do not reconnect audio.
- **Depends on:** C05, C06, C07, C08.
- **Done when:** both frozen 20-event and primary 40-event 8-microphone random-3D cases
  meet their geometry/TDOA thresholds or return an explicit justified ambiguity.

### Milestone D — planar observables and constraints

#### D01 — Implement planar rank-2 factorization and metric equations

- **Status:** complete — rank-2 factorization feeds an explicit linear five-parameter planar H/b metric recovery path with exact/nonexact diagnostics (`a1f177fe`).

- **Scope:** Implement only the receiver-rank-2/source-rank-3 algebraic path for exact
  nondegenerate planar measurements.
- **Depends on:** B01, T04.
- **Done when:** exact nondegenerate planar fixtures reconstruct the observable metric
  quantities without projecting a 3-D solution onto a plane.

#### D02 — Represent unsigned source height in results

- **Status:** complete — planar results expose projected source coordinates, unsigned plane-normal heights, representative positive-height coordinates, and an all-false sign-known mask unless an external sign convention is supplied (`a1f177fe`, `ddd026e2`).

- **Scope:** Extend planar hypothesis/result types to carry in-plane source coordinates,
  unsigned plane-normal distance, and sign-ambiguity diagnostics.
- **Depends on:** T01, D01.
- **Done when:** independent event sign flips produce the same observable result and no
  hidden sign choice is made.

#### D03 — Add nondegenerate planar exact/noisy gates

- **Status:** complete — exact 20/40-event planar TDOA gates plus seeded-noise metric/TDOA regressions are implemented (`a1f177fe`, `ddd026e2`).

- **Scope:** Add accuracy and conditioning tests for the nondegenerate planar fixture at
  exact and seeded noisy settings.
- **Depends on:** A06, D01, D02.
- **Done when:** exact recovery passes tight tolerances and noisy tests meet documented
  observable-quantity thresholds.

#### D04 — Implement cross/continuous-ambiguity diagnostics

- **Status:** complete — planar metric/conic rank diagnostics expose the exact Myotis cross as a one-dimensional continuous metric family with a concrete null direction; near-cross conditioning is also tested (`a1f177fe`, `490733f8`, `155b1e70`).

- **Scope:** Detect the conic/two-line cross degeneracy and surface a continuous-ambiguity
  status before candidate ranking.
- **Depends on:** T00, D01.
- **Done when:** the exact synthetic Myotis cross returns ambiguous/degenerate and is not
  forced into a unique right-angle geometry.

#### D05 — Add one explicit planar anchor constraint

- **Status:** complete — an explicit exact right-angle arm constraint lifts the cross metric rank, records provenance/residual, and the ablation restores the unconstrained ambiguity (`d2ba9263`, `9c9208c`, `155b1e70`).

- **Scope:** Support one documented physical constraint that breaks the cross ambiguity,
  such as a known inter-arm angle or cross-arm distance.
- **Depends on:** D04.
- **Done when:** the constrained cross recovers uniquely and removing the constraint
  restores the ambiguity.

#### D06 — Compare planar and 3-D models on held-out data

- **Status:** complete — planar and general-3D solvers use the same deterministic held-out event split, compare predictive scores with structural conditioning, select clear planar/3-D cases, and return ambiguity for near-ties (`04a5d525`, `5327e0f9`, `c57a7278`).

- **Scope:** Implement model comparison using the same validation basis and a documented
  indistinguishability tolerance.
- **Depends on:** C06, C07, D03.
- **Done when:** clearly planar/3-D fixtures select appropriately and near-ties return
  model ambiguity instead of choosing by training residual.

### Milestone D completion note

Milestone D is implemented on `rewrite/stratified-tdoa-calibration`. The branch now
contains a mixed-dimensional rank-2 planar metric backend, unsigned source-height
semantics, exact/noisy planar TDOA gates, continuous cross-family diagnostics with
explicit metric null directions, an exact right-angle physical constraint, constraint
ablation tests, and held-out planar-versus-general-3D model comparison.

The unconstrained Myotis cross is deliberately not assigned a unique right angle. Its
metric design has a one-dimensional nullspace and is returned as degenerate. Supplying
the explicit right-angle constraint lifts the metric design to full rank while leaving
per-event source-side signs unresolved; removing the constraint restores the family.

Independent numerical execution of the committed planar equations reproduces exact
nondegenerate scenes to machine precision. A seeded 2 us TDOA-noise check remains well
inside the milestone's observable-geometry/TDOA tolerances, while near-cross and
slightly nonplanar tests expose conditioning/model uncertainty instead of silently
projecting data to a plane.

As with prior milestones, feature-branch pushes do not trigger this repository's CI
workflow and the execution environment cannot clone the branch from GitHub. The full
Ruff/Ty/Python 3.11–3.14/build matrix therefore still requires a PR or manual workflow
dispatch; this completion note does not claim that workflow has run.

### Milestone E — audio and WAV integration

#### E01 — Port transient detection only

- **Status:** complete — PR #7 transient detection was ported without geometry/lag tracking and is covered on frozen 20/40-event 8/12/16/24-microphone pulse fixtures (`bf110f12`).

- **Scope:** Reintroduce event detection from PR #7 without geometry or lag tracking.
- **Depends on:** A04.
- **Done when:** event-count/timing unit tests pass on the synthetic pulse fixtures.

#### E02 — Port event TDOA association into the new measurement contract

- **Status:** complete — event arrival association now emits immutable mic-0 reference-star `EventTDOAMeasurements` with confidence, sigma, validity, covariance metadata, sub-sample peaks, and tracker-enabled/disabled diagnostics (`fd6343d3`).

- **Scope:** Reintroduce arrival/TDOA estimation, emit the independent reference-star
  basis, validity/confidence metadata, and retain temporal tracking only as an optional
  association heuristic.
- **Depends on:** A05, E01.
- **Done when:** tracker-enabled and tracker-disabled diagnostics are testable and both
  produce valid `EventTDOAMeasurements`.

#### E03 — Integrate the 8-microphone audio pipeline

- **Status:** complete — the 8-microphone 20/40-event rendered-audio path is audio -> detection -> event TDOA contract -> stratified solver, with no motion prior or posterior branch selection (`f574019e`, `7911fc9b`).

- **Scope:** Connect E02 to the stratified solver for only the 8-microphone 20/40-event
  rendered-audio cases.
- **Depends on:** C09, E02.
- **Done when:** both 8-microphone audio gates pass without motion priors or
  posterior-based branch selection.

#### E04 — Extend audio gates to 12/16/24 microphones

- **Status:** complete — the stratified backend expands the 8-microphone seed solution to 12/16/24 microphones by deterministic extra-receiver completion; 20/40-event audio gates cover all requested counts (`a33e9ec6`, `f0aa3d27`, `b6285508`).

- **Scope:** Parameterize the working E03 path across the remaining microphone counts.
- **Depends on:** E03.
- **Done when:** 20- and 40-event 12/16/24 cases pass the unchanged thresholds with the
  same public pipeline.

#### E05 — Restore the PCM16 WAV regression

- **Status:** complete — WAV loading forwards the stratified event API and PCM16 20/40-event regressions preserve the relaxed WAV geometry thresholds (`235c8757`).

- **Scope:** Connect the working audio pipeline through WAV loading and PCM16 scaling.
- **Depends on:** E03.
- **Done when:** the PCM16 8-microphone regression passes geometry/TDOA checks with
  uncertainty assertions limited to implemented deterministic diagnostics.

#### E06 — Export solver diagnostics through API/JSON

- **Status:** complete — public results/JSON now expose backend/model/status, hypothesis/class counts, conditioning, ambiguity, lineage counts, held-out robust validation, rejected hypotheses, extra-microphone completion, and deterministic refinement/rollback diagnostics; solved/ambiguous/degenerate serialization is tested.

- **Scope:** Wire status, model, hypothesis counts, conditioning, ambiguity, lineage,
  inlier/rejection, and refinement fields into result serialization.
- **Depends on:** C08, D02.
- **Done when:** deterministic JSON snapshot/schema tests cover success, ambiguous, and
  degenerate outcomes.

### Milestone F — real Myotis

#### F01 — Produce the real-Myotis geometry conditioning report

- **Status:** complete — receiver-only conditioning now reports affine rank, singular-value ratios, best-fit-plane residual, planar metric/conic ranks and condition numbers, continuous metric null directions, cross-like degeneracy, and explicit identifiable/ambiguous quantities (`1d41b7f8`, `ad1b75dc`, `00d6a05e`).

- **Scope:** Analyze only the supplied reference microphone coordinates for affine rank,
  planar/cross degeneracy, singular-value gaps, and condition estimates.
- **Depends on:** A07, D04.
- **Done when:** the report states which geometric quantities are identifiable before
  any audio calibration is interpreted.

#### F02 — Run and record the blind real-Myotis calibration

- **Status:** complete — the blind runner validates/hashes inputs, records channel mapping/config/seed/revision, runs the production audio pipeline without reference geometry/source states, and evaluates only afterward with source-time overlap semantics; missing external data returns `not_run/missing_input_paths` (`1d3b160e`, `c4d790f2`, `5b5974b2`).

- **Scope:** Run the production audio pipeline without reference-driven tuning and save
  input/config hashes plus diagnostics/evaluation JSON.
- **Depends on:** E06, F01.
- **Done when:** the run is reproducible and reports success, ambiguity, degeneracy, or
  unavailable-data honestly.

#### F03 — Add optional constrained-Myotis evaluation

- **Status:** complete — an explicit right-angle constraint can be applied to the same extracted measurements through the 8+-microphone planar solver, with separate blind/constrained/ablation reports, provenance, projected-source/unsigned-height evaluation, and signed source RMS only under an explicit half-space sign (`6e0e6df4`, `20f21d2d`, `d2d2c7a3`, `def971ee`, `be7aa289`, `47a46f4b`).

- **Scope:** If a legitimate physical anchor/orientation is available, rerun using that
  explicit constraint and label it separately from the blind result.
- **Depends on:** D05, F02.
- **Done when:** blind and constrained outputs are clearly separated and removing the
  constraint restores the blind ambiguity behavior where applicable.

### Milestone F completion note

Milestone F is implemented on `rewrite/stratified-tdoa-calibration`. The repository
now has a receiver-only Myotis conditioning report, a blind production-audio runner,
and a separately labeled constrained planar rerun with explicit constraint provenance
and ablation on the same extracted TDOA measurements.

Reference geometry/source states are evaluation-only in the blind run. Channel mapping
is applied only when matching the estimate back to the reference. The conditioning
report is computed separately and cannot influence branch selection. Post-hoc source
evaluation preserves the existing source-time-overlap behavior.

For constrained planar results, evaluation reports microphone error, projected-source
error, and unsigned plane-normal height error. A signed 3-D source RMS is emitted only
when a half-space sign is explicitly supplied relative to the reported reference plane
normal. Removing the constraint reruns the planar model and records whether
ambiguity/degeneracy returns.

The actual real Myotis WAV/reference files are not bundled in this repository and are
not available in this execution session. Therefore this milestone does **not** claim a
successful empirical real-Myotis calibration. With no supplied paths the runner
records `not_run/missing_input_paths`; with paths it records hashes, full configuration,
solver diagnostics, evaluation, and success/ambiguity/degeneracy/failure honestly.

As with prior milestones, feature-branch pushes do not trigger the full repository CI
workflow in this environment. Ruff/Ty/Python 3.11–3.14/build verification still
requires a PR or manual workflow dispatch; this note does not claim that matrix ran.

### Milestone G — cleanup and release surface

#### G01 — Remove obsolete Bayesian geometry internals

- **Status:** complete — obsolete Bayesian public exports, solver/initializer modules, and dedicated backend tests were removed; production calibration has a single stratified backend with no hidden fallback (`135f3e9e`, `dcf06b93`, `08e2adbd`, `8daab73f`).

- **Scope:** Delete the old Bayesian solver/initializer and dead internal imports only
  after replacement gates are green.
- **Depends on:** E04, E05.
- **Done when:** tests/imports pass with a single production calibration backend and no
  fallback silently invokes the removed solver.

#### G02 — Clean up CLI options and public API

- **Status:** complete — the public API/CLI exports stratified result/constraint types, explicit dimensional-model selection, subset budgets, deterministic post-selection `wls`/`huber` refinement controls, and planar constraint provenance while rejecting removed Bayesian/motion-prior options.

- **Scope:** Remove/deprecate posterior/motion-prior options, add only necessary
  dimensional-model/subset/refinement diagnostics controls, and update type exports.
- **Depends on:** G01, E06.
- **Done when:** CLI help and API tests expose the stratified terminology only, including
  explicit deterministic refinement mode, evaluation budget, and improvement tolerance controls.

#### G03 — Finalize user-facing docs and examples

- **Status:** complete — README, algorithm/JSON/validation docs, package metadata, and runnable examples now use the final event-driven stratified API and document ambiguity, planar unsigned heights, explicit constraints, and real-Myotis availability semantics (`51f58704`, `081c1d3a`, `4b7dda63`, `7a798961`, `de0d5881`, `61830d32`, `f8371a0c`, `c2f25e92`).

- **Scope:** Update README, algorithm docs, JSON docs, validation instructions, and
  examples to match the implemented solver and ambiguity semantics.
- **Depends on:** G02, F02.
- **Done when:** documented commands run against the final API and no text claims unique
  recovery for the unconstrained cross or signed planar source height.

### Milestone G completion note

Milestone G completes the rewrite release surface. The obsolete Bayesian solver and
initializer are deleted, top-level imports expose stratified terminology only, and the
CLI/API explicitly represent dimensional model, subset budgets, constraints, and the
current no-refinement state.

User-facing documentation and examples now match the event-driven production path.
Planar results are documented as projection plus unsigned height; the conventional
positive-normal representative is never described as uniquely signed truth. Exact
cross geometry remains an ambiguity/degeneracy case unless an explicit physical
constraint is supplied.

Package metadata is bumped to 0.6.0 for the breaking backend/API replacement.

The repository still cannot claim the GitHub Actions Python 3.11-3.14 matrix passed
from feature-branch pushes in this session; a pull request or manual workflow dispatch
is required for that final CI signal.

### Ticket execution order

A practical critical path is:

`T02 -> B01 -> B02 -> B04 -> B05 -> C01 -> C02 -> C03 -> C04 -> C05 -> C06 -> C07 -> C08 -> C09 -> E03`.

In parallel, `T00/T01 -> A06 -> D01-D06` can proceed once the planar math contract is
ready, while fixture/API work `A01-A05` can proceed independently of the generated
minimal solver. Real-data tickets `F01-F03` should stay behind the corresponding
diagnostic and audio-integration gates.

Each ticket should close with its focused tests. If a ticket discovers a new algebraic
or identifiability problem, open a new ticket rather than broadening the current one
until its scope is no longer independently reviewable.
