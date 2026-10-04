# Stratified TDOA algorithm

## Measurement model

For receiver `r_i`, source event `s_j`, and range

```text
d_ij = ||r_i - s_j||
```

the mic-0 reference-star TDOA is

```text
tau_ij = t_ij - t_0j = (d_ij - d_0j) / c.
```

The implementation uses meter-valued relative arrivals

```text
f_ij = c * tau_ij
```

and event offsets

```text
o_j = -d_0j
d_ij = f_ij - o_j.
```

Offsets may be negative; corrected ranges may not.

## Audio frontend

The production frontend detects discrete transients from smoothed channel-energy
envelopes. Around each event, normalized envelope correlation yields sub-sample lag
candidates for every channel.

Optional temporal tracking can choose a smooth candidate sequence. It is strictly a
correspondence heuristic; the geometry solver receives no source-motion smoothness
factor.

Per-channel arrivals are converted into an independent mic-0 reference-star basis.
Shared-reference covariance is retained in `EventTDOAMeasurements`. Missing or invalid
measurements stay invalid and are never zero-filled.

## Arrival gauge

Literal zero-reference arrivals make one useful linear formulation rank deficient.
The solver can apply event-wise gauge shifts

```text
f'_ij = f_ij + q_j
o'_j  = o_j + q_j
```

without changing corrected ranges. Gauge selection is deterministic and scale-aware.

## Linear offset anchors

Exact anchor cases are implemented for:

- 9 receivers / 5 events in rank-3 geometry,
- 7 receivers / 4 events in rank-2 receiver geometry.

Both solve conditioned linear offset systems after re-gauging.

## Production 7r/6s stage

The production 3-D minimal solver uses 7 receivers and 6 events. A checked-in generated
rank template supplies primary rank-four equations. Runtime enumeration uses a
deterministic Halton start schedule and verifies every retained candidate against all 75
rank-four minors.

Candidates are deduplicated, checked for nonnegative corrected ranges, and assigned
finite-difference Jacobian-rank diagnostics.

`search_stabilized` is a numerical completeness diagnostic, not a symbolic proof that
no isolated real root exists outside the bounded search region.

## Expansion

Minimal hypotheses are expanded to more events through column-space membership of the
receiver-differenced squared-range matrix. Additional microphones are localized from
recovered source states and absolute ranges.

Generation, completion, and validation measurement masks are tracked separately.

## Affine factorization

Corrected ranges produce the compacted cross-Gram matrix

```text
Q = -0.5 * C_m.T @ (d**2) @ C_n.
```

SVD yields rank-3 or rank-2 affine factors and exposes singular values, discarded
energy, and reconstruction error.

## 3-D metric recovery

For rank-3 factors `Q = X Y.T`, the Euclidean upgrade uses

```text
r_i = L x_i
s_j = solve(L.T, b + y_j)
H = L.T L > 0.
```

Receiver equations are linear in the symmetric metric entries and `b`; source
constraints close the remaining metric freedom. Real branches are checked for positive
definiteness and against original corrected ranges.

The solver never clips metric eigenvalues to manufacture a Euclidean solution.

## Planar receivers / 3-D sources

Rank-2 receiver factors use a five-parameter planar metric system. Each source event
then exposes:

- a 2-D projection in the receiver plane,
- unsigned distance from that plane.

Reflecting any individual source event through the receiver plane preserves all ranges.
Therefore source height sign is not identifiable by default. Results carry unsigned
height plus a `height_sign_known` mask.

## Continuous planar ambiguity

Some planar arrays lie on a nontrivial conic. The exact two-line Myotis cross has a
one-dimensional continuous metric nullspace. The solver reports this before candidate
ranking and does not assign a unique right angle.

An explicit physical constraint, such as a known arm angle, can add an independent
metric equation. Constraint type, receiver IDs, value, exactness, and provenance are
explicit inputs. Removing the constraint must restore the unconstrained ambiguity when
the data remain degenerate.

## Robust validation and consensus

Held-out residuals use heteroscedastic scales and Huber scoring. When shared-reference
arrival covariance is present, held-out residual blocks use conditional covariance after
accounting for measurements used to localize the validation source.

Equivalent hypotheses are clustered by receiver-only rigid alignment. Consensus counts
independent minimal subsets rather than duplicate polynomial roots.

## Dimensional-model comparison

Planar and general-3D models use the same deterministic held-out event split. Evidence
combines held-out robust scores with structural conditioning. Near-equal evidence returns
`ambiguous`; training residual is not used to force a winner.

## Coordinate gauge

Geometry is observable only up to rigid transformation/reflection. Canonicalization uses
conditioning-aware receiver choices rather than fixed first microphones. Gauge fixing is
a representation choice, not measured handedness.

## Refinement

Optional deterministic measurement-only refinement runs only after a solved stratified
geometry has been selected. `wls` uses covariance-aware weighted least squares and
`huber` uses the same whitened TDOA residuals with a fixed robust loss. The geometry is
optimized in a reduced rigid gauge, planar source heights remain unsigned, and explicit
planar angle anchors are held fixed. Refinement never participates in branch/model selection
and never replaces the frozen held-out validation score. The pre-refinement calibration is
retained for rollback. `refinement_max_nfev` and
`refinement_improvement_tolerance` make the optimizer budget and acceptance threshold
explicit. JSON records those controls with pre/post coordinates, objective values,
termination, TDOA RMS, and whether the candidate refinement was accepted.

## Limitations

The current release assumes synchronized channels and known speed of sound. Reverberation,
incorrect direct-path peaks, channel-response mismatch, asynchronous clocks, weak source
trajectories, and degenerate receiver layouts can reduce or remove identifiability.
Those conditions produce weak/ambiguous/degenerate outcomes rather than a hidden fallback
estimator.
