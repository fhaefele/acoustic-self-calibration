# Algorithm

## Overview

The calibration pipeline is event driven. It does not assume that every uniformly
spaced audio frame contains a usable source observation.

For a synchronized multichannel recording it performs:

1. transient-event detection,
2. multi-hypothesis inter-channel delay extraction,
3. temporally coherent delay-track selection,
4. cycle-consistent pairwise TDOA construction,
5. low-rank geometry initialization,
6. joint Bayesian/MAP refinement of microphones and one source state per event,
7. optional Laplace uncertainty estimation.

The separation between acoustic correspondence and geometry estimation is important
for pulsed sources such as clicks, chirps, claps, and echolocation calls. Silence,
reverberant tails, and arbitrary frame centers are not source states.

## Event detection

Let `x[n, i]` be the synchronized waveform for microphone `i`. A short moving-average
energy envelope is computed for every channel. Peaks are found subject to a minimum
time separation and a prominence threshold relative to the strongest event on that
channel.

If no event channel is specified, all channels are scored and the channel with the
largest summed retained-event prominence is used as the event-time reference. This
reference only defines the recording timestamps used for source states; it does not
fix a geometric microphone reference.

The detector therefore outputs discrete sample indices

```text
n_0, n_1, ..., n_(T-1)
```

and event times

```text
t_k = n_k / f_s.
```

## Multi-hypothesis arrival-delay association

For each detected event and each non-reference channel, the frontend correlates a
short smoothed-energy template from the event channel against a surrounding search
region. The search region is wide enough to contain the configured maximum acoustic
inter-channel delay.

Instead of taking only the strongest correlation peak, the frontend keeps several
local peak candidates. This matters for repeated pulses, neighbouring calls,
periodic waveforms, and reflected arrivals where the locally strongest peak need not
be the direct-path correspondence.

For one channel, event `k` therefore has candidates

```text
{ (d_kj, q_kj) }
```

where `d_kj` is a candidate arrival delay and `q_kj` is its normalized correlation
confidence.

A dynamic-programming track chooses one candidate per event. The path cost combines
negative log correlation confidence with a temporal transition penalty on changes in
arrival delay. The transition scale grows with the elapsed time between events, so a
moving source may change delay more between widely separated calls than between
closely spaced calls.

This is a correspondence prior, not a geometric trajectory fit. It is applied before
the 3-D solver and is intended to reject implausible jumps between competing acoustic
peaks.

## Cycle-consistent TDOAs

The selected arrival delay of microphone `i` for event `k` is `a_ki`. Pairwise TDOAs
are derived from these arrival delays:

```text
tau_k,(i,j) = a_kj - a_ki.
```

Because every pair is derived from the same per-channel arrival vector, cycle
consistency is exact. For example,

```text
tau_(a,b) + tau_(b,c) = tau_(a,c).
```

This avoids feeding mutually contradictory independently selected pairwise peaks to
the geometry solver.

The default measurement graph is the microphone-0 reference star. Redundant and all
pair graphs remain available, but derived pair measurements are statistically
correlated because they share channel-arrival estimates.

## State

For `M` stationary microphones and `T` detected source events, the core unknowns are
microphone positions `m_i in R^3` and event source positions `s_k in R^3`. Optional
nuisance variables include per-channel clock offset, linear clock drift, and sound
speed.

One source state corresponds to one detected acoustic emission. There is no source
state for silence between events.

## Pairwise TDOA factors

For oriented microphone pair `(a, b)` and event `k`:

```text
h_ab,k = (||s_k-m_b|| - ||s_k-m_a||)/c
         + (offset_b-offset_a)
         + (drift_b-drift_a)*(t_k-t_center)
```

The measured TDOA has a per-observation standard deviation derived from the selected
correlation confidence. The MAP problem uses standardized residuals. A Cauchy robust
likelihood is the default to reduce sensitivity to remaining wrong associations and
multipath outliers.

## Geometric gauge

TDOAs are invariant to a global rigid transform. The optimizer uses a canonical
gauge:

- mic 0 at `(0,0,0)`,
- mic 1 on `+x`,
- mic 2 in the `+xy` half-plane,
- mic 3 on the `+z` side.

That removes the six rigid-body degrees of freedom and fixes a handedness convention.

## Low-rank initialization

Reference-channel range differences are converted into a cross-distance model.
Unknown source-to-reference ranges are optimized so the doubly centered squared-
distance matrix is approximately rank three. A rank-3 factorization then gives
microphone/source coordinates up to an affine ambiguity, which is resolved by fitting
the recovered cross distances.

This gives the nonlinear MAP optimizer a geometry-aware initial point without
requiring known microphone coordinates.

## Motion prior

A Gaussian constant-velocity prior penalizes changes in source velocity between
successive detected events:

```text
v_k = (s_k - s_(k-1)) / dt_k
r_motion = (v_(k+1) - v_k) / sigma_dv
```

The prior uses the actual irregular event intervals. It is optional and regularizes
noisy event localization while allowing arbitrary smooth 3-D motion.

## MAP optimization

The factor graph is solved by sparse nonlinear least squares. In Gaussian mode this
is the MAP solution for Gaussian factors. In robust mode, iteratively reweighted
least squares uses Cauchy data weights

```text
1 / (1 + r^2)
```

on standardized TDOA residuals. Motion and nuisance-parameter priors remain ordinary
Gaussian quadratic factors and are not robustified.

## Uncertainty

After MAP convergence, a local Laplace approximation uses `J^T J` as the information
matrix. Source-state blocks are marginalized with a Schur complement before reporting
uncertainty for microphone positions and optional global nuisance variables.

## Scale and sound speed

With unknown geometry, TDOA alone does not identify both scene scale and sound speed.
Scaling all distances and `c` together leaves time differences unchanged. Therefore
sound-speed estimation requires at least one known metric microphone baseline.

## Reference evaluation

Reference data is not used by calibration. It is evaluation-only.

A single rigid transform is fit from estimated microphones to reference microphones.
That transform is applied unchanged to the estimated source states. Reference source
positions are interpolated to detected event times over the time interval where
reference and estimate overlap.

## Practical limitations

The current frontend is a free-field/direct-path-oriented baseline. It retains
multiple delay hypotheses and uses temporal continuity, but it does not yet perform a
full joint multipath hypothesis search across all microphones. Strong reverberation,
occlusion, channel-response mismatch, overlapping sources, or missed/extra event
associations can still cause failure.

Posterior covariance is local and should not be interpreted as a guarantee when the
correspondence or geometry posterior is multimodal.
