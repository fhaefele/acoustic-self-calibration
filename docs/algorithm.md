# Algorithm

## State

For `M` stationary microphones and `T` source frames, the core unknowns are microphone positions `m_i in R^3` and source states `s_t in R^3`. Optional nuisance variables include per-channel clock offset, linear clock drift, and sound speed.

## Pairwise TDOA factors

For oriented pair `(a, b)`:

```text
h_ab,t = (||s_t-m_b|| - ||s_t-m_a||)/c
         + (offset_b-offset_a)
         + (drift_b-drift_a)*(t-t_center)
```

The measured TDOA has a per-observation standard deviation derived from the GCC-PHAT confidence score. The MAP problem uses standardized residuals. A Cauchy loss is the default to reduce sensitivity to wrong peaks and multipath outliers.

## Redundant measurement graph

A single reference microphone is mathematically sufficient but statistically brittle: one poor reference channel contaminates every TDOA. The default graph keeps the full mic-0 star required by the initializer and adds a second reference set, providing redundancy without the cost of all `M(M-1)/2` pairs.

## Geometric gauge

TDOAs are invariant to a global rigid transform. The optimizer uses a canonical gauge:

- mic 0 at `(0,0,0)`,
- mic 1 on `+x`,
- mic 2 in the `+xy` half-plane,
- mic 3 on the `+z` side.

That removes the six rigid-body degrees of freedom and fixes a handedness convention.

## Low-rank initialization

Reference-channel range differences are converted into a cross-distance model. Unknown source-to-reference ranges are optimized so the doubly centered squared-distance matrix is approximately rank three. A rank-3 factorization then gives microphone/source coordinates up to an affine ambiguity, which is resolved by fitting the cross distances.

This is substantially more stable than starting the nonlinear optimizer from arbitrary random geometry.

## Motion prior

A Gaussian constant-velocity prior penalizes changes in source velocity:

```text
v_t = (s_t - s_(t-1)) / dt_t
r_motion = (v_(t+1) - v_t) / sigma_dv
```

The prior is optional. It regularizes noisy audio while still allowing arbitrary smooth 3-D motion.

## MAP optimization

The factor graph is solved by sparse nonlinear least squares with an analytic sparse Jacobian. In Gaussian mode this is exactly MAP for Gaussian factors. In robust mode, iteratively reweighted least squares (IRLS) uses Cauchy data weights `1 / (1 + r^2)` on standardized TDOA residuals. Motion and nuisance-parameter priors remain ordinary Gaussian quadratic factors and are never robustified.

## Uncertainty

After MAP convergence, a local Laplace approximation uses `J^T J` as the information matrix. Source-state blocks are marginalized with a Schur complement before reporting uncertainty for microphone positions and optional global nuisance variables.

## Scale and sound speed

With unknown geometry, TDOA alone does not identify both scene scale and sound speed. Scaling all distances and `c` together leaves time differences unchanged. Therefore sound-speed estimation requires at least one known metric microphone baseline.

## Practical limitations

The present renderer and likelihood assume a dominant direct path. Real deployments need stronger handling for reverberation, occlusion, channel-response mismatch, and asynchronous clocks. Posterior covariance is local and should not be mistaken for a guarantee when the problem is multi-modal or poorly initialized.
