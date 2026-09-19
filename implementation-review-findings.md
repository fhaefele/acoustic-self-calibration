# Implementation review findings

Review target: `rewrite/stratified-tdoa-calibration` against
`docs/STRATIFIED_IMPLEMENTATION_PLAN.md`.

- [x] **Public calibration status omits `degenerate`.**
  The planar solver can return `degenerate`, but `CalibrationStatus` does not include it.

- [x] **Production solvers require mic-0, contiguous microphone IDs, and first-eight seed channels.**
  A reference-star basis centered on another microphone is mathematically equivalent but
  rejected. Original microphone IDs should survive internal re-referencing and seed selection.
  Fixed by covariance-aware reference selection, measurement-quality seed ordering, explicit
  original-ID mapping, and forced inclusion of explicitly constrained planar receivers.

- [x] **Optional deterministic measurement-only refinement is not implemented.**
  The implementation plan requires refinement from a valid selected geometry while preserving
  the pre-refinement solution and frozen validation score. Fixed with covariance-aware WLS/Huber
  post-selection refinement, rollback preservation, reduced-gauge optimization, CLI/API controls,
  and JSON pre/post diagnostics.

- [x] **Extra-microphone completion lineage is not preserved in returned hypotheses/JSON.**
  For arrays larger than eight microphones, extra receiver measurements are used to complete
  geometry but are absent from the selected hypothesis's generation/completion/validation masks.

- [x] **Continuous planar ambiguity diagnostics are dropped from the public result/JSON.**
  The metric layer computes continuous nullity and null directions, but callers cannot recover
  those diagnostics from a degenerate planar calibration result.

- [x] **The public hypothesis contract is less structured than the implementation plan specifies.**
  `FullGeometryHypothesis` does not explicitly carry model, seed receiver/event IDs, metric
  branch ID, arrival-gauge information, offset scope, per-state completion status, and
  generating-equation residual diagnostics.

## Verification note

The branch currently has no GitHub Actions run at its reviewed head. The configured CI matrix
therefore remains an outstanding verification signal after these fixes.
