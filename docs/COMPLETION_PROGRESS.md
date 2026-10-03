# Completion evidence

Implementation plan: [SELF_CALIBRATION_COMPLETION_PLAN.md](SELF_CALIBRATION_COMPLETION_PLAN.md).
Starting revision: `31b64b8`. Each entry records completed checks; an absent entry
does not mean its milestone is complete.

## M0 — Contract and identifiability

The executable contract is `benchmarks/contract.json`. The four-angle counterexample
in `tests/test_cross_identifiability.py` verifies all 492 ranges with one fixed
alternative array per angle and strictly one-sided sources.

The committed Myotis files have a hash/provenance record in `data/myotis/provenance.json`.
The import commit labels the WAV real; the later reference-frame note calls it
rendered. Original acquisition evidence is unavailable in this repository, so
recorded-versus-derived provenance and the independent construction angle are
**unverified**. These labels are not inferred from matching ground truth.

Checks: cross counterexample and existing Myotis geometry/planar metric tests;
Ruff format/lint. The evaluation reference has not been changed.

## M1 — Chirp fixtures and source metrics

`chirps.py` adds windowed linear/logarithmic sweeps, alternating directions,
call-to-call variation, rapid schedules, ultrasonic bands, and signal-relative
noise. Legacy noise-pulse fixtures retain their original renderer and seeds.
`benchmarking.py` reports source errors, one-to-one matching, and coverage;
missing events and wrong sources prevent a success result.

Checks: 5 chirp/benchmark tests passed, including an independent static-path delay
oracle. Ruff and Ty passed. The seed-10 exact planar chirp case recovered all
20 source states and geometry to numerical precision. Audio accuracy remains a
later gate; this exact measurement test is not an audio acceptance claim.

## M2 — Public common-side assumption

Audio, WAV and CLI accept the declared common-side assumption and reject its use
with the spatial model. JSON distinguishes declared orientation from unsigned
representatives. Finite events keep their sign mask when other events are missing;
refinement is followed by the same physical orientation check. Strictly on-plane
events use the existing scale-relative 1e-12 tolerance and reject the strict-side
input. This assumption cannot detect acoustically invisible side crossings.

Checks: 26 API/CLI/export/planar tests passed; Ruff and Ty passed. A clean seed-10
chirp trial achieved 1.3 mm microphone and 8.2 mm source RMS, but was classified
`weakly_identified`, so it does **not** pass the standard acceptance gate. The
noise/conditioning classification remains an M6 investigation.

## M2a — Structured construction implementation

`ArrayConfiguration` is shared by audio/WAV and CLI `--array-config`. It records
membership, directed rays, exact angle, grid labels, independent optional equal
spacing relationships, and provenance. There are no metric pitches or square-cell
assumptions. Separated arm endpoints support arrays without a junction microphone.
Non-right metric recovery enumerates the signed feasible metric roots; noisy
full-rank equations are fitted under an exact nonlinear constraint.

Structured polishing uses reduced line/grid coordinates at every iterate. Unknown
spacings remain free; local uncertainty is evaluated within that parameter space.
Unknown-angle two-arm inputs retain a metric-family diagnosis, and thin two-row
grids do not claim unique row separation. Frozen pre-polish validation is retained.

Checks: 16 focused tests passed, including direct 60/75/90-degree cross/T chirps,
9-channel uneven/equal grid chirps through WAV, and angle/two-row ablations.
All tested recovery scenes met 5/10 cm and complete-event coverage. Ruff/Ty passed.
The broader preregistered channel/event/noise matrix remains an M6/M8 gate;
these focused checks alone do not establish full acceptance.

## M3 — Microphone-relative output frame

The public audio/WAV result defaults to microphone ID 0 as origin (or accepts
`--output-origin-microphone`). Baselines use distance and ID tie breaks, so the
first three channels may be collinear. Declared common-side sources occupy z>0;
planar microphones occupy z=0. JSON includes origin/axis IDs, units and the rigid
transform. Final, selected-candidate and refinement-audit scenes receive that
same transform; an audit origin can differ after refinement and is labeled.

Source-state timestamps now represent inferred emission features rather than
receiver arrivals. Receiver event times remain in the measurements section.
This distinction prevents propagation delay from shifting reference comparisons.

Checks: public-frame and numerical frame tests cover origin tolerance, unchanged
TDOAs, rigid-motion/channel-permutation invariance and exported time semantics.
Export/evaluation fixtures now declare the same emission-time basis explicitly.
