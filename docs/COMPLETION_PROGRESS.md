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
