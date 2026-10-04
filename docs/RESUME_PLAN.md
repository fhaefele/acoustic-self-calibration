# Completion resume sequence

Use [AGENT_HANDOFF.md](AGENT_HANDOFF.md), the authoritative
[completion plan](SELF_CALIBRATION_COMPLETION_PLAN.md), and the measured
[progress record](COMPLETION_PROGRESS.md). Preserve uncommitted work.

1. Inspect every archived benchmark's expected IDs, completion flag, code hashes,
   status/error/coverage counts and missing cases. Resume only matching code and
   manifests. Keep all failed development/evaluation outcomes.
2. Diagnose any exact geometry failure before frontend changes. For audio failures,
   separate association/delay extraction from completion and uncertainty. Preserve
   frozen held-out evidence and do not supply truth to calibration.
3. Complete M6's full exact/noisy development matrix; expand audio from 8/12-channel
   20-call cases to all preregistered dimensions when the initial cases pass.
   Run the reserved evaluation seeds without tuning on them.
4. Complete M7 direct-sound stress characterization. Record partial coverage,
   inaccurate solved outcomes and non-success reasons separately; no echo work.
5. Myotis's public strict acceptance still requires resolved weak identification,
   honest event coverage and verified construction evidence. Do not relabel weak
   geometry as solved or assume that 12 channels imply uniqueness.
6. Finish M8 evidence/plots/docs and verify Ruff, Ty, full pytest, build, CLI and
   Python 3.11–3.14 CI. Report missing/unmet acceptance explicitly in PR #8.
7. Commit and push coherent checked changes as authorized. Do not merge the PR.

Commands and manifests are documented in `VALIDATION.md`; archived reports are
under `benchmarks/results/`. A filtered run is an explicitly partial result.
