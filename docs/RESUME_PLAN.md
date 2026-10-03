# Resume plan (2026-09-24)

Execute per `docs/AGENT_HANDOFF.md`. Preserve uncommitted changes. Never use truth for selection; never loosen gates.

## Tracks

**A. Benchmarks (background, `OPENBLAS_NUM_THREADS=1`, logs in `/tmp/opencode/asc/`)**
- Planar exact + noisy (events 20/40, seeds 0-2, both layouts/spans); room exact + noisy (8/16 mics).
- Fresh seeds 3-5 for stars. Pass bar: 0 crosses `solved`; stars solved; report weak/failed counts.

**B. Audio T-004 (after seed-family fix)**
- `test_stratified_audio_spec`, `test_stratified_audio_8mic`, `test_stratified_audio_large_arrays`, `test_stratified_wav`, `test_end_to_end_audio`.
- Fix failures in code/selection only; gates stay 0.15/0.18/0.22 m, 60 µs.

**C. Foreground fixes/tickets**
1. Make `_seed_event_families` prefix-stable; widen start-stability tests.
2. T-006: Myotis constrained/unconstrained, IDs, unsigned heights.
3. T-007: model selection; add expansion membership-constraint error test.
4. T-008: ruff, ty, focused suite, CLI smoke, build; no debug artifacts (none found).
5. T-009: CI draft-trigger sanity, py matrix where possible, update stale notes; keep PR draft while gates fail.

**D. Commits** (only after checks): seed/stability; planar gates+half-space; general-3D polish; fixtures; CI/notes; handoff results update.

## Final report
What passed / failed (numbers) / uncertain (local CI matrix, fresh-seed rates).
