# Archived chirp evidence

Rows retain all failures and expected case IDs. Stress characterization is separate from recovery acceptance.

## chirp_development_exact.json.gz

Completed 732/732 expected rows; complete=True.
Recorded source revision: `d1b611da1cdcf1ba11a0f8a6a7c433bc489815a6`.

Standard recovery: 552/552 pass.

| Class | Status | Count |
| --- | --- | ---: |
| negative | degenerate | 56 |
| negative | failed | 4 |
| standard | solved | 552 |
| stress | solved | 87 |
| stress | weakly_identified | 33 |

Unsuccessful required cases:

| Case | Status | Mic RMS (m) | Source RMS (m) | Unresolved |
| --- | --- | ---: | ---: | ---: |
| `thin-grid/8/20/2.0/10/exact` | failed | unavailable | unavailable | 20 |
| `thin-grid/8/20/4.0/10/exact` | failed | unavailable | unavailable | 20 |
| `thin-grid/8/40/2.0/10/exact` | failed | unavailable | unavailable | 40 |
| `thin-grid/8/40/4.0/10/exact` | failed | unavailable | unavailable | 40 |

## chirp_development_arrival_noise.json.gz

Completed 732/732 expected rows; complete=True.
Recorded source revision: `e14ffa4fff82f1fb7e486056c012bdf316191338`.

Standard recovery: 525/552 pass.

| Class | Status | Count |
| --- | --- | ---: |
| negative | degenerate | 60 |
| standard | failed | 11 |
| standard | solved | 539 |
| standard | weakly_identified | 2 |
| stress | failed | 54 |
| stress | solved | 29 |
| stress | weakly_identified | 37 |

Unsuccessful required cases:

| Case | Status | Mic RMS (m) | Source RMS (m) | Unresolved |
| --- | --- | ---: | ---: | ---: |
| `cross60/8/20/2.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross60/8/20/2.0/11/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross75/8/20/2.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross75/8/20/2.0/12/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross75/8/40/2.0/10/arrival-noise` | solved | 0.000748762 | 0.0093822 | 5 |
| `cross90/12/20/4.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross90/8/40/2.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `cross90/8/40/4.0/10/arrival-noise` | solved | 0.000587962 | 0.00700722 | 5 |
| `generic/24/20/2.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `grid-equal/16/20/4.0/12/arrival-noise` | solved | 0.000402248 | 0.00370413 | 6 |
| `grid-equal/24/20/4.0/12/arrival-noise` | failed | unavailable | unavailable | 20 |
| `grid-equal/24/40/2.0/11/arrival-noise` | solved | 0.000630429 | 0.00544519 | 1 |
| `grid-equal/24/40/2.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `grid-equal/24/40/4.0/11/arrival-noise` | solved | 0.000209619 | 0.00400065 | 1 |
| `grid-equal/9/40/4.0/11/arrival-noise` | solved | 0.000605966 | 0.00552605 | 6 |
| `grid-free/16/20/2.0/10/arrival-noise` | solved | 0.00111981 | 0.0070591 | 2 |
| `grid-free/16/20/2.0/11/arrival-noise` | solved | 0.000393439 | 0.00570317 | 2 |
| `grid-free/16/20/2.0/12/arrival-noise` | solved | 0.000181619 | 0.00371028 | 1 |
| `grid-free/24/40/2.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `grid-free/24/40/4.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `grid-free/9/40/4.0/11/arrival-noise` | solved | 0.000619804 | 0.00629205 | 8 |
| `star/16/40/2.0/12/arrival-noise` | weakly_identified | 0.00115638 | 0.00686011 | 0 |
| `star/16/40/4.0/12/arrival-noise` | weakly_identified | 0.00107062 | 0.00686463 | 0 |
| `t60/8/20/2.0/10/arrival-noise` | solved | 0.00475416 | 0.0253275 | 1 |
| `t75/8/20/2.0/10/arrival-noise` | solved | 0.00464335 | 0.0240615 | 1 |
| `t75/8/20/4.0/12/arrival-noise` | solved | 0.00369686 | 0.0264579 | 1 |
| `t90/8/40/4.0/12/arrival-noise` | solved | 0.00158286 | 0.0114642 | 1 |

## chirp_development_audio_initial.json.gz

Completed 96/96 expected rows; complete=True.
Recorded source revision: `e14ffa4fff82f1fb7e486056c012bdf316191338`.

Standard recovery: 94/96 pass.

| Class | Status | Count |
| --- | --- | ---: |
| standard | failed | 2 |
| standard | solved | 94 |

Unsuccessful required cases:

| Case | Status | Mic RMS (m) | Source RMS (m) | Unresolved |
| --- | --- | ---: | ---: | ---: |
| `room-rectangular/12/20/2.0/12/float-40db` | failed | unavailable | unavailable | 20 |
| `room-rectangular/12/20/2.0/12/pcm-40db` | failed | unavailable | unavailable | 20 |

## chirp_arrival_noise_failure_rechecks.json

Completed 27/27 expected rows; complete=True.
Recorded source revision: `69810e78265363e22a901b44ee61fd52a45d77fc`.

Standard recovery: 0/27 pass.

| Class | Status | Count |
| --- | --- | ---: |
| standard | failed | 11 |
| standard | weakly_identified | 16 |

Unsuccessful required cases:

| Case | Status | Mic RMS (m) | Source RMS (m) | Unresolved |
| --- | --- | ---: | ---: | ---: |
| `cross60/8/20/2.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross60/8/20/2.0/11/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross75/8/20/2.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross75/8/20/2.0/12/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross75/8/40/2.0/10/arrival-noise` | weakly_identified | 0.000748762 | 0.0093822 | 5 |
| `cross90/12/20/4.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `cross90/8/40/2.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `cross90/8/40/4.0/10/arrival-noise` | weakly_identified | 0.000587962 | 0.00700722 | 5 |
| `generic/24/20/2.0/10/arrival-noise` | failed | unavailable | unavailable | 20 |
| `grid-equal/16/20/4.0/12/arrival-noise` | weakly_identified | 0.000402236 | 0.00370408 | 6 |
| `grid-equal/24/20/4.0/12/arrival-noise` | failed | unavailable | unavailable | 20 |
| `grid-equal/24/40/2.0/11/arrival-noise` | weakly_identified | 0.000630429 | 0.00544519 | 1 |
| `grid-equal/24/40/2.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `grid-equal/24/40/4.0/11/arrival-noise` | weakly_identified | 0.000209619 | 0.00400065 | 1 |
| `grid-equal/9/40/4.0/11/arrival-noise` | weakly_identified | 0.000605966 | 0.00552605 | 6 |
| `grid-free/16/20/2.0/10/arrival-noise` | weakly_identified | 0.00111981 | 0.0070591 | 2 |
| `grid-free/16/20/2.0/11/arrival-noise` | weakly_identified | 0.000393438 | 0.00570317 | 2 |
| `grid-free/16/20/2.0/12/arrival-noise` | weakly_identified | 0.000181617 | 0.00371028 | 1 |
| `grid-free/24/40/2.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `grid-free/24/40/4.0/12/arrival-noise` | failed | unavailable | unavailable | 40 |
| `grid-free/9/40/4.0/11/arrival-noise` | weakly_identified | 0.000619804 | 0.00629205 | 8 |
| `star/16/40/2.0/12/arrival-noise` | weakly_identified | 0.00115638 | 0.00686011 | 0 |
| `star/16/40/4.0/12/arrival-noise` | weakly_identified | 0.00107062 | 0.00686463 | 0 |
| `t60/8/20/2.0/10/arrival-noise` | weakly_identified | 0.00475416 | 0.0253275 | 1 |
| `t75/8/20/2.0/10/arrival-noise` | weakly_identified | 0.00464335 | 0.0240615 | 1 |
| `t75/8/20/4.0/12/arrival-noise` | weakly_identified | 0.00369686 | 0.0264579 | 1 |
| `t90/8/40/4.0/12/arrival-noise` | weakly_identified | 0.00158286 | 0.0114642 | 1 |

