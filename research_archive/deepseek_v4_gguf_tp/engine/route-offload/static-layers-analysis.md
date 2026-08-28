# DeepSeek V4 GGUF-TP cold-expert offload route-skew report

**Decision: NO-GO** — At least one observed layer requires H>=248 for 99% visit coverage.

The preregistered gate is GO only when every observed layer reaches 99% coverage at H≤224; NO-GO when any requires H≥248.

## Workloads

| Workload | Median H99 | Worst H99 | Full consecutive reuse data |
| --- | ---: | ---: | --- |
| `deepswe-pilot-final-context` | 249 | 249 | yes |
| `deepswe-12task-corpus` | 251 | 251 | yes |

## Per-layer coverage

Full H=1..256 curves are in the adjacent JSON report. Selected points:

| Layer | deepswe-pilot-final-context H99 | deepswe-pilot-final-context cov@224 | deepswe-pilot-final-context cov@248 | deepswe-12task-corpus H99 | deepswe-12task-corpus cov@224 | deepswe-12task-corpus cov@248 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 249 | 0.944179 | 0.988996 | 251 | 0.930343 | 0.985597 |
| 1 | 248 | 0.944734 | 0.990421 | 251 | 0.932088 | 0.985100 |
| 2 | 249 | 0.945778 | 0.989712 | 251 | 0.934172 | 0.985751 |

## Interpretation constraints

- Coverage is per-layer routed-expert visit coverage, not a performance claim.
- Cross-session transfer and consecutive/LRU evidence must be inspected before designing a cache even when the H99 gate says GO.
- Missing consecutive route sequences leave the LRU proxy unverified; histogram coverage alone cannot establish temporal locality.
