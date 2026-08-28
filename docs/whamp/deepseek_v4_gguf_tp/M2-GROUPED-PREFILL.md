<!-- markdownlint-disable MD060 -->

# M2 — grouped SM86 expert prefill

Decision: **pass the expert-prefill component gate.** Proceed to dense prefill and the TP4 layer slice. This is not an end-to-end prefill claim.

## Causal sequence

All arms use full 256-expert/top-6 TP4 geometry, raw GGUF weights, Q8_1 activations, vLLM's capture-safe expert alignment, the 230 W / 1650 MHz safety policy, and five exclusive GPU0 trials.

| IQ2 gate/up arm, uniform M256 | Captured kernel | Net incl. alignment | Decision |
|---|---:|---:|---|
| indexed DP4A | 6.303 ms | 6.303 ms | baseline |
| shared WMMA N16/block16 | 7.973 ms | 8.037 ms | reject: padding + shared decode |
| shared MMA N8/block8 | 6.406 ms | 6.469 ms | reject: parity only |
| D2R MMA N8/block8 | **3.931 ms** | **3.995 ms** | keep: 1.56× net |

The winning IQ2 kernel constructs `m16n8k32` A fragments directly from raw IQ2 grid/sign bytes, stages only activation codes, and preserves the indexed path as M≤4/fallback. The grouped Q2_K down kernel folds its two scale nibbles into INT8 A codes and applies the two per-16 min corrections from activation-code sums outside MMA.

## Full grouped expert result

Five trials, 250 warm + 500 measured graph replays, GPU0 only, ≤1 process/sample, max clock 1650 MHz:

| Routing | M | Indexed gate+up+down | Grouped gate+up | Grouped down | One alignment | Grouped total | Net speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 64 | 2.560 ms | 2.438 | 1.454 | 0.063 | 3.954 | 0.65× |
| uniform | 128 | 5.133 | 2.932 | 1.618 | 0.064 | 4.614 | 1.11× |
| uniform | 256 | **10.219** | **3.932** | **2.082** | **0.065** | **6.079** | **1.68×** |
| concentrated | 256 | 8.796 | 2.465 | 1.697 | 0.065 | 4.227 | 2.08× |

At uniform M256, grouped expert work is **0.023745 ms/token/layer**, or **1.021 ms/token across 43 layers**, giving a 979 tok/s expert-only ceiling. The complete 550 tok/s floor allows 1.818 ms/token, leaving approximately 0.797 ms/token for dense, attention, collectives, routing, SwiGLU, and other work. This makes the floor plausible enough to continue; the 700 target and complete prefill remain unproven.

Grouped execution deliberately loses below roughly M128 for uniform routing. Dispatch must therefore retain indexed decode and choose grouped only in the measured prefill regime; the exact crossover remains an empirical runtime policy.

## Correctness, dispatch, and safety

- Final GPU suite: **22/22** IQ2/Q2 tests pass, including independent dequantized references, MMVQ-versus-MMQ class-B windows, block-8 expert mapping, and CUDA Graph replay.
- Compute Sanitizer: grouped tests **2/2**, memcheck 0 errors; racecheck 0 hazards/warnings.
- SM86 device code:
    - IQ2 cubin SHA-256 `a5a27f7808369fbb3a8d64c6b142fe2bd8cd1fe80cf9014804b0e79443ce5327`;
    - Q2 cubin SHA-256 `5a43777e2cdb877dec2f80f7408d4372c72d28dbb59c6eb420d398334a388dea`;
    - both contain `IMMA.16832.S8.S8` in the named grouped kernels.
- Llama.cpp remains intentionally offline during the authorized batch window; an eight-hour restore watchdog is armed. Final-service health is restored at the next deliberate checkpoint.

Evidence: `evidence/m2-grouped-prefill/`.
