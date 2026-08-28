<!-- markdownlint-disable MD060 -->

# M2 — indexed-expert prefill falsifier

Decision: **current indexed decode kernels cannot satisfy prefill. Grouped/MMA execution is mandatory.**

M0 captured decode only, so it did not provide a prefill M distribution. The inherited single-request scheduler caps `max_num_batched_tokens=256`; this screen therefore uses geometric M={16,32,64,128,256} and will later bind the final prefill gate to scheduler-observed chunks. Full 256-expert resident geometry, top-6, and exact TP4 projection shapes are used.

Two deterministic routing boundaries separate reuse effects:

- `uniform`: cyclic IDs distribute assignments across all 256 experts;
- `concentrated`: every token selects the same six experts, maximizing cache/reuse opportunity.

Each result captures shared BF16→Q8_1 quantization plus native indexed gate/up and down. It intentionally omits SwiGLU, routing, dense, attention, and collectives; therefore it is a lower bound on layer time.

## Exclusive RTX 3090 result

Five trials, 250 warm + 500 measured graph replays, GPU0 only, ≤1 process/sample, max clock 1650 MHz, canonical final zero-swap:

| Routing | M | Expert ms/token/layer | 43-layer expert cost | Expert-only ceiling |
|---|---:|---:|---:|---:|
| uniform | 16 | 0.03860 | 1.660 ms/token | 602 tok/s |
| uniform | 32 | 0.04070 | 1.750 | 571 |
| uniform | 64 | 0.04002 | 1.721 | 581 |
| uniform | 128 | 0.04005 | 1.722 | 581 |
| uniform | 256 | **0.04008** | **1.723** | **580** |
| concentrated | 256 | 0.03450 | 1.483 | 674 |

The project floor of 550 tok/s permits 1.818 ms/token for the complete model. Uniform M=256 leaves only 0.095 ms/token for all non-expert work; even the concentrated best boundary leaves 0.335 ms/token. Both are impossible given measured dense and inherited attention/collective work. This directly gates the next move: compact tokens by expert, stage raw IQ2_XXS/Q2_K tiles once per expert token group, and use SM86 INT8 MMA/DP4A reuse. The indexed kernels remain the M≤4 decode path and correctness fallback.

Evidence: `evidence/m2-expert-prefill-baseline/`.
