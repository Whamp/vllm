# M2 — Q8_0 dense prefill screen

Decision: **pass the dense-prefill component gate.** The combined changed-component budget supports proceeding to the TP4 slice but does not itself prove 550 tok/s.

## Workload binding

The measured promotion-style prompt is 8,984 tokens and the inherited profile fixes `max_num_batched_tokens=256`. A one-request cache-busted prefill therefore consists of 35 full M256 chunks plus one 24-token tail: **99.7% of tokens execute at M256**. The geometric M={16,32,64,128,256} sweep captures the full scheduler domain; M256 is the representative throughput point and smaller M values characterize the tail lose-condition.

## Exclusive RTX 3090 results

Five trials per shape and M, 250 warm + 500 measured graph replays, GPU0 only, ≤1 process/sample, max clock 1650 MHz. Every prepared weight+scale payload remains byte-equal to raw Q8_0.

At M256:

| Component | Measured contribution |
|---|---:|
| five ordinary Q8 projections ×43 layers | **0.06494 ms/token** |
| grouped-diagonal `wo_a` ×43 layers | **0.03179 ms/token** |
| vocabulary head once | **0.00664 ms/token** |
| **Q8 dense total** | **0.10337 ms/token** |
| grouped routed experts ×43 layers | **1.02105 ms/token** |
| **all changed components** | **1.12442 ms/token** |

The 550 tok/s floor allows 1.81818 ms/token, leaving **0.69376 ms/token** for inherited attention, collectives, routing, SwiGLU, norms, cache work, scheduling, and gaps. The 700 target allows 1.42857 ms/token and leaves only 0.30415 ms/token; it remains uncertain.

At M128, grouped experts plus dense components consume approximately 1.664 ms/token, leaving only ~0.154 ms/token at the floor. This is a real lose-condition for sustained small chunks, but the bound promotion workload pays it only in one short tail. Mixed-request scheduling must be re-evaluated later rather than inheriting the single-request conclusion.

## Scope and uncertainty

These are exclusive microbenchmarks summed by semantic execution count. They do not capture overlap, router/SwiGLU materialization, real TP collectives, attention, or graph interactions. The next gate is the TP4 graph-captured layer slice with real gate/up→SwiGLU→down dataflow and real all-reduce. Only that slice may project complete decode/prefill.

Evidence: `evidence/m2-q8-prefill/`.
