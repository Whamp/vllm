# CAPACITY.md — M1 per-rank GPU residency and context table

Status: M1 model (measured anchors + exact artifact bytes + labeled estimates).
Winner is **INSTANCE VALUE** — M5 measures actual residency. Formulas are
explicit so every delta can be recomputed.

## Anchors

- Physical RTX 3090 VRAM: 24 GiB nameplate; measured usable executor/driver
  envelope on the WNA16 service: **94.242188 GiB aggregate = 23.560547
  GiB/rank**.
- Verified GGUF payload: 80.7594 GiB (1,328 tensors, no MTP), exact family
  split in `TP-MAPPING.md` / `evidence/gguf-inventory.json`.
- Post-TP registered weights (exact raw-byte basis): **21.1893 GiB/rank**.
- KV density (measured WNA16 grouped-cache anchor): **5.832 KiB/token/rank**
  (third review independently recomputed); cache admission is nonlinear,
  so linear conversions are decision estimates, not runtime promises.
- Comparable WNA16-quality registered residency: 20.69 GiB/rank at 156K;
  promoted Marlin profile reached 230,144 with 1.28 GiB KV and ~93 MiB
  physical free. The GGUF adds **~0.50 GiB/rank** weights.

## Base profile (design target, no BF16 wo_a cache)

| Bucket | GiB/rank | Evidence |
|---|---:|---|
| registered model weights | 21.1893 | exact TP-MAPPING table; Q8_0/SoA repacks byte-neutral before tile padding |
| non-Torch CUDA/NCCL/JIT state | 0.5305 | measured WNA16 156K residency audit (comparable; remeasure M5) |
| CUDA graphs | 0.19 | conservative measured-stack anchor |
| peak activation | 0.35 | measured WNA16 startup profile |
| **fixed subtotal** | **22.2598** | estimate anchored to exact weights |
| usable envelope | 23.5605 | measured |
| **left for KV + allocator/headroom** | **1.3007** | subtraction |
| 140K KV @5.832 KiB/tok | 0.7786 | linear estimate |
| residual allocator/physical headroom | **0.5221** | tight; below normal 1 GiB release guard |

Point estimate from the direct WNA16 delta: 230,144 tokens −
(0.50 GiB / 5.832 KiB/token) ≈ **140–142K**, consistent with PLAN §10's
139.1K after graph-pool differences. **M1 gate: pass narrowly** for the plan's
≥140K threshold, with a material caveat: it likely cannot also satisfy the
normal 1 GiB physical-headroom release margin. M5 must choose an operating
context below the measured ceiling if sustained-agent safety requires that
margin.

## Sensitivity / lose conditions

| Delta | GiB/rank | Approx context impact | Decision |
|---|---:|---:|---|
| wo_a BF16 cache | +0.672 | −100K to −120K (measured nonlinear precedent ~−92K) | **fatal/rejected**; WOA-DESIGN Marlin-diagonal is mandatory |
| router F16→fp32 weights | +0.084 | −~15K | only if class-B tie-break window fails; risks <140K |
| token_embd fp32 upcast | +0.247 | −~44K | reject; bf16 cast + window |
| indexer/compressor fp32 upcast | +~0.90 | −~160K | impossible; bf16 fast path + class-B window mandatory |
| Marlin tile padding | unknown | M2/M4 measure; budget ≤0.05 GiB/rank (~9K) | **SOURCE GAP** — fail M1/M2 gate if larger without compensating lever |
| aligned-SoA routed repack | 0 | 0 | hard byte-neutral contract |
| 1 GiB physical headroom | reserves extra ~0.48 over base residual | max context only ~50–60K by linear model | not compatible with 140K unless fixed/runtime state shrinks |

## Will's M1 capacity decision

Will accepted the measured **140–142K on-GPU context floor with approximately 0.52 GiB projected physical headroom** as this service's initial contract on 2026-08-17. This resolves PLAN §12.4 and permits M5 bring-up at that floor. The acceptance does not convert the estimate into a measurement or waive M5's residency falsifier; larger-context work remains a separate follow-up if the engine passes promotion gates.

## Will's headroom decision (2026-08-18)

Will accepted the measured M5 profile (71–73 MiB idle physical headroom after
long-context JIT) as **normal and release-acceptable for this packed TP
profile**. Rationale: vLLM preallocates the KV pool at startup
(`gpu-memory-utilization 0.98` by design), so low idle free VRAM is the
configured steady state, not creeping exhaustion; the profile already survived
the late-allocation events (full load, Marlin repacks, long-context JIT,
119,730-token NIAH) with zero swap. The **1 GiB physical-headroom release
guard does not apply to deliberately packed TP profiles**; it remains in force
for dynamically sized profiles. Release evidence for this engine is instead:
zero serving-process swap, verify-stress-class boundary tests (including
tool-prefill spikes) passing at the operating context, and stable long-context
runs. **The low-headroom warning is not a promotion blocker at M9.** Any OOM
at or below the operating context reopens this decision.

## M1 conclusion

- **Capacity gate outcome:** ≥140K is plausible but tight on exact weight
  accounting; 430K remains llama.cpp-exclusive. No safe room exists for
  fallback caches or broad fp32 upcasts.
- **Implementation consequences:** int8 Marlin-diagonal wo_a is on the
  critical path; indexer/compressor must take the bf16 merged path; every
  loader allocation and tile pad is reported per rank; runtime starts at a
  conservative explicit context and auto-fit is not trusted (known 1M-profile
  OOM history).
- **Falsifier:** M5 measured registered weights + fixed runtime exceed
  22.78 GiB/rank before KV (leaving <0.78 GiB) → 140K cannot fit; stop or
  return to Will with a named/sized reclaim lever — do not hide it with
  throughput-killing CPU weight offload.
