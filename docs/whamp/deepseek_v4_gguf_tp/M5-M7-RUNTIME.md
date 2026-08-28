# M5–M7 — full runtime acceptance

Decision: **M5, M6 functional canaries/NIAH, and M7 performance floors pass.** M6's explicit class-B layer comparison and M8 one-cell DeepSWE pilot (`M8-DEEPSWE.md`) remain required before promotion. The profile is a measured capacity ceiling; its low idle headroom is accepted by Will as normal for a packed TP profile (see `CAPACITY.md` → "Will's headroom decision (2026-08-18)").

## Bring-up

M5 attempt 1 allocated and loaded the exact 21.19 GiB/rank raw plan, then failed during LM-head Q8 preparation because the original repacker expanded codes to a 1,010 MiB INT64 temporary with ~902 MiB free. This was a transient implementation defect, not steady-state capacity.

Whamp/vLLM `3ec20cebe` replaced whole-tensor INT64 expansion with 2,048-row INT32 chunks. Attempt 2 then passed:

- image `sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf`;
- GGUF SHA-256 `ca22ae2f…` and 86,720,111,488 bytes;
- full load in 271.90 s;
- model loading 21.53 GiB/rank;
- consumed weights + non-Torch 22.01 GiB/rank;
- peak activation 0.27 GiB/rank;
- actual CUDA graph pool 0.06 GiB/rank;
- KV cache 0.81 GiB/rank / 154,519 tokens;
- 1.10× concurrency at 140,000 context;
- zero serving-process swap after RAM-gated normalization.

The service reached API readiness with Ampere FlashMLA decode, Triton sparse-prefill/indexer fallback, HIERARCHICAL then PYNCCL collective dispatch, breakable CUDA graphs, and the custom GGUF quant/load method.

## Functional and long-context correctness

The live service returned:

- exact deterministic `GGUF TP READY`;
- a valid automatic `get_weather({"city":"Paris"})` call;
- coherent post-tool continuation;
- exact `NEEDLE-GGUF-842731` retrieval from a 119,730-token prompt, normal stop, 230.02 s wall time;
- zero residual requests/KV and zero swap afterward.

Quick quality scored **27/30 pass@1 and 27/30 pass@3**:

- ToolCall-15: 12/15;
- InstructFollow-15: 15/15.

The prior WNA quick gate was 25/30 pass@1 and 26/30 pass@3. These packs are smoke evidence, not a substitute for M8.

## Matched performance

### Decode

Three warmups plus five measured 512-token length-capped generations:

- **76.6973 wall tok/s mean**;
- 0.0334% CV;
- every measured response generated 512 tokens with `finish_reason=length`.

This exceeds the 58 floor and 70 target. The same inherited WNA speed stack measured 74.98 tok/s, so GGUF-TP is approximately 2.3% faster in this matched single-stream screen.

### Prefill

Three cache-busted ~9K prompts with unique prefixes before filler:

- 548.23, 553.36, 554.07 tok/s;
- **551.89 tok/s mean**.

This clears the 550 floor by only 0.34% and misses the 700 target. It is 37.8% below the WNA speed stack's 887.52 tok/s. Prefill is the leading promotion risk and has essentially no regression margin.

### Concurrency 2

Two simultaneous 512-token requests completed normally:

- 61.16 and 61.05 tok/s per stream;
- 121.86 aggregate tok/s;
- zero swap, zero residual requests/KV.

This covers short requests only, not two 140K contexts.

## Capacity headroom

Idle physical headroom was 101–102 MiB after readiness and 71–73 MiB after long-context JIT. On 2026-08-18 Will reviewed this and accepted it as **normal for a packed vLLM TP profile and not a promotion blocker** — the KV pool is preallocated by design, so low idle free VRAM is the configured steady state, and the profile survived all late-allocation events (load, repacks, long-context JIT, 119,730-token NIAH) with zero swap. The 1 GiB release guard is scoped to dynamically sized profiles and does not apply here. Full decision text and its reopen condition (any OOM at or below operating context): `CAPACITY.md` → "Will's headroom decision (2026-08-18)". The measured profile remains a capacity ceiling: do not raise the operating context without remeasuring.

## Next gates

M6 first requires the pre-registered class-B decoder-layer and final-logit comparison in `M6-LAYER-ORACLE-SPEC.md`. The diagnostic implementations are prepared but have not run on GPU.

M8 is the **one-cell DeepSWE pilot only** per `M8-DEEPSWE.md` (Will 2026-08-18): **approved to execute** GGUF-TP on `superjson-error-stack-serialization` rep0 under plan `sha256:7ac3e4c4…`; compare to reused llama.cpp baseline; pass = Will's closeness judgment. The ≥72-cell multi-seed grid is **cancelled — do not run**. M6 must pass before M8 counts toward promotion.

Evidence: `evidence/m5-m7-runtime/`.
