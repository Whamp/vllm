<!-- markdownlint-disable MD060 -->

# FP4 DS-MLA cache report

Date: 2026-08-20
Target: server60, 4× RTX 3090 SM86, TP=4
Model: pinned Antirez DeepSeek-V4-Flash-0731 GGUF
Status: validated opt-in; FP8 remains the production default

## Decision

`fp4_ds_mla` is a working native cache format, not a Triton fallback or a
storage-only prototype. It increased the measured cache-token pool by 15.1%
with effectively unchanged decode and concurrency-2 throughput, a 3.1–4.4%
prefill cost, identical quick-quality results, and exact 136K NIAH retrieval.

It does **not** replace `fp8_ds_mla` by default. Both 148K profiles are measured
capacity ceilings, and the FP4 stress ladder left only 31 MiB/card, below the
normal 1 GiB sustained-service guard. The implementation ships as a separate,
digest-pinned Compose profile so the existing FP8 path and rollback remain
unchanged.

## Source-grounded format

The implementation directly extends the AppMana SM86 sparse-MLA path at
`AppMana/forks-flash-mla-int@7f41a5b`. That path already separated physical
cache-row dequantization from downstream BF16 attention for its native INT8
format. FP4 adds a third explicit cache format; it does not reuse INT8 or FP8
branches through an ambiguous boolean.

One `fp4_ds_mla` token row is 368 bytes:

| section | representation | bytes |
|---|---|---:|
| NoPE latent, 448 values | packed E2M1, even value in low nibble | 224 |
| RoPE latent, 64 values | BF16, unchanged | 128 |
| NoPE scales | 14 UE8M0 group-32 scales + 2 zero pad bytes | 16 |

E2M1 magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`; the high nibble bit is the
sign. UE8M0 stores `exponent + 127`. Quantization uses round-to-nearest-even
boundaries `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0`. The existing
`fp8_ds_mla` row remains byte-for-byte unchanged at 584 bytes.

## Gated causal hypotheses

### H1 — storage and capacity

**Gate:** every writer, allocator, page spec, SWA spec, reader, and native
attention operator must agree on the 368-byte physical row.
**Prediction:** raw row storage falls 37.0%; the whole-service cache-token pool
rises materially but by less than 37% because sliding-window/compressor state,
indexer cache, graphs, model weights, and workspaces remain.
**Outcome:** passed. The same 0.8 GiB startup pool rose from 156,373 to 180,039
tokens (+15.1%).

### H2 — native attention performance

**Gate:** native AppMana decode and prefill must consume compressed rows
directly, generate SM86 device code, agree with an independent FP32/BF16
oracle, replay deterministically under CUDA Graphs, and pass memcheck/racecheck.
**Prediction:** decode remains near FP8 because selected rows are dequantized
before the unchanged BF16 attention MMA; prefill may regress modestly because
E2M1 decode and scale handling add work.
**Outcome:** passed. Decode changed by +0.7%, concurrency-2 by +1.0%, and prefill
by -3.1% at 10K / -4.4% at 93K.

### H3 — quality

**Gate:** exact long-context NIAH retrieval plus matched benchlocal quick packs
must not show a material regression.
**Prediction:** FP4 cache quantization may perturb attention but should preserve
the tested task and retrieval behavior if errors remain bounded.
**Outcome:** passed for the tested scope. FP4 and FP8 both scored 27/30 with the
same TC-05, TC-06, and TC-07 failures. FP4 recalled the exact secret at the
136K stress rung. This is evidence for the tested workloads, not a universal
claim that FP4 cache is quality-equivalent.

## Implementation ownership

### Native FlashMLA

`Whamp/forks-flash-mla-int@81a06aa6`:

- explicit `Sparse_mla_cache_format::{FP8,INT8,FP4}_DS_MLA` dispatch;
- centralized `fp4_ds_mla.cuh` physical constants and E2M1 decode;
- native FP4 sparse decode using the selected-row dequantization pre-pass;
- native FP4 sparse prefill using compressed-row `cp.async` staging and
  in-shared-memory dequantization;
- Python stable-ABI wrappers for decode and prefill;
- adversarial physical-layout, numerical-parity, and CUDA-Graph tests.

### vLLM

`Whamp/vllm@633815f68`:

- `DeepseekV4CacheLayout` as the single physical-layout owner;
- `CacheConfig`, global dtype mapping, quant mode, MLA/SWA page accounting;
- Triton and stable-extension FP4 writers;
- Triton gather/dequant reader for diagnostics and non-native seams;
- native AppMana decode and prefill dispatch on SM8x;
- FP4-aware compressor configuration and scratch-workspace selection;
- fail-closed cache-shape checks before native decode;
- independent writer/reader and production-compressor tests.

The SM86 Triton writer uses a software E2M1 packer. The existing vectorized
`cvt.rn.satfinite.e2m1x2.f32` helper emits Blackwell-only PTX and is not legal
on RTX 3090.

## Bring-up defects caught

The full-model gate found four integration defects that isolated kernel tests
did not:

1. missing global `fp4_ds_mla -> torch.uint8` dtype mapping;
2. missing physical-row accounting on `SlidingWindowMLASpec`;
3. an FP8-only hard-coded 584-byte SWA backend shape;
4. a FP4-only Triton constexpr passed to 128-wide indexer kernels that do not
   declare it.

Each defect now has a focused regression. The final candidate reached TP=4 API
readiness with native FP4 decode and prefill and no silent fallback.

## Matched server60 results

Both cache formats used the same GGUF, runtime knobs, TP topology, 148K model
length, max_num_seqs=2, max_num_batched_tokens=256, 230 W power limit,
210–1650 MHz safety range, three warmups, and five measured runs. Swap was
normalized and serving-process swap was zero before accepted measurements.

| result | FP8 | FP4 | FP4 delta |
|---|---:|---:|---:|
| cache tokens | 156,373 | 180,039 | +15.1% |
| narrative decode | 79.84 tok/s | 80.36 tok/s | +0.7% |
| code decode | 79.82 tok/s | 80.37 tok/s | +0.7% |
| concurrency-2 aggregate | 126.02 tok/s | 127.27 tok/s | +1.0% |
| cache-busted prefill, 10K | 541.79 tok/s | 524.87 tok/s | -3.1% |
| cache-busted prefill, 93K | 518.82 tok/s | 495.79 tok/s | -4.4% |
| quick quality | 27/30 | 27/30 | same three failures |

FP4 also passed verify-full, deterministic generation, streaming, tool calls,
reasoning, post-tool continuation, coding/reasoning stress probes, NIAH at 94K
and 136K, and concurrency 2. The concurrency evidence is for short 512-token
requests, not two simultaneous 148K requests.

## Validation inventory

- 129 relevant CPU tests passed before final source formatting.
- 26 focused vLLM contracts passed from the exact post-fix source.
- 13 isolated FP4 writer/reader tests and the production compressor test passed
  on RTX 3090.
- 30 final formatted-wheel FP4/FP8/INT8 tests passed.
- Compute Sanitizer memcheck: 0 errors.
- Compute Sanitizer racecheck: 0 hazards.
- SM86 cubins are present in both the FlashMLA native library and vLLM stable
  extension.
- Full-model startup selected native FP4 sparse decode and native FP4 sparse
  prefill; FP8 indexer cache remained in use.
- The accepted BuildKit manifest list `sha256:20ad3a698bc6934d5abb12ccd32ebfa510ac3d0baa4660681226072ec362e164`
  referenced platform manifest `sha256:eb94d5049bf4d8d55c335ac1d2445382a811b7312d28e3e73088011a8103e181`
  and passed verify-full in both FP4 and FP8 modes with zero swap. FP4 reported
  180,039 cache tokens; FP8 reported 156,738, proving the new image retains the
  original cache path without a silent FP4 substitution.
- BuildKit's default provenance attestation made the outer manifest-list digest
  nondeterministic. The final builder disables provenance and reproduced the
  accepted platform manifest `eb94d504…` in two consecutive builds; Compose
  pins that stable content manifest.
- Raw compact evidence is checksum-bound in `evidence/server60-20260820/`.

## Reproducibility and rollback

`FP4-MANIFEST.json` pins:

- production base image `sha256:f91e8283…`;
- Whamp/vLLM commit/tree and all 14 runtime overlay hashes;
- Whamp/forks-flash-mla-int commit/tree and final wheel/native-library hashes;
- stable-extension hash and SM86 target;
- final image digest and Compose profile.

`build-fp4-kv-image.sh` verifies every input before building. The FP8
`base.yml` profile remains unchanged except for its sibling-variant header.
The canonical Antirez llama.cpp service remains the separate engine rollback.

After exact-image acceptance, server60 was restored to the unchanged FP8
production container on base image `sha256:f91e8283…`: healthy, restart
`unless-stopped`, zero restarts, zero serving-process swap, no candidate
containers, and the 230 W / 210–1650 MHz safety policy intact. See
`evidence/server60-20260820/final-service-state.txt`.
