# FORMAT-CONTRACT.md — GGUF block formats for the DeepSeek-V4-Flash GGUF-TP engine

Status: v1 (M1). Generated from pinned source, never from prose memory.
Authoritative source: Whamp/llama.cpp `0379cf4bf889f3d28038a005210c4bc193fc8ba1`
(local study checkout `/home/will/projects/llama.cpp-ds4-study`).
Every section cites file:line at that revision. Any change to this file
requires re-derivation from source and a passing L0 oracle (§5).

Reference implementations (MIT, vendored/adapted under `cuda/mmq/`):
antirez/ds4 `84cc882` — `cuda/mmq/vecdotq.cuh`, `cuda/mmq/ggml-common.h`,
`cuda/mmq/test/iq2_host_tables.h`.

## 0. Constants

| Name | Value | Source |
|---|---|---|
| QK_K | 256 | ggml-common.h (super-block size for K-quants) |
| QK8_0 | 32 | ggml-common.h:240 |
| QK8_1 | 32 | ggml-common.h:251 |

## 1. block_q8_0 — 34 bytes / 32 weights (2.125 bpw data)

Layout (ggml-common.h:240-246):

```c
typedef struct {
    ggml_half d;       // fp16 delta
    int8_t  qs[32];    // quants
} block_q8_0;          // 34 bytes
```

Decode — exact (ggml-quants.c:491 `dequantize_row_q8_0`):

```
for i in 0..31:  y[i] = fp32(d) * qs[i]        // single fp32 multiply
```

Used for: attention projections, shared experts, output head in this GGUF.
Runtime path: repacked to int8 group-32 (§6, lossy-in-last-bits), NOT
consumed natively by dense kernels; block decode above is the oracle
reference for the repack tolerance window.

## 2. block_q8_1 — 36 bytes / 32 values (activations only; not in GGUF)

Layout (ggml-common.h:251-259): fp16 `d`, fp16 `s = d * sum(qs)` (union
`ds` as fp16x2), 32×int8. Appears only as the runtime-quantized activation
side of vec_dot kernels. Recorded here because the rewritten expert kernels
replace Q8_1-activation dot products with bf16-activation MMA + in-mainloop
weight unpack; Q8_1 is the semantic reference for what the integer dot must
reproduce.

## 3. block_q2_K — 84 bytes / 256 weights (2.625 bpw)

Layout (ggml-common.h:289-299):

```c
typedef struct {
    uint8_t scales[16];  // per-16-weight sub-block: low nibble = scale, high nibble = min
    uint8_t qs[64];      // 2-bit quants, packed 4/byte
    ggml_half d;         // super-block scale (fp16)
    ggml_half dmin;      // super-block min-scale (fp16)
} block_q2_K;            // 84 bytes
```

**Struct member order caveat:** C struct declares `scales, qs, dm` in that
order; byte offsets are scales@0..15, qs@16..79, d@80..81, dmin@82..83.
(Confirm via `offsetof` in the L0 oracle harness — struct padding rules
make member order a decode-relevant fact, not cosmetic.)

Decode — exact operation order (ggml-quants.c:899-929
`dequantize_row_q2_K`). Weight model per sub-block: `x = dl * q - ml`.

```
d   = fp32(d)
min = fp32(dmin)
is = 0; qbase = 0                        # byte offset into qs
for n_chunk in 0..1:                     # two 128-weight chunks
    shift = 0
    for j in 0..3:                       # 8 sub-blocks per chunk
        sc = scales[is++];  dl = d * (sc & 0xF);  ml = min * (sc >> 4)
        for l in 0..15:  y = dl * ((qs[qbase + l]       >> shift) & 3) - ml
        sc = scales[is++];  dl = d * (sc & 0xF);  ml = min * (sc >> 4)
        for l in 0..15:  y = dl * ((qs[qbase + 16 + l] >> shift) & 3) - ml
        shift += 2
    qbase += 32
```

Sub-block → output ordering: sub-block `s` (0..15) covers weights
`s*16 .. s*16+15`; its 2-bit fields come from `qs[s*4 + (weight&3)... ]`
via the shift schedule above (sub-blocks 0,1 read qs[0..15]+qs[16..31] at
shift 0; 2,3 at shift 2; etc. within each 128-chunk).

Adversarial corpus must cover: sc low/high nibble extremes (0xF/0xF),
d=0, dmin≫d, all-ones qs patterns, and the two-chunk boundary.

## 4. block_iq2_xxs — 66 bytes / 256 weights (2.0625 bpw)

Layout (ggml-common.h:366-375):

```c
typedef struct {
    ggml_half d;         // fp16 scale
    uint16_t qs[32];     // 8 groups of 4 bytes
} block_iq2_xxs;         // 66 bytes (2-byte aligned — the DwarfStar
                         // alignment complaint; see PLAN §4.2/§4.6)
```

Decode — exact operation order (ggml-quants.c:2412-2437
`dequantize_row_iq2_xxs`). Per 32-weight group `ib32` (0..7):
view `qs[4*ib32 .. 4*ib32+8)` as `aux32[0], aux32[1]` (LE), and `aux8[0..3]`
as the four bytes of aux32[0].

```
d  = fp32(d)
# db uses the float path: 0.5f + (aux32[1] >> 28) is [0.5 .. 15.5], then *0.25f
db = (d * (0.5f + (aux32[1] >> 28))) * 0.25f
for l in 0..3:
    grid8 = iq2xxs_grid[aux8[l]]            # 8 x uint8, table = 2 KiB
    signs = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127]   # 7-bit selector
    for j in 0..7:
        y[...] = (db * grid8[j]) * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f)
```

Equivalent integer form used by the GPU dot (vecdotq.cuh:985-1014
`vec_dot_iq2_xxs_q8_1`): `ls = (aux32[1] >> 27) | 1` (i.e. `2*db_bits`),
grid bytes are SIGNED in the dot (`sumi += grid[j] * sign` with sign from
ksigns/kmask), accumulator scaled `sumi * ls / 8 * d`. The L0 oracle uses
the CPU float path (§5); the integer form is cross-checked in the kernel
oracle (class B) — both are pinned here so neither drifts.

Tables: `iq2xxs_grid` (256×8 uint8, generated in ggml-quants.c at build of
this revision — the L0 harness compiles the pinned source, so table bytes
come from the same revision, and the harness additionally snapshots the
table SHA-256 into the evidence bundle), `ksigns_iq2xs` (128 uint8),
`kmask_iq2xs = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80}`.

Adversarial corpus must cover: aux8 values 0 and 255 (LUT boundary),
aux32[1]>>28 ∈ {0,15} (sub-scale extremes), sign-selector byte extremes,
d=0, and group-boundary bytes at ib32=0 and ib32=7.

## 5. L0 oracle (class A) — specification

Purpose: prove our format understanding bit-exactly against the pinned
implementation before any kernel is written.

- **Reference A (pinned):** the three `dequantize_row_*` functions compiled
  from `llama.cpp-ds4-study` ggml source at `0379cf4bf` (verbatim
  extraction with provenance header if standalone compilation requires it).
- **Reference B (independent):** a decoder written only from this document,
  in Python/NumPy using explicit float32 dtype and the exact operation
  order above (fp16→fp32 conversion is lossless; every multiply/subtract
  issued as float32 in the stated order → bitwise-comparable).
- **Inputs:** ≥10,000 random blocks per format (seeded, reproducible) plus
  the §3/§4 adversarial corpora plus a full all-zeros and all-0xFF block.
- **Pass:** 100% bitwise-equal fp32 outputs across all blocks and formats.
  Any mismatch is a contract bug: fix this document or reference B, never
  the pinned reference.
- Evidence: script + seed + table SHA-256 snapshots archived under
  `.research/gguf-tp-engine/evidence/`.

## 6. GGUF tensor-side contract (per-tensor)

From `src/models/deepseek4.cpp` at `0379cf4bf` (lines 131-160):

| GGUF tensor | Logical shape | Format | vLLM destination (TP rule per PLAN §4.7) |
|---|---|---|---|
| `ffn_gate_exps` | {n_embd=4096 (K), n_ff_exp (N), n_expert=256 (E)} | IQ2_XXS | routed w13 gate slot |
| `ffn_up_exps` | {n_embd (K), n_ff_exp (N), n_expert} | IQ2_XXS | routed w13 up slot |
| `ffn_down_exps` | {n_ff_exp (N), n_embd (K), n_expert} | **Q2_K — note K/N swapped vs gate/up** | routed w2 |
| `wq_a` / `attn_kv` | separate tensors | Q8_0 | `fused_wqa_wkv` slots (disable_tp replicated) |
| attention/shared/output | per-tensor | Q8_0 | dense int8-g32 repack (§1 note) |
| `token_embd` | {n_vocab, n_embd} | F16 | vocab-sharded |

Byte-level row stride: a GGUF quantized row of logical width W stores
`ceil(W/QK)*sizeof(block)` bytes contiguously; row-major over the first
dimension. The M1 inventory verifies actual strides from the pinned GGUF
headers on server60 (read-only).

## 7. Aligned-SoA repack contract (class A by construction)

The load-time repack (PLAN §4.2) rearranges `block_iq2_xxs` data into
separate `d[]` (fp16) and 64-byte-aligned `qs[]` arrays per block WITHOUT
changing any decoded value: same integer inputs, same operation order —
only byte placement changes. Its gate: repacked-artifact decode == raw
decode bitwise (L0 extension), plus a content hash over the repacked
artifact bound in the loader (DwarfStar `ds4_repack.cu:1-6` discipline).
Q2_K aligned variant: same principle for `scales/qs/d/dmin` streams.

## 8. Change control

This contract is frozen for M2–M4. Format questions discovered during
implementation are resolved by re-reading the pinned source and amending
this file with a new line-cited entry + oracle rerun — never by editing
kernel code against memory.
