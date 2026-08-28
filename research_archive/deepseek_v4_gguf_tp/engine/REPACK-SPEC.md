# REPACK-SPEC.md — aligned-SoA experiment contract (M1 spec; IQ2 rejected)

Source of the hypothesis: antirez/ds4 `84cc882`
`cuda/mmq/test/proto_iq2_aligned.cu:1-22` (MIT; attribution retained per
PLAN §4.6). Discipline gate: DwarfStar `ds4_repack.cu:1-6` — producers
byte-identical, content hash is the gate.

## Why

`block_iq2_xxs` is 66 bytes and 2-byte aligned, so the code stream forces
two 16-bit loads per 32-bit weight word (`get_int_b2`). Our measured
serving-shape MMVQ ceiling (346–358 GB/s vs 713 GB/s Q8_0 on one 1650 MHz
3090) is consistent with DwarfStar's ~142-of-~200 GB/s alignment
diagnosis on their silicon. **Re-measured on SM86 at M2 — not assumed.**

## IQ2_XXS decision (M2)

Rejected for the production DP4A path: exact TP4 N512 single-matrix aligned
was 9.90 µs vs raw 8.89 µs; indexed raw reaches 247.35 GB/s without a derived
artifact. The layout below remains reproducible reference evidence only.

## Layout (IQ2_XXS reference)

For a tensor region of `nb` blocks (a whole expert matrix row-block span
per rank), raw = `nb × 66 B` interleaved. Repacked = three parallel
streams, total bytes identical (**66·nb**, no padding):

```
d[]        : nb × fp16          (block scales, 2·nb bytes)
grid[]     : nb × 32 × uint8    (grid-index bytes = aux32[0] per group, 32·nb)
lsign[]    : nb × 32 × uint8    (aux32[1] per group: 7-bit sign selectors ×4
                                 + shared 4-bit ls in high nibble of byte 3)
```

64-byte alignment per stream start; per-block rows of `grid[]`/`lsign[]`
are 32 B (naturally 2-wide coalescable; the 64 B claim in the proto refers
to per-block qs alignment — we keep 32 B rows and 64 B stream alignment,
which preserves the full-width load win without padding bytes; the M2 A/B
microbench owns the final choice, byte-neutrality is the hard constraint).

Decoded-value identity: decode(streams) must equal decode(raw blocks)
**bitwise in fp32** (class-A extension — same integer inputs, same
operation order, only placement changes). The `ls` reconstruction
`aux32[1] >> 27 | 1` reads from `lsign[]` bytes 4k+3 exactly as before.

## Layout (Q2_K)

Same principle, four streams (byte-neutral 84·nb): `d[]`, `dmin[]` (fp16
each), `scales[]` (16 B/block), `qs[]` (64 B/block). Q2_K's raw block is
already member-interleaved; SoA separation removes the 2-byte-aligned d/dmin
reads inside the hot loop.

## Producer and gates

- Produced **in-process at model load** (no on-disk artifact; the GGUF is
  never mutated). Cost: one pass over 72.56 GiB ≈ memcpy-bound (~seconds on
  PCIe gen3; acceptable at startup, measured at M4).
- **Content hash**: SHA-256 over (d ‖ grid ‖ lsign) per tensor, recorded in
  the load manifest and asserted on every load (deterministic → identical
  hashes across ranks; mismatch = fail-closed).
- **Class-A extension**: L0 oracle extended to prove streams-decode ==
  blocks-decode bitwise over the same random+adversarial corpora.
- **Capacity**: byte-neutral by construction; CAPACITY.md carries zero
  delta for the repack itself (loader scratch during transform is transient
  and bounded per-tensor-stream).
