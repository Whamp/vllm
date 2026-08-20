# FP4 DS-MLA capacity baseline

This report records the allocation baseline before enabling an MXFP4 sparse-indexer cache on RTX 3090. The runtime uses `fp4_ds_mla` for the main MLA cache and FP8 for the sparse-indexer cache.

## Pinned inputs

- Runtime source base: `Whamp/vllm@81593507f`
- FP4 integration source: `Whamp/vllm@633815f6889d9d033aefa04bf40cb270d5b6a3f1`
- Accounting image: `sha256:4f11d99672280c34ad32c271c20195bb76aaad35c3cdc780177f946dd5cfacd6`
- Main FP4 image: `sha256:eb94d5049bf4d8d55c335ac1d2445382a811b7312d28e3e73088011a8103e181`
- Hardware: four RTX 3090 GPUs, TP=4
- Safety policy: 230 W power limit and 210-1650 MHz graphics-clock range
- Cache profile: `fp4_ds_mla`, `max_num_seqs=2`, `max_num_batched_tokens=256`

## Measured capacity

| Configured context | KV tokens | KV allocation | Packed blocks | Near-ceiling NIAH | Serving swap | Stress headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160,000 | 179,743 | 841,727,040 B | 965 | 156,649 tokens | 0 KiB | 27 MiB |
| 170,000 | 178,887 | 825,154,176 B | 946 | 166,470 tokens | 0 KiB | 27-28 MiB |
| 175,000 | 178,050 | 814,687,104 B | 934 | 173,058 tokens | 0 KiB | 27-28 MiB |

All three profiles reached API readiness and passed the fast stress probes, tool and reasoning checks, and exact needle retrieval near their configured ceilings. The planner's storage size matched storage-deduplicated runtime allocation on every rank with a zero-byte reconciliation delta.

None is release-safe under the project's 1 GiB free-VRAM gate. The 175K profile is the highest tested functional ceiling, not a promotion candidate.

## Packed layout at 175K

The planner packs 167 cache specs into five groups. Every allocated block uses the largest group stride, 872,256 bytes.

Group 0 contains:

- 21 ratio-4 FP8 indexer rows: 132 physical bytes per token, 8,704-byte pages
- 21 ratio-4 FP4 MLA rows: 368 physical bytes per token, 23,584-byte pages
- 20 ratio-128 FP4 MLA rows: 368 physical bytes per token, 1,056-byte pages

Those specs use 699,168 bytes of each group-0 block. The 872,256-byte global stride leaves 173,088 bytes at the end of group 0.

Changing the indexer row from 132-byte FP8 to 68-byte MXFP4 is predicted to reduce its aligned page from 8,704 to 4,608 bytes. Group 0 would shrink to 613,152 bytes, but its tail would grow to 259,104 bytes. The global stride would remain 872,256 bytes because group 3 is still the largest group.

At 175K, the indexer change should reduce logical model-length bytes by 58,834,944 bytes per rank. It should not increase the physical KV pool or token capacity unless packing or group composition also changes. The post-port runtime report must confirm this prediction.

## Evidence

`evidence/capacity_before_indexer/` contains the resolved Compose files, image inspection, startup and allocation logs, planner JSON, GPU and swap snapshots, verification output, final container state, and release snapshots for all three profiles. `SHA256SUMS` binds every file.
