# Qwen3.8 native NVFP4 PLE gather result

## Decision

Promote the native NVFP4 PLE gather with Loktar's PLE meta-construction fix. The production profile also raises `max_num_seqs` from 2 to 4.

The promoted image is:

```text
sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef
```

It contains:

- Kernel2 revision `42b918e36fa3bdd04e3d7bd7ad4a9c7695b9624f`;
- PLE meta-construction revision `d014fb2c0d4063843acf92d5fdbbc2e198b2e604`;
- native PLE gather revision `9a0f67dcec40424f3c093abdbadf52b58730f187`.

The native gather keeps the existing Torch implementation as the fallback when the environment variable is unset or the output is not BF16.

## Actual sidecar correctness and timing

The server60 gate used the 128 production mmap shards, width 160, and 320,001,536 total rows. It compared 260 rows, including duplicates and shard boundaries. Native BF16 output matched the production Torch arithmetic bit for bit.

| Rows | Torch mean | Native mean | Mean speedup |
| ---: | ---: | ---: | ---: |
| 16 | 2.199706 ms | 0.014079 ms | 156.24x |
| 32 | 3.388903 ms | 0.019132 ms | 177.13x |

These timings include the production mmap tensors. They do not include request scheduling or GPU delivery.

## Matched service measurements

The service comparisons used the same four RTX 3090 GPUs, image base, model, FP8 KV cache, 0.98 GPU memory utilization, hierarchical all-reduce grouping `0,1;2,3`, and `FULL_DECODE_ONLY` CUDA Graph mode. Pair 1 ran control then candidate. Pair 2 reversed the order. Each matrix contained five measured decode rounds at each concurrency after three warmups.

| Pair | Variant | C1 decode | C2 decode |
| --- | --- | ---: | ---: |
| 1 | Loktar control | 51.50 tok/s | 80.13 tok/s |
| 1 | Native candidate | 59.89 tok/s | 104.02 tok/s |
| 1 | Change | +16.30% | +29.81% |
| 2 | Native candidate | 67.39 tok/s | 107.47 tok/s |
| 2 | Loktar control | 51.91 tok/s | 80.04 tok/s |
| 2 | Change | +29.81% | +34.27% |

A separate matched `max_num_seqs=4` probe measured:

| Variant | C4 decode | C4 prefill |
| --- | ---: | ---: |
| Loktar control | 120.82 tok/s | 1551.26 tok/s |
| Native candidate | 167.46 tok/s | 1638.73 tok/s |
| Change | +38.61% | +5.64% |

The immutable final image repeated C4 at 163.10 tok/s. A later warmed run reached 185.61 tok/s, but the promotion claim uses the matched 167.46 tok/s result.

## Capacity and acceptance

The Loktar control and native candidate both reported:

```text
Available KV cache memory: 2.82 GiB
GPU KV cache size: 425,497 tokens
Maximum concurrency for 262,144 tokens per request: 1.62x
```

Ready-state GPU memory was identical at 23,087 MiB used and 1,040 MiB free per rank. The PLE worker held 276 MiB on each rank in both variants. The candidate had no unexplained residency increase.

The immutable final image passed:

- deterministic output with the exact answer `PARIS`;
- tool selection and post-tool continuation;
- multimodal image input;
- reasoning output;
- two-stream decode;
- prefix caching and `FULL_DECODE_ONLY` CUDA Graph startup;
- needle retrieval at 261,544 API prompt tokens;
- health, model identity, restart, swap, and embedded-artifact hash checks.

Production used profile SHA-256 `ad55f9159a517e3ac816361d73473e90ae3ff1b794eba59a0324d5dd7d51b0fb` and restore script SHA-256 `d18463d486f803864fa1fb6b09040fb574f63fa0b87b80e6ef83c82a4035e34f`.

## Deviations from the initial gate list

The initial plan called for five service-level A/B pairs and a matched Nsight readiness trace before promotion. The campaign instead ran two reverse-ordered service pairs. Those pairs contained ten measured rounds per variant at C1 and C2. It also ran a matched C4 pair and repeated the full acceptance pack on the immutable image.

The readiness trace was not collected before promotion. Will authorized promotion after the repeated service gains, bit-exact production-sidecar test, unchanged capacity, and full acceptance result. This is a recorded deviation, not a passed trace gate. Collect a fresh trace before attributing later fan-out or async-scheduling results to PLE readiness changes.

## Evidence

The checksum-bound local receipt set is:

```text
/home/will/build/qwen38-ple-native-lookup-evidence/SHA256SUMS
```

The server60 campaign directory is:

```text
/home/will/build/qwen38-ple-native-lookup
```
