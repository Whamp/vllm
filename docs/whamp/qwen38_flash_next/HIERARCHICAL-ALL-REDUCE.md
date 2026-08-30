# Qwen3.8 hierarchical all-reduce

## Decision

Promote island-aware hierarchical all-reduce for server60's Qwen3.8 FP8-QSA
profile. The exact production instance improved matched single-stream decode from
43.77 to 50.34 tokens/s, a 15.0% gain, while cache-busted prefill moved from
1,529.25 to 1,538.14 tokens/s. The profile kept native 262,144-token context,
vision, tool use, post-tool continuation, and the BF16 rollback.

The deployment sets:

```text
VLLM_HIER_ALL_REDUCE=0,1;2,3
```

Ranks 0 and 1 form one PCIe island. Ranks 2 and 3 form the other. Payloads that
fail the hierarchical backend's dtype, shape, or size checks continue through
PYNCCL.

## Why this change

A matched Nsight Systems trace of the promoted FP8-QSA service found BF16 NCCL
ring all-reduce consumed 62.5% of summed kernel time in the c=1 segment and
65.8% in the c=2 segment. Across six measured decode episodes, c=1 spent 4.221
ms of summed collective kernel time per generated token. The service used
PYNCCL because the image did not contain the island-aware backend.

Whamp/vLLM main already contained the required implementation at commit
`3d8f13bc19e028d0e8d9ac0a9a4899fb1bc22872`. The deployed backend file is
byte-identical to that revision with SHA-256
`299a7c6dbbfd620c9d8b904ba0f4fd731f7dfec3cf494b1b24618aa2095adc39`.
The older Qwen image needed a 32-line compatibility hook in its installed
`cuda_communicator.py`. The hook adds the environment parser, backend
construction, dispatch ordering, and startup logging without replacing other
Qwen-specific runtime files.

## Gated hypothesis

- **Outcome:** improve single-stream decode by at least 5% without reducing
  prefill, context, or capability.
- **Critical segment:** decode-sized BF16 all-reduces across four PCIe-only RTX
  3090 GPUs.
- **Move:** reduce cross-island synchronization and traffic with two equal
  islands, while retaining PYNCCL for unsupported payloads.
- **Gate:** the trace placed collective work on the decode path, canonical source
  already implemented the backend, and server60's topology matches islands
  `0,1;2,3`.
- **Lose condition:** numerical disagreement, CUDA-Graph failure, a slower
  serving result, or a payload outside the backend's `should_use` contract.
- **Shifted cost:** one compatibility overlay on the older image and raw CUDA IPC
  buffers owned by each communicator.
- **Falsifier:** no end-to-end gain after the microbenchmark changed, or any
  capability, stability, memory, or prefill regression.

## Mechanism gate

The four-GPU RTX 3090 gate compared the hierarchical backend with NCCL on BF16
payloads and exercised both alternating CUDA-Graph buffers. Numerical results
matched NCCL within BF16 reduction tolerance, and deterministic graph replay
passed.

The table reports the slowest rank's median time over 80 measured iterations.

| Elements | Hierarchical | NCCL | Ratio |
| ---: | ---: | ---: | ---: |
| 2,560 | 80.90 us | 107.52 us | 0.756x |
| 4,096 | 91.65 us | 109.57 us | 0.839x |
| 10,240 | 93.15 us | 109.57 us | 0.850x |
| 65,536 | 173.06 us | 220.16 us | 0.786x |
| 262,144 | 448.51 us | 534.53 us | 0.841x |

Startup then reported `HIERARCHICAL` before `PYNCCL` for both the TP and EP
communicators. This proves runtime dispatch, not only source eligibility.

## Serving results

The comparison used the same Intel AutoRound checkpoint, Primitive NVFP4 PLE,
calibrated FP8 E4M3 QSA cache, 262,144-token context, batch-token budget 1,024,
maximum sequences 2, GPU-memory utilization 0.95, and fixed 230 W GPU policy.
The only runtime changes were the compatibility image and hierarchical island
environment variable.

| Measurement | FP8 QSA plus PYNCCL | Hierarchical exact final | Change |
| --- | ---: | ---: | ---: |
| Decode, 5-run mean | 43.77 tok/s | 50.34 tok/s | +15.0% |
| Cache-busted prefill, 3-run mean | 1,529.25 tok/s | 1,538.14 tok/s | +0.6% |
| Concurrency-2 aggregate | 53.25 tok/s | 59.00 tok/s | +10.8% |
| KV-cache capacity | 314,261 tokens | 315,039 tokens | +778 tokens |

The exact-final decode runs ranged from 48.57 to 52.55 tokens/s with a 3.47%
population coefficient of variation. The exact-final prefill runs ranged from
1,535.40 to 1,541.84 tokens/s with a 0.18% coefficient of variation.

## Capability and stability

The promoted image passed:

- deterministic `PARIS` generation;
- automatic multiply-tool selection and correct post-tool continuation;
- a synthetic multimodal request;
- exact `VIOLET ORBIT 9137` retrieval from a 261,544-token API prompt;
- two simultaneous 256-token streams at 59.00 aggregate tokens/s;
- BenchLocal quick at 26/30, matching the accepted baseline exactly;
- zero container restarts and zero host or serving-process swap;
- no logged OOM, allocator retry, JIT failure, CUDA error, or engine failure;
- unchanged NVML allocation before and after long-context acceptance;
- zero running requests, zero waiting requests, and zero KV use after the run.

The production container uses restart policy `unless-stopped`. The rollback
contracts preserve both the prior FP8/PYNCCL profile and the earlier BF16-QSA
profile. The fixed `gpu-power-limit.service` remained active at 230 W throughout
validation.

## Warmed stability acceptance

The final acceptance used diagnostic image
`sha256:63ccf7ea4983d950b739b1e9bc2a5fcbaba1a8537ccfbc5b7fecfa1747db6c85`,
which layers opt-in memory reporting over the promoted image without changing the
model, QSA, collective, scheduler, or serving configuration.

The first full acceptance warmed deterministic generation, tool and post-tool
turns, multimodal input, concurrency 2, and a 261,544-token NIAH request. The
allocator made a one-time first-use reservation: per rank, reserved memory grew
by 360-380 MiB from startup warmup to execution step 500, while active allocation
grew by only 5.37 MiB. This is allocator cache, not retained model or request
state.

The warmed state then held:

- all recorded allocator counters were byte-identical between execution steps
  500 and 750 on all four ranks;
- a separate first-block-nonced 261,549-token NIAH request passed in 188.55
  seconds;
- 162 one-second NVML samples per GPU showed zero memory growth during that cold
  long-context request;
- minimum free memory was 1,370 MiB on rank 0 and 1,394 MiB on ranks 1-3;
- maximum serving-process swap was zero;
- post-run metrics reported zero running or waiting requests and zero KV-cache
  use.

The diagnostic container reported `OOMKilled=false`, and its live log scan found
no OOM, allocator-retry, CUDA-error, fatal, traceback, or engine-failure record.
The diagnostic was removed after collection. The normal production image was
restored healthy with restart policy `unless-stopped`, restart count zero, swap
disabled, `vm.overcommit_memory=0`, the 230 W safety service active, and no
rollback timer.

## Production identity

| Item | Value |
| --- | --- |
| Base FP8-QSA image | `sha256:61971a78222e89335d84a7f4d72b0e8842619a4a29564582a58ff328af48abb9` |
| Hierarchical production image | `sha256:4b59067e269f313a78f0a698e79261230fb02e3712f42ffd54b3e9ec9be9705a` |
| Base communicator SHA-256 | `a9a0242175c3d3e0d46d586a38cb76139e6e4e92d15032d5b1e33237af2f6757` |
| Deployed communicator SHA-256 | `8d6b202ab8838da0196fd3af3a292e30185e5c19d4df32373401428e96d86cd7` |
| Hierarchical backend SHA-256 | `299a7c6dbbfd620c9d8b904ba0f4fd731f7dfec3cf494b1b24618aa2095adc39` |
| Islands | `0,1;2,3` |
| Endpoint | `http://server60:30002/v1` |
| Model context | 262,144 tokens |
| KV cache | calibrated FP8 E4M3 QSA |

The compact evidence bundle is under
[`evidence/qwen38-hier-allreduce-20260830`](evidence/qwen38-hier-allreduce-20260830/README.md).
