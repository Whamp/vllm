# Qwen3.8 exact-production trace blocker

## Decision

Do not use this campaign as a model trace. No valid model `.nsys-rep` or SQLite
file was captured.

The corrected Nsight-launched service reached a healthy API, but it fit 423,164
KV-cache tokens instead of the promoted production contract of 425,497. The
runner rejected it before collection. Changing the trace feature set or KV
allocation would have created a different experiment, so the campaign stopped
and released the GPUs to the queued shared-expert test.

## Fixed contract

The attempted trace kept the promoted image and serving configuration:

```text
sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef
```

It preserved:

- TP4 and expert parallelism;
- `max_num_seqs=4` and `max_num_batched_tokens=1024`;
- `gpu_memory_utilization=0.98`;
- FP8 E4M3 KV cache;
- Mamba align mode and prefix caching;
- CPU PLE offload with native NVFP4 lookup;
- hierarchical all-reduce `0,1;2,3`;
- implicit async scheduling;
- FULL_DECODE_ONLY CUDA Graphs;
- the 262,144-token model length.

Nsight Systems 2025.5.2.266 launched the server with CUDA, NVTX, and OS runtime
tracing, node-level CUDA Graph tracing, and CUDA-event tracing. CPU sampling and
context-switch collection were disabled.

## Capacity gate

| Service | KV-cache tokens | Delta |
| --- | ---: | ---: |
| Promoted production | 425,497 | control |
| Nsight-launched service | 423,164 | -2,333, or -0.5483% |

The Nsight-launched service loaded the expected model and native-PLE artifacts,
captured all four decode graphs, and exposed a healthy API. Concurrent PLE
fan-out remained disabled. The fitted-capacity mismatch was the first failed
identity gate, so the runner stopped before `nsys start`.

This result proves only that the instrumented startup did not preserve fitted
capacity. It does not identify which profiler feature accounts for the memory
difference.

## Attempt ledger

1. The first orchestration attempt received SIGHUP before model launch. The
   runner was moved into a separate terminal session and its signal handling
   was corrected.
2. The Ubuntu package rejected `nccl` as a direct trace source before model
   launch. The corrected `cuda,nvtx,osrt` source set passed an exact parser
   test. NCCL kernels would still appear in CUDA activity.
3. The corrected service reached a healthy API but failed the capacity gate
   above. No collection began.

No fourth attempt ran. The queued shared-expert candidate already had a
completed CPU review and needed the GPUs.

## Restoration and handoff

The failure path restored and verified the pinned production service. The
handoff procedure then stopped it and recorded:

- zero GPU compute applications;
- zero swap devices;
- `vm.overcommit_memory=0`;
- no trace container;
- no active trace or restore units and timers;
- no orphan trace, restore, or benchmark jobs.

The shared-expert owner acknowledged the receipt before receiving
`GPUS RELEASED` for:

```text
perf/qwen38-shared-expert-overlap
9cbc55c7ac78bfbedb9a7c5b314726f996ea710d
```

That owner accepted watchdog, rollback, and final production-restoration
responsibility.

## Evidence

The campaign checkpoint is under:

```text
/home/will/build/qwen38-nsys-trace
```

Important files:

- `TRACE-BLOCKER.md`
- `evidence/pre-trace-server.log`
- `evidence/sessions-before-start.txt`
- `evidence/trace-path.json`
- `evidence/trace-args.json`
- `evidence/cleanup-trace-container.log`
- `runner-failure-hup-1/`
- `runner-failure-invalid-nccl-2/`
- `GPU-RELEASE.txt`

The 68-file evidence manifest has SHA-256:

```text
81bbf473ba31eeb5e9ba6aa82a28a3e544436e548ea18abb0c5fd7ce3846ab66
```

The GPU-release receipt has SHA-256:

```text
533da6f487eff497497dfd0db0f3088527f5d163dd18a760fe5b9d301e6327aa
```

The pinned restore command remains:

```text
/home/will/inference/runtime/qwen38-qsa-fp8-candidate/restore-fp8.sh
```

Its SHA-256 is:

```text
d18463d486f803864fa1fb6b09040fb574f63fa0b87b80e6ef83c82a4035e34f
```
