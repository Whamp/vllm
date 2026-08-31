# Qwen3.8 shared-expert early launch

Status: GPU-validated, default-off, and rejected after the post-BIOS x8 retest.

This experiment changes when vLLM submits Qwen3.8's shared expert. The current CUDA path submits routed-expert work on the main stream, then submits the shared expert on an auxiliary stream and waits before combining both results. CUDA can overlap those queues because the auxiliary stream waits at an earlier synchronization point. The experiment submits the same shared expert before gate and routed-expert dispatch, then waits at the same result-consumption boundary.

The change may improve decode if the current host launch order leaves shared-expert work exposed. It may do nothing on NVIDIA. Upstream's B200 trace showed the old and early-launch CUDA schedules already had the same overlap structure. RTX 3090 behavior remains an instance value that needs a direct trace and same-image benchmark.

## Why this is worth one bounded test

Qwen3.8 has one shared expert in each of 48 MoE layers. The shared and routed experts consume the same layer input but produce independent outputs until vLLM combines them. This creates a legal overlap opportunity.

Only the exposed part of shared-expert execution can become a gain. The experiment does not remove a kernel, reduce weight traffic, or change arithmetic. If the current CUDA scheduler already overlaps both streams, moving the host submission earlier has no useful effect.

The evidence points in both directions:

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| [vLLM PR #38990](https://github.com/vllm-project/vllm/pull/38990) | Losing CUDA shared-expert overlap caused 9-14% GLM-5 decode regressions on eight Blackwell GPUs. A Nemotron report measured 15-20%. | The current RTX 3090 path has not lost overlap. |
| [vLLM PR #48223](https://github.com/vllm-project/vllm/pull/48223) | Earlier submission repaired ROCm overlap and improved many MI300 decode cases by roughly 2-7%. | ROCm stream behavior and DPA workloads do not predict server60. Some tested TP and concurrency points regressed. |
| [vLLM PR #52024](https://github.com/vllm-project/vllm/pull/52024) | The first early-launch patch was reverted after a Qwen3.5 race corrupted accuracy. | Reversion alone does not make the schedule invalid when ownership is correct. |
| [vLLM PR #52033](https://github.com/vllm-project/vllm/pull/52033) | Upstream fixed the ROCm input-alias race and restored the patch. Its NVIDIA B200 trace showed unchanged overlap structure. | It did not benchmark RTX 3090 or the Intel Qwen3.8 checkpoint. |

The current Whamp model inherits `Qwen3NextSparseMoeBlock`. It constructs a separate `Qwen3NextMLP` shared expert and gives it to `FusedMoEFactory`. `SharedExperts` already selects an auxiliary CUDA stream for batches at or below `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`, which defaults to 256 tokens.

## Causal hypothesis

**Outcome.** Improve warmed Qwen3.8 decode by at least 3% at concurrency 1 or concurrency 2. Keep the other decode result within 2% and keep cache-busted prefill within 2%.

**Critical segment.** The per-layer interval from gate submission through availability of both routed and shared expert outputs.

**Move.** Submit the existing shared expert before gate and routed-expert dispatch. Use per-DBO-slot CUDA events to order its input and output. Wait only where vLLM already consumes the shared result.

**Gate.** An RTX 3090 timeline must show an earlier shared-expert start and less exposed shared-expert time. The same-image selector A/B must meet the outcome threshold.

**Lose condition.** Both streams contend for the same SM or memory resources, event work costs more than it hides, the current path already overlaps fully, or concurrency changes which operation sets the critical path.

**Shifted cost.** Four CUDA events per layer, more stream dependencies, new failure modes around slot reuse, and another runtime configuration to maintain.

**Falsifier.** Reject the experiment if the timeline does not move the shared-expert critical segment, if warmed decode improves by less than 3%, or if correctness, CUDA Graph replay, race safety, prefill, concurrency, memory stability, or service health regresses.

## Implementation boundary

`VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH=1` enables the experiment. The default is off.

The change is limited to:

- `SharedExperts`, which owns auxiliary-stream events and two DBO output slots;
- `MoERunner._forward_impl`, which chooses the submission point;
- `MoERunner._apply_quant_method`, which waits before consuming the result;
- `vllm.envs`, which defines the default-off selector.

When the selector is off, vLLM retains the existing `maybe_sync_shared_experts_stream` and `_run_in_aux_stream` path. The selector does not change ROCm behavior, PLE construction, the shared-expert token threshold, modular-kernel-owned overlap, weights, kernels, arithmetic, quantization, routing, collectives, cache layout, context capacity, or production defaults.

The early path tracks `idle`, `in-flight`, and `ready` states independently for both DBO slots. It rejects double launch, wait-before-launch, double wait, and result consumption before the main stream has waited. It records the input on the auxiliary stream and the output on the consuming main stream. A terminal event orders partially submitted auxiliary work before a host-side shared-expert exception clears the slot for retry.

## CPU evidence

The focused tests were red on unmodified `Whamp/vLLM main` because the selector, early submission, and lifecycle state did not exist. They now cover:

- default-off configuration;
- explicit main-to-aux input-event and aux-to-main output-event order;
- unchanged legacy ordering when the selector is off;
- launch before gate and dispatch;
- cleanup after a host-side shared-expert failure, including cleanup failure;
- no CUDA-event allocation when the selector is set on ROCm;
- modular-kernel ownership and unchanged runner fallback ordering;
- generated two-slot operation sequences up to 100 steps.

The property test compares production state with an independent `idle -> in-flight -> ready -> idle` model. It generates starts, waits, consumes, and error-path discards for both DBO slots, including invalid sequences. A planted defect that failed to clear a waited slot shrank to two operations: `start(0), wait(0)`.

The exact production delivery adapter also has tests. It reconstructs the legacy runner from source commit `42b918e36fa3bdd04e3d7bd7ad4a9c7695b9624f`, reconstructs the installed shared-expert baseline from `617d38d97b4dd8a90ad0ffaf15a4f64412470b25`, compiles all transformed files, and rejects source drift or duplicate application.

CPU tests cannot prove CUDA event behavior, CUDA Graph capture, race freedom, timeline overlap, numerical output, or serving performance.

## Server60 result

The bounded GPU run used candidate image
`sha256:39de8fdfb787592cf06819268a817c8a4087d84658e2300d7adb5ad136b59bb3`
over the exact production base. The one-GPU operator suite passed eight eager
and CUDA Graph cases. Compute Sanitizer reported zero memcheck errors and zero
racecheck hazards.

The non-restarting TP=4 candidate retained the native 262,144-token API limit
and passed deterministic output, automatic tool use, post-tool continuation,
multimodal input, two-stream decode, and exact retrieval from a 261,544-token
API prompt. It stayed at zero serving-process swap with no allocator failure.

The same-image unprofiled A/B measured:

| Metric | Selector off | Selector on | Change |
| --- | ---: | ---: | ---: |
| C1 decode | 59.3775 tok/s | 58.8669 tok/s | -0.8598% |
| C2 aggregate decode | 103.2025 tok/s | 104.5260 tok/s | +1.2825% |
| C1 cache-busted prefill | 1,614.3558 tok/s | 1,618.7557 tok/s | +0.2725% |
| C2 aggregate prefill | 1,629.2663 tok/s | 1,632.5239 tok/s | +0.1999% |

This initial run did not clear the preregistered 3% decode threshold. It left
the mechanism open only because C2 decode and both prefill results improved.

Matched pre-BIOS Nsight Systems traces used the same profiler-adjusted
423,164-token KV capacity for both arms. CUDA Graph replay used two streams with
the selector off and three with it on. Across four GPUs, median C1 graph spans
fell by about 2.49% and overlap time rose by about 2.65%. Median C2 graph spans
grew by about 8.91% and overlap time fell by about 10.04%. Node-level tracing
perturbs timing, so these figures show a real but phase-dependent schedule
change. They are not serving benchmarks.

### Post-BIOS x8 retest

The BIOS change raised GPU0 from x4 to x8. Server60 then negotiated widths of
x8, x16, x8, and x16. The retest ran selector-off and selector-on arms in both
forward and reverse order. Each side therefore includes ten measured decode
runs across two independent startups. C4 was diagnostic only.

| Metric | Selector off | Selector on | Change | Forward | Reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 decode | 61.588 tok/s | 55.450 tok/s | -9.967% | -9.593% | -10.341% |
| C2 aggregate decode | 104.646 tok/s | 105.588 tok/s | +0.900% | +2.161% | -0.360% |
| C4 aggregate decode | 188.895 tok/s | 188.382 tok/s | -0.272% | -0.252% | -0.291% |
| C1 cache-busted prefill | 1,684.410 tok/s | 1,673.877 tok/s | -0.625% | -0.695% | -0.556% |
| C2 aggregate prefill | 1,692.938 tok/s | 1,687.643 tok/s | -0.313% | -0.332% | -0.294% |
| C4 aggregate prefill | 1,695.037 tok/s | 1,694.027 tok/s | -0.060% | -0.080% | -0.040% |

The C1 regression reproduced in both launch orders. C2 did not reach the 3%
promotion threshold, and C4 was flat. All four arms ran at zero worker swap
with no allocator or EngineCore failures. Power, clocks, and temperatures were
comparable under the fixed 230 W and 1650 MHz safety limits.

Fresh post-BIOS Nsight traces used 0.979 GPU-memory utilization for both arms
because profiler startup overhead narrowly failed the 0.98 reservation check.
Both retained the 262,144-token API limit. Across four GPUs, enabling early
launch increased median C1 graph span by 7.31% and C2 span by 6.29%. GPU busy
time moved only 0.84% and 1.62%. The added third stream therefore changed the
schedule without shortening the critical path.

The x8 result closes the prior topology question for this workload. The old x4
link did not cause the mixed early-launch result. Keep the selector default-off
and reject this implementation as a production optimization.

The initial compact evidence bundle is
[`evidence/qwen38-shared-expert-early-launch-20260831/`](evidence/qwen38-shared-expert-early-launch-20260831/README.md).
The post-BIOS benchmark and fresh trace are in
[`evidence/qwen38-shared-expert-post-bios-x8-20260831/`](evidence/qwen38-shared-expert-post-bios-x8-20260831/README.md).

## Delivery contract

The current handoff identity is:

| Item | Value |
| --- | --- |
| Base production image | `sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef` |
| Installed `envs.py` | `cbe6272965f373a2e490a4b16171a4301277b5ad0f6889654183b3d1d1121b4c` |
| Installed `moe_runner.py` | `a19ecf2417c80b93c0fe5594e74528f686665847732196f9b29ef967b38110b0` |
| Installed `shared_experts.py` | `4216c125cecc05181a94f745101603e90e7c06bb98030036dd1c1acfe5095ff6` |
| Compose project/profile | `qwen38-qsa-fp8-candidate` / `qwen38-flash-next` |
| Profile-resolved production Compose SHA-256 | `0b8b5e857b3b550107817c7c682a89b76c2442ed3a6d3ccda23b39948e4045ab` |
| Production restore command | `/home/will/inference/runtime/qwen38-qsa-fp8-candidate/restore-fp8.sh` |
| Restore script SHA-256 | `d18463d486f803864fa1fb6b09040fb574f63fa0b87b80e6ef83c82a4035e34f` |

`MANIFEST.json` is the identity authority. `scripts/qwen38_verify_shared_expert_rollback.sh` verifies every repository-owned delivery artifact plus the restore script, profile-resolved production Compose, Compose project/profile, service name, and immutable image. `scripts/qwen38_shared_expert_restore_watchdog.sh arm` creates the user-systemd restore timer. When the timer fires, `scripts/qwen38_execute_shared_expert_restore.sh` reruns the verifier immediately before invoking the restore script.

`scripts/qwen38_build_shared_expert_early_launch_image.sh` reads the same manifest, extracts only the three Python files, runs the fail-closed patcher, builds a thin image, labels every input and output hash, and verifies the files inside the image. It does not start a service or use a GPU.

If the production image changes, stop. Rebase the three-file adapter against the new production source and rerun the CPU contract tests. Do not weaken the hashes.

## GPU acceptance protocol

The completed pre-BIOS and post-BIOS runs used this protocol.

1. Run `scripts/qwen38_verify_shared_expert_rollback.sh`. Then verify zero-swap state and the 230 W / 210-1650 MHz safety policy.
2. Run `scripts/qwen38_shared_expert_restore_watchdog.sh arm`. Require its timer to be active before stopping production.
3. Build the thin candidate without GPU access. Verify its labels and installed source hashes.
4. Run a bounded operator test on one RTX 3090. Compare selector-off and selector-on shared outputs exactly. Exercise M=1, M=2, both DBO slots, the threshold boundary, and fallback cases.
5. Capture and replay the operator in CUDA Graphs. Require deterministic output and no stale slot state.
6. Run Compute Sanitizer memcheck and racecheck over eager and graph paths. Require zero errors and zero hazards.
7. Launch a non-restarting TP=4 candidate with an exact rollback watchdog. Confirm the early-launch log appears once on every rank.
8. Run deterministic text, arithmetic, automatic tool, post-tool, multimodal, two-stream, and long-context NIAH checks.
9. Capture one phase-separated decode trace. Require earlier shared-expert submission, real overlap, and a shorter shared-plus-routed critical segment. A second stream alone is not success.
10. Run the existing matched matrix with the same image and only the selector changed: three warmups and five measured decode runs at concurrency 1 and 2, plus cache-busted prefill at both concurrency levels. Record worker swap, allocator retries, VRAM growth, power, clocks, and thermals.
11. Promote only if the causal trace and performance threshold pass together. Otherwise remove the candidate and record a no-go.
12. Restore the exact production service. After every health and safety check passes, run `scripts/qwen38_shared_expert_restore_watchdog.sh cancel`. Verify health, model identity, restart policy, zero swap, safety controls, and absence of rollback timers before releasing server60.

## Decision

Reject the current early-launch mechanism as a server60 production
optimization. The post-BIOS x8 retest reproduced a 9.97% C1 decode regression,
left C2 below the 3% threshold, and showed no C4 benefit. The fresh trace showed
longer C1 and C2 graph spans even though GPU busy time changed little.

Keep the implementation default-off on its experiment branch as a tested
reference. Do not promote or retest it unchanged. A future attempt needs a new
mechanism that removes the added dependency tail rather than moving the same
shared-expert work to another stream.
