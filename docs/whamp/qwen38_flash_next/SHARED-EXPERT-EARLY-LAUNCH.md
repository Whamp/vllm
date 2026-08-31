# Qwen3.8 shared-expert early launch

Status: CPU-prepared, default-off, GPU-unvalidated.

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

`Mtplx` may change the production image before releasing server60. If any identity differs, stop. Rebase the three-file adapter against the new production source and rerun the CPU contract tests. Do not weaken the hashes.

## Preregistered GPU acceptance

Do not begin until `Mtplx` explicitly releases all four GPUs and reports service, timer, swap, and cleanup state.

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

Proceed to the bounded GPU gate after `Mtplx` releases server60. Do not merge the selector into production defaults before that gate. The likely outcomes are a small decode gain or a clean no-go. Upstream NVIDIA evidence makes a large gain unlikely.
