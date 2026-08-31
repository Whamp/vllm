# Shared-expert early-launch delivery

This directory contains the thin-image contract for the default-off Qwen3.8
CUDA shared-expert early-launch experiment.

The first server60 gate completed on 2026-08-31 and missed the 3% decode
promotion threshold. A post-BIOS retest then raised GPU0 from x4 to x8 and ran
both selector orders. Early launch regressed C1 decode by 9.97%, improved C2 by
only 0.90%, and left C4 flat. Fresh traces showed longer C1 and C2 CUDA Graph
spans. The current mechanism is rejected and remains default-off. See the
[initial evidence](../../evidence/qwen38-shared-expert-early-launch-20260831/README.md)
and [post-BIOS evidence](../../evidence/qwen38-shared-expert-post-bios-x8-20260831/README.md).

Do not deploy this candidate. Keep it as a tested reference. If a future design
changes the mechanism, update and review `MANIFEST.json` before any GPU run.

After an explicit GPU release, verify and arm rollback before stopping production:

```bash
scripts/qwen38_verify_shared_expert_rollback.sh
scripts/qwen38_shared_expert_restore_watchdog.sh arm
```

Then build without GPU access from a checkout of the reviewed branch:

```bash
REPORT_DIR=/home/will/build/qwen38-shared-expert-early-launch/report \
TARGET_IMAGE=whamp/vllm:qwen38-shared-expert-early-launch \
  scripts/qwen38_build_shared_expert_early_launch_image.sh
```

The builder:

1. requires the exact base image ID in `MANIFEST.json`;
2. verifies the three installed runtime source hashes;
3. applies `scripts/qwen38_patch_shared_expert_early_launch.py`;
4. records every derived output hash;
5. builds a three-file Python overlay;
6. verifies the files inside the resulting image.

The image remains behaviorally identical until the service sets:

```bash
VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH=1
```

The builder does not start a service. The watchdog invokes `qwen38_execute_shared_expert_restore.sh`, which re-verifies every pinned rollback identity when the timer fires. Follow the preregistered GPU acceptance and rollback sequence in [the experiment report](../../SHARED-EXPERT-EARLY-LAUNCH.md). Cancel the watchdog only after the exact production service is healthy and every final-state check passes.
