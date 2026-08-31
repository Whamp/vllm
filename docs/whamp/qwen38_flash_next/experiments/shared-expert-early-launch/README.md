# Shared-expert early-launch delivery

This directory contains the thin-image contract for the default-off Qwen3.8
CUDA shared-expert early-launch experiment.

The first server60 gate completed on 2026-08-31. It did not clear the 3% decode
promotion threshold, but C2 aggregate decode and both prefill measures improved.
The candidate remains default-off for a matched retest after the planned BIOS
work on server60's PCIe Gen3 x4 link. The current test does not establish that
the x4 link caused the mixed result. See
[the evidence bundle](../../evidence/qwen38-shared-expert-early-launch-20260831/README.md).

Do not run it while another agent owns server60. Do not run it after the
production image changes without updating and reviewing `MANIFEST.json`.

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
