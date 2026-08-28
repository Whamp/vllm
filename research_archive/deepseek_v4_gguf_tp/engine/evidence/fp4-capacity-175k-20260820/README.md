# Antirez FP4 KV 175K capacity evidence

Source: `/home/will/build/vllm-fp4/goal-capacity/175k` on server60
Copied into Git: 2026-08-28
Test date: 2026-08-20

This bundle closes a gap in the original FP4 report, which recorded the matched 148K FP4 versus FP8 comparison but not the later 160K, 170K, and 175K capacity campaign.

The 175K profile reported 178,050 KV tokens and 1.02x concurrency. `verify-stress-99pct.log` records exact recall at 173,058 tokens, all functional stress probes passing, 27 MiB free VRAM, and zero serving-process swap. `verify-full.log` records completion, tool, streaming, reasoning, and anti-degeneration success. `final-container-state.txt` records healthy status and zero restarts.

The profile was not promoted because 27 MiB per card was too little operating margin. This was a safety decision, not a functional failure.

Run `sha256sum -c SHA256SUMS` from this directory to verify the copied evidence.
