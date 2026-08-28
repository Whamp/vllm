# DeepSeek V4 FP4 DS-MLA cache

[REPORT.md](REPORT.md) records the FP4 cache format, implementation, matched FP8 comparison, defects found during testing, and the decision not to make FP4 the default.

Implementation commits:

- [`Whamp/forks-flash-mla-int@81a06aa6`](https://github.com/Whamp/forks-flash-mla-int/commit/81a06aa6feb608bcba687a40acf60ee87d14f2da) contains native SM86 FP4 sparse-MLA decode and prefill.
- [`Whamp/vllm@633815f6`](https://github.com/Whamp/vllm/commit/633815f6889d9d033aefa04bf40cb270d5b6a3f1) contains cache allocation, writers, readers, paging, attention selection, and tests.

The build manifest and Compose profile remain in the Whamp/club-3090 deployment fork. [EVIDENCE.md](../EVIDENCE.md) points to the raw logs and checksums now archived in Whamp/vLLM.
