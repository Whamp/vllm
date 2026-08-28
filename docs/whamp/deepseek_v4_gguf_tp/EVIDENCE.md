# Raw evidence

The readable reports are in this directory. Raw logs, numerical dumps, traces, manifests, and route captures are stored on a separate Whamp/vLLM branch so they do not add 134 MB to main.

Immutable archive:

- Commit: [`123d78ec92c498dbd4ea4dc335f717f1f5b4a94e`](https://github.com/Whamp/vllm/tree/123d78ec92c498dbd4ea4dc335f717f1f5b4a94e/research_archive/deepseek_v4_gguf_tp)
- Branch: `archive/deepseek-v4-gguf-tp-evidence-20260828`
- Files: 3,205 checksummed files plus the checksum manifest
- Size: about 134 MB

The archive contains:

- Native Antirez GGUF-TP implementation and runtime evidence
- FP4 DS-MLA kernel, quality, capacity, and performance evidence
- Unsloth IQ1_S and IQ1_M cross-engine oracle dumps and reports
- The three historical writing drafts

Run `sha256sum -c SHA256SUMS` from the archive directory to verify it. The generated `host-differential/host_diff` executable was excluded. Its source and build instructions remain in the archive.

Earlier copies remain in Whamp/club-3090 branches for provenance, but Whamp/vLLM is now the permanent home.
