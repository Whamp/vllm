# Qwen3.8 hierarchical all-reduce evidence

This directory preserves the compact acceptance record for server60's promoted
Qwen3.8 FP8-QSA hierarchical all-reduce profile.

## Layout

- `runtime-overlay/` contains the exact two-file compatibility overlay as deterministic gzip,
  the offline Docker build, source hashes, and four-GPU numerical and timing
  gate. Decompress `cuda_communicator.py.gz` and `hier_all_reduce.py.gz` into
  the build context before running `build.sh`. The exact executed gate is stored as
  `gate.executed.py.gz`.
- `production/` contains the promoted Compose contract plus FP8/PYNCCL and BF16
  rollback scripts.
- `results/` contains the collective gate, matched benchmarks, long-context and
  multimodal acceptance, BenchLocal quick result, exact-final state, GPU state,
  and server log. BenchLocal's generated Markdown is preserved as
  `benchlocal-quick.md.gz`.

`SHA256SUMS` covers every file in this directory. Verify it with:

```bash
sha256sum -c SHA256SUMS
```

## Result

The production service improved matched decode from 43.77 to 50.34 tokens/s and
concurrency-2 aggregate throughput from 53.25 to 59.00 tokens/s. Prefill stayed
flat at 1,538.14 tokens/s. Exact retrieval passed from a 261,544-token API
prompt, BenchLocal quick matched the accepted 26/30 score, and the final service
had zero restarts and zero swap.
