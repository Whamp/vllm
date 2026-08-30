# Qwen3.8 calibrated QSA FP8 evidence

This directory preserves the kernel, calibration, serving, production, and
llama-benchy evidence for server60's promoted QSA E4M3 cache profile.

Run this from the directory to verify the archive:

```bash
sha256sum -c SHA256SUMS
```

## Contents

`kernel-gates/` contains the executed RTX 3090 scripts, the 54-profile SM86
search, and the accepted result. `calibration/` contains the workload list and
four TP-rank reports. `qsa-fp8-scales.json` is the promoted merged scale file.

`serving/` contains startup logs, model identity, the internal decode/prefill
benchmark, and the 261,544-token NIAH acceptance result.

`llama-benchy/` contains Will's original c=1 and c=2 result files, the exact
llama-benchy result calculation source, the launcher, and a deterministic
phase-correct summary. The source checkout was at commit
`b220b7c9cae7af2d6bd9ebf6bfa9ac066cb40780`. The source is stored byte-for-byte as deterministic `results.py.gz`; its decompressed SHA-256
is `2c7ac6e6a8f8781a013af96276f805e4d2fe0b7f449b4e3ee7db8cade838acc3`.

`production/` contains the digest-pinned Compose profile, BF16 and FP8 restore
scripts, image Dockerfile, exact deployed runtime overlay, and the tested build
contract. The production image remains pinned to
`sha256:61971a78222e89335d84a7f4d72b0e8842619a4a29564582a58ff328af48abb9`.
A rebuild verifies the base image, overlay file hashes, and labels. Its OCI
manifest can differ because build metadata is not part of the runtime contract.

## llama-benchy correction

The c=2 cold-run `tg_throughput` spans from the earliest first token to the
latest last token across both requests. It includes the interval where one
request decodes while the other still prefills. It measures mixed phase
interference, not steady concurrent decode.

Use the cached records for decode scaling. They measure 48.21 tokens/s at c=1
and 65.68 aggregate tokens/s at c=2, or 1.36x aggregate scaling and 68.1%
parallel efficiency.
