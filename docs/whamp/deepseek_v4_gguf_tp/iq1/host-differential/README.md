# Host differential proof — reproduction

Proves the GPU-free IQ1/Q456 kernel rewrite preserves arithmetic exactly,
without a GPU.

```sh
cd /home/will/projects/club-3090/.worktrees/gguf-tp-iq1-unsloth/.research/gguf-tp-iq1/host-differential

# Regenerate the old-tables include from the git parent revision (optional;
# the committed iq1_iq3_tables_old.inc was generated this way):
git -C /home/will/projects/vllm/.worktrees/gguf-tp-iq1-unsloth \
  show 'e7982b484~1:csrc/libtorch_stable/quantization/gguf_dsv4/iq1_iq3_tables.cuh' \
  | python3 -c '
import sys
body = sys.stdin.read().split("#pragma once", 1)[1]
for name in ("kIqSigns", "kIq3XXSGrid", "kIq1SGrid"):
    body = body.replace(name, name + "Old")
body = body.replace("__device__ ", "").replace("__constant__ ", "")
# keep only the outer namespace wrapper
lines, seen = [], False
for line in body.split("\n"):
    if line.strip() == "namespace vllm::gguf_dsv4 {":
        if seen:
            continue
        seen = True
    lines.append(line)
text = "\n".join(lines).rstrip()
print("// AUTO-GENERATED from Whamp/vLLM git parent e7982b484~1 -- do not edit.")
print("#include <cstdint>")
print(text)
' > iq1_iq3_tables_old.inc

g++ -O2 -std=c++17 \
  -I/home/will/projects/vllm/.worktrees/gguf-tp-iq1-unsloth/csrc/libtorch_stable/quantization/gguf_dsv4 \
  -o host_diff test_host_differential.cpp
./host_diff
```

Expected output ends with `ALL HOST DIFFERENTIAL CHECKS PASSED`.

Notes:

- The harness includes the REAL `iq1_iq3_tables.cuh` via `-I`, with
  `__device__`/`__forceinline__` defined empty and host shims for `__dp4a`
  (four signed byte products) and half→float conversion.
- The new `q45_group_dot` is transcribed into the harness; if the kernel
  changes again, update the transcription and re-run before trusting it.
