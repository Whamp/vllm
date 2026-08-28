# DeepSeek V4 GGUF runtime history on server60

Last updated: 2026-08-28

This is the short path to the DeepSeek V4 GGUF work on server60. It records the Antirez and Unsloth artifacts, both serving engines, the cache experiments, measured throughput, validated context, and the reason each profile was or was not promoted.

The most important correction is simple:

> The native GGUF tensor-parallel engine was not limited to 148K context. The 148K FP8 profile was the conservative production rollback. Antirez plus FP4 KV passed the full functional suite and exact retrieval at 173,058 tokens with a 175,000-token limit. The Unsloth IQ1 artifacts exposed larger aggregate KV pools at a 148K configured limit, but their larger configured-context limits were never tested because they were much slower.

## How to read the numbers

These measurements answer different questions. Do not substitute one for another.

- **Configured context** is the server's `max_model_len` or llama.cpp context reservation.
- **KV tokens** is the aggregate token capacity reported for the allocated cache. It is not proof that one request was served at that length.
- **Validated recall** is the longest prompt that returned the exact planted value.
- **Decode** is generated tokens per second under the stated benchmark.
- **Prefill** is prompt-processing throughput. Different prompt lengths can produce different rates.
- **Headroom** is physical free VRAM after startup or load. A profile can fit while remaining unsafe for sustained use.

## Result matrix

| Artifact and runtime | Cache | Configured context | Reported KV tokens | Longest validated recall | Decode | Prefill | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Antirez IQ2_XXS, specialized llama.cpp | Q8_0 K/V | 430,080 | not comparable to vLLM pool reporting | 395,282 | 33 to 38 tok/s | about 913 tok/s at the 395K test | Canonical high-context llama.cpp profile |
| Antirez IQ2_XXS, native GGUF-TP, first full acceptance | FP8 DS-MLA | 140,000 | 154,519 | 119,730 | 76.70 tok/s | 551.89 tok/s at about 9K | Passed M5 to M8 and established the engine |
| Antirez IQ2_XXS, native GGUF-TP production profile | FP8 DS-MLA | 148,000 | 156,373 to 156,738 | 136K in later stress | 79.84 narrative, 79.82 code | 541.79 at 10K, 518.82 at 93K | Conservative production rollback |
| Antirez IQ2_XXS, native GGUF-TP FP4 comparison | FP4 DS-MLA | 148,000 | 180,039 | 136K | 80.36 narrative, 80.37 code | 524.87 at 10K, 495.79 at 93K | Validated opt-in, not default |
| Antirez IQ2_XXS, native GGUF-TP FP4 capacity run | FP4 DS-MLA | 175,000 | 178,050 | **173,058** | not rebenchmarked at 175K | 466.4 tok/s at the 173K recall rung | Functional high-context success, rejected for 27 MiB headroom |
| Unsloth UD-IQ1_S, native GGUF-TP | FP8 DS-MLA | 148,000 | **421,244** | 136,032 | 40.80 engine, 40.31 wall | 251.62 at 10K, 250.79 at 90K | Functional, but too slow and only about 1 MiB free VRAM |
| Unsloth UD-IQ1_M, native GGUF-TP | FP8 DS-MLA | 148,000 | **223,001** | no larger-context campaign | 44.79 engine, 44.25 wall | 275.39 at 10K | Functional, but too slow for deeper evaluation |
| Antirez IQ2_XXS, experimental DCP=4 | FP8 DS-MLA | 148,000 | 155,810 with explicit 400 MB pool | 136K | 37.58 narrative, 37.53 code | 296.7 to 460.9 across tested depths | Correct but 53% slower than production |
| Antirez IQ2_XXS, experimental DCP=4 ceiling | FP8 DS-MLA | 262,144 | 373,421 | 240K probe timed out | not rebenchmarked | not accepted | Startup success only, unsafe and unvalidated at the ceiling |

### What the table establishes

- The highest fully validated native GGUF-TP context was **175K with FP4 KV**, including exact recall at 173,058 tokens.
- The fastest native GGUF-TP profile was the Antirez artifact at about **80 tok/s**.
- UD-IQ1_S had the largest reported non-DCP KV pool, **421,244 tokens**, but the server remained capped at 148K during that campaign. No result proves a 421K request.
- DCP reached 262K startup, but the 240K request timed out and decode at the validated 148K profile fell to about 37.5 tok/s.
- The specialized llama.cpp profile still held the largest validated context, 395,282 tokens inside a 430,080-token reservation, at much lower decode throughput.

## Artifacts and quantization

### Antirez IQ2_XXS

Pinned GGUF:

- file size: 86,720,111,488 bytes
- SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- tensor payload: 80.7594 GiB
- routed gate and up projections: IQ2_XXS
- routed down projections: Q2_K
- attention, shared experts, and output: mostly Q8_0

The specialized Whamp/llama.cpp service and the native vLLM GGUF-TP service consumed the same model artifact through different execution designs.

### Unsloth UD-IQ1_S

Pinned Hugging Face revision:

- repository: `unsloth/DeepSeek-V4-Flash-0731-GGUF`
- revision: `109848da2469efe1f1aab9e11acea08a065ccd4`
- three-shard size: 82,539,237,792 bytes, 76.870655 GiB
- routed gate/up: IQ1_S on 30 layers and IQ2_XXS on 13
- routed down: IQ3_XXS on 41 layers and MXFP4 on layers 26 and 42
- shared experts: Q5_K, Q6_K, and Q8_0 by tensor
- embedding and output: Q4_K

The exact shard sizes and SHA-256 values live in `ARTIFACT-IDENTITIES.json` on Whamp/club-3090 commit `44fe8a81`.

### Unsloth UD-IQ1_M

The same immutable repository revision contains UD-IQ1_M:

- three-shard size: 86,901,313,952 bytes, 80.933155 GiB
- routed gate/up: IQ1_M on 22 layers and IQ2_XXS on 21
- routed down and non-routed allocation: the same format families as UD-IQ1_S

UD-IQ1_M was slightly larger than the Antirez artifact but produced a larger KV pool than the Antirez FP8 profile because its transformed runtime residency differed.

## Runtime history

### 1. Specialized llama.cpp established the quality and context reference

The canonical Antirez fast-prefill profile used Whamp/llama.cpp revision `0379cf4bf889f3d28038a005210c4bc193fc8ba1` and image digest `sha256:a96bd947d63eb81d8baf9f6f5ecb26669476383976717237450fbb5727b03745`.

Its profile reserved 430,080 tokens with Q8_0 K/V, batch 8192, ubatch 384, parallel 1, layer split `1,1,0.95,1.05`, and a required 429,568-token warmup. It recalled the exact value at 395,282 tokens, processed that prompt at about 913 tok/s, and decoded at 33 to 38 tok/s.

This profile remained the high-context and known-quality rollback. It was not the same runtime as the later native GGUF-TP engine.

### 2. Native GGUF-TP established fast execution without requantizing the model

The native engine loaded the Antirez GGUF directly into custom SM86 kernels. It used tensor parallelism across all four RTX 3090s, native IQ2_XXS and Q2_K routed-expert kernels, Q8_0 Marlin linears, Marlin-diagonal `wo_a`, Ampere FlashMLA sparse decode, and hierarchical all-reduce.

The first complete 140K run passed deterministic output, automatic tools, post-tool continuation, exact 119,730-token recall, a 27/30 quick gate, and the one-cell SuperJSON DeepSWE pilot. It measured 76.70 decode tok/s and 551.89 prefill tok/s. The later seq2 148K profile measured about 79.8 decode tok/s.

This work disproved the assumption that preserving the GGUF formats required llama.cpp's serial layer split. It also showed that file size alone does not predict runtime capacity. The initial profile loaded 21.53 GiB per rank and left only about 100 MiB free after cache allocation.

Primary report: `M5-M7-RUNTIME.md`. Behavioral gate: `M8-DEEPSWE.md`.

### 3. FP4 DS-MLA raised validated native context to 175K

`fp4_ds_mla` reduced one DeepSeek MLA cache row from 584 bytes to 368 bytes while retaining BF16 RoPE values. The 148K matched comparison increased the cache pool from 156,373 to 180,039 tokens. Decode stayed flat. Prefill fell by 3.1% at 10K and 4.4% at 93K. FP8 and FP4 both scored 27/30 with the same quick-gate failures.

The later capacity campaign tested 160K, 170K, and 175K. At 175K:

- startup reported 178,050 cache tokens, only 3,050 above the configured request;
- the stress suite recalled exact values at 115K and 173,058 tokens;
- tool-prefill, IDE-agent, multi-turn, coding, and reasoning probes passed;
- `verify-full.sh` passed basic completion, tools, streaming, streaming tools, reasoning, and anti-degeneration checks;
- serving-process swap stayed zero;
- the container remained healthy with zero restarts;
- only 27 to 28 MiB VRAM remained per card.

This was a real functional success and a safety rejection. The test did not fail. It was not promoted because 27 MiB leaves no practical margin for late allocations, JIT, or sustained mixed work.

The checksum-bound evidence copied from server60 is in `evidence/fp4-capacity-175k-20260820/`.

### 4. Native Unsloth IQ1 support worked after a real format bug was fixed

The engine added split-GGUF loading and native IQ1_S, IQ1_M, IQ3_XXS, MXFP4, Q4_K, Q5_K, and Q6_K execution. The initial synthetic IQ1 tests passed by construction while both the test oracle and CUDA kernel used the wrong grid-nibble ordering. A real-weight replay against pinned llama.cpp exposed the defect. llama.cpp used lane-interleaved IQ1 grid nibbles and the required delta corrections.

The corrected kernels passed:

- real Unsloth weight replays;
- 21 corrected IQ1 cases;
- the full 47-test native-format numerical and CUDA Graph suite;
- Compute Sanitizer memcheck;
- grouped-kernel racecheck;
- SM86 cubin and SASS checks.

Canonical vLLM correction: `c9c376b85`. Production-compatible port: `ed311edca`.

#### UD-IQ1_S outcome

The corrected model reached TP=4 readiness at a 148K configured limit. Startup reported 421,244 aggregate KV tokens and 2.85x declared concurrency. This capacity was not exercised as one 421K request.

It passed deterministic generation, automatic tools, post-tool continuation, `verify-full.sh`, stress probes through 136,032 tokens, and a 26/30 quick quality gate. Its performance was the blocker:

- narrative decode: 40.80 engine tok/s, 40.31 wall tok/s;
- code decode: 40.79 engine tok/s, 40.08 wall tok/s;
- cache-busted prefill: 251.62 tok/s at 10K and 250.79 at 90K;
- about 1 MiB free VRAM per card in the final benchmark state.

#### UD-IQ1_M outcome

UD-IQ1_M also reached TP=4 readiness at 148K. Startup reported 223,001 aggregate KV tokens and 1.51x declared concurrency. It passed deterministic completion, automatic tools, post-tool continuation, streaming, reasoning, and the anti-degeneration probe.

Measured performance:

- narrative decode: 44.79 engine tok/s, 44.25 wall tok/s;
- code decode: 44.77 engine tok/s, 44.01 wall tok/s;
- cache-busted 10K prefill: 275.39 tok/s.

The campaign stopped before the 90K prefill run, full quality pack, or a higher configured-context test because it was still about half as fast as Antirez.

The Unsloth work therefore established functional support, not a production recommendation. Its large cache pools remain useful capacity evidence but are not validated maximum contexts.

### 5. Decode context parallelism traded most of the speed for cache capacity

The DCP path sharded compressed MLA history across the four TP ranks while keeping sliding-window and compressor state replicated. The implementation required exact compressed-entry ownership, global sparse-indexer top-k, partial FlashMLA decode, FP32 LSE merge, and one-time attention-sink application.

After fixing replicated-SWA double counting and C128 physical-slot translation, the 148K DCP=4 profile passed exact recall through 136K and the functional stress suite. It decoded at about 37.5 tok/s, 53% below production.

The 262,144-token profile reached readiness with 373,421 KV tokens, but left 75 MiB at idle and 11 MiB during the 240K probe. That request timed out. DCP remained experimental and production returned to the Antirez FP8 profile.

Primary report: `DCP-SM86.md`.

## Decisions and non-decisions

### Promoted or retained

- The native Antirez GGUF-TP engine became the fast serving path.
- The 148K FP8 DS-MLA profile remained the conservative rollback.
- The specialized Antirez llama.cpp profile remained the high-context rollback.
- FP4 DS-MLA remained a validated opt-in cache format.

### Not promoted

- FP4 at 175K fit and worked, but 27 MiB of headroom was too small.
- UD-IQ1_S and UD-IQ1_M worked, but decode and prefill were about half the Antirez speed.
- DCP worked at 148K, but its 37.5 tok/s decode rate was too slow.
- DCP at 262K started but did not pass the 240K request.

### Never established

- A 421K UD-IQ1_S request was never served. `421,244` is an aggregate pool count.
- A 223K UD-IQ1_M request was never served. `223,001` is an aggregate pool count.
- Antirez FP4 decode was not rebenchmarked at the 175K limit. The 80.36 tok/s result belongs to the matched 148K comparison.
- DCP never produced a safe, validated 262K service.

The separate safetensors WNA16 service is intentionally excluded from this matrix. It reached different speed and context results, but it did not preserve the GGUF quantization and later failed the broader coding-agent comparison. Mixing those results with native GGUF-TP was one source of earlier confusion.

## Evidence map

### In this research directory

- `M5-M7-RUNTIME.md`: first native Antirez GGUF-TP full-model acceptance
- `M8-DEEPSWE.md`: one-cell coding-agent behavioral gate
- `CAPACITY.md`: original native-engine residency model and measured fit
- `DCP-SM86.md`: DCP implementation, correctness fixes, performance, and 262K limit
- `evidence/fp4-capacity-175k-20260820/`: copied 175K startup, allocation, stress, functional, VRAM, swap, and container-state evidence
- `evidence/m5-m7-runtime/`: first full-model Antirez runtime bundle
- `evidence/dcp-sm86-20260828/`: DCP result bundle

### Other Whamp/club-3090 commits

- [`32263ba5`](https://github.com/Whamp/club-3090/commit/32263ba51cf421c2e4785f200654d160af143b91): FP4 DS-MLA implementation, matched report, and 148K evidence
- [`44fe8a81`](https://github.com/Whamp/club-3090/commit/44fe8a81b9ebaa385f241f80c637d2eff86a83b7): Unsloth shard identities and source-format profiles
- [`20c39d29`](https://github.com/Whamp/club-3090/commit/20c39d29353bb86001dd5b2508b6c4db367187c8): IQ1 measured baseline and performance-preparation record
- [`a816987f`](https://github.com/Whamp/club-3090/commit/a816987f1c4c1ee89c256747ca37b6d63b8446a5): DCP implementation and server60 evidence

### Whamp/vllm implementation lines

- `3ec20cebe`: native Antirez GGUF-TP full-model runtime used for M5 to M8
- `633815f6889d9d033aefa04bf40cb270d5b6a3f1`: FP4 DS-MLA vLLM integration
- `81a06aa6feb608bcba687a40acf60ee87d14f2da`: FP4 native FlashMLA implementation in `Whamp/forks-flash-mla-int`
- `c9c376b85`: corrected canonical IQ1 lane mapping
- `ed311edca`: corrected production-compatible IQ1 port
- `00793b3e52`: final experimental DCP branch formatting head; functional changes precede it on the same branch

## Last verified server state

After the DCP campaign on 2026-08-28, server60 returned to `dsv4-gguf-tp-prod` on image `sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf`, with restart policy `unless-stopped`, zero restarts, zero serving-process swap, and a successful deterministic response. `gpu-power-limit.service` remained active at 230 W and 210 to 1650 MHz on all four RTX 3090s.

This is a historical state assertion, not a live health check. Verify the current service before operational work.
