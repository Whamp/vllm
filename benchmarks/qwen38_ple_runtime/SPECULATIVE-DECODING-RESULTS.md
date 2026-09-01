# Qwen3.8 speculative decoding results

- Date: 2026-09-01
- Host: server60, 4× RTX 3090, PCIe Gen3, no NVLink
- Model: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`
- Runtime: Whamp/vLLM image `sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef`

## Verdict

Native MTP materially accelerates decode on this host. The balanced profile is:

- native MTP with two draft tokens;
- routed MTP experts quantized to symmetric RTN W4A16, group size 128, GPTQ layout;
- FP8 E4M3 QSA KV cache with scales calibrated for the quantized draft;
- async scheduling;
- CUDA graph capture sizes `[1, 2, 4, 8, 12, 16, 24]`.

The profile improves primary matched decode throughput by 32.8% at C1, 24.6% at C2, and 17.5% at C4. It retains 317,179 KV-cache tokens, 74.5% of the 425,497-token no-spec baseline, and serves the native 262,144-token context at 1.21× maximum concurrency.

Keep the no-spec profile available. It retains the most cache and prefills 6.7–7.5% faster. The MTP profile is the better default when decode latency or aggregate generation throughput matters more than the final 25.5% of cache capacity.

## Matched primary results

All rows use the same cache-busted service benchmark, 256 generated tokens, and the C1/C2/C4 concurrency curve.

| Profile | C1 decode tok/s | C2 decode tok/s | C4 decode tok/s | C1/C2/C4 prefill tok/s | KV tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| No spec, FP8 QSA | 67.53 | 117.84 | 206.87 | 1688.71 / 1696.14 / 1701.77 | 425,497 |
| MTP K1, BF16 experts, FP8 QSA | 78.73 | 132.51 | 233.37 | 1585.33 / 1600.42 / 1609.35 | 205,920 |
| MTP K2, BF16 experts, FP8 QSA | 86.83 | 143.22 | 243.56 | 1565.55 / 1582.02 / 1589.38 | 196,800 |
| MTP K3, BF16 experts, FP8 QSA | 89.34 | 148.94 | 228.88 | 1561.80 / 1566.22 / 1587.77 | 190,400 |
| **MTP K2, INT4 experts, FP8 QSA** | **89.67** | **146.85** | **243.11** | 1488.89 / 1539.60 / 1588.84 | **317,179** |

The candidate's initial C1 prefill point was low. A dedicated five-round replication measured 1561.26 / 1579.07 / 1587.18 tok/s at C1/C2/C4 with population CV below 0.11%. Use those stable prefill values for the balance comparison.

Decode-only replication measured:

- C1: 91.22 tok/s over ten rounds, CV 3.44%;
- C4: 255.49 tok/s over ten rounds, CV 6.02%.

The primary candidate run accepted 63.48% of draft tokens with mean acceptance length 2.270. The BF16-expert FP8 K2 profile accepted 62.55% with mean length 2.251. INT4 draft experts did not reduce matched-workload acceptance.

The deployed image cannot embed speculative metrics in each API response. Eight requests were therefore run in isolation with Prometheus snapshots before and after each request. Counter deltas give exact per-request metrics when the engine has no concurrent traffic. Acceptance ranged from 56.2% to 86.7%, averaged 68.8%, and produced mean acceptance length 2.376. Position-level acceptance and request metadata are in `runtime/qwen38-spec-decode-study/mtp-k2-int4experts-fp8-cg12/per-request-spec-metrics.json`.

## Context and concurrency proof

The candidate started with 317,179 GPU KV-cache tokens. A real API stress test submitted two distinct prompts concurrently, each with 150,020 input tokens. Both requests completed without an OOM, restart, or capacity error. The service remained healthy with restart count zero.

The test used unrelated word sequences to prevent prefix sharing between requests. `ignore_eos=true` forced fixed-length output, so special-token text after each correct summary is expected and is not a model-quality result.

## Output check

A six-prompt greedy suite covered arithmetic, code, Chinese, JSON, factual recall, and GPU systems explanations. Every candidate answer and every no-spec production answer was semantically correct. Candidate and production text matched exactly on two prompts and differed in wording or equivalent implementation on four.

Exact text is not a valid gate on this runtime. A second no-spec production run matched its own first run on only one of six prompts despite `temperature=0`. Expert-parallel and async execution can change floating-point tie-breaking. The candidate did not introduce a unique determinism failure. Draft acceptance remains the useful direct check because speculative verification still uses the target model.

Evidence:

- `runtime/qwen38-spec-decode-study/deterministic-equivalence-comparison.json`
- `runtime/qwen38-spec-decode-study/deterministic-nospec-repeatability.json`

## Why K2 C4 originally collapsed

The first BF16 K2 run fell from about 240 tok/s at C3 to about 52 tok/s at C4. Nsight traces initially exposed long hierarchical all-reduce waits, but two controls disproved that as the cause:

- raising the one-shot/two-shot threshold to 65,536 elements left C4 at 52.51 tok/s;
- disabling hierarchical all-reduce left C4 at 51.23 tok/s.

The deployed V2 CUDA graph manager rounded capture sizes to K2's three-token verification width. Its candidate set contained 3, 6, 9, 18, and 24 tokens, then rejected 18 and 24 because `max_num_seqs × (K + 1) = 12`. It never captured the required 12-token C4 graph. C3 replayed a nine-token graph; C4 ran the 48-layer target eagerly.

Adding capture size 12 recovered C4 to 206.93 tok/s in the quick probe and 238.49 tok/s in the first full BF16 run. Fork main already contains the general fix in commit `d3d79ffc1` (`Capture the widest uniform decode batch by default`); the deployed image predates it.

Nsight evidence:

- `runtime/qwen38-spec-decode-study/nsys-k2-bf16/k2-c3-64t.nsys-rep`
- `runtime/qwen38-spec-decode-study/nsys-k2-bf16/k2-c4-64t.nsys-rep`

## Selective INT4 draft conversion

The original MTP sidecar contains 5,214,301,696 tensor bytes. Its 1,536 routed-expert matrices account for 96.5% of the file. The converter changes only those matrices:

- source expert `.weight` tensors become `.qweight`, `.scales`, and `.qzeros`;
- MTP attention, embeddings, hyperconnection mixers, norms, gates, and shared experts remain bit-identical BF16;
- the target-model shards remain untouched;
- the source snapshot and production release remain read-only.

The derived sidecar contains 1,488,580,096 tensor bytes and saves 3,725,721,600 bytes (3.47 GiB). Independent verification found:

- 1,536 qweight, 1,536 scales, and 1,536 qzeros tensors;
- all 29 non-expert sidecar tensors bit-identical;
- zero shape, scale, or qzero errors;
- every sidecar tensor present in the rewritten index;
- INC resolves only `mtp.layers.48.mlp.experts` to W4 group 128;
- MTP attention still resolves to 16-bit;
- the service logs `Using MarlinExperts`.

A stratified sample of 45 expert matrices measured aggregate cosine similarity 0.9921, NRMSE 12.6%, and SNR 18.0 dB. This is RTN quantization, not AutoRound optimization. Runtime acceptance and target verification, rather than weight-space error alone, determine whether the draft remains useful.

Converter commit: `5b7092294` on `perf/qwen38-int4-mtp-experts`.

Derived view:

- `runtime/qwen38-spec-decode-study/model-mtp-int4-experts`
- sidecar SHA-256 `674e7102079286693b1fa5e5ddec5619575010f07fa1d5dab6c6ebcdfc2db2fa`

Fresh cumulative QSA scales:

- `runtime/qwen38-spec-decode-study/qsa-calibration-k2-int4experts/qsa-fp8-scales-mtp-k2-int4experts.json`
- SHA-256 `b3dc568549dfd3735ffc81c7e637770777087d2e5ddcfa8e52cf18fdc549cb18`

## Other speculative methods

### MTP depth

K3 improves C1 and C2 slightly but loses 6.0% at C4 versus FP8 K2 and consumes another 6,400 cache tokens. Draft acceptance falls to 51.83%. Stop at K2 for the balanced profile; K4 is not justified.

### GPU n-gram

GPU n-gram requires the V1 runner and compilation mode 3 in this deployed image. Against a matched V1+mode3 no-spec control:

- novel prose loses 87–88% at C1/C2/C4 because acceptance is 13.08%;
- deterministic repetition gains 56% / 43% / 35% with 96.97% acceptance.

N-gram is a per-request accelerator for repeatable content, not a global default.

### DFlash, DFlash2, and DSpark

Qwen4Exp validation permits only `mtp`, `ngram`, and `ngram_gpu`. It rejects DFlash, DFlash2, DSpark, and other draft-model methods with:

> Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear n-gram proposers

Testing those methods would require a new architecture integration and a compatible Qwen3.8 draft checkpoint. No such candidate was part of this campaign.

## Why speculation helps now

The current host epoch differs from the one in which prior speculative tests failed:

- one GPU moved from PCIe Gen3 x4 to x8;
- the current topology has GPU0/GPU2 at x8 and GPU1/GPU3 at x16;
- the runtime now has optimized QSA, PLE, hierarchical communication, async scheduling, and CUDA graph coverage;
- this is the first speculative-decoding test on Qwen3.8-Flash-Next itself.

Measured one-second PCIe peaks during MTP K1 were about 1.7–2.0 GB/s per GPU, below Gen3 x8 saturation. The successful MTP result therefore does not prove that PCIe was the sole cause of earlier failures. The older x4 link remains a plausible contributor, but prior runs have no traces or matched acceptance data. The defensible conclusion is that hardware topology, runtime overhead, CUDA graph coverage, and workload acceptance all changed; the old failures cannot isolate one cause.

## Production recommendation

Use `compose-mtp-k2-int4experts-fp8-cg12.yml` as the promotion candidate. Preserve the current no-spec production profile as the rollback and as an explicit maximum-capacity/prefill profile.

The campaign restored no-spec production after testing. The restore procedure printed `QWEN38_RELEASE_VERIFIED=1` and `QWEN38_PRODUCTION_RESTORED=1`; the production container is healthy with zero restarts.

Before replacing the production default:

1. canary representative user traffic while watching acceptance, TTFT, prefill, and cache pressure;
2. run the relevant domain-quality evaluation if production traffic requires a stronger gate than target verification and the six-prompt semantic check;
3. keep capture size 12 until the deployed image includes commit `d3d79ffc1` or an equivalent backport.

Primary candidate evidence is under:

- `runtime/qwen38-spec-decode-study/mtp-k2-int4experts-fp8-cg12/`
- `runtime/qwen38-spec-decode-study/qsa-calibration-k2-int4experts/`
- `runtime/qwen38-spec-decode-study/model-mtp-int4-experts/`
