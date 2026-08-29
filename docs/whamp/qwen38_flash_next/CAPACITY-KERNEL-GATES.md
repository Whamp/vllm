# Qwen3.8 capacity-kernel gates

## Decision

The first purpose-specific compressed-hyperconnection and direct Q8-K/Q4-V QSA implementations are rejected for production. Both passed their independent numerical and deterministic CUDA-Graph gates on an RTX 3090, but both failed their preregistered serving-shape performance thresholds by large margins. Neither implementation was wired into a model image or used for a full-model launch.

The thresholds are unchanged:

- hyperconnection candidate time must be no greater than `1 / 0.90` of BF16 time at M=1, M=2, and M=256;
- QSA reader time must be no greater than 1.25 times BF16 at M=1 and M=256;
- hyperconnection normalized RMSE must be at most 0.02 with cosine at least 0.9999;
- QSA normalized RMSE must be at most 0.17 with cosine at least 0.985;
- CUDA-Graph replay must be finite and bitwise deterministic.

## Environment

The gates ran on one server60 NVIDIA GeForce RTX 3090, compute capability 8.6, under the fixed 230 W safety policy. The exact control image was `sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b`. Each attempt used a fail-closed two-hour production-restore watchdog. The accepted 202,400-token Intel AutoRound service was restored after every attempt.

The hyperconnection source was Whamp/vLLM branch `feat/qwen38-hyperconnection-int8-sm86` at commit `070ffd51032a4261d9b051d8e8274d19c99fa8ba`. The QSA source was branch `feat/qwen38-qsa-q8k-q4v-sm86` at commit `7de82f834dfd28642c13b461143fc13c8a71e515`.

## Hyperconnection INT8 result

The representation used K-group-128 FP16 scales for the merged down/injection projection and one FP16 scale per output row for the up projection. It retained no BF16 weight copy. The tested layer-0 shapes were `[336, 10240]` and `[10240, 320]`.

| Matrix | M | NRMSE | Cosine | BF16 µs | Candidate µs | BF16/candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged down | 1 | 0.011363 | 0.999936 | 27.707 | 111.469 | 0.249 |
| merged down | 2 | 0.010612 | 0.999944 | 28.018 | 112.073 | 0.250 |
| merged down | 256 | 0.010193 | 0.999948 | 49.132 | 91.003 | 0.540 |
| up | 1 | 0.012643 | 0.999922 | 23.050 | 113.268 | 0.204 |
| up | 2 | 0.013099 | 0.999914 | 22.766 | 113.994 | 0.200 |
| up | 256 | 0.012166 | 0.999926 | 34.611 | 85.238 | 0.406 |

The tested matrices saved 3,386,880 and 3,256,320 bytes respectively relative to BF16 storage. The mechanism is therefore capacity-effective but execution-inefficient. The two-launch Triton decode path and per-prefill dequantization are rejected.

A later candidate is justified only if it changes the mechanism, such as one SM86 CUDA kernel that folds activation quantization into the skinny DP4A projection. Retiling the rejected Triton path is not sufficient evidence for another attempt.

## Direct Q8-K/Q4-V QSA result

The mixed row stores symmetric Q8 transformed keys and asymmetric packed Q4 transformed values. It uses 5,472 bytes per token and rank versus 13,056 bytes for BF16 QSA state, a 58.1% storage reduction. The direct reader quantizes transformed queries, computes integer query-key scores, folds value scales into quantized probabilities, and multiplies those probabilities by packed Q4 values.

| M | NRMSE | Cosine | BF16 µs | Candidate µs | Candidate/BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.116188 | 0.993452 | 133.673 | 562.268 | 4.206 |
| 256 | 0.115115 | 0.993530 | 604.928 | 1,038.536 | 1.717 |

The mixed writer took 371.507 µs for 2,051 rows. M=256 used the same 64-token tile and eight split-K schedule class as the BF16 path, so its 1.72-times reader cost is not explained by the M=1 split schedule alone. The direct Triton implementation is rejected.

Another QSA attempt requires a different mechanism and direct attribution. It must first show that a fused SM86 CUDA path can remove separate query RHT, query quantization, merge, inverse RHT, and copy work while making the integer core materially faster. The existing 1.25-times reader threshold remains binding.

## Evidence

The checksum-bound evidence is under [`evidence/qwen38-capacity-kernel-gates-20260829`](evidence/qwen38-capacity-kernel-gates-20260829/README.md). It includes exact result JSON, gate scripts, runner, and executed candidate sources.
