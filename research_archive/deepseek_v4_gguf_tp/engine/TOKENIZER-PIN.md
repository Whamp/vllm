# TOKENIZER-PIN.md — M1 tokenizer contract and parity evidence

## Decision (PLAN §4.4)

The GGUF-TP engine uses the **HF `tokenizer.json`** (pinned SHA-256
`8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`, from
artifact revision `75d9286c…`, verified byte-identical on server60) with
`tokenizer_mode="deepseek_v4"` **pinned explicitly in the bootstrap config
before module/tokenizer construction** — never left to architecture
auto-selection (auto-selection exists at `vllm/config/model.py:644-660` and
would silently fall back to generic HF mode if the GGUF bootstrap omitted
the architecture hint). The GGUF's embedded chat template is **not** used by
the engine; conversation rendering is vLLM's `DeepseekV4Tokenizer`
(`vllm/tokenizers/deepseek_v4.py`, `encode_messages`).

## Parity evidence (static, M1) — PASS

Compared: GGUF-embedded tokenizer (extracted read-only from the pinned blob,
`evidence/gguf-tokenizer.json`) vs HF `tokenizer.json`.

- Base BPE vocab: ids 0–127,999 identical (every id → same token string).
- Added/special tokens: all 1,280 identical in id and content (GGUF
  `token_type` 3 = special, 4 = user-defined; HF `added_tokens`
  special/non-special correspond). Verified including the DSML set:
  `<think>`=128821, `</think>`=128822, `｜DSML｜`=128825, `<dsml:`=128840,
  `</dsml:`=128841, and all `<｜tool▁…｜>` markers — same ids on both sides.
- Merges: 127,741 entries, identical content **and order**.
- Model: gpt2 BPE, pre `joyai-llm` (GGUF) ↔ HF pre-tokenizer regex sequence.
- Control ids: bos=0, eos=1, pad=1 both sides; `add_bos/add_eos=false`.

**Consequence:** the token alphabet is provably identical, so any
encode-level difference between llama.cpp and the new engine can only come
from (a) pre-tokenization regex behavior or (b) conversation rendering —
not vocabulary. Both are covered by the runtime golden tests below.

## Runtime golden tests (M5, executable spec)

Text/API → token-ID goldens, run against the live engine
(`/v1/chat/completions` with `echo`/debug token ids, or offline
`DeepseekV4Tokenizer` in the container) and against the canonical llama.cpp
service's `/tokenize` (authoritative GGUF-side encoder, same blob):

1. Ordinary chat: 3 turns, system + user + assistant.
2. Reasoning: `thinking=high`/`max` renderings.
3. Tool call: assistant DSML tool-call turn.
4. Post-tool continuation: tool-result block then assistant.
5. Code-heavy content and unicode edge cases.

Pass rule: for identical *rendered text*, token ids identical between the
engine tokenizer and llama.cpp `/tokenize`; for identical *conversations*,
engine render via `encode_messages` compared against llama.cpp's
GGUF-template render token-by-token with differences limited to the
documented template divergence (below) — any other divergence is a defect.

## Known, accepted divergence

llama.cpp renders conversations with the GGUF's embedded Jinja template
(4,988 bytes, `tokenizer.chat_template`); vLLM renders with
`encode_messages`. The DSML-fix history (Whamp/vLLM `9a2ffbb4`) already
established these renderings are close but not byte-identical. Consequences
are bounded: both produce valid DS4 conversations consumed by the same
model; parity of *model behavior* is gated at M6 (per-layer/forward
windows) and M8 (DeepSWE), not assumed here.
