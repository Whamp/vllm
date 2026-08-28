# DeepSeek V4 GGUF-TP evidence archive

This branch preserves the raw evidence behind the DeepSeek V4 GGUF tensor-parallel research. The readable reports live in `docs/whamp/deepseek_v4_gguf_tp` on Whamp/vLLM main.

Archived sources:

- `engine/` came from Whamp/club-3090 branch `feat/gguf-tp-engine` at commit `7f8ab8314acbc7cb9c8994ad48b3795d5b176e2e`.
- `fp4_kv/` came from Whamp/club-3090 branch `feat/deepseek-v4-gguf-tp-q4-kv` at commit `32263ba51cf421c2e4785f200654d160af143b91`.
- `iq1/` combines committed IQ1 reports at commit `ced745427b4da4e7933232f9b961d2b6bee8d54d` with the untracked cross-engine oracle dumps present on 2026-08-28.

The generated `host-differential/host_diff` executable was excluded. Its source and build instructions are archived.

Run `sha256sum -c SHA256SUMS` from this directory to verify the archive.
