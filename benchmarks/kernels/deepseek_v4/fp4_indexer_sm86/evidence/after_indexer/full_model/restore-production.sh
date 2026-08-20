#!/usr/bin/env bash
set -euo pipefail
env -u VLLM_IMAGE CLUB3090_RESTART=no ESTATE_CONTAINER=dsv4-fp4-indexer-sm86 docker compose -p dsv4-fp4-indexer-sm86 -f /home/will/build/vllm-fp4/indexer-sm86-full-model-175k/compose.yml down --remove-orphans || true
env -u VLLM_IMAGE docker compose -p dsv4-gguf-tp-prod -f /home/will/inference/runtime/gguf-tp-prod/base.yml up -d
