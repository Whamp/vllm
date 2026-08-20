#!/usr/bin/env bash
set -euo pipefail
env -u VLLM_IMAGE CLUB3090_RESTART=no ESTATE_CONTAINER=dsv4-fp4-indexer-final docker compose -p dsv4-fp4-indexer-final -f /home/will/build/vllm-fp4/indexer-sm86-review-final/compose.yml down --remove-orphans || true
