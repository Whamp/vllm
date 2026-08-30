#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
runtime=/home/will/inference/runtime/qwen38-qsa-fp8-candidate
base_tag=vllm/vllm-openai:qwen38-rope-bound-dev
expected_base=sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b
tag=${1:-vllm/vllm-openai:qwen38-qsa-fp8-e4m3-rebuild}
root=$runtime/root
cleanup() { rm -rf "$root"; }
trap cleanup EXIT
[[ "$(docker image inspect "$base_tag" --format '{{.Id}}')" == "$expected_base" ]]
rm -rf "$root"
mkdir -p "$root"
tar -xzf "$runtime/runtime-overlay.tar.gz" -C "$root"
(cd "$root" && sha256sum -c "$runtime/runtime-overlay.sha256")
(
  cd "$runtime"
  docker build --pull=false --network=none --provenance=false \
    -f Dockerfile.fp8-qsa -t "$tag" .
)
docker run --rm --entrypoint sha256sum "$tag" \
  /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/common/qsa_fp8_calibration.py \
  /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/common/qsa_fp8.py \
  /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py \
  /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/qsa.py \
  | grep -F -f <(cut -d" " -f1 "$runtime/runtime-overlay.sha256") >/dev/null
[[ "$(docker image inspect "$tag" --format '{{index .Config.Labels "io.whamp.qwen38.base-image"}}')" == "$expected_base" ]]
[[ "$(docker image inspect "$tag" --format '{{index .Config.Labels "io.whamp.qwen38.qsa-kv-cache"}}')" == fp8_e4m3_direct_sm86 ]]
docker image inspect "$tag" --format '{{.Id}}'
