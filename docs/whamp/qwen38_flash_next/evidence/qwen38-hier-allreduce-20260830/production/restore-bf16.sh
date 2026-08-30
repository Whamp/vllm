#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
runtime=/home/will/inference/runtime/qwen38-qsa-fp8-candidate
expected_fp8_image=sha256:4b59067e269f313a78f0a698e79261230fb02e3712f42ffd54b3e9ec9be9705a
container=qwen38-qsa-fp8-candidate
if docker inspect "$container" >/dev/null 2>&1; then
  [[ "$(docker inspect "$container" --format '{{.Image}}')" == "$expected_fp8_image" ]]
fi
docker compose -p qwen38-qsa-fp8-candidate \
  --profile qwen38-flash-next -f "$runtime/production.yml" \
  down --remove-orphans
exec /home/will/inference/runtime/qwen38-util-0968/rollback.sh
