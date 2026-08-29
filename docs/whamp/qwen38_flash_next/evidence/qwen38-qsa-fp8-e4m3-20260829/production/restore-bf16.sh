#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
runtime=/home/will/inference/runtime/qwen38-qsa-fp8-candidate
expected_fp8_image=sha256:61971a78222e89335d84a7f4d72b0e8842619a4a29564582a58ff328af48abb9
container=qwen38-qsa-fp8-candidate
if docker inspect "$container" >/dev/null 2>&1; then
  [[ "$(docker inspect "$container" --format '{{.Image}}')" == "$expected_fp8_image" ]]
fi
docker compose -p qwen38-qsa-fp8-candidate \
  --profile qwen38-flash-next -f "$runtime/production.yml" \
  down --remove-orphans
exec /home/will/inference/runtime/qwen38-util-0968/rollback.sh
