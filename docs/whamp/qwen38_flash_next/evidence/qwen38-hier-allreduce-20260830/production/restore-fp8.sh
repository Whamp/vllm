#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
runtime=/home/will/inference/runtime/qwen38-qsa-fp8-candidate
container=qwen38-qsa-fp8-candidate
expected_image=sha256:4b59067e269f313a78f0a698e79261230fb02e3712f42ffd54b3e9ec9be9705a
trap 'sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null || true' EXIT
docker rm -f qwen38-qsa-fp8-hier-ab >/dev/null 2>&1 || true
docker rm -f "$container" >/dev/null 2>&1 || true
sudo -n swapoff -a
sudo -n sysctl -q -w vm.overcommit_memory=1 >/dev/null
docker compose -p qwen38-qsa-fp8-candidate --profile qwen38-flash-next -f "$runtime/production.yml" up -d
for poll in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:30002/health >/dev/null 2>&1; then break; fi
  [[ "$(docker inspect "$container" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]
  if ((poll==240)); then docker logs --tail 1000 "$container" >&2; exit 1; fi
  sleep 15
done
sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null
[[ "$(docker inspect "$container" --format '{{.Image}}')" == "$expected_image" ]]
[[ "$(docker inspect "$container" --format '{{.State.Health.Status}}')" == healthy ]]
[[ "$(docker inspect "$container" --format '{{.RestartCount}}')" == 0 ]]
[[ "$(docker inspect "$container" --format '{{.HostConfig.RestartPolicy.Name}}')" == unless-stopped ]]
[[ "$(swapon --show --noheadings | wc -l)" == 0 ]]
for pid in $(docker top "$container" -o pid | tail -n +2); do
  [[ "$(awk '/^VmSwap:/{print $2}' /proc/"$pid"/status)" == 0 ]]
done
curl -fsS http://127.0.0.1:30002/v1/models | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; assert len(d)==1; assert d[0]["id"]=="qwen3.8-flash-next-intel-autoround-w4a16"; assert d[0]["max_model_len"]==262144'
echo "FP8 QSA plus hierarchical all-reduce Qwen3.8 service restored healthy with zero swap."
