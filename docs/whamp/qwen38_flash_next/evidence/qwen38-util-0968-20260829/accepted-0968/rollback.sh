#!/usr/bin/env bash
set -euo pipefail
runtime=/home/will/inference/runtime/qwen38-util-0968
current_runtime=/home/will/inference/runtime/qwen38-rope-bound
candidate=qwen38-util-0968-candidate
production=qwen38-flash-next-intel-autoround-vllm
production_image=sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b
trap 'sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null || true' EXIT
docker compose -p qwen38-util-0968 --profile qwen38-flash-next -f "$runtime/candidate.yml" down --remove-orphans || true
sudo -n swapoff -a
sudo -n sysctl -q -w vm.overcommit_memory=1 >/dev/null
if docker inspect "$production" >/dev/null 2>&1; then docker start "$production" >/dev/null; else docker compose -p qwen38-rope-prod --profile qwen38-flash-next -f "$current_runtime/production.yml" up -d; fi
for poll in $(seq 1 240); do
 if curl -fsS http://127.0.0.1:30001/health >/dev/null 2>&1; then break; fi
 [[ "$(docker inspect "$production" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]
 if (( poll==240 )); then docker logs --tail 500 "$production" >&2; exit 1; fi
 sleep 15
done
sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null
trap - EXIT
[[ "$(docker inspect "$production" --format '{{.Image}}')" == "$production_image" ]]
[[ "$(docker inspect "$production" --format '{{.RestartCount}}')" == 0 ]]
[[ "$(cat /proc/swaps | wc -l)" == 1 ]]
for pid in $(docker top "$production" -o pid | tail -n +2); do [[ "$(awk '/^VmSwap:/{print $2}' /proc/$pid/status)" == 0 ]]; done
echo 'RoPE-bound Qwen3.8 service restored healthy with zero swap.'
