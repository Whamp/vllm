#!/usr/bin/env bash
# shellcheck disable=SC2086
set -euo pipefail
runtime=/home/will/inference/runtime/qwen38-util-0968
production=qwen38-flash-next-intel-autoround-vllm
image=sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b
trap 'sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null || true' EXIT
docker compose -p qwen38-util-0968 --profile qwen38-flash-next -f "$runtime/candidate.yml" down --remove-orphans || true
docker compose -p qwen38-rope-prod --profile qwen38-flash-next -f "$runtime/production.yml" down --remove-orphans || true
docker rm -f "$production" >/dev/null 2>&1 || true
sudo -n swapoff -a
sudo -n sysctl -q -w vm.overcommit_memory=1 >/dev/null
docker compose -p qwen38-rope-prod --profile qwen38-flash-next -f "$runtime/control.yml" up -d
for poll in $(seq 1 240); do
 if curl -fsS http://127.0.0.1:30001/health >/dev/null 2>&1; then break; fi
 [[ "$(docker inspect "$production" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]
 if ((poll==240)); then docker logs --tail500 "$production" >&2; exit 1; fi
 sleep 15
done
sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null
trap - EXIT
[[ "$(docker inspect "$production" --format '{{.Image}}')" == "$image" ]]
[[ "$(docker inspect "$production" --format '{{.RestartCount}}')" == 0 ]]
[[ "$(cat /proc/swaps|wc -l)" == 1 ]]
for pid in $(docker top "$production" -o pid|tail -n+2); do [[ "$(awk '/^VmSwap:/{print $2}' /proc/$pid/status)" == 0 ]]; done
curl -fsS http://127.0.0.1:30001/v1/models | grep -q '"max_model_len":167600'
echo '0.95 Qwen3.8 control restored healthy with zero swap.'
