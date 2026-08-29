#!/usr/bin/env bash
set -euo pipefail

runtime=/home/will/inference/runtime/qwen38-rope-bound
candidate_compose=$runtime/candidate.yml
promoted_compose=$runtime/production.yml
control_compose=/home/will/inference/runtime/qwen38-qsa-1024/control.yml
production_container=qwen38-flash-next-intel-autoround-vllm
production_image=sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3
production_url=http://127.0.0.1:30001

restore_overcommit() {
  sudo -n sysctl -q -w vm.overcommit_memory=0 >/dev/null || true
}
trap restore_overcommit EXIT

docker compose -p qwen38-rope-bound --profile qwen38-flash-next \
  -f "$candidate_compose" down --remove-orphans || true
docker compose -p qwen38-rope-prod --profile qwen38-flash-next \
  -f "$promoted_compose" down --remove-orphans || true
docker rm -f "$production_container" >/dev/null 2>&1 || true
sudo -n swapoff -a
sudo -n sysctl -q -w vm.overcommit_memory=1 >/dev/null
actual=$(docker image inspect "$production_image" --format '{{.Id}}')
[[ "$actual" == "$production_image" ]]
docker tag "$production_image" vllm/vllm-openai:qwen38-flash-next
docker compose -p qwen38-rope-prod --profile qwen38-flash-next \
  -f "$control_compose" up -d

for attempt in $(seq 1 240); do
  if curl -fsS "$production_url/health" >/dev/null 2>&1; then
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$production_container" 2>/dev/null || true)" != true ]]; then
    docker logs --tail 300 "$production_container" >&2 || true
    exit 1
  fi
  if (( attempt == 240 )); then
    docker logs --tail 300 "$production_container" >&2 || true
    exit 1
  fi
  sleep 15
done
restore_overcommit
trap - EXIT

[[ "$(docker inspect -f '{{.Image}}' "$production_container")" == "$production_image" ]]
[[ "$(docker inspect -f '{{.RestartCount}}' "$production_container")" == 0 ]]
[[ "$(cat /proc/swaps | wc -l)" == 1 ]]
[[ "$(systemctl is-active gpu-power-limit.service)" == active ]]
for pid in $(docker top "$production_container" -o pid | tail -n +2); do
  swap_kib=$(awk '/^VmSwap:/{print $2}' "/proc/$pid/status")
  [[ "${swap_kib:-0}" == 0 ]]
done
curl -fsS "$production_url/v1/models" \
  | grep -q 'qwen3.8-flash-next-intel-autoround-w4a16'
echo 'Control Qwen3.8 service restored healthy with zero swap.'
