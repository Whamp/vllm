#!/usr/bin/env bash
set -euo pipefail
r=/home/will/inference/runtime/qwen38-hier-allreduce-ab
base_tag=vllm/vllm-openai:qwen38-qsa-fp8-e4m3-dev
base_id=sha256:61971a78222e89335d84a7f4d72b0e8842619a4a29564582a58ff328af48abb9
[[ "$(docker image inspect "$base_tag" --format '{{.Id}}')" == "$base_id" ]]
docker build --network=none -t vllm/vllm-openai:qwen38-qsa-fp8-hier-ab "$r"
image_id=$(docker image inspect vllm/vllm-openai:qwen38-qsa-fp8-hier-ab --format '{{.Id}}')
docker run --rm --entrypoint sha256sum "$image_id" \
 /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py \
 /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/hier_all_reduce.py
