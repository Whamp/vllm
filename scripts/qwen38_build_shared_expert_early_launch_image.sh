#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

readonly DEFAULT_TARGET_IMAGE="whamp/vllm:qwen38-shared-expert-early-launch"
readonly INSTALLED_ROOT="/usr/local/lib/python3.12/dist-packages"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
patch_script="$repo_root/scripts/qwen38_patch_shared_expert_early_launch.py"
identity_manifest="${QWEN38_SHARED_EXPERT_MANIFEST:-$repo_root/docs/whamp/qwen38_flash_next/experiments/shared-expert-early-launch/MANIFEST.json}"
target_image="${TARGET_IMAGE:-$DEFAULT_TARGET_IMAGE}"
report_dir="${REPORT_DIR:-$repo_root/.build/qwen38-shared-expert-early-launch}"
mkdir -p "$report_dir"
context="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-shared-expert.XXXXXX")"
base_tag="whamp/qwen38-shared-expert-base:$$"
extract_container="qwen38-shared-expert-extract-$$"

cleanup() {
    docker rm -f "$extract_container" >/dev/null 2>&1 || true
    docker image rm "$base_tag" >/dev/null 2>&1 || true
    rm -rf "$context"
}
trap cleanup EXIT

mapfile -t delivery_identity < <(
    python3 - "$identity_manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest["current_production_contract"]["base_image_id"])
print(manifest["delivery"]["patch_script"]["sha256"])
print(manifest["delivery"]["build_script"]["sha256"])
print(manifest["delivery"]["dockerfile"]["path"])
print(manifest["delivery"]["dockerfile"]["sha256"])
PY
)
if [[ "${#delivery_identity[@]}" -ne 5 ]]; then
    echo "Shared-expert delivery manifest is incomplete" >&2
    exit 1
fi
expected_base_image_id="${delivery_identity[0]}"
expected_patch_script_sha256="${delivery_identity[1]}"
expected_build_script_sha256="${delivery_identity[2]}"
dockerfile="$repo_root/${delivery_identity[3]}"
expected_dockerfile_sha256="${delivery_identity[4]}"

actual_patch_script_sha256="$(sha256sum "$patch_script" | cut -d' ' -f1)"
if [[ "$actual_patch_script_sha256" != "$expected_patch_script_sha256" ]]; then
    echo "Shared-expert patch script SHA-256 mismatch" >&2
    exit 1
fi
actual_build_script_sha256="$(sha256sum "$0" | cut -d' ' -f1)"
if [[ "$actual_build_script_sha256" != "$expected_build_script_sha256" ]]; then
    echo "Shared-expert build script SHA-256 mismatch" >&2
    exit 1
fi
actual_dockerfile_sha256="$(sha256sum "$dockerfile" | cut -d' ' -f1)"
if [[ "$actual_dockerfile_sha256" != "$expected_dockerfile_sha256" ]]; then
    echo "Shared-expert Dockerfile SHA-256 mismatch" >&2
    exit 1
fi

actual_base_image_id="$(docker image inspect "$expected_base_image_id" --format '{{.Id}}')"
if [[ "$actual_base_image_id" != "$expected_base_image_id" ]]; then
    echo "Qwen3.8 shared-expert base image mismatch" >&2
    exit 1
fi

docker tag "$expected_base_image_id" "$base_tag"
docker create --name "$extract_container" "$base_tag" >/dev/null

runtime_root="$context/runtime"
runner_dir="$runtime_root/vllm/model_executor/layers/fused_moe/runner"
mkdir -p "$runner_dir"
docker cp \
    "$extract_container:$INSTALLED_ROOT/vllm/envs.py" \
    "$runtime_root/vllm/envs.py"
docker cp \
    "$extract_container:$INSTALLED_ROOT/vllm/model_executor/layers/fused_moe/runner/moe_runner.py" \
    "$runner_dir/moe_runner.py"
docker cp \
    "$extract_container:$INSTALLED_ROOT/vllm/model_executor/layers/fused_moe/runner/shared_experts.py" \
    "$runner_dir/shared_experts.py"
docker rm "$extract_container" >/dev/null

manifest="$report_dir/runtime-patch-manifest.json"
python3 "$patch_script" \
    --root "$runtime_root" \
    --identity-manifest "$identity_manifest" \
    --manifest "$manifest" >/dev/null

mapfile -t patch_hashes < <(
    python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for path in (
    "vllm/envs.py",
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
    "vllm/model_executor/layers/fused_moe/runner/shared_experts.py",
):
    print(manifest["files"][path]["input_sha256"])
    print(manifest["files"][path]["output_sha256"])
PY
)
if [[ "${#patch_hashes[@]}" -ne 6 ]]; then
    echo "Shared-expert runtime manifest is incomplete" >&2
    exit 1
fi

docker build \
    --build-arg "BASE_IMAGE=$base_tag" \
    --build-arg "BASE_IMAGE_ID=$expected_base_image_id" \
    --build-arg "ENV_INPUT_SHA256=${patch_hashes[0]}" \
    --build-arg "ENV_OUTPUT_SHA256=${patch_hashes[1]}" \
    --build-arg "MOE_RUNNER_INPUT_SHA256=${patch_hashes[2]}" \
    --build-arg "MOE_RUNNER_OUTPUT_SHA256=${patch_hashes[3]}" \
    --build-arg "SHARED_EXPERTS_INPUT_SHA256=${patch_hashes[4]}" \
    --build-arg "SHARED_EXPERTS_OUTPUT_SHA256=${patch_hashes[5]}" \
    --file "$dockerfile" \
    --tag "$target_image" \
    "$context" | tee "$report_dir/image-build.log"

docker image inspect "$target_image" > "$report_dir/image-inspect.json"
docker run --rm --network none --entrypoint /bin/sh "$target_image" -c \
    "sha256sum \
$INSTALLED_ROOT/vllm/envs.py \
$INSTALLED_ROOT/vllm/model_executor/layers/fused_moe/runner/moe_runner.py \
$INSTALLED_ROOT/vllm/model_executor/layers/fused_moe/runner/shared_experts.py" \
    > "$report_dir/image-runtime-files.sha256"

python3 - "$manifest" "$report_dir/image-runtime-files.sha256" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
observed = {
    line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
    for line in Path(sys.argv[2]).read_text().splitlines()
}
installed_root = "/usr/local/lib/python3.12/dist-packages/"
for path, identity in manifest["files"].items():
    installed_path = installed_root + path
    assert observed[installed_path] == identity["output_sha256"], installed_path
PY

printf 'TARGET_IMAGE=%s\n' "$target_image"
printf 'TARGET_IMAGE_ID=%s\n' "$(docker image inspect "$target_image" --format '{{.Id}}')"
printf 'REPORT_DIR=%s\n' "$report_dir"
