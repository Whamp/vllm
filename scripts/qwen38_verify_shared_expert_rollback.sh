#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${QWEN38_SHARED_EXPERT_MANIFEST:-$repo_root/docs/whamp/qwen38_flash_next/experiments/shared-expert-early-launch/MANIFEST.json}"

python3 - "$manifest" "$repo_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
repo_root = Path(sys.argv[2])
for name, artifact in manifest["delivery"].items():
    path = repo_root / artifact["path"]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != artifact["sha256"]:
        raise RuntimeError(f"Shared-expert delivery artifact mismatch: {name}")
PY

mapfile -t rollback_identity < <(
    python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text())["current_production_contract"]
for key in (
    "base_image_id",
    "service_name",
    "compose_project",
    "compose_profile",
    "container_name",
    "served_model_name",
    "host_port",
    "production_compose",
    "resolved_compose_sha256",
    "restore_script",
    "restore_script_sha256",
):
    print(contract[key])
PY
)
if [[ "${#rollback_identity[@]}" -ne 11 ]]; then
    echo "Shared-expert rollback manifest is incomplete" >&2
    exit 1
fi

base_image_id="${rollback_identity[0]}"
service_name="${rollback_identity[1]}"
compose_project="${rollback_identity[2]}"
compose_profile="${rollback_identity[3]}"
container_name="${rollback_identity[4]}"
served_model_name="${rollback_identity[5]}"
host_port="${rollback_identity[6]}"
production_compose="${rollback_identity[7]}"
resolved_compose_sha256="${rollback_identity[8]}"
restore_script="${rollback_identity[9]}"
restore_script_sha256="${rollback_identity[10]}"

actual_restore_sha256="$(sha256sum "$restore_script" | cut -d' ' -f1)"
if [[ "$actual_restore_sha256" != "$restore_script_sha256" ]]; then
    echo "Shared-expert rollback script SHA-256 mismatch" >&2
    exit 1
fi

compose_args=(
    compose
    -p "$compose_project"
    --profile "$compose_profile"
    -f "$production_compose"
)
actual_compose_sha256="$(
    docker "${compose_args[@]}" config | sha256sum | cut -d' ' -f1
)"
if [[ "$actual_compose_sha256" != "$resolved_compose_sha256" ]]; then
    echo "Shared-expert resolved production Compose SHA-256 mismatch" >&2
    exit 1
fi

actual_base_image_id="$(docker image inspect "$base_image_id" --format '{{.Id}}')"
if [[ "$actual_base_image_id" != "$base_image_id" ]]; then
    echo "Shared-expert rollback image mismatch" >&2
    exit 1
fi

if ! docker "${compose_args[@]}" config --services | grep -Fxq "$service_name"; then
    echo "Shared-expert rollback service is absent from production Compose" >&2
    exit 1
fi

printf 'ROLLBACK_READY=1\n'
printf 'BASE_IMAGE_ID=%s\n' "$base_image_id"
printf 'SERVICE_NAME=%s\n' "$service_name"
printf 'COMPOSE_PROJECT=%s\n' "$compose_project"
printf 'COMPOSE_PROFILE=%s\n' "$compose_profile"
printf 'CONTAINER_NAME=%s\n' "$container_name"
printf 'SERVED_MODEL_NAME=%s\n' "$served_model_name"
printf 'HOST_PORT=%s\n' "$host_port"
printf 'RESTORE_SCRIPT=%s\n' "$restore_script"
