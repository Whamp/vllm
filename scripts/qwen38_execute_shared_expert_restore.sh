#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${QWEN38_SHARED_EXPERT_MANIFEST:-$repo_root/docs/whamp/qwen38_flash_next/experiments/shared-expert-early-launch/MANIFEST.json}"
verifier="$repo_root/scripts/qwen38_verify_shared_expert_rollback.sh"

QWEN38_SHARED_EXPERT_MANIFEST="$manifest" "$verifier" >/dev/null
restore_script="$(
    python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text())["current_production_contract"]
print(contract["restore_script"])
PY
)"
exec /usr/bin/bash "$restore_script"
