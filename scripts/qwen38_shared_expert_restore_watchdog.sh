#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

readonly UNIT_NAME="qwen38-shared-expert-restore"
readonly DEFAULT_DELAY="2h"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${QWEN38_SHARED_EXPERT_MANIFEST:-$repo_root/docs/whamp/qwen38_flash_next/experiments/shared-expert-early-launch/MANIFEST.json}"
verifier="$repo_root/scripts/qwen38_verify_shared_expert_rollback.sh"
restore_executor="$repo_root/scripts/qwen38_execute_shared_expert_restore.sh"
action="${1:-status}"

case "$action" in
    arm)
        "$verifier" >/dev/null
        if systemctl --user is-active --quiet "$UNIT_NAME.timer"; then
            echo "Shared-expert restore watchdog is already active" >&2
            exit 1
        fi
        systemd-run \
            --user \
            --unit "$UNIT_NAME" \
            --on-active "${RESTORE_DELAY:-$DEFAULT_DELAY}" \
            --timer-property Persistent=true \
            --collect \
            /usr/bin/env \
            "QWEN38_SHARED_EXPERT_MANIFEST=$manifest" \
            "$restore_executor"
        systemctl --user is-active --quiet "$UNIT_NAME.timer"
        ;;
    cancel)
        systemctl --user stop "$UNIT_NAME.timer" "$UNIT_NAME.service" \
            >/dev/null 2>&1 || true
        systemctl --user reset-failed "$UNIT_NAME.timer" "$UNIT_NAME.service" \
            >/dev/null 2>&1 || true
        ;;
    status)
        systemctl --user status "$UNIT_NAME.timer" --no-pager
        ;;
    *)
        echo "Usage: $0 {arm|cancel|status}" >&2
        exit 2
        ;;
esac
