# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import qwen38_patch_shared_expert_early_launch as runtime_patch

REPO_ROOT = Path(__file__).parents[1]
LEGACY_RUNTIME_REVISION = "42b918e36fa3bdd04e3d7bd7ad4a9c7695b9624f"
INSTALLED_SHARED_EXPERTS_REVISION = "617d38d97b4dd8a90ad0ffaf15a4f64412470b25"
MANIFEST_PATH = REPO_ROOT / (
    "docs/whamp/qwen38_flash_next/experiments/shared-expert-early-launch/MANIFEST.json"
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def _git_source(revision: str, path: Path) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _write_exact_runtime(root: Path) -> dict[Path, str]:
    revisions = {
        runtime_patch.ENV_PATH: INSTALLED_SHARED_EXPERTS_REVISION,
        runtime_patch.MOE_RUNNER_PATH: LEGACY_RUNTIME_REVISION,
        runtime_patch.SHARED_EXPERTS_PATH: INSTALLED_SHARED_EXPERTS_REVISION,
    }
    source_hashes = {}
    for relative_path, revision in revisions.items():
        content = _git_source(revision, relative_path)
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        source_hashes[relative_path] = hashlib.sha256(content).hexdigest()
    return source_hashes


def test_runtime_patcher_transforms_exact_legacy_sources(tmp_path: Path) -> None:
    input_hashes = _write_exact_runtime(tmp_path)

    manifest = runtime_patch.patch_runtime(
        tmp_path,
        dry_run=False,
        expected_input_sha256=input_hashes,
    )

    assert manifest["selector"] == "VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH"
    assert manifest["default_enabled"] is False
    assert manifest["dry_run"] is False
    for relative_path in runtime_patch.PATCHERS:
        patched = (tmp_path / relative_path).read_text()
        compile(patched, str(relative_path), "exec")
        assert (
            manifest["files"][str(relative_path)]["input_sha256"]
            == (input_hashes[relative_path])
        )
        assert manifest["files"][str(relative_path)]["output_sha256"] == (
            hashlib.sha256(patched.encode()).hexdigest()
        )

    assert (
        "VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH"
        in (tmp_path / runtime_patch.ENV_PATH).read_text()
    )
    assert (
        "maybe_forward_async" in (tmp_path / runtime_patch.MOE_RUNNER_PATH).read_text()
    )
    assert (
        "maybe_forward_async"
        in (tmp_path / runtime_patch.SHARED_EXPERTS_PATH).read_text()
    )


def test_runtime_patcher_rejects_source_drift(tmp_path: Path) -> None:
    input_hashes = _write_exact_runtime(tmp_path)
    drifted_path = tmp_path / runtime_patch.MOE_RUNNER_PATH
    drifted_path.write_bytes(drifted_path.read_bytes() + b"\n# source drift\n")

    with pytest.raises(RuntimeError, match="runtime source mismatch"):
        runtime_patch.patch_runtime(
            tmp_path,
            dry_run=True,
            expected_input_sha256=input_hashes,
        )


def test_runtime_patcher_rejects_duplicate_application(tmp_path: Path) -> None:
    input_hashes = _write_exact_runtime(tmp_path)
    runtime_patch.patch_runtime(
        tmp_path,
        dry_run=False,
        expected_input_sha256=input_hashes,
    )

    with pytest.raises(RuntimeError, match="runtime source mismatch"):
        runtime_patch.patch_runtime(
            tmp_path,
            dry_run=False,
            expected_input_sha256=input_hashes,
        )


def _write_rollback_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    restore_script = tmp_path / "restore.sh"
    restore_script.write_text("#!/bin/sh\nexit 0\n")
    restore_script.chmod(0o755)
    compose_path = tmp_path / "production.yml"
    compose_path.write_text("services: {}\n")
    rendered_compose = tmp_path / "resolved-compose.yml"
    rendered_compose.write_text("services:\n  qwen38-flash-next: {}\n")
    base_image_id = "sha256:" + "a" * 64

    contract = {
        "base_image_id": base_image_id,
        "service_name": "qwen38-flash-next",
        "compose_project": "qwen38-test-project",
        "compose_profile": "qwen38-flash-next",
        "container_name": "qwen38-test",
        "served_model_name": "qwen38-test-model",
        "host_port": 30002,
        "production_compose": str(compose_path),
        "resolved_compose_sha256": hashlib.sha256(
            rendered_compose.read_bytes()
        ).hexdigest(),
        "restore_script": str(restore_script),
        "restore_script_sha256": hashlib.sha256(
            restore_script.read_bytes()
        ).hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    delivery = json.loads(MANIFEST_PATH.read_text())["delivery"]
    manifest_path.write_text(
        json.dumps(
            {
                "current_production_contract": contract,
                "delivery": delivery,
            }
        )
        + "\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_DOCKER_LOG"\n'
        'if [ "$1" = compose ]; then\n'
        '  case " $* " in\n'
        "    *' config --services '*) printf '%s\\n' \"$FAKE_SERVICE_NAME\" ;;\n"
        '    *) cat "$FAKE_COMPOSE_OUTPUT" ;;\n'
        "  esac\n"
        'elif [ "$1" = image ] && [ "$2" = inspect ]; then\n'
        "  printf '%s\\n' \"$FAKE_IMAGE_ID\"\n"
        "else\n"
        "  exit 64\n"
        "fi\n"
    )
    fake_docker.chmod(0o755)
    env: dict[str, str] = {
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
        "QWEN38_SHARED_EXPERT_MANIFEST": str(manifest_path),
        "FAKE_COMPOSE_OUTPUT": str(rendered_compose),
        "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
        "FAKE_SERVICE_NAME": str(contract["service_name"]),
        "FAKE_IMAGE_ID": base_image_id,
    }
    return restore_script, env


def _add_fake_user_systemd(tmp_path: Path, env: dict[str, str]) -> None:
    fake_bin = Path(env["PATH"].split(":", 1)[0])
    timer_state = tmp_path / "timer-active"
    systemd_run_log = tmp_path / "systemd-run.log"
    fake_systemctl = fake_bin / "systemctl"
    fake_systemctl.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'case " $* " in\n'
        "  *' is-active '*) [ -e \"$FAKE_TIMER_STATE\" ] ;;\n"
        "  *' stop '*) rm -f \"$FAKE_TIMER_STATE\" ;;\n"
        "  *' reset-failed '*) exit 0 ;;\n"
        "  *' status '*) [ -e \"$FAKE_TIMER_STATE\" ] ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n"
    )
    fake_systemctl.chmod(0o755)
    fake_systemd_run = fake_bin / "systemd-run"
    fake_systemd_run.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" > "$FAKE_SYSTEMD_RUN_LOG"\n'
        'touch "$FAKE_TIMER_STATE"\n'
    )
    fake_systemd_run.chmod(0o755)
    env.update(
        {
            "FAKE_TIMER_STATE": str(timer_state),
            "FAKE_SYSTEMD_RUN_LOG": str(systemd_run_log),
        }
    )


def test_rollback_verifier_accepts_exact_manifest(tmp_path: Path) -> None:
    _, env = _write_rollback_fixture(tmp_path)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts/qwen38_verify_shared_expert_rollback.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ROLLBACK_READY=1" in result.stdout
    assert "HOST_PORT=30002" in result.stdout
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text()
    assert "-p qwen38-test-project" in docker_log
    assert "--profile qwen38-flash-next" in docker_log


def test_rollback_verifier_rejects_changed_restore_script(tmp_path: Path) -> None:
    restore_script, env = _write_rollback_fixture(tmp_path)
    restore_script.write_text("#!/bin/sh\nexit 1\n")

    result = subprocess.run(
        [str(REPO_ROOT / "scripts/qwen38_verify_shared_expert_rollback.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "rollback script SHA-256 mismatch" in result.stderr


def test_restore_watchdog_reverifies_at_trigger_time(tmp_path: Path) -> None:
    restore_script, env = _write_rollback_fixture(tmp_path)
    _add_fake_user_systemd(tmp_path, env)
    watchdog = REPO_ROOT / "scripts/qwen38_shared_expert_restore_watchdog.sh"
    executor = REPO_ROOT / "scripts/qwen38_execute_shared_expert_restore.sh"

    subprocess.run(
        [str(watchdog), "arm"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    systemd_run = Path(env["FAKE_SYSTEMD_RUN_LOG"]).read_text()
    assert str(executor) in systemd_run
    assert str(restore_script) not in systemd_run

    subprocess.run(
        [str(executor)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    restore_script.write_text("#!/bin/sh\nexit 1\n")
    failed_restore = subprocess.run(
        [str(executor)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed_restore.returncode == 1
    assert "rollback script SHA-256 mismatch" in failed_restore.stderr


def test_delivery_manifest_matches_sources_and_tools() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    production_sources = manifest["current_production_contract"]["installed_sources"]
    assert production_sources == {
        str(path): sha256
        for path, sha256 in runtime_patch.load_runtime_identities(MANIFEST_PATH).items()
    }

    for path, expected_sha256 in manifest["canonical_candidate_sources"].items():
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == (
            expected_sha256
        )

    for artifact in manifest["delivery"].values():
        assert (
            hashlib.sha256((REPO_ROOT / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )
