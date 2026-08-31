#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501  # Exact source fragments must remain byte-for-byte searchable.
"""Patch the exact Qwen3.8 production runtime for shared-expert early launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

ENV_PATH = Path("vllm/envs.py")
MOE_RUNNER_PATH = Path("vllm/model_executor/layers/fused_moe/runner/moe_runner.py")
SHARED_EXPERTS_PATH = Path(
    "vllm/model_executor/layers/fused_moe/runner/shared_experts.py"
)


def load_runtime_identities(manifest_path: Path) -> dict[Path, str]:
    """Load exact installed-source hashes from the delivery manifest."""
    payload = json.loads(manifest_path.read_text())
    sources = payload["current_production_contract"]["installed_sources"]
    if not isinstance(sources, dict):
        raise TypeError("Shared-expert manifest installed_sources must be an object")

    identities: dict[Path, str] = {}
    for path, sha256 in sources.items():
        if not isinstance(path, str) or not isinstance(sha256, str):
            raise TypeError("Shared-expert manifest source identities must be strings")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"Shared-expert manifest has invalid SHA-256 for {path}")
        identities[Path(path)] = sha256

    if set(identities) != set(PATCHERS):
        raise ValueError("Shared-expert manifest source inventory is incomplete")
    return identities


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Qwen shared-expert early-launch patch mismatch for {label}: "
            f"expected one match, found {count}"
        )
    return text.replace(old, new)


def patch_envs_source(text: str) -> str:
    """Add the default-off CUDA shared-expert early-launch selector."""
    text = _replace_once(
        text,
        "    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool = False\n",
        "    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool = False\n"
        "    VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH: bool = False\n",
        "environment type declaration",
    )
    registry_entry = (
        '    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(\n'
        '        int(os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0"))\n'
        "    ),\n"
    )
    return _replace_once(
        text,
        registry_entry,
        registry_entry
        + "    # Opt-in experiment: enqueue CUDA shared experts before routed expert\n"
        + "    # dispatch instead of relying on later host launch ordering for overlap.\n"
        + '    "VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH": lambda: bool(\n'
        + '        int(os.getenv("VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH", "0"))\n'
        + "    ),\n",
        "environment registry",
    )


def patch_shared_experts_source(text: str) -> str:
    """Add event-ordered early launch while retaining the legacy fallback."""
    text = _replace_once(
        text,
        "        self.enable_dbo = enable_dbo\n"
        "        self._output: list[torch.Tensor | None] = [None, None]\n"
        "        self._layer = layer\n",
        "        self.enable_dbo = enable_dbo\n"
        "        self._output: list[torch.Tensor | None] = [None, None]\n"
        "        self._async_in_flight = [False, False]\n"
        "        self._layer = layer\n",
        "DBO slot state",
    )
    text = _replace_once(
        text,
        "        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts\n",
        "        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts\n"
        "        self._cuda_early_launch = (\n"
        "            envs.VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH\n"
        "            and current_platform.is_cuda()\n"
        "        )\n",
        "early-launch selector snapshot",
    )
    text = _replace_once(
        text,
        "            if self._stream is not None:\n"
        '                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")\n'
        "\n"
        "    # TODO(bnell): Hack for elastic_ep. Get rid of this\n",
        "            if self._stream is not None:\n"
        '                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")\n'
        "\n"
        "        self._input_ready_event: list[torch.cuda.Event] = []\n"
        "        self._output_ready_event: list[torch.cuda.Event] = []\n"
        "        if self._stream is not None and self._cuda_early_launch:\n"
        "            self._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]\n"
        "            self._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]\n"
        '            logger.info_once("Enabled early launch for CUDA MoE shared experts")\n'
        "\n"
        "    # TODO(bnell): Hack for elastic_ep. Get rid of this\n",
        "per-DBO CUDA events",
    )
    text = _replace_once(
        text,
        "    def maybe_sync_shared_experts_stream(\n",
        '''    def maybe_forward_async(self, shared_experts_input: torch.Tensor) -> bool:
        """Enqueue CUDA shared experts before routed expert dispatch.

        Returns whether the caller must invoke :meth:`wait` before consuming
        :attr:`output`. The default-off selector preserves the legacy launch
        order and synchronization path.
        """
        if not self._cuda_early_launch:
            return False
        if (
            self._determine_shared_experts_order(shared_experts_input)
            != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False

        assert self._stream is not None
        idx = self._output_idx
        if self._output[idx] is not None or self._async_in_flight[idx]:
            raise RuntimeError(
                f"CUDA shared-expert early-launch slot {idx} is occupied"
            )
        if len(self._input_ready_event) != 2 or len(self._output_ready_event) != 2:
            raise RuntimeError("CUDA shared-expert early-launch events are unavailable")

        shared_experts_input.record_stream(self._stream)
        self._input_ready_event[idx].record(current_stream())
        output_event_recorded = False
        try:
            with torch.cuda.stream(self._stream):
                self._input_ready_event[idx].wait(self._stream)
                try:
                    self._output[idx] = self._layer(shared_experts_input)
                finally:
                    self._output_ready_event[idx].record(self._stream)
                    output_event_recorded = True
        except BaseException:
            try:
                if output_event_recorded:
                    self._output_ready_event[idx].wait(current_stream())
                else:
                    current_stream().wait_stream(self._stream)
            except BaseException as cleanup_error:
                logger.exception(
                    "CUDA shared-expert launch cleanup failed while preserving "
                    "the shared-layer error: %r",
                    cleanup_error,
                )
            self._output[idx] = None
            raise

        self._async_in_flight[idx] = True
        logger.debug_once("Early-launched CUDA MoE shared experts")
        return True

    def wait(self) -> None:
        """Order the current CUDA stream after one early shared-expert launch."""
        idx = self._output_idx
        if not self._async_in_flight[idx]:
            raise RuntimeError(
                f"CUDA shared-expert early-launch slot {idx} is not in flight"
            )
        self._output_ready_event[idx].wait(current_stream())
        self._async_in_flight[idx] = False

    def wait_and_discard_async_output(self) -> None:
        """Finish and discard early output after the routed path raises."""
        idx = self._output_idx
        try:
            if self._async_in_flight[idx]:
                self._output_ready_event[idx].wait(current_stream())
        finally:
            self._async_in_flight[idx] = False
            self._output[idx] = None

    def maybe_sync_shared_experts_stream(
''',
        "early-launch methods",
    )
    return _replace_once(
        text,
        "    @property\n"
        "    def output(self) -> torch.Tensor:\n"
        "        assert self._output[self._output_idx] is not None\n"
        "        output = self._output[self._output_idx]\n"
        "        self._output[self._output_idx] = None\n"
        "        return output",
        "    @property\n"
        "    def output(self) -> torch.Tensor:\n"
        "        idx = self._output_idx\n"
        "        if self._async_in_flight[idx]:\n"
        "            raise RuntimeError(\n"
        '                f"CUDA shared-expert early-launch slot {idx} was not waited"\n'
        "            )\n"
        "        output = self._output[idx]\n"
        "        if output is None:\n"
        "            raise RuntimeError(\n"
        '                f"CUDA shared-expert early-launch slot {idx} has no output"\n'
        "            )\n"
        "        if self._cuda_early_launch:\n"
        "            output.record_stream(current_stream())\n"
        "        self._output[idx] = None\n"
        "        return output",
        "output lifecycle",
    )


def patch_moe_runner_source(text: str) -> str:
    """Move only opted-in shared-expert submission before routed dispatch."""
    text = _replace_once(
        text,
        "        shared_experts_input: torch.Tensor | None,\n"
        "        input_ids: torch.Tensor | None = None,\n"
        "    ) -> tuple[torch.Tensor | None, torch.Tensor]:\n"
        '        """Run expert routing and the fused MoE kernel via the quant method.\n',
        "        shared_experts_input: torch.Tensor | None,\n"
        "        input_ids: torch.Tensor | None = None,\n"
        "        shared_experts_overlapping: bool = False,\n"
        "    ) -> tuple[torch.Tensor | None, torch.Tensor]:\n"
        '        """Run expert routing and the fused MoE kernel via the quant method.\n',
        "quant method signature",
    )
    text = _replace_once(
        text,
        '        (shared_expert_output, fused_expert_output).\n        """\n',
        "        (shared_expert_output, fused_expert_output).\n"
        "\n"
        "        ``shared_experts_overlapping`` is true only when the shared expert was\n"
        "        already enqueued on its auxiliary CUDA stream. That path waits here at\n"
        "        the same result-consumption boundary used by the legacy launch order.\n"
        '        """\n',
        "quant method contract",
    )
    text = _replace_once(
        text,
        "        self._maybe_apply_shared_experts(\n"
        "            shared_experts_input,\n"
        "            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,\n"
        "        )\n"
        "\n"
        "        return (\n",
        "        if shared_experts_overlapping:\n"
        "            assert self._shared_experts is not None\n"
        "            self._shared_experts.wait()\n"
        "        else:\n"
        "            # Preserve the existing CUDA launch and synchronization path when\n"
        "            # the early-launch experiment is disabled or inapplicable.\n"
        "            self._maybe_apply_shared_experts(\n"
        "                shared_experts_input,\n"
        "                SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,\n"
        "            )\n"
        "\n"
        "        return (\n",
        "result-boundary wait",
    )
    text = _replace_once(
        text,
        "        # Sync aux and main stream for shared expert multi-stream overlap.\n"
        "        self._maybe_sync_shared_experts_stream(shared_experts_input)\n"
        "\n"
        "        # If the Runner holds the gate, apply it after the stream sync,\n",
        "        # Opt-in CUDA experiment: enqueue shared experts before gate, routing,\n"
        "        # and routed-expert dispatch. If it does not apply, preserve the legacy\n"
        "        # stream synchronization and later host launch order exactly.\n"
        "        shared_experts_overlapping = False\n"
        "        if self._shared_experts is not None:\n"
        "            assert shared_experts_input is not None\n"
        "            shared_experts_overlapping = (\n"
        "                self._shared_experts.maybe_forward_async(shared_experts_input)\n"
        "            )\n"
        "        if not shared_experts_overlapping:\n"
        "            self._maybe_sync_shared_experts_stream(shared_experts_input)\n"
        "\n"
        "        # If the Runner holds the gate, apply it after the stream sync,\n",
        "pre-dispatch launch",
    )
    text = _replace_once(
        text,
        "                shared_experts_input=shared_experts_input,\n"
        "                input_ids=input_ids,\n"
        "            )\n",
        "                shared_experts_input=shared_experts_input,\n"
        "                input_ids=input_ids,\n"
        "                shared_experts_overlapping=shared_experts_overlapping,\n"
        "            )\n",
        "overlap state forwarding",
    )
    return _replace_once(
        text,
        """        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states,
                router_logits,
            )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
                shared_experts_overlapping=shared_experts_overlapping,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )""",
        """        try:
            # If the Runner holds the gate, apply it after the stream sync,
            # so it can run overlapped with the
            # NOTE: in future PR, MoE runner will always hold the gate.
            if self.gate is not None:
                if self._fse_fuse_gate:
                    self._maybe_fuse_gate_weights()
                    router_logits = F.linear(hidden_states, self._combined_gate_weight)
                else:
                    router_logits, _ = self.gate(hidden_states)

            with self._sequence_parallel_context():
                # TODO(bnell): parts of the dispatch/combine steps will go away once
                # #32567 lands and the remaining kernels are made MKs.  The PCP
                # code will probably remain
                hidden_states, router_logits = self._maybe_dispatch(
                    hidden_states,
                    router_logits,
                )

                shared_output, hidden_states = self._apply_quant_method(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=input_ids,
                    shared_experts_overlapping=shared_experts_overlapping,
                )

                return self._maybe_combine(
                    shared_output,
                    hidden_states,
                )
        except BaseException:
            if shared_experts_overlapping:
                assert self._shared_experts is not None
                try:
                    self._shared_experts.wait_and_discard_async_output()
                except BaseException as cleanup_error:
                    logger.exception(
                        "CUDA shared-expert cleanup failed while preserving "
                        "the routed-path error: %r",
                        cleanup_error,
                    )
            raise""",
        "routed-path error cleanup",
    )


PATCHERS = {
    ENV_PATH: patch_envs_source,
    MOE_RUNNER_PATH: patch_moe_runner_source,
    SHARED_EXPERTS_PATH: patch_shared_experts_source,
}


def patch_runtime(
    root: Path,
    *,
    dry_run: bool,
    expected_input_sha256: Mapping[Path, str],
) -> dict[str, object]:
    """Patch one hash-bound runtime tree and return its input/output manifest."""
    files: dict[str, dict[str, str]] = {}
    outputs: dict[Path, str] = {}
    for relative_path, patcher in PATCHERS.items():
        path = root / relative_path
        original = path.read_bytes()
        input_sha256 = _sha256_bytes(original)
        expected_sha256 = expected_input_sha256[relative_path]
        if input_sha256 != expected_sha256:
            raise RuntimeError(
                f"Qwen shared-expert runtime source mismatch for {relative_path}: "
                f"expected {expected_sha256}, got {input_sha256}"
            )
        patched = patcher(original.decode())
        output_sha256 = _sha256_bytes(patched.encode())
        outputs[path] = patched
        files[str(relative_path)] = {
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
        }

    if not dry_run:
        for path, patched in outputs.items():
            path.write_text(patched)

    return {
        "schema_version": 1,
        "selector": "VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH",
        "default_enabled": False,
        "dry_run": dry_run,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root containing the installed vllm package",
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        required=True,
        help="Reviewed manifest containing exact installed-source hashes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and derive output hashes without writing files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path for the atomic JSON result manifest",
    )
    args = parser.parse_args()

    manifest = patch_runtime(
        args.root,
        dry_run=args.dry_run,
        expected_input_sha256=load_runtime_identities(args.identity_manifest),
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        tmp_path = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        tmp_path.write_text(rendered)
        tmp_path.replace(args.manifest)
    print(rendered, end="")


if __name__ == "__main__":
    main()
