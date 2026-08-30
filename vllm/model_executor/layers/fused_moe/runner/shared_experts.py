# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = init_logger(__name__)


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._async_in_flight = [False, False]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts
        self._cuda_early_launch = (
            envs.VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH and current_platform.is_cuda()
        )

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        in_ple_offload_process = False
        if envs.VLLM_PLE_CPU_OFFLOAD:
            from vllm.model_executor.layers.ple_offload_layer import (
                is_offload_process,
            )

            in_ple_offload_process = is_offload_process()

        if in_ple_offload_process:
            # PLE discovery builds the non-PLE model only for module traversal;
            # no MoE forward ever runs in this process.  Creating an auxiliary
            # CUDA stream here defeats meta construction and initializes a CUDA
            # context before the worker imports its actual IPC destinations.
            self._stream = None
        elif envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

        self._input_ready_event: list[torch.cuda.Event] = []
        self._output_ready_event: list[torch.cuda.Event] = []
        if self._stream is not None and self._cuda_early_launch:
            self._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
            self._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
            logger.info_once("Enabled early launch for CUDA MoE shared experts")

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-safe backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain

        # Both these comm backends have been shown to be safe for shared expert overlap.
        _EPLB_OVERLAP_SAFE_BACKENDS = (
            "allgather_reducescatter",
            "flashinfer_nvlink_one_sided",
        )

        parallel_config = self._moe_config.moe_parallel_config
        if getattr(self._layer, "shard_sequence_parallel", False):
            # TODO: we may enable this to optimize further
            return True
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend not in _EPLB_OVERLAP_SAFE_BACKENDS
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_forward_async(self, shared_experts_input: torch.Tensor) -> bool:
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
        self,
        shared_experts_input: torch.Tensor,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    @property
    def output(self) -> torch.Tensor:
        idx = self._output_idx
        if self._async_in_flight[idx]:
            raise RuntimeError(
                f"CUDA shared-expert early-launch slot {idx} was not waited"
            )
        output = self._output[idx]
        if output is None:
            raise RuntimeError(
                f"CUDA shared-expert early-launch slot {idx} has no output"
            )
        if self._cuda_early_launch:
            output.record_stream(current_stream())
        self._output[idx] = None
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None
