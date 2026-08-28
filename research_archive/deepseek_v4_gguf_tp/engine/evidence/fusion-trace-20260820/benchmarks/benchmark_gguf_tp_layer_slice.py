# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a graph-captured TP4 DeepSeek V4 GGUF decoder-layer slice."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import vllm._C_stable_libtorch  # noqa: F401
import vllm._moe_C_stable_libtorch  # noqa: F401

from benchmarks.kernels.benchmark_gguf_iq2_xxs import make_seeded_packed_iq2_xxs
from benchmarks.kernels.benchmark_gguf_q2_k import make_q2_k_weights
from benchmarks.kernels.benchmark_gguf_q8_0_dense import (
    DenseQ8Shape,
    make_q8_0_weights,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.gguf_dsv4.q8_0_marlin import (
    GGUFQ8MarlinWeights,
    apply_gguf_q8_0_marlin,
    prepare_gguf_q8_0_marlin,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_dsv4_wo_a_marlin_diagonal,
)

EXPERT_COUNT = 256
TOPK = 6
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE_PER_RANK = 512
SWIGLU_LIMIT = 10.0


@dataclass
class LayerSliceWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    dense: dict[str, GGUFQ8MarlinWeights]


class PreparedWoA(torch.nn.Module):
    """Expose prepared Q8_0 wo_a through the existing diagonal helper seam."""

    def __init__(self, prepared: GGUFQ8MarlinWeights):
        super().__init__()
        self.prepared = prepared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return apply_gguf_q8_0_marlin(inputs, self.prepared)


def initialize_tp4() -> tuple[int, torch.device]:
    """Initialize one torchrun rank with the production hierarchical backend."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(f"GGUF layer slice requires TP4, got world_size={world_size}")
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="nccl",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )
    return rank, device


def set_iq2_fixture_scales(packed: np.ndarray, input_columns: int) -> None:
    """Bound synthetic IQ2 values without changing byte layout or codebooks."""
    row_bytes = input_columns // 256 * 66
    scale_bytes = np.frombuffer(np.float16(1.0e-4).tobytes(), dtype=np.uint8)
    rows = packed.reshape(-1, row_bytes)
    for block in range(input_columns // 256):
        rows[:, block * 66 : block * 66 + 2] = scale_bytes


def set_q2_fixture_scales(packed: np.ndarray, input_columns: int) -> None:
    """Bound synthetic Q2 scale/min halves for chained layer execution."""
    row_bytes = input_columns // 256 * 84
    scale_bytes = np.frombuffer(np.float16(1.0e-4).tobytes(), dtype=np.uint8)
    min_bytes = np.frombuffer(np.float16(5.0e-5).tobytes(), dtype=np.uint8)
    rows = packed.reshape(-1, row_bytes)
    for block in range(input_columns // 256):
        rows[:, block * 84 + 80 : block * 84 + 82] = scale_bytes
        rows[:, block * 84 + 82 : block * 84 + 84] = min_bytes


def set_q8_fixture_scales(raw: torch.Tensor, shape: DenseQ8Shape) -> None:
    """Bound synthetic Q8 scales while retaining codes and block geometry."""
    scale_bytes = torch.from_numpy(
        np.frombuffer(np.float16(1.0e-4).tobytes(), dtype=np.uint8).copy()
    )
    blocks = raw.view(shape.output_rows, shape.input_columns // 32, 34)
    blocks[:, :, :2] = scale_bytes


def prepare_layer_slice_weights(rank: int, device: torch.device) -> LayerSliceWeights:
    """Prepare exact rank-local GGUF projection shapes on one GPU."""
    gate_rows = INTERMEDIATE_SIZE_PER_RANK
    gate_columns = HIDDEN_SIZE
    gate_row_bytes = gate_columns // 256 * 66
    gate = make_seeded_packed_iq2_xxs(
        EXPERT_COUNT * gate_rows, gate_columns, 20260900 + rank
    ).reshape(EXPERT_COUNT, gate_rows, gate_row_bytes)
    up = make_seeded_packed_iq2_xxs(
        EXPERT_COUNT * gate_rows, gate_columns, 20261000 + rank
    ).reshape(EXPERT_COUNT, gate_rows, gate_row_bytes)
    down = make_q2_k_weights(
        EXPERT_COUNT,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE_PER_RANK,
    )
    set_iq2_fixture_scales(gate, gate_columns)
    set_iq2_fixture_scales(up, gate_columns)
    set_q2_fixture_scales(down, INTERMEDIATE_SIZE_PER_RANK)

    dense_shapes = (
        DenseQ8Shape("fused_wqa_wkv", 1536, 4096, 1),
        DenseQ8Shape("wq_b", 8192, 1024, 1),
        DenseQ8Shape("wo_a", 2048, 4096, 1),
        DenseQ8Shape("wo_b", 4096, 2048, 1),
        DenseQ8Shape("shared_gate_up", 1024, 4096, 1),
        DenseQ8Shape("shared_down", 4096, 512, 1),
    )
    dense = {}
    for shape in dense_shapes:
        raw = make_q8_0_weights(shape)
        set_q8_fixture_scales(raw, shape)
        dense[shape.name] = prepare_gguf_q8_0_marlin(
            raw.to(device),
            input_columns=shape.input_columns,
            scale_dtype=torch.bfloat16,
        )

    return LayerSliceWeights(
        gate=torch.from_numpy(gate).to(device),
        up=torch.from_numpy(up).to(device),
        down=torch.from_numpy(down).to(device),
        dense=dense,
    )


class GGUFDecoderLayerSlice:
    """Own caller-allocated buffers for one decode or prefill graph shape."""

    def __init__(
        self,
        token_count: int,
        grouped_experts: bool,
        weights: LayerSliceWeights,
        device: torch.device,
    ):
        self.token_count = token_count
        self.grouped_experts = grouped_experts
        self.weights = weights
        self.hidden = torch.randn(
            token_count, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
        )
        self.topk_ids = (
            torch.arange(token_count * TOPK, device=device, dtype=torch.int32)
            .remainder(EXPERT_COUNT)
            .reshape(token_count, TOPK)
        )
        self.topk_weights = torch.rand(
            token_count, TOPK, device=device, dtype=torch.float32
        )
        self.topk_weights /= self.topk_weights.sum(dim=1, keepdim=True)
        self.gate_scales = torch.empty(
            token_count, HIDDEN_SIZE // 32, device=device, dtype=torch.float16
        )
        self.gate_codes = torch.empty_like(self.hidden, dtype=torch.int8)
        self.gate_output = torch.empty(
            token_count,
            TOPK,
            INTERMEDIATE_SIZE_PER_RANK,
            device=device,
            dtype=torch.float32,
        )
        self.up_output = torch.empty_like(self.gate_output)
        self.down_scales = torch.empty(
            token_count * TOPK,
            INTERMEDIATE_SIZE_PER_RANK // 32,
            device=device,
            dtype=torch.float16,
        )
        self.down_codes = torch.empty(
            token_count * TOPK,
            INTERMEDIATE_SIZE_PER_RANK,
            device=device,
            dtype=torch.int8,
        )
        self.down_output = torch.empty(
            token_count, TOPK, HIDDEN_SIZE, device=device, dtype=torch.float32
        )
        self.wo_a = PreparedWoA(weights.dense["wo_a"])
        if grouped_experts:
            (
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
            ) = moe_align_block_size(
                topk_ids=self.topk_ids,
                block_size=8,
                num_experts=EXPERT_COUNT,
            )
        else:
            self.sorted_ids = None
            self.expert_ids = None
            self.num_tokens_padded = None

    def run(self) -> torch.Tensor:
        """Execute Q8 attention, routed/shared experts, and two TP reductions."""
        dense = self.weights.dense
        fused_wqa_wkv = apply_gguf_q8_0_marlin(self.hidden, dense["fused_wqa_wkv"])
        q_heads = apply_gguf_q8_0_marlin(fused_wqa_wkv[:, :1024], dense["wq_b"])
        grouped_q = q_heads.view(self.token_count, 2, HIDDEN_SIZE)
        wo_a = _apply_dsv4_wo_a_marlin_diagonal(
            grouped_q,
            self.wo_a,
            n_local_groups=2,
            o_lora_rank=1024,
        )
        attention_partial = apply_gguf_q8_0_marlin(wo_a.flatten(1), dense["wo_b"])
        hidden = tensor_model_parallel_all_reduce(attention_partial)

        torch.ops._C.gguf_quantize_bf16_to_q8_1(
            hidden, self.gate_scales, self.gate_codes
        )
        if self.grouped_experts:
            torch.ops._C.gguf_iq2_xxs_q8_1_grouped_gate_up(
                self.gate_scales,
                self.gate_codes,
                self.weights.gate,
                self.weights.up,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.gate_output,
                self.up_output,
                TOPK,
            )
        else:
            torch.ops._C.gguf_iq2_xxs_q8_1_indexed_gate_up(
                self.gate_scales,
                self.gate_codes,
                self.weights.gate,
                self.weights.up,
                self.topk_ids,
                self.gate_output,
                self.up_output,
            )
        torch.ops._C.gguf_swiglu_weighted_q8_1(
            self.gate_output,
            self.up_output,
            self.topk_weights,
            self.down_scales,
            self.down_codes,
            SWIGLU_LIMIT,
        )
        if self.grouped_experts:
            torch.ops._C.gguf_q2_k_q8_1_grouped_down(
                self.down_scales,
                self.down_codes,
                self.weights.down,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.down_output,
            )
        else:
            torch.ops._C.gguf_q2_k_q8_1_indexed_down(
                self.down_scales,
                self.down_codes,
                self.weights.down,
                self.topk_ids,
                self.down_output,
            )
        routed_partial = self.down_output.sum(dim=1)

        shared_gate_up = apply_gguf_q8_0_marlin(hidden, dense["shared_gate_up"]).float()
        shared_gate = torch.clamp(shared_gate_up[:, :512], max=SWIGLU_LIMIT)
        shared_up = torch.clamp(
            shared_gate_up[:, 512:], min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT
        )
        shared_mid = (F.silu(shared_gate) * shared_up).to(torch.bfloat16)
        shared_partial = apply_gguf_q8_0_marlin(shared_mid, dense["shared_down"])
        ffn_partial = (routed_partial + shared_partial.float()).to(torch.bfloat16)
        return tensor_model_parallel_all_reduce(ffn_partial)


def capture_and_time_slice(
    layer_slice: GGUFDecoderLayerSlice,
    iterations: int,
    warmup: int,
    profile: bool = False,
) -> tuple[float, torch.Tensor]:
    """Capture one static layer shape and return rank-local event timing."""
    for _ in range(3):
        eager_output = layer_slice.run()
    torch.accelerator.synchronize()
    if not torch.isfinite(eager_output).all():
        raise RuntimeError("GGUF TP4 layer slice eager output is non-finite")
    graph_output = torch.empty_like(eager_output)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output.copy_(layer_slice.run())
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    dist.barrier()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("gguf_tp_decode_layer_slice_replays")
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    dist.barrier()
    if profile:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()
    elapsed_ms = start.elapsed_time(end) / iterations
    if not torch.isfinite(graph_output).all():
        raise RuntimeError("GGUF TP4 layer slice graph output is non-finite")
    return elapsed_ms, graph_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-iterations", type=int, default=2000)
    parser.add_argument("--prefill-iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--profile-decode",
        action="store_true",
        help="Bracket only indexed-decode graph replays with cudaProfilerStart/Stop",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rank, device = initialize_tp4()
    weights = prepare_layer_slice_weights(rank, device)
    tp_group = get_tp_group()
    hier_comm = tp_group.device_communicator.hier_ar_comm
    if hier_comm is None:
        raise RuntimeError("GGUF layer slice requires hierarchical all-reduce")

    results = []
    for token_count, grouped, iterations in (
        (1, False, args.decode_iterations),
        (256, True, args.prefill_iterations),
    ):
        layer_slice = GGUFDecoderLayerSlice(token_count, grouped, weights, device)
        elapsed_ms, output = capture_and_time_slice(
            layer_slice,
            iterations,
            args.warmup,
            profile=args.profile_decode and token_count == 1,
        )
        if not torch.isfinite(output).all():
            raise RuntimeError("GGUF TP4 layer slice produced non-finite output")
        probe = torch.empty(
            token_count, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
        )
        results.append(
            {
                "tokens": token_count,
                "expert_path": "grouped" if grouped else "indexed",
                "graph_ms": elapsed_ms,
                "iterations": iterations,
                "hierarchical_should_use": bool(hier_comm.should_use(probe)),
            }
        )

    all_rank_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_rank_results, results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "rank": rank,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "results": results,
    }
    (output_dir / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if rank == 0:
        (output_dir / "all-ranks.json").write_text(
            json.dumps(all_rank_results, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(all_rank_results, indent=2))
    dist.barrier()


if __name__ == "__main__":
    main()
