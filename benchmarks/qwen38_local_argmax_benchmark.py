# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched TP4 draft-readout benchmark; run with torchrun on an idle GPU set."""

import gc
import json
import os
import statistics

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


@torch.inference_mode()
def benchmark_readout(rank):
    torch.manual_seed(915 + rank)
    head = ParallelLMHead(248320, 2560, bias=False, params_dtype=torch.bfloat16)
    head = head.cuda()
    head.weight.normal_(std=0.02)
    processor = LogitsProcessor(248320)
    processor.use_all_gather = True
    torch.manual_seed(1942)
    hidden = torch.randn(24, 2560, dtype=torch.bfloat16).cuda()
    modes = ("full", "pairs-gather", "pairs-allreduce")
    batches = (1, 2, 4, 12, 24)
    graphs = {}
    for batch in batches:
        inputs = hidden[:batch]
        expected = processor(head, inputs).argmax(dim=-1)
        for mode in modes:
            os.environ["VLLM_LOCAL_ARGMAX_ALLREDUCE"] = (
                "1" if mode == "pairs-allreduce" else "0"
            )

            def run_readout(mode=mode, inputs=inputs):
                if mode == "full":
                    return processor(head, inputs).argmax(dim=-1)
                return processor.get_top_tokens(head, inputs)

            for _ in range(3):
                run_readout()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(torch.device("cuda", rank)) as context,
                torch.cuda.graph(graph, stream=context.stream),
            ):
                output = run_readout()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (rank, batch, mode)
            graphs[mode, batch] = (graph, output)

    # Reverse the second pass to bound drift; timings exclude setup and profiling.
    for pass_id, order in enumerate((modes, modes[::-1])):
        for mode in order:
            for batch in batches if pass_id == 0 else batches[::-1]:
                graph, output = graphs[mode, batch]
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                elapsed = []
                for _ in range(7):
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(100):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end) / 100)
                all_ranks = [None] * dist.get_world_size()
                dist.all_gather_object(all_ranks, elapsed)
                if rank == 0:
                    critical = [
                        max(values[index] for values in all_ranks)
                        for index in range(len(elapsed))
                    ]
                    record = {
                        "pass": pass_id,
                        "mode": mode,
                        "batch": batch,
                        "rank_max_median_ms": statistics.median(critical),
                        "rank_gpu_ms": all_ranks,
                    }
                    print("LOCAL_ARGMAX_BENCHMARK=" + json.dumps(record), flush=True)

    # Cross-rank ties: each graph must still select the first global vocabulary id.
    head.weight.zero_()
    for (mode, batch), (graph, output) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            output, torch.zeros(batch, dtype=torch.int64, device="cuda")
        ), (rank, mode, batch)
    if rank == 0:
        print("QWEN38_TP4_LOCAL_ARGMAX_EQUALITY_PASSED=1", flush=True)
    graphs.clear()
    del graph, output
    gc.collect()
    torch.cuda.synchronize()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise ValueError("Qwen3.8 local-argmax benchmark requires exactly TP4")
    torch.cuda.set_device(rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=4, disable_custom_all_reduce=True
        )
    )
    with set_current_vllm_config(config):
        set_custom_all_reduce(False)
        init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=world)
        try:
            benchmark_readout(rank)
        finally:
            destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
