#!/usr/bin/env python3
"""Verify GGUF-TP planned targets against a full DeepSeek V4 meta model."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import vllm.models.deepseek_v4.nvidia.model as model_module
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.config.device import DeviceConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.layers.quantization.gguf_dsv4 import GGUFDSV4QuantConfig
from vllm.model_executor.model_loader.gguf_dsv4_index import (
    GGUF_TYPES,
    GGUFTensorEntry,
)
from vllm.model_executor.model_loader.gguf_dsv4_plan import (
    build_gguf_dsv4_load_plan,
)
from vllm.models.deepseek_v4.ampere.ampere_sparse import (
    DeepseekV4AmpereMLAAttention,
)
from vllm.platforms import current_platform


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-port", type=int, default=29677)
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    model_config = ModelConfig(
        model=str(args.model_config_dir),
        tokenizer=str(args.model_config_dir),
        tokenizer_mode="deepseek_v4",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=140000,
        trust_remote_code=True,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=world_size, pipeline_parallel_size=1
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=SchedulerConfig.default_factory(),
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla"),
        quant_config=GGUFDSV4QuantConfig(),
    )
    # Probe-only: skip eager CUDA scratch and instantiate explicit device tensors
    # on meta without changing production constructors.
    object.__setattr__(parallel_config, "enable_dbo", True)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=(
                "env://" if world_size > 1 else f"tcp://127.0.0.1:{args.init_port}"
            ),
            backend="gloo",
        )
        ensure_model_parallel_initialized(world_size, 1)
        torch.cuda.Stream = lambda *unused_args, **unused_kwargs: None
        model_module._select_dsv4_attn_cls = lambda unused: DeepseekV4AmpereMLAAttention
        current_platform.device_type = "meta"
        with torch.device("meta"):
            model = model_module.DeepseekV4ForCausalLM(vllm_config=vllm_config)

    inventory = json.loads(args.inventory.read_text())
    type_ids = {spec.name: type_id for type_id, spec in GGUF_TYPES.items()}
    entries = tuple(
        GGUFTensorEntry(
            tensor["name"],
            type_ids[tensor["type"]],
            tuple(tensor["dims"]),
            tensor["offset"],
        )
        for tensor in inventory["tensors"]
    )
    plan = build_gguf_dsv4_load_plan(entries, tp_rank=rank, tp_size=world_size)
    parameters = dict(model.named_parameters())
    planned_targets = {item.target_name for item in plan}
    parameter_targets = set(parameters)
    if planned_targets != parameter_targets:
        raise AssertionError(
            {
                "missing_parameters": sorted(planned_targets - parameter_targets),
                "unexpected_parameters": sorted(parameter_targets - planned_targets),
            }
        )

    planned_elements = defaultdict(int)
    source_types = defaultdict(set)
    for item in plan:
        spec = GGUF_TYPES[type_ids[item.source_type]]
        if spec.block_elements == 1:
            planned_elements[item.target_name] += item.target_nbytes // spec.block_bytes
        else:
            planned_elements[item.target_name] += item.target_nbytes
        source_types[item.target_name].add(item.source_type)
    mismatches = []
    for name, parameter in parameters.items():
        if parameter.numel() != planned_elements[name]:
            mismatches.append(
                {
                    "target": name,
                    "parameter_numel": parameter.numel(),
                    "planned_elements": planned_elements[name],
                }
            )
    if mismatches:
        raise AssertionError({"element_count_mismatches": mismatches})

    report = {
        "schema_version": 1,
        "rank": rank,
        "tp_size": world_size,
        "model_config_sha256": _sha256(args.model_config_dir / "config.json"),
        "inventory_sha256": _sha256(args.inventory),
        "inventory_tensors": len(entries),
        "planned_targets": len(planned_targets),
        "model_parameters": len(parameters),
        "name_sets_equal": True,
        "element_counts_equal": True,
        "parameter_dtypes": dict(
            sorted(
                Counter(
                    str(parameter.dtype) for parameter in parameters.values()
                ).items()
            )
        ),
        "source_type_sets": dict(
            sorted(
                Counter(
                    "+".join(sorted(types)) for types in source_types.values()
                ).items()
            )
        ),
    }
    reports = [None for _ in range(world_size)]
    dist.all_gather_object(reports, report)
    if rank == 0:
        aggregate = {
            "schema_version": 1,
            "tp_size": world_size,
            "ranks": reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
        print(json.dumps(aggregate, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
