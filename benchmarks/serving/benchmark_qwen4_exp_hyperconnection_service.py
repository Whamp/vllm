# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen4Exp hyperconnection service throughput at c=1 and c=2."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def request_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def streamed_decode(
    endpoint: str,
    model: str,
    run_id: str,
    output_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Decode benchmark nonce {run_id}. Write a detailed uninterrupted "
                    "technical essay about designing a resilient distributed storage "
                    "engine. Continue until the token limit."
                ),
            }
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_delta = None
    usage = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            content = line[6:]
            if content == "[DONE]":
                break
            event = json.loads(content)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if first_delta is None and any(
                    delta.get(key)
                    for key in ("content", "reasoning", "reasoning_content")
                ):
                    first_delta = time.perf_counter()
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if first_delta is None or usage is None:
        raise RuntimeError("Qwen hyperconnection decode omitted timing or usage")
    completion_tokens = usage["completion_tokens"]
    return {
        "start": start,
        "first_delta": first_delta,
        "end": end,
        "wall_seconds": end - start,
        "ttft_seconds": first_delta - start,
        "decode_seconds": end - first_delta,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "decode_tokens_per_second": completion_tokens / (end - first_delta),
        "finish_reason": finish_reason,
    }


def cache_busted_prefill_prompt(run_id: str, line_count: int = 850) -> str:
    lines = [f"PREFILL-NONCE-{run_id} starts this request so no prefix can be reused."]
    lines.extend(
        (
            f"Record {index}: distributed systems need explicit invariants, "
            "measured limits, and deterministic recovery."
        )
        for index in range(line_count)
    )
    lines.append("Reply with one word naming the main topic.")
    return "\n".join(lines)


def prefill_probe(endpoint: str, model: str, run_id: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": cache_busted_prefill_prompt(run_id)}],
        "temperature": 0,
        "max_tokens": 1,
    }
    start = time.perf_counter()
    response = request_json(endpoint, "/v1/chat/completions", payload)
    end = time.perf_counter()
    usage = response["usage"]
    return {
        "start": start,
        "end": end,
        "wall_seconds": end - start,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens_per_second": usage["prompt_tokens"] / (end - start),
        "finish_reason": response["choices"][0]["finish_reason"],
    }


def run_concurrent_round(
    operation,
    endpoint: str,
    model: str,
    label: str,
    concurrency: int,
    *operation_args: Any,
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                operation,
                endpoint,
                model,
                f"{label}-stream-{stream}",
                *operation_args,
            )
            for stream in range(concurrency)
        ]
        runs = [future.result() for future in futures]
    start = min(run["start"] for run in runs)
    end = max(run["end"] for run in runs)
    result = {
        "runs": runs,
        "wall_seconds": end - start,
    }
    if "completion_tokens" in runs[0] and "first_delta" in runs[0]:
        first_delta = min(run["first_delta"] for run in runs)
        completion_tokens = sum(run["completion_tokens"] for run in runs)
        result["aggregate_decode_tokens_per_second"] = completion_tokens / (
            end - first_delta
        )
        result["aggregate_wall_tokens_per_second"] = completion_tokens / (end - start)
        result["mean_per_stream_decode_tokens_per_second"] = statistics.mean(
            run["decode_tokens_per_second"] for run in runs
        )
    else:
        prompt_tokens = sum(run["prompt_tokens"] for run in runs)
        result["aggregate_prompt_tokens_per_second"] = prompt_tokens / (end - start)
        result["mean_per_stream_prompt_tokens_per_second"] = statistics.mean(
            run["prompt_tokens_per_second"] for run in runs
        )
    return result


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "population_cv_percent": statistics.pstdev(values)
        / statistics.mean(values)
        * 100,
    }


def benchmark_concurrency(
    endpoint: str,
    model: str,
    label: str,
    concurrency: int,
    decode_tokens: int,
) -> dict[str, Any]:
    decode_warmups = [
        run_concurrent_round(
            streamed_decode,
            endpoint,
            model,
            f"{label}-c{concurrency}-decode-warmup-{index}",
            concurrency,
            decode_tokens,
        )
        for index in range(3)
    ]
    decode_measured = [
        run_concurrent_round(
            streamed_decode,
            endpoint,
            model,
            f"{label}-c{concurrency}-decode-measured-{index}",
            concurrency,
            decode_tokens,
        )
        for index in range(5)
    ]
    prefill_warmups = [
        run_concurrent_round(
            prefill_probe,
            endpoint,
            model,
            f"{label}-c{concurrency}-prefill-warmup-{index}",
            concurrency,
        )
        for index in range(1)
    ]
    prefill_measured = [
        run_concurrent_round(
            prefill_probe,
            endpoint,
            model,
            f"{label}-c{concurrency}-prefill-measured-{index}",
            concurrency,
        )
        for index in range(3)
    ]
    return {
        "decode": {
            "warmups": decode_warmups,
            "measured": decode_measured,
            "aggregate_decode_tokens_per_second": summarize(
                [run["aggregate_decode_tokens_per_second"] for run in decode_measured]
            ),
            "aggregate_wall_tokens_per_second": summarize(
                [run["aggregate_wall_tokens_per_second"] for run in decode_measured]
            ),
        },
        "prefill": {
            "warmups": prefill_warmups,
            "measured": prefill_measured,
            "aggregate_prompt_tokens_per_second": summarize(
                [run["aggregate_prompt_tokens_per_second"] for run in prefill_measured]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decode-tokens", type=int, default=256)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    models = request_json(endpoint, "/v1/models")
    if [record["id"] for record in models["data"]] != [args.model]:
        raise RuntimeError(f"Qwen hyperconnection model identity mismatch: {models}")
    result = {
        "schema_version": 1,
        "label": args.label,
        "endpoint": endpoint,
        "model": args.model,
        "model_record": models["data"][0],
        "decode_tokens": args.decode_tokens,
        "concurrency": {
            str(concurrency): benchmark_concurrency(
                endpoint,
                args.model,
                args.label,
                concurrency,
                args.decode_tokens,
            )
            for concurrency in (1, 2)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
