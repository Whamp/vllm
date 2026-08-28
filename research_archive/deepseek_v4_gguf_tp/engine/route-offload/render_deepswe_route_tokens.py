#!/usr/bin/env python3
"""Render compaction-aware DeepSWE requests with the production DSV4 tokenizer.

Run this script inside the pinned GGUF-TP vLLM image with the model view mounted.
It needs no GPU and writes exact token IDs plus a checksum-bound manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

from vllm.entrypoints.chat_utils import (  # ty: ignore[unresolved-import]
    parse_chat_messages,
)
from vllm.tokenizers.deepseek_v4 import (  # ty: ignore[unresolved-import]
    DeepseekV4Tokenizer,
)


class RenderedRouteRequest(TypedDict):
    """One request and exact rendered-token identity in the output manifest."""

    task: str
    token_count: int
    request_sha256: str
    token_ids_sha256: str
    token_ids_file: str


def render_deepswe_route_requests(
    *, requests_dir: Path, model_dir: Path, output_dir: Path
) -> dict[str, object]:
    """Render every `*.request.json` file and return its stable manifest."""
    request_paths = sorted(requests_dir.glob("*.request.json"))
    if not request_paths:
        raise ValueError(f"No route replay requests found under {requests_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = DeepseekV4Tokenizer.from_pretrained(model_dir, trust_remote_code=True)
    text_model_config = SimpleNamespace(
        multimodal_config=None, enable_prompt_embeds=False
    )
    requests: list[RenderedRouteRequest] = []
    for path in request_paths:
        request = json.loads(path.read_text(encoding="utf-8"))
        conversation, multimodal_data, multimodal_uuids = parse_chat_messages(
            request["messages"], text_model_config, content_format="string"
        )
        if multimodal_data is not None or multimodal_uuids is not None:
            raise ValueError(f"Route replay request {path} unexpectedly contains media")
        token_ids = tokenizer.apply_chat_template(
            conversation=conversation,
            messages=request["messages"],
            tools=request["tools"],
            add_generation_prompt=True,
            continue_final_message=False,
            reasoning_effort=request["reasoning_effort"],
            enable_thinking=True,
            tokenize=True,
            return_dict=False,
        )
        token_bytes = (
            json.dumps(token_ids, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        task = path.name.removesuffix(".request.json")
        token_file = output_dir / f"{task}.token-ids.json"
        token_file.write_bytes(token_bytes)
        requests.append(
            {
                "task": task,
                "token_count": len(token_ids),
                "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
                "token_ids_file": token_file.name,
            }
        )
    manifest = {"schema_version": 1, "requests": requests}
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (output_dir / "render-manifest-v2.json").write_bytes(manifest_bytes)
    return {
        "requests": len(requests),
        "total_tokens": sum(request["token_count"] for request in requests),
        "maximum_tokens": max(request["token_count"] for request in requests),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render_deepswe_route_requests(
        requests_dir=args.requests_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
