#!/usr/bin/env python3
"""Build a compaction-aware DeepSWE chat request for GGUF-TP route capture.

The builder reads Pi's version-3 session tree, applies Pi 0.84's active-context
compaction rule, converts Pi content blocks to the OpenAI chat-completions shape,
and emits a max-one-token request that exercises the recorded prompt through the
production DeepSeek V4 renderer and model. It first proves the conversion against
the captured real second provider request from the same run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following "
    "summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

JsonObject = dict[str, Any]


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON document from path."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_pi_session_entries(path: Path) -> list[JsonObject]:
    """Load nonempty JSONL entries from a Pi version-3 session."""
    entries: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise TypeError(f"Pi session entry {line_number} is not an object")
            entries.append(entry)
    if not entries or entries[0].get("type") != "session":
        raise ValueError("Pi session is missing its version header")
    if entries[0].get("version") != 3:
        raise ValueError(
            f"Expected Pi session version 3, got {entries[0].get('version')!r}"
        )
    return entries


def build_pi_session_path(entries: list[JsonObject], leaf_id: str) -> list[JsonObject]:
    """Walk Pi parent IDs from leaf to root and return chronological entries."""
    by_id = {entry["id"]: entry for entry in entries if "id" in entry}
    path: list[JsonObject] = []
    current_id: str | None = leaf_id
    seen: set[str] = set()
    while current_id is not None:
        if current_id in seen:
            raise ValueError(f"Pi session parent cycle at {current_id}")
        seen.add(current_id)
        try:
            entry = by_id[current_id]
        except KeyError as error:
            raise ValueError(f"Pi session parent {current_id} is missing") from error
        path.append(entry)
        current_id = entry.get("parentId")
    path.reverse()
    return path


def build_compaction_aware_entries(entries: list[JsonObject]) -> list[JsonObject]:
    """Reproduce Pi's latest-compaction active-context entry ordering."""
    leaf = next((entry for entry in reversed(entries) if "id" in entry), None)
    if leaf is None:
        raise ValueError("Pi session has no leaf entry")
    path = build_pi_session_path(entries, leaf["id"])
    latest_compaction = next(
        (entry for entry in reversed(path) if entry.get("type") == "compaction"), None
    )
    if latest_compaction is None:
        return path

    compaction_index = path.index(latest_compaction)
    first_kept_id = latest_compaction.get("firstKeptEntryId")
    if not first_kept_id:
        raise ValueError("Pi compaction is missing firstKeptEntryId")
    kept: list[JsonObject] = []
    found_first_kept = False
    for entry in path[:compaction_index]:
        if entry.get("id") == first_kept_id:
            found_first_kept = True
        if found_first_kept:
            kept.append(entry)
    if not found_first_kept:
        raise ValueError(f"Pi compaction first-kept entry {first_kept_id} is absent")
    return [latest_compaction, *kept, *path[compaction_index + 1 :]]


def join_text_blocks(content: Any) -> str:
    """Join Pi text blocks in order, rejecting unsupported replay content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(
            f"Expected Pi content list or string, got {type(content).__name__}"
        )
    text: list[str] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            text.append(block.get("text", ""))
        elif block_type in {"thinking", "toolCall"}:
            continue
        else:
            raise ValueError(f"Unsupported Pi content block for replay: {block_type!r}")
    return "".join(text)


def convert_pi_message_to_openai(message: JsonObject) -> JsonObject:
    """Convert one Pi user, assistant, or tool-result message to OpenAI chat shape."""
    role = message.get("role")
    if role == "user":
        content = message.get("content", [])
        if not isinstance(content, list):
            raise ValueError("Pi user message content must be a list")
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": block.get("text", "")}
                for block in content
                if block.get("type") == "text"
            ],
        }
    if role == "toolResult":
        tool_call_id = message.get("toolCallId")
        if not tool_call_id:
            raise ValueError("Pi tool result is missing toolCallId")
        return {
            "role": "tool",
            "content": join_text_blocks(message.get("content", [])),
            "tool_call_id": tool_call_id,
        }
    if role != "assistant":
        raise ValueError(f"Unsupported Pi message role for replay: {role!r}")

    content = message.get("content", [])
    if not isinstance(content, list):
        raise TypeError("Pi assistant message content must be a list")
    text_content = "".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )
    result: JsonObject = {
        "role": "assistant",
        "content": text_content if text_content else None,
    }
    reasoning = "".join(
        block.get("thinking", "")
        for block in content
        if block.get("type") == "thinking"
    )
    if reasoning:
        result["reasoning_content"] = reasoning
    tool_calls = []
    for block in content:
        if block.get("type") != "toolCall":
            continue
        arguments = json.dumps(
            block.get("arguments", {}), ensure_ascii=False, separators=(",", ":")
        )
        tool_calls.append(
            {
                "id": block["id"],
                "type": "function",
                "function": {"name": block["name"], "arguments": arguments},
            }
        )
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def convert_pi_entry_to_openai(entry: JsonObject) -> JsonObject | None:
    """Convert a context-producing Pi session entry to OpenAI chat shape."""
    entry_type = entry.get("type")
    if entry_type == "message":
        return convert_pi_message_to_openai(entry["message"])
    if entry_type == "compaction":
        summary = entry.get("summary")
        if not isinstance(summary, str):
            raise ValueError("Pi compaction is missing its summary text")
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": COMPACTION_SUMMARY_PREFIX
                    + summary
                    + COMPACTION_SUMMARY_SUFFIX,
                }
            ],
        }
    return None


def validate_second_provider_request(
    entries: list[JsonObject], provider_request: JsonObject
) -> None:
    """Prove message conversion against the captured real second wire request."""
    converted_messages = [
        convert_pi_entry_to_openai(entry)
        for entry in entries
        if entry.get("type") == "message"
    ]
    expected_messages = provider_request.get("messages")
    if not isinstance(expected_messages, list) or len(expected_messages) < 2:
        raise ValueError("Captured second provider request has no message sequence")
    expected_session_messages = expected_messages[1:]
    actual_session_messages = [
        message
        for message in converted_messages[: len(expected_session_messages)]
        if message
    ]
    if actual_session_messages != expected_session_messages:
        for index, (actual, expected) in enumerate(
            zip(actual_session_messages, expected_session_messages, strict=False)
        ):
            if actual != expected:
                raise ValueError(
                    "Pi-to-OpenAI conversion differs from captured provider request "
                    f"at session message {index}: actual={actual!r}, expected={expected!r}"
                )
        raise ValueError(
            "Pi-to-OpenAI conversion message count differs from captured provider request"
        )


def build_route_replay_request(
    entries: list[JsonObject], provider_request: JsonObject
) -> tuple[JsonObject, dict[str, Any]]:
    """Build one max-one-token request from Pi's final active context."""
    context_entries = build_compaction_aware_entries(entries)
    context_messages = [
        message
        for entry in context_entries
        if (message := convert_pi_entry_to_openai(entry)) is not None
    ]
    system_message = provider_request["messages"][0]
    request = {
        "model": provider_request["model"],
        "messages": [system_message, *context_messages],
        "stream": False,
        "max_tokens": 1,
        "tools": provider_request["tools"],
        "reasoning_effort": provider_request.get("reasoning_effort", "max"),
        "temperature": 0,
        "top_p": provider_request.get("top_p", 0.95),
    }
    metadata = {
        "source_entry_count": len(entries),
        "context_entry_count": len(context_entries),
        "context_message_count": len(context_messages),
        "assistant_message_count": sum(
            message.get("role") == "assistant" for message in context_messages
        ),
        "tool_message_count": sum(
            message.get("role") == "tool" for message in context_messages
        ),
        "compaction_count": sum(
            entry.get("type") == "compaction" for entry in context_entries
        ),
    }
    return request, metadata


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--provider-request-1", type=Path, required=True)
    parser.add_argument("--provider-request-2", type=Path, required=True)
    parser.add_argument("--output-request", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()

    entries = load_pi_session_entries(args.session)
    first_request = load_json(args.provider_request_1)
    second_request = load_json(args.provider_request_2)
    validate_second_provider_request(entries, second_request)
    request, metadata = build_route_replay_request(entries, first_request)
    write_json_atomic(args.output_request, request)
    request_bytes = args.output_request.read_bytes()
    metadata.update(
        {
            "source_session_path": str(args.session),
            "source_session_sha256": hashlib.sha256(
                args.session.read_bytes()
            ).hexdigest(),
            "provider_request_1_path": str(args.provider_request_1),
            "provider_request_1_sha256": hashlib.sha256(
                args.provider_request_1.read_bytes()
            ).hexdigest(),
            "provider_request_2_path": str(args.provider_request_2),
            "provider_request_2_sha256": hashlib.sha256(
                args.provider_request_2.read_bytes()
            ).hexdigest(),
            "request_path": str(args.output_request),
            "request_bytes": len(request_bytes),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "provider_request_2_conversion": "exact",
        }
    )
    write_json_atomic(args.output_metadata, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
