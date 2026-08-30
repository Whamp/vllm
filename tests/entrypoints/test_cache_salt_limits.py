# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.pooling.base.protocol import PoolingBasicRequestMixin
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MAX_CACHE_SALT_LENGTH = 1024

REQUEST_CASES: tuple[tuple[type[BaseModel], dict[str, Any]], ...] = (
    (
        AnthropicMessagesRequest,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
        },
    ),
    (
        ChatCompletionRequest,
        {"messages": [{"role": "user", "content": "hello"}]},
    ),
    (
        BatchChatCompletionRequest,
        {"messages": [[{"role": "user", "content": "hello"}]]},
    ),
    (CompletionRequest, {"prompt": "hello"}),
    (ResponsesRequest, {"input": "hello"}),
    (PoolingBasicRequestMixin, {}),
    (GenerateRequest, {"token_ids": [1], "sampling_params": {}}),
)


@pytest.mark.parametrize(("request_type", "payload"), REQUEST_CASES)
def test_cache_salt_accepts_1024_unicode_code_points(
    request_type: type[BaseModel], payload: dict[str, Any]
) -> None:
    cache_salt = "é" * MAX_CACHE_SALT_LENGTH

    request = request_type.model_validate({**payload, "cache_salt": cache_salt})

    assert request.cache_salt == cache_salt


@pytest.mark.parametrize(("request_type", "payload"), REQUEST_CASES)
def test_cache_salt_rejects_more_than_1024_code_points(
    request_type: type[BaseModel], payload: dict[str, Any]
) -> None:
    cache_salt = "é" * (MAX_CACHE_SALT_LENGTH + 1)

    with pytest.raises(ValidationError, match="cache_salt"):
        request_type.model_validate({**payload, "cache_salt": cache_salt})


@pytest.mark.parametrize(("request_type", "payload"), REQUEST_CASES)
def test_cache_salt_rejects_empty_string(
    request_type: type[BaseModel], payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="cache_salt"):
        request_type.model_validate({**payload, "cache_salt": ""})


@pytest.mark.parametrize("request_type", [case[0] for case in REQUEST_CASES])
def test_cache_salt_openapi_schema_exposes_length_bound(
    request_type: type[BaseModel],
) -> None:
    variants = request_type.model_json_schema()["properties"]["cache_salt"]["anyOf"]
    string_schema = next(item for item in variants if item.get("type") == "string")

    assert string_schema["minLength"] == 1
    assert string_schema["maxLength"] == MAX_CACHE_SALT_LENGTH


def test_batch_chat_cache_salt_survives_single_request_conversion() -> None:
    cache_salt = "s" * MAX_CACHE_SALT_LENGTH
    messages = [{"role": "user", "content": "hello"}]
    request = BatchChatCompletionRequest(
        messages=[messages],
        cache_salt=cache_salt,
    )

    single_request = request.to_chat_completion_request(messages)

    assert single_request.cache_salt == cache_salt
