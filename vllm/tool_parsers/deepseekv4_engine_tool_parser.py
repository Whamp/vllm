# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import DeepSeekV4ParserToolAdapter

DEEPSEEK_V4_TOOL_CALLS_END = "</｜DSML｜tool_calls>"


class DeepSeekV4EngineToolParser(DeepSeekV4ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "deepseek_v4"

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if (
            not isinstance(request, ChatCompletionRequest)
            or not request.tools
            or request.tool_choice == "none"
        ):
            return request

        # A completed DSML block is one assistant tool-call turn. Without this
        # boundary DeepSeek V4 can continue emitting new assistant/tool blocks
        # in the same generation. The parser sees the delimiter before the
        # serving layer removes the matched stop string from returned text.
        if request.stop is None:
            stop = []
        elif isinstance(request.stop, str):
            stop = [request.stop]
        else:
            stop = list(request.stop)
        if DEEPSEEK_V4_TOOL_CALLS_END not in stop:
            stop.append(DEEPSEEK_V4_TOOL_CALLS_END)
        request.stop = stop
        return request
