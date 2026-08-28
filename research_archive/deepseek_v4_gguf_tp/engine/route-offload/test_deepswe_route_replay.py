#!/usr/bin/env python3
"""Tests for the compaction-aware DeepSWE route replay builder."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build_deepswe_route_replay.py")
SPEC = importlib.util.spec_from_file_location("build_deepswe_route_replay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class DeepSweRouteReplayTests(unittest.TestCase):
    def test_latest_compaction_precedes_kept_and_post_compaction_entries(self):
        entries = [
            {"type": "session", "version": 3},
            {"type": "model_change", "id": "m", "parentId": None},
            {
                "type": "message",
                "id": "u",
                "parentId": "m",
                "message": {"role": "user", "content": []},
            },
            {
                "type": "message",
                "id": "kept",
                "parentId": "u",
                "message": {"role": "assistant", "content": []},
            },
            {
                "type": "compaction",
                "id": "compact",
                "parentId": "kept",
                "firstKeptEntryId": "kept",
                "summary": "summary",
            },
            {
                "type": "message",
                "id": "post",
                "parentId": "compact",
                "message": {"role": "user", "content": []},
            },
        ]
        context = replay.build_compaction_aware_entries(entries)
        self.assertEqual(
            [entry["id"] for entry in context], ["compact", "kept", "post"]
        )

    def test_tool_only_assistant_uses_null_openai_content(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "inspect"},
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "bash",
                    "arguments": {"command": "pwd"},
                },
            ],
        }
        converted = replay.convert_pi_message_to_openai(message)
        self.assertIsNone(converted["content"])
        self.assertEqual(converted["reasoning_content"], "inspect")
        self.assertEqual(
            converted["tool_calls"][0]["function"]["arguments"],
            '{"command":"pwd"}',
        )


if __name__ == "__main__":
    unittest.main()
