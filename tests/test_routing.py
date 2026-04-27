"""
Tests for the lead-to-gold routing logic in claw_cli.

Locks in the contract that:
  - read-only tools route to SMALL_MODEL
  - any write/exec tool forces the main model
  - intent-based pick at iteration 1 only triggers on specific phrasings
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import claw_cli  # noqa: E402


class SmallModelSafeToolsTests(unittest.TestCase):
    """The set of tools the small model is allowed to compose responses for."""

    def test_read_only_tools_are_safe(self):
        # These must stay safe for cheap routing — no filesystem mutation.
        for name in ("read_file", "glob_search", "grep_search", "memory_search"):
            self.assertIn(name, claw_cli.SMALL_MODEL_SAFE_TOOLS, name)

    def test_write_and_exec_tools_are_NOT_safe(self):
        # If any of these slip into SAFE, weak models would corrupt state.
        for name in ("write_file", "edit_file", "bash", "scaffold_project", "memory_save"):
            self.assertNotIn(name, claw_cli.SMALL_MODEL_SAFE_TOOLS, name)


class PickModelForTaskTests(unittest.TestCase):
    """Coarse iteration-1 router based on user message phrasing."""

    def test_empty_messages_returns_default(self):
        self.assertEqual(
            claw_cli._pick_model_for_task([], "main-model"),
            "main-model",
        )

    def test_short_status_question_routes_small(self):
        msgs = [{"role": "user", "content": "what is the status?"}]
        self.assertEqual(claw_cli._pick_model_for_task(msgs, "main-model"), claw_cli.SMALL_MODEL)

    def test_short_show_me_routes_small(self):
        msgs = [{"role": "user", "content": "show me the file"}]
        self.assertEqual(claw_cli._pick_model_for_task(msgs, "main-model"), claw_cli.SMALL_MODEL)

    def test_long_message_uses_main(self):
        # Even with trigger words, a long message implies real work
        long_msg = "what is the right way to refactor this enormous module " * 5
        msgs = [{"role": "user", "content": long_msg}]
        self.assertEqual(claw_cli._pick_model_for_task(msgs, "main-model"), "main-model")

    def test_no_trigger_words_uses_main(self):
        msgs = [{"role": "user", "content": "build me a website"}]
        self.assertEqual(claw_cli._pick_model_for_task(msgs, "main-model"), "main-model")


class PickModelForToolCallTests(unittest.TestCase):
    """The per-tool-call router that decides who runs the next iteration."""

    @staticmethod
    def _tc(name, args=None):
        return {"function": {"name": name, "arguments": args or {}}}

    def test_no_tool_calls_returns_None(self):
        # No previous tool calls → no routing decision (caller falls back).
        self.assertIsNone(claw_cli._pick_model_for_tool_call(None))
        self.assertIsNone(claw_cli._pick_model_for_tool_call([]))

    def test_all_safe_tools_route_small(self):
        calls = [self._tc("read_file"), self._tc("grep_search")]
        self.assertEqual(claw_cli._pick_model_for_tool_call(calls), claw_cli.SMALL_MODEL)

    def test_any_unsafe_tool_returns_None(self):
        calls = [self._tc("read_file"), self._tc("write_file")]
        self.assertIsNone(claw_cli._pick_model_for_tool_call(calls))

    def test_bash_forces_main(self):
        calls = [self._tc("bash")]
        self.assertIsNone(claw_cli._pick_model_for_tool_call(calls))

    def test_unknown_tool_treated_as_unsafe(self):
        # If a model invents a tool name, conservative default is main model.
        calls = [self._tc("do_a_dance")]
        self.assertIsNone(claw_cli._pick_model_for_tool_call(calls))

    def test_handles_dict_without_function_wrapper(self):
        # Some code paths may pass {"name": ...} directly.
        calls = [{"name": "read_file"}]
        self.assertEqual(claw_cli._pick_model_for_tool_call(calls), claw_cli.SMALL_MODEL)


if __name__ == "__main__":
    unittest.main()
