"""Regression tests for _validate_tool_args (pre-tool argument validator)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import claw_cli  # noqa: E402


class ToolArgValidatorTests(unittest.TestCase):
    def test_well_formed_call_returns_none(self):
        self.assertIsNone(claw_cli._validate_tool_args(
            "write_file", {"file_path": "x.py", "content": "ok"}
        ))

    def test_missing_required_arg_caught(self):
        err = claw_cli._validate_tool_args("write_file", {"file_path": "x.py"})
        self.assertIsNotNone(err)
        self.assertIn("content", err)
        self.assertIn("missing", err.lower())

    def test_empty_required_arg_treated_as_missing(self):
        # Empty string is not useful — treat it as missing so the model fixes it.
        err = claw_cli._validate_tool_args("read_file", {"file_path": ""})
        self.assertIsNotNone(err)
        self.assertIn("file_path", err)

    def test_alias_redirected_to_right_key(self):
        # qwen2.5-coder sometimes uses 'new_string' on a write_file call.
        err = claw_cli._validate_tool_args(
            "write_file", {"file_path": "x.py", "new_string": "ok"}
        )
        self.assertIsNotNone(err)
        self.assertIn("'new_string'", err)
        self.assertIn("'content'", err)

    def test_path_alias_redirects(self):
        err = claw_cli._validate_tool_args(
            "read_file", {"path": "x.py"}
        )
        self.assertIsNotNone(err)
        self.assertIn("'path'", err)
        self.assertIn("'file_path'", err)

    def test_invalid_arg_for_tool(self):
        # 'content' isn't valid on edit_file (which uses old_string/new_string).
        err = claw_cli._validate_tool_args(
            "edit_file", {"file_path": "x.py", "content": "stuff"}
        )
        self.assertIsNotNone(err)
        self.assertIn("does not accept 'content'", err)

    def test_unknown_tool_returns_none(self):
        # Tools without a contract (db_schema, env_manage, etc.) must not error.
        self.assertIsNone(claw_cli._validate_tool_args("db_schema", {}))

    def test_bash_command_required(self):
        err = claw_cli._validate_tool_args("bash", {})
        self.assertIsNotNone(err)
        self.assertIn("command", err)

    def test_bash_alias_cmd(self):
        err = claw_cli._validate_tool_args("bash", {"cmd": "ls"})
        self.assertIsNotNone(err)
        self.assertIn("'cmd'", err)
        self.assertIn("'command'", err)

    def test_edit_file_complete_call(self):
        self.assertIsNone(claw_cli._validate_tool_args(
            "edit_file", {"file_path": "x.py", "old_string": "old", "new_string": "new"}
        ))


if __name__ == "__main__":
    unittest.main()
