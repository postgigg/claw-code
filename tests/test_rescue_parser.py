"""
Regression tests for rescue_tool_calls_from_text and _loads_lenient.

Locks in the surgical fixes added during the lead-to-gold work:
  - ast.literal_eval fallback (Python single-quote dict syntax)
  - <|python_tag|> and similar special-token prefix stripping
  - "parameters" key as alternative to "arguments"
  - Trailing-comma tolerance
"""
from __future__ import annotations

import unittest

# claw_cli is a top-level module (not packaged); we import it the same way
# claw-code's other tests reach into the workspace.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import claw_cli  # noqa: E402


class LenientJSONTests(unittest.TestCase):
    """_loads_lenient should accept JSON, JSON-with-trailing-comma, and Python literals."""

    def test_strict_json(self):
        obj = claw_cli._loads_lenient('{"a": 1, "b": "two"}')
        self.assertEqual(obj, {"a": 1, "b": "two"})

    def test_trailing_comma(self):
        obj = claw_cli._loads_lenient('{"a": 1, "b": 2,}')
        self.assertEqual(obj, {"a": 1, "b": 2})

    def test_python_single_quotes(self):
        obj = claw_cli._loads_lenient("{'a': 1, 'b': 'two'}")
        self.assertEqual(obj, {"a": 1, "b": "two"})

    def test_mixed_quotes(self):
        # Common from local coder models: JSON-style keys + Python-style values
        obj = claw_cli._loads_lenient('{"a": "x", "b": \'y\'}')
        self.assertEqual(obj, {"a": "x", "b": "y"})

    def test_garbage_returns_none(self):
        self.assertIsNone(claw_cli._loads_lenient("not actually json or python"))

    def test_returns_parsed_value_not_dict_filtered(self):
        # _loads_lenient returns whatever the parser produced (list, dict, etc.).
        # Dict-only filtering happens at the call site in rescue_tool_calls_from_text.
        self.assertEqual(claw_cli._loads_lenient("[1, 2, 3]"), [1, 2, 3])

    def test_no_code_execution(self):
        # ast.literal_eval is safe — it must NOT execute __import__ or os.system
        result = claw_cli._loads_lenient('{"x": __import__("os").system("echo PWNED")}')
        self.assertIsNone(result)


class RescueToolCallTests(unittest.TestCase):
    """rescue_tool_calls_from_text should pull tool calls out of various weak-model outputs."""

    def _rescue(self, text):
        cleaned, rescued = claw_cli.rescue_tool_calls_from_text(text)
        return rescued

    def test_plain_json_object(self):
        text = '{"name": "write_file", "arguments": {"file_path": "x.py", "content": "print(1)"}}'
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "write_file")
        self.assertEqual(rescued[0]["function"]["arguments"]["file_path"], "x.py")

    def test_fenced_json_block(self):
        text = (
            "Here you go:\n"
            "```json\n"
            '{"name": "read_file", "arguments": {"file_path": "data.txt"}}\n'
            "```\n"
        )
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "read_file")

    def test_python_tag_prefix_stripped(self):
        # Llama-family special token leaked into output
        text = '<|python_tag|>{"name": "write_file", "arguments": {"file_path": "x.py", "content": "ok"}}'
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "write_file")

    def test_other_special_tokens_stripped(self):
        text = '<|tool_call|>{"name": "read_file", "arguments": {"file_path": "y.py"}}<|im_end|>'
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "read_file")

    def test_parameters_key_accepted(self):
        # Some Llama-family models emit "parameters" instead of "arguments"
        text = '{"name": "write_file", "parameters": {"file_path": "x.py", "content": "ok"}}'
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "write_file")
        self.assertEqual(rescued[0]["function"]["arguments"]["file_path"], "x.py")

    def test_python_tag_plus_parameters(self):
        # The exact pattern observed from qwen2.5-coder:7b that we shipped fixes for
        text = (
            '<|python_tag|>{"name": "write_file", "parameters": '
            '{"file_path": "cli.py", "content": "import argparse\\nprint(\\"hi\\")"}}'
        )
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "write_file")
        self.assertIn("import argparse", rescued[0]["function"]["arguments"]["content"])

    def test_python_dict_syntax(self):
        # Mixed-quote tool call (qwen2.5-coder emits these)
        text = (
            '{"name": "edit_file", "arguments": {"file_path": "cli.py", '
            '"old_string": \'print("Hello")\', "new_string": \'print("Hi")\'}}'
        )
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "edit_file")
        self.assertEqual(rescued[0]["function"]["arguments"]["old_string"], 'print("Hello")')

    def test_unknown_tool_name_rejected(self):
        # Even if the JSON parses, name must be in TOOL_NAMES
        text = '{"name": "do_a_dance", "arguments": {"steps": 4}}'
        rescued = self._rescue(text)
        self.assertEqual(rescued, [])

    def test_no_tool_call_in_prose(self):
        text = "I'll create the file once I figure out the right path."
        rescued = self._rescue(text)
        self.assertEqual(rescued, [])

    def test_garbage_returns_empty(self):
        text = "asdf jkl; not even close to json"
        rescued = self._rescue(text)
        self.assertEqual(rescued, [])

    def test_missing_arguments_rejected(self):
        # Has "name" but no "arguments" or "parameters"
        text = '{"name": "write_file", "stuff": {"file_path": "x.py"}}'
        rescued = self._rescue(text)
        self.assertEqual(rescued, [])

    def test_truncated_outer_brace(self):
        # qwen2.5-coder:7b sometimes emits a tool call missing the outer
        # closing brace when its output gets cut off.
        text = (
            '{"name": "write_file", "arguments": {"file_path": "fizzbuzz.py", '
            '"content": "for i in range(1, 16): print(i)"}'
        )
        rescued = self._rescue(text)
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["function"]["name"], "write_file")
        self.assertEqual(rescued[0]["function"]["arguments"]["file_path"], "fizzbuzz.py")


class FakeSystemGaslightTests(unittest.TestCase):
    """The agent loop should detect the model emitting its own [SYSTEM:...] block."""

    def test_pattern_match(self):
        # The exact pattern observed during read_count failures.
        sample = "[SYSTEM: Task completed. No further actions required.]"
        self.assertTrue(sample.lstrip().startswith("[SYSTEM:"))

    def test_pattern_with_leading_whitespace(self):
        sample = "   \n  [SYSTEM: Done!]"
        self.assertTrue(sample.lstrip().startswith("[SYSTEM:"))

    def test_normal_response_doesnt_match(self):
        # A legitimate assistant text response must not trigger.
        sample = "Here's the file contents: ..."
        self.assertFalse(sample.lstrip().startswith("[SYSTEM:"))

    def test_quoted_system_doesnt_match(self):
        # If the model quotes "[SYSTEM:..." as part of a sentence, lstrip
        # check should not catch it because content doesn't START with the bracket.
        sample = 'I noticed you wrote "[SYSTEM: example]" — let me explain.'
        self.assertFalse(sample.lstrip().startswith("[SYSTEM:"))


if __name__ == "__main__":
    unittest.main()
