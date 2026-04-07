#!/usr/bin/env python3
"""
Rattlesnake -- a free, local agent CLI powered by Ollama.

Usage:
    rattlesnake                               # interactive REPL
    rattlesnake "fix the bug"                 # one-shot prompt
    rattlesnake --model qwen2.5-coder:14b
    rattlesnake --file screenshot.png "whats this?"
"""

import base64
import json
import os
import shlex
import shutil
import sys
import glob as glob_mod
import re
import subprocess
import threading
import datetime
import time
import urllib.request
import urllib.error
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template
import hashlib
import math
from textwrap import dedent

# Fix Windows console encoding + enable VT100 ANSI support
_ANSI_CURSOR_OK = True  # whether cursor-up/erase-line ANSI codes work
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")  # enable ANSI escape codes on Windows
    # Enable proper VT100 processing on Windows 10+
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(-11)
        # Get current mode
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        _ANSI_CURSOR_OK = False  # VT100 not available, skip cursor manipulation

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("CLAW_MODEL", "qwen/qwen3.6-plus:free")
DEFAULT_VISION_MODEL = os.environ.get("CLAW_VISION_MODEL", "llama3.2-vision:11b")
MAX_ITERATIONS = 128
TOOL_OUTPUT_LIMIT = 12000
_ONE_SHOT_MODE = False  # set True in one-shot CLI invocations (no interactive input)
# Timeout for requests (configurable via env — weaker models need more time)
OLLAMA_STREAM_TIMEOUT = int(os.environ.get("CLAW_TIMEOUT", "1800"))  # 30 min default
OLLAMA_SYNC_TIMEOUT = int(os.environ.get("CLAW_TIMEOUT", "1800"))    # 30 min default
CWD = os.getcwd()
SCRIPT_DIR = Path(__file__).resolve().parent
SMALL_MODEL = os.environ.get("CLAW_SMALL_MODEL", "deepseek/deepseek-chat-v3-0324")  # fast model for simple tasks
SESSIONS_DIR = Path.home() / ".claw" / "sessions"
SNAPSHOTS_DIR = Path.home() / ".claw" / "snapshots"

# Inner monologue & reflection
THINKING_ENABLED = os.environ.get("CLAW_THINKING", "1") != "0"
REFLECTION_ENABLED = os.environ.get("CLAW_REFLECTION", "1") != "0"
QUALITY_GATE_ENABLED = os.environ.get("CLAW_QUALITY_GATE", "1") != "0"
REFLECTION_AFTER_N = 2        # reflect after every N tool rounds
ANTIREPEAT_THRESHOLD = 3      # trigger after N similar actions
THINK_RETRY_ON_MISSING = True # retry once if model omits <think> tags
WIRING_ENABLED = os.environ.get("CLAW_WIRING", "1") != "0"

# Multi-provider support
PROVIDER = os.environ.get("CLAW_PROVIDER", "openrouter")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# Windows: detect Git Bash for Unix command translation
_HAS_GIT_BASH = shutil.which("bash") is not None if sys.platform == "win32" else False

# Template and API registry search paths (bundled first, then user-level)
# Data directory search paths: source dir → pip install share dir → user home
_SHARE_DIR = Path(sys.prefix) / "share" / "claw-code"
TEMPLATES_DIRS = [
    SCRIPT_DIR / "templates",
    _SHARE_DIR / "templates",
    Path.home() / ".claw" / "templates",
]
APIS_DIR_PATHS = [
    SCRIPT_DIR / "apis",
    _SHARE_DIR / "apis",
    Path.home() / ".claw" / "apis",
]
PROMPTS_DIR_PATHS = [
    SCRIPT_DIR / "prompts",
    _SHARE_DIR / "prompts",
    Path.home() / ".claw" / "prompts",
]

# Verification display
VERIFY_OK = "\033[38;2;44;190;90m\u2713\033[0m"   # green checkmark
VERIFY_FAIL = "\033[38;2;220;60;80m\u2717\033[0m"  # red X

# ---------------------------------------------------------------------------
# LLM provider abstraction
# ---------------------------------------------------------------------------

# Known cloud model context sizes
_CLOUD_CONTEXT_SIZES = {
    "claude": 200000, "gpt-4o": 128000, "gpt-4": 128000, "gpt-3.5": 16385,
    "llama": 131072, "qwen": 32768, "mistral": 32768, "deepseek": 65536,
    "gemma": 8192, "phi": 16384, "command-r": 128000,
}

class LLMProvider:
    """Abstract base for LLM providers."""
    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        raise NotImplementedError
    def get_context_size(self, model):
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Ollama local inference."""
    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"num_ctx": num_ctx},
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=OLLAMA_STREAM_TIMEOUT)
                break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if resp is None:
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_BASE} after 3 attempts. Is it running?\n{last_err}"
            )

        _prompt_est = sum(len(m.get("content", "").split()) for m in messages)
        _token_tracker.add(prompt=_prompt_est)

        if stream:
            _comp_tokens = 0
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        _comp_tokens += len(content.split())
                    if chunk.get("done"):
                        actual = chunk.get("eval_count", _comp_tokens)
                        _token_tracker.add(completion=actual)
                    yield chunk
                except json.JSONDecodeError:
                    continue
        else:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            _token_tracker.add(completion=data.get("eval_count", len(data.get("message", {}).get("content", "").split())))
            yield data

    def get_context_size(self, model):
        """Query Ollama for model context size via /api/show."""
        try:
            payload = json.dumps({"name": model}).encode("utf-8")
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            info = json.loads(resp.read().decode("utf-8"))
            # Parse num_ctx from modelfile parameters
            modelfile = info.get("modelfile", "") or info.get("parameters", "")
            for line in str(modelfile).split("\n"):
                if "num_ctx" in line:
                    parts = line.strip().split()
                    for p in parts:
                        if p.isdigit():
                            return int(p)
            # Check model_info for context_length
            model_info = info.get("model_info", {})
            for key, val in model_info.items():
                if "context" in key.lower() and isinstance(val, (int, float)):
                    return int(val)
        except Exception:
            pass
        return 8192  # safe fallback


class OpenRouterProvider(LLMProvider):
    """OpenRouter API (OpenAI-compatible with SSE streaming)."""
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set. Export it or use --api-key.")

        payload = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        # Always set max_tokens — without this, providers use tiny defaults
        payload["max_tokens"] = min(num_ctx, 16384) if num_ctx else 16384

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/claw-project/rattlesnake",
                "X-Title": "Rattlesnake CLI",
            },
            method="POST",
        )

        _prompt_est = sum(len(m.get("content", "").split()) for m in messages)
        _token_tracker.add(prompt=_prompt_est)

        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=OLLAMA_STREAM_TIMEOUT)
                break
            except urllib.error.HTTPError as e:
                # Read the response body for detailed error info
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    err_body = ""
                last_err = f"HTTP {e.code}: {err_body}" if err_body else e
                if e.code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                # 4xx errors — don't retry, they won't change
                if e.code >= 400 and e.code < 500:
                    break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if resp is None:
            raise ConnectionError(f"Cannot reach OpenRouter after 3 attempts.\n{last_err}")

        if stream:
            yield from self._parse_sse_stream(resp)
        else:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            yield self._openai_to_ollama(data)

    def _parse_sse_stream(self, resp):
        """Parse SSE stream from OpenAI-compatible endpoint."""
        _comp_tokens = 0
        accumulated_tool_calls = {}  # index -> {id, name, arguments}
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                # Emit final chunk with accumulated tool calls
                final = {"done": True, "message": {"content": "", "role": "assistant"}}
                if accumulated_tool_calls:
                    final["message"]["tool_calls"] = [
                        {"id": tc.get("id", f"call_{i}"), "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for i, tc in enumerate(accumulated_tool_calls.values())
                    ]
                _token_tracker.add(completion=_comp_tokens)
                yield final
                return
            try:
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "") or ""
                if content:
                    _comp_tokens += len(content.split())

                # Handle streamed tool calls
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.get("id"):
                            accumulated_tool_calls[idx]["id"] = tc["id"]
                        if tc.get("function", {}).get("name"):
                            accumulated_tool_calls[idx]["name"] = tc["function"]["name"]
                        if tc.get("function", {}).get("arguments"):
                            accumulated_tool_calls[idx]["arguments"] += tc["function"]["arguments"]

                # Yield content chunks in Ollama format
                ollama_chunk = {
                    "message": {"role": "assistant", "content": content},
                    "done": False,
                }
                # If finish_reason is present, attach completed tool calls
                finish = chunk.get("choices", [{}])[0].get("finish_reason")
                if finish == "tool_calls" and accumulated_tool_calls:
                    ollama_chunk["message"]["tool_calls"] = [
                        {"id": tc.get("id", f"call_{i}"), "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for i, tc in enumerate(accumulated_tool_calls.values())
                    ]
                yield ollama_chunk
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

    def _openai_to_ollama(self, data):
        """Convert OpenAI response format to Ollama format."""
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        result = {
            "message": {"role": "assistant", "content": msg.get("content", "") or ""},
            "done": True,
        }
        if msg.get("tool_calls"):
            result["message"]["tool_calls"] = []
            for i, tc in enumerate(msg["tool_calls"]):
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
                result["message"]["tool_calls"].append(
                    {"id": tc.get("id", f"call_{i}"), "function": {"name": fn.get("name", ""), "arguments": args}}
                )
        usage = data.get("usage", {})
        _token_tracker.add(completion=usage.get("completion_tokens", 0))
        return result

    def get_context_size(self, model):
        model_lower = model.lower()
        return next((v for k, v in _CLOUD_CONTEXT_SIZES.items() if k in model_lower), 128000)


class OpenAIProvider(LLMProvider):
    """OpenAI API."""
    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Export it or use --api-key.")

        payload = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            method="POST",
        )

        _prompt_est = sum(len(m.get("content", "").split()) for m in messages)
        _token_tracker.add(prompt=_prompt_est)

        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=OLLAMA_STREAM_TIMEOUT)
                break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if resp is None:
            raise ConnectionError(f"Cannot reach OpenAI API after 3 attempts.\n{last_err}")

        # Reuse OpenRouter's SSE parser — same format
        _or = OpenRouterProvider()
        if stream:
            yield from _or._parse_sse_stream(resp)
        else:
            body = resp.read().decode("utf-8")
            yield _or._openai_to_ollama(json.loads(body))

    def get_context_size(self, model):
        model_lower = model.lower()
        return next((v for k, v in _CLOUD_CONTEXT_SIZES.items() if k in model_lower), 128000)


class DashScopeProvider(LLMProvider):
    """Alibaba Cloud DashScope API (OpenAI-compatible) for Qwen models."""
    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    _last_request_time = 0.0  # class-level rate limiter
    _MIN_REQUEST_GAP = float(os.environ.get("DASHSCOPE_COOLDOWN", "1.5"))  # seconds between requests

    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY not set. Export it or use --api-key.")

        # Rate limit: enforce minimum gap between requests
        now = time.time()
        elapsed = now - DashScopeProvider._last_request_time
        if elapsed < self._MIN_REQUEST_GAP:
            time.sleep(self._MIN_REQUEST_GAP - elapsed)
        DashScopeProvider._last_request_time = time.time()

        # Strip OpenRouter-style prefixes (e.g. "qwen/qwen3-235b-a22b:free" -> "qwen3-235b-a22b")
        clean_model = model
        if "/" in clean_model:
            clean_model = clean_model.split("/", 1)[1]
        # Remove :free or :extended suffixes
        if ":" in clean_model:
            clean_model = clean_model.split(":")[0]
        # Map to DashScope model IDs (they use different naming)
        _DASHSCOPE_MODEL_MAP = {
            "qwen3-235b-a22b": "qwen-plus-latest",
            "qwen3-32b": "qwen-turbo-latest",
            "qwen3-30b-a3b": "qwen-turbo-latest",
            "qwen3.6-plus": "qwen3.5-plus",
        }
        clean_model = _DASHSCOPE_MODEL_MAP.get(clean_model, clean_model)

        # DashScope requires tool_call arguments to be valid JSON strings — sanitize
        clean_messages = []
        for msg in messages:
            m = dict(msg)
            if m.get("tool_calls"):
                fixed_tcs = []
                for tc in m["tool_calls"]:
                    tc = dict(tc)
                    if "function" in tc:
                        fn = dict(tc["function"])
                        args = fn.get("arguments", "{}")
                        if isinstance(args, dict):
                            fn["arguments"] = json.dumps(args)
                        elif isinstance(args, str):
                            # Validate it's valid JSON, fix if not
                            try:
                                json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                fn["arguments"] = json.dumps({"input": args})
                        tc["function"] = fn
                    fixed_tcs.append(tc)
                m["tool_calls"] = fixed_tcs
            clean_messages.append(m)

        payload = {"model": clean_model, "messages": clean_messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        payload["max_tokens"] = min(num_ctx, 16384) if num_ctx else 16384

        data = json.dumps(payload).encode("utf-8")

        _prompt_est = sum(len(m.get("content", "").split()) for m in messages)
        _token_tracker.add(prompt=_prompt_est)

        resp = None
        last_err = None
        # DashScope rate-limits aggressively — use more retries with longer backoff
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.BASE_URL, data=data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
                    method="POST",
                )
                resp = urllib.request.urlopen(req, timeout=OLLAMA_STREAM_TIMEOUT)
                break
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    err_body = ""
                last_err = f"HTTP {e.code}: {err_body}" if err_body else e
                # Retry on 429/500/503 (rate limits and server errors)
                if (e.code == 429 or e.code >= 500) and attempt < 4:
                    wait = min(3 * (2 ** attempt), 30)  # 3s, 6s, 12s, 24s
                    time.sleep(wait)
                    continue
                if e.code >= 400 and e.code < 500:
                    break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                if attempt < 4:
                    time.sleep(3 * (2 ** attempt))
        if resp is None:
            raise ConnectionError(f"Cannot reach DashScope API after 5 attempts.\n{last_err}")

        # Reuse OpenRouter's SSE parser — same OpenAI-compatible format
        _or = OpenRouterProvider()
        if stream:
            yield from _or._parse_sse_stream(resp)
        else:
            body = resp.read().decode("utf-8")
            yield _or._openai_to_ollama(json.loads(body))

    def get_context_size(self, model):
        model_lower = model.lower()
        if "qwen3" in model_lower or "qwen-plus" in model_lower:
            return 131072
        return next((v for k, v in _CLOUD_CONTEXT_SIZES.items() if k in model_lower), 32768)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API."""
    BASE_URL = "https://api.anthropic.com/v1/messages"

    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set. Export it or use --api-key.")

        # Separate system message
        system_text = ""
        api_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_text += m.get("content", "") + "\n"
            elif m.get("role") == "tool":
                # Anthropic uses tool_result blocks
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m.get("tool_use_id", "tool_0"), "content": m.get("content", "")}]
                })
            else:
                api_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        # Ensure messages alternate user/assistant
        api_messages = self._fix_message_order(api_messages)

        payload = {"model": model, "messages": api_messages, "max_tokens": min(num_ctx, 8192), "stream": stream}
        if system_text.strip():
            payload["system"] = system_text.strip()

        # Convert tools to Anthropic format
        if tools:
            payload["tools"] = self._convert_tools(tools)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        _prompt_est = sum(len(m.get("content", "").split()) for m in messages)
        _token_tracker.add(prompt=_prompt_est)

        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=OLLAMA_STREAM_TIMEOUT)
                break
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if resp is None:
            raise ConnectionError(f"Cannot reach Anthropic API after 3 attempts.\n{last_err}")

        if stream:
            yield from self._parse_anthropic_stream(resp)
        else:
            body = resp.read().decode("utf-8")
            yield self._anthropic_to_ollama(json.loads(body))

    def _fix_message_order(self, messages):
        """Ensure messages alternate user/assistant for Anthropic."""
        if not messages:
            return [{"role": "user", "content": "Hello"}]
        fixed = []
        for m in messages:
            if fixed and fixed[-1]["role"] == m["role"]:
                # Merge same-role messages
                if isinstance(fixed[-1]["content"], str) and isinstance(m["content"], str):
                    fixed[-1]["content"] += "\n" + m["content"]
                else:
                    fixed.append(m)
            else:
                fixed.append(m)
        if fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "Begin."})
        return fixed

    def _convert_tools(self, tools):
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", {})
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def _parse_anthropic_stream(self, resp):
        """Parse Anthropic SSE stream."""
        _comp_tokens = 0
        tool_calls = []
        current_tool = None
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
                event_type = event.get("type", "")
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            _comp_tokens += len(text.split())
                        yield {"message": {"role": "assistant", "content": text}, "done": False}
                    elif delta.get("type") == "input_json_delta":
                        if current_tool is not None:
                            current_tool["arguments"] += delta.get("partial_json", "")
                elif event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {"name": block.get("name", ""), "arguments": ""}
                        tool_calls.append(current_tool)
                elif event_type == "content_block_stop":
                    current_tool = None
                elif event_type == "message_stop":
                    final = {"message": {"role": "assistant", "content": ""}, "done": True}
                    if tool_calls:
                        final["message"]["tool_calls"] = []
                        for tc in tool_calls:
                            try:
                                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            final["message"]["tool_calls"].append(
                                {"function": {"name": tc["name"], "arguments": args}}
                            )
                    _token_tracker.add(completion=_comp_tokens)
                    yield final
                    return
                elif event_type == "message_delta":
                    usage = event.get("usage", {})
                    if usage.get("output_tokens"):
                        _token_tracker.add(completion=usage["output_tokens"])
            except (json.JSONDecodeError, KeyError):
                continue

    def _anthropic_to_ollama(self, data):
        """Convert Anthropic response to Ollama format."""
        content_parts = data.get("content", [])
        text = ""
        tool_calls = []
        for block in content_parts:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "function": {"name": block.get("name", ""), "arguments": block.get("input", {})}
                })
        result = {"message": {"role": "assistant", "content": text}, "done": True}
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls
        usage = data.get("usage", {})
        _token_tracker.add(completion=usage.get("output_tokens", 0))
        return result

    def get_context_size(self, model):
        return 200000  # All Claude models support 200K


# Provider singleton
_provider_instance = None

def _get_provider():
    """Return the active LLM provider based on PROVIDER config."""
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance
    p = PROVIDER.lower()
    if p == "openrouter":
        _provider_instance = OpenRouterProvider()
    elif p == "openai":
        _provider_instance = OpenAIProvider()
    elif p == "anthropic":
        _provider_instance = AnthropicProvider()
    elif p == "dashscope":
        _provider_instance = DashScopeProvider()
    else:
        _provider_instance = OllamaProvider()
    return _provider_instance


def _get_model_context_size(model):
    """Auto-detect context window size for the active model."""
    provider = _get_provider()
    return provider.get_context_size(model)


# ---------------------------------------------------------------------------
# permission modes + plan mode state
# ---------------------------------------------------------------------------

class PermissionMode:
    AUTO_ACCEPT = "auto-accept"   # all tools run without asking
    EDIT_CONFIRM = "edit-confirm" # write/edit/bash need confirmation
    PLAN_ONLY = "plan-only"       # only plan, don't execute

CURRENT_MODE = PermissionMode.AUTO_ACCEPT  # default: auto-accept
ACTIVE_PLAN_FILE = None                     # path to current plan .md

# ---------------------------------------------------------------------------
# CLAW.md hook -- project-level instructions
# ---------------------------------------------------------------------------

def load_claw_md():
    """Load CLAW.md from CWD and parent dirs (like Claude Code's CLAUDE.md)."""
    instructions = []
    seen = set()
    # check CWD and parents up to 3 levels
    check_dir = Path(CWD)
    for _ in range(4):
        for name in ("CLAW.md", "claw.md", ".claw.md"):
            md_path = check_dir / name
            if md_path.is_file():
                # dedupe (Windows is case-insensitive)
                resolved = str(md_path.resolve()).lower()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    content = md_path.read_text(encoding="utf-8", errors="replace")
                    instructions.append((str(md_path), content))
                except Exception:
                    pass
        parent = check_dir.parent
        if parent == check_dir:
            break
        check_dir = parent
    return instructions

def load_memories_for_context():
    """Load hot-tier memories (+ pinned) for system prompt injection."""
    _ensure_memory_dir()
    entries = _load_all_memories()
    if not entries:
        return ""

    # Filter: hot tier or pinned
    hot_entries = [e for e in entries if e.get("tier") == "hot" or e.get("pinned", False)]
    if not hot_entries:
        return ""

    # Sort by weighted score: access_count / sqrt(days_ago)
    now = datetime.datetime.now()
    def _score(e):
        try:
            last = datetime.datetime.fromisoformat(e.get("last_accessed", e.get("saved_at", now.isoformat())))
        except (ValueError, TypeError):
            last = now
        days_ago = max((now - last).days, 1)
        return e.get("access_count", 0) / (days_ago ** 0.5)

    hot_entries.sort(key=_score, reverse=True)
    hot_entries = hot_entries[:10]  # cap at 10

    lines = []
    for e in hot_entries:
        pin = " [pinned]" if e.get("pinned") else ""
        lines.append(f"[{e.get('category','?')}] {e.get('key','?')}: {e.get('value','')}{pin}")

    return "\n".join(lines)

def load_active_plan():
    """Load the active plan file if one exists."""
    global ACTIVE_PLAN_FILE
    # check for plan files in CWD
    for name in ("PLAN.md", "plan.md", ".plan.md"):
        plan_path = Path(CWD) / name
        if plan_path.is_file():
            ACTIVE_PLAN_FILE = str(plan_path)
            try:
                return plan_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
    return ""


def _extract_conversation_context_for_plan(messages):
    """
    Extract user and assistant messages from the conversation to build plan context.
    Skips system messages and tool results. Returns a compressed summary string,
    or empty string if no meaningful conversation has happened.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or role in ("system", "tool"):
            continue
        # Skip internal system injections
        if content.startswith("[SYSTEM:"):
            continue
        if role == "user":
            # Trim very long user messages (e.g., file contents)
            text = content[:500] if len(content) > 500 else content
            parts.append(f"User: {text}")
        elif role == "assistant":
            # Only keep substantive assistant responses (skip tool-call-only turns)
            text = content.strip()
            if text and len(text) > 10:
                parts.append(f"Assistant: {text[:300]}")
    if not parts:
        return ""
    # Cap total context to ~3000 chars to keep the plan prompt focused
    combined = "\n\n".join(parts)
    return combined[:3000]

# ---------------------------------------------------------------------------
# edit confirmation (for edit-confirm mode)
# ---------------------------------------------------------------------------

def confirm_tool_execution(tool_name, tool_args):
    """Ask user to confirm a tool execution. Returns True if approved."""
    global CURRENT_MODE
    if CURRENT_MODE == PermissionMode.AUTO_ACCEPT:
        return True

    # read-only tools never need confirmation
    safe_tools = {"read_file", "glob_search", "grep_search", "memory_search", "memory_save", "ask_user", "db_schema", "env_manage"}
    if tool_name in safe_tools:
        return True

    # show what's about to happen
    print(f"\n  {C.WARNING}{BLACK_CIRCLE} {C.BOLD}Confirm action{C.RESET}")
    print(f"  {C.TEXT}Tool: {tool_name}{C.RESET}")
    if tool_name == "bash":
        print(f"  {C.TEXT}Command: {tool_args.get('command', '')}{C.RESET}")
    elif tool_name in ("write_file", "edit_file"):
        print(f"  {C.TEXT}File: {tool_args.get('file_path', '')}{C.RESET}")

    sys.stdout.write(f"  {C.CLAW}Allow? (y/n/always)>{C.RESET} ")
    sys.stdout.flush()
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if answer in ("a", "always"):
        CURRENT_MODE = PermissionMode.AUTO_ACCEPT
        print(f"  {C.SUCCESS}{BLACK_CIRCLE} Switched to auto-accept mode{C.RESET}")
        return True
    return answer in ("y", "yes", "")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
PDF_EXTS = {".pdf"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
BINARY_EXTS = {".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".bin", ".whl",
               ".pyc", ".o", ".a", ".jar", ".class"}

# ---------------------------------------------------------------------------
# colors
# ---------------------------------------------------------------------------

def _rgb(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def _rgb_bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"

class C:
    RESET      = "\033[0m"
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    ITALIC     = "\033[3m"
    UNDERLINE  = "\033[4m"

    # -- claw brand palette --
    CLAW       = _rgb(87, 199, 170)      # teal-green (primary brand)
    CLAW_DIM   = _rgb(60, 140, 120)      # muted teal
    TEXT       = _rgb(220, 220, 220)      # light gray text
    SUBTLE     = _rgb(130, 130, 130)      # secondary text
    SUCCESS    = _rgb(44, 190, 90)        # green
    ERROR      = _rgb(220, 60, 80)        # red
    WARNING    = _rgb(210, 170, 50)       # amber
    TOOL       = _rgb(87, 145, 247)       # blue (tool calls)
    TOOL_BG    = _rgb(30, 40, 60)         # dark blue bg for tool labels
    PERMISSION = _rgb(87, 105, 247)       # medium blue
    DIFF_ADD   = _rgb(105, 219, 124)      # green for additions
    DIFF_REM   = _rgb(255, 168, 180)      # red for removals
    THINK      = _rgb(180, 130, 220)      # soft purple for inner monologue
    THINK_DIM  = _rgb(120, 90, 160)       # muted purple for borders

    # -- fallback ansi --
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[37m"

# -- display figures (matching professional CLI conventions) --
BLACK_CIRCLE = "●"
OPEN_CIRCLE  = "○"
HALF_CIRCLE  = "~"
BLOCKQUOTE   = "|"
DASH         = "-"

def cprint(color, *args, **kw):
    print(f"{color}{''.join(str(a) for a in args)}{C.RESET}", **kw)


# ---------------------------------------------------------------------------
# UI components: progress bar, diff, file tree, token budget, markdown, bell
# ---------------------------------------------------------------------------

def _progress_bar(current, total, width=30, label="", show_percent=True):
    """Render a colored progress bar string. Does NOT print — returns string."""
    if total <= 0:
        frac = 1.0
    else:
        frac = min(current / total, 1.0)
    filled = int(width * frac)
    empty = width - filled

    # Gradient from teal to green as progress increases
    r, g, b = _lerp_color(frac, 87, 199, 170, 44, 190, 90)
    bar_color = f"\033[38;2;{r};{g};{b}m"

    bar = f"{bar_color}{'█' * filled}{C.SUBTLE}{'░' * empty}{C.RESET}"
    pct = f" {C.TEXT}{int(frac * 100)}%{C.RESET}" if show_percent else ""
    count = f" {C.SUBTLE}({current}/{total}){C.RESET}" if total > 0 else ""
    lbl = f"{C.DIM}{label} {C.RESET}" if label else ""
    return f"  {lbl}{bar}{pct}{count}"


def _print_progress(current, total, width=30, label=""):
    """Print a progress bar, overwriting the current line."""
    bar = _progress_bar(current, total, width, label)
    sys.stdout.write(f"\r{bar}  ")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _diff_display(old_text, new_text, filepath="", context_lines=3):
    """Print a colored inline diff of old vs new text."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    import difflib
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filepath}" if filepath else "before",
        tofile=f"b/{filepath}" if filepath else "after",
        n=context_lines,
    )

    output_lines = []
    for line in diff:
        line = line.rstrip("\n")
        if line.startswith("+++") or line.startswith("---"):
            output_lines.append(f"  {C.BOLD}{C.SUBTLE}{line}{C.RESET}")
        elif line.startswith("@@"):
            output_lines.append(f"  {C.TOOL}{line}{C.RESET}")
        elif line.startswith("+"):
            output_lines.append(f"  {C.DIFF_ADD}{line}{C.RESET}")
        elif line.startswith("-"):
            output_lines.append(f"  {C.DIFF_REM}{line}{C.RESET}")
        else:
            output_lines.append(f"  {C.DIM}{line}{C.RESET}")

    if output_lines:
        # Show at most 30 diff lines to keep it concise
        for line in output_lines[:30]:
            print(line)
        if len(output_lines) > 30:
            print(f"  {C.SUBTLE}... {len(output_lines) - 30} more lines{C.RESET}")
    return len(output_lines)


def _file_tree(directory, max_depth=3, max_files=30):
    """Print a file tree with sizes. Skips node_modules, .git, __pycache__, venv."""
    skip = {"node_modules", ".git", ".next", "__pycache__", "venv", ".venv",
            "dist", "build", "env", ".env", ".cache", "coverage"}
    pdir = Path(directory)
    if not pdir.is_dir():
        return

    printed = [0]

    def _walk(current, prefix="", depth=0):
        if printed[0] >= max_files or depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        entries = [e for e in entries if e.name not in skip]
        for i, entry in enumerate(entries):
            if printed[0] >= max_files:
                break
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if entry.is_dir():
                sub_count = 0
                try:
                    sub_count = sum(1 for _ in entry.iterdir())
                except PermissionError:
                    pass
                print(f"  {C.SUBTLE}{prefix}{connector}{C.RESET}{C.TOOL}{entry.name}/{C.RESET} {C.SUBTLE}({sub_count}){C.RESET}")
                printed[0] += 1
                _walk(entry, prefix + extension, depth + 1)
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                print(f"  {C.SUBTLE}{prefix}{connector}{C.RESET}{C.TEXT}{entry.name}{C.RESET} {C.SUBTLE}({size_str}){C.RESET}")
                printed[0] += 1

    print(f"  {C.TOOL}{pdir.name}/{C.RESET}")
    _walk(pdir)
    if printed[0] >= max_files:
        print(f"  {C.SUBTLE}... (truncated at {max_files} entries){C.RESET}")


def _token_budget_bar(used, total=8192, label="Context"):
    """Print a token budget bar showing how much context is consumed."""
    frac = min(used / total, 1.0)
    width = 20
    filled = int(width * frac)
    empty = width - filled

    # Color: green when low, amber when mid, red when almost full
    if frac < 0.5:
        r, g, b = _lerp_color(frac * 2, 44, 190, 90, 210, 170, 50)
    else:
        r, g, b = _lerp_color((frac - 0.5) * 2, 210, 170, 50, 220, 60, 80)
    bar_color = f"\033[38;2;{r};{g};{b}m"

    bar = f"{bar_color}{'█' * filled}{C.SUBTLE}{'░' * empty}{C.RESET}"
    print(f"  {C.DIM}{label}:{C.RESET} [{bar}] {bar_color}{used:,}{C.RESET}{C.SUBTLE}/{total:,} tokens{C.RESET}")


def _render_markdown(text):
    """Render markdown-ish text with ANSI formatting for terminal display."""
    lines = text.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                result.append(f"  {C.SUBTLE}{'─' * 40}{C.RESET}")
            else:
                result.append(f"  {C.SUBTLE}{'─' * 40}{C.RESET}")
            continue

        if in_code_block:
            result.append(f"  {C.DIM}  {line}{C.RESET}")
            continue

        # Headers
        if line.startswith("### "):
            result.append(f"  {C.CLAW}{C.BOLD}{line[4:]}{C.RESET}")
            continue
        if line.startswith("## "):
            result.append(f"  {C.CLAW}{C.BOLD}{C.UNDERLINE}{line[3:]}{C.RESET}")
            continue
        if line.startswith("# "):
            result.append(f"  {C.CLAW}{C.BOLD}{C.UNDERLINE}{line[2:]}{C.RESET}")
            continue

        # Bold: **text**
        line = re.sub(r'\*\*(.+?)\*\*', f'{C.BOLD}\\1{C.RESET}{C.TEXT}', line)
        # Italic: *text* (but not inside **)
        line = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', f'{C.ITALIC}\\1{C.RESET}{C.TEXT}', line)
        # Inline code: `text`
        line = re.sub(r'`([^`]+?)`', f'{C.DIM}{C.CLAW_DIM}\\1{C.RESET}{C.TEXT}', line)
        # Bullet points
        if line.strip().startswith("- "):
            indent = len(line) - len(line.lstrip())
            content = line.strip()[2:]
            result.append(f"  {' ' * indent}{C.CLAW}•{C.RESET} {C.TEXT}{content}{C.RESET}")
            continue
        # Numbered lists
        m = re.match(r'^(\s*)\d+\.\s+(.+)', line)
        if m:
            indent = len(m.group(1))
            content = m.group(2)
            result.append(f"  {' ' * indent}{C.CLAW}›{C.RESET} {C.TEXT}{content}{C.RESET}")
            continue

        # Regular text
        if line.strip():
            result.append(f"  {C.TEXT}{line}{C.RESET}")
        else:
            result.append("")

    return "\n".join(result)


def _bell():
    """Ring terminal bell to notify user of completion."""
    sys.stdout.write("\a")
    sys.stdout.flush()


def _status_bar(model="", turn=0, tokens=0, mode="", turn_tokens=None):
    """Print a persistent status bar at the bottom of output."""
    try:
        w = min(os.get_terminal_size().columns, 120)
    except (OSError, ValueError):
        w = 80

    parts = []
    if model:
        parts.append(f"{C.CLAW}⚡{C.RESET} {C.DIM}{model}{C.RESET}")
    if turn > 0:
        parts.append(f"{C.SUBTLE}turn {turn}{C.RESET}")
    if tokens > 0:
        parts.append(f"{C.SUBTLE}{_format_tokens(tokens)} tok{C.RESET}")
    if turn_tokens:
        up_tok, down_tok = turn_tokens
        parts.append(f"{C.SUBTLE}↑{_format_tokens(up_tok)} ↓{_format_tokens(down_tok)}{C.RESET}")
    if mode:
        parts.append(f"{C.SUBTLE}{mode}{C.RESET}")

    bar_content = f" {C.DIM}·{C.RESET} ".join(parts)

    # Dim separator line
    print(f"  {C.SUBTLE}{'─' * (w - 4)}{C.RESET}")
    print(f"  {bar_content}")


# ---------------------------------------------------------------------------
# context compression -- keep conversation within 8K window
# ---------------------------------------------------------------------------

def _estimate_tokens(messages):
    """Rough token estimate: ~1.3 words per token for English text."""
    return int(sum(len(m.get("content", "").split()) for m in messages) * 1.3)


def _compress_messages(messages, max_tokens=None):
    """
    Compress conversation to fit within the model's context window.
    Keeps: system prompt + last 4 messages intact.
    Compresses: old tool results → one-line summaries, old assistant text → truncated.
    Drops: oldest messages first if still over budget.
    """
    if max_tokens is None:
        max_tokens = 5500  # fallback default
    if not messages:
        return messages
    if _estimate_tokens(messages) <= max_tokens:
        return messages  # fits, no compression needed

    # Always keep system prompt (first message if role=system)
    system = []
    rest = messages[:]
    if rest and rest[0].get("role") == "system":
        system = [rest[0]]
        rest = rest[1:]

    # Always keep last 4 messages (last 2 turns) intact
    tail_size = min(4, len(rest))
    keep_tail = rest[-tail_size:] if tail_size > 0 else []
    middle = rest[:-tail_size] if tail_size > 0 and len(rest) > tail_size else []

    # Compress middle messages
    compressed = []
    for msg in middle:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Drop ephemeral reflection prompts entirely
        if role == "user" and isinstance(content, str) and content.startswith("[SYSTEM: Pause and reflect"):
            continue
        if role == "tool":
            # Tool results → one-line summary
            summary = content[:80].replace("\n", " ").strip()
            if len(content) > 80:
                summary += "..."
            compressed.append({"role": "tool", "content": f"[prior: {summary}]"})
        elif role == "assistant":
            # Assistant text → truncate to key info
            if len(content) > 200:
                compressed.append({"role": "assistant", "content": content[:200] + "..."})
            else:
                compressed.append(msg)
        else:
            # User messages → keep but truncate if huge
            if len(content) > 300:
                compressed.append({"role": role, "content": content[:300] + "..."})
            else:
                compressed.append(msg)

    result = system + compressed + keep_tail

    # If still too big, drop oldest compressed messages one at a time
    while _estimate_tokens(result) > max_tokens and compressed:
        compressed.pop(0)
        result = system + compressed + keep_tail

    # Last resort: if still too big, only keep system + tail
    if _estimate_tokens(result) > max_tokens:
        result = system + keep_tail

    return result


# ---------------------------------------------------------------------------
# spinner animation
# ---------------------------------------------------------------------------

import random

SPINNER_GLYPHS = ["*", "o", "O", "@", "*", "o", "O", "@"]

SPINNER_VERBS = [
    "Thinking", "Analyzing", "Processing", "Computing", "Evaluating",
    "Reasoning", "Considering", "Examining", "Working", "Reflecting",
    "Contemplating", "Synthesizing", "Formulating", "Constructing",
]

class Spinner:
    """Animated spinner that runs in a background thread."""

    def __init__(self, label=None, color=C.CLAW):
        self.label = label or random.choice(SPINNER_VERBS)
        self.color = color
        self._stop = threading.Event()
        self._thread = None

    def _animate(self):
        i = 0
        while not self._stop.is_set():
            glyph = SPINNER_GLYPHS[i % len(SPINNER_GLYPHS)]
            text = f"\r  {self.color}{glyph}{C.RESET} {C.DIM}{C.ITALIC}{self.label}...{C.RESET}"
            sys.stdout.write(text + " " * 10)
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.10)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_msg=None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
        if final_msg:
            print(final_msg)

    def update(self, label):
        self.label = label


# ---------------------------------------------------------------------------
# interactive questions (ask_user tool)
# ---------------------------------------------------------------------------

_pending_question = {"active": False, "answer": None}

def tool_ask_user(args):
    """Ask the user a question, optionally with selectable choices."""
    question = args.get("question", "")
    choices = args.get("choices", [])
    default = args.get("default", "")

    if not question:
        return "Error: no question provided"

    # In one-shot mode, there's no interactive user — auto-proceed
    if _ONE_SHOT_MODE:
        print(f"\n  {C.CLAW}{BLACK_CIRCLE} {C.BOLD}Question{C.RESET}")
        print(f"  {C.TEXT}{question}{C.RESET}")
        print(f"  {C.DIM}-> (one-shot mode: auto-proceeding){C.RESET}\n")
        return ("User is not available (one-shot non-interactive mode). "
                "DO NOT ask more questions. Make your best judgment call and proceed immediately. "
                "Skip all remaining discovery steps and BUILD NOW. "
                "Use write_file and bash tools to create files and install dependencies.")

    print(f"\n  {C.CLAW}{BLACK_CIRCLE} {C.BOLD}Question{C.RESET}")
    print(f"  {C.TEXT}{question}{C.RESET}")

    if choices:
        print()
        for i, choice in enumerate(choices, 1):
            print(f"    {C.CLAW}{i}.{C.RESET} {C.TEXT}{choice}{C.RESET}")
        print()
        hint = f"1-{len(choices)}"
        if default:
            hint += f", default: {default}"
        sys.stdout.write(f"  {C.CLAW}choose ({hint})>{C.RESET} ")
        sys.stdout.flush()
        try:
            answer = input().strip()
            if not answer and default:
                answer = default
            # if numeric, map to choice text
            try:
                idx = int(answer) - 1
                if 0 <= idx < len(choices):
                    answer = choices[idx]
            except ValueError:
                pass
        except (EOFError, KeyboardInterrupt):
            answer = default or "(no answer)"
    else:
        if default:
            sys.stdout.write(f"  {C.CLAW}answer (default: {default})>{C.RESET} ")
        else:
            sys.stdout.write(f"  {C.CLAW}answer>{C.RESET} ")
        sys.stdout.flush()
        try:
            answer = input().strip()
            if not answer and default:
                answer = default
        except (EOFError, KeyboardInterrupt):
            answer = default or "(no answer)"

    print(f"  {C.DIM}-> {answer}{C.RESET}\n")
    return f"User answered: {answer}"


# ---------------------------------------------------------------------------
# persistent memory system
# ---------------------------------------------------------------------------

MEMORY_DIR = Path.home() / ".claw" / "memory"

_STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "in",
               "of", "and", "for", "it", "on", "at", "by", "or", "be",
               "do", "no", "not", "but", "with", "this", "that", "from",
               "has", "have", "had", "its", "my", "your", "we", "they",
               "he", "she", "me", "you", "can", "will", "just", "so"}

_memory_cache = {"entries": None, "mtime": 0}
_last_maintenance = 0

def _ensure_memory_dir():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

def _extract_keywords(key, value):
    """Extract keywords from key and value for warm matching."""
    text = f"{key} {value}".lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique[:20]

def _migrate_memory_entry(entry):
    """Lazy migrate v1 entry to v2 schema. Returns migrated entry (not written to disk)."""
    if entry.get("schema_version") == 2:
        return entry
    now = datetime.datetime.now().isoformat()
    saved_at = entry.get("saved_at", now)
    entry["schema_version"] = 2
    entry.setdefault("tier", "warm")
    entry.setdefault("pinned", False)
    entry.setdefault("last_accessed", saved_at)
    entry.setdefault("access_count", 0)
    entry.setdefault("demotion_strikes", 0)
    entry.setdefault("saved_at", now)
    if "related_keywords" not in entry:
        entry["related_keywords"] = _extract_keywords(entry.get("key", ""), entry.get("value", ""))
    return entry

def _save_memory_entry(entry, filepath=None):
    """Atomic write: write to .tmp then os.replace()."""
    if filepath is None:
        cat_dir = MEMORY_DIR / entry.get("category", "general")
        cat_dir.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r'[^\w\-]', '_', entry.get("key", ""))[:60]
        filepath = cat_dir / f"{safe_key}.json"
    save_data = {k: v for k, v in entry.items() if not k.startswith("_")}
    tmp = filepath.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(filepath))
    _memory_cache["entries"] = None  # invalidate cache

def _load_all_memories():
    """Load all memory entries with per-file mtime caching."""
    _ensure_memory_dir()

    all_json = []
    for d in MEMORY_DIR.iterdir():
        if d.is_dir():
            all_json.extend(d.glob("*.json"))

    if not all_json:
        _memory_cache["entries"] = []
        _memory_cache["mtime"] = 0
        return []

    current_mtime = max(f.stat().st_mtime for f in all_json)

    if _memory_cache["entries"] is not None and current_mtime <= _memory_cache["mtime"]:
        return _memory_cache["entries"]

    entries = []
    for f in all_json:
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
            entry = _migrate_memory_entry(entry)
            entry["_filepath"] = str(f)
            entries.append(entry)
        except Exception:
            pass

    _memory_cache["entries"] = entries
    _memory_cache["mtime"] = current_mtime
    return entries

def _classify_tier(entry):
    """Determine what tier an entry should be in based on access patterns."""
    now = datetime.datetime.now()
    try:
        last = datetime.datetime.fromisoformat(entry.get("last_accessed", entry.get("saved_at", now.isoformat())))
    except (ValueError, TypeError):
        last = now
    days_since = (now - last).days
    access_count = entry.get("access_count", 0)

    if days_since <= 3:
        return "hot"
    if access_count >= 3 and days_since <= 14:
        return "hot"
    if days_since <= 30:
        return "warm"
    return "cold"

def _run_memory_maintenance():
    """Reclassify tiers with 2-strike demotion rule."""
    entries = _load_all_memories()
    tier_rank = {"hot": 2, "warm": 1, "cold": 0}
    for entry in entries:
        if entry.get("pinned", False):
            entry["demotion_strikes"] = 0
            continue

        ideal_tier = _classify_tier(entry)
        current_tier = entry.get("tier", "warm")

        # Promotion: immediate
        if tier_rank.get(ideal_tier, 0) > tier_rank.get(current_tier, 0):
            entry["tier"] = ideal_tier
            entry["demotion_strikes"] = 0
            _save_memory_entry(entry, Path(entry["_filepath"]))
            continue

        # Demotion: requires 2 consecutive strikes
        if tier_rank.get(ideal_tier, 0) < tier_rank.get(current_tier, 0):
            entry["demotion_strikes"] = entry.get("demotion_strikes", 0) + 1
            if entry["demotion_strikes"] >= 2:
                if current_tier == "hot":
                    entry["tier"] = "warm"
                elif current_tier == "warm":
                    entry["tier"] = "cold"
                entry["demotion_strikes"] = 0
            _save_memory_entry(entry, Path(entry["_filepath"]))
        else:
            # At correct tier, reset strikes
            if entry.get("demotion_strikes", 0) > 0:
                entry["demotion_strikes"] = 0
                _save_memory_entry(entry, Path(entry["_filepath"]))

def _maybe_run_maintenance():
    """Run maintenance at most every 5 minutes."""
    global _last_maintenance
    if time.time() - _last_maintenance > 300:
        _run_memory_maintenance()
        _last_maintenance = time.time()

def _auto_search_warm_memories(msg):
    """Auto-search warm/cold memories by keyword overlap with user message."""
    msg_keywords = set(_extract_keywords("", msg))
    if not msg_keywords:
        return ""

    long_keywords = [k for k in msg_keywords if len(k) >= 4]
    if not long_keywords:
        return ""

    entries = _load_all_memories()
    scored = []
    for entry in entries:
        if entry.get("tier") == "hot" or entry.get("pinned", False):
            continue  # hot/pinned already injected in system prompt
        entry_kw = set(entry.get("related_keywords", []))
        overlap = msg_keywords & entry_kw
        if len(overlap) < 2:
            continue
        if not any(len(k) >= 4 for k in overlap):
            continue
        scored.append((len(overlap), entry))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    now = datetime.datetime.now().isoformat()
    lines = []
    for _score, entry in top:
        tier_tag = entry.get("tier", "w")[0].upper()
        lines.append(f"[{entry.get('category','?')}/{tier_tag}] {entry.get('key','?')}: {entry.get('value','')}")
        # Bump access on warm auto-search hit
        entry["access_count"] = entry.get("access_count", 0) + 1
        entry["last_accessed"] = now
        new_tier = _classify_tier(entry)
        if new_tier != entry.get("tier"):
            entry["tier"] = new_tier
            entry["demotion_strikes"] = 0
        if "_filepath" in entry:
            _save_memory_entry(entry, Path(entry["_filepath"]))

    return "\n".join(lines)

def tool_memory_save(args):
    """Save something to persistent memory (v2 schema, tier=warm)."""
    key = args.get("key", "").strip()
    value = args.get("value", "").strip()
    category = args.get("category", "general").strip()

    if not key or not value:
        return "Error: key and value are required"

    _ensure_memory_dir()
    cat_dir = MEMORY_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    safe_key = re.sub(r'[^\w\-]', '_', key)[:60]
    filepath = cat_dir / f"{safe_key}.json"

    now = datetime.datetime.now().isoformat()

    # Preserve access stats on overwrite
    old_access_count = 0
    old_last_accessed = now
    old_tier = "warm"
    old_pinned = False
    old_demotion_strikes = 0
    if filepath.exists():
        try:
            old_entry = json.loads(filepath.read_text(encoding="utf-8"))
            old_entry = _migrate_memory_entry(old_entry)
            old_access_count = old_entry.get("access_count", 0)
            old_last_accessed = old_entry.get("last_accessed", now)
            old_tier = old_entry.get("tier", "warm")
            old_pinned = old_entry.get("pinned", False)
            old_demotion_strikes = old_entry.get("demotion_strikes", 0)
        except Exception:
            pass

    entry = {
        "schema_version": 2,
        "key": key,
        "value": value,
        "category": category,
        "tier": old_tier,
        "pinned": old_pinned,
        "saved_at": now,
        "last_accessed": old_last_accessed,
        "access_count": old_access_count,
        "related_keywords": _extract_keywords(key, value),
        "demotion_strikes": old_demotion_strikes,
    }
    _save_memory_entry(entry, filepath)
    _maybe_run_maintenance()
    return f"Saved to memory: [{category}] {key}"

def tool_memory_search(args):
    """Search persistent memory across all tiers."""
    query = args.get("query", "").strip().lower()
    category = args.get("category", "").strip()

    _ensure_memory_dir()
    entries = _load_all_memories()

    if category:
        entries = [e for e in entries if e.get("category", "") == category]

    results = []
    query_keywords = set(_extract_keywords("", query)) if query else set()

    for entry in entries:
        if not query:
            results.append((0, entry))
            continue
        # Match against key, value, and related_keywords
        key_lower = entry.get("key", "").lower()
        val_lower = entry.get("value", "").lower()
        if query in key_lower or query in val_lower:
            results.append((10, entry))
            continue
        entry_kw = set(entry.get("related_keywords", []))
        overlap = query_keywords & entry_kw
        if overlap:
            results.append((len(overlap), entry))

    if not results:
        return "No memories found."

    results.sort(key=lambda x: x[0], reverse=True)

    now = datetime.datetime.now().isoformat()
    lines = []
    for _score, r in results[:20]:
        tier_tag = r.get("tier", "w")[0].upper()
        lines.append(f"[{r.get('category','?')}/{tier_tag}] {r.get('key','?')}: {r.get('value','')[:200]}")
        # Bump access stats on search hit
        r["access_count"] = r.get("access_count", 0) + 1
        r["last_accessed"] = now
        new_tier = _classify_tier(r)
        if new_tier != r.get("tier"):
            r["tier"] = new_tier
            r["demotion_strikes"] = 0
        if "_filepath" in r:
            _save_memory_entry(r, Path(r["_filepath"]))

    return "\n".join(lines)

def tool_memory_delete(args):
    """Delete a memory entry by key."""
    key = args.get("key", "").strip()
    category = args.get("category", "").strip()

    if not key:
        return "Error: key is required"

    _ensure_memory_dir()
    safe_key = re.sub(r'[^\w\-]', '_', key)[:60]

    deleted = False
    search_dirs = []
    if category:
        cat_dir = MEMORY_DIR / category
        if cat_dir.exists():
            search_dirs = [cat_dir]
    else:
        search_dirs = [d for d in MEMORY_DIR.iterdir() if d.is_dir()]

    for d in search_dirs:
        filepath = d / f"{safe_key}.json"
        if filepath.exists():
            filepath.unlink()
            deleted = True

    if deleted:
        _memory_cache["entries"] = None  # invalidate cache
    return f"Deleted memory: {key}" if deleted else f"Memory not found: {key}"


# ---------------------------------------------------------------------------
# sub-agents -- parallel execution with tree display
# ---------------------------------------------------------------------------

# tree drawing chars
T_PIPE  = "|"
T_TEE   = "|-"
T_ELBOW = "`-"
T_HOOK  = "  `-"
T_SPACE = "  "

class SubAgentResult:
    """Result from a single sub-agent run."""
    def __init__(self, name):
        self.name = name
        self.status = "queued"      # queued -> running -> done / error
        self.tool_uses = 0
        self.tokens = 0
        self.elapsed = 0.0
        self.output = ""
        self.error = None

def _format_time(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m}m {s}s"

def _format_tokens(n):
    if n < 1000:
        return f"{n}"
    return f"{n/1000:.1f}k"

def _ollama_chat_sync(messages, model, tools=None, num_ctx=8192):
    """Non-streaming Ollama call for sub-agents (thread-safe)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Estimate prompt tokens
    _prompt_est = sum(len(m.get("content", "").split()) for m in messages)

    # Retry with exponential backoff (3 attempts)
    last_err = None
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=OLLAMA_SYNC_TIMEOUT)
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            # Track tokens globally
            comp = data.get("eval_count", len(data.get("message", {}).get("content", "").split()))
            _token_tracker.add(prompt=_prompt_est, completion=comp)
            return data
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s
    raise ConnectionError(f"Ollama unreachable after 3 attempts: {last_err}")

def _run_subagent_worker(task_prompt, model, result_obj, system_prompt, drip_context=None):
    """Worker function for a single sub-agent thread."""
    _thread_local.is_subagent = True
    result_obj.status = "running"
    start = time.time()

    # If drip context is available, inject compressed context into the prompt
    if drip_context is not None:
        compressed = drip_context.compress()
        if compressed:
            task_prompt = f"{task_prompt}\n\n## Shared Context (from other agents)\n{compressed}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    try:
        iterations = 0
        while iterations < 12:
            iterations += 1
            ctx_msgs = _compress_messages(messages, max_tokens=5500)
            resp = _ollama_chat_sync(ctx_msgs, model, tools=TOOL_DEFS)
            msg = resp.get("message", {})
            content = msg.get("content", "")

            # count tokens from eval_count if available
            result_obj.tokens += resp.get("eval_count", len(content.split()))

            tool_calls = msg.get("tool_calls", [])

            # rescue tool calls from text
            if not tool_calls and content:
                _, rescued = rescue_tool_calls_from_text(content)
                if rescued:
                    tool_calls = rescued

            if not tool_calls:
                result_obj.output = content
                break

            # execute tools
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                result_obj.tool_uses += 1
                tool_result = execute_tool(name, args)
                messages.append({"role": "tool", "content": tool_result})

        result_obj.status = "done"
    except Exception as e:
        result_obj.status = "error"
        result_obj.error = str(e)
    finally:
        _thread_local.is_subagent = False
        result_obj.elapsed = time.time() - start


def _draw_agent_tree(results, show_spinner_frame=0):
    """Draw the tree display for sub-agents. Returns list of lines."""
    lines = []
    n = len(results)
    for i, r in enumerate(results):
        is_last = (i == n - 1)
        branch = T_ELBOW if is_last else T_TEE
        pipe   = T_SPACE if is_last else f"{T_PIPE} "

        # status icon
        if r.status == "done":
            icon = f"{C.SUCCESS}{BLACK_CIRCLE}{C.RESET}"
        elif r.status == "error":
            icon = f"{C.ERROR}{BLACK_CIRCLE}{C.RESET}"
        elif r.status == "running":
            glyph = SPINNER_GLYPHS[show_spinner_frame % len(SPINNER_GLYPHS)]
            icon = f"{C.CLAW}{glyph}{C.RESET}"
        else:  # queued
            icon = f"{C.SUBTLE}{OPEN_CIRCLE}{C.RESET}"

        # stats
        stats = ""
        if r.tool_uses > 0:
            stats += f" {C.SUBTLE}{r.tool_uses} tool uses{C.RESET}"
        if r.tokens > 0:
            stats += f" {C.SUBTLE}{_format_tokens(r.tokens)} tokens{C.RESET}"
        if r.elapsed > 0 and r.status in ("done", "error"):
            stats += f" {C.SUBTLE}{_format_time(r.elapsed)}{C.RESET}"

        # name
        name_color = C.TEXT if r.status in ("done", "error") else C.SUBTLE
        lines.append(f"   {C.SUBTLE}{branch}{C.RESET} {icon} {name_color}{r.name}{C.RESET}{stats}")

        # result hook
        if r.status == "done" and r.output:
            preview = r.output[:120].replace("\n", " ").strip()
            if len(r.output) > 120:
                preview += "..."
            lines.append(f"   {C.SUBTLE}{pipe} {T_HOOK}{C.RESET} {C.DIM}{preview}{C.RESET}")
        elif r.status == "error" and r.error:
            lines.append(f"   {C.SUBTLE}{pipe} {T_HOOK}{C.RESET} {C.ERROR}{r.error[:100]}{C.RESET}")

    return lines


def run_subagents(task_descriptions, model, system_prompt):
    """
    Run multiple sub-agents in parallel with live tree display.

    task_descriptions: list of (name, prompt) tuples
    Returns: list of SubAgentResult
    """
    results = [SubAgentResult(name) for name, _ in task_descriptions]
    threads = []

    # header
    n = len(results)
    print(f"\n  {C.TOOL}{BLACK_CIRCLE} {C.BOLD}Running {n} agents...{C.RESET}")

    # launch all threads
    for i, (name, prompt) in enumerate(task_descriptions):
        t = threading.Thread(
            target=_run_subagent_worker,
            args=(prompt, model, results[i], system_prompt),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # live display loop
    frame = 0
    prev_lines = 0
    start_time = time.time()

    while any(t.is_alive() for t in threads):
        # move cursor up to overwrite previous tree
        if prev_lines > 0:
            sys.stdout.write(f"\033[{prev_lines}A")

        tree_lines = _draw_agent_tree(results, frame)

        # timing footer
        elapsed = time.time() - start_time
        done_count = sum(1 for r in results if r.status in ("done", "error"))
        total_tokens = sum(r.tokens for r in results)
        footer = f"   {C.SUBTLE}{_format_time(elapsed)} elapsed | {done_count}/{n} complete | {_format_tokens(total_tokens)} tokens{C.RESET}"
        tree_lines.append(footer)

        for line in tree_lines:
            # clear the line first, then print
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()

        prev_lines = len(tree_lines)
        frame += 1
        time.sleep(0.15)

    # final draw
    if prev_lines > 0:
        sys.stdout.write(f"\033[{prev_lines}A")
    tree_lines = _draw_agent_tree(results, frame)
    elapsed = time.time() - start_time
    total_tokens = sum(r.tokens for r in results)
    total_tools = sum(r.tool_uses for r in results)
    footer = f"   {C.SUCCESS}{BLACK_CIRCLE}{C.RESET} {C.SUBTLE}All done | {_format_time(elapsed)} | {total_tools} tool uses | {_format_tokens(total_tokens)} tokens{C.RESET}"
    tree_lines.append(footer)

    for line in tree_lines:
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.flush()
    print()

    return results


def run_background_agent(name, prompt, model, system_prompt, callback=None):
    """
    Launch a single background agent that runs independently.
    Returns a SubAgentResult that updates in real-time.
    Optional callback(result) called when done.
    """
    result = SubAgentResult(name)

    def _worker():
        _run_subagent_worker(prompt, model, result, system_prompt)
        if callback:
            try:
                callback(result)
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return result, t


def decompose_plan_for_parallel(plan_content):
    """
    Analyze a plan and identify which steps can run in parallel.
    Returns list of step groups: [[step1, step2], [step3], [step4, step5]]
    where steps in the same group can run concurrently.
    """
    lines = plan_content.split("\n")
    unchecked = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            step_text = stripped.replace("- [ ]", "").strip()
            unchecked.append(step_text)

    if not unchecked:
        return []

    # group steps by dependency analysis
    # steps that create independent files/features can run in parallel
    # steps that say "after" or reference previous steps must be sequential
    groups = []
    current_group = []

    dependency_keywords = [
        "after", "then", "next", "update", "modify", "edit", "fix",
        "integrate", "connect", "wire", "link", "based on", "using the"
    ]

    for step in unchecked:
        step_lower = step.lower()
        has_dependency = any(kw in step_lower for kw in dependency_keywords)

        if has_dependency and current_group:
            # this step depends on previous — flush current group, start new sequential step
            groups.append(current_group)
            current_group = [step]
        else:
            # independent step — can run in parallel with current group
            current_group.append(step)

        # cap parallel group size at 4 (don't overwhelm the model)
        if len(current_group) >= 4:
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

    return groups


def run_plan_steps_parallel(steps, model, system_prompt, plan_content, manifest, drip_context=None):
    """
    Run a group of independent plan steps in parallel using sub-agents.
    Optionally accepts a DripContext for shared state between agents.
    Returns list of SubAgentResult.
    """
    tasks = []
    for step in steps:
        # build rich context for each agent
        api_context = get_api_context_for_prompt(step)
        manifest_summary = manifest.get_context_summary()

        step_prompt = (
            f"Execute this plan step: {step}\n\n"
            f"## Full Plan\n{plan_content}\n\n"
        )
        if manifest_summary:
            step_prompt += f"## Files created so far\n{manifest_summary}\n\n"
        if api_context:
            step_prompt += f"## API Reference\n{api_context}\n\n"
        # Inject drip context if available
        if drip_context is not None:
            compressed = drip_context.compress()
            if compressed:
                step_prompt += f"## Shared Drip Context\n{compressed}\n\n"
        step_prompt += (
            f"IMPORTANT:\n"
            f"- Focus ONLY on this specific step\n"
            f"- Write COMPLETE code — no TODOs, no placeholders\n"
            f"- Follow the design system — modern, clean, no slop\n"
            f"- After completing, update PLAN.md to mark this step done\n"
        )
        tasks.append((step[:60], step_prompt))

    if len(tasks) == 1:
        # single step — don't bother with sub-agents, just return the prompt
        return None, tasks[0][1]

    # run in parallel
    results = run_subagents(tasks, model, system_prompt)
    return results, None


# ---------------------------------------------------------------------------
# enhanced spinner with timing + token count
# ---------------------------------------------------------------------------

class TimedSpinner(Spinner):
    """Spinner that shows elapsed time and token count."""

    def __init__(self, label=None, color=C.CLAW):
        super().__init__(label, color)
        self._start_time = None
        self.tokens = 0

    def _animate(self):
        i = 0
        self._start_time = time.time()
        while not self._stop.is_set():
            glyph = SPINNER_GLYPHS[i % len(SPINNER_GLYPHS)]
            elapsed = time.time() - self._start_time
            time_str = _format_time(elapsed)
            tok_str = ""
            if self.tokens > 0:
                tok_str = f" | {_format_tokens(self.tokens)} tokens"
            text = (
                f"\r  {self.color}{glyph}{C.RESET} "
                f"{C.DIM}{C.ITALIC}{self.label}...{C.RESET} "
                f"{C.SUBTLE}({time_str}{tok_str}){C.RESET}"
            )
            sys.stdout.write(text + " " * 10)
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.10)

    def add_tokens(self, n):
        self.tokens += n


# ---------------------------------------------------------------------------
# file handling -- images, PDFs, text, video
# ---------------------------------------------------------------------------

def _resolve(p):
    """Resolve a path relative to CWD. Rejects paths outside the working directory."""
    raw = str(p).strip().strip('"').strip("'")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(CWD) / path
    # Security: verify resolved path stays inside CWD
    try:
        resolved = path.resolve()
        cwd_resolved = Path(CWD).resolve()
        # os.path.commonpath raises ValueError if paths are on different drives (Windows)
        if sys.platform == "win32":
            common = os.path.commonpath([str(resolved).lower(), str(cwd_resolved).lower()])
            if common.lower() != str(cwd_resolved).lower():
                raise ValueError(f"Path escapes working directory: {p}")
        else:
            common = os.path.commonpath([str(resolved), str(cwd_resolved)])
            if common != str(cwd_resolved):
                raise ValueError(f"Path escapes working directory: {p}")
    except ValueError:
        raise ValueError(f"Path escapes working directory: {p}")
    return path

def _canon(p):
    """Canonical string path for tracker lookups."""
    return str(_resolve(p).resolve())

def _file_size_str(size):
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.0f}KB"
    else:
        return f"{size / (1024*1024):.1f}MB"

def read_image_b64(path):
    """Read an image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def extract_pdf_text(path):
    """Extract text from a PDF. Tries pdftotext first, then PyPDF2."""
    # try pdftotext (poppler)
    try:
        r = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # try PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Page {i+1} ---\n{text}")
        if pages:
            return "\n\n".join(pages)
    except ImportError:
        pass
    except Exception:
        pass

    return None

def extract_video_info(path):
    """Get basic video metadata via ffprobe if available."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            info = json.loads(r.stdout)
            fmt = info.get("format", {})
            duration = float(fmt.get("duration", 0))
            mins = int(duration // 60)
            secs = int(duration % 60)
            streams = info.get("streams", [])
            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            res = ""
            if video_streams:
                v = video_streams[0]
                res = f"{v.get('width', '?')}x{v.get('height', '?')}"
            return f"Video: {mins}m{secs}s, {res}, {_file_size_str(os.path.getsize(path))}"
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return f"Video file: {_file_size_str(os.path.getsize(path))} (install ffprobe for metadata)"


class Attachment:
    """Represents a file attached to a message."""
    def __init__(self, path):
        self.path = _resolve(path)
        self.name = self.path.name
        self.ext = self.path.suffix.lower()
        self.size = self.path.stat().st_size if self.path.exists() else 0
        self.kind = self._detect_kind()
        self.content = None      # text content for PDFs/text files
        self.b64_image = None    # base64 for images
        self.error = None

        self._load()

    def _detect_kind(self):
        if self.ext in IMAGE_EXTS:
            return "image"
        if self.ext in PDF_EXTS:
            return "pdf"
        if self.ext in VIDEO_EXTS:
            return "video"
        if self.ext in BINARY_EXTS:
            return "binary"
        return "text"

    def _load(self):
        if not self.path.exists():
            self.error = f"File not found: {self.path}"
            return
        if not self.path.is_file():
            self.error = f"Not a file: {self.path}"
            return

        try:
            if self.kind == "image":
                self.b64_image = read_image_b64(self.path)

            elif self.kind == "pdf":
                text = extract_pdf_text(self.path)
                if text:
                    self.content = text
                else:
                    self.error = "Could not extract PDF text (install pdftotext or PyPDF2)"

            elif self.kind == "video":
                self.content = extract_video_info(self.path)

            elif self.kind == "binary":
                self.content = f"[Binary file: {self.name}, {_file_size_str(self.size)}]"

            else:  # text
                text = self.path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 50000:
                    text = text[:50000] + f"\n... [truncated, {len(text)} total chars]"
                self.content = text

        except Exception as e:
            self.error = str(e)

    def summary(self):
        size = _file_size_str(self.size)
        if self.error:
            return f"{C.RED}[!] {self.name} -- {self.error}{C.RESET}"
        if self.kind == "image":
            return f"{C.GREEN}[+] {self.name} (image, {size}){C.RESET}"
        if self.kind == "pdf":
            chars = len(self.content) if self.content else 0
            return f"{C.GREEN}[+] {self.name} (PDF, {size}, {chars} chars extracted){C.RESET}"
        if self.kind == "video":
            return f"{C.GREEN}[+] {self.name} (video, {size}){C.RESET}"
        if self.kind == "binary":
            return f"{C.YELLOW}[~] {self.name} (binary, {size}){C.RESET}"
        chars = len(self.content) if self.content else 0
        return f"{C.GREEN}[+] {self.name} ({size}, {chars} chars){C.RESET}"


# ---------------------------------------------------------------------------
# tool definitions (sent to the model)
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash/shell command and return stdout+stderr. Use for running programs, git, npm, pip, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk. Returns the file contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (absolute or relative to cwd)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based). Optional."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Optional."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with a new string. The old_string must appear exactly once in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text"
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob_search",
            "description": "Find files matching a glob pattern. Example: '**/*.py', 'src/**/*.ts'",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to cwd."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Defaults to cwd."
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob to filter files, e.g. '*.py'"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a question. Optionally provide a list of choices for them to select from. Use this when you need clarification or confirmation before proceeding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask"
                    },
                    "choices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of choices. User will pick one."
                    },
                    "default": {
                        "type": "string",
                        "description": "Default answer if user presses Enter"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Save information to persistent memory. Use this to remember user preferences, project context, decisions, or anything that should persist across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short label for this memory (e.g. 'preferred_framework', 'project_structure')"
                    },
                    "value": {
                        "type": "string",
                        "description": "The information to remember"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category: 'user', 'project', 'preference', 'context'. Default: 'general'"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search persistent memory for previously saved information. Use this at the start of conversations or when context seems relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term. Leave empty to list all memories."
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category. Optional."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "db_schema",
            "description": "Generate, view, or manage database schemas. Supports PostgreSQL, SQLite, Prisma, Drizzle, and Supabase. Use this when planning or building projects that need a database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'generate' (create schema from description), 'view' (show current schema files), 'migrate' (show migration commands), 'seed' (generate seed data)"
                    },
                    "orm": {
                        "type": "string",
                        "description": "ORM/database: 'prisma', 'drizzle', 'postgresql', 'sqlite', 'supabase'. Default: auto-detect from project."
                    },
                    "description": {
                        "type": "string",
                        "description": "For 'generate': describe the data model. E.g. 'users with posts and comments, each post has tags'"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "For 'view': specific schema file to read. Optional — auto-detects if omitted."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "env_manage",
            "description": "Manage environment variables safely. Create .env.example templates, check for missing vars, and validate .env files. NEVER writes actual secrets — only templates with placeholder values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'scan' (find all env vars used in project), 'template' (generate .env.example), 'check' (validate .env against .env.example), 'gitignore' (ensure .env is gitignored)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Project directory to scan. Defaults to cwd."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information, documentation, error solutions, or API references. Use when you need current information or are unsure about something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the content of a URL (webpage, documentation, API reference). Returns the page text with HTML tags stripped. Use after web_search to read a specific result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch (must start with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scaffold_project",
            "description": "Analyze a project description and generate a structured file manifest. Returns the planned file tree for confirmation before creating files. Use this for new project scaffolding — it extracts intent, asks minimal questions, and generates files in the correct dependency order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What to build (e.g., 'todo app with auth')"
                    },
                    "stack": {
                        "type": "string",
                        "description": "Optional: tech stack (e.g., 'nextjs, supabase, tailwind')"
                    },
                    "features": {
                        "type": "string",
                        "description": "Optional: comma-separated features (e.g., 'auth, crud, dashboard')"
                    }
                },
                "required": ["description"]
            }
        }
    },
]

# ---------------------------------------------------------------------------
# tool implementations
# ---------------------------------------------------------------------------

def _check_bash_safety(cmd):
    """Check if a bash command is safe to execute. Returns (safe, reason)."""
    stripped = cmd.strip()
    lower = stripped.lower()

    # --- Destructive filesystem operations ---
    # Block root deletion (rm -rf /)
    if re.search(r'\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\s*$', stripped) or \
       re.search(r'\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\*', stripped) or \
       re.search(r'\brm\s+.*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/\s*$', stripped) or \
       re.search(r'\brm\s+.*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/\*', stripped):
        return False, "Blocked: attempted root filesystem deletion"
    # Block sudo
    if re.search(r'\bsudo\b', stripped):
        return False, "Blocked: sudo is not allowed"
    # Block pipe-to-shell (curl/wget piped to bash/sh)
    if re.search(r'\b(curl|wget)\b.*\|\s*(bash|sh|zsh)\b', stripped):
        return False, "Blocked: piping remote content to shell is not allowed"
    # Block destructive system commands
    if re.search(r'\b(mkfs|mkfs\.\w+)\b', stripped):
        return False, "Blocked: filesystem formatting is not allowed"
    if re.search(r'\bdd\s+if=', stripped):
        return False, "Blocked: dd with if= is not allowed"
    # Block fork bombs
    if re.search(r':\(\)\s*\{.*\}', stripped) or ':(){ :|:& };:' in stripped:
        return False, "Blocked: fork bomb detected"
    # Block chmod 777 on root
    if re.search(r'\bchmod\s+777\s+/', stripped):
        return False, "Blocked: chmod 777 on root paths is not allowed"

    # --- Secret / credential leak prevention ---
    # Block commands that would print env secrets to stdout
    if re.search(r'^(printenv|env|set)\s*$', stripped):
        return False, "Blocked: dumping all environment variables could leak secrets. Use 'echo $SPECIFIC_VAR' instead."
    # Block echoing known secret patterns
    if re.search(r'\becho\s+.*\$(STRIPE_SECRET|DATABASE_URL|SECRET_KEY|PRIVATE_KEY|API_KEY|SERVICE_ROLE|JWT_SECRET)', stripped):
        return False, "Blocked: echoing secret environment variables. Check .env file directly instead."
    # Block cat on common secret files outside project
    if re.search(r'\b(cat|less|more|head|tail)\s+.*(\.ssh/|/etc/shadow|/etc/passwd|credentials|\.aws/)', stripped):
        return False, "Blocked: reading system credential files is not allowed"
    # Block base64 encoding of key files (exfiltration)
    if re.search(r'\bbase64\b.*\.(pem|key|p12|pfx|jks)', stripped):
        return False, "Blocked: encoding key files is not allowed"

    # --- Network security ---
    # Block reverse shells
    if re.search(r'\b(nc|ncat|netcat)\s+.*-[a-zA-Z]*e\s', stripped):
        return False, "Blocked: netcat with -e (reverse shell) is not allowed"
    if re.search(r'/dev/tcp/', stripped):
        return False, "Blocked: /dev/tcp redirect (reverse shell pattern) is not allowed"
    if re.search(r'\bpython[23]?\s+-c\s+.*socket.*connect', stripped):
        return False, "Blocked: Python socket reverse shell pattern detected"
    # Block adding SSH keys to authorized_keys
    if re.search(r'>>?\s*~?/?\.ssh/authorized_keys', stripped):
        return False, "Blocked: modifying SSH authorized_keys is not allowed"

    # --- Package/supply chain safety ---
    # Block npm/pip install from suspicious URLs (not registries)
    if re.search(r'\bnpm\s+install\s+https?://', stripped) and 'registry.npmjs.org' not in stripped:
        return False, "Blocked: installing npm packages from non-registry URLs. Use package names instead."
    if re.search(r'\bpip\s+install\s+.*--index-url\s+(?!https://pypi)', stripped):
        return False, "Blocked: pip install from non-PyPI index. Use default PyPI registry."
    # Block npm scripts that bypass audit
    if re.search(r'\bnpm\s+(audit\s+fix\s+--force|install\s+--ignore-scripts\s+--no-audit)', stripped):
        return False, "Blocked: bypassing npm security audit. Run 'npm audit' first."

    # --- Git credential safety ---
    # Block storing credentials in git config
    if re.search(r'\bgit\s+config\s+.*credential', lower):
        return False, "Blocked: modifying git credential config is not allowed"
    # Block adding secrets to git
    if re.search(r'\bgit\s+add\s+.*\.(env|pem|key|p12)(\s|$)', stripped) and '.example' not in stripped and '.env.example' not in stripped:
        return False, "Blocked: staging secret files (.env, .pem, .key). Use .env.example instead."

    # --- Process/system safety ---
    # Block killing system processes
    if re.search(r'\bkill\s+-9\s+1\b', stripped) or re.search(r'\bkillall\b', stripped):
        return False, "Blocked: killing system processes is not allowed"

    return True, ""


def _check_write_size(content):
    """Check if content is within the write size limit. Returns (ok, reason)."""
    size = len(content.encode("utf-8")) if isinstance(content, str) else len(content)
    if size > 1_000_000:  # 1MB limit
        return False, f"Content too large: {size} bytes (limit: 1MB)"
    return True, ""


def _translate_command_for_windows(cmd):
    """Translate common Unix commands to Windows equivalents."""
    # Commands that already work natively on Windows -- don't translate
    native_cmds = ("npm", "npx", "node", "python", "python3", "pip", "pip3",
                   "git", "code", "cargo", "go", "java", "javac", "docker",
                   "yarn", "pnpm", "bun", "deno", "ruby", "gem", "dotnet",
                   "mkdir", "dir", "del", "copy", "move", "type", "where",
                   "rmdir", "xcopy", "powershell", "cmd", "start", "echo",
                   "set", "call", "if", "for", "reg", "sc", "net", "icacls")
    stripped = cmd.strip()
    first_word = stripped.split()[0] if stripped.split() else ""
    # Already a Windows-native command
    if first_word in native_cmds:
        return cmd
    # Contains backslashes in paths — already Windows-style, don't wrap in bash
    if '\\' in stripped:
        return cmd

    # Translation map for common Unix commands (try these BEFORE Git Bash fallback)
    translations = {
        "ls": "dir /B",
        "ls -la": "dir",
        "ls -l": "dir",
        "ls -a": "dir /A",
        "cat": "type",
        "rm": "del /Q",
        "which": "where",
        "grep": "findstr",
        "cp": "copy",
        "mv": "move",
        "touch": "type nul >",
    }

    # Handle compound commands (&&, ||)
    if "&&" in cmd:
        parts = cmd.split("&&")
        translated_parts = [_translate_command_for_windows(p.strip()) for p in parts]
        return " && ".join(translated_parts)
    if "||" in cmd:
        parts = cmd.split("||")
        translated_parts = [_translate_command_for_windows(p.strip()) for p in parts]
        return " || ".join(translated_parts)

    # Check for rm -rf (special case)
    rm_rf_match = re.match(r'^rm\s+(-rf|-r\s+-f|-f\s+-r)\s+(.+)$', stripped)
    if rm_rf_match:
        target = rm_rf_match.group(2).strip()
        return f'rmdir /S /Q {target}'

    # Check for mkdir -p
    mkdir_p_match = re.match(r'^mkdir\s+-p\s+(.+)$', stripped)
    if mkdir_p_match:
        target = mkdir_p_match.group(1).strip()
        return f'mkdir {target}'

    # Check for cp -r
    cp_r_match = re.match(r'^cp\s+(-r|-R)\s+(.+)\s+(.+)$', stripped)
    if cp_r_match:
        src = cp_r_match.group(2).strip()
        dst = cp_r_match.group(3).strip()
        return f'xcopy /E /I {src} {dst}'

    # Simple command translations
    for unix_cmd, win_cmd in translations.items():
        if stripped == unix_cmd or stripped.startswith(unix_cmd + " "):
            rest = stripped[len(unix_cmd):].strip()
            if rest:
                return f"{win_cmd} {rest}"
            return win_cmd

    # Last resort: if Git Bash is available, wrap unrecognized Unix commands
    if _HAS_GIT_BASH:
        escaped = cmd.replace('\\', '\\\\').replace('"', '\\"')
        return f'bash -c "{escaped}"'

    return cmd


def tool_bash(args):
    cmd = args.get("command", "")
    if not cmd.strip():
        return _tool_error("bash", "empty command", "Provide a shell command to run.")

    # Security check
    safe, reason = _check_bash_safety(cmd)
    if not safe:
        return _tool_error("bash", reason)

    # Windows command translation
    if sys.platform == "win32":
        cmd = _translate_command_for_windows(cmd)

    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=CWD
        )
        out = ""
        if r.stdout:
            out += r.stdout
        if r.stderr:
            out += ("\n" if out else "") + r.stderr
        if r.returncode != 0:
            out += f"\n[exit code: {r.returncode}]"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return _tool_error("bash", "command timed out after 120 seconds", "Try a shorter-running command or break it into steps.")
    except Exception as e:
        return _tool_error("bash", str(e))

def tool_read_file(args):
    try:
        fp = _resolve(args["file_path"])
    except ValueError as e:
        return _tool_error("read_file", str(e))
    if not fp.exists():
        return _tool_error("read_file", f"file not found: {fp}", "Check the path and try again.")
    if not fp.is_file():
        return _tool_error("read_file", f"not a file: {fp}")
    # Size guard: reject files > 5MB
    try:
        fsize = fp.stat().st_size
        if fsize > 5_000_000:
            return _tool_error("read_file", f"file too large: {fsize} bytes (limit: 5MB)", "Read a specific section with offset/limit.")
    except OSError:
        pass
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return _tool_error("read_file", f"reading file: {e}")
    offset = max(args.get("offset", 1), 1) - 1
    limit = args.get("limit", len(lines))
    selected = lines[offset:offset + limit]
    numbered = []
    for i, line in enumerate(selected, start=offset + 1):
        numbered.append(f"{i:>5}\t{line}")
    # Track this file as read for read-before-write guard
    with _read_guard_lock:
        _files_read_this_turn.add(_canon(args["file_path"]))
    return "\n".join(numbered) or "(empty file)"

def tool_write_file(args):
    try:
        fp = _resolve(args["file_path"])
    except ValueError as e:
        return _tool_error("write_file", str(e))
    content = args.get("content", "")
    # Size guard
    ok, reason = _check_write_size(content)
    if not ok:
        return _tool_error("write_file", reason, "Break the content into smaller files.")
    # Security: block writing actual secret files — only .env.example allowed
    fname = fp.name.lower()
    if fname in (".env", ".env.local", ".env.production", ".env.development", ".env.staging"):
        # Check if content has real-looking secrets (not placeholders)
        has_real_key = bool(re.search(r'(sk_live_|sk-ant-api|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36})', content))
        if has_real_key:
            return _tool_error("write_file",
                f"Blocked: refusing to write real credentials to {fname}. "
                "Never hardcode secrets. Use .env.example with placeholders instead.",
                "Use env_manage action='template' to generate a safe .env.example.")
    # Block writing private key files
    if fp.suffix.lower() in (".pem", ".key", ".p12", ".pfx", ".jks"):
        return _tool_error("write_file",
            f"Blocked: writing private key files ({fp.suffix}) is not allowed.",
            "Use environment variables or a secrets manager for credentials.")
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {fp}"
    except Exception as e:
        return _tool_error("write_file", f"writing file: {e}")

def tool_edit_file(args):
    try:
        fp = _resolve(args["file_path"])
    except ValueError as e:
        return _tool_error("edit_file", str(e))
    if not fp.exists():
        return _tool_error("edit_file", f"file not found: {fp}", "Use read_file first to verify the path.")
    try:
        content = fp.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"
    old = args["old_string"]
    new = args["new_string"]
    count = content.count(old)
    if count == 0:
        return f"Error: old_string not found in {fp}"
    if count > 1:
        return f"Error: old_string appears {count} times -- must be unique. Add more context."
    new_content = content.replace(old, new, 1)
    fp.write_text(new_content, encoding="utf-8")
    # Show colored diff inline
    _diff_display(old, new, filepath=str(fp.name), context_lines=2)
    return f"Edited {fp} (replaced 1 occurrence)"

def tool_glob_search(args):
    pattern = args["pattern"]
    base = args.get("path", CWD)
    base = _resolve(base)
    matches = sorted(glob_mod.glob(str(base / pattern), recursive=True))
    if not matches:
        return "No files matched."
    result = []
    for m in matches[:200]:
        try:
            result.append(str(Path(m).relative_to(CWD)))
        except ValueError:
            result.append(m)
    # Track globbed files for read-before-write guard
    with _read_guard_lock:
        for m in matches[:200]:
            try:
                _files_searched_this_turn.add(str(Path(m).resolve()))
            except Exception:
                pass
    out = "\n".join(result)
    if len(matches) > 200:
        out += f"\n... and {len(matches) - 200} more"
    return out

def tool_grep_search(args):
    pattern = args["pattern"]
    base = args.get("path", CWD)
    base = _resolve(base)
    include = args.get("include", "")
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    results = []
    if base.is_file():
        files_to_search = [base]
    else:
        glob_pat = include if include else "**/*"
        files_to_search = [
            Path(p) for p in glob_mod.glob(str(base / glob_pat), recursive=True)
            if Path(p).is_file()
        ]
    for fp in files_to_search[:500]:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                try:
                    rel = str(fp.relative_to(CWD))
                except ValueError:
                    rel = str(fp)
                results.append(f"{rel}:{i}: {line.rstrip()}")
                if len(results) >= 100:
                    break
        if len(results) >= 100:
            break
    if not results:
        return "No matches found."
    # Track searched files for read-before-write guard
    with _read_guard_lock:
        for r in results:
            parts = r.rsplit(":", 2)
            if len(parts) >= 3:
                try:
                    _files_searched_this_turn.add(_canon(parts[0]))
                except (ValueError, Exception):
                    pass
    return "\n".join(results)


# ---------------------------------------------------------------------------
# db_schema tool — database-agnostic schema management
# ---------------------------------------------------------------------------

def _detect_orm(base_dir):
    """Auto-detect which ORM/database the project uses."""
    base = Path(base_dir)
    if (base / "prisma" / "schema.prisma").exists():
        return "prisma"
    if any(base.rglob("drizzle.config.*")):
        return "drizzle"
    # Check package.json
    pkg = base / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "prisma" in deps or "@prisma/client" in deps:
                return "prisma"
            if "drizzle-orm" in deps:
                return "drizzle"
            if "@supabase/supabase-js" in deps:
                return "supabase"
        except Exception:
            pass
    # Check for requirements.txt or Python files
    if (base / "requirements.txt").exists():
        try:
            reqs = (base / "requirements.txt").read_text(encoding="utf-8").lower()
            if "sqlalchemy" in reqs:
                return "postgresql"
            if "django" in reqs:
                return "postgresql"
        except Exception:
            pass
    if any(base.rglob("*.db")) or any(base.rglob("*.sqlite")):
        return "sqlite"
    return "postgresql"


def tool_db_schema(args):
    """Handle db_schema tool calls."""
    action = args.get("action", "").strip().lower()
    orm = args.get("orm", "").strip().lower()
    desc = args.get("description", "").strip()
    fp = args.get("file_path", "")

    if not orm:
        orm = _detect_orm(CWD)

    if action == "view":
        # Show current schema files
        base = Path(CWD)
        schema_files = []
        if fp:
            try:
                p = _resolve(fp)
                if p.exists():
                    content = p.read_text(encoding="utf-8", errors="replace")
                    return f"Schema: {p}\n\n{content}"
            except Exception as e:
                return f"Error reading schema: {e}"

        # Auto-detect schema files
        patterns = {
            "prisma": ["prisma/schema.prisma"],
            "drizzle": ["src/db/schema.ts", "src/db/schema.js", "db/schema.ts", "drizzle/schema.ts"],
            "supabase": ["supabase/migrations/*.sql", "sql/*.sql"],
            "postgresql": ["schema.sql", "migrations/*.sql", "sql/*.sql", "db/schema.sql"],
            "sqlite": ["schema.sql", "*.db"],
        }
        for pattern in patterns.get(orm, ["schema.sql", "*.sql"]):
            for match in sorted(glob_mod.glob(str(base / pattern), recursive=True)):
                schema_files.append(match)
        if not schema_files:
            return f"No schema files found for ORM '{orm}'. Use db_schema with action='generate' to create one."
        result = f"Schema files detected (ORM: {orm}):\n"
        for sf in schema_files[:10]:
            try:
                content = Path(sf).read_text(encoding="utf-8", errors="replace")
                rel = Path(sf).relative_to(base) if Path(sf).is_relative_to(base) else sf
                result += f"\n--- {rel} ---\n{content[:3000]}\n"
            except Exception:
                result += f"\n--- {sf} (unreadable) ---\n"
        return result

    elif action == "generate":
        if not desc:
            return "Error: 'description' is required for action='generate'. Describe your data model."
        # Load database patterns
        db_patterns = load_api_patterns("database")
        if not db_patterns:
            return "Error: database.json API patterns not found."
        pattern = db_patterns.get("schema_patterns", {}).get(orm, db_patterns.get("schema_patterns", {}).get("postgresql", {}))
        # Return the patterns + description for the model to use
        result = f"Database schema patterns for '{orm}':\n\n"
        result += f"Description: {desc}\n\n"
        for key, val in pattern.items():
            if key == "description":
                continue
            result += f"### {key}\n```\n{val}\n```\n\n"
        result += "Use these patterns as a base. Adapt the tables/columns to match the description above."
        result += "\n\n⚠️ SECURITY REMINDERS:\n"
        for req in db_patterns.get("security_requirements", [])[:5]:
            result += f"  - {req}\n"
        return result

    elif action == "migrate":
        db_patterns = load_api_patterns("database")
        if not db_patterns:
            return "Error: database.json not found."
        migration_info = db_patterns.get("migration_patterns", {}).get(orm, {})
        if not migration_info:
            return f"No migration patterns found for '{orm}'. Supported: prisma, drizzle, raw_sql."
        result = f"Migration commands for '{orm}':\n"
        for cmd_name, cmd_val in migration_info.items():
            result += f"  {cmd_name}: {cmd_val}\n"
        return result

    elif action == "seed":
        return (
            f"Seed data generation for '{orm}':\n\n"
            "Use the write_file tool to create a seed script. Examples:\n"
            "  Prisma:  prisma/seed.ts  → run with: npx prisma db seed\n"
            "  Drizzle: src/db/seed.ts  → run with: npx tsx src/db/seed.ts\n"
            "  Raw SQL: seed.sql        → run with: psql -f seed.sql\n"
            "  SQLite:  seed.py         → run with: python seed.py\n\n"
            "SECURITY: Use realistic but FAKE data for seeds. Never use real user data."
        )

    return f"Unknown action '{action}'. Use: generate, view, migrate, seed."


# ---------------------------------------------------------------------------
# env_manage tool — secure environment variable management
# ---------------------------------------------------------------------------

# Files that MUST be gitignored — never committed
_SECRET_FILE_PATTERNS = {
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.staging", ".env.test",
}
_SECRET_FILE_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}

# Known env var categories
_ENV_VAR_DOCS = {
    "DATABASE_URL": ("Database connection string", "postgresql://user:password@localhost:5432/dbname"),
    "STRIPE_SECRET_KEY": ("Stripe secret key (server-side only)", "sk_test_..."),
    "NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY": ("Stripe publishable key (client-safe)", "pk_test_..."),
    "STRIPE_WEBHOOK_SECRET": ("Stripe webhook endpoint secret", "whsec_..."),
    "NEXT_PUBLIC_SUPABASE_URL": ("Supabase project URL", "https://xxx.supabase.co"),
    "NEXT_PUBLIC_SUPABASE_ANON_KEY": ("Supabase anonymous key (client-safe)", "eyJ..."),
    "SUPABASE_SERVICE_ROLE_KEY": ("Supabase service role key (server-side ONLY)", "eyJ..."),
    "JWT_SECRET": ("JWT signing secret", "your-256-bit-secret"),
    "SESSION_SECRET": ("Session encryption secret", "random-32-char-string"),
    "RESEND_API_KEY": ("Resend email API key", "re_..."),
    "OPENAI_API_KEY": ("OpenAI API key", "sk-..."),
    "ANTHROPIC_API_KEY": ("Anthropic API key", "sk-ant-..."),
    "AWS_ACCESS_KEY_ID": ("AWS access key ID", "AKIA..."),
    "AWS_SECRET_ACCESS_KEY": ("AWS secret access key", "wJal..."),
    "REDIS_URL": ("Redis connection URL", "redis://localhost:6379"),
    "SMTP_HOST": ("SMTP server hostname", "smtp.gmail.com"),
    "SMTP_USER": ("SMTP username", "user@example.com"),
    "SMTP_PASS": ("SMTP password", "app-specific-password"),
}

# Env vars that are safe to expose client-side (public prefixes)
_PUBLIC_PREFIXES = ("NEXT_PUBLIC_", "REACT_APP_", "VITE_", "NUXT_PUBLIC_")


def tool_env_manage(args):
    """Handle env_manage tool calls."""
    action = args.get("action", "").strip().lower()
    base = Path(args.get("path", CWD))

    if action == "scan":
        # Find all env vars used in the project
        env_vars = {}  # var_name -> list of files
        for ext in ("*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.mjs", "*.cjs"):
            for fp in base.rglob(ext):
                if "node_modules" in str(fp) or ".next" in str(fp) or "__pycache__" in str(fp):
                    continue
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                # JS/TS: process.env.VAR
                for m in re.finditer(r'process\.env\.(\w+)', content):
                    env_vars.setdefault(m.group(1), []).append(str(fp))
                # Python: os.environ
                for m in re.finditer(r'os\.environ(?:\.get)?\s*[\[(]\s*["\'](\w+)["\']', content):
                    env_vars.setdefault(m.group(1), []).append(str(fp))
                # Vite: import.meta.env.VITE_
                for m in re.finditer(r'import\.meta\.env\.(\w+)', content):
                    env_vars.setdefault(m.group(1), []).append(str(fp))

        if not env_vars:
            return "No environment variables detected in project source files."

        result = f"Found {len(env_vars)} environment variables:\n\n"
        for var, files in sorted(env_vars.items()):
            is_public = any(var.startswith(p) for p in _PUBLIC_PREFIXES)
            is_secret = not is_public
            doc = _ENV_VAR_DOCS.get(var, ("", ""))
            marker = "🔒 SECRET" if is_secret else "🌐 PUBLIC"
            desc = f" — {doc[0]}" if doc[0] else ""
            result += f"  {marker} {var}{desc}\n"
            for f in files[:3]:
                try:
                    rel = Path(f).relative_to(base)
                except ValueError:
                    rel = f
                result += f"    └─ {rel}\n"
        return result

    elif action == "template":
        # Generate .env.example from scanned vars
        env_vars = set()
        for ext in ("*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.mjs", "*.cjs"):
            for fp in base.rglob(ext):
                if "node_modules" in str(fp) or ".next" in str(fp):
                    continue
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                for m in re.finditer(r'process\.env\.(\w+)', content):
                    env_vars.add(m.group(1))
                for m in re.finditer(r'os\.environ(?:\.get)?\s*[\[(]\s*["\'](\w+)["\']', content):
                    env_vars.add(m.group(1))
                for m in re.finditer(r'import\.meta\.env\.(\w+)', content):
                    env_vars.add(m.group(1))

        skip = {"NODE_ENV", "PATH", "HOME", "USER", "SHELL", "TERM", "PWD", "CI"}
        env_vars -= skip

        if not env_vars:
            return "No environment variables detected. Nothing to template."

        lines = ["# Environment Variables", "# Copy this to .env and fill in real values", "#", "# ⚠️  NEVER commit .env to git — only commit .env.example", ""]
        for var in sorted(env_vars):
            doc = _ENV_VAR_DOCS.get(var, ("", ""))
            is_public = any(var.startswith(p) for p in _PUBLIC_PREFIXES)
            if doc[0]:
                lines.append(f"# {doc[0]}{' (client-safe)' if is_public else ' (server-side ONLY)' if not is_public else ''}")
            placeholder = doc[1] if doc[1] else "your-value-here"
            lines.append(f"{var}={placeholder}")
            lines.append("")

        template_content = "\n".join(lines)
        out_path = base / ".env.example"
        try:
            out_path.write_text(template_content, encoding="utf-8")
            return f"Created {out_path} with {len(env_vars)} variables.\n\n{template_content}"
        except Exception as e:
            return f"Error writing .env.example: {e}\n\nTemplate content:\n{template_content}"

    elif action == "check":
        # Validate .env against .env.example
        example = base / ".env.example"
        env_file = None
        for name in (".env", ".env.local", ".env.development"):
            candidate = base / name
            if candidate.exists():
                env_file = candidate
                break

        if not example.exists():
            return "No .env.example found. Run env_manage with action='template' first."
        if not env_file:
            return "No .env file found. Copy .env.example to .env and fill in real values."

        # Parse both files
        def parse_env(path):
            vals = {}
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r'^(\w+)\s*=\s*(.*)', line)
                if m:
                    vals[m.group(1)] = m.group(2)
            return vals

        required = parse_env(example)
        actual = parse_env(env_file)

        missing = set(required.keys()) - set(actual.keys())
        extra = set(actual.keys()) - set(required.keys())
        placeholder = {k for k, v in actual.items() if v in ("your-value-here", "", "sk_test_...", "eyJ...")}

        result = f"Checking {env_file.name} against .env.example:\n\n"
        if missing:
            result += f"❌ MISSING ({len(missing)}):\n"
            for v in sorted(missing):
                result += f"  {v}\n"
        if placeholder:
            result += f"\n⚠️  PLACEHOLDER VALUES ({len(placeholder)}):\n"
            for v in sorted(placeholder):
                result += f"  {v}={actual[v]}\n"
        if extra:
            result += f"\nℹ️  Extra vars not in .env.example ({len(extra)}):\n"
            for v in sorted(extra):
                result += f"  {v}\n"
        if not missing and not placeholder:
            result += "✅ All required variables are set with real values.\n"
        return result

    elif action == "gitignore":
        # Ensure .env files are gitignored
        gi_path = base / ".gitignore"
        existing = ""
        if gi_path.exists():
            try:
                existing = gi_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        lines_to_add = []
        for pattern in sorted(_SECRET_FILE_PATTERNS):
            if pattern not in existing:
                lines_to_add.append(pattern)
        for ext in sorted(_SECRET_FILE_EXTENSIONS):
            glob_pat = f"*{ext}"
            if glob_pat not in existing and ext not in existing:
                lines_to_add.append(glob_pat)

        if not lines_to_add:
            return "✅ .gitignore already covers all secret file patterns."

        addition = "\n# Secrets — auto-added by Rattlesnake security\n" + "\n".join(lines_to_add) + "\n"
        try:
            with open(gi_path, "a", encoding="utf-8") as f:
                f.write(addition)
            return f"Updated .gitignore with {len(lines_to_add)} secret patterns:\n{addition}"
        except Exception as e:
            return f"Error updating .gitignore: {e}"

    return f"Unknown action '{action}'. Use: scan, template, check, gitignore."


def tool_web_search(args):
    """Search the web using DuckDuckGo HTML and return top results."""
    query = args.get("query", "").strip()
    if not query:
        return _tool_error("web_search", "empty query", "Provide a search query.")
    num = min(args.get("num_results", 5), 10)
    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Rattlesnake CLI)"})
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
        results = []
        for m in re.finditer(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', html):
            link = m.group(1)
            title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if title and link and len(results) < num:
                results.append(f"- [{title}]({link})")
        # Extract snippets
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html)
        for i, snip in enumerate(snippets[:len(results)]):
            clean = re.sub(r'<[^>]+>', '', snip).strip()
            if clean:
                results[i] += f"\n  {clean[:200]}"
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return _tool_error("web_search", f"search failed: {e}", "Check internet connection.")


def tool_web_fetch(args):
    """Fetch a URL and return its text content (HTML tags stripped)."""
    url = args.get("url", "").strip()
    if not url:
        return _tool_error("web_fetch", "empty URL", "Provide a URL to fetch.")
    if not url.startswith("http"):
        return _tool_error("web_fetch", "invalid URL — must start with http:// or https://")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Rattlesnake CLI)"})
        resp = urllib.request.urlopen(req, timeout=15)
        raw = resp.read().decode("utf-8", errors="replace")
        # Strip scripts, styles, then all tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Cap to keep within context budget
        if len(text) > 4000:
            text = text[:4000] + "\n... [truncated, full page was larger]"
        return text if text else "Page returned no readable text content."
    except Exception as e:
        return _tool_error("web_fetch", f"fetch failed: {e}")


TOOL_HANDLERS = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "glob_search": tool_glob_search,
    "grep_search": tool_grep_search,
    "ask_user": tool_ask_user,
    "memory_save": tool_memory_save,
    "memory_search": tool_memory_search,
    "db_schema": tool_db_schema,
    "env_manage": tool_env_manage,
    "web_search": tool_web_search,
    "web_fetch": tool_web_fetch,
}


# ---------------------------------------------------------------------------
# plugin system -- load user extensions from ~/.claw/plugins/
# ---------------------------------------------------------------------------

PLUGINS_DIR = Path.home() / ".claw" / "plugins"

def _load_plugins():
    """Load .py plugin files from ~/.claw/plugins/. Each must export TOOLS list."""
    if not PLUGINS_DIR.exists():
        return 0
    import importlib.util
    count = 0
    for pf in sorted(PLUGINS_DIR.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(pf.stem, str(pf))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for tool in getattr(mod, "TOOLS", []):
                name = tool.get("name")
                handler = tool.get("handler")
                if not name or not handler:
                    continue
                TOOL_DEFS.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", f"Plugin tool: {name}"),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
                    }
                })
                TOOL_HANDLERS[name] = handler
                count += 1
        except Exception as e:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Plugin {pf.name} failed: {e}{C.RESET}")
    return count


# ---------------------------------------------------------------------------
# token tracking (feature 1)
# ---------------------------------------------------------------------------

class TokenTracker:
    """Track tokens across the entire session."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_requests = 0
    def add(self, prompt=0, completion=0):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_requests += 1
    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens
    def summary(self):
        return f"{_format_tokens(self.total)} total ({_format_tokens(self.prompt_tokens)} in, {_format_tokens(self.completion_tokens)} out, {self.total_requests} calls)"

_token_tracker = TokenTracker()

# ---------------------------------------------------------------------------
# output quality guards (telemetry + helpers)
# ---------------------------------------------------------------------------

_guard_stats = {"ramble_truncations": 0, "dedup_fires": 0, "html_strips": 0, "read_guard_blocks": 0, "auto_symbol_searches": 0, "todo_resolves": 0}

_CODE_LINE_RE = re.compile(
    r'^(?:'
    r'[\t ]{4,}'            # indented ≥4 spaces/tabs
    r'|.*[{};]'             # braces or semicolons
    r'|(?:import |export |from |function |const |let |var |class |def |return |if |for |while |async |await )' # keywords at start
    r'|```'                  # fenced code blocks
    r')',
)


def _is_code_dump(text):
    """Return True if >60% of lines look like code rather than prose."""
    lines = text.split("\n")
    if len(lines) < 5:
        return False
    code_lines = sum(1 for ln in lines if _CODE_LINE_RE.match(ln))
    return code_lines / len(lines) > 0.6


def _truncate_at_sentence(text, limit):
    """Truncate text at the last sentence boundary before *limit* chars."""
    if len(text) <= limit:
        return text
    region = text[:limit]
    # Find last sentence-ending punctuation followed by whitespace/newline
    best = -1
    for pat in ('. ', '.\n', '?\n', '!\n', '? ', '! '):
        idx = region.rfind(pat)
        if idx > best:
            best = idx
    if best > 0:
        return region[:best + 1]
    # Fallback: break at last space
    space_idx = region.rfind(' ')
    if space_idx > 0:
        return region[:space_idx]
    return region


def _deduplicate_response(text):
    """Sliding-window duplicate detector.

    Split into 300-char chunks overlapping by 100. If any chunk hash appears
    2+ times at positions >500 chars apart AND the region is >400 chars,
    truncate at the earliest duplicate.  Returns (cleaned_text, was_deduped).
    """
    if len(text) < 800:
        return text, False
    # Normalize whitespace for hashing
    import hashlib
    norm = re.sub(r'\s+', ' ', text)
    chunk_size = 300
    step = 200  # overlap of 100 chars
    seen = {}  # hash -> first position in original
    dup_start = None
    pos = 0
    while pos + chunk_size <= len(norm):
        chunk = norm[pos:pos + chunk_size]
        h = hashlib.md5(chunk.encode()).hexdigest()
        if h in seen:
            gap = pos - seen[h]
            if gap > 500:
                dup_start = pos
                break
        else:
            seen[h] = pos
        pos += step
    if dup_start is None:
        return text, False
    # Map normalized position back to original: use ratio
    ratio = len(text) / len(norm) if norm else 1
    orig_pos = min(int(dup_start * ratio), len(text))
    # Ensure the duplicate region is >400 chars
    if len(text) - orig_pos < 400:
        return text, False
    cleaned = text[:orig_pos].rstrip() + "\n\n[... repeated content removed]"
    return cleaned, True


# Structural HTML tags to strip from display (not <code>, <pre>, <a>)
_STRUCTURAL_HTML_RE = re.compile(r'</?(?:h[1-6]|br|hr|div|p)(?:\s[^>]*)?>', re.IGNORECASE)


def _strip_structural_html(text):
    """Strip structural HTML tags from text for terminal display.

    Preserves content inside triple-backtick fences.
    Returns (cleaned_text, did_strip).
    """
    parts = text.split('```')
    changed = False
    for i in range(0, len(parts), 2):  # only outside fences (even indices)
        new_part = _STRUCTURAL_HTML_RE.sub('', parts[i])
        if new_part != parts[i]:
            changed = True
            parts[i] = new_part
    if not changed:
        return text, False
    return '```'.join(parts), True


# ---------------------------------------------------------------------------
# undo / rollback (feature 2)
# ---------------------------------------------------------------------------

_undo_stack = []  # list of (action, filepath, old_content)

# ---------------------------------------------------------------------------
# read-before-write enforcement globals
# ---------------------------------------------------------------------------
_files_read_this_turn = set()       # canonical paths from read_file
_files_searched_this_turn = set()   # canonical paths from grep_search/glob_search
_read_guard_lock = threading.Lock() # protects both sets from concurrent subagent access
_thread_local = threading.local()   # per-thread state (e.g. is_subagent flag)
READ_GUARD_ENABLED = True           # feature flag — disable without reverting code
AUTO_SYMBOL_SEARCH = True           # feature flag for post-edit symbol search
TODO_RESOLVER_ENABLED = True        # feature flag for auto-TODO resolution


def _find_related_references(edit_args):
    """After edit_file, grep for symbols from old_string to show related references."""
    old_string = edit_args.get("old_string", "")
    edited_file = edit_args.get("file_path", "")
    if not old_string or not edited_file or len(old_string) < 10:
        return ""

    # Extract ONLY declaration-style identifiers (not every word)
    decl_re = re.compile(
        r'(?:def |class |function |const |let |var |export (?:default )?(?:function |class )?)'
        r'(\w{3,})'
    )
    names = set(decl_re.findall(old_string))

    # Also: CamelCase (likely class/component) and snake_case with 4+ chars
    SKIP = {"True", "False", "None", "self", "this", "return", "import", "from", "else",
            "elif", "async", "await", "print", "string", "number", "boolean", "export", "default"}
    for ident in re.findall(r'\b([A-Z][a-zA-Z0-9]{3,}|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', old_string):
        if ident not in SKIP:
            names.add(ident)

    if not names:
        return ""

    # Deduplicate, take top 5 by length (most specific)
    symbols = sorted(set(names), key=len, reverse=True)[:5]

    try:
        edited_resolved = str(_resolve(edited_file).resolve())
    except (ValueError, Exception):
        edited_resolved = ""

    refs = []
    for sym in symbols:
        if len(refs) >= 15:  # GLOBAL cap across all symbols
            break
        try:
            grep_out = tool_grep_search({"pattern": re.escape(sym), "path": str(CWD)})
            if grep_out and grep_out != "No matches found.":
                for line in grep_out.split("\n"):
                    if len(refs) >= 15:
                        break
                    # Extract file path — rsplit from right to handle Windows drive letters
                    parts = line.rsplit(":", 2)
                    if len(parts) < 3:
                        continue
                    ref_file = parts[0]
                    try:
                        if str((_resolve(ref_file)).resolve()) == edited_resolved:
                            continue  # skip the file we just edited
                    except Exception:
                        pass
                    refs.append(f"  {line.strip()}")
        except Exception:
            continue

    if not refs:
        refs_text = ""
    else:
        # Final dedup
        seen = set()
        unique = [r for r in refs if r not in seen and not seen.add(r)]
        refs_text = "\n".join(unique[:15])

    # --- GRAPH-BASED IMPACT ANALYSIS (Phase 4) ---
    if _project_graph is not None and edited_file:
        try:
            _ef_resolved = str(_resolve(edited_file).resolve())
            _ef_rel = str(Path(_ef_resolved).relative_to(Path(CWD).resolve())).replace('\\', '/')
            file_importers = _project_graph.importers.get(_ef_rel, set())
            if file_importers:
                # Extract declaration-style identifiers from old_string
                file_exports = set(_project_graph.exports.get(_ef_rel, []))
                if file_exports:
                    # Find symbols in old_string that are also this file's exports
                    affected_exports = file_exports & names if names else set()
                    if affected_exports:
                        impact_lines = [f"\n\u26a0 IMPACT: {len(file_importers)} file(s) import from {_ef_rel}:"]
                        for imp_file in sorted(file_importers)[:10]:
                            impact_lines.append(f"  \u2192 {imp_file}")
                        if len(file_importers) > 10:
                            impact_lines.append(f"  ... and {len(file_importers) - 10} more")
                        impact_lines.append(f"  Exports affected: {', '.join(sorted(affected_exports))}")
                        refs_text += "\n".join(impact_lines)
        except Exception:
            pass

    return refs_text


def _scan_all_todos(written_files):
    """Scan files written this turn for TODOs/stubs. Returns list of dicts."""
    todos = []
    for fp_str in written_files:
        fp = Path(fp_str)
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = content.splitlines()
        ext = fp.suffix.lower()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            lower = stripped.lower()
            todo_entry = None

            # TODO/FIXME/HACK/XXX comments
            m = re.search(r'(//|#|/\*|\*)\s*(TODO|FIXME|HACK|XXX)\b[:\s]*(.*)', stripped, re.IGNORECASE)
            if m:
                todo_entry = {"type": "todo_comment", "text": m.group(3).strip() or m.group(2), "line": i}

            # Empty function bodies
            if ext == ".py" and stripped == "pass":
                for j in range(max(0, i - 4), i - 1):
                    prev = lines[j].strip()
                    if prev.startswith("def ") or prev.startswith("class "):
                        todo_entry = {"type": "empty_body", "text": prev, "line": j + 1}
                        break

            # JS/TS stubs
            if ext in (".js", ".ts", ".jsx", ".tsx"):
                if 'throw new Error("Not implemented")' in stripped or \
                   "throw new Error('Not implemented')" in stripped:
                    todo_entry = {"type": "not_implemented", "text": stripped, "line": i}
                if stripped in ("...", "...;"):
                    todo_entry = {"type": "ellipsis_stub", "text": "incomplete code block", "line": i}

            # Placeholder text in any file
            if '"placeholder"' in lower or "'placeholder'" in lower:
                todo_entry = {"type": "placeholder", "text": stripped[:80], "line": i}
            if 'lorem ipsum' in lower:
                todo_entry = {"type": "placeholder", "text": "Lorem ipsum text", "line": i}
            if '"not implemented"' in lower or "'not implemented'" in lower:
                todo_entry = {"type": "not_implemented", "text": stripped[:80], "line": i}

            if todo_entry:
                ctx_start = max(0, i - 4)
                ctx_end = min(len(lines), i + 3)
                context_lines = lines[ctx_start:ctx_end]
                todo_entry["file"] = str(fp)
                todo_entry["context"] = "\n".join(f"  {n + ctx_start + 1:>5} | {l}" for n, l in enumerate(context_lines))
                todos.append(todo_entry)

    return todos[:30]


def _build_todo_resolver_prompt(todos):
    """Build a context-rich prompt for the model to resolve all TODOs."""
    if not todos:
        return None

    profile = _get_cached_profile()
    domain = _infer_project_domain()

    lines = [
        "[SYSTEM: TODO RESOLVER — You left incomplete code. A project manager has reviewed your work and found the following issues that MUST be resolved NOW.]\n"
    ]

    lines.append("## Project Context")
    if profile:
        if profile.framework:
            lines.append(f"- Framework: {profile.framework}")
        if profile.styling:
            lines.append(f"- Styling: {profile.styling}")
        if profile.base_info.get("type"):
            lines.append(f"- Project type: {profile.base_info['type']}")
    if domain:
        lines.append(f"- Brand: {domain['brand']}")
        lines.append(f"- Tone: {domain['tone']}")
    lines.append("")

    by_file = {}
    for t in todos:
        by_file.setdefault(t["file"], []).append(t)

    lines.append(f"## {len(todos)} Incomplete Items Found\n")

    for filepath, file_todos in by_file.items():
        rel = filepath
        try:
            rel = str(Path(filepath).relative_to(CWD))
        except ValueError:
            pass
        lines.append(f"### File: `{rel}`")
        for t in file_todos:
            icon = {"todo_comment": "TODO", "empty_body": "EMPTY", "not_implemented": "STUB",
                    "ellipsis_stub": "STUB", "placeholder": "PLACEHOLDER"}.get(t["type"], "ISSUE")
            lines.append(f"\n**[{icon}] Line {t['line']}:** {t['text']}")
            lines.append("```")
            lines.append(t["context"])
            lines.append("```")
        lines.append("")

    lines.append("## Your Task")
    lines.append("For EACH item above:")
    lines.append("1. **read_file** the file to see current state")
    lines.append("2. **Understand** the surrounding code, imports, and what the function/component should do")
    lines.append("3. **edit_file** to replace the TODO/stub/placeholder with a COMPLETE, production-quality implementation")
    lines.append("4. Match the existing code style, use the project's framework/styling conventions")
    if domain:
        lines.append(f"5. Content should match the brand tone: {domain['tone']}")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Do NOT create new TODOs — resolve them completely")
    lines.append("- Do NOT skip any item — implement ALL of them")
    lines.append("- Write real, working code — not more stubs")
    lines.append("- Keep edits surgical — only replace the incomplete parts")

    return "\n".join(lines)


def _snapshot_for_undo(tool_name, tool_args):
    """Snapshot a file before write/edit for undo support."""
    if tool_name not in ("write_file", "edit_file"):
        return
    fp_str = tool_args.get("file_path", "")
    if not fp_str:
        return
    try:
        fp = _resolve(fp_str)
        old_content = fp.read_text(encoding="utf-8") if fp.exists() else None
        _undo_stack.append((tool_name, str(fp), old_content))
        # Keep stack bounded
        if len(_undo_stack) > 50:
            _undo_stack.pop(0)
    except Exception:
        pass

def _perform_undo():
    """Undo the last write/edit operation. Returns description of what was undone."""
    if not _undo_stack:
        return "Nothing to undo."
    tool_name, filepath, old_content = _undo_stack.pop()
    fp = Path(filepath)
    if old_content is None:
        # File didn't exist before — delete it
        if fp.exists():
            fp.unlink()
            return f"Deleted {fp.name} (was newly created)"
        return f"File {fp.name} already doesn't exist."
    else:
        fp.write_text(old_content, encoding="utf-8")
        return f"Reverted {fp.name} to previous version ({len(old_content)} chars)"


# ---------------------------------------------------------------------------
# session save / resume (feature 3)
# ---------------------------------------------------------------------------

def _save_session(messages, session_id=None):
    """Save conversation messages to disk."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not session_id:
        session_id = f"session_{int(time.time())}"
    session_file = SESSIONS_DIR / f"{session_id}.json"
    data = {"id": session_id, "timestamp": time.time(), "cwd": CWD, "messages": messages}
    session_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return str(session_file)

def _load_session(session_id=None):
    """Load the most recent session or a specific one. Returns (messages, session_id) or (None, None)."""
    if not SESSIONS_DIR.exists():
        return None, None
    if session_id:
        sf = SESSIONS_DIR / f"{session_id}.json"
        if sf.exists():
            data = json.loads(sf.read_text(encoding="utf-8"))
            return data.get("messages", []), data.get("id", session_id)
        return None, None
    # Find most recent
    sessions = sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not sessions:
        return None, None
    data = json.loads(sessions[0].read_text(encoding="utf-8"))
    return data.get("messages", []), data.get("id", "")

def _list_sessions(limit=10):
    """List recent sessions."""
    if not SESSIONS_DIR.exists():
        return []
    sessions = sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for sf in sessions[:limit]:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            msg_count = len(data.get("messages", []))
            ts = data.get("timestamp", 0)
            cwd = data.get("cwd", "")
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"
            result.append((data.get("id", sf.stem), time_str, msg_count, cwd))
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# codebase map (feature 4)
# ---------------------------------------------------------------------------

def _build_codebase_map(project_dir, max_tokens=1500):
    """
    Scan the project and build a compact map of files, exports, and structure.
    Returns a string suitable for injection into the system prompt.
    """
    pdir = Path(project_dir)
    skip_dirs = {"node_modules", ".git", ".next", "__pycache__", "venv", ".venv",
                 "dist", "build", "env", ".cache", "coverage", ".tox", "egg-info"}
    code_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".sql"}

    files_info = []
    total_chars = 0

    for root, dirs, files in os.walk(pdir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        rel_root = Path(root).relative_to(pdir)

        for fname in sorted(files):
            fp = Path(root) / fname
            if fp.suffix.lower() not in code_exts:
                continue
            rel_path = str(rel_root / fname).replace("\\", "/")
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]

            try:
                size = fp.stat().st_size
                if size > 100_000:  # skip huge files
                    files_info.append(f"  {rel_path} ({size // 1024}KB)")
                    continue
                content = fp.read_text(encoding="utf-8", errors="replace")[:3000]

                # Extract key exports/definitions
                exports = []
                if fp.suffix in (".py",):
                    for m in re.finditer(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE):
                        exports.append(m.group(1))
                elif fp.suffix in (".js", ".ts", ".tsx", ".jsx"):
                    for m in re.finditer(r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)', content):
                        exports.append(m.group(1))

                exp_str = f" → {', '.join(exports[:8])}" if exports else ""
                files_info.append(f"  {rel_path}{exp_str}")
                total_chars += len(files_info[-1])
            except Exception:
                files_info.append(f"  {rel_path}")

            # Cap at max_tokens (~4 chars per token)
            if total_chars > max_tokens * 4:
                files_info.append(f"  ... ({len(files_info)} files shown)")
                break
        if total_chars > max_tokens * 4:
            break

    if not files_info:
        return ""

    return "## Codebase Map\n" + "\n".join(files_info)

_build_codebase_map._cache = None
_build_codebase_map._cache_cwd = None


# ---------------------------------------------------------------------------
# graph-aware context injection (Phase 2)
# ---------------------------------------------------------------------------

def _graph_token_budget(ctx_size):
    """Token budget for graph context based on model context size."""
    if ctx_size <= 4096:
        return 200
    if ctx_size <= 8192:
        return 600
    if ctx_size <= 32768:
        return 1500
    return 3000


def _build_graph_context(seed_files, max_tokens=1500):
    """Build focused graph context from seed files for prompt injection.
    Returns a string suitable for injection, or '' if graph unavailable.
    """
    global _project_graph
    if _project_graph is None:
        try:
            _get_project_graph()
        except Exception:
            return ""
    if _project_graph is None:
        return ""

    subgraph = _project_graph.get_subgraph(seed_files)
    if not subgraph:
        return ""

    lines = [f"## Project Graph (focused on {len(subgraph)} files)"]
    total_chars = 0
    char_limit = max_tokens * 4  # ~4 chars per token heuristic

    for fpath in sorted(subgraph.keys()):
        info = subgraph[fpath]
        exports = info.get('exports', [])
        importers = info.get('importers', set())
        exp_str = f" → {', '.join(exports[:8])}" if exports else ""
        line = f"  {fpath}{exp_str}"
        if importers:
            # Only show importers that are in the subgraph or are direct dependents
            dep_list = sorted(importers)[:5]
            line += f"\n    ← imported by: {', '.join(dep_list)}"
            if len(importers) > 5:
                line += f" (+{len(importers) - 5} more)"
        total_chars += len(line)
        if total_chars > char_limit:
            lines.append(f"  ... (truncated, {len(subgraph)} files total)")
            break
        lines.append(line)

    if _project_graph.is_partial:
        lines.append(f"(graph covers {len(_project_graph.file_hashes)} of {_project_graph._total_files_on_disk} files — some dependencies may be missing)")

    return "\n".join(lines)


# Git context cache (Phase 3b)
_git_context_cache = ""
_git_context_seeds = []  # modified files from git for seeding graph context


def _capture_git_context():
    """Capture git status/diff/log. Returns formatted string or '' if not a git repo."""
    global _git_context_cache, _git_context_seeds
    pdir = Path(CWD)
    if not (pdir / ".git").exists():
        _git_context_cache = ""
        _git_context_seeds = []
        return ""

    parts = []
    modified_files = []
    timestamp = time.strftime('%H:%M:%S')

    # Branch name
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, cwd=str(pdir)
        )
        if r.returncode == 0 and r.stdout.strip():
            parts.append(f"Branch: {r.stdout.strip()}")
    except Exception:
        pass

    # Modified/untracked files
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=str(pdir)
        )
        if r.returncode == 0 and r.stdout.strip():
            mod = []
            untracked = 0
            for line in r.stdout.strip().splitlines()[:20]:
                status = line[:2].strip()
                fname = line[3:].strip()
                if status == '??':
                    untracked += 1
                else:
                    mod.append(fname)
                    modified_files.append(fname)
            if mod:
                summary = ', '.join(mod[:6])
                if len(mod) > 6:
                    summary += f" (+{len(mod) - 6} more)"
                if untracked:
                    summary += f" (+{untracked} untracked)"
                parts.append(f"Modified: {summary}")
            elif untracked:
                parts.append(f"Untracked: {untracked} file(s)")
    except Exception:
        pass

    # Recent commits
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True, text=True, timeout=5, cwd=str(pdir)
        )
        if r.returncode == 0 and r.stdout.strip():
            commits = []
            for line in r.stdout.strip().splitlines()[:3]:
                msg = line.split(' ', 1)[1] if ' ' in line else line
                commits.append(f'"{msg.strip()}"')
            if commits:
                parts.append(f"Recent: {' → '.join(commits)}")
    except Exception:
        pass

    if not parts:
        _git_context_cache = ""
        _git_context_seeds = []
        return ""

    result = f"## Git Context (captured {timestamp})\n" + "\n".join(parts)
    _git_context_cache = result
    _git_context_seeds = modified_files
    return result


# ---------------------------------------------------------------------------
# auto-test runner (feature 5)
# ---------------------------------------------------------------------------

def _detect_test_framework(project_dir):
    """Detect which test framework is used and the command to run tests."""
    pdir = Path(project_dir)

    # Node: check package.json scripts
    pkg_path = pdir / "package.json"
    if pkg_path.exists():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                test_script = scripts["test"]
                if "jest" in test_script or "vitest" in test_script:
                    return "jest/vitest", "npm test"
                if "mocha" in test_script:
                    return "mocha", "npm test"
                if test_script != 'echo "Error: no test specified" && exit 1':
                    return "npm", "npm test"
        except Exception:
            pass
        # Check for test files
        if list(pdir.rglob("*.test.js")) or list(pdir.rglob("*.test.ts")) or list(pdir.rglob("*.spec.js")):
            return "jest", "npx jest"

    # Python: pytest or unittest
    if list(pdir.rglob("test_*.py")) or list(pdir.rglob("*_test.py")) or (pdir / "tests").is_dir():
        if (pdir / "pytest.ini").exists() or (pdir / "pyproject.toml").exists() or (pdir / "setup.cfg").exists():
            return "pytest", "python -m pytest -v"
        return "pytest", "python -m pytest -v"

    # Rust
    if (pdir / "Cargo.toml").exists():
        return "cargo", "cargo test"

    # Go
    if list(pdir.rglob("*_test.go")):
        return "go", "go test ./..."

    return None, None


def _run_tests(project_dir, model=None, messages=None):
    """Run detected tests and return (success, summary)."""
    framework, cmd = _detect_test_framework(project_dir)
    if not cmd:
        return True, "No test framework detected."

    print(f"  {C.TOOL}{BLACK_CIRCLE} Running tests: {framework} (`{cmd}`){C.RESET}")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=project_dir)
        output = (r.stdout or "") + "\n" + (r.stderr or "")
        if r.returncode == 0:
            print(f"    {VERIFY_OK} Tests passed")
            return True, f"Tests passed ({framework})"
        else:
            print(f"    {VERIFY_FAIL} Tests failed (exit {r.returncode})")
            # Feed failure to model if available
            if model and messages is not None:
                excerpt = output[-2000:]
                messages.append({"role": "user", "content":
                    f"Tests failed (`{cmd}`, exit {r.returncode}):\n```\n{excerpt}\n```\nFix the failing tests."})
                try:
                    run_agent_turn(messages, model, use_tools=True)
                except Exception:
                    pass
            return False, f"Tests failed ({framework})"
    except subprocess.TimeoutExpired:
        return False, "Tests timed out (120s)"
    except Exception as e:
        return False, f"Test error: {e}"


# ---------------------------------------------------------------------------
# multi-model routing (feature 6)
# ---------------------------------------------------------------------------

_SIMPLE_TASK_PATTERNS = re.compile(
    r'^(read_file|glob_search|grep_search|memory_search|ask_user)$'
)

def _should_use_small_model(tool_name):
    """Return True if this tool call can be handled by the fast small model."""
    return bool(_SIMPLE_TASK_PATTERNS.match(tool_name))

def _pick_model_for_task(messages, default_model):
    """Analyze the latest message and decide if we can use the small model."""
    if not messages:
        return default_model
    last = messages[-1].get("content", "")
    # Short questions, status checks, simple reads → small model
    if len(last) < 100 and any(w in last.lower() for w in ("what is", "show me", "list", "how many", "status")):
        return SMALL_MODEL
    return default_model


# ---------------------------------------------------------------------------
# watch mode (feature 7)
# ---------------------------------------------------------------------------

class FileWatcher:
    """Background file watcher that detects changes in the project."""

    def __init__(self, directory):
        self.directory = directory
        self._stop = threading.Event()
        self._thread = None
        self._file_hashes = {}
        self.changes = []  # list of (filepath, change_type)
        self._skip = {"node_modules", ".git", ".next", "__pycache__", "venv", ".venv", "dist", "build"}

    def _hash_file(self, fp):
        try:
            return fp.stat().st_mtime
        except Exception:
            return None

    def _scan(self):
        pdir = Path(self.directory)
        current = {}
        for root, dirs, files in os.walk(pdir):
            dirs[:] = [d for d in dirs if d not in self._skip]
            for fname in files:
                fp = Path(root) / fname
                rel = str(fp.relative_to(pdir))
                current[rel] = self._hash_file(fp)
        return current

    def _watch_loop(self):
        self._file_hashes = self._scan()
        while not self._stop.is_set():
            self._stop.wait(2.0)  # check every 2 seconds
            if self._stop.is_set():
                break
            new_hashes = self._scan()
            for path, mtime in new_hashes.items():
                if path not in self._file_hashes:
                    self.changes.append((path, "created"))
                elif self._file_hashes[path] != mtime:
                    self.changes.append((path, "modified"))
            for path in self._file_hashes:
                if path not in new_hashes:
                    self.changes.append((path, "deleted"))
            self._file_hashes = new_hashes
            # Update project graph incrementally if it exists
            if _project_graph is not None and self.changes:
                try:
                    _project_graph.update_incremental(self.changes)
                    _project_graph.save()
                except Exception:
                    pass

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def pop_changes(self):
        """Return and clear pending changes."""
        changes = self.changes[:]
        self.changes.clear()
        return changes

_file_watcher = None  # global watcher instance


# ---------------------------------------------------------------------------
# conversation export (feature 8)
# ---------------------------------------------------------------------------

def _export_conversation(messages, fmt="md"):
    """Export conversation to file. Returns filepath."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if fmt == "json":
        fp = SESSIONS_DIR / f"export_{ts}.json"
        fp.write_text(json.dumps(messages, indent=2, default=str), encoding="utf-8")
    else:
        fp = SESSIONS_DIR / f"export_{ts}.md"
        lines = [f"# Rattlesnake Session Export\n", f"*Exported: {time.strftime('%Y-%m-%d %H:%M')}*\n"]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue  # skip system prompt
            elif role == "user":
                lines.append(f"\n## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"\n## Rattlesnake\n\n{content}\n")
            elif role == "tool":
                lines.append(f"\n> **Tool result:**\n> {content[:500]}\n")
        fp.write_text("\n".join(lines), encoding="utf-8")
    return str(fp)


def _screenshot_qa(image_path, vision_model):
    """Send a screenshot to the vision model for UI/UX review."""
    fp = Path(image_path)
    if not fp.exists():
        return "Screenshot file not found."
    try:
        img_b64 = read_image_b64(str(fp))
        qa_messages = [
            {"role": "system", "content": "You are a UI/UX reviewer. Analyze this screenshot. Check: layout, colors, readability, spacing, missing elements, broken styling. Be specific and concise."},
            {"role": "user", "content": "Review this screenshot for visual quality issues.", "images": [img_b64]},
        ]
        result = ""
        for chunk in ollama_chat(qa_messages, vision_model, tools=None, stream=True):
            content = chunk.get("message", {}).get("content", "")
            if content:
                result += content
        return result if result else "Vision model returned no feedback."
    except Exception as e:
        return f"Screenshot QA failed: {e}"


# ---------------------------------------------------------------------------
# verification gate -- anti-hallucination for file writes
# ---------------------------------------------------------------------------

def _tool_error(tool_name, msg, hint=""):
    """Return a structured error message for tool failures."""
    error = f"Error [{tool_name}]: {msg}"
    if hint:
        error += f"\nHint: {hint}"
    return error


def _validate_file_syntax(filepath):
    """Validate file syntax by extension. Returns (valid, error_detail)."""
    fp = Path(filepath)
    if not fp.exists():
        return True, ""
    try:
        size = fp.stat().st_size
        if size > 500_000:  # skip files > 500KB
            return True, ""
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return True, ""  # can't read = skip validation

    ext = fp.suffix.lower()

    # Python: compile check
    if ext == ".py":
        try:
            compile(content, str(fp), "exec")
        except SyntaxError as e:
            return False, f"Python syntax error at line {e.lineno}: {e.msg}"

    # JSON: parse check
    elif ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"JSON parse error at line {e.lineno} col {e.colno}: {e.msg}"

    # JS/TS/JSX/TSX: balanced brackets check
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        # Remove string literals and comments to avoid false positives
        cleaned = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'`[^`]*`', '""', cleaned)
        cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', '""', cleaned)
        cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "''", cleaned)
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        line_num = 1
        for ch in cleaned:
            if ch == '\n':
                line_num += 1
            elif ch in '({[':
                stack.append((ch, line_num))
            elif ch in ')}]':
                if not stack:
                    return False, f"Unmatched '{ch}' at line {line_num}"
                top_ch, top_line = stack[-1]
                if top_ch != pairs[ch]:
                    return False, f"Mismatched '{pairs[ch]}' (line {top_line}) closed by '{ch}' at line {line_num}"
                stack.pop()
        if stack:
            unclosed_ch, unclosed_line = stack[-1]
            return False, f"Unclosed '{unclosed_ch}' opened at line {unclosed_line}"

    # HTML: basic tag matching
    elif ext in (".html", ".htm"):
        block_tags = {"div", "section", "article", "header", "footer", "nav", "main",
                      "aside", "form", "table", "thead", "tbody", "tr", "ul", "ol", "li",
                      "head", "body", "html", "script", "style", "p", "span", "a", "button"}
        tag_stack = []
        for m in re.finditer(r'<(/?)(\w+)[^>]*?(/?)>', content):
            is_close = m.group(1) == '/'
            tag_name = m.group(2).lower()
            is_self_close = m.group(3) == '/'
            if tag_name not in block_tags or is_self_close:
                continue
            if is_close:
                if tag_stack and tag_stack[-1] == tag_name:
                    tag_stack.pop()
            else:
                tag_stack.append(tag_name)
        if tag_stack:
            return False, f"Unclosed HTML tag(s): {', '.join(tag_stack)}"

    # CSS: balanced braces
    elif ext == ".css":
        depth = 0
        line_num = 1
        for ch in content:
            if ch == '\n':
                line_num += 1
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth < 0:
                    return False, f"Unmatched '}}' at line {line_num}"
        if depth > 0:
            return False, f"Unclosed '{{' ({depth} unclosed)"

    return True, ""


def _fix_json_syntax(content: str) -> tuple:
    """
    Attempt to repair common JSON corruption from LLM output.
    Returns (fixed_content, was_fixed). Never makes valid JSON invalid.
    """
    original = content
    fixed = content

    # Strip BOM
    if fixed.startswith('\ufeff'):
        fixed = fixed[1:]

    # Remove single-line JS comments (// ...)
    fixed = re.sub(r'(?m)^\s*//.*$', '', fixed)
    # Remove inline // comments (but not inside strings)
    fixed = re.sub(r'(?<=[,{\[\]\d"true"false"null])\s*//[^\n]*', '', fixed)
    # Remove multi-line JS comments (/* ... */)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)

    # Remove trailing commas before } or ]
    fixed = re.sub(r',\s*([\]}])', r'\1', fixed)

    # Truncate trailing garbage after root object closes (balanced-brace scan)
    depth = 0
    root_end = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(fixed):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{' or ch == '[':
            depth += 1
        elif ch == '}' or ch == ']':
            depth -= 1
            if depth == 0:
                root_end = i + 1
                break
    if root_end and root_end < len(fixed.rstrip()):
        fixed = fixed[:root_end].rstrip() + '\n'

    # Guard: only return fixed content if it actually parses
    try:
        json.loads(fixed)
        was_fixed = (fixed.strip() != original.strip())
        return fixed, was_fixed
    except json.JSONDecodeError:
        return content, False


def _scan_for_incomplete_code(filepath):
    """Scan a file for incomplete/placeholder code. Returns list of issue strings."""
    fp = Path(filepath)
    if not fp.exists():
        return []
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    issues = []
    ext = fp.suffix.lower()
    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        lower = stripped.lower()

        # All files: TODO/FIXME comments
        if re.search(r'(//|#|/\*|\*)\s*(TODO|FIXME|HACK|XXX)\b', stripped, re.IGNORECASE):
            issues.append(f"Line {i}: TODO/FIXME comment: {stripped[:80]}")

        # All files: placeholder text
        if '"placeholder"' in lower or "'placeholder'" in lower:
            issues.append(f"Line {i}: placeholder text")
        if '"not implemented"' in lower or "'not implemented'" in lower:
            issues.append(f"Line {i}: 'Not implemented' text")
        if 'lorem ipsum' in lower:
            issues.append(f"Line {i}: Lorem ipsum placeholder")

        # Python-specific
        if ext == ".py":
            if stripped == "pass":
                # Check if it's in an except block or abstract method — those are OK
                is_except = False
                is_abstract = False
                for j in range(max(0, i - 3), i - 1):
                    prev = lines[j].strip()
                    if prev.startswith("except") or prev.startswith("except:"):
                        is_except = True
                    if "@abstractmethod" in prev or "raise NotImplementedError" in prev:
                        is_abstract = True
                if not is_except and not is_abstract:
                    issues.append(f"Line {i}: empty function body (pass)")
            # FastAPI sync endpoint (def instead of async def when fastapi imported)
            if 'fastapi' in content.lower():
                if re.match(r'@(app|router)\.(get|post|put|patch|delete)', stripped):
                    # Look ahead for sync def
                    if i < len(lines):
                        next_line = lines[i].strip() if i < len(lines) else ""
                        if next_line.startswith("def ") and not next_line.startswith("async def"):
                            issues.append(f"Line {i+1}: sync endpoint in FastAPI — use async def")
            # Hardcoded secret patterns
            if re.search(r'(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*["\'][^"\']{8,}', stripped, re.IGNORECASE):
                if not stripped.startswith('#') and 'os.environ' not in stripped and 'os.getenv' not in stripped:
                    issues.append(f"Line {i}: possible hardcoded secret")

        # JS/TS-specific
        if ext in (".js", ".ts", ".jsx", ".tsx"):
            if 'throw new Error("Not implemented")' in stripped or \
               "throw new Error('Not implemented')" in stripped:
                issues.append(f"Line {i}: throw 'Not implemented'")
            if 'console.log("placeholder")' in lower or \
               "console.log('placeholder')" in lower:
                issues.append(f"Line {i}: console.log placeholder")
            # Standalone ... (not spread operator like ...arr)
            if stripped == "..." or stripped == "...;":
                issues.append(f"Line {i}: standalone ellipsis (incomplete code)")
            # Client component missing 'use client' (scoped: .tsx/.jsx with JSX return + hooks)
            if ext in (".tsx", ".jsx") and i == 1:
                has_jsx_return = bool(re.search(r'return\s*\(?\s*<', content))
                has_hooks = bool(re.search(r'\buse[A-Z]\w+\s*\(', content))
                has_use_client = "'use client'" in content or '"use client"' in content
                has_next = 'from "next' in content or "from 'next" in content
                if has_jsx_return and has_hooks and not has_use_client and has_next:
                    issues.append(f"Line 1: Missing 'use client' directive — component uses hooks with JSX")
            # One-shot framer-motion animate={{}} without state management
            if 'animate={{' in stripped and 'framer-motion' in content:
                if 'key=' not in content and 'AnimatePresence' not in content and 'useAnimate' not in content:
                    issues.append(f"Line {i}: one-shot framer-motion animate={{{{}}}} — needs key/AnimatePresence/useAnimate")

        # HTML-specific
        if ext in (".html", ".htm"):
            if '"your content here"' in lower or "'your content here'" in lower:
                issues.append(f"Line {i}: 'Your content here' placeholder")
            if '"sample text"' in lower or "'sample text'" in lower:
                issues.append(f"Line {i}: 'Sample text' placeholder")

    # All files: unused import detection (top 3 only)
    unused_imports = []
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        for m in re.finditer(r'import\s+(?:\{([^}]+)\}|(\w+))', content):
            names = []
            if m.group(1):
                names = [n.strip().split(' as ')[-1].strip() for n in m.group(1).split(',')]
            elif m.group(2) and m.group(2) not in ('type', 'from'):
                names = [m.group(2)]
            rest_of_file = content[m.end():]
            import_line = content[:m.start()].count("\n") + 1
            for name in names:
                if name and len(name) > 1 and not re.search(r'\b' + re.escape(name) + r'\b', rest_of_file):
                    unused_imports.append(f"Line {import_line}: unused import '{name}'")
    elif ext == ".py":
        for m in re.finditer(r'^(?:from\s+\S+\s+)?import\s+(.+)$', content, re.MULTILINE):
            imports_str = m.group(1)
            import_line = content[:m.start()].count("\n") + 1
            for name in imports_str.split(','):
                name = name.strip().split(' as ')[-1].strip()
                if name and len(name) > 1:
                    rest_of_file = content[m.end():]
                    if not re.search(r'\b' + re.escape(name) + r'\b', rest_of_file):
                        unused_imports.append(f"Line {import_line}: unused import '{name}'")
    issues.extend(unused_imports[:3])

    return issues[:20]  # cap at 20 issues per file


def _auto_lint(filepath):
    """Run the project's linter on a file if available. Returns (ok, output) or (None, None)."""
    fp = Path(filepath)
    ext = fp.suffix.lower()
    try:
        if ext == ".py":
            # Try ruff (fast), then flake8
            for linter in ["ruff check", "flake8"]:
                if shutil.which(linter.split()[0]):
                    r = subprocess.run(f"{linter} \"{fp}\"", shell=True, capture_output=True,
                                       text=True, timeout=15, cwd=CWD)
                    return r.returncode == 0, (r.stdout + r.stderr).strip()[:500]
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            eslint = Path(CWD) / "node_modules" / ".bin" / ("eslint.cmd" if sys.platform == "win32" else "eslint")
            if eslint.exists():
                r = subprocess.run(f"\"{eslint}\" --no-error-on-unmatched-pattern \"{fp}\"",
                                   shell=True, capture_output=True, text=True, timeout=15, cwd=CWD)
                return r.returncode == 0, (r.stdout + r.stderr).strip()[:500]
        elif ext == ".css":
            stylelint = Path(CWD) / "node_modules" / ".bin" / ("stylelint.cmd" if sys.platform == "win32" else "stylelint")
            if stylelint.exists():
                r = subprocess.run(f"\"{stylelint}\" \"{fp}\"", shell=True, capture_output=True,
                                   text=True, timeout=15, cwd=CWD)
                return r.returncode == 0, (r.stdout + r.stderr).strip()[:500]
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None, None


def _check_and_install_imports(filepath):
    """For Python files, detect imports and auto-install missing packages."""
    fp = Path(filepath)
    if fp.suffix != ".py":
        return []
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    imports = set()
    for m in re.finditer(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE):
        imports.add(m.group(1))

    # Standard library modules (Python 3.x common)
    stdlib = {
        "os", "sys", "json", "re", "time", "datetime", "pathlib", "collections",
        "itertools", "functools", "typing", "abc", "math", "random", "hashlib",
        "base64", "io", "copy", "string", "textwrap", "logging", "unittest",
        "subprocess", "threading", "multiprocessing", "socket", "http", "urllib",
        "csv", "xml", "html", "email", "argparse", "configparser", "sqlite3",
        "shutil", "glob", "tempfile", "pickle", "struct", "dataclasses", "enum",
        "contextlib", "inspect", "traceback", "warnings", "signal", "platform",
        "decimal", "fractions", "statistics", "secrets", "uuid", "pprint",
        "importlib", "ast", "dis", "token", "tokenize", "types", "weakref",
        "array", "queue", "heapq", "bisect", "operator", "codecs", "zlib",
        "gzip", "bz2", "lzma", "zipfile", "tarfile", "shelve", "dbm",
    }

    # Common import→pip name mappings
    pip_names = {
        "cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn",
        "yaml": "PyYAML", "bs4": "beautifulsoup4", "dotenv": "python-dotenv",
        "jwt": "PyJWT", "Crypto": "pycryptodome", "gi": "PyGObject",
        "attr": "attrs", "dateutil": "python-dateutil",
    }

    third_party = imports - stdlib
    installed = []
    for pkg in third_party:
        pip_name = pip_names.get(pkg, pkg)
        try:
            __import__(pkg)
        except ImportError:
            r = subprocess.run(f"pip install {pip_name}", shell=True, capture_output=True,
                               text=True, timeout=60, cwd=CWD)
            if r.returncode == 0:
                installed.append(pip_name)
    return installed


# ---------------------------------------------------------------------------
# Lead-to-Gold Engine — post-generation HTML enhancement
# ---------------------------------------------------------------------------

_TAILWIND_V3_CDN = '<script src="https://cdn.tailwindcss.com"></script>'

_INTER_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
)

_TAILWIND_CONFIG_BLOCK = """<script>
tailwind.config = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: { 50:'#ecfdf5',100:'#d1fae5',200:'#a7f3d0',300:'#6ee7b7',400:'#34d399',500:'#10b981',600:'#059669',700:'#047857',800:'#065f46',900:'#064e3b' },
      },
      fontFamily: { sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'] },
    },
  },
}
</script>"""

_GLASS_CSS = """<style>
  :root{--bg-primary:#09090b;--bg-secondary:#18181b;--bg-tertiary:#27272a;--surface:#1c1c1f;--surface-hover:#252529;--border:#2e2e33;--border-subtle:#232328;--text-primary:#fafafa;--text-secondary:#a1a1aa;--text-tertiary:#71717a;--text-muted:#52525b;--accent:#6366f1;--accent-hover:#818cf8;--accent-muted:rgba(99,102,241,0.12);--success:#22c55e;--warning:#f59e0b;--error:#ef4444;--info:#3b82f6}
  .glass{backdrop-filter:blur(16px) saturate(180%);-webkit-backdrop-filter:blur(16px) saturate(180%);background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:1rem}
  @keyframes fade-up{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
  @keyframes slide-in{from{opacity:0;transform:translateX(-12px)}to{opacity:1;transform:translateX(0)}}
  @keyframes scale-in{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
  .animate-fade-up{animation:fade-up .5s ease-out both}
  .animate-slide-in{animation:slide-in .4s ease-out both}
  .animate-scale-in{animation:scale-in .3s ease-out both}
  [data-animate]{opacity:0;transform:translateY(12px);transition:opacity .6s ease-out,transform .6s ease-out}
  [data-animate].visible{opacity:1;transform:translateY(0)}
</style>"""

_SCROLL_REVEAL_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded',function(){
  var io=new IntersectionObserver(function(entries){entries.forEach(function(e){if(e.isIntersecting){e.target.classList.add('visible');io.unobserve(e.target)}})},{threshold:0.1});
  document.querySelectorAll('[data-animate]').forEach(function(el){io.observe(el)});
});
</script>"""

_DARK_TOGGLE_HTML = """<!-- dark mode toggle -->
<button onclick="document.documentElement.classList.toggle('dark');localStorage.setItem('dm',document.documentElement.classList.contains('dark'))" class="fixed top-4 right-4 z-50 p-2.5 rounded-xl bg-white/80 dark:bg-gray-800/80 backdrop-blur border border-gray-200 dark:border-gray-700 shadow-lg hover:scale-105 active:scale-95 transition-all duration-200" aria-label="Toggle dark mode">
  <svg class="w-5 h-5 text-gray-700 dark:text-gray-300 hidden dark:block" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 3v1m0 16v1m8.66-13.66l-.71.71M4.05 19.95l-.71.71M21 12h-1M4 12H3m16.66 7.66l-.71-.71M4.05 4.05l-.71-.71M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
  <svg class="w-5 h-5 text-gray-700 dark:text-gray-300 block dark:hidden" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z"/></svg>
</button>
<script>if(localStorage.getItem('dm')==='true')document.documentElement.classList.add('dark');</script>"""

_INPUT_CLASSES = "w-full px-3 py-2 rounded-md text-sm bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)] transition-colors duration-150"
_BUTTON_CLASSES = "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg-primary)] disabled:opacity-50 disabled:cursor-not-allowed"
_SELECT_CLASSES = "w-full px-3 py-2 rounded-md text-sm bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)] transition-colors duration-150 appearance-none cursor-pointer"
_TEXTAREA_CLASSES = "w-full px-3 py-2 rounded-md text-sm bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)] transition-colors duration-150 resize-y min-h-[100px]"

_FAVICON_DATA_URI = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect rx='20' width='100' height='100' fill='%23059669'/><path d='M30 55l15 15 25-30' stroke='white' stroke-width='8' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>"


def _is_unstyled(tag_match):
    """Check if an HTML element lacks bg-, text-color, AND padding utilities."""
    cls_m = re.search(r'class\s*=\s*"([^"]*)"', tag_match)
    if not cls_m:
        return True
    classes = cls_m.group(1)
    has_bg = bool(re.search(r'\bbg-', classes))
    has_text_color = bool(re.search(r'\btext-(white|black|gray|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|slate|zinc|neutral|stone)\b', classes))
    has_padding = bool(re.search(r'\b(p-|px-|py-|pl-|pr-|pt-|pb-)', classes))
    return not (has_bg and has_text_color and has_padding)


def _inject_classes(tag_html, new_classes):
    """Merge new classes into an existing tag's class attribute, or add one."""
    cls_m = re.search(r'class\s*=\s*"([^"]*)"', tag_html)
    if cls_m:
        existing = cls_m.group(1).strip()
        merged = existing + " " + new_classes if existing else new_classes
        return tag_html[:cls_m.start(1)] + merged + tag_html[cls_m.end(1):]
    # No class attribute — insert before the closing >
    insert_pos = tag_html.rfind('>')
    if tag_html[insert_pos - 1] == '/':
        insert_pos -= 1
    return tag_html[:insert_pos] + f' class="{new_classes}"' + tag_html[insert_pos:]


def _count_unique_tw_classes(html):
    """Count unique Tailwind utility classes in the HTML."""
    all_classes = re.findall(r'class\s*=\s*"([^"]*)"', html)
    unique = set()
    for cls_str in all_classes:
        for c in cls_str.split():
            if re.match(r'^(bg-|text-|p-|px-|py-|m-|mx-|my-|flex|grid|rounded|shadow|border|w-|h-|min-|max-|font-|tracking-|leading-|transition|duration|hover:|dark:|focus:|active:)', c):
                unique.add(c)
    return len(unique)


def _enhance_html(raw_html):
    """
    Lead-to-Gold Engine: post-process raw model HTML into production quality.
    Transforms CDN, fonts, config, unstyled elements, invisible text,
    base styles, meta tags, animations, and spacing.
    """
    # --- Escape hatches ---
    if '<!-- no-enhance -->' in raw_html:
        return raw_html
    if len(raw_html) > 200_000:
        return raw_html
    if _count_unique_tw_classes(raw_html) >= 20:
        return raw_html

    html = raw_html

    # 1a. Fix CDN — replace old Tailwind v2 or missing CDN
    html = re.sub(
        r'<(?:link|script)[^>]*unpkg\.com/tailwindcss@[^>]*>',
        _TAILWIND_V3_CDN, html
    )
    html = re.sub(
        r'<script[^>]*cdn\.tailwindcss\.com/[^>]*v=2[^>]*>[^<]*</script>',
        _TAILWIND_V3_CDN, html
    )
    if 'cdn.tailwindcss.com' not in html and 'tailwindcss' not in html.lower():
        # Inject into <head>
        if '<head>' in html:
            html = html.replace('<head>', '<head>\n    ' + _TAILWIND_V3_CDN, 1)
        elif '<head ' in html:
            head_close = html.find('>', html.find('<head '))
            if head_close > 0:
                html = html[:head_close + 1] + '\n    ' + _TAILWIND_V3_CDN + html[head_close + 1:]
        else:
            html = _TAILWIND_V3_CDN + '\n' + html

    # 1b. Inject Inter font if missing
    if 'fonts.googleapis.com' not in html or 'Inter' not in html:
        if '<head>' in html:
            html = html.replace('<head>', '<head>\n    ' + _INTER_FONT_LINKS, 1)
        elif '</head>' in html:
            html = html.replace('</head>', '    ' + _INTER_FONT_LINKS + '\n</head>', 1)
    # Add font-family to body if not present
    if "font-family" not in html and "'Inter'" not in html:
        body_m = re.search(r'<body([^>]*)>', html)
        if body_m:
            style_attr = body_m.group(0)
            if 'style=' in style_attr:
                html = html.replace(style_attr, style_attr.replace('style="', "style=\"font-family:'Inter',system-ui,sans-serif;"))
            else:
                html = html.replace(body_m.group(0), f'<body{body_m.group(1)} style="font-family:\'Inter\',system-ui,sans-serif;">')

    # 1c. Inject Tailwind config for custom colors
    if 'cdn.tailwindcss.com' in html and 'tailwind.config' not in html:
        html = html.replace(_TAILWIND_V3_CDN, _TAILWIND_V3_CDN + '\n    ' + _TAILWIND_CONFIG_BLOCK)

    # 1d. Upgrade unstyled HTML elements
    for tag, classes in [('input', _INPUT_CLASSES), ('select', _SELECT_CLASSES), ('textarea', _TEXTAREA_CLASSES)]:
        def _upgrade_tag(m, _cls=classes):
            if _is_unstyled(m.group(0)):
                return _inject_classes(m.group(0), _cls)
            return m.group(0)
        html = re.sub(rf'<{tag}\b[^>]*>', _upgrade_tag, html)

    # Buttons
    def _upgrade_button(m):
        if _is_unstyled(m.group(0)):
            return _inject_classes(m.group(0), _BUTTON_CLASSES)
        return m.group(0)
    html = re.sub(r'<button\b[^>]*>', _upgrade_button, html)

    # 1e. Fix invisible text (white on white)
    def _fix_invisible_button(m):
        tag = m.group(0)
        cls_m = re.search(r'class\s*=\s*"([^"]*)"', tag)
        if not cls_m:
            return tag
        classes = cls_m.group(1)
        if 'text-white' in classes and not re.search(r'\bbg-', classes):
            return _inject_classes(tag, 'bg-emerald-600')
        return tag
    html = re.sub(r'<button\b[^>]*>', _fix_invisible_button, html)

    # Add default text color to body if missing
    body_m = re.search(r'<body\b[^>]*>', html)
    if body_m:
        body_tag = body_m.group(0)
        body_cls = re.search(r'class\s*=\s*"([^"]*)"', body_tag)
        body_classes = body_cls.group(1) if body_cls else ""
        if not re.search(r'\btext-(gray|white|black|slate|zinc|neutral|stone)', body_classes):
            html = html.replace(body_tag, _inject_classes(body_tag, 'text-gray-900 dark:text-gray-100'))

    # 1f. Inject base styles if missing
    if 'backdrop-filter' not in html and '@keyframes' not in html:
        if '</head>' in html:
            html = html.replace('</head>', _GLASS_CSS + '\n</head>', 1)
        elif '</style>' in html:
            html = html.replace('</style>', '</style>\n' + _GLASS_CSS, 1)

    # Dark mode on body
    body_m = re.search(r'<body\b[^>]*>', html)
    if body_m:
        body_tag = body_m.group(0)
        body_cls = re.search(r'class\s*=\s*"([^"]*)"', body_tag)
        body_classes = body_cls.group(1) if body_cls else ""
        if 'dark:bg-' not in body_classes:
            html = html.replace(body_tag, _inject_classes(body_tag, 'dark:bg-gray-950'))
        if 'transition' not in body_classes:
            html = html.replace(
                re.search(r'<body\b[^>]*>', html).group(0),
                _inject_classes(re.search(r'<body\b[^>]*>', html).group(0), 'transition-colors duration-200')
            )

    # 1g. Inject meta tags, favicon, OG data
    head_inject = ""
    if '<meta name="viewport"' not in html and "<meta name='viewport'" not in html:
        head_inject += '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    if '<link rel="icon"' not in html and "<link rel='icon'" not in html:
        head_inject += f'    <link rel="icon" href="{_FAVICON_DATA_URI}">\n'
    if '<meta name="theme-color"' not in html:
        head_inject += '    <meta name="theme-color" content="#059669">\n'

    # Security: CSP meta tag for static HTML (the only security header that works as meta)
    # X-Frame-Options and X-Content-Type-Options don't work reliably as meta tags — they're HTTP response headers. CSP is the exception.
    if '<meta http-equiv="Content-Security-Policy"' not in html:
        head_inject += "    <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; script-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com;\">\n"
    if '<meta name="referrer"' not in html:
        head_inject += '    <meta name="referrer" content="strict-origin-when-cross-origin">\n'

    # OG tags from <title>
    title_m = re.search(r'<title>([^<]+)</title>', html)
    if title_m and '<meta property="og:title"' not in html:
        t = title_m.group(1).strip()
        head_inject += f'    <meta property="og:title" content="{t}">\n'
        head_inject += f'    <meta property="og:description" content="{t}">\n'

    if head_inject:
        if '</head>' in html:
            html = html.replace('</head>', head_inject + '</head>', 1)
        elif '<head>' in html:
            html = html.replace('<head>', '<head>\n' + head_inject, 1)

    # 1h. Inject scroll animations + dark mode toggle
    if 'IntersectionObserver' not in html:
        # Add data-animate to sections
        def _add_data_animate(m):
            tag = m.group(0)
            if 'data-animate' not in tag:
                return tag.replace('<section', '<section data-animate', 1)
            return tag
        html = re.sub(r'<section\b[^>]*>', _add_data_animate, html)
        if '</body>' in html:
            html = html.replace('</body>', _SCROLL_REVEAL_SCRIPT + '\n</body>', 1)

    # Dark mode toggle
    if 'Toggle dark mode' not in html and 'dark-mode-toggle' not in html:
        if '</body>' in html:
            html = html.replace('</body>', _DARK_TOGGLE_HTML + '\n</body>', 1)

    # 1i. Fix spacing
    body_m = re.search(r'<body\b[^>]*>', html)
    if body_m:
        body_tag = body_m.group(0)
        body_cls = re.search(r'class\s*=\s*"([^"]*)"', body_tag)
        body_classes = body_cls.group(1) if body_cls else ""
        if 'min-h-screen' not in body_classes:
            html = html.replace(body_tag, _inject_classes(body_tag, 'min-h-screen'))

    return html


def _inject_design_snippets():
    """Return a concise design cheat-sheet for the system prompt — copy-pasteable components."""
    return dedent("""\

        ## HTML Design Cheat Sheet (copy-paste these EXACTLY)

        **Tailwind v3 CDN (ALWAYS use this, never v2):**
        ```
        <script src="https://cdn.tailwindcss.com"></script>
        ```

        **Inter font (ALWAYS include):**
        ```
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        ```

        **Styled input with label:**
        ```html
        <div class="space-y-2">
          <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Email</label>
          <input class="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500 transition-all" type="email" placeholder="you@example.com" />
        </div>
        ```

        **Button system (primary + secondary):**
        ```html
        <button class="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white text-sm font-semibold rounded-xl transition-all duration-200 shadow-lg shadow-emerald-500/25">Primary</button>
        <button class="px-5 py-2.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm font-medium rounded-xl transition-all">Secondary</button>
        ```

        **Glass card wrapper:**
        ```html
        <div class="backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-8 shadow-xl">
          <!-- card content -->
        </div>
        ```
    """)


def _scrub_placeholders(filepath):
    """Scrub placeholder text from ANY file type. Called by _verify_file_write()."""
    fp = Path(filepath)
    if not fp.exists() or fp.stat().st_size == 0:
        return False
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    original = content

    # Domain-aware brand name
    _domain = _infer_project_domain()
    _brand = _domain["brand"] if _domain else "Nexus Labs"

    # Universal patterns (all file types)
    _universal_replacements = [
        (r'Lorem ipsum[^"\'<\n]*', "We build tools that help teams move faster, communicate better, and ship with confidence."),
        (r'Your Company', _brand),
        (r'Company Name', _brand),
        (r'John Doe', 'Alex Morgan'),
        (r'Jane Doe', 'Sarah Chen'),
        (r'John Smith', 'Alex Morgan'),
        (r'Jane Smith', 'Sarah Chen'),
        (r'user@example\.com', 'hello@nexuslabs.io'),
        (r'contact@example\.com', 'team@nexuslabs.io'),
        (r'test@test\.com', 'demo@nexuslabs.io'),
        (r'foo@bar\.com', 'info@nexuslabs.io'),
        (r'example@email\.com', 'hello@nexuslabs.io'),
        (r'email@example\.com', 'hello@nexuslabs.io'),
        (r'\+?1?[\s-]?\(?123\)?[\s-]?456[\s-]?7890', '+1 (555) 932-4178'),
        (r'123 (?:Main|Fake|Test) St(?:reet)?', '742 Evergreen Ave, Suite 200'),
    ]
    for pattern, replacement in _universal_replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    # Python-specific
    if fp.suffix == ".py":
        content = re.sub(r'#\s*TODO:?\s*implement', '# Implementation', content)

    # JS/TS-specific
    if fp.suffix in (".js", ".ts", ".jsx", ".tsx"):
        content = re.sub(r'console\.log\(["\']placeholder["\']\)', '', content)

    changed = content != original
    if changed:
        try:
            fp.write_text(content, encoding="utf-8")
        except Exception:
            return False
    return changed


# ---------------------------------------------------------------------------
# quality gate — score generated code and block slop
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS = {
    "restaurant": {"keywords": ["restaurant", "menu", "reservation", "dining", "food"],
                    "brand": "The Copper Table", "tone": "warm, inviting"},
    "saas": {"keywords": ["saas", "subscription", "pricing", "tier", "plan"],
             "brand": "Nexus Labs", "tone": "professional, modern"},
    "ecommerce": {"keywords": ["shop", "cart", "product", "checkout", "store"],
                   "brand": "Haven Supply Co.", "tone": "clean, trustworthy"},
    "healthcare": {"keywords": ["patient", "clinic", "appointment", "health", "medical"],
                    "brand": "Meridian Health", "tone": "calm, authoritative"},
    "education": {"keywords": ["course", "student", "lesson", "learn", "teacher"],
                   "brand": "Bright Path Academy", "tone": "encouraging, clear"},
}


def _infer_project_domain():
    """Scan PLAN.md, README.md, package.json for domain keywords."""
    texts = []
    for fname in ["PLAN.md", "README.md", "package.json"]:
        fp = Path(CWD) / fname
        if fp.exists():
            try:
                texts.append(fp.read_text(encoding="utf-8", errors="replace")[:2000].lower())
            except Exception:
                pass
    combined = " ".join(texts)
    if not combined:
        return None
    best_domain = None
    best_score = 0
    for domain, info in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in info["keywords"] if kw in combined)
        if score > best_score:
            best_score = score
            best_domain = domain
    if best_score >= 2:
        return _DOMAIN_KEYWORDS[best_domain]
    return None


# --- Boilerplate fingerprinting ---

_TUTORIAL_FINGERPRINTS = {
    "flask-hello": {"signals": ["from flask import", "app = Flask", '@app.route("/")', "Hello"],
                     "threshold": 3},
    "express-hello": {"signals": ["require('express')", "app.get('/'", "app.listen("],
                       "threshold": 3},
    "react-counter": {"signals": ["useState(0)", "setCount", "count +", "onClick"],
                       "threshold": 3},
    "generic-todo": {"signals": ["addTodo", "removeTodo", "todoList", "TodoItem"],
                      "threshold": 3},
    "generic-crud": {"signals": ["getAll", "getById", "create", "update", "delete"],
                      "threshold": 4},
}

_GENERIC_FUNCTION_NAMES = {"handleClick", "handleSubmit", "handleChange", "getData",
                            "fetchData", "processData", "handleError", "doSomething",
                            "init", "setup", "render", "update", "create", "delete"}


def _boilerplate_fingerprint(content, filepath):
    """Check if content matches known tutorial/boilerplate patterns.
    Returns (is_boilerplate, matched_pattern, generic_name_ratio)."""
    content_lower = content.lower()
    for pattern_name, fp_info in _TUTORIAL_FINGERPRINTS.items():
        matches = sum(1 for sig in fp_info["signals"] if sig.lower() in content_lower)
        if matches >= fp_info["threshold"]:
            # Count generic function names
            found_generic = sum(1 for name in _GENERIC_FUNCTION_NAMES if name in content)
            func_defs = content.count("def ") + content.count("function ") + content.count("const ") + content.count("=> ")
            ratio = found_generic / max(func_defs, 1)
            return (True, pattern_name, ratio)
    # Even without pattern match, check generic name ratio
    found_generic = sum(1 for name in _GENERIC_FUNCTION_NAMES if name in content)
    func_defs = content.count("def ") + content.count("function ") + content.count("const ") + content.count("=> ")
    ratio = found_generic / max(func_defs, 1)
    return (False, None, ratio)


# --- Language-specific idiom checkers ---

def _check_python_idioms(content):
    """Check Python code for anti-patterns. Returns list of (line_num, issue)."""
    issues = []
    lines = content.split("\n")
    has_fastapi = 'fastapi' in content.lower() or 'from fastapi' in content
    os_environ_count = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # .format() string formatting -> suggest f-strings
        if '.format(' in stripped and not stripped.startswith('#'):
            issues.append((i, "Use f-strings instead of .format()"))
        # os.path.* -> suggest pathlib
        if 'os.path.' in stripped and not stripped.startswith('#'):
            issues.append((i, "Consider using pathlib instead of os.path"))
        # Bare except:
        if stripped == "except:" or stripped.startswith("except: "):
            issues.append((i, "Bare except — specify exception type"))
        # type() comparisons
        if re.search(r'type\(.+\)\s*==', stripped):
            issues.append((i, "Use isinstance() instead of type() comparison"))
        # Count os.environ usage
        if 'os.environ' in stripped or 'os.getenv' in stripped:
            os_environ_count += 1
        # Star imports
        if re.match(r'from\s+\S+\s+import\s+\*', stripped):
            issues.append((i, "Star import (from X import *) — import specific names"))
        # Mutable default arguments
        if re.search(r'def\s+\w+\(.*=\s*(\[\]|\{\}|set\(\))', stripped):
            issues.append((i, "Mutable default argument — use None and assign in body"))
        # Global variable mutation
        if re.match(r'global\s+\w+', stripped):
            issues.append((i, "Global variable mutation — use function parameters or class attributes"))
        # Hardcoded localhost URLs
        if not stripped.startswith('#'):
            if re.search(r'["\']https?://(localhost|127\.0\.0\.1)', stripped):
                if 'os.environ' not in stripped and 'os.getenv' not in stripped and 'settings' not in stripped.lower():
                    issues.append((i, "Hardcoded localhost URL — use environment variable"))
    # Empty function bodies (just pass)
    for m in re.finditer(r'def \w+\([^)]*\):\s*\n\s+pass\s*$', content, re.MULTILINE):
        lineno = content[:m.start()].count("\n") + 1
        issues.append((lineno, "Empty function body (just pass)"))

    # --- Universal Python checks ---

    # A. Sync def in FastAPI/async context (def endpoint when async def expected)
    if has_fastapi:
        for m in re.finditer(r'@(app|router)\.(get|post|put|patch|delete)\b', content):
            rest = content[m.end():]
            def_match = re.search(r'\ndef\s+', rest)
            async_match = re.search(r'\nasync\s+def\s+', rest)
            if def_match:
                if not async_match or def_match.start() < async_match.start():
                    lineno = content[:m.end() + def_match.start()].count("\n") + 2
                    issues.append((lineno, "Sync endpoint in FastAPI — use async def for I/O-bound endpoints"))

    # B. print() for logging (should use logging module) — only in non-script files
    has_main_guard = 'if __name__' in content
    if not has_main_guard or has_fastapi:
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.match(r'print\s*\(', stripped) and not stripped.startswith('#'):
                issues.append((i, "Use logging module instead of print() for production code"))

    # C. Hardcoded secrets (API_KEY = "sk-...", PASSWORD = "...")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.search(r'(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*["\'][^"\']{8,}', stripped, re.IGNORECASE):
            if not stripped.startswith('#') and 'os.environ' not in stripped and 'os.getenv' not in stripped and 'settings.' not in stripped.lower():
                issues.append((i, "Possible hardcoded secret — use environment variables"))

    # D. Missing if __name__ == "__main__" guard in entry point files
    if 'uvicorn.run' in content and 'if __name__' not in content:
        issues.append((0, "Missing if __name__ == '__main__' guard around uvicorn.run"))

    # E. Raw os.environ instead of pydantic Settings or python-dotenv (3+ calls)
    if os_environ_count >= 3:
        issues.append((0, f"Excessive raw os.environ usage ({os_environ_count} calls) — use Pydantic Settings or python-dotenv"))

    # F. Long functions (50+ lines)
    func_starts = []
    for i, line in enumerate(lines):
        if re.match(r'\s*(async\s+)?def\s+', line):
            func_starts.append(i)
    for idx, start in enumerate(func_starts):
        end = func_starts[idx + 1] if idx + 1 < len(func_starts) else len(lines)
        func_len = end - start
        if func_len > 50:
            issues.append((start + 1, f"Function is {func_len} lines long — consider breaking into smaller functions"))

    # G. Missing type hints on public functions
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r'def\s+[a-z]\w*\(', stripped) and not stripped.startswith('def _'):
            if '->' not in stripped and ') :' not in stripped:
                # Check if it has parameter type hints at least
                if ':' not in stripped.split('(')[1].split(')')[0] if ')' in stripped else True:
                    issues.append((i, "Public function missing type hints — add parameter and return types"))

    # H. except Exception with just pass
    for m in re.finditer(r'except\s+\w+.*:\s*\n\s+pass\s*$', content, re.MULTILINE):
        lineno = content[:m.start()].count("\n") + 1
        issues.append((lineno, "Exception caught and silently ignored (except + pass) — log or handle"))

    return issues


def _check_js_idioms(content, filepath=""):
    """Check JS/TS code for anti-patterns. Returns list of (line_num, issue)."""
    issues = []
    lines = content.split("\n")
    has_jsx_return = bool(re.search(r'return\s*\(?\s*<', content))
    # Shared SSR detection (used by Fixes 5 and 6)
    has_use_client = "'use client'" in content or '"use client"' in content
    filepath_fwd = str(filepath).replace('\\', '/')
    is_ssr_path = bool(filepath_fwd) and '/app/' in filepath_fwd
    ext = Path(filepath).suffix.lower() if filepath else ''
    has_next_import = 'from "next' in content or "from 'next" in content or 'from "next/' in content or "from 'next/" in content
    is_test_file = bool(re.search(r'\.(test|spec)\.[jt]sx?$', str(Path(filepath).name) if filepath else ''))
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # var declarations
        if re.match(r'\bvar\s+', stripped):
            issues.append((i, "Use const/let instead of var"))
        # == comparisons (not ===)
        if re.search(r'[^!=]==[^=]', stripped) and not stripped.startswith('//'):
            issues.append((i, "Use === instead of =="))
        # React.Component class
        if 'extends React.Component' in stripped:
            issues.append((i, "Use functional components instead of class components"))
        # console.log in production code (not test files, not debug comments)
        if not is_test_file and re.match(r'console\.(log|warn|error|info|debug)\s*\(', stripped):
            if not stripped.startswith('//'):
                issues.append((i, "console.log left in production code — remove or use proper logger"))
        # Nested ternary (hard to read)
        if stripped.count(' ? ') >= 2 and stripped.count(' : ') >= 2:
            issues.append((i, "Nested ternary — extract to variable or use if/else for readability"))
        # Empty catch block
        if stripped == 'catch' or re.match(r'catch\s*\([^)]*\)\s*\{\s*\}', stripped):
            issues.append((i, "Empty catch block — handle or log the error"))
    # .then() chains (3+ deep)
    then_chain = re.findall(r'\.then\(', content)
    if len(then_chain) >= 3:
        issues.append((0, "Deep .then() chain — consider async/await"))
    # Missing key prop in .map() JSX
    for m in re.finditer(r'\.map\([^)]*\)\s*=>\s*[({]', content):
        # Check if there's a key= in the next 200 chars
        snippet = content[m.end():m.end()+200]
        if 'key=' not in snippet and 'key =' not in snippet:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "Missing key prop in .map() JSX"))

    # --- Universal JS/TS checks ---

    # A. Unused imports (import X but X never appears in rest of file)
    for m in re.finditer(r'import\s+(?:\{([^}]+)\}|(\w+))', content):
        names = []
        if m.group(1):
            names = [n.strip().split(' as ')[-1].strip() for n in m.group(1).split(',')]
        elif m.group(2) and m.group(2) not in ('type', 'from'):
            names = [m.group(2)]
        import_line = content[:m.start()].count("\n") + 1
        rest_of_file = content[m.end():]
        for name in names:
            if name and len(name) > 1 and not re.search(r'\b' + re.escape(name) + r'\b', rest_of_file):
                issues.append((import_line, f"Unused import: '{name}'"))

    # B. Missing 'use client' on components with hooks/events + JSX return
    # Detect even without explicit Next.js imports — any .tsx/.jsx with hooks needs 'use client' in Next.js
    if has_jsx_return:
        has_hooks = bool(re.search(r'\b(useState|useEffect|useRef|useCallback|useMemo|useReducer|useContext|useLayoutEffect|use[A-Z]\w+)\s*\(', content))
        has_event_handlers = bool(re.search(r'\bon[A-Z]\w+=\{', content))
        has_framer = 'framer-motion' in content
        if not has_use_client and (has_hooks or has_event_handlers or has_framer):
            issues.append((1, "Missing 'use client' directive — file uses hooks/events/framer-motion with JSX"))

    # C. One-shot framer-motion animations (animate={{}} without key/AnimatePresence/useAnimate)
    if 'framer-motion' in content or 'motion.' in content:
        has_animate_prop = bool(re.search(r'animate=\{\{', content))
        has_key_or_presence = bool(re.search(r'\bkey=', content)) or 'AnimatePresence' in content or 'useAnimate' in content
        if has_animate_prop and not has_key_or_presence:
            for m in re.finditer(r'animate=\{\{', content):
                lineno = content[:m.start()].count("\n") + 1
                issues.append((lineno, "One-shot framer-motion animate={{}} — use key prop, AnimatePresence, or useAnimate for triggers"))
                break  # Report once

    # D. Browser API usage in server-rendered files
    # Reports up to 5 occurrences per file to avoid whack-a-mole
    if not has_use_client and is_ssr_path:
        browser_apis = re.compile(
            r'\b(window\.|document\.|localStorage\.|sessionStorage\.|'
            r'navigator\.|alert\s*\(|confirm\s*\(|prompt\s*\()')
        in_block_comment = False
        browser_hits = 0
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Track block comment state
            if '/*' in stripped and '*/' not in stripped:
                in_block_comment = True
                continue
            if in_block_comment:
                if '*/' in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith('//'):
                continue
            # Strip inline /* ... */ comments before scanning
            stripped = re.sub(r'/\*.*?\*/', '', stripped)
            # Skip typeof guards
            if 'typeof window' in stripped or 'typeof document' in stripped or 'typeof navigator' in stripped:
                continue
            bm = browser_apis.search(stripped)
            if bm:
                issues.append((i, f"Browser API '{bm.group(1).rstrip('.')}' in server-rendered file — "
                                  f"add 'use client' or guard with typeof check"))
                browser_hits += 1
                if browser_hits >= 5:
                    break
    elif not has_use_client:
        # Original module-level check for non-app-dir files
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if indent == 0 and not stripped.startswith('//') and not stripped.startswith('*'):
                if re.search(r'\b(window\.|document\.|localStorage\.|sessionStorage\.)', stripped):
                    if 'typeof window' not in stripped and 'typeof document' not in stripped:
                        issues.append((i, "Unguarded browser API at module level — wrap in typeof check"))
                        break

    # E. Raw <img> when next is imported (should use next/image)
    if has_next_import:
        for m in re.finditer(r'<img\b', content):
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "Use next/image instead of raw <img> in Next.js"))

    # F. TypeScript 'any' overuse (3+ occurrences)
    any_count = len(re.findall(r':\s*any\b', content))
    if any_count >= 3:
        issues.append((0, f"Excessive 'any' type usage ({any_count}) — use proper types"))

    # G. Hardcoded localhost/API URLs
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith('//') and not stripped.startswith('*'):
            if re.search(r'["\']https?://(localhost|127\.0\.0\.1)', stripped):
                if 'process.env' not in stripped and 'import.meta.env' not in stripped:
                    issues.append((i, "Hardcoded localhost URL — use environment variable"))

    # H. useEffect/setTimeout/setInterval without cleanup in React
    if has_jsx_return:
        for m in re.finditer(r'useEffect\(\s*\(\)\s*=>\s*\{', content):
            effect_start = m.end()
            # Find matching closing — look for setInterval/setTimeout
            effect_snippet = content[effect_start:effect_start + 500]
            if ('setInterval' in effect_snippet or 'setTimeout' in effect_snippet or
                'addEventListener' in effect_snippet or 'subscribe' in effect_snippet):
                if 'return' not in effect_snippet.split('}')[0] and 'clearInterval' not in effect_snippet and 'clearTimeout' not in effect_snippet:
                    lineno = content[:m.start()].count("\n") + 1
                    issues.append((lineno, "useEffect with timer/listener but no cleanup return — will cause memory leak"))

    # I. Index as key in map
    for m in re.finditer(r'\.map\(\s*\([^,)]*,\s*(\w+)\)', content):
        index_name = m.group(1)
        snippet = content[m.end():m.end() + 300]
        if f'key={{{index_name}}}' in snippet or f'key={{ {index_name} }}' in snippet:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, f"Array index '{index_name}' used as key — unstable for reorderable lists"))

    # J. Placeholder href="#" links (lazy non-functional navigation)
    placeholder_links = re.findall(r'href=["\']#["\']', content)
    if len(placeholder_links) >= 2:
        issues.append((0, f"{len(placeholder_links)} placeholder href='#' links — use real routes with next/link"))

    # K. Raw <a> tag in Next.js (should use next/link Link component)
    if has_next_import:
        a_tags = re.findall(r'<a\s+', content)
        if len(a_tags) >= 2 and 'next/link' not in content:
            issues.append((0, f"Raw <a> tags ({len(a_tags)}) in Next.js — use next/link <Link> component"))

    # L. Structural conflict: default component + HTTP method handlers in same file
    if not has_jsx_return:
        has_jsx_return = bool(re.search(r'return\s*\(?\s*<', content))
    http_methods_exported = re.findall(
        r'export\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b', content)
    has_default_export = bool(re.search(r'export\s+default\s+(?:async\s+)?function\b', content))
    if has_default_export and has_jsx_return and http_methods_exported:
        issues.append((1, f"Structural conflict: file exports default component AND HTTP handlers "
                          f"({', '.join(http_methods_exported)}) — split into page + route files"))

    # M. Untyped callback parameters in TypeScript (strict mode failure)
    if ext in ('.ts', '.tsx'):
        # Match: .method(param) { / .method(param) => / = (param) => / function(param) {
        cb_pattern = re.compile(
            r'(?:\.\w+\s*\(\s*|=\s*\(\s*|function\s*\(\s*)'
            r'([a-zA-Z_]\w*)\s*\)'
            r'\s*(?:=>|\{|:)')
        cb_hits = 0
        for cb_m in cb_pattern.finditer(content):
            param = cb_m.group(1)
            # Skip single-character params (too common, low signal)
            if len(param) <= 1:
                continue
            # Check: no colon between param name and closing paren = no type annotation
            match_text = cb_m.group(0)
            after_param = match_text.split(param, 1)[-1].split(')')[0]
            if ':' not in after_param:
                lineno = content[:cb_m.start()].count("\n") + 1
                issues.append((lineno, f"Untyped callback parameter '{param}' — add type annotation for strict mode"))
                cb_hits += 1
                if cb_hits >= 5:
                    break  # cap at 5 per file

    # N. Client-only library in server component
    if not has_use_client and is_ssr_path:
        for imp_m in re.finditer(r'''import\s+.*?from\s+['"]([@\w/.-]+)['"]''', content):
            mod_name = imp_m.group(1)
            bare = mod_name.split('/')[0] if not mod_name.startswith('@') else '/'.join(mod_name.split('/')[:2])
            if bare in _CLIENT_ONLY_MODULES:
                lineno = content[:imp_m.start()].count("\n") + 1
                issues.append((lineno, f"Client-only library '{bare}' imported in server component — "
                                       f"add 'use client' or extract to client child component"))

    return issues


def _check_css_idioms(content):
    """Check CSS for anti-patterns. Returns list of (line_num, issue)."""
    issues = []
    important_count = content.count("!important")
    if important_count > 2:
        issues.append((0, f"Excessive !important usage ({important_count} occurrences)"))
    # Nesting depth (SCSS/CSS-in-JS)
    max_depth = 0
    depth = 0
    for ch in content:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == '}':
            depth = max(depth - 1, 0)
    if max_depth > 4:
        issues.append((0, f"Nesting depth {max_depth} — keep under 4 levels"))

    # --- Universal CSS checks ---
    lines = content.split("\n")

    # A. @keyframes inside @layer utilities (should be @layer base)
    in_layer_utilities = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if '@layer utilities' in stripped:
            in_layer_utilities = True
        elif '@layer' in stripped and 'utilities' not in stripped:
            in_layer_utilities = False
        if in_layer_utilities and '@keyframes' in stripped:
            issues.append((i, "@keyframes inside @layer utilities — move to @layer base"))

    # A2. Duplicate @layer block detection
    layer_counts = {}
    for layer_m in re.finditer(r'@layer\s+([\w-]+)\s*\{', content):
        name = layer_m.group(1)
        layer_counts[name] = layer_counts.get(name, 0) + 1
    for name, count in layer_counts.items():
        if count > 1:
            issues.append((0, f"Duplicate @layer {name} block ({count} occurrences) — merge into one"))

    # B. Animations without prefers-reduced-motion media query
    has_animation = bool(re.search(r'(animation:|@keyframes\s)', content))
    has_reduced_motion = 'prefers-reduced-motion' in content
    if has_animation and not has_reduced_motion:
        issues.append((0, "Animations defined without @media (prefers-reduced-motion) — add reduced-motion fallback"))

    # C. 3D transforms without perspective/preserve-3d
    has_3d = bool(re.search(r'(rotateX|rotateY|rotate3d|translateZ|translate3d|perspective\()', content))
    has_perspective = 'perspective:' in content or 'perspective(' in content
    has_preserve_3d = 'preserve-3d' in content
    if has_3d and not (has_perspective and has_preserve_3d):
        if not has_perspective:
            issues.append((0, "3D transforms used without perspective on parent"))
        if not has_preserve_3d:
            issues.append((0, "3D transforms used without transform-style: preserve-3d"))

    # D. @apply outside globals.css (breaks CSS Modules)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if '@apply' in stripped:
            issues.append((i, "@apply usage — only use in globals.css to avoid CSS Modules issues"))
            break  # Report once

    # E. Missing :focus-visible on custom interactive elements
    has_custom_buttons = bool(re.search(r'(button|\.btn|a\[|\.link)', content))
    has_focus_visible = ':focus-visible' in content or ':focus' in content
    if has_custom_buttons and not has_focus_visible:
        issues.append((0, "Custom interactive element styles without :focus-visible — add focus states for accessibility"))

    # F. Hardcoded color values (should use CSS custom properties) — only in files 20+ lines
    if len(lines) > 20:
        hex_colors = re.findall(r':\s*#[0-9a-fA-F]{3,8}\b', content)
        rgb_colors = re.findall(r':\s*rgb[a]?\(', content)
        hardcoded_count = len(hex_colors) + len(rgb_colors)
        has_css_vars = 'var(--' in content
        if hardcoded_count >= 5 and not has_css_vars:
            issues.append((0, f"Hardcoded colors ({hardcoded_count}) without CSS custom properties — use var(--color-*) for maintainability"))

    # G. px on font-size (should use rem/em for accessibility)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.search(r'font-size:\s*\d+px', stripped):
            issues.append((i, "font-size in px — use rem or em for accessibility (respects user font preferences)"))

    # H. Magic z-index values
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        z_match = re.search(r'z-index:\s*(\d+)', stripped)
        if z_match and int(z_match.group(1)) > 100:
            issues.append((i, f"Magic z-index value ({z_match.group(1)}) — use CSS custom property or documented scale"))

    # I. No media queries in files with 50+ lines (missing responsive design)
    if len(lines) > 50 and '@media' not in content and '@tailwind' not in content:
        issues.append((0, "No @media queries in 50+ line stylesheet — consider responsive design"))

    return issues


def _check_html_idioms(content):
    """Check HTML for anti-patterns. Returns list of (line_num, issue)."""
    issues = []
    # Excessive inline styles
    inline_count = len(re.findall(r'style="', content))
    if inline_count > 3:
        issues.append((0, f"Excessive inline styles ({inline_count}) — use CSS classes"))
    # Missing alt on img tags
    for m in re.finditer(r'<img\b[^>]*>', content):
        tag = m.group()
        if 'alt=' not in tag:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "Missing alt attribute on <img>"))
    # Div soup (5+ nested divs without semantic elements)
    div_depth = 0
    max_div_depth = 0
    for m in re.finditer(r'<(/?)div\b', content):
        if m.group(1) == '/':
            div_depth = max(div_depth - 1, 0)
        else:
            div_depth += 1
            max_div_depth = max(max_div_depth, div_depth)
    if max_div_depth >= 5:
        issues.append((0, f"Div soup — {max_div_depth} nested divs. Use semantic HTML (section, article, nav, etc.)"))

    # --- Universal HTML checks ---

    # A. Missing <meta name="viewport"> (responsive breakage)
    if '<html' in content or '<!DOCTYPE' in content.upper() or '<!doctype' in content:
        if '<meta' in content and 'viewport' not in content:
            issues.append((0, "Missing <meta name=\"viewport\"> — required for responsive design"))

    # B. Missing lang attribute on <html>
    html_tag_match = re.search(r'<html\b([^>]*)>', content)
    if html_tag_match:
        if 'lang=' not in html_tag_match.group(1):
            lineno = content[:html_tag_match.start()].count("\n") + 1
            issues.append((lineno, "Missing lang attribute on <html> — required for accessibility"))

    # C. Forms without associated <label> elements
    form_count = len(re.findall(r'<(input|select|textarea)\b', content))
    label_count = len(re.findall(r'<label\b', content))
    if form_count > 0 and label_count == 0:
        issues.append((0, f"Form inputs ({form_count}) without any <label> elements — required for accessibility"))

    # D. Animations/transitions without prefers-reduced-motion
    has_animation = bool(re.search(r'(animation:|transition:|@keyframes)', content))
    has_reduced_motion = 'prefers-reduced-motion' in content
    if has_animation and not has_reduced_motion:
        issues.append((0, "Animations/transitions without prefers-reduced-motion media query"))

    # E. Missing meta description
    if '<head' in content and '<meta' in content:
        if 'name="description"' not in content and "name='description'" not in content:
            issues.append((0, "Missing <meta name=\"description\"> — required for SEO"))

    # F. Script tags without defer/async
    for m in re.finditer(r'<script\b([^>]*)src=', content):
        attrs = m.group(1)
        if 'defer' not in attrs and 'async' not in attrs and 'type="module"' not in attrs:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "<script src> without defer/async — blocks rendering"))

    # G. Missing ARIA on interactive custom elements
    for m in re.finditer(r'<div\b[^>]*(onclick|@click|onClick)', content):
        tag_content = content[m.start():m.start() + 300].split('>')[0]
        if 'role=' not in tag_content and 'aria-' not in tag_content:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "Interactive <div> without role/aria attributes — use <button> or add role=\"button\""))

    # H. Missing favicon
    if '<head' in content and '</head>' in content:
        head_section = content[content.index('<head'):content.index('</head>')]
        if 'rel="icon"' not in head_section and "rel='icon'" not in head_section and 'favicon' not in head_section:
            issues.append((0, "Missing favicon link — add <link rel=\"icon\">"))

    return issues


def _resolve_tsconfig_paths():
    """Read tsconfig.json compilerOptions.paths. Returns dict {alias_prefix: target_dir}."""
    aliases = {'@/': 'src/', '~/': ''}
    for name in ('tsconfig.json', 'jsconfig.json'):
        tsc = Path(CWD) / name
        if not tsc.exists():
            continue
        try:
            raw = re.sub(r'//.*$', '', tsc.read_text(encoding='utf-8', errors='replace'), flags=re.MULTILINE)
            raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
            data = json.loads(raw)
            paths = data.get('compilerOptions', {}).get('paths', {})
            base_url = data.get('compilerOptions', {}).get('baseUrl', '.')
            for alias_pattern, targets in paths.items():
                if targets and alias_pattern.endswith('/*'):
                    prefix = alias_pattern[:-1]
                    target_dir = targets[0].replace('/*', '/') if targets[0].endswith('/*') else targets[0]
                    if base_url != '.':
                        target_dir = base_url.rstrip('/') + '/' + target_dir.lstrip('/')
                    aliases[prefix] = target_dir
        except Exception:
            pass
        break
    return aliases

_tsconfig_paths_cache = None
_tsconfig_paths_mtime = 0

def _get_tsconfig_paths():
    """Cached tsconfig paths — invalidates when tsconfig.json mtime changes."""
    global _tsconfig_paths_cache, _tsconfig_paths_mtime
    mtime = 0
    for name in ('tsconfig.json', 'jsconfig.json'):
        tsc = Path(CWD) / name
        if tsc.exists():
            try:
                mtime = tsc.stat().st_mtime
            except OSError:
                pass
            break
    if _tsconfig_paths_cache is None or mtime != _tsconfig_paths_mtime:
        _tsconfig_paths_cache = _resolve_tsconfig_paths()
        _tsconfig_paths_mtime = mtime
    return _tsconfig_paths_cache


def _check_import_coherence(content, filepath):
    """Check that imports reference real packages/files. Returns list of (line_num, issue)."""
    issues = []
    fp = Path(filepath)
    ext = fp.suffix.lower()

    if ext in (".js", ".ts", ".jsx", ".tsx"):
        # Check JS/TS imports against package.json
        pkg_path = Path(CWD) / "package.json"
        known_deps = set()
        if pkg_path.exists():
            try:
                pkg = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
                known_deps.update(pkg.get("dependencies", {}).keys())
                known_deps.update(pkg.get("devDependencies", {}).keys())
            except Exception:
                pass
        for m in re.finditer(r'(?:import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))', content):
            mod = m.group(1) or m.group(2)
            if mod.startswith('.'):
                # Relative import — check if file exists
                target = (fp.parent / mod).resolve()
                # Try common extensions
                exists = any(target.with_suffix(s).exists() for s in ['', '.js', '.ts', '.jsx', '.tsx', '.json', '/index.js', '/index.ts'])
                if not exists and not any((target.parent / (target.name + s)).exists() for s in ['.js', '.ts', '.jsx', '.tsx']):
                    lineno = content[:m.start()].count("\n") + 1
                    issues.append((lineno, f"Relative import '{mod}' — target file not found"))
            else:
                # Package import — check against package.json
                pkg_name = mod.split('/')[0]
                if pkg_name.startswith('@'):
                    pkg_name = '/'.join(mod.split('/')[:2])
                # Skip node builtins
                node_builtins = {'fs', 'path', 'os', 'url', 'http', 'https', 'crypto', 'stream', 'util', 'events', 'child_process', 'buffer', 'querystring', 'assert', 'net', 'tls', 'zlib'}
                if pkg_name not in node_builtins and pkg_name not in known_deps and known_deps:
                    lineno = content[:m.start()].count("\n") + 1
                    issues.append((lineno, f"Package '{pkg_name}' not in package.json dependencies"))

            # Self-import detection (relative + all alias patterns)
            aliases = _get_tsconfig_paths()
            self_import_resolved = None

            if mod.startswith('.'):
                try:
                    base_resolved = (fp.parent / mod).resolve()
                    # Try exact, then extensions, then index files
                    for suffix in ['', '.ts', '.tsx', '.js', '.jsx']:
                        c = base_resolved.with_suffix(suffix) if suffix else base_resolved
                        if c.exists() and c.resolve() == fp.resolve():
                            self_import_resolved = c
                            break
                    if not self_import_resolved:
                        for idx in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
                            c = base_resolved / idx
                            if c.exists() and c.resolve() == fp.resolve():
                                self_import_resolved = c
                                break
                except Exception:
                    pass
            else:
                for alias_prefix, target_dir in aliases.items():
                    if not mod.startswith(alias_prefix):
                        continue
                    remainder = mod[len(alias_prefix):]
                    for base in [Path(CWD) / target_dir.rstrip('/'), Path(CWD)]:
                        candidate = (base / remainder).resolve() if remainder else base.resolve()
                        # Try extensions + index resolution
                        for suffix in ['', '.ts', '.tsx', '.js', '.jsx']:
                            c = candidate.with_suffix(suffix) if suffix else candidate
                            try:
                                if c.exists() and c.resolve() == fp.resolve():
                                    self_import_resolved = c
                                    break
                            except Exception:
                                pass
                        if not self_import_resolved:
                            for idx in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
                                c = candidate / idx
                                try:
                                    if c.exists() and c.resolve() == fp.resolve():
                                        self_import_resolved = c
                                        break
                                except Exception:
                                    pass
                        if self_import_resolved:
                            break  # found match — stop trying bases
                    if self_import_resolved:
                        break  # found match — stop trying aliases

            if self_import_resolved:
                lineno = content[:m.start()].count("\n") + 1
                issues.append((lineno, f"Self-import — file imports from itself via '{mod}'"))

    elif ext == ".py":
        # Check Python imports against known stdlib + requirements
        known_pkgs = set()
        for req_file in ["requirements.txt", "pyproject.toml"]:
            rp = Path(CWD) / req_file
            if rp.exists():
                try:
                    txt = rp.read_text(encoding="utf-8", errors="replace")
                    if req_file == "requirements.txt":
                        for line in txt.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("#"):
                                known_pkgs.add(re.split(r'[>=<\[!;]', line)[0].strip().lower().replace('-', '_'))
                    elif req_file == "pyproject.toml":
                        for m2 in re.finditer(r'"([^"]+)"', txt):
                            known_pkgs.add(m2.group(1).split('[')[0].strip().lower().replace('-', '_'))
                except Exception:
                    pass
        python_stdlib = {
            'os', 'sys', 're', 'json', 'math', 'time', 'datetime', 'pathlib', 'collections',
            'itertools', 'functools', 'typing', 'abc', 'io', 'hashlib', 'base64', 'copy',
            'shutil', 'glob', 'subprocess', 'threading', 'multiprocessing', 'socket', 'http',
            'urllib', 'email', 'html', 'xml', 'csv', 'sqlite3', 'logging', 'unittest', 'pdb',
            'argparse', 'configparser', 'textwrap', 'string', 'struct', 'enum', 'dataclasses',
            'contextlib', 'warnings', 'traceback', 'inspect', 'importlib', 'pkgutil', 'ast',
            'dis', 'token', 'tokenize', 'pprint', 'tempfile', 'random', 'secrets', 'uuid',
            'decimal', 'fractions', 'statistics', 'array', 'heapq', 'bisect', 'operator',
            'difflib', 'calendar', 'locale', 'gettext', 'unicodedata', 'codecs',
        }
        for m in re.finditer(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE):
            mod = m.group(1).lower()
            if mod not in python_stdlib and mod not in known_pkgs and known_pkgs:
                # Could be a local module — check if file exists
                local_mod = Path(CWD) / (mod + ".py")
                local_pkg = Path(CWD) / mod / "__init__.py"
                if not local_mod.exists() and not local_pkg.exists():
                    lineno = content[:m.start()].count("\n") + 1
                    issues.append((lineno, f"Module '{mod}' not in requirements or stdlib"))

    # Detect fetch() calls to nonexistent API routes
    api_dir = None
    for candidate in [Path(CWD) / 'src' / 'app' / 'api', Path(CWD) / 'app' / 'api']:
        if candidate.is_dir():
            api_dir = candidate
            break

    file_ext = fp.suffix.lower()
    if api_dir and file_ext in ('.js', '.ts', '.jsx', '.tsx'):

        def _route_exists(base, segs):
            """Check if route.ts exists for the given path segments, handling dynamic/catch-all."""
            if not segs:
                return (base / 'route.ts').exists() or (base / 'route.js').exists()
            seg = segs[0]
            rest = segs[1:]
            if (base / seg).is_dir() and _route_exists(base / seg, rest):
                return True
            try:
                for d in base.iterdir():
                    if not d.is_dir() or not d.name.startswith('['):
                        continue
                    if d.name.startswith('[[...') or d.name.startswith('[...'):
                        # Catch-all: matches all remaining segments
                        if (d / 'route.ts').exists() or (d / 'route.js').exists():
                            return True
                    else:
                        # Single dynamic segment
                        if _route_exists(d, rest):
                            return True
            except OSError:
                pass
            return False

        # Three URL patterns:
        #   1. String literal: fetch('/api/tasks')
        #   2. Template literal WITHOUT interpolation: fetch(`/api/tasks`) — treat as string
        #   3. Template literal WITH interpolation: fetch(`/api/tasks/${id}`) — extract prefix
        for fetch_m in re.finditer(
            r'''fetch\s*\(\s*(?:'''
            r'''['"](/api/[^'"]+)['"]'''           # group 1: string literal
            r'''|`(/api/[^`$]*)\$\{'''             # group 2: template WITH interpolation (prefix before ${)
            r'''|`(/api/[^`$]*)`'''                # group 3: template WITHOUT interpolation (full URL)
            r''')''', content):
            string_path = fetch_m.group(1)     # from 'string' or "string"
            interp_prefix = fetch_m.group(2)   # from `prefix${...}`
            backtick_full = fetch_m.group(3)   # from `full/path`

            # Template without interpolation = treat as exact string
            api_path = string_path or backtick_full or interp_prefix
            is_interpolated = bool(interp_prefix)
            if not api_path:
                continue

            segments = [s for s in api_path.replace('/api/', '').split('/') if s]

            if is_interpolated and len(segments) > 1:
                # Template with ${}: /api/tasks/${id} → check /api/tasks/ has route OR /api/tasks/[param]/
                found = _route_exists(api_dir, segments) or _route_exists(api_dir, segments[:-1])
            else:
                # String literal or template without ${}: exact match only
                found = _route_exists(api_dir, segments)

            if not found:
                lineno = content[:fetch_m.start()].count("\n") + 1
                display = api_path + ('...' if is_interpolated else '')
                issues.append((lineno, f"fetch('{display}') — no matching route.ts found in app/api/"))

    return issues


def _slop_score(content, filepath):
    """Score generated code quality. Starts at 100, deductions for issues.
    Returns (score, issues_list) where issues_list contains (line, desc, deduction)."""
    score = 100
    issues = []
    ext = Path(filepath).suffix.lower()

    # --- TODO/FIXME/HACK comments ---
    todo_cap = 0
    for m in re.finditer(r'(?:#|//|/\*)\s*(?:TODO|FIXME|HACK|XXX)\b', content, re.IGNORECASE):
        if todo_cap < 25:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "TODO/FIXME/HACK comment", -5))
            score -= 5
            todo_cap += 5

    # --- Empty function bodies ---
    empty_cap = 0
    if ext == ".py":
        for m in re.finditer(r'def \w+\([^)]*\):\s*\n\s+pass\s*(?:\n|$)', content):
            if empty_cap < 24:
                lineno = content[:m.start()].count("\n") + 1
                issues.append((lineno, "Empty function body (just pass)", -8))
                score -= 8
                empty_cap += 8
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        for m in re.finditer(r'(?:function\s+\w+|=>\s*)\s*\{\s*\}', content):
            if empty_cap < 24:
                lineno = content[:m.start()].count("\n") + 1
                issues.append((lineno, "Empty function body", -8))
                score -= 8
                empty_cap += 8

    # --- Generic variable names ---
    generic_cap = 0
    generic_vars = re.compile(r'\b(?:data|result|temp|info|stuff|value|item|obj|arr|tmp)\s*[=:]', re.IGNORECASE)
    for m in generic_vars.finditer(content):
        if generic_cap < 15:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, f"Generic variable name: {m.group().strip()}", -3))
            score -= 3
            generic_cap += 3

    # --- Redundant comments restating code ---
    redundant_cap = 0
    lines = content.split("\n")
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        next_line = lines[i + 1].strip()
        if (line.startswith('#') or line.startswith('//')) and next_line:
            comment_text = re.sub(r'^[#/]+\s*', '', line).lower().strip()
            code_text = re.sub(r'[^a-zA-Z\s]', ' ', next_line).lower().strip()
            if comment_text and code_text:
                # Simple word overlap check
                comment_words = set(comment_text.split())
                code_words = set(code_text.split())
                if len(comment_words) >= 2 and len(comment_words & code_words) / len(comment_words) > 0.6:
                    if redundant_cap < 10:
                        issues.append((i + 1, "Redundant comment restating code", -2))
                        score -= 2
                        redundant_cap += 2

    # --- Generic error messages ---
    generic_err_cap = 0
    generic_errs = re.compile(r'["\'](?:Something went wrong|An error occurred|Error occurred|Unknown error|Oops)["\']', re.IGNORECASE)
    for m in generic_errs.finditer(content):
        if generic_err_cap < 12:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "Generic error message — be specific", -4))
            score -= 4
            generic_err_cap += 4

    # --- Lorem ipsum / placeholder text ---
    lorem_cap = 0
    for m in re.finditer(r'lorem ipsum|placeholder\s+text|dummy\s+text|sample\s+text', content, re.IGNORECASE):
        if lorem_cap < 18:
            lineno = content[:m.start()].count("\n") + 1
            issues.append((lineno, "Lorem ipsum / placeholder text", -6))
            score -= 6
            lorem_cap += 6

    # --- Tutorial boilerplate ---
    is_bp, bp_pattern, generic_ratio = _boilerplate_fingerprint(content, filepath)
    bp_cap = 0
    if is_bp and bp_cap < 15:
        issues.append((0, f"Matches tutorial boilerplate pattern: {bp_pattern}", -5))
        score -= 5
        bp_cap += 5
    if generic_ratio > 0.5 and bp_cap < 15:
        issues.append((0, f"High generic function name ratio: {generic_ratio:.0%}", -5))
        score -= 5
        bp_cap += 5

    # --- Language-specific anti-patterns ---
    lang_cap = 0
    lang_issues = []
    if ext == ".py":
        lang_issues = _check_python_idioms(content)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        lang_issues = _check_js_idioms(content, filepath)
    elif ext in (".css", ".scss", ".sass"):
        lang_issues = _check_css_idioms(content)
    elif ext in (".html", ".htm"):
        lang_issues = _check_html_idioms(content)
    for lineno, desc in lang_issues:
        if lang_cap < 20:
            issues.append((lineno, f"Anti-pattern: {desc}", -3))
            score -= 3
            lang_cap += 3

    # --- Query filter with string interpolation — injection risk warning (informational, -2) ---
    if ext in ('.js', '.ts', '.jsx', '.tsx'):
        qi_deducted = 0
        for qm in re.finditer(r'\.(or|filter|match|textSearch)\s*\(\s*`[^`]*\$\{', content):
            if qi_deducted < 6:  # cap: 3 occurrences × -2 = -6 max
                lineno = content[:qm.start()].count("\n") + 1
                issues.append((lineno, "Query filter with string interpolation — validate input before query", -2))
                score -= 2
                qi_deducted += 2

    # --- Security checks (cap: 15pts, all languages) ---
    sec_cap = 0
    # Hardcoded secrets in any language
    if re.search(r'(?:api[_-]?key|secret|password|token|private[_-]?key)\s*[:=]\s*["\'][^"\']{8,}', content, re.IGNORECASE):
        if sec_cap < 15:
            issues.append((0, "Hardcoded secret/credential detected", -10))
            score -= 10
            sec_cap += 10
    # Hardcoded URLs (localhost, IPs)
    if re.search(r'["\']https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)', content):
        if 'process.env' not in content and 'import.meta.env' not in content and 'os.environ' not in content and 'os.getenv' not in content and 'settings' not in content.lower():
            if sec_cap < 15:
                issues.append((0, "Hardcoded localhost/IP URL — use environment variable", -5))
                score -= 5
                sec_cap += 5
    # innerHTML / dangerouslySetInnerHTML without sanitization
    if 'dangerouslySetInnerHTML' in content or '.innerHTML' in content:
        if 'sanitize' not in content.lower() and 'DOMPurify' not in content:
            if sec_cap < 15:
                issues.append((0, "innerHTML/dangerouslySetInnerHTML without sanitization — XSS risk", -5))
                score -= 5
                sec_cap += 5

    # --- Framework correctness (cap: 30pts, all languages) ---
    fw_cap = 0

    # React/Next.js
    if ext in (".jsx", ".tsx"):
        has_jsx_return = bool(re.search(r'return\s*\(?\s*<', content))
        has_next_import = 'from "next' in content or "from 'next" in content
        has_use_client = "'use client'" in content or '"use client"' in content
        has_hooks = bool(re.search(r'\buse[A-Z]\w+\s*\(', content))
        has_event_handlers = bool(re.search(r'\bon[A-Z]\w+=\{', content))
        has_framer = 'framer-motion' in content
        # Missing 'use client' on entry-point client component: -10
        if has_jsx_return and has_next_import and not has_use_client and (has_hooks or has_event_handlers or has_framer):
            if fw_cap < 30:
                issues.append((1, "Missing 'use client' directive on interactive component", -10))
                score -= 10
                fw_cap += 10
        # One-shot framer-motion animation: -8
        if has_framer and bool(re.search(r'animate=\{\{', content)):
            if 'key=' not in content and 'AnimatePresence' not in content and 'useAnimate' not in content:
                if fw_cap < 30:
                    issues.append((0, "One-shot framer-motion animate={{}} without key/AnimatePresence/useAnimate", -8))
                    score -= 8
                    fw_cap += 8
        # Unguarded browser API at module level: -6
        for line_content in content.split("\n"):
            if not line_content.startswith((' ', '\t')) and not line_content.strip().startswith('//'):
                if re.search(r'\b(window\.|document\.|localStorage\.|sessionStorage\.)', line_content):
                    if 'typeof window' not in line_content and 'typeof document' not in line_content:
                        if fw_cap < 30:
                            issues.append((0, "Unguarded browser API at module level", -6))
                            score -= 6
                            fw_cap += 6
                        break
        # console.log left in production TSX/JSX: -4
        if re.search(r'^\s*console\.(log|warn|info|debug)\s*\(', content, re.MULTILINE):
            if fw_cap < 30:
                issues.append((0, "console.log in production component — remove before shipping", -4))
                score -= 4
                fw_cap += 4
        # Excessive TypeScript 'any': -6
        any_count = len(re.findall(r':\s*any\b', content))
        if any_count >= 3:
            if fw_cap < 30:
                issues.append((0, f"Excessive 'any' type ({any_count}) — defeats TypeScript's purpose", -6))
                score -= 6
                fw_cap += 6
        # Placeholder href="#" links: -8
        placeholder_ct = len(re.findall(r'href=["\']#["\']', content))
        if placeholder_ct >= 2:
            if fw_cap < 30:
                issues.append((0, f"Placeholder href='#' links ({placeholder_ct}) — navigation is non-functional", -8))
                score -= 8
                fw_cap += 8
        # Raw <a> in Next.js instead of Link: -4
        if ('from "next' in content or "from 'next" in content) and 'next/link' not in content:
            a_count = len(re.findall(r'<a\s+', content))
            if a_count >= 2:
                if fw_cap < 30:
                    issues.append((0, f"Raw <a> tags ({a_count}) in Next.js — use next/link", -4))
                    score -= 4
                    fw_cap += 4

    # Python
    elif ext == ".py":
        has_fastapi = 'fastapi' in content.lower() or 'from fastapi' in content
        # Sync def in async context (FastAPI): -6
        if has_fastapi:
            for m in re.finditer(r'@(app|router)\.(get|post|put|patch|delete)\b', content):
                rest = content[m.end():]
                def_match = re.search(r'\ndef\s+', rest)
                async_match = re.search(r'\nasync\s+def\s+', rest)
                if def_match and (not async_match or def_match.start() < async_match.start()):
                    if fw_cap < 30:
                        issues.append((0, "Sync endpoint in FastAPI — use async def", -6))
                        score -= 6
                        fw_cap += 6
                    break
        # Hardcoded secrets: -10
        if re.search(r'(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*["\'][^"\']{8,}', content, re.IGNORECASE):
            if fw_cap < 30:
                issues.append((0, "Hardcoded secret detected", -10))
                score -= 10
                fw_cap += 10
        # print() instead of logging in non-script: -4
        has_main_guard = 'if __name__' in content
        if not has_main_guard and has_fastapi:
            if re.search(r'^\s*print\s*\(', content, re.MULTILINE):
                if fw_cap < 30:
                    issues.append((0, "print() in production code — use logging module", -4))
                    score -= 4
                    fw_cap += 4
        # Star imports: -4
        if re.search(r'^from\s+\S+\s+import\s+\*', content, re.MULTILINE):
            if fw_cap < 30:
                issues.append((0, "Star import (from X import *) — pollutes namespace", -4))
                score -= 4
                fw_cap += 4
        # Mutable default arguments: -4
        if re.search(r'def\s+\w+\(.*=\s*(\[\]|\{\}|set\(\))', content):
            if fw_cap < 30:
                issues.append((0, "Mutable default argument — causes shared state bugs", -4))
                score -= 4
                fw_cap += 4

    # CSS
    elif ext in (".css", ".scss"):
        # Animations without prefers-reduced-motion: -6
        has_animation = bool(re.search(r'(animation:|@keyframes\s)', content))
        if has_animation and 'prefers-reduced-motion' not in content:
            if fw_cap < 30:
                issues.append((0, "Animations without prefers-reduced-motion", -6))
                score -= 6
                fw_cap += 6
        # Keyframes in wrong layer: -4
        if '@layer utilities' in content and '@keyframes' in content:
            # Check if keyframes is inside utilities layer
            in_utils = False
            for css_line in content.split("\n"):
                if '@layer utilities' in css_line:
                    in_utils = True
                elif '@layer' in css_line and 'utilities' not in css_line:
                    in_utils = False
                if in_utils and '@keyframes' in css_line:
                    if fw_cap < 30:
                        issues.append((0, "@keyframes in @layer utilities — use @layer base", -4))
                        score -= 4
                        fw_cap += 4
                    break
        # 3D transforms without perspective: -4
        has_3d = bool(re.search(r'(rotateX|rotateY|rotate3d|translateZ|translate3d)', content))
        if has_3d and 'preserve-3d' not in content:
            if fw_cap < 30:
                issues.append((0, "3D transforms without transform-style: preserve-3d", -4))
                score -= 4
                fw_cap += 4
        # Hardcoded colors without CSS vars in substantial files: -4
        css_lines = content.split("\n")
        if len(css_lines) > 20:
            hex_ct = len(re.findall(r':\s*#[0-9a-fA-F]{3,8}\b', content))
            rgb_ct = len(re.findall(r':\s*rgb[a]?\(', content))
            if (hex_ct + rgb_ct) >= 5 and 'var(--' not in content:
                if fw_cap < 30:
                    issues.append((0, f"Hardcoded colors ({hex_ct + rgb_ct}) without CSS custom properties", -4))
                    score -= 4
                    fw_cap += 4
        # px font-size (accessibility issue): -4
        if re.search(r'font-size:\s*\d+px', content):
            if fw_cap < 30:
                issues.append((0, "font-size in px — use rem for accessibility", -4))
                score -= 4
                fw_cap += 4

    # HTML
    elif ext in (".html", ".htm"):
        # Missing viewport meta: -4
        if ('<html' in content or '<!doctype' in content.lower()) and 'viewport' not in content:
            if fw_cap < 30:
                issues.append((0, "Missing <meta name=\"viewport\">", -4))
                score -= 4
                fw_cap += 4
        # Missing lang attribute: -4
        html_match = re.search(r'<html\b([^>]*)>', content)
        if html_match and 'lang=' not in html_match.group(1):
            if fw_cap < 30:
                issues.append((0, "Missing lang attribute on <html>", -4))
                score -= 4
                fw_cap += 4
        # Forms without labels: -4
        input_count = len(re.findall(r'<(input|select|textarea)\b', content))
        label_ct = len(re.findall(r'<label\b', content))
        if input_count > 0 and label_ct == 0:
            if fw_cap < 30:
                issues.append((0, f"Form inputs without <label> elements", -4))
                score -= 4
                fw_cap += 4
        # Missing meta description: -4
        if '<head' in content and 'name="description"' not in content and "name='description'" not in content:
            if fw_cap < 30:
                issues.append((0, "Missing <meta name=\"description\"> for SEO", -4))
                score -= 4
                fw_cap += 4
        # Script without defer/async: -4
        if re.search(r'<script\b[^>]*src=(?!.*(?:defer|async|type="module"))', content):
            if fw_cap < 30:
                issues.append((0, "Script tag without defer/async — blocks rendering", -4))
                score -= 4
                fw_cap += 4
        # Interactive div without ARIA: -4
        if re.search(r'<div\b[^>]*(onclick|@click|onClick)', content):
            if fw_cap < 30:
                issues.append((0, "Interactive <div> without role/aria — use <button> or add ARIA", -4))
                score -= 4
                fw_cap += 4

    # --- Import/dependency coherence ---
    import_cap = 0
    import_issues = _check_import_coherence(content, filepath)
    for lineno, desc in import_issues:
        if import_cap < 16:
            issues.append((lineno, f"Import issue: {desc}", -4))
            score -= 4
            import_cap += 4

    return (max(score, 0), issues)


# --- Quality gate state tracking ---
_gate_rejections = {}  # filepath -> rejection count


def _apply_quality_gate(filepath, score, issues):
    """Apply quality gate logic. Returns None to allow, or feedback string to block."""
    global _gate_rejections
    rejections = _gate_rejections.get(filepath, 0)

    if score >= 65:
        # PASS — minor issues noted
        if issues:
            minor_notes = [f"  Line {ln}: {desc}" for ln, desc, _ in issues[:3]]
            _log_quality_score(filepath, score, issues)
            return None  # allow, issues are minor
        _log_quality_score(filepath, score, issues)
        return None

    if score >= 50:
        # WARN but allow
        _log_quality_score(filepath, score, issues)
        warn_lines = [f"  Line {ln}: {desc} ({ded:+d})" for ln, desc, ded in issues[:5]]
        warn_text = "\n".join(warn_lines)
        # Return None to allow, but the warning will be added to tool result
        return None

    # Score < 50 — potential BLOCK
    if rejections >= 3:
        # Safety valve — allow after 3 rejections to prevent infinite loops
        _log_quality_score(filepath, score, issues)
        warn_lines = [f"  Line {ln}: {desc}" for ln, desc, _ in issues[:5]]
        return None  # allow with warnings

    _gate_rejections[filepath] = rejections + 1
    _log_quality_score(filepath, score, issues)

    if rejections >= 2:
        # ESCALATE — explicit constraints
        constraints = []
        for ln, desc, ded in issues:
            constraints.append(f"- MUST FIX: {desc}" + (f" (line {ln})" if ln > 0 else ""))
        constraint_text = "\n".join(constraints[:8])
        return (
            f"QUALITY GATE BLOCKED (score: {score}/100, attempt {rejections + 1}/3)\n"
            f"Rewrite this file following these MANDATORY rules:\n{constraint_text}\n"
            f"Do NOT repeat the same code. Apply ALL fixes before writing again."
        )

    # First/second rejection — structured feedback
    feedback_lines = []
    for ln, desc, ded in issues:
        fix = f"Line {ln}: {desc} ({ded:+d} pts)" if ln > 0 else f"{desc} ({ded:+d} pts)"
        feedback_lines.append(f"  {fix}")
    feedback_text = "\n".join(feedback_lines[:8])
    return (
        f"QUALITY GATE BLOCKED (score: {score}/100)\n"
        f"Issues found:\n{feedback_text}\n"
        f"Fix these issues and rewrite the file. Score must be >= 50 to pass."
    )


def _log_quality_score(filepath, score, issues):
    """Append quality score to ~/.claw/quality_log.jsonl."""
    try:
        import datetime
        log_dir = Path.home() / ".claw"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "quality_log.jsonl"
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "filepath": str(filepath),
            "score": score,
            "issues": [{"line": ln, "desc": desc, "deduction": ded} for ln, desc, ded in issues[:10]],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging is best-effort


# ---------------------------------------------------------------------------
# CSS / Tailwind quality gates — post-generation enhancement
# ---------------------------------------------------------------------------

_GLOBALS_CSS_TEMPLATE = Template("""\
/* --- Rattlesnake Design System --- */
@import url('https://fonts.googleapis.com/css2?family=$heading_font_url:wght@400;600;700&family=$body_font_url:wght@400;500;600&family=$mono_font_url:wght@400;500&display=swap');

:root {
$css_variables
  --font-heading: '$heading_font', serif;
  --font-body: '$body_font', sans-serif;
  --font-mono: '$mono_font', monospace;
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

html {
  font-family: var(--font-body);
  color: var(--text-primary);
  background: var(--bg-primary);
  -webkit-font-smoothing: antialiased;
  scroll-behavior: smooth;
}

h1, h2, h3, h4, h5, h6 { font-family: var(--font-heading); }
code, pre, kbd { font-family: var(--font-mono); }

::selection { background: var(--accent-muted); color: var(--text-primary); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}

@keyframes animate-in {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}
.animate-in { animation: animate-in 0.15s ease-out; }

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.skeleton {
  background: linear-gradient(90deg, rgba(128,128,128,0.1) 25%, rgba(128,128,128,0.2) 50%, rgba(128,128,128,0.1) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 0.25rem;
}
/* --- End Rattlesnake Design System --- */
""")

_MIDNIGHT_PALETTE_VARS = {
    "--bg-primary": "#09090b",
    "--bg-secondary": "#18181b",
    "--bg-tertiary": "#27272a",
    "--surface": "#1c1c1f",
    "--surface-hover": "#252529",
    "--surface-active": "#2e2e33",
    "--border": "#2e2e33",
    "--border-subtle": "#232328",
    "--text-primary": "#fafafa",
    "--text-secondary": "#a1a1aa",
    "--text-tertiary": "#71717a",
    "--text-muted": "#52525b",
    "--accent": "#6366f1",
    "--accent-hover": "#818cf8",
    "--accent-muted": "rgba(99, 102, 241, 0.12)",
    "--success": "#22c55e",
    "--success-muted": "rgba(34, 197, 94, 0.1)",
    "--warning": "#f59e0b",
    "--warning-muted": "rgba(245, 158, 11, 0.1)",
    "--error": "#ef4444",
    "--error-muted": "rgba(239, 68, 68, 0.1)",
    "--info": "#3b82f6",
    "--info-muted": "rgba(59, 130, 246, 0.1)",
}


def _build_globals_css(palette_vars=None, typography=None):
    """Build the full globals.css injection string from palette and typography selections."""
    if palette_vars is None:
        palette_vars = _MIDNIGHT_PALETTE_VARS
    if typography is None:
        typography = {"heading": "JetBrains Mono", "body": "Inter", "mono": "JetBrains Mono"}
    css_var_lines = "\n".join(f"  {k}: {v};" for k, v in palette_vars.items())
    return _GLOBALS_CSS_TEMPLATE.safe_substitute(
        heading_font=typography.get("heading", "JetBrains Mono"),
        body_font=typography.get("body", "Inter"),
        mono_font=typography.get("mono", "JetBrains Mono"),
        heading_font_url=typography.get("heading", "JetBrains Mono").replace(" ", "+"),
        body_font_url=typography.get("body", "Inter").replace(" ", "+"),
        mono_font_url=typography.get("mono", "JetBrains Mono").replace(" ", "+"),
        css_variables=css_var_lines,
    )


_GLOBALS_CSS_INJECTION = _build_globals_css()  # midnight defaults — backward compat alias


def _enhance_globals_css(content, palette_vars=None, typography=None):
    """Inject design system styles into globals.css. Handles new, old, and missing markers."""
    # Path 1: New marker found — already has new design system
    if "Rattlesnake Design System" in content:
        return content
    css_block = _build_globals_css(palette_vars, typography)
    # Path 2: Old marker found — upgrade
    old_start = "/* --- Rattlesnake UI quality base styles --- */"
    old_end = "/* --- End Rattlesnake UI quality base styles --- */"
    if old_start in content:
        start_idx = content.index(old_start)
        if old_end in content:
            end_idx = content.index(old_end) + len(old_end)
            return content[:start_idx] + css_block + content[end_idx:]
        else:
            # Orphaned start marker — inject after it without stripping
            return content[:start_idx + len(old_start)] + "\n" + css_block + content[start_idx + len(old_start):]
    # Path 3: No marker — insert after @tailwind/@import/@layer directives
    lines = content.split("\n")
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("@tailwind") or stripped.startswith("@import"):
            insert_idx = i + 1
    for i in range(insert_idx, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("@layer") or stripped == "}" or stripped == "":
            insert_idx = i + 1
        else:
            break
    lines.insert(insert_idx, css_block)
    return "\n".join(lines)


def _enhance_tailwind_config(content, use_design_system=False):
    """Enforce darkMode: 'class' and inject theme defaults if missing."""
    modified = content
    changes = False

    # 1. Ensure darkMode: 'class' (not 'media') — skip when using design system palettes
    if not use_design_system:
        if "darkMode" not in content:
            config_obj_match = re.search(
                r'(?:export\s+default|module\.exports\s*=|(?:const|let|var)\s+\w+(?:\s*:\s*\w+)?\s*=)\s*\{',
                content
            )
            if config_obj_match:
                idx = config_obj_match.end() - 1
                modified = content[:idx+1] + "\n  darkMode: 'class'," + content[idx+1:]
                changes = True
        elif "'media'" in content or '"media"' in content:
            modified = modified.replace("'media'", "'class'").replace('"media"', '"class"')
            changes = True

    # 2. Ensure extend has theme colors — full CSS variable mapping
    has_any_colors = bool(re.search(r'\bcolors\s*:', content))
    if not has_any_colors and "extend" in modified:
        color_block = """
      colors: {
        'bg-primary': 'var(--bg-primary)',
        'bg-secondary': 'var(--bg-secondary)',
        'bg-tertiary': 'var(--bg-tertiary)',
        'surface': 'var(--surface)',
        'surface-hover': 'var(--surface-hover)',
        'surface-active': 'var(--surface-active)',
        'border': 'var(--border)',
        'border-subtle': 'var(--border-subtle)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-tertiary': 'var(--text-tertiary)',
        'text-muted': 'var(--text-muted)',
        'accent': 'var(--accent)',
        'accent-hover': 'var(--accent-hover)',
        'accent-muted': 'var(--accent-muted)',
        'success': 'var(--success)',
        'success-muted': 'var(--success-muted)',
        'warning': 'var(--warning)',
        'warning-muted': 'var(--warning-muted)',
        'error': 'var(--error)',
        'error-muted': 'var(--error-muted)',
        'info': 'var(--info)',
        'info-muted': 'var(--info-muted)',
      },"""
        extend_match = re.search(r'extend\s*:\s*\{', modified)
        if extend_match:
            insert_pos = extend_match.end()
            modified = modified[:insert_pos] + color_block + modified[insert_pos:]
            changes = True

    # 3. Ensure fontFamily with code font if not present
    if "fontFamily" not in modified and "extend" in modified:
        font_block = """
      fontFamily: {
        sans: ['var(--font-body)', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'Cascadia Code', 'Fira Code', 'monospace'],
        heading: ['var(--font-heading)', 'serif'],
      },"""
        extend_match = re.search(r'extend\s*:\s*\{', modified)
        if extend_match:
            insert_pos = extend_match.end()
            modified = modified[:insert_pos] + font_block + modified[insert_pos:]
            changes = True

    return modified if changes else content


def _enhance_config_security(content, framework=None):
    """Inject security headers into framework config if not present. Returns modified content.

    Supports Next.js config injection. Other frameworks use middleware-based security
    (not config-based), so they return unchanged content.
    """
    fw = framework or "nextjs"
    if fw != "nextjs":
        # Express/Fastify/etc. use helmet/middleware — not config-based headers
        return content
    return _enhance_next_config_security(content)


def _enhance_next_config_security(content):
    """Inject security headers into next.config if not present. Returns modified content."""
    # Skip if already has security headers
    if "X-Content-Type-Options" in content:
        return content
    # Skip large files (likely already complex)
    if len(content) > 20000:
        return content
    # Skip wrapper patterns — don't inject inside function calls
    if re.search(r'(?:defineConfig|withPlugins|createConfig)\s*\(', content):
        return content
    # Find config object declaration
    config_match = re.search(
        r'((?:const\s+\w+\s*=|export\s+default|module\.exports\s*=)\s*)\{',
        content
    )
    if not config_match:
        return content
    # Check if headers() already exists
    if re.search(r'\bheaders\s*\(\s*\)', content):
        return content

    headers_block = """
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
          // Add 'unsafe-eval' to script-src ONLY if your setup specifically requires it (e.g., certain MDX plugins)
          { key: 'Content-Security-Policy', value: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com; connect-src 'self' https://*.supabase.co" },
          // Uncomment for production with HTTPS. Do NOT enable on localhost — HSTS on plain HTTP
          // will lock out local development. Next.js dev server uses HTTP by default.
          // { key: 'Strict-Transport-Security', value: 'max-age=31536000; includeSubDomains' },
        ],
      },
    ];
  },"""

    # Insert after opening brace of config object
    insert_pos = config_match.end()  # position right after the {
    modified = content[:insert_pos] + headers_block + content[insert_pos:]
    return modified


def _verify_file_write(tool_name, tool_args, original_result):
    """After write_file or edit_file, verify the file actually exists, has content, valid syntax, and no placeholders."""
    fp_str = tool_args.get("file_path", "")
    if not fp_str:
        return original_result
    try:
        fp = _resolve(fp_str)
    except ValueError:
        return original_result
    if not fp.exists():
        return original_result + f"\n{VERIFY_FAIL} VERIFICATION FAILED: File {fp} does NOT exist on disk after {tool_name}."
    size = fp.stat().st_size
    if size == 0:
        return original_result + f"\n{VERIFY_FAIL} VERIFICATION FAILED: File {fp} exists but is EMPTY (0 bytes)."

    # --- HTML quality gate: post-validate + Lead-to-Gold enhance ---
    ext = fp.suffix.lower()
    if ext in (".html", ".htm"):
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
            # Post-validate: fix wiring, placeholders, dark mode, etc.
            validated, html_fixes = _post_validate_html(raw)
            if html_fixes:
                original_result += f"\n{VERIFY_OK} HTML quality gate: {len(html_fixes)} fix(es) applied"
            # Lead-to-Gold: enhance HTML
            enhanced = _enhance_html(validated)
            if enhanced != raw:
                fp.write_text(enhanced, encoding="utf-8")
                original_result += f"\n✨ Lead-to-Gold: enhanced HTML ({len(enhanced) - len(raw):+d} chars)"
        except Exception:
            pass

    # --- CSS quality gate: inject base styles for globals.css in Tailwind projects ---
    if ext == ".css" and fp.name.lower() in ("globals.css", "global.css", "app.css", "style.css"):
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
            css_enhanced = _enhance_globals_css(raw)
            if css_enhanced != raw:
                fp.write_text(css_enhanced, encoding="utf-8")
                original_result += f"\n✨ CSS quality gate: injected base UI styles ({len(css_enhanced) - len(raw):+d} chars)"
        except Exception:
            pass

    # --- Tailwind config enforcement ---
    if ext in (".ts", ".js") and fp.name.lower() in ("tailwind.config.ts", "tailwind.config.js"):
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
            tw_enhanced = _enhance_tailwind_config(raw)
            if tw_enhanced != raw:
                fp.write_text(tw_enhanced, encoding="utf-8")
                original_result += f"\n✨ Tailwind config gate: enforced dark mode + theme defaults ({len(tw_enhanced) - len(raw):+d} chars)"
        except Exception:
            pass

    size = fp.stat().st_size
    result = original_result + f"\n{VERIFY_OK} Verified: {fp} exists ({size} bytes)"

    # Syntax validation
    valid, detail = _validate_file_syntax(str(fp))
    if not valid:
        result += f"\n{VERIFY_FAIL} SYNTAX ERROR: {detail}"
    else:
        result += f"\n{VERIFY_OK} Syntax OK"

    # Completeness scan
    incomplete = _scan_for_incomplete_code(str(fp))
    if incomplete:
        issues_str = "; ".join(incomplete[:5])
        result += f"\n{VERIFY_FAIL} INCOMPLETE CODE: {issues_str}"
    else:
        result += f"\n{VERIFY_OK} Completeness OK"

    # Universal placeholder scrub (all file types)
    if _scrub_placeholders(str(fp)):
        result += f"\n{VERIFY_OK} Placeholder scrub: cleaned placeholder text"

    # Auto-lint (if linter available)
    lint_ok, lint_output = _auto_lint(str(fp))
    if lint_ok is not None:
        if lint_ok:
            result += f"\n{VERIFY_OK} Lint OK"
        else:
            result += f"\n{VERIFY_FAIL} LINT: {lint_output[:200]}"

    # Auto-install missing Python imports
    if ext == ".py":
        installed = _check_and_install_imports(str(fp))
        if installed:
            result += f"\n{VERIFY_OK} Auto-installed: {', '.join(installed)}"

    return result


def _check_bash_result(result):
    """Check if a bash command result indicates failure (non-zero exit code)."""
    return "[exit code:" in result and "[exit code: 0]" not in result

# ---------------------------------------------------------------------------
# template system -- pre-built project scaffolds
# ---------------------------------------------------------------------------

def list_templates():
    """List all available project templates."""
    templates = {}
    for tdir in TEMPLATES_DIRS:
        if not tdir.exists():
            continue
        for child in tdir.iterdir():
            if child.is_dir() and (child / "template.json").exists():
                try:
                    meta = json.loads((child / "template.json").read_text(encoding="utf-8"))
                    templates[child.name] = {
                        "path": str(child),
                        "name": meta.get("name", child.name),
                        "description": meta.get("description", ""),
                        "stack": meta.get("stack", []),
                        "features": meta.get("features", []),
                    }
                except Exception:
                    templates[child.name] = {"path": str(child), "name": child.name, "description": ""}
    return templates


def load_template_meta(name):
    """Load template metadata by name. Returns (template_dir_path, meta_dict) or (None, None)."""
    for tdir in TEMPLATES_DIRS:
        candidate = tdir / name
        if candidate.is_dir() and (candidate / "template.json").exists():
            try:
                meta = json.loads((candidate / "template.json").read_text(encoding="utf-8"))
                return str(candidate), meta
            except Exception:
                return str(candidate), {}
    return None, None


# ---------------------------------------------------------------------------
# Build spec generation — blueprint-driven scaffold builds
# ---------------------------------------------------------------------------

# SaaS entity patterns for keyword extraction
_ENTITY_PATTERNS = {
    "task": ["task management", "todo", "project management", "kanban", "issue tracker"],
    "product": ["e-commerce", "ecommerce", "store", "shop", "marketplace", "inventory"],
    "booking": ["booking", "appointment", "reservation", "scheduling", "calendar"],
    "user": ["social", "community", "profile", "membership", "team"],
    "post": ["blog", "cms", "content", "article", "news", "feed"],
    "message": ["chat", "messaging", "communication", "inbox", "notification"],
    "order": ["ordering", "delivery", "food", "restaurant", "menu"],
    "course": ["learning", "education", "tutorial", "lesson", "training"],
    "listing": ["real estate", "rental", "property", "classified", "directory"],
    "event": ["event", "conference", "meetup", "ticket", "rsvp"],
    "invoice": ["invoice", "billing", "accounting", "finance", "payment"],
    "patient": ["health", "medical", "clinic", "wellness", "fitness"],
    "recipe": ["recipe", "cooking", "food blog", "meal plan"],
    "workout": ["workout", "exercise", "gym", "fitness tracker"],
    "job": ["job board", "hiring", "recruitment", "career", "resume"],
}

# Default entity sets per template type
_TEMPLATE_DEFAULT_ENTITIES = {
    "nextjs-supabase": ["item", "category"],
    "nextjs-stripe": ["product", "order"],
    "react-vite": ["item", "category"],
    "express-api": ["resource", "user"],
}


def _extract_entities_from_desc(desc: str) -> list:
    """Extract domain entities from project description using keyword matching + defaults."""
    if not desc:
        return ["item", "category"]
    desc_lower = desc.lower()
    entities = []
    for entity, keywords in _ENTITY_PATTERNS.items():
        if any(kw in desc_lower for kw in keywords):
            entities.append(entity)
    if len(entities) < 2:
        # Creative descriptions like "Tinder for dogs" — provide sensible defaults
        entities = entities or ["item"]
        if len(entities) < 2:
            entities.append("category")
    return entities[:5]  # cap at 5 entities


_TEAM_KEYWORDS = ["team", "workspace", "collaboration", "members", "organization",
                  "org", "multi-tenant", "invite", "shared", "real-time", "colleague"]


def _is_team_app(desc: str) -> bool:
    """Detect if project description implies multi-user/team functionality."""
    if not desc:
        return False
    desc_lower = desc.lower()
    return any(kw in desc_lower for kw in _TEAM_KEYWORDS)


def _generate_build_spec(template_name: str, project_desc: str, project_dir: str, design_selections=None) -> str:
    """Generate a structured BUILD SPEC that tells the model exactly what to create."""
    pdir = Path(project_dir)
    entities = _extract_entities_from_desc(project_desc)

    # List existing files (so model knows NOT to recreate them)
    existing_files = []
    for fpath in sorted(pdir.rglob("*")):
        if not fpath.is_file():
            continue
        parts = fpath.parts
        if any(skip in parts for skip in ("node_modules", ".next", "dist", ".git")):
            continue
        try:
            rel = str(fpath.relative_to(pdir)).replace("\\", "/")
        except ValueError:
            continue
        existing_files.append(rel)

    spec_lines = [
        "# BUILD SPEC — Follow this blueprint exactly",
        "",
        "## EXISTING FILES (DO NOT recreate these)",
    ]
    for f in existing_files:
        spec_lines.append(f"- {f}")

    # Schema section
    spec_lines.extend(["", "## SCHEMA (create in supabase/migrations/ or lib/schema.ts)"])
    for entity in entities:
        plural = entity + "s"
        spec_lines.append(f"### Table: {plural}")
        spec_lines.append(f"  - id (uuid, primary key, default gen_random_uuid())")
        spec_lines.append(f"  - user_id (uuid, references auth.users)")
        spec_lines.append(f"  - name (text, not null)")
        spec_lines.append(f"  - description (text)")
        spec_lines.append(f"  - status (text, default 'active')")
        spec_lines.append(f"  - created_at (timestamptz, default now())")
        spec_lines.append(f"  - updated_at (timestamptz, default now())")

    # Lib/helpers section
    is_team = _is_team_app(project_desc)
    spec_lines.extend(["", "## LIB/HELPERS (create in lib/ or src/lib/)"])
    if is_team:
        spec_lines.append("IMPORTANT: This is a TEAM app. List queries MUST filter by workspace membership, not just creator_id.")
        spec_lines.append("Users should see all items in their workspace, not just items they personally created.")
    else:
        spec_lines.append("This is a personal app. Filter queries by the authenticated user's ID (user_id).")
    spec_lines.append("")
    for entity in entities:
        plural = entity + "s"
        title = entity.title()
        spec_lines.append(f"### lib/{plural}.ts")
        if is_team:
            spec_lines.append(f"  - get{title}s(userId: string, workspaceId?: string): Promise<{title}[]>")
            spec_lines.append(f"    → Filter by workspace membership, NOT by creator_id alone")
        else:
            spec_lines.append(f"  - get{title}s(userId: string): Promise<{title}[]>")
            spec_lines.append(f"    → Filter by user_id (owner)")
        spec_lines.append(f"  - get{title}ById(id: string): Promise<{title} | null>")
        spec_lines.append(f"  - create{title}(data: Create{title}Input): Promise<{title}>")
        spec_lines.append(f"  - update{title}(id: string, data: Partial<{title}>): Promise<{title}>")
        spec_lines.append(f"  - delete{title}(id: string): Promise<void>")

    # Validation section
    spec_lines.extend(["", "## VALIDATION (create in lib/validations.ts)"])
    spec_lines.append("Write manual validation functions — NO external libraries (no Zod, no Yup).")
    spec_lines.append("Use typeof checks, string length, required field presence, etc.")
    for entity in entities:
        title = entity.title()
        spec_lines.append(f"  - validate{title}Input(data: unknown): {{ success: boolean, data?: Create{title}Input, error?: string }}")

    # API routes section
    spec_lines.extend(["", "## API ROUTES (create in app/api/)"])
    for entity in entities:
        plural = entity + "s"
        title = entity.title()
        spec_lines.append(f"### /api/{plural}")
        spec_lines.append(f"  - GET  /api/{plural}        → list all for current user")
        spec_lines.append(f"  - POST /api/{plural}        → validate body with validate{title}Input(), then create")
        spec_lines.append(f"  - GET  /api/{plural}/[id]   → get single {entity}")
        spec_lines.append(f"  - PUT  /api/{plural}/[id]   → validate body (partial), then update")
        spec_lines.append(f"  - DELETE /api/{plural}/[id]  → delete {entity}")

    # Components section
    spec_lines.extend(["", "## COMPONENTS (create in components/ or src/components/)"])
    for entity in entities:
        title = entity.title()
        spec_lines.append(f"### {title} components")
        spec_lines.append(f"  - {title}Card.tsx — displays single {entity} with actions")
        spec_lines.append(f"  - {title}Form.tsx — 'use client' create/edit form with validation")
        spec_lines.append(f"  - {title}List.tsx — 'use client' list with loading/empty states")
    spec_lines.append("### Shared components")
    spec_lines.append("  - Sidebar.tsx — navigation sidebar with links to all pages")
    spec_lines.append("  - Header.tsx — top bar with user info and actions")

    # Pages section
    spec_lines.extend(["", "## PAGES (create in app/ directory)"])
    spec_lines.append("### app/dashboard/layout.tsx — dashboard shell layout")
    spec_lines.append("  - Server component that wraps ALL dashboard pages")
    spec_lines.append("  - Renders Sidebar + Header ONCE, children render inside main content area")
    spec_lines.append("  - Auth check: redirect to /login if no user")
    spec_lines.append("  - Individual pages must NOT import Sidebar or Header — the layout handles it")
    spec_lines.append("")
    spec_lines.append("### app/dashboard/page.tsx — main dashboard with overview stats")
    for entity in entities:
        plural = entity + "s"
        spec_lines.append(f"### app/dashboard/{plural}/page.tsx — list/manage {plural}")
        spec_lines.append(f"### app/dashboard/{plural}/new/page.tsx — create new {entity}")

    # File creation order
    spec_lines.extend([
        "",
        "## FILE CREATION ORDER (follow exactly)",
        "1. Database schema / migrations",
        "2. Type definitions (types.ts)",
        "3. Validation functions (lib/validations.ts)",
        "4. Lib helpers (CRUD functions)",
        "5. API route handlers (with validation)",
        "6. Reusable components (Card, Form, List)",
        "7. Shared components (Sidebar, Header)",
        "8. Dashboard layout (app/dashboard/layout.tsx with Sidebar + Header)",
        "9. Pages (dashboard, entity pages) — do NOT import Sidebar/Header, layout handles it",
        "10. Add any missing dependencies to package.json LAST",
    ])

    # Design system section
    if design_selections:
        palette_name = design_selections.get("palette_name", "midnight")
        typography = design_selections.get("typography", {})
        spec_lines.extend(["", "## DESIGN SYSTEM (follow exactly)", ""])
        spec_lines.append(f"### Palette: {palette_name}")
        spec_lines.append("All colors use CSS variables. NEVER use raw Tailwind colors (gray-800, blue-500).")
        spec_lines.append("Use: bg-[var(--bg-primary)], text-[var(--text-secondary)], border-[var(--border)], etc.")
        spec_lines.append("")
        spec_lines.append(f"### Typography: {typography.get('name', 'technical')}")
        spec_lines.append(f"Heading: {typography.get('heading', 'JetBrains Mono')}, Body: {typography.get('body', 'Inter')}, Mono: {typography.get('mono', 'JetBrains Mono')}")
        spec_lines.append("")
        spec_lines.append("### Component → Recipe mapping:")
        spec_lines.append("- Sidebar component → use `sidebar` recipe from design context")
        spec_lines.append("- Header component → use `header` recipe")
        for entity in entities:
            title = entity.title()
            spec_lines.append(f"- {title}Card → use `card.default` recipe, `badge` for status")
            spec_lines.append(f"- {title}Form → use `input.text` + `input.label` + `input.error_text` + `button.primary`")
            spec_lines.append(f"- {title}List → use `data_table` recipe with `pagination`")
        spec_lines.append("- Dashboard stat cards → use `stat_card` recipe (max 4 per row)")
        spec_lines.append("- All modals → use `modal` recipe")
        spec_lines.append("- All empty states → use `empty_state` recipe (REQUIRED on every list/table)")
        spec_lines.append("- All loading states → use `skeleton_loader` recipe (REQUIRED on every async fetch)")
        spec_lines.append("- All error states → use `alert_banner.error` recipe")
        spec_lines.append("- For page types not listed above → follow general patterns from design context, do not improvise visual decisions")

    # Security section (always included)
    spec_lines.extend(["", "## SECURITY (follow exactly)", ""])
    spec_lines.append("### Input Validation")
    spec_lines.append("- POST/PUT/PATCH: validate request body BEFORE processing")
    spec_lines.append("- GET with params: validate query parameters and URL params")
    spec_lines.append("- Use validate*Input() functions from lib/validations.ts")
    spec_lines.append("")
    spec_lines.append("### Authentication")
    spec_lines.append("- Call getUser() or getSession() at the start of EVERY protected handler")
    spec_lines.append("- Return 401 if no valid session")
    spec_lines.append("- Filter data by user_id to prevent data leakage")
    spec_lines.append("")
    spec_lines.append("### Security Headers")
    spec_lines.append("- next.config.js must include headers() with: X-Content-Type-Options, X-Frame-Options,")
    spec_lines.append("  Content-Security-Policy, Referrer-Policy, Permissions-Policy, Strict-Transport-Security")
    spec_lines.append("")
    spec_lines.append("### Error Handling")
    spec_lines.append("- NEVER expose stack traces or internal errors to users")
    spec_lines.append("- Return generic error messages (400/401/404/500) with safe descriptions")
    spec_lines.append("- Wrap all DB operations in try/catch")
    spec_lines.append("")
    spec_lines.append("### Secrets")
    spec_lines.append("- All secrets in .env (never hardcoded)")
    spec_lines.append("- Server secrets WITHOUT NEXT_PUBLIC_ prefix")
    spec_lines.append("- .env.example with placeholders only")

    return "\n".join(spec_lines)


def _pluralize(word):
    """Simple English pluralization for entity names."""
    if word.endswith('y') and word[-2:] not in ('ay', 'ey', 'oy', 'uy'):
        return word[:-1] + 'ies'
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    return word + 's'


def _sanitize_generated_code(content):
    """Strip trailing JSON wrapper garbage that local models leave in generated code."""
    if not content:
        return content
    # Pattern: code ends with `\n}\n} or "\n}\n} from JSON tool call wrapper
    # Strip trailing backticks + braces
    content = re.sub(r'[`"]\s*\n\s*\}\s*\n\s*\}\s*$', '', content)
    # Strip orphan closing braces at the very end that don't match opens
    lines = content.rstrip().split('\n')
    while lines and re.match(r'^\s*[}\]]\s*$', lines[-1]):
        # Count braces in the whole content to see if this is orphaned
        full = '\n'.join(lines)
        opens = full.count('{') + full.count('[')
        closes = full.count('}') + full.count(']')
        if closes > opens:
            lines.pop()
        else:
            break
    content = '\n'.join(lines)
    # Strip trailing backtick on its own line
    content = re.sub(r'\n`\s*$', '', content)
    return content.rstrip() + '\n'


def _extract_scaffold_file_list(entities, project_desc, is_team=False):
    """
    Derive the ordered file list for scaffold, matching _generate_build_spec structure.
    Returns list of dicts: [{"path": "lib/types.ts", "purpose": "...", "depends_on": [...]}, ...]
    """
    files = []

    # 1. Types
    files.append({"path": "lib/types.ts",
                  "purpose": "TypeScript type definitions and interfaces for all entities",
                  "depends_on": []})

    # 2. Validation
    entity_titles = [e.title() for e in entities]
    files.append({"path": "lib/validations.ts",
                  "purpose": f"Input validation functions: {', '.join(f'validate{t}Input()' for t in entity_titles)}",
                  "depends_on": ["lib/types.ts"]})

    # 3. CRUD helpers per entity
    for entity in entities:
        plural = _pluralize(entity)
        title = entity.title()
        team_note = " Filter by workspace membership." if is_team else " Filter by user_id."
        files.append({"path": f"lib/{plural}.ts",
                      "purpose": f"CRUD functions for {plural}: get{title}s, get{title}ById, create{title}, update{title}, delete{title}. Use createClient from '@/lib/supabase/server'.{team_note}",
                      "depends_on": ["lib/types.ts"]})

    # 4. API routes per entity
    for entity in entities:
        plural = _pluralize(entity)
        title = entity.title()
        files.append({"path": f"app/api/{plural}/route.ts",
                      "purpose": f"GET (list) + POST (create) handlers for {plural}. Use validate{title}Input() for POST. Import createClient from '@/lib/supabase/server'.",
                      "depends_on": [f"lib/{plural}.ts", "lib/validations.ts", "lib/types.ts"]})
        files.append({"path": f"app/api/{plural}/[id]/route.ts",
                      "purpose": f"GET/PUT/DELETE handlers for a single {entity} by ID. Import createClient from '@/lib/supabase/server'.",
                      "depends_on": [f"lib/{plural}.ts", "lib/types.ts"]})

    # 5. Components per entity
    for entity in entities:
        plural = _pluralize(entity)
        title = entity.title()
        files.append({"path": f"components/{title}Card.tsx",
                      "purpose": f"Display card component for a single {entity}. Shows name, status, actions. Must start with 'use client'.",
                      "depends_on": ["lib/types.ts"]})
        files.append({"path": f"components/{title}Form.tsx",
                      "purpose": f"'use client' form component for creating/editing a {entity}. Use manual validation (typeof checks), NOT Zod. Must start with 'use client'. Import useState from react.",
                      "depends_on": ["lib/types.ts"]})
        files.append({"path": f"components/{title}List.tsx",
                      "purpose": f"'use client' list component for {plural}. Fetches from /api/{plural}. Shows loading/empty states. Must start with 'use client'. Import useState, useEffect from react.",
                      "depends_on": ["lib/types.ts", f"components/{title}Card.tsx"]})

    # 6. Shared components
    entity_plurals = [_pluralize(e) for e in entities]
    files.append({"path": "components/Sidebar.tsx",
                  "purpose": f"Navigation sidebar with links to dashboard and entity pages: {', '.join(entity_plurals)}.",
                  "depends_on": []})
    files.append({"path": "components/Header.tsx",
                  "purpose": "Top bar with user email display and sign-out button.",
                  "depends_on": []})

    # 7. Dashboard layout
    files.append({"path": "app/dashboard/layout.tsx",
                  "purpose": "Dashboard shell layout. Server component (NO 'use client'). Auth check: use redirect() from next/navigation (NOT useRouter). Renders Sidebar + Header + children.",
                  "depends_on": ["components/Sidebar.tsx", "components/Header.tsx"]})

    # 8. Dashboard page
    files.append({"path": "app/dashboard/page.tsx",
                  "purpose": f"Main dashboard with overview stats for {', '.join(entity_plurals)}.",
                  "depends_on": ["lib/types.ts"]})

    # 9. Entity pages
    for entity in entities:
        plural = _pluralize(entity)
        title = entity.title()
        files.append({"path": f"app/dashboard/{plural}/page.tsx",
                      "purpose": f"List/manage {plural} page. Uses {title}List component. Must start with 'use client'.",
                      "depends_on": [f"components/{title}List.tsx"]})
        files.append({"path": f"app/dashboard/{plural}/new/page.tsx",
                      "purpose": f"Create new {entity} page. Uses {title}Form component. Must start with 'use client'.",
                      "depends_on": [f"components/{title}Form.tsx"]})

    return files


def _extract_export_summary(content, filepath):
    """Extract a brief summary of exports from file content."""
    exports = []
    for m in re.finditer(r'export\s+(?:type|interface)\s+(\w+)', content):
        exports.append(m.group(1))
    for m in re.finditer(r'export\s+(?:async\s+)?(?:function|const)\s+(\w+)', content):
        exports.append(m.group(1))
    for m in re.finditer(r'export\s+default\s+(?:async\s+)?(?:function|class)\s+(\w+)', content):
        exports.append(f"default {m.group(1)}")
    if not exports:
        # Check for export { ... }
        for m in re.finditer(r'export\s*\{([^}]+)\}', content):
            exports.extend(n.strip() for n in m.group(1).split(",") if n.strip())
    return ", ".join(exports[:15]) if exports else "(no named exports)"


def _append_to_context(context_path, filepath, export_summary):
    """Append a created file entry to CONTEXT.md."""
    current = context_path.read_text(encoding="utf-8")
    entry = f"- src/{filepath}: {export_summary}\n"
    if "## Created Files" in current:
        current = current.replace("## Created Files\n", f"## Created Files\n{entry}", 1)
    else:
        current += f"\n{entry}"
    context_path.write_text(current, encoding="utf-8")


def _extract_code_from_response(text, expected_ext=".tsx"):
    """Extract code from a text response when model dumps code instead of using write_file."""
    if not text or len(text) < 15:
        return None

    # Tier 1: find largest code block with language tag
    lang_pattern = r'```(?:tsx|ts|typescript|javascript|jsx|json)\s*\n(.*?)```'
    blocks = re.findall(lang_pattern, text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()

    # Tier 2: any code block > 20 chars
    any_blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    valid = [b.strip() for b in any_blocks if len(b.strip()) > 20]
    if valid:
        return max(valid, key=len)

    # Tier 3: heuristic — from first import/export/comment to last }
    lines = text.split('\n')
    start = end = -1
    for i, line in enumerate(lines):
        if start == -1 and re.match(r'^\s*(import |export |"use client"|//|/\*)', line):
            start = i
        if re.match(r'^\s*[}\]];?\s*$', line):
            end = i
    if start >= 0 and end > start:
        code = '\n'.join(lines[start:end + 1]).strip()
        if len(code) > 20:
            return code

    return None


def scaffold_from_template(template_name, project_name, target_dir=None):
    """
    Copy a template into target_dir, replacing {{PROJECT_NAME}} placeholders.
    Returns (success_bool, message_string).
    """
    tpl_path, meta = load_template_meta(template_name)
    if not tpl_path:
        available = list(list_templates().keys())
        return False, f"Template '{template_name}' not found. Available: {', '.join(available) or 'none'}"

    target = Path(target_dir or CWD) / project_name
    if target.exists() and any(target.iterdir()):
        return False, f"Directory {target} already exists and is not empty."

    # copy template tree
    shutil.copytree(tpl_path, str(target), dirs_exist_ok=True)

    # remove template.json from the copy (it's metadata, not project code)
    tpl_json = target / "template.json"
    if tpl_json.exists():
        tpl_json.unlink()

    # replace placeholders in all text files
    replacements = {"{{PROJECT_NAME}}": project_name}
    variables = meta.get("variables", {})
    for key, default_val in variables.items():
        replacements[f"{{{{{key}}}}}"] = default_val

    # always override PROJECT_NAME
    replacements["{{PROJECT_NAME}}"] = project_name

    text_extensions = {".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".css", ".html", ".sql", ".env", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
    for fpath in target.rglob("*"):
        if fpath.is_file() and fpath.suffix in text_extensions:
            try:
                content = fpath.read_text(encoding="utf-8")
                for placeholder, value in replacements.items():
                    content = content.replace(placeholder, value)
                fpath.write_text(content, encoding="utf-8")
            except Exception:
                pass
        # also handle extensionless dot files
        if fpath.is_file() and fpath.name.startswith(".") and fpath.suffix == "":
            try:
                content = fpath.read_text(encoding="utf-8")
                for placeholder, value in replacements.items():
                    content = content.replace(placeholder, value)
                fpath.write_text(content, encoding="utf-8")
            except Exception:
                pass

    # count files
    file_count = sum(1 for _ in target.rglob("*") if _.is_file())
    return True, f"Scaffolded '{template_name}' into {target} ({file_count} files)"


def _post_scaffold_enhance(project_dir, design_selections=None):
    """Run CSS/Tailwind quality gates on project files after scaffold + customization."""
    palette_vars = design_selections.get("palette_vars") if design_selections else None
    typography = design_selections.get("typography") if design_selections else None
    has_design_system = bool(design_selections)
    pdir = Path(project_dir)
    for fpath in pdir.rglob("*"):
        if not fpath.is_file():
            continue
        fname = fpath.name.lower()
        ext = fpath.suffix.lower()
        if ext == ".css" and fname in ("globals.css", "global.css", "app.css", "style.css"):
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                enhanced = _enhance_globals_css(raw, palette_vars=palette_vars, typography=typography)
                if enhanced != raw:
                    fpath.write_text(enhanced, encoding="utf-8")
            except Exception:
                pass
        if ext in (".ts", ".js") and fname in ("tailwind.config.ts", "tailwind.config.js"):
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                enhanced = _enhance_tailwind_config(raw, use_design_system=has_design_system)
                if enhanced != raw:
                    fpath.write_text(enhanced, encoding="utf-8")
            except Exception:
                pass
        # Security: enhance config with security headers (framework-aware)
        if fname in ("next.config.mjs", "next.config.js", "next.config.ts"):
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                enhanced = _enhance_config_security(raw, framework="nextjs")
                if enhanced != raw:
                    fpath.write_text(enhanced, encoding="utf-8")
            except Exception:
                pass


def _repair_json_files(project_dir: str) -> list:
    """Scan all .json files (skip node_modules), repair broken ones. Returns list of repaired paths."""
    pdir = Path(project_dir)
    repaired = []
    for fpath in pdir.rglob("*.json"):
        # Skip node_modules, .next, dist, etc.
        parts = fpath.parts
        if any(skip in parts for skip in ("node_modules", ".next", "dist", ".git")):
            continue
        try:
            raw = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Check if already valid
        try:
            json.loads(raw)
            continue  # valid, skip
        except json.JSONDecodeError:
            pass
        # Try to repair
        fixed, was_fixed = _fix_json_syntax(raw)
        if was_fixed:
            try:
                fpath.write_text(fixed, encoding="utf-8")
                try:
                    rel = str(fpath.relative_to(pdir))
                except ValueError:
                    rel = str(fpath)
                repaired.append(rel)
            except Exception:
                pass
    return repaired


# ---------------------------------------------------------------------------
# API knowledge registry -- anti-hallucination for external APIs
# ---------------------------------------------------------------------------

def load_api_registry():
    """Load the API registry index."""
    for adir in APIS_DIR_PATHS:
        registry_file = adir / "registry.json"
        if registry_file.exists():
            try:
                return json.loads(registry_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {"apis": {}}


def load_api_patterns(api_name):
    """Load API patterns for a specific API (e.g., 'stripe', 'supabase')."""
    for adir in APIS_DIR_PATHS:
        api_file = adir / f"{api_name}.json"
        if api_file.exists():
            try:
                return json.loads(api_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def find_relevant_apis(query):
    """Given a query string, find APIs whose keywords match. Returns list of api_name strings."""
    registry = load_api_registry()
    query_lower = query.lower()
    matched = []
    for api_name, info in registry.get("apis", {}).items():
        keywords = info.get("keywords", [])
        if any(kw in query_lower for kw in keywords) or api_name in query_lower:
            matched.append(api_name)
    return matched


def get_api_context_for_prompt(query):
    """Build API context string to inject into the system prompt based on the query/plan."""
    relevant = find_relevant_apis(query)
    if not relevant:
        return ""
    parts = []
    for api_name in relevant:
        patterns = load_api_patterns(api_name)
        if not patterns:
            continue
        section = f"\n### {patterns.get('name', api_name).upper()} API Reference\n"
        section += f"SDK: {patterns.get('sdk', 'unknown')}\n"
        section += f"Setup: {patterns.get('setup', {}).get('install', '')}\n\n"
        # include key patterns
        for pname, pdata in patterns.get("patterns", {}).items():
            section += f"**{pname}**: {pdata.get('description', '')}\n"
            section += f"```\n{pdata.get('code', '')}\n```\n\n"
        # include common mistakes
        mistakes = patterns.get("common_mistakes", [])
        if mistakes:
            section += "**Common mistakes to AVOID:**\n"
            for m in mistakes:
                section += f"- {m}\n"
        parts.append(section)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# build-verify-fix loop -- self-healing after project generation
# ---------------------------------------------------------------------------

def detect_project_type(project_dir):
    """
    Detect the project type, framework, ORM, and determine the correct
    build/dev/start commands. Handles complex enterprise stacks.
    Returns a dict with: type, framework, commands, dev_cmd, has_typescript,
    has_prisma, has_docker, entry_file, etc.
    """
    pdir = Path(project_dir)
    info = {
        "type": "unknown",
        "framework": None,
        "commands": [],         # build pipeline: [(step_name, cmd)]
        "dev_cmd": None,        # the command to start dev server
        "has_typescript": False,
        "has_prisma": False,
        "has_docker": False,
        "entry_file": None,     # main HTML/py file for static/python
    }

    # --- Node / JS project ---
    if (pdir / "package.json").exists():
        info["type"] = "node"
        try:
            pkg = json.loads((pdir / "package.json").read_text(encoding="utf-8"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            scripts = pkg.get("scripts", {})

            # Detect framework
            if "next" in deps:
                info["framework"] = "nextjs"
            elif "nuxt" in deps or "nuxt3" in deps:
                info["framework"] = "nuxt"
            elif "vite" in deps:
                info["framework"] = "vite"
            elif "react-scripts" in deps:
                info["framework"] = "cra"
            elif "svelte" in deps or "@sveltejs/kit" in deps:
                info["framework"] = "svelte"
            elif "express" in deps:
                info["framework"] = "express"
            elif "fastify" in deps:
                info["framework"] = "fastify"

            # Detect ORM / DB
            if "@prisma/client" in deps or "prisma" in deps:
                info["has_prisma"] = True
            if "drizzle-orm" in deps:
                info["has_drizzle"] = True
            if "mongoose" in deps:
                info["has_mongoose"] = True

            # TypeScript
            if "typescript" in deps or (pdir / "tsconfig.json").exists():
                info["has_typescript"] = True

            # Build commands — install first, always
            info["commands"] = [("install", "npm install")]

            # Prisma: generate client after install, before build
            if info["has_prisma"]:
                info["commands"].append(("prisma_generate", "npx prisma generate"))

            # Build if script exists
            if "build" in scripts:
                info["commands"].append(("build", "npm run build"))

            # Lint if script exists (non-fatal)
            if "lint" in scripts:
                info["commands"].append(("lint", "npm run lint"))

            # Dev command — pick the right one
            if "dev" in scripts:
                info["dev_cmd"] = "npm run dev"
            elif "start" in scripts:
                info["dev_cmd"] = "npm start"
            elif info["framework"] == "nextjs":
                info["dev_cmd"] = "npx next dev"
            elif info["framework"] == "vite":
                info["dev_cmd"] = "npx vite"

        except Exception:
            # package.json exists but can't parse — basic fallback
            info["commands"] = [("install", "npm install")]
            info["dev_cmd"] = "npm run dev"

    # --- Python project ---
    elif (pdir / "requirements.txt").exists() or (pdir / "pyproject.toml").exists():
        info["type"] = "python"

        if (pdir / "requirements.txt").exists():
            info["commands"] = [("install", "pip install -r requirements.txt")]
        elif (pdir / "pyproject.toml").exists():
            info["commands"] = [("install", "pip install -e .")]

        # Detect framework by scanning py files (limit to top-level + 1 deep)
        py_files = list(pdir.glob("*.py")) + list(pdir.glob("*/*.py"))
        for f in py_files[:30]:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")[:5000]
                if "FastAPI" in content or "fastapi" in content.lower():
                    info["framework"] = "fastapi"
                    info["entry_file"] = str(f.relative_to(pdir))
                    break
                if "Django" in content or "django" in content.lower():
                    info["framework"] = "django"
                    break
                if "Flask" in content or "flask" in content.lower():
                    info["framework"] = "flask"
                    info["entry_file"] = str(f.relative_to(pdir))
                    break
                if "streamlit" in content.lower():
                    info["framework"] = "streamlit"
                    info["entry_file"] = str(f.relative_to(pdir))
                    break
            except Exception:
                pass

        # Prisma in Python (prisma-client-py)
        if (pdir / "schema.prisma").exists() or (pdir / "prisma").is_dir():
            info["has_prisma"] = True
            info["commands"].append(("prisma_generate", "prisma generate"))

        # Dev command
        if info["framework"] == "fastapi":
            entry = info.get("entry_file", "main.py")
            module = entry.replace("/", ".").replace("\\", ".").replace(".py", "")
            info["dev_cmd"] = f"python -m uvicorn {module}:app --host 0.0.0.0 --port 8000 --reload"
        elif info["framework"] == "django":
            info["dev_cmd"] = "python manage.py runserver"
        elif info["framework"] == "flask":
            entry = info.get("entry_file", "app.py")
            info["dev_cmd"] = f"python {entry}"
        elif info["framework"] == "streamlit":
            entry = info.get("entry_file", "app.py")
            info["dev_cmd"] = f"streamlit run {entry}"

    # --- Static HTML project ---
    elif any(pdir.glob("*.html")):
        info["type"] = "static"
        info["commands"] = []  # no build steps
        # Find the main HTML file
        for name in ("index.html", "home.html", "main.html"):
            if (pdir / name).exists():
                info["entry_file"] = name
                break
        if not info["entry_file"]:
            html_files = sorted(pdir.glob("*.html"))
            if html_files:
                info["entry_file"] = html_files[0].name

    # --- Docker ---
    if (pdir / "Dockerfile").exists() or (pdir / "docker-compose.yml").exists() or (pdir / "docker-compose.yaml").exists():
        info["has_docker"] = True

    # --- .env check ---
    info["has_env"] = (pdir / ".env").exists()
    info["has_env_example"] = (pdir / ".env.example").exists()

    return info


# ---------------------------------------------------------------------------
# Structured error parser (Phase 3a)
# ---------------------------------------------------------------------------

# Source file extension pattern for error matching
_SRC_EXT_RE = r'\.(?:tsx?|jsx?|py|mjs|cjs)'

_ERROR_PATTERNS = [
    # TypeScript: src/app/page.tsx(14,5): error TS2345: ...
    re.compile(r'^(.+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)$', re.MULTILINE),
    # Generic: ./src/app/page.tsx:14:5 (only source file extensions)
    re.compile(rf'^[./]*(.+?{_SRC_EXT_RE}):(\d+):(\d+)', re.MULTILINE),
    # Module not found
    re.compile(r"Module not found:\s*Can't resolve '([^']+)'\s+in\s+'([^']+)'"),
    # Python traceback: File "path/to/file.py", line 42, in function_name
    re.compile(r'File "(.+?)", line (\d+)(?:, in (\w+))?'),
]


def _parse_build_errors(output):
    """Parse build error output into structured dicts.
    Returns list of {file, line, column, message, category}.
    """
    errors = []
    seen = set()

    # TypeScript errors: file(line,col): error TSxxxx: message
    for m in re.finditer(r'^(.+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)$', output, re.MULTILINE):
        key = (m.group(1), m.group(2), m.group(4))
        if key not in seen:
            seen.add(key)
            errors.append({
                'file': m.group(1).strip(),
                'line': int(m.group(2)),
                'column': int(m.group(3)),
                'message': f"{m.group(4)}: {m.group(5).strip()}",
                'category': 'typescript',
            })

    # Generic source:line:col errors
    for m in re.finditer(rf'^[./]*(.+?{_SRC_EXT_RE}):(\d+):(\d+)', output, re.MULTILINE):
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            # Grab rest of line for message
            line_start = m.end()
            line_end = output.find('\n', line_start)
            msg = output[line_start:line_end].strip(' :') if line_end > line_start else ''
            errors.append({
                'file': m.group(1).strip(),
                'line': int(m.group(2)),
                'column': int(m.group(3)),
                'message': msg[:200],
                'category': 'build',
            })

    # Module not found
    for m in re.finditer(r"Module not found:\s*Can't resolve '([^']+)'\s+in\s+'([^']+)'", output):
        key = ('module_not_found', m.group(1))
        if key not in seen:
            seen.add(key)
            errors.append({
                'file': m.group(2).strip(),
                'line': 0,
                'column': 0,
                'message': f"Can't resolve '{m.group(1)}'",
                'category': 'module_not_found',
            })

    # Python tracebacks
    for m in re.finditer(r'File "(.+?)", line (\d+)(?:, in (\w+))?', output):
        fpath = m.group(1).strip()
        # Skip stdlib/venv paths
        if 'site-packages' in fpath or '/lib/python' in fpath:
            continue
        key = (fpath, m.group(2))
        if key not in seen:
            seen.add(key)
            func = m.group(3) or ''
            errors.append({
                'file': fpath,
                'line': int(m.group(2)),
                'column': 0,
                'message': f"in {func}" if func else '',
                'category': 'python_traceback',
            })

    return errors


def _format_structured_errors(errors, raw_output, project_dir=None):
    """Format parsed errors with optional graph context. Falls back to raw if no errors parsed."""
    if not errors:
        return raw_output[-2000:] if len(raw_output) > 2000 else raw_output

    lines = [f"Parsed {len(errors)} error(s):"]
    error_files = set()
    for err in errors[:15]:  # cap at 15 errors
        loc = f"{err['file']}:{err['line']}"
        if err['column']:
            loc += f":{err['column']}"
        lines.append(f"  [{err['category']}] {loc} — {err['message']}")
        error_files.add(err['file'])

    if len(errors) > 15:
        lines.append(f"  ... and {len(errors) - 15} more")

    # Add graph context for affected files
    if _project_graph is not None and error_files:
        # Normalize paths relative to project
        seeds = []
        for ef in error_files:
            norm = ef.replace('\\', '/')
            if norm.startswith('./'):
                norm = norm[2:]
            seeds.append(norm)
        graph_ctx = _build_graph_context(seeds, max_tokens=400)
        if graph_ctx:
            lines.append("")
            lines.append(graph_ctx)

    return '\n'.join(lines)


def build_and_verify(project_dir, messages, model, auto_stub=False):
    """
    Run the build-verify-fix loop on a project directory.
    Handles: Node (Next.js, Vite, Express, CRA), Python (FastAPI, Django, Flask),
    static HTML, Prisma, Docker, .env setup.
    Returns (success_bool, summary_string).
    """
    pdir = Path(project_dir)
    if not pdir.exists():
        return False, f"Project directory {pdir} does not exist."

    project_info = detect_project_type(str(pdir))
    ptype = project_info["type"]
    framework = project_info.get("framework")

    print(f"  {C.CLAW}{BLACK_CIRCLE} Detected: {ptype}" +
          (f" ({framework})" if framework else "") +
          f"{C.RESET}")

    if ptype == "unknown":
        return True, "Could not detect project type — skipping build verification."

    if ptype == "static":
        html_files = list(pdir.glob("*.html"))
        entry = project_info.get("entry_file") or (html_files[0].name if html_files else None)
        if html_files:
            print(f"  {C.SUCCESS}{BLACK_CIRCLE} Static project: {len(html_files)} HTML file(s) — open {entry} in browser{C.RESET}")
            return True, f"Static site. Open {entry} in a browser."
        return True, "Static project — no build steps needed."

    results = []
    overall_success = True
    MAX_FIX_ATTEMPTS = 3

    # --- Pre-flight: .env setup ---
    if project_info.get("has_env_example") and not project_info.get("has_env"):
        print(f"  {C.TOOL}{BLACK_CIRCLE} Creating .env from .env.example...{C.RESET}")
        try:
            example_content = (pdir / ".env.example").read_text(encoding="utf-8")
            (pdir / ".env").write_text(example_content, encoding="utf-8")
            results.append(f"{VERIFY_OK} .env: created from .env.example")
        except Exception as e:
            results.append(f"{VERIFY_FAIL} .env: could not create — {e}")

    # --- Run build pipeline commands ---
    for step_name, cmd in project_info["commands"]:
        success = False
        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            print(f"  {C.TOOL}{BLACK_CIRCLE} {step_name}: `{cmd}` (attempt {attempt}/{MAX_FIX_ATTEMPTS}){C.RESET}")
            try:
                r = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=180, cwd=str(pdir)
                )
                output = ""
                if r.stdout:
                    output += r.stdout
                if r.stderr:
                    output += ("\n" if output else "") + r.stderr

                if r.returncode == 0:
                    print(f"    {VERIFY_OK} {step_name} passed")
                    results.append(f"{VERIFY_OK} {step_name}: passed")
                    success = True
                    break
                else:
                    print(f"    {VERIFY_FAIL} {step_name} failed (exit {r.returncode})")
                    # Parse structured errors (Phase 3a) — fall back to raw on 0 matches
                    parsed_errors = _parse_build_errors(output)
                    error_excerpt = _format_structured_errors(parsed_errors, output, project_dir=str(pdir))
                    if attempt < MAX_FIX_ATTEMPTS:
                        fix_prompt = (
                            f"The '{step_name}' step (`{cmd}`) failed with exit code {r.returncode}.\n"
                            f"Project type: {ptype}" + (f" ({framework})" if framework else "") + "\n"
                            f"Error output:\n```\n{error_excerpt}\n```\n\n"
                            f"Fix the error in the project files at: {pdir}\n"
                            f"If a dependency is missing, add it. If a config is wrong, fix it."
                        )
                        messages.append({"role": "user", "content": fix_prompt})
                        try:
                            run_agent_turn(messages, model, use_tools=True)
                        except Exception as e:
                            print(f"    {C.ERROR}Fix attempt failed: {e}{C.RESET}")
                    else:
                        results.append(f"{VERIFY_FAIL} {step_name}: failed after {MAX_FIX_ATTEMPTS} attempts")
            except subprocess.TimeoutExpired:
                print(f"    {VERIFY_FAIL} {step_name} timed out (180s)")
                results.append(f"{VERIFY_FAIL} {step_name}: timed out")
                break
            except Exception as e:
                print(f"    {VERIFY_FAIL} {step_name} error: {e}")
                results.append(f"{VERIFY_FAIL} {step_name}: {e}")
                break

        if not success:
            overall_success = False

    # --- Startup test: use the detected dev_cmd ---
    start_cmd = project_info.get("dev_cmd")

    if start_cmd:
        print(f"  {C.TOOL}{BLACK_CIRCLE} Startup test: `{start_cmd}`{C.RESET}")
        try:
            proc = subprocess.Popen(
                start_cmd, shell=True, cwd=str(pdir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            time.sleep(6)  # give it 6 seconds to crash or not
            if proc.poll() is not None:
                _, stderr = proc.communicate(timeout=3)
                err_text = stderr.decode("utf-8", errors="replace")[-500:]
                print(f"    {VERIFY_FAIL} App crashed on startup")
                results.append(f"{VERIFY_FAIL} startup: crashed — {err_text[:200]}")
                overall_success = False

                # Feed crash to model for fixing
                if messages is not None:
                    crash_prompt = (
                        f"The app crashed on startup with `{start_cmd}`.\n"
                        f"Error: {err_text}\n"
                        f"Project: {ptype}" + (f" ({framework})" if framework else "") + f" at {pdir}\n"
                        f"Fix the crash. Check: missing dependencies, wrong entry point, "
                        f"missing .env vars, import errors, port conflicts."
                    )
                    messages.append({"role": "user", "content": crash_prompt})
                    try:
                        run_agent_turn(messages, model, use_tools=True)
                    except Exception:
                        pass
            else:
                print(f"    {VERIFY_OK} App started successfully (6s without crash)")
                results.append(f"{VERIFY_OK} startup: {start_cmd} runs OK")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception as e:
            results.append(f"{VERIFY_FAIL} startup test: {e}")
    elif ptype != "static":
        print(f"  {C.SUBTLE}  No dev command detected — skipping startup test{C.RESET}")

    # --- AUTO-WIRING SCAN (using WiringAgent) ---
    print(f"  {C.TOOL}{BLACK_CIRCLE} Auto-wiring scan...{C.RESET}")
    wiring = WiringAgent(str(pdir), auto_stub=auto_stub)
    wiring.run_full_scan()
    if wiring.issues:
        wiring.auto_fix()
        _display_wiring_report(wiring)
        print(f"    {VERIFY_FAIL} Found {len(wiring.issues)} wiring issue(s)")
        results.append(f"{VERIFY_FAIL} wiring: {len(wiring.issues)} issues found")
        if wiring.manual_needed and messages is not None:
            messages.append({"role": "user", "content": wiring.format_prompt()})
            try:
                run_agent_turn(messages, model, use_tools=True)
            except Exception as e:
                print(f"    {C.ERROR}Wiring fix failed: {e}{C.RESET}")
            # Re-scan after fix
            wiring2 = WiringAgent(str(pdir), auto_stub=auto_stub)
            wiring2.run_full_scan()
            if wiring2.issues:
                results.append(f"{VERIFY_FAIL} wiring: {len(wiring2.issues)} issues remain after fix attempt")
            else:
                print(f"    {VERIFY_OK} All wiring issues resolved")
                results.append(f"{VERIFY_OK} wiring: all issues resolved")
    else:
        print(f"    {VERIFY_OK} No wiring issues found")
        results.append(f"{VERIFY_OK} wiring: all components connected")

    # --- EDGE CASE DETECTION ---
    features = detect_built_features(str(pdir))
    if features:
        print(f"  {C.TOOL}{BLACK_CIRCLE} Edge case scan: {len(features)} feature(s) detected{C.RESET}")
        edge_prompt = generate_edge_case_prompt(features)
        if edge_prompt and messages is not None:
            messages.append({"role": "user", "content": edge_prompt})
            try:
                run_agent_turn(messages, model, use_tools=True)
                results.append(f"{VERIFY_OK} edge cases: {len(features)} features hardened")
            except Exception as e:
                print(f"    {C.ERROR}Edge case hardening failed: {e}{C.RESET}")

    summary = "\n".join(results) if results else "No verification steps run."

    # --- AUTO-TEST ---
    test_fw, test_cmd = _detect_test_framework(str(pdir))
    if test_cmd:
        test_ok, test_summary = _run_tests(str(pdir), model, messages)
        results.append(f"{VERIFY_OK if test_ok else VERIFY_FAIL} tests: {test_summary}")
        if not test_ok:
            overall_success = False

    # Show file tree of the project
    print(f"\n  {C.CLAW}{BLACK_CIRCLE} Project structure:{C.RESET}")
    _file_tree(str(pdir), max_depth=2, max_files=25)

    # --- FIX 12: Print "how to run" instructions based on project type ---
    print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} How to run:{C.RESET}")
    if ptype == "static":
        entry = project_info.get("entry_file", "index.html")
        print(f"    Open {entry} in your browser: start {entry}")
    elif ptype == "node":
        dev = project_info.get("dev_cmd", "npm run dev")
        print(f"    Run: {dev}")
        if framework in ("nextjs", "vite", "cra", "nuxt"):
            print(f"    Opens at: http://localhost:3000")
        elif framework in ("express", "fastify"):
            print(f"    Opens at: http://localhost:3000")
    elif ptype == "python":
        dev = project_info.get("dev_cmd")
        if dev:
            print(f"    Run: {dev}")
            if framework == "fastapi":
                print(f"    Opens at: http://localhost:8000")
            elif framework == "django":
                print(f"    Opens at: http://localhost:8000")
            elif framework == "flask":
                print(f"    Opens at: http://localhost:5000")
            elif framework == "streamlit":
                print(f"    Opens at: http://localhost:8501")
        else:
            print(f"    Run the main Python file in the project directory")

    # Bell notification — build done
    _bell()

    return overall_success, summary


# ---------------------------------------------------------------------------
# auto-wiring verification -- detect disconnected components
# ---------------------------------------------------------------------------

_WIRING_SKIP_DIRS = {"node_modules", ".git", ".next", "__pycache__", "venv", "dist", "build", ".venv", "env"}

# Known native Node modules that need serverExternalPackages in Next.js
_NATIVE_NODE_MODULES = {"better-sqlite3", "sharp", "bcrypt", "argon2", "canvas", "sqlite3",
                        "pg-native", "cpu-features", "ssh2", "@mapbox/node-pre-gyp"}

# Server-only modules that must not be imported in 'use client' files
_SERVER_ONLY_MODULES = {"openai", "better-sqlite3", "pg", "mysql2", "mongoose", "prisma",
                        "@prisma/client", "nodemailer", "sharp", "fs", "path", "crypto",
                        "child_process", "bcrypt", "argon2", "jsonwebtoken"}

# Client-only modules that crash during SSR. Review quarterly.
_CLIENT_ONLY_MODULES = {
    "recharts", "chart.js", "react-chartjs-2", "plotly.js", "react-plotly.js", "visx",
    "three", "@react-three/fiber", "@react-three/drei",
    "@hello-pangea/dnd", "react-beautiful-dnd", "@dnd-kit/core", "@dnd-kit/sortable",
    "framer-motion", "react-spring", "lottie-react",
    "react-player", "react-confetti",
    "react-hot-toast", "sonner",
    "@tanstack/react-query", "react-query",
    "mapbox-gl", "react-map-gl",
    "react-hook-form",
}


# ---------------------------------------------------------------------------
# Project dependency graph (multi-file reasoning)
# ---------------------------------------------------------------------------

PROJECT_INDEX_DIR = Path.home() / ".claw" / "project_index"
_GRAPH_MAX_FILES = 500
_project_graph = None

# Extensions to skip when resolving imports (non-source assets)
_ASSET_EXTS = {'.css', '.scss', '.sass', '.less', '.json', '.png', '.jpg', '.jpeg',
               '.gif', '.svg', '.ico', '.webp', '.mp4', '.webm', '.woff', '.woff2',
               '.ttf', '.eot', '.mp3', '.wav', '.pdf'}

# Source file extensions for graph building
_GRAPH_SOURCE_EXTS = {'.js', '.ts', '.jsx', '.tsx', '.py', '.mjs', '.cjs'}


def _walk_project_files(project_dir, max_files=500):
    """Walk project, return list of (rel_path, abs_path) for source files."""
    pdir = Path(project_dir)
    results = []
    for root, dirs, files in os.walk(pdir):
        dirs[:] = [d for d in dirs if d not in _WIRING_SKIP_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            fp = Path(root) / fname
            if fp.suffix.lower() not in _GRAPH_SOURCE_EXTS:
                continue
            rel = str(fp.relative_to(pdir)).replace("\\", "/")
            results.append((rel, fp))
            if len(results) >= max_files:
                return results
    return results


class ProjectGraph:
    """Bidirectional dependency graph built from import/require statements."""

    VERSION = 1

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()
        self.imports = {}       # file -> set of files it imports
        self.importers = {}     # file -> set of files that import it
        self.exports = {}       # file -> list of exported symbol names
        self.barrels = set()    # set of barrel files (index.ts/js with only re-exports)
        self.file_hashes = {}   # file -> (mtime, size)
        self.is_partial = False
        self.built_at = 0.0
        self.version = self.VERSION
        self._total_files_on_disk = 0

    def _parse_file(self, rel_path, content):
        """Parse a file for imports and exports.
        Returns (import_specifiers: set[str], exported_symbols: list[str], is_barrel: bool)
        """
        ext = Path(rel_path).suffix.lower()
        import_specs = set()
        exported = []
        is_barrel = False

        if ext in ('.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs'):
            # Static ES imports: import X from './path', import { X } from '@/path'
            for m in re.finditer(r'''import\s+.*?from\s+['"]([@\w/.-]+)['"]''', content):
                import_specs.add(m.group(1))
            # Re-exports: export { X } from './path' — captures import edge
            for m in re.finditer(r'''export\s+\{[^}]*\}\s+from\s+['"]([@\w/.-]+)['"]''', content):
                import_specs.add(m.group(1))
            # CommonJS: require('./path')
            for m in re.finditer(r'''require\s*\(\s*['"]([@\w/.-]+)['"]\s*\)''', content):
                import_specs.add(m.group(1))

            # Named exports
            for m in re.finditer(r'export\s+(?:const|function|class|let|var|type|interface)\s+(\w+)', content):
                exported.append(m.group(1))
            # Default exports: export default function/class Name
            for m in re.finditer(r'export\s+default\s+(?:function|class)\s+(\w+)', content):
                if m.group(1) not in exported:
                    exported.append(m.group(1))

            # Barrel detection: index.ts/js files with only import/export/closing-brace lines
            basename = Path(rel_path).stem.lower()
            if basename == 'index':
                meaningful = []
                in_block_comment = False
                for line in content.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if in_block_comment:
                        if '*/' in stripped:
                            in_block_comment = False
                        continue
                    if stripped.startswith('/*'):
                        if '*/' not in stripped:
                            in_block_comment = True
                        continue
                    if stripped.startswith('//'):
                        continue
                    meaningful.append(stripped)
                if meaningful:
                    barrel_re = re.compile(r'^(import |export |\})')
                    is_barrel = all(barrel_re.match(ln) for ln in meaningful)

        elif ext == '.py':
            # Python: import mod, from mod import X
            for m in re.finditer(r'^import\s+(\w+)', content, re.MULTILINE):
                import_specs.add(m.group(1))
            for m in re.finditer(r'^from\s+(\w+(?:\.\w+)*)\s+import', content, re.MULTILINE):
                spec = m.group(1)
                if not spec.startswith('.'):
                    import_specs.add(spec)
            # Python exports: def/class at module level
            for m in re.finditer(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE):
                exported.append(m.group(1))

        return import_specs, exported, is_barrel

    def _resolve_import(self, importing_file, import_specifier):
        """Resolve import specifier to relative path. Returns None if unresolvable."""
        spec = import_specifier

        # Skip asset imports
        ext_of_spec = Path(spec).suffix.lower()
        if ext_of_spec in _ASSET_EXTS:
            return None

        # Only resolve project-local specifiers
        is_local = spec.startswith('.') or spec.startswith('@/') or spec.startswith('src/')
        if not is_local and not importing_file.endswith('.py'):
            return None

        importing_dir = str(Path(importing_file).parent)

        if spec.startswith('@/'):
            remainder = spec[2:]
            candidates_base = []
            if (self.project_dir / 'src').is_dir():
                candidates_base.append('src/' + remainder)
            candidates_base.append(remainder)
        elif spec.startswith('./') or spec.startswith('../'):
            resolved = (Path(importing_dir) / spec).as_posix()
            parts = []
            for p in resolved.split('/'):
                if p == '..':
                    if parts:
                        parts.pop()
                elif p != '.':
                    parts.append(p)
            candidates_base = ['/'.join(parts)]
        elif spec.startswith('src/'):
            candidates_base = [spec]
        elif importing_file.endswith('.py'):
            candidates_base = [spec.replace('.', '/')]
        else:
            return None

        for base_str in candidates_base:
            # Direct match if has extension
            if Path(base_str).suffix:
                if (self.project_dir / base_str).is_file():
                    return base_str
                continue

            # Extension probing
            if importing_file.endswith('.py'):
                exts_to_try = ['.py', '/__init__.py']
            else:
                exts_to_try = ['.ts', '.tsx', '.js', '.jsx',
                               '/index.ts', '/index.tsx', '/index.js', '/index.jsx']
            for try_ext in exts_to_try:
                candidate = base_str + try_ext
                if (self.project_dir / candidate).is_file():
                    return candidate

        return None

    def build_full(self):
        """Walk up to _GRAPH_MAX_FILES files, parse imports/exports, build graph."""
        file_list = _walk_project_files(self.project_dir, max_files=_GRAPH_MAX_FILES)
        self._total_files_on_disk = len(file_list)

        if len(file_list) >= _GRAPH_MAX_FILES:
            all_files = _walk_project_files(self.project_dir, max_files=_GRAPH_MAX_FILES + 100000)
            self._total_files_on_disk = len(all_files)
            self.is_partial = len(all_files) > _GRAPH_MAX_FILES
        else:
            self.is_partial = False

        self.imports.clear()
        self.importers.clear()
        self.exports.clear()
        self.barrels.clear()
        self.file_hashes.clear()

        parsed = {}
        for rel_path, abs_path in file_list:
            try:
                stat = abs_path.stat()
                self.file_hashes[rel_path] = (stat.st_mtime, stat.st_size)
                content = abs_path.read_text(encoding='utf-8', errors='replace')
                parsed[rel_path] = self._parse_file(rel_path, content)
            except Exception:
                continue

        for rel_path, (import_specs, exported, is_barrel) in parsed.items():
            self.exports[rel_path] = exported
            if is_barrel:
                self.barrels.add(rel_path)
            resolved_imports = set()
            for spec in import_specs:
                target = self._resolve_import(rel_path, spec)
                if target and target in self.file_hashes:
                    resolved_imports.add(target)
            self.imports[rel_path] = resolved_imports
            for target in resolved_imports:
                if target not in self.importers:
                    self.importers[target] = set()
                self.importers[target].add(rel_path)

        self.built_at = time.time()

    def update_incremental(self, changed_files):
        """Incremental update. changed_files: list of (rel_path, change_type)."""
        if not changed_files:
            return

        # Pass 0: Handle deletions first
        for rel_path, change_type in changed_files:
            rel_path = rel_path.replace('\\', '/')
            if change_type != 'deleted':
                continue
            self.file_hashes.pop(rel_path, None)
            self.exports.pop(rel_path, None)
            self.barrels.discard(rel_path)
            old_imports = self.imports.pop(rel_path, set())
            for target in old_imports:
                if target in self.importers:
                    self.importers[target].discard(rel_path)
            if rel_path in self.importers:
                for importer in list(self.importers[rel_path]):
                    if importer in self.imports:
                        self.imports[importer].discard(rel_path)
                del self.importers[rel_path]

        # Pass 1: Re-read and re-parse changed/created files
        new_parsed = {}
        for rel_path, change_type in changed_files:
            rel_path = rel_path.replace('\\', '/')
            if change_type == 'deleted':
                continue
            abs_path = self.project_dir / rel_path
            if abs_path.suffix.lower() not in _GRAPH_SOURCE_EXTS:
                continue
            try:
                stat = abs_path.stat()
                self.file_hashes[rel_path] = (stat.st_mtime, stat.st_size)
                content = abs_path.read_text(encoding='utf-8', errors='replace')
                new_parsed[rel_path] = self._parse_file(rel_path, content)
            except Exception:
                continue

        # Pass 2: Rebuild edges for changed/created files in batch
        for rel_path, (import_specs, exported, is_barrel) in new_parsed.items():
            old_imports = self.imports.get(rel_path, set())
            for target in old_imports:
                if target in self.importers:
                    self.importers[target].discard(rel_path)

            self.exports[rel_path] = exported
            if is_barrel:
                self.barrels.add(rel_path)
            else:
                self.barrels.discard(rel_path)

            resolved_imports = set()
            for spec in import_specs:
                target = self._resolve_import(rel_path, spec)
                if target and target in self.file_hashes:
                    resolved_imports.add(target)
            self.imports[rel_path] = resolved_imports
            for target in resolved_imports:
                if target not in self.importers:
                    self.importers[target] = set()
                self.importers[target].add(rel_path)

    def _stale_files(self):
        """Fresh directory walk to find modified/added/removed files.
        Returns (modified, added, removed) lists of relative paths.
        """
        file_list = _walk_project_files(self.project_dir, max_files=_GRAPH_MAX_FILES)
        current = {}
        for rel_path, abs_path in file_list:
            try:
                stat = abs_path.stat()
                current[rel_path] = (stat.st_mtime, stat.st_size)
            except Exception:
                continue

        modified, added, removed = [], [], []
        for rel_path, (mtime, size) in current.items():
            if rel_path not in self.file_hashes:
                added.append(rel_path)
            elif self.file_hashes[rel_path] != (mtime, size):
                modified.append(rel_path)
        for rel_path in self.file_hashes:
            if rel_path not in current:
                removed.append(rel_path)
        return modified, added, removed

    def get_subgraph(self, seed_files, depth=1):
        """BFS from seeds, follow both directions, skip barrel pass-throughs.
        Returns dict {file: {imports, importers, exports, is_barrel}}.
        Empty seeds -> top 20 most-connected files.
        """
        if not seed_files:
            connectivity = {}
            for f in self.file_hashes:
                score = len(self.importers.get(f, set())) + len(self.imports.get(f, set()))
                if score > 0:
                    connectivity[f] = score
            top = sorted(connectivity, key=connectivity.get, reverse=True)[:20]
            return {f: {'imports': self.imports.get(f, set()),
                        'importers': self.importers.get(f, set()),
                        'exports': self.exports.get(f, []),
                        'is_barrel': f in self.barrels} for f in top}

        visited = set()
        queue = []
        for seed in seed_files:
            norm = seed.replace('\\', '/')
            if norm in self.file_hashes:
                queue.append((norm, 0))
                visited.add(norm)

        result = {}
        while queue:
            current, d = queue.pop(0)
            file_imports = self.imports.get(current, set())
            file_importers = self.importers.get(current, set())
            result[current] = {
                'imports': file_imports,
                'importers': file_importers,
                'exports': self.exports.get(current, []),
                'is_barrel': current in self.barrels,
            }
            if d < depth:
                for neighbor in file_imports | file_importers:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
        return result

    def get_dependents(self, file_path):
        """Direct importers only (not transitive)."""
        return self.importers.get(file_path.replace('\\', '/'), set())

    def _cache_path(self):
        """Cache file path for this project."""
        import hashlib
        PROJECT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        project_hash = hashlib.md5(str(self.project_dir).encode()).hexdigest()[:16]
        return PROJECT_INDEX_DIR / f"{project_hash}.json"

    def save(self):
        """Persist graph to disk with atomic write."""
        cache_path = self._cache_path()
        data = {
            'version': self.version,
            'project_dir': str(self.project_dir),
            'built_at': self.built_at,
            'is_partial': self.is_partial,
            'total_files_on_disk': self._total_files_on_disk,
            'imports': {k: sorted(v) for k, v in self.imports.items()},
            'importers': {k: sorted(v) for k, v in self.importers.items()},
            'exports': self.exports,
            'barrels': sorted(self.barrels),
            'file_hashes': {k: list(v) for k, v in self.file_hashes.items()},
        }
        tmp = cache_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
        os.replace(str(tmp), str(cache_path))

    def load(self):
        """Load graph from disk cache. Returns True if loaded."""
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            if data.get('version') != self.VERSION:
                return False
            if data.get('project_dir') != str(self.project_dir):
                return False
            self.built_at = data.get('built_at', 0)
            self.is_partial = data.get('is_partial', False)
            self._total_files_on_disk = data.get('total_files_on_disk', 0)
            self.imports = {k: set(v) for k, v in data.get('imports', {}).items()}
            self.importers = {k: set(v) for k, v in data.get('importers', {}).items()}
            self.exports = data.get('exports', {})
            self.barrels = set(data.get('barrels', []))
            self.file_hashes = {k: tuple(v) for k, v in data.get('file_hashes', {}).items()}
            return True
        except Exception:
            return False

    def inspect(self, file_path=None):
        """Debug dump. No file = full stats. With file = that file's edges."""
        if file_path is None:
            total_edges = sum(len(v) for v in self.imports.values())
            lines = [
                f"Project Graph Stats:",
                f"  Files indexed: {len(self.file_hashes)}",
                f"  Total edges: {total_edges}",
                f"  Barrel files: {len(self.barrels)}",
                f"  Partial: {self.is_partial}",
            ]
            if self.is_partial:
                lines.append(f"  Total on disk: {self._total_files_on_disk}")
            if self.barrels:
                lines.append(f"  Barrels: {', '.join(sorted(self.barrels)[:10])}")
            lines.append(f"  Built: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.built_at))}")
            return '\n'.join(lines)
        norm = file_path.replace('\\', '/')
        if norm not in self.file_hashes:
            return f"File '{norm}' not in graph."
        imp = sorted(self.imports.get(norm, set()))
        dep = sorted(self.importers.get(norm, set()))
        exp = self.exports.get(norm, [])
        lines = [
            f"Graph: {norm}",
            f"  Exports: {', '.join(exp) if exp else '(none)'}",
            f"  Imports ({len(imp)}): {', '.join(imp) if imp else '(none)'}",
            f"  Imported by ({len(dep)}): {', '.join(dep) if dep else '(none)'}",
            f"  Barrel: {norm in self.barrels}",
        ]
        return '\n'.join(lines)


def _get_project_graph():
    """Lazy singleton for the project dependency graph."""
    global _project_graph
    if _project_graph is not None:
        modified, added, removed = _project_graph._stale_files()
        if modified or added or removed:
            changes = ([(f, 'modified') for f in modified] +
                       [(f, 'created') for f in added] +
                       [(f, 'deleted') for f in removed])
            _project_graph.update_incremental(changes)
            _project_graph.save()
        return _project_graph

    _project_graph = ProjectGraph(CWD)
    if _project_graph.load():
        modified, added, removed = _project_graph._stale_files()
        if modified or added or removed:
            changes = ([(f, 'modified') for f in modified] +
                       [(f, 'created') for f in added] +
                       [(f, 'deleted') for f in removed])
            _project_graph.update_incremental(changes)
            _project_graph.save()
    else:
        _project_graph.build_full()
        _project_graph.save()
    return _project_graph


def _detect_active_frameworks(project_dir):
    """Detect which frameworks are active in the project."""
    pdir = Path(project_dir)
    frameworks = set()

    # Check config files
    if (pdir / "next.config.mjs").exists() or (pdir / "next.config.js").exists() or (pdir / "next.config.ts").exists():
        frameworks.add("nextjs")
    if (pdir / "svelte.config.js").exists() or (pdir / "svelte.config.ts").exists():
        frameworks.add("sveltekit")
    if (pdir / "nuxt.config.ts").exists() or (pdir / "nuxt.config.js").exists():
        frameworks.add("nuxt")
    if (pdir / "remix.config.js").exists() or (pdir / "remix.config.ts").exists():
        frameworks.add("remix")
    if (pdir / "astro.config.mjs").exists() or (pdir / "astro.config.ts").exists():
        frameworks.add("astro")

    # Also check package.json dependencies
    pkg = pdir / "package.json"
    if pkg.exists():
        try:
            import json as _json
            data = _json.loads(pkg.read_text(encoding="utf-8"))
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "next" in deps: frameworks.add("nextjs")
            if "@sveltejs/kit" in deps: frameworks.add("sveltekit")
            if "nuxt" in deps: frameworks.add("nuxt")
            if "@remix-run/react" in deps or "@remix-run/node" in deps: frameworks.add("remix")
            if "astro" in deps: frameworks.add("astro")
            if "@react-router/dev" in deps: frameworks.add("remix")  # React Router v7
        except Exception:
            pass

    return frameworks


_MAGIC_FILES_BY_FRAMEWORK = {
    "nextjs": {
        "page.tsx", "page.jsx", "page.js", "page.ts",
        "layout.tsx", "layout.jsx", "layout.js", "layout.ts",
        "loading.tsx", "loading.jsx", "loading.js",
        "error.tsx", "error.jsx", "error.js",
        "not-found.tsx", "not-found.jsx",
        "template.tsx", "template.jsx",
        "route.ts", "route.js",
        "default.tsx", "default.jsx",
        "middleware.ts", "middleware.js",
    },
    "sveltekit": {
        "+page.svelte", "+layout.svelte", "+error.svelte",
        "+page.server.ts", "+page.server.js",
        "+layout.server.ts", "+layout.server.js",
        "+server.ts", "+server.js",
        "+page.ts", "+page.js",
        "+layout.ts", "+layout.js",
    },
    "nuxt": {
        "app.vue",
    },
    "remix": {
        "root.tsx", "root.jsx",
        "entry.client.tsx", "entry.server.tsx",
        "entry.client.jsx", "entry.server.jsx",
    },
    "astro": set(),
}

_MAGIC_EXPORTS_BY_FRAMEWORK = {
    "nextjs": {
        "metadata", "generateMetadata", "generateStaticParams",
        "revalidate", "dynamic", "dynamicParams", "fetchCache",
        "runtime", "preferredRegion", "maxDuration",
        "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
    },
    "sveltekit": {
        "load", "actions", "entries", "prerender", "ssr", "csr",
    },
    "nuxt": {
        "definePageMeta", "useAsyncData", "useFetch",
    },
    "remix": {
        "loader", "action", "meta", "links", "headers",
        "ErrorBoundary", "HydrateFallback", "shouldRevalidate",
    },
    "astro": {
        "getStaticPaths", "prerender", "partial",
    },
}


def _is_framework_magic_path(fp, pdir, frameworks):
    """Check if file is in a framework-magic directory (not just by filename)."""
    name = fp.name
    # Check filename-based magic
    for fw in frameworks:
        if name in _MAGIC_FILES_BY_FRAMEWORK.get(fw, set()):
            return True
    # Check directory-based magic
    try:
        rel = str(fp.relative_to(pdir)).replace("\\", "/")
    except ValueError:
        return False
    if "nuxt" in frameworks:
        if rel.startswith("pages/") and name.endswith(".vue"): return True
        if rel.startswith("layouts/") and name.endswith(".vue"): return True
        if rel.startswith("server/api/"): return True
    if "remix" in frameworks:
        if rel.startswith("app/routes/"): return True
    if "astro" in frameworks:
        if rel.startswith("src/pages/") and name.endswith(".astro"): return True
    return False


def _check_native_module_config(pdir, frameworks, issues, file_contents=None):
    """A1: Check that native Node modules have serverExternalPackages in Next.js config."""
    if "nextjs" not in frameworks:
        return
    pkg = pdir / "package.json"
    # Collect native modules from BOTH package.json AND actual imports in source files
    native_used = set()
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = set(data.get("dependencies", {}).keys()) | set(data.get("devDependencies", {}).keys())
            native_used |= deps & _NATIVE_NODE_MODULES
        except Exception:
            pass
    # Also scan actual imports — catches modules imported but missing from package.json
    if file_contents:
        for fp, content in file_contents.items():
            for m in re.finditer(r'''import\s+.*?from\s+['"]([@\w/.-]+)['"]''', content):
                mod = m.group(1)
                bare = mod.split('/')[0] if not mod.startswith('@') else '/'.join(mod.split('/')[:2])
                if bare in _NATIVE_NODE_MODULES:
                    native_used.add(bare)
    if not native_used:
        return
    # Check next.config for serverExternalPackages
    for cfg_name in ("next.config.mjs", "next.config.js", "next.config.ts"):
        cfg = pdir / cfg_name
        if cfg.exists():
            try:
                content = cfg.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for mod in native_used:
                if mod not in content:
                    issues.append({
                        "type": "missing_native_config",
                        "file": cfg_name,
                        "module": mod,
                        "message": f"Native module '{mod}' used but not in serverExternalPackages in {cfg_name} — Next.js will fail to bundle it"
                    })
            return
    # No next.config found at all — flag all native modules
    issues.append({
        "type": "missing_native_config",
        "file": "next.config.mjs",
        "module": ", ".join(sorted(native_used)),
        "message": f"Native module(s) {', '.join(sorted(native_used))} used but no next.config with serverExternalPackages found"
    })


def _check_client_server_boundary(pdir, file_contents, issues):
    """A2: Check that 'use client' files don't import server-only modules."""
    for fp, content in file_contents.items():
        if "'use client'" not in content and '"use client"' not in content:
            continue
        # Check direct imports from this client file
        for m in re.finditer(r'''import\s+.*?from\s+['"]([@\w/.-]+)['"]''', content):
            mod = m.group(1)
            # Get bare module name (handle scoped packages like @prisma/client)
            bare_mod = mod.split('/')[0] if not mod.startswith('@') else '/'.join(mod.split('/')[:2])
            if bare_mod in _SERVER_ONLY_MODULES:
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "client_server_violation",
                    "file": str(rel),
                    "module": mod,
                    "message": f"'use client' file {fp.name} imports server-only module '{mod}' — extract to a separate server module or API route"
                })


# Node.js built-in modules (skip in missing-deps check)
_NODE_BUILTINS = {
    "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
    "constants", "crypto", "dgram", "diagnostics_channel", "dns", "domain",
    "events", "fs", "http", "http2", "https", "inspector", "module", "net",
    "os", "path", "perf_hooks", "process", "punycode", "querystring",
    "readline", "repl", "stream", "string_decoder", "sys", "timers",
    "tls", "trace_events", "tty", "url", "util", "v8", "vm", "wasi",
    "worker_threads", "zlib", "node", "next", "react", "react-dom",
}

# Hardcoded light-mode Tailwind classes that break in dark mode
_LIGHT_ONLY_CLASSES = re.compile(
    r'\b(?:text-(?:gray|slate|zinc|neutral|stone)-(?:9|8|7)00'
    r'|bg-(?:white|gray-50|gray-100|slate-50|slate-100)'
    r'|border-(?:gray|slate)-(?:2|3)00'
    r'|hover:bg-(?:gray|slate)-(?:50|100))\b'
)


def _check_missing_package_deps(pdir, file_contents, issues):
    """Check that every imported npm package is listed in package.json."""
    pkg = pdir / "package.json"
    if not pkg.exists():
        return
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
        listed = set(data.get("dependencies", {}).keys()) | set(data.get("devDependencies", {}).keys())
    except Exception:
        return
    # Scan all JS/TS files for external imports
    missing = {}  # module -> first file that imports it
    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        if ext not in (".js", ".ts", ".jsx", ".tsx"):
            continue
        for m in re.finditer(r'''import\s+.*?from\s+['"]([^'"./][^'"]*)['"]''', content):
            mod = m.group(1)
            # Get bare package name
            bare = mod.split('/')[0] if not mod.startswith('@') else '/'.join(mod.split('/')[:2])
            if bare in _NODE_BUILTINS or bare.startswith("node:") or bare.startswith("@/"):
                continue
            if bare in listed:
                continue
            if bare not in missing:
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                missing[bare] = str(rel)
    for mod, first_file in missing.items():
        issues.append({
            "type": "missing_package_dep",
            "file": "package.json",
            "module": mod,
            "first_import": first_file,
            "message": f"Package '{mod}' is imported (first in {first_file}) but not listed in package.json — add it to dependencies"
        })


def _check_import_export_mismatch(pdir, file_contents, issues):
    """Check that named vs default imports match the target file's export style."""
    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        if ext not in (".js", ".ts", ".jsx", ".tsx"):
            continue
        for m in re.finditer(r'''import\s+\{\s*(\w+)\s*\}\s*from\s+['"](\.[^'"]+)['"]''', content):
            symbol = m.group(1)
            import_path = m.group(2)
            # Resolve target file
            base_dir = fp.parent
            target = base_dir / import_path
            resolved = None
            for try_ext in ("", ".ts", ".tsx", ".js", ".jsx"):
                candidate = Path(str(target) + try_ext)
                if candidate.exists() and candidate in file_contents:
                    resolved = candidate
                    break
            if not resolved:
                continue
            target_content = file_contents[resolved]
            # Check: target ONLY has default export, no matching named export
            has_named = bool(re.search(r'export\s+(?:const|function|class|let|var|type|interface)\s+' + re.escape(symbol) + r'\b', target_content))
            has_default = bool(re.search(r'export\s+default\b', target_content))
            if has_default and not has_named:
                try:
                    rel = fp.relative_to(pdir)
                    trel = resolved.relative_to(pdir)
                except ValueError:
                    rel, trel = fp, resolved
                issues.append({
                    "type": "import_export_mismatch",
                    "file": str(rel),
                    "target": str(trel),
                    "symbol": symbol,
                    "message": f"Named import '{{ {symbol} }}' from {trel} but that file uses 'export default' — use 'import {symbol} from ...' instead"
                })


def _check_self_redirect(pdir, file_contents, active_frameworks, issues):
    """Check for pages that redirect() to themselves — infinite redirect loop."""
    if "nextjs" not in active_frameworks:
        return
    app_dir = None
    for candidate in (pdir / "app", pdir / "src" / "app"):
        if candidate.is_dir():
            app_dir = candidate
            break
    if not app_dir:
        return
    for fp, content in file_contents.items():
        if fp.name not in ("page.tsx", "page.jsx", "page.ts", "page.js"):
            continue
        # Find redirect() calls
        for m in re.finditer(r'''redirect\s*\(\s*[`'"](/[^`'"]*)[`'"]\s*\)''', content):
            redirect_target = m.group(1)
            # Compute this page's route
            try:
                rel_route = "/" + str(fp.parent.relative_to(app_dir)).replace("\\", "/")
                rel_route = re.sub(r'/\([^)]+\)', '', rel_route)  # remove route groups
                if rel_route == "/.":
                    rel_route = "/"
            except ValueError:
                continue
            if redirect_target.rstrip("/") == rel_route.rstrip("/"):
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "self_redirect",
                    "file": str(rel),
                    "route": rel_route,
                    "message": f"Page {rel} redirects to itself ('{redirect_target}') — infinite redirect loop. Create the resource or redirect to a different route."
                })


def _check_dark_mode_conflicts(pdir, file_contents, active_frameworks, issues):
    """Check for hardcoded light-mode Tailwind classes when app defaults to dark mode."""
    if "nextjs" not in active_frameworks:
        return
    # Detect if dark mode is the default
    dark_default = False
    for fp, content in file_contents.items():
        if fp.name in ("layout.tsx", "layout.jsx", "layout.ts", "layout.js"):
            if "'dark'" in content or '"dark"' in content:
                if 'className' in content and 'dark' in content:
                    dark_default = True
                    break
    if not dark_default:
        return
    # Scan all page/component files for hardcoded light-mode classes
    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        if ext not in (".tsx", ".jsx"):
            continue
        # Skip layout files (they set up dark mode)
        if fp.name.startswith("layout"):
            continue
        matches = _LIGHT_ONLY_CLASSES.findall(content)
        if len(matches) >= 3:  # Only flag if 3+ light-mode classes (not incidental)
            try:
                rel = fp.relative_to(pdir)
            except ValueError:
                rel = fp
            sample = ", ".join(set(matches[:5]))
            issues.append({
                "type": "dark_mode_conflict",
                "file": str(rel),
                "count": len(matches),
                "message": f"{len(matches)} hardcoded light-mode classes in {rel} ({sample}) — app defaults to dark mode. Use theme-aware classes (bg-background, text-foreground) or dark: variants."
            })


def _check_nonfunctional_server_ui(pdir, file_contents, active_frameworks, issues):
    """Check for interactive UI elements (input, button, textarea) in server components without handlers."""
    if "nextjs" not in active_frameworks:
        return
    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        if ext not in (".tsx", ".jsx"):
            continue
        # Skip if 'use client'
        if "'use client'" in content or '"use client"' in content:
            continue
        # Skip API routes and non-page files
        if "route." in fp.name:
            continue
        # Check for interactive elements
        has_input = bool(re.search(r'<(?:input|textarea)\b', content))
        has_button = bool(re.search(r'<button\b', content))
        if not (has_input or has_button):
            continue
        # Check for any event handlers
        has_handlers = bool(re.search(r'\bon(?:Click|Change|Submit|KeyDown|KeyUp|Input|Focus|Blur)\s*[={]', content))
        if has_handlers:
            continue
        # Check if it's a form with action (server action pattern — OK, forms work without JS)
        has_form_action = bool(re.search(r'<form\s[^>]*action\s*=', content))
        if has_form_action:
            continue
        # Count interactive elements without handlers
        input_count = len(re.findall(r'<(?:input|textarea)\b', content))
        button_count = len(re.findall(r'<button\b', content))
        if input_count > 0:  # Has actual input fields without handlers in server component
            try:
                rel = fp.relative_to(pdir)
            except ValueError:
                rel = fp
            issues.append({
                "type": "nonfunctional_server_ui",
                "file": str(rel),
                "inputs": input_count,
                "buttons": button_count,
                "message": f"Server component {rel} has {input_count} input(s) and {button_count} button(s) but no event handlers or 'use client' — the UI is non-interactive. Extract to a client component."
            })


def scan_wiring_issues(project_dir):
    """Scan for disconnected components: orphaned exports, broken imports, missing env vars, dead routes."""
    pdir = Path(project_dir)
    if not pdir.exists():
        return []

    active_frameworks = _detect_active_frameworks(project_dir)

    issues = []
    files_by_ext = {}
    file_count = 0

    # Collect all source files (cap at 200)
    for fp in pdir.rglob("*"):
        if file_count >= 200:
            break
        if not fp.is_file():
            continue
        # Skip excluded dirs
        if any(skip in fp.parts for skip in _WIRING_SKIP_DIRS):
            continue
        ext = fp.suffix.lower()
        if ext in (".js", ".ts", ".jsx", ".tsx", ".py", ".html"):
            files_by_ext.setdefault(ext, []).append(fp)
            file_count += 1

    # Read all file contents into cache
    file_contents = {}
    for ext_files in files_by_ext.values():
        for fp in ext_files:
            try:
                file_contents[fp] = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    all_content = "\n".join(file_contents.values())
    js_ts_files = []
    for ext in (".js", ".ts", ".jsx", ".tsx"):
        js_ts_files.extend(files_by_ext.get(ext, []))

    # 1. Orphaned exports (JS/TS): exported but never imported
    # Collect magic export names for active frameworks
    _active_magic_exports = set()
    for fw in active_frameworks:
        _active_magic_exports.update(_MAGIC_EXPORTS_BY_FRAMEWORK.get(fw, set()))

    for fp in js_ts_files:
        content = file_contents.get(fp, "")
        is_magic = _is_framework_magic_path(fp, pdir, active_frameworks)
        # Find named exports
        for m in re.finditer(r'export\s+(?:const|function|class|let|var|type|interface)\s+(\w+)', content):
            symbol = m.group(1)
            # Skip framework-magic exports
            if is_magic and symbol in _active_magic_exports:
                continue
            # Check if imported anywhere else
            import_pattern = re.compile(r'import\b.*\b' + re.escape(symbol) + r'\b')
            used = False
            for other_fp, other_content in file_contents.items():
                if other_fp == fp:
                    continue
                if import_pattern.search(other_content):
                    used = True
                    break
            if not used:
                # Skip default exports in framework-magic files
                if is_magic:
                    continue
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "orphaned_export",
                    "file": str(rel),
                    "symbol": symbol,
                    "message": f"'{symbol}' is exported from {rel} but never imported anywhere"
                })

    # 2. Unrendered React components: PascalCase exports in .tsx/.jsx not used as <Component>
    react_files = files_by_ext.get(".tsx", []) + files_by_ext.get(".jsx", [])
    for fp in react_files:
        # Skip framework-magic files — they are rendered by the router
        if _is_framework_magic_path(fp, pdir, active_frameworks):
            continue
        content = file_contents.get(fp, "")
        for m in re.finditer(r'export\s+(?:default\s+)?(?:function|const)\s+([A-Z]\w+)', content):
            comp_name = m.group(1)
            # Check if used as <ComponentName in any other file
            usage_pattern = re.compile(r'<' + re.escape(comp_name) + r'[\s/>]')
            rendered = False
            for other_fp, other_content in file_contents.items():
                if other_fp == fp:
                    continue
                if usage_pattern.search(other_content):
                    rendered = True
                    break
            if not rendered:
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "unrendered_component",
                    "file": str(rel),
                    "symbol": comp_name,
                    "message": f"React component '{comp_name}' in {rel} is never rendered (<{comp_name}>) anywhere"
                })

    # 3. Broken imports: import from relative path that doesn't exist
    for fp in js_ts_files:
        content = file_contents.get(fp, "")
        for m in re.finditer(r'''import\s+.*?from\s+['"](\.{1,2}/[^'"]+)['"]''', content):
            import_path = m.group(1)
            base_dir = fp.parent
            target = base_dir / import_path
            # Try with extensions
            found = False
            candidates = [target]
            for try_ext in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js", "/index.jsx"):
                candidates.append(Path(str(target) + try_ext))
            for c in candidates:
                if c.exists():
                    found = True
                    break
            if not found:
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "broken_import",
                    "file": str(rel),
                    "import_path": import_path,
                    "message": f"Broken import in {rel}: '{import_path}' — target file not found"
                })

    # 4. Missing env vars: process.env.X / os.environ used but not in .env.example
    env_vars_used = set()
    for fp, content in file_contents.items():
        # JS/TS: process.env.VAR_NAME
        for m in re.finditer(r'process\.env\.(\w+)', content):
            env_vars_used.add(m.group(1))
        # Python: os.environ["VAR"] or os.environ.get("VAR")
        for m in re.finditer(r'os\.environ(?:\.get)?\s*\[\s*["\'](\w+)["\']', content):
            env_vars_used.add(m.group(1))
        for m in re.finditer(r'os\.environ\.get\s*\(\s*["\'](\w+)["\']', content):
            env_vars_used.add(m.group(1))

    if env_vars_used:
        env_example = pdir / ".env.example"
        env_file = pdir / ".env"
        env_local = pdir / ".env.local"
        defined_vars = set()
        for ef in (env_example, env_file, env_local):
            if ef.exists():
                try:
                    for line in ef.read_text(encoding="utf-8", errors="replace").splitlines():
                        m = re.match(r'^(\w+)\s*=', line.strip())
                        if m:
                            defined_vars.add(m.group(1))
                except Exception:
                    pass
        # Standard vars to skip
        skip_vars = {"NODE_ENV", "PATH", "HOME", "USER", "SHELL", "TERM", "PWD",
                     "NEXT_PUBLIC_", "REACT_APP_"}
        for var in env_vars_used:
            if var in defined_vars:
                continue
            if any(var.startswith(s) for s in skip_vars if s.endswith("_")):
                continue
            if var in skip_vars:
                continue
            issues.append({
                "type": "missing_env_var",
                "variable": var,
                "message": f"Environment variable '{var}' is used but not defined in .env.example or .env"
            })

    # 5. Route dead-ends (Next.js): Link href / router.push to non-existent pages
    pages_dir = None
    app_dir = None
    for candidate in (pdir / "app", pdir / "src" / "app"):
        if candidate.is_dir():
            app_dir = candidate
            break
    for candidate in (pdir / "pages", pdir / "src" / "pages"):
        if candidate.is_dir():
            pages_dir = candidate
            break

    if app_dir or pages_dir:
        # Collect existing routes
        existing_routes = {"/"}
        route_dir = app_dir or pages_dir
        for route_file in route_dir.rglob("*"):
            if route_file.name in ("page.tsx", "page.jsx", "page.js", "page.ts",
                                     "route.tsx", "route.ts", "route.js"):
                try:
                    rel_route = "/" + str(route_file.parent.relative_to(route_dir)).replace("\\", "/")
                    rel_route = re.sub(r'/\([^)]+\)', '', rel_route)  # remove route groups
                    if rel_route == "/.":
                        rel_route = "/"
                    existing_routes.add(rel_route)
                except ValueError:
                    pass
            elif route_file.suffix in (".tsx", ".jsx", ".js", ".ts") and pages_dir:
                try:
                    rel_route = "/" + str(route_file.relative_to(pages_dir)).replace("\\", "/")
                    rel_route = re.sub(r'\.\w+$', '', rel_route)  # remove extension
                    if rel_route.endswith("/index"):
                        rel_route = rel_route[:-6] or "/"
                    existing_routes.add(rel_route)
                except ValueError:
                    pass

        # Find link targets
        for fp, content in file_contents.items():
            for m in re.finditer(r'''(?:href|push|replace)\s*(?:=\s*|[(])\s*['"](/[a-zA-Z0-9/_-]+)['"]''', content):
                target = m.group(1)
                # Skip API routes and dynamic segments
                if target.startswith("/api/") or "[" in target:
                    continue
                if target not in existing_routes:
                    try:
                        rel = fp.relative_to(pdir)
                    except ValueError:
                        rel = fp
                    issues.append({
                        "type": "dead_route",
                        "file": str(rel),
                        "target": target,
                        "message": f"Link to '{target}' in {rel} — no matching page found"
                    })

    # 6. Placeholder links: href="#" in nav/sidebar components (lazy non-functional links)
    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        if ext in (".tsx", ".jsx", ".js", ".ts", ".html"):
            placeholder_links = re.findall(r'href=["\']#["\']', content)
            if len(placeholder_links) >= 2:
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                # Extract link labels to suggest real routes
                labels = re.findall(r'href=["\']#["\'][^>]*>([^<]+)<', content)
                label_str = ", ".join(labels[:5]) if labels else "multiple links"
                issues.append({
                    "type": "placeholder_links",
                    "file": str(rel),
                    "count": len(placeholder_links),
                    "labels": labels[:5],
                    "message": f"{len(placeholder_links)} placeholder href='#' links in {rel} ({label_str}) — create real routes and use next/link with proper hrefs"
                })

    # 7. Nav links without matching pages: extract nav labels and check if routes exist
    if app_dir:
        for fp, content in file_contents.items():
            ext = fp.suffix.lower()
            if ext not in (".tsx", ".jsx"):
                continue
            # Detect sidebar/nav components by name or content patterns
            fname = fp.stem.lower()
            is_nav = fname in ('sidebar', 'nav', 'navbar', 'navigation', 'header', 'menu')
            is_nav = is_nav or bool(re.search(r'(sidebar|nav|menu)', content.lower()))
            if not is_nav:
                continue
            # Find link labels that suggest pages should exist
            _homepage_aliases = {"/overview", "/home", "/dashboard", "/main", "/index", "/start", "/landing"}
            link_labels = re.findall(r'(?:href|to)=["\'][^"\']*["\'][^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*<', content)
            for label in link_labels:
                route = "/" + label.lower().replace(" ", "-")
                if route == "/":
                    continue
                # Skip homepage aliases if "/" route exists
                if route in _homepage_aliases and "/" in existing_routes:
                    continue
                if route not in existing_routes:
                    try:
                        rel = fp.relative_to(pdir)
                    except ValueError:
                        rel = fp
                    issues.append({
                        "type": "missing_nav_page",
                        "file": str(rel),
                        "label": label,
                        "expected_route": route,
                        "message": f"Nav link '{label}' in {rel} suggests route '{route}' should exist — create app{route}/page.tsx"
                    })

    # 8. Python broken imports: from models.X import Y where models/X.py doesn't exist
    py_files = files_by_ext.get(".py", [])
    for fp in py_files:
        content = file_contents.get(fp, "")
        for m in re.finditer(r'from\s+([\w.]+)\s+import', content):
            module_path = m.group(1)
            # Only check local imports (not stdlib/pip packages)
            parts = module_path.split(".")
            if len(parts) >= 2:
                # Check if it looks like a local module
                first_part = parts[0]
                local_dirs = {"models", "schemas", "routes", "services", "middleware", "utils", "lib", "api", "core", "config", "database", "db"}
                if first_part in local_dirs:
                    # Build expected file path
                    expected = pdir / (module_path.replace(".", "/") + ".py")
                    expected_init = pdir / module_path.replace(".", "/") / "__init__.py"
                    if not expected.exists() and not expected_init.exists():
                        try:
                            rel = fp.relative_to(pdir)
                        except ValueError:
                            rel = fp
                        issues.append({
                            "type": "broken_python_import",
                            "file": str(rel),
                            "import_path": module_path,
                            "message": f"Broken import in {rel}: 'from {module_path} import ...' — file {module_path.replace('.', '/')}.py does not exist"
                        })

    # 9. FastAPI/Express route references to missing handler files
    for fp in py_files:
        content = file_contents.get(fp, "")
        # Detect router includes: app.include_router / from routes.X import router
        for m in re.finditer(r'from\s+(routes\.(\w+))\s+import', content):
            route_module = m.group(1)
            expected = pdir / (route_module.replace(".", "/") + ".py")
            if not expected.exists():
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "missing_route_handler",
                    "file": str(rel),
                    "module": route_module,
                    "message": f"Route handler '{route_module}' imported in {rel} but {route_module.replace('.', '/')}.py doesn't exist — create it"
                })

    # 10. Native module config check (Next.js serverExternalPackages)
    _check_native_module_config(pdir, active_frameworks, issues, file_contents)

    # 11. Client/server boundary violations ('use client' importing server-only modules)
    _check_client_server_boundary(pdir, file_contents, issues)

    # 12. Missing package.json dependencies (imported but not listed)
    _check_missing_package_deps(pdir, file_contents, issues)

    # 13. Import/export mismatch (named import from default-export file)
    _check_import_export_mismatch(pdir, file_contents, issues)

    # 14. Self-redirect detection (page redirects to itself = infinite loop)
    _check_self_redirect(pdir, file_contents, active_frameworks, issues)

    # 15. Dark-mode color conflicts (hardcoded light classes in dark-mode app)
    _check_dark_mode_conflicts(pdir, file_contents, active_frameworks, issues)

    # 16. Non-functional server UI (interactive elements without handlers in server components)
    _check_nonfunctional_server_ui(pdir, file_contents, active_frameworks, issues)

    return issues


def _format_wiring_report(issues):
    """Format wiring issues into a prompt the model can fix."""
    if not issues:
        return ""
    lines = ["## Auto-Wiring Issues Detected", "The following connection problems were found. Fix each one:", ""]
    by_type = {}
    for issue in issues:
        by_type.setdefault(issue["type"], []).append(issue)

    type_labels = {
        "orphaned_export": "Orphaned Exports (exported but never imported)",
        "unrendered_component": "Unrendered Components (defined but never used in JSX)",
        "broken_import": "Broken Imports (import targets don't exist)",
        "missing_env_var": "Missing Environment Variables",
        "dead_route": "Dead-End Routes (links to non-existent pages)",
        "missing_native_config": "Missing Native Module Config (serverExternalPackages)",
        "client_server_violation": "Client/Server Boundary Violations ('use client' imports server-only modules)",
    }
    for issue_type, items in by_type.items():
        lines.append(f"### {type_labels.get(issue_type, issue_type)}")
        for item in items[:10]:
            lines.append(f"- {item['message']}")
        lines.append("")

    lines.append("Fix these issues by creating missing files, adding missing imports, wiring up components, or defining missing env vars.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# extended wiring checks — CSS, API, asset coherence
# ---------------------------------------------------------------------------

def _detect_tailwind(project_dir):
    """Detect if project uses Tailwind CSS."""
    pdir = Path(project_dir)
    if (pdir / "tailwind.config.js").exists() or (pdir / "tailwind.config.ts").exists():
        return True
    pkg_path = pdir / "package.json"
    if pkg_path.exists():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "tailwindcss" in deps:
                return True
        except Exception:
            pass
    return False


def _detect_css_in_js(project_dir):
    """Detect if project uses CSS-in-JS (styled-components, emotion, vanilla-extract, panda)."""
    pdir = Path(project_dir)
    pkg_path = pdir / "package.json"
    if pkg_path.exists():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            css_in_js_pkgs = {"styled-components", "@emotion/react", "@emotion/styled",
                              "@vanilla-extract/css", "@pandacss/dev"}
            if css_in_js_pkgs & set(deps.keys()):
                return True
        except Exception:
            pass
    return False


def scan_css_coherence(project_dir, file_contents):
    """Check that CSS classes used in HTML/JSX exist in CSS files."""
    pdir = Path(project_dir)
    issues = []

    # Skip entirely for CSS-in-JS projects
    if _detect_css_in_js(project_dir):
        return issues

    is_tailwind = _detect_tailwind(project_dir)

    # Tailwind utility pattern: lowercase-start, dashes/slashes, no dots or spaces
    _tw_utility_re = re.compile(r'^[a-z][\w-]*(?:/[\w.]+)?$')

    # Collect CSS selectors from .css / .scss / .module.css files
    css_selectors = set()
    for fp, content in file_contents.items():
        if fp.suffix.lower() in (".css", ".scss"):
            for m in re.finditer(r'\.([a-zA-Z_][\w-]*)', content):
                css_selectors.add(m.group(1))

    # Collect class usage from HTML/JSX/TSX files
    for fp, content in file_contents.items():
        if fp.suffix.lower() not in (".html", ".jsx", ".tsx"):
            continue
        for m in re.finditer(r'(?:class|className)\s*=\s*["\']([^"\']+)["\']', content):
            classes = m.group(1).split()
            for cls in classes:
                # Skip Tailwind utilities
                if is_tailwind and _tw_utility_re.match(cls):
                    continue
                # Skip common non-Tailwind utilities
                if cls in ("sr-only", "visually-hidden", "clearfix"):
                    continue
                # Check if class is defined in CSS
                if cls not in css_selectors:
                    try:
                        rel = fp.relative_to(pdir)
                    except ValueError:
                        rel = fp
                    issues.append({
                        "type": "css_orphan",
                        "file": str(rel),
                        "detail": cls,
                        "message": f"CSS class '{cls}' used in {rel} but not defined in any CSS file"
                    })

    return issues


def scan_api_coherence(project_dir, file_contents):
    """Check that fetch/axios API calls point to existing API routes."""
    pdir = Path(project_dir)
    issues = []

    # Detect framework
    framework = None
    app_dir = None
    pages_api_dir = None

    for candidate in (pdir / "app" / "api", pdir / "src" / "app" / "api"):
        if candidate.is_dir():
            framework = "nextjs-app"
            app_dir = candidate
            break
    if not framework:
        for candidate in (pdir / "pages" / "api", pdir / "src" / "pages" / "api"):
            if candidate.is_dir():
                framework = "nextjs-pages"
                pages_api_dir = candidate
                break
    if not framework:
        # Check for Express patterns
        for fp, content in file_contents.items():
            if re.search(r'(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*[\'"]\/api\/', content):
                framework = "express"
                break

    # For unsupported frameworks, skip with info
    if not framework:
        return issues

    # Build known API routes
    known_routes = set()

    if framework == "nextjs-app" and app_dir:
        for route_file in app_dir.rglob("route.*"):
            if route_file.suffix.lower() in (".ts", ".tsx", ".js", ".jsx"):
                try:
                    rel = str(route_file.parent.relative_to(app_dir.parent)).replace("\\", "/")
                    route = "/" + rel
                    route = re.sub(r'/\([^)]+\)', '', route)  # remove route groups
                    known_routes.add(route)
                except ValueError:
                    pass

    elif framework == "nextjs-pages" and pages_api_dir:
        for route_file in pages_api_dir.rglob("*"):
            if route_file.is_file() and route_file.suffix.lower() in (".ts", ".tsx", ".js", ".jsx"):
                try:
                    rel = str(route_file.relative_to(pages_api_dir)).replace("\\", "/")
                    rel = re.sub(r'\.\w+$', '', rel)  # remove extension
                    if rel.endswith("/index"):
                        rel = rel[:-6] or ""
                    route = "/api/" + rel
                    known_routes.add(route)
                except ValueError:
                    pass

    elif framework == "express":
        for fp, content in file_contents.items():
            for m in re.finditer(r'(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*[\'"](\/api\/[^\'"]+)[\'"]', content):
                route = m.group(2)
                # Normalize: remove trailing params like :id
                route_base = re.sub(r'/:[^/]+', '', route)
                known_routes.add(route)
                known_routes.add(route_base)

    # Scan for API calls in JS/TS files
    for fp, content in file_contents.items():
        if fp.suffix.lower() not in (".js", ".ts", ".jsx", ".tsx"):
            continue
        for m in re.finditer(r'''(?:fetch|axios\.(?:get|post|put|delete|patch))\s*\(\s*[\'"](\/api\/[^\'"]+)[\'"]''', content):
            called_route = m.group(1)
            # Strip query params
            called_route = called_route.split("?")[0]
            # Skip dynamic segments
            if "${" in called_route or "{" in called_route:
                continue
            # Check if route exists (try exact and with param wildcards)
            if called_route not in known_routes:
                # Try removing trailing segments (could be :id params)
                base = "/".join(called_route.split("/")[:-1])
                if base not in known_routes and called_route not in known_routes:
                    try:
                        rel = fp.relative_to(pdir)
                    except ValueError:
                        rel = fp
                    issues.append({
                        "type": "missing_api_route",
                        "file": str(rel),
                        "detail": called_route,
                        "message": f"API call to '{called_route}' in {rel} — no matching route handler found"
                    })

    return issues


def scan_asset_references(project_dir, file_contents):
    """Check that static src/href references point to existing files."""
    pdir = Path(project_dir)
    issues = []

    for fp, content in file_contents.items():
        if fp.suffix.lower() not in (".html", ".jsx", ".tsx"):
            continue

        for m in re.finditer(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', content):
            ref = m.group(1)

            # Skip: external URLs, anchors, data URIs, javascript:, mailto:
            if any(ref.startswith(p) for p in ("http", "//", "#", "data:", "javascript:", "mailto:")):
                continue

            # Skip dynamic paths with template literals or JSX expressions
            if any(c in ref for c in ("{", "`", "${")):
                continue

            # Skip root-relative /api/ paths (handled by api coherence)
            if ref.startswith("/api/"):
                continue

            # Resolve path
            if ref.startswith("/"):
                # Root-relative: check from public/ or project root
                candidates = [
                    pdir / "public" / ref.lstrip("/"),
                    pdir / ref.lstrip("/"),
                ]
            else:
                # Relative to source file
                candidates = [
                    fp.parent / ref,
                    pdir / "public" / ref,
                ]

            found = any(c.exists() for c in candidates)
            if not found:
                try:
                    rel = fp.relative_to(pdir)
                except ValueError:
                    rel = fp
                issues.append({
                    "type": "missing_asset",
                    "file": str(rel),
                    "detail": ref,
                    "message": f"Asset reference '{ref}' in {rel} — file not found"
                })

    return issues


def _scan_render_coherence(project_dir, auto_stub=False):
    """
    Check that pages importing from @/components/ have matching component files,
    that component files export a function, and that functions return JSX.
    Optionally auto-stub missing components when auto_stub=True.
    """
    pdir = Path(project_dir)
    issues = []

    # Find all page/component files
    page_files = []
    component_dir = None
    for candidate in (pdir / "components", pdir / "src" / "components",
                      pdir / "app", pdir / "src" / "app"):
        if candidate.is_dir():
            if "components" in candidate.name:
                component_dir = candidate
            for f in candidate.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".tsx", ".jsx", ".ts", ".js"):
                    page_files.append(f)

    # Also check all pages in app/ directory
    for app_candidate in (pdir / "app", pdir / "src" / "app"):
        if app_candidate.is_dir():
            for f in app_candidate.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".tsx", ".jsx") and f not in page_files:
                    page_files.append(f)

    # Scan pages for component imports from @/components/
    for page_file in page_files:
        try:
            content = page_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Find imports like: import ComponentName from '@/components/ComponentName'
        for m in re.finditer(
            r'''import\s+(?:\{[^}]+\}|\w+)\s+from\s+['"]@/components/([^'"]+)['"]''',
            content
        ):
            import_path = m.group(1)
            # Resolve to file path
            if not component_dir:
                # Try to find components dir
                for cd in (pdir / "components", pdir / "src" / "components"):
                    if cd.is_dir():
                        component_dir = cd
                        break
            if not component_dir:
                continue

            # Check if component file exists
            comp_candidates = [
                component_dir / f"{import_path}.tsx",
                component_dir / f"{import_path}.ts",
                component_dir / f"{import_path}.jsx",
                component_dir / f"{import_path}.js",
                component_dir / import_path / "index.tsx",
                component_dir / import_path / "index.ts",
            ]
            comp_file = None
            for c in comp_candidates:
                if c.exists():
                    comp_file = c
                    break

            try:
                page_rel = str(page_file.relative_to(pdir)).replace("\\", "/")
            except ValueError:
                page_rel = str(page_file)

            if not comp_file:
                # Missing component
                comp_name = import_path.split("/")[-1]
                issues.append({
                    "type": "missing_component",
                    "file": page_rel,
                    "detail": import_path,
                    "message": f"Page {page_rel} imports '@/components/{import_path}' but component file does not exist"
                })

                # Auto-stub if requested
                if auto_stub and component_dir:
                    stub_path = component_dir / f"{import_path}.tsx"
                    stub_path.parent.mkdir(parents=True, exist_ok=True)
                    stub_content = (
                        "'use client';\n"
                        f"export default function {comp_name}() {{\n"
                        "  return (\n"
                        '    <div className="p-4 border border-dashed border-gray-600 rounded-lg">\n'
                        f'      <p className="text-gray-400">{comp_name} — placeholder</p>\n'
                        "    </div>\n"
                        "  );\n"
                        "}\n"
                    )
                    try:
                        stub_path.write_text(stub_content, encoding="utf-8")
                    except Exception:
                        pass
                continue

            # Component file exists — check it has an export and JSX
            try:
                comp_content = comp_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            try:
                comp_rel = str(comp_file.relative_to(pdir)).replace("\\", "/")
            except ValueError:
                comp_rel = str(comp_file)

            # Check for exported function
            has_export = bool(re.search(
                r'export\s+(default\s+)?function\s+\w+|export\s+default\s+\w+|module\.exports',
                comp_content
            ))
            if not has_export:
                issues.append({
                    "type": "no_export_component",
                    "file": comp_rel,
                    "detail": import_path,
                    "message": f"Component {comp_rel} has no exported function"
                })

            # Check for JSX return
            has_jsx = bool(re.search(r'return\s*\(?\s*<', comp_content))
            if not has_jsx and has_export:
                issues.append({
                    "type": "no_jsx_component",
                    "file": comp_rel,
                    "detail": import_path,
                    "message": f"Component {comp_rel} does not return JSX"
                })

    return issues


def _scan_api_validation(project_dir, file_contents, framework=None) -> list:
    """Detect POST/PUT/PATCH API route handlers that accept request.json() without input validation,
    GET handlers using query params without validation, and missing auth checks.
    Supports Next.js, SvelteKit, Remix, Express, Fastify, Hono, and generic Node.js frameworks."""
    pdir = Path(project_dir)
    issues = []
    _VALIDATION_PATTERNS = re.compile(
        r'validate|parse|safeParse|z\.object|zod|schema|if\s*\(\s*!body\.|typeof\s+body\.|\.trim\(\)|\.length'
    )
    _QUERY_VALIDATION = re.compile(
        r'validate|parse|safeParse|z\.|schema|parseInt|Number\(|isNaN|typeof\s|\.trim\(\)'
    )
    # Broadened auth patterns — word boundaries to avoid false positives
    _AUTH_PATTERNS = re.compile(
        r'\bgetUser\b|\bgetSession\b|\bgetServerSession\b|\buseSession\b'
        r'|\bcurrentUser\b|auth\(\)\.protect'
        r'|\bverifyToken\b|\bverifyIdToken\b|\bgetAuth\(\)'
        r'|\breq\.user\b|\bpassport\.\w+'
        r'|\bjwt\.verify\b|\bjwt\.decode\b'
        r'|@login_required|\bis_authenticated\b'
        r'|\bget_current_user\b|Depends\(\s*get_current'
        r'|createClient.*cookies'
    )
    _AUTH_SKIP_PATHS = ["/auth/", "/login", "/signup", "/register", "/webhook", "/health", "/public/", "/api/cron"]

    # Multi-framework handler patterns
    # Next.js: export async function POST/GET/...
    _NEXTJS_HANDLER = re.compile(r'export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE)\b')
    # SvelteKit: export const GET/POST in +server.ts
    _SVELTEKIT_HANDLER = re.compile(r'export\s+(?:const|async\s+function|function)\s+(GET|POST|PUT|PATCH|DELETE)\b')
    # Remix: export function loader/action
    _REMIX_HANDLER = re.compile(r'export\s+(?:async\s+)?function\s+(loader|action)\b')
    # Express/Fastify/Hono: app.post(...), router.get(...)
    _EXPRESS_HANDLER = re.compile(r'(?:app|router|server)\.(get|post|put|patch|delete)\s*\(')
    # Request body patterns across frameworks
    _BODY_ACCESS = re.compile(r'request\.json\(\)|req\.json\(\)|req\.body|request\.formData\(\)')

    # Determine which file filter to use based on framework
    def _is_api_file(fp, rel):
        name = fp.name.lower()
        # Normalize: prepend / for consistent segment matching
        nrel = "/" + rel
        if framework in ("express", "fastify", "hono", "koa"):
            # Express-like: any .ts/.js file in routes/, api/, controllers/
            if any(seg in nrel for seg in ("/routes/", "/api/", "/controllers/")):
                return True
            return False
        if framework == "sveltekit":
            return name in ("+server.ts", "+server.js")
        if framework == "remix":
            return name in ("route.tsx", "route.ts", "route.jsx", "route.js") or \
                   "/routes/" in nrel
        # Default: Next.js-style (also covers unknown)
        if "/api/" in nrel and name in ("route.ts", "route.js"):
            return True
        return False

    # Project-level middleware auth suppression
    middleware_has_auth = False
    for fp, content in file_contents.items():
        fname = fp.name.lower()
        if fname in ("middleware.ts", "middleware.js"):
            try:
                mrel = str(fp.relative_to(pdir)).replace("\\", "/")
            except ValueError:
                continue
            if mrel.count("/") <= 1:
                if re.search(r'\bgetUser\b|\bgetSession\b|auth\(|\bverifyToken\b|\breq\.user\b', content):
                    middleware_has_auth = True

    for fp, content in file_contents.items():
        try:
            rel = str(fp.relative_to(pdir)).replace("\\", "/")
        except ValueError:
            continue
        if not _is_api_file(fp, rel):
            continue
        lines = content.split("\n")

        # Select handler pattern based on framework
        if framework in ("express", "fastify", "hono", "koa"):
            handler_re = _EXPRESS_HANDLER
        elif framework == "sveltekit":
            handler_re = _SVELTEKIT_HANDLER
        elif framework == "remix":
            handler_re = _REMIX_HANDLER
        else:
            handler_re = _NEXTJS_HANDLER

        for i, line in enumerate(lines):
            # Flag POST, PUT, and PATCH handlers without input validation
            handler_match = handler_re.search(line)
            if handler_match:
                method = handler_match.group(1).upper()
                if method in ("POST", "PUT", "PATCH", "ACTION"):
                    # Look for request body access in the handler
                    handler_block = "\n".join(lines[i:i + 30])
                    if _BODY_ACCESS.search(handler_block):
                        # Check if validation appears within 10 lines after body access
                        body_line_idx = None
                        for j in range(i, min(i + 30, len(lines))):
                            if _BODY_ACCESS.search(lines[j]):
                                body_line_idx = j
                                break
                        if body_line_idx is not None:
                            validation_window = "\n".join(lines[body_line_idx:body_line_idx + 10])
                            if not _VALIDATION_PATTERNS.search(validation_window):
                                issues.append({
                                    "type": "unvalidated_api_input",
                                    "file": rel,
                                    "detail": f"{method} handler at line {i + 1}",
                                    "message": f"{method} handler in {rel} accepts request body without input validation"
                                })

                # Flag GET/LOADER handlers using query params without validation
                if method in ("GET", "LOADER"):
                    handler_body = "\n".join(lines[i:i + 40])
                    if re.search(r'searchParams|req\.query|url\.searchParams|request\.nextUrl\.searchParams|request\.url', handler_body):
                        if not _QUERY_VALIDATION.search(handler_body):
                            issues.append({
                                "type": "unvalidated_get_params",
                                "file": rel,
                                "detail": f"{method} handler at line {i + 1}",
                                "message": f"{method} handler in {rel} uses query parameters without validation"
                            })

                # Auth check — scoped to handler function body
                if not middleware_has_auth:
                    if any(sp in rel for sp in _AUTH_SKIP_PATHS):
                        continue
                    body_end = min(i + 50, len(lines))
                    for j in range(i + 1, len(lines)):
                        stripped = lines[j].lstrip()
                        if stripped.startswith("}") and len(lines[j]) - len(stripped) <= 2:
                            body_end = j
                            break
                    handler_body = "\n".join(lines[i:body_end])
                    if not _AUTH_PATTERNS.search(handler_body):
                        issues.append({
                            "type": "missing_api_auth",
                            "file": rel,
                            "detail": f"{method} handler at line {i + 1}",
                            "message": f"{method} handler in {rel} may be missing authentication check"
                        })
    return issues


def _scan_dashboard_layout(project_dir, file_contents) -> list:
    """Detect missing dashboard layout or duplicated Sidebar/Header imports in pages."""
    pdir = Path(project_dir)
    issues = []
    dashboard_dir = pdir / "app" / "dashboard"
    if not dashboard_dir.is_dir():
        # Also check src/app/dashboard
        dashboard_dir = pdir / "src" / "app" / "dashboard"
        if not dashboard_dir.is_dir():
            return issues

    # Count page files in dashboard directory tree
    page_files = []
    for fp in dashboard_dir.rglob("page.tsx"):
        page_files.append(fp)
    for fp in dashboard_dir.rglob("page.jsx"):
        page_files.append(fp)

    if len(page_files) < 3:
        return issues

    # Check if dashboard layout exists
    layout_exists = (dashboard_dir / "layout.tsx").exists() or (dashboard_dir / "layout.jsx").exists()
    if not layout_exists:
        try:
            rel = str(dashboard_dir.relative_to(pdir)).replace("\\", "/")
        except ValueError:
            rel = str(dashboard_dir)
        issues.append({
            "type": "missing_dashboard_layout",
            "file": rel,
            "detail": f"{len(page_files)} dashboard pages without layout",
            "message": f"Dashboard has {len(page_files)} pages but no dashboard/layout.tsx — Sidebar/Header should be in a shared layout"
        })

    # Check if pages duplicate Sidebar/Header imports
    sidebar_import_count = 0
    for fp in page_files:
        content = file_contents.get(fp, "")
        if not content:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
        if re.search(r'import\s+.*(?:Sidebar|Header)', content):
            sidebar_import_count += 1

    if sidebar_import_count >= 3:
        issues.append({
            "type": "duplicated_layout_components",
            "file": "app/dashboard/",
            "detail": f"{sidebar_import_count} pages import Sidebar/Header",
            "message": f"{sidebar_import_count} dashboard pages import Sidebar/Header directly — move to dashboard/layout.tsx instead"
        })

    return issues


def _scan_hardcoded_date_ranges(project_dir, file_contents) -> list:
    """Detect components using hardcoded date offsets instead of data-driven ranges."""
    pdir = Path(project_dir)
    issues = []
    _HARDCODED_DATE_PATTERNS = re.compile(
        r'getDate\(\)\s*[-+]\s*\d{2,}'       # getDate() - 7, getDate() + 21
        r'|addDays\(\s*(?:today|now)'          # addDays(today, ...)
        r'|subDays\(\s*(?:today|now)'          # subDays(today, ...)
        r'|setDate\([^)]*[-+]\s*\d{2,}\)'     # setDate(getDate() - 14)
    )
    flagged_files = set()
    for fp, content in file_contents.items():
        if fp in flagged_files:
            continue
        ext = fp.suffix.lower()
        if ext not in (".tsx", ".jsx", ".ts", ".js"):
            continue
        # Only check client components (likely visualization)
        if "'use client'" not in content and '"use client"' not in content:
            continue
        if _HARDCODED_DATE_PATTERNS.search(content):
            try:
                rel = str(fp.relative_to(pdir)).replace("\\", "/")
            except ValueError:
                continue
            flagged_files.add(fp)
            issues.append({
                "type": "possible_hardcoded_date_range",
                "file": rel,
                "detail": "hardcoded date arithmetic detected",
                "message": f"Component {rel} may use hardcoded date offsets — consider deriving range from data"
            })
    return issues


# ---------------------------------------------------------------------------
# wiring agent — automated cross-file verification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Design quality scanning — detect AI design slop + conversion anti-patterns
# ---------------------------------------------------------------------------

_DESIGN_SCAN_PATTERNS = None
_CONVERSION_SCAN_PATTERNS = None


def _get_design_scan_patterns():
    """Lazily compile design scan regexes from design.json."""
    global _DESIGN_SCAN_PATTERNS
    if _DESIGN_SCAN_PATTERNS is not None:
        return _DESIGN_SCAN_PATTERNS
    design_data = _load_design_json()
    if not design_data:
        _DESIGN_SCAN_PATTERNS = []
        return []
    raw = design_data.get("slop_scan_patterns_design", {}).get("patterns", [])
    compiled = []
    for p in raw:
        regex_str = p.get("regex")
        if not regex_str:
            continue
        try:
            compiled.append({**p, "_regex": re.compile(regex_str, re.MULTILINE)})
        except re.error:
            pass
    _DESIGN_SCAN_PATTERNS = compiled
    return _DESIGN_SCAN_PATTERNS


def _get_conversion_scan_patterns():
    """Lazily compile conversion scan regexes from design.json."""
    global _CONVERSION_SCAN_PATTERNS
    if _CONVERSION_SCAN_PATTERNS is not None:
        return _CONVERSION_SCAN_PATTERNS
    design_data = _load_design_json()
    if not design_data:
        _CONVERSION_SCAN_PATTERNS = []
        return []
    raw = design_data.get("slop_scan_patterns_conversion", {}).get("patterns", [])
    compiled = []
    for p in raw:
        regex_str = p.get("regex")
        if regex_str:
            try:
                compiled.append({**p, "_regex": re.compile(regex_str, re.MULTILINE)})
            except re.error:
                pass
        else:
            compiled.append(p)  # non-regex checks (file_pattern + check)
    _CONVERSION_SCAN_PATTERNS = compiled
    return _CONVERSION_SCAN_PATTERNS


def _stem_matches_exception(filepath, exceptions):
    """Check if file stem matches any exception keyword."""
    if not exceptions:
        return False
    stem = Path(filepath).stem
    stem_lower = stem.lower()
    stem_parts = set(re.split(r'[-_]', stem_lower))
    for exc in exceptions:
        exc_lower = exc.lower()
        if stem_lower == exc_lower or exc_lower in stem_parts:
            return True
    return False


def _scan_design_quality(project_dir, file_contents):
    """Scan for design slop patterns. Returns list of issue dicts."""
    patterns = _get_design_scan_patterns()
    if not patterns:
        return []
    pdir = Path(project_dir)
    issues = []

    # Check if project uses design system CSS vars — skip raw_tailwind_colors if not
    has_design_system = False
    for fp, content in file_contents.items():
        if "var(--bg-primary)" in content or "var(--surface)" in content:
            has_design_system = True
            break

    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        if ext not in (".tsx", ".jsx", ".html", ".htm"):
            continue
        try:
            rel = str(fp.relative_to(pdir)).replace("\\", "/")
        except ValueError:
            rel = str(fp)
        fname = fp.name

        for pattern in patterns:
            pname = pattern.get("name", "")
            # Skip raw_tailwind_colors for pre-design-system projects
            if pname == "raw_tailwind_colors" and not has_design_system:
                continue
            # Check exceptions on filename stem
            exceptions = pattern.get("exceptions", [])
            if _stem_matches_exception(fp, exceptions):
                continue
            regex = pattern.get("_regex")
            if not regex:
                continue
            # Special handling: missing_transitions — scan line by line
            if pname == "missing_transitions":
                match_count = 0
                for line_no, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        match_count += 1
                        if match_count <= 5:
                            issues.append({
                                "type": f"design_{pname}",
                                "severity": pattern.get("severity", "low"),
                                "file": rel,
                                "line": line_no,
                                "message": pattern.get("message", ""),
                            })
                if match_count > 5:
                    issues.append({
                        "type": f"design_{pname}",
                        "severity": pattern.get("severity", "low"),
                        "file": rel,
                        "message": f"...and {match_count - 5} more instances in {fname}",
                    })
                continue
            # Standard: match against whole file
            matches = list(regex.finditer(content))
            if not matches:
                continue
            for i, m in enumerate(matches[:5]):
                line_no = content[:m.start()].count("\n") + 1
                issues.append({
                    "type": f"design_{pname}",
                    "severity": pattern.get("severity", "medium"),
                    "file": rel,
                    "line": line_no,
                    "message": pattern.get("message", ""),
                })
            if len(matches) > 5:
                issues.append({
                    "type": f"design_{pname}",
                    "severity": pattern.get("severity", "medium"),
                    "file": rel,
                    "message": f"...and {len(matches) - 5} more instances in {fname}",
                })
    return issues


# ---------------------------------------------------------------------------
# Security quality scanning — detect vulnerabilities + misconfigurations
# ---------------------------------------------------------------------------

_SECURITY_SCAN_PATTERNS = None


def _get_security_scan_patterns():
    """Lazily compile security scan regexes from security.json."""
    global _SECURITY_SCAN_PATTERNS
    if _SECURITY_SCAN_PATTERNS is not None:
        return _SECURITY_SCAN_PATTERNS
    sec_data = _load_security_json()
    if not sec_data:
        _SECURITY_SCAN_PATTERNS = []
        return []
    raw = sec_data.get("security_scan_patterns", {}).get("patterns", [])
    compiled = []
    for p in raw:
        entry = dict(p)
        regex_str = p.get("regex")
        if regex_str:
            try:
                entry["_regex"] = re.compile(regex_str, re.MULTILINE)
            except re.error:
                continue
        neg_regex_str = p.get("negative_regex")
        if neg_regex_str:
            try:
                entry["_neg_regex"] = re.compile(neg_regex_str, re.MULTILINE)
            except re.error:
                pass
        compiled.append(entry)
    _SECURITY_SCAN_PATTERNS = compiled
    return _SECURITY_SCAN_PATTERNS


def _scan_security_quality(project_dir, file_contents):
    """Scan for security vulnerabilities. Returns list of issue dicts."""
    patterns = _get_security_scan_patterns()
    if not patterns:
        return []
    pdir = Path(project_dir)
    issues = []

    # Migration path: detect if project is security-aware
    security_aware = False
    has_middleware = False
    middleware_has_auth = False
    for fp, content in file_contents.items():
        fname = fp.name.lower()
        if "X-Content-Type-Options" in content:
            security_aware = True
        if fname in ("middleware.ts", "middleware.js"):
            # Check if it's in project root (not nested deep)
            try:
                rel = str(fp.relative_to(pdir)).replace("\\", "/")
            except ValueError:
                continue
            if rel.count("/") <= 1:  # root or src/
                has_middleware = True
                if re.search(r'getUser|getSession|auth\(|verifyToken', content):
                    middleware_has_auth = True

    # Patterns to skip for non-security-aware projects
    skip_for_legacy = {"missing_auth_check", "missing_rate_limit"}

    for fp, content in file_contents.items():
        ext = fp.suffix.lower()
        try:
            rel = str(fp.relative_to(pdir)).replace("\\", "/")
        except ValueError:
            continue

        for pattern in patterns:
            pname = pattern.get("name", "")

            # Migration path: skip certain patterns for legacy projects
            if not security_aware and pname in skip_for_legacy:
                continue

            # File type filter
            file_types = pattern.get("file_types", [])
            if file_types and ext not in file_types:
                continue

            # Exception check on filename stem
            exceptions = pattern.get("exceptions", [])
            if exceptions and _stem_matches_exception(fp, exceptions):
                continue

            # --- Special handling for missing_auth_check ---
            if pname == "missing_auth_check":
                # Skip entirely if middleware handles auth
                if middleware_has_auth:
                    continue
                # Only check API route files
                if "/api/" not in rel or fp.name not in ("route.ts", "route.js"):
                    continue
                # Path-based exceptions
                skip_paths = pattern.get("skip_paths", [])
                if any(sp in rel for sp in skip_paths):
                    continue
                # Check if handler has auth keywords
                auth_kws = pattern.get("auth_keywords", [])
                has_auth = any(kw in content for kw in auth_kws if ".*" not in kw)
                if not has_auth and not re.search(r'createClient.*cookies', content):
                    has_auth = False
                else:
                    has_auth = True
                if not has_auth and re.search(r'export\s+async\s+function\s+(?:GET|POST|PUT|PATCH|DELETE)\b', content):
                    issues.append({
                        "type": f"security_{pname}",
                        "severity": pattern.get("severity", "low"),
                        "file": rel,
                        "message": pattern.get("message", ""),
                    })
                continue

            # --- Special handling for missing_rate_limit ---
            if pname == "missing_rate_limit":
                if not security_aware:
                    continue
                path_filter = pattern.get("path_filter", [])
                if not any(pf in rel for pf in path_filter):
                    continue
                rate_kws = pattern.get("rate_limit_keywords", [])
                regex = pattern.get("_regex")
                if regex and regex.search(content):
                    if not any(rk in content for rk in rate_kws):
                        issues.append({
                            "type": f"security_{pname}",
                            "severity": pattern.get("severity", "medium"),
                            "file": rel,
                            "message": pattern.get("message", ""),
                        })
                continue

            # --- Standard regex patterns ---
            regex = pattern.get("_regex")
            if not regex:
                continue

            # Same-file skip check (e.g., bcrypt in file skips cleartext_password)
            same_file_skip = pattern.get("same_file_skip", [])
            if same_file_skip and any(sk in content for sk in same_file_skip):
                continue

            matches = list(regex.finditer(content))
            if not matches:
                continue

            # Negative regex: if present in same line context, skip match
            neg_regex = pattern.get("_neg_regex")
            filtered_matches = []
            for m in matches:
                if neg_regex:
                    # Check surrounding context (50 lines after match)
                    line_start = content.rfind("\n", 0, m.start()) + 1
                    context_end = min(m.end() + 2000, len(content))
                    context = content[line_start:context_end]
                    if neg_regex.search(context):
                        continue
                filtered_matches.append(m)

            for m in filtered_matches[:5]:
                line_no = content[:m.start()].count("\n") + 1
                issues.append({
                    "type": f"security_{pname}",
                    "severity": pattern.get("severity", "medium"),
                    "file": rel,
                    "line": line_no,
                    "message": pattern.get("message", ""),
                })
            if len(filtered_matches) > 5:
                issues.append({
                    "type": f"security_{pname}",
                    "severity": pattern.get("severity", "medium"),
                    "file": rel,
                    "message": f"...and {len(filtered_matches) - 5} more instances in {fp.name}",
                })

    return issues


def _scan_conversion_quality(project_dir, file_contents):
    """Scan for conversion anti-patterns. Returns list of issue dicts."""
    patterns = _get_conversion_scan_patterns()
    if not patterns:
        return []
    pdir = Path(project_dir)
    issues = []

    for pattern in patterns:
        pname = pattern.get("name", "")
        regex = pattern.get("_regex")
        file_pattern_str = pattern.get("file_pattern")
        check_desc = pattern.get("check")

        # Determine which files to scan
        target_files = {}
        if file_pattern_str:
            try:
                fp_regex = re.compile(file_pattern_str, re.IGNORECASE)
            except re.error:
                continue
            for fp, content in file_contents.items():
                ext = fp.suffix.lower()
                if ext not in (".tsx", ".jsx", ".html", ".htm"):
                    continue
                try:
                    rel = str(fp.relative_to(pdir)).replace("\\", "/")
                except ValueError:
                    rel = str(fp)
                if fp_regex.search(rel):
                    target_files[fp] = (content, rel)
        else:
            for fp, content in file_contents.items():
                ext = fp.suffix.lower()
                if ext not in (".tsx", ".jsx", ".html", ".htm"):
                    continue
                try:
                    rel = str(fp.relative_to(pdir)).replace("\\", "/")
                except ValueError:
                    rel = str(fp)
                target_files[fp] = (content, rel)

        if regex:
            # Regex match check
            for fp, (content, rel) in target_files.items():
                matches = list(regex.finditer(content))
                for m in matches[:5]:
                    line_no = content[:m.start()].count("\n") + 1
                    issues.append({
                        "type": f"conversion_{pname}",
                        "severity": pattern.get("severity", "medium"),
                        "file": rel,
                        "line": line_no,
                        "message": pattern.get("message", ""),
                    })
        elif check_desc and file_pattern_str:
            # Content-absence check — scoped to matching files only
            absence_keywords = []
            if "no_risk_reversal" in pname:
                absence_keywords = ["guarantee", "money-back", "risk-free", "cancel anytime", "refund"]
            elif "missing_social_proof" in pname:
                absence_keywords = ["testimonial", "review", "customer", "logo", "trust", "social proof"]
            if absence_keywords:
                for fp, (content, rel) in target_files.items():
                    content_lower = content.lower()
                    if not any(kw in content_lower for kw in absence_keywords):
                        issues.append({
                            "type": f"conversion_{pname}",
                            "severity": pattern.get("severity", "medium"),
                            "file": rel,
                            "message": pattern.get("message", check_desc),
                        })
    return issues


class WiringAgent:
    """Automated cross-file verification after generation batches."""

    def __init__(self, project_dir, auto_stub=False):
        self.project_dir = project_dir
        self.auto_stub = auto_stub
        self.issues = []
        self.auto_fixed = []
        self.manual_needed = []
        self._file_contents = None

    def run_full_scan(self):
        """Run all wiring checks with shared file cache."""
        self._file_contents = self._build_file_cache()
        self._framework = _detect_project_framework(self.project_dir)
        self._ensure_graph()
        self.issues = scan_wiring_issues(self.project_dir)
        self.issues.extend(scan_css_coherence(self.project_dir, self._file_contents))
        self.issues.extend(scan_api_coherence(self.project_dir, self._file_contents))
        self.issues.extend(scan_asset_references(self.project_dir, self._file_contents))
        self.issues.extend(_scan_render_coherence(self.project_dir, auto_stub=self.auto_stub))
        self.issues.extend(_scan_api_validation(self.project_dir, self._file_contents, framework=self._framework))
        self.issues.extend(_scan_dashboard_layout(self.project_dir, self._file_contents))
        self.issues.extend(_scan_hardcoded_date_ranges(self.project_dir, self._file_contents))
        self.issues.extend(_scan_design_quality(self.project_dir, self._file_contents))
        self.issues.extend(_scan_conversion_quality(self.project_dir, self._file_contents))
        self.issues.extend(_scan_security_quality(self.project_dir, self._file_contents))
        self._deduplicate_with_quality_gate()
        self._classify()
        return self.issues

    def auto_fix(self):
        """Fix what can be auto-fixed."""
        pdir = Path(self.project_dir)
        fixed = []
        for issue in self.auto_fixed:
            if issue["type"] == "missing_env_var":
                # Append placeholder to .env.example
                env_example = pdir / ".env.example"
                var = issue.get("variable", "")
                if var and env_example.exists():
                    try:
                        content = env_example.read_text(encoding="utf-8", errors="replace")
                        if var not in content:
                            with open(str(env_example), "a", encoding="utf-8") as f:
                                f.write(f"\n{var}=\n")
                            fixed.append(issue)
                    except Exception:
                        pass
            elif issue["type"] == "env_setup_needed":
                # Copy .env.example to .env
                env_example = pdir / ".env.example"
                env_file = pdir / ".env"
                if env_example.exists() and not env_file.exists():
                    try:
                        env_file.write_text(env_example.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                        fixed.append(issue)
                    except Exception:
                        pass
        return fixed

    def format_report(self):
        """Format colored terminal report."""
        return _display_wiring_report_text(self)

    def format_prompt(self):
        """Format for model injection."""
        extended_labels = {
            "css_orphan": "CSS Class Issues (class used but not defined in CSS)",
            "missing_api_route": "Missing API Routes (fetch/axios calls to non-existent routes)",
            "missing_asset": "Missing Assets (src/href references to non-existent files)",
            "missing_component": "Missing Components (imported but file does not exist)",
            "no_export_component": "Components Without Exports (file exists but no exported function)",
            "no_jsx_component": "Components Without JSX (exported function but no JSX return)",
            "unvalidated_api_input": "Unvalidated API Input (POST/PUT/PATCH accepts request body without validation)",
            "unvalidated_get_params": "Unvalidated GET Parameters (query params used without validation)",
            "missing_api_auth": "Missing API Authentication (handler without getUser/getSession check)",
            "missing_dashboard_layout": "Missing Dashboard Layout (multiple pages without shared layout)",
            "duplicated_layout_components": "Duplicated Layout Components (Sidebar/Header imported in multiple pages)",
            "possible_hardcoded_date_range": "Hardcoded Date Ranges (component uses fixed date offsets instead of data-driven)",
            "security_sql_injection": "SQL Injection Risk (string interpolation in database queries)",
            "security_wildcard_cors": "Wildcard CORS (origin: '*' allows any website access)",
            "security_exposed_server_secret": "Exposed Server Secret (client env prefix on server-side secret)",
            "security_innerHTML_xss": "XSS Risk (innerHTML/dangerouslySetInnerHTML without sanitization)",
            "security_eval_usage": "Unsafe eval() (arbitrary code execution risk)",
            "security_hardcoded_api_key": "Hardcoded API Key (move to .env)",
            "security_cleartext_password": "Cleartext Password (use bcrypt/argon2 hashing)",
            "security_jwt_in_localstorage": "JWT in localStorage (use httpOnly cookies)",
            "security_missing_auth_check": "Missing Auth Check (API handler without authentication)",
            "security_missing_rate_limit": "Missing Rate Limit (auth endpoint without rate limiting)",
            "security_unvalidated_redirect": "Unvalidated Redirect (redirect from user input)",
            "security_missing_input_length": "Missing Input Length (text input without maxLength)",
            "security_unvalidated_file_upload": "Unvalidated File Upload (no type/size check)",
            "security_insecure_cookie": "Insecure Cookie (missing httpOnly/secure/sameSite)",
        }
        # Build combined report
        all_issues = self.manual_needed
        if not all_issues:
            return ""
        lines = ["## Wiring Issues Detected", "Fix each issue below:", ""]
        by_type = {}
        for issue in all_issues:
            by_type.setdefault(issue["type"], []).append(issue)

        type_labels = {
            "orphaned_export": "Orphaned Exports",
            "unrendered_component": "Unrendered Components",
            "broken_import": "Broken Imports",
            "missing_env_var": "Missing Environment Variables",
            "dead_route": "Dead-End Routes",
            "missing_native_config": "Missing Native Module Config",
            "client_server_violation": "Client/Server Boundary Violations",
        }
        type_labels.update(extended_labels)

        for issue_type, items in by_type.items():
            lines.append(f"### {type_labels.get(issue_type, issue_type)}")
            for item in items[:10]:
                lines.append(f"- {item['message']}")
            if len(items) > 10:
                lines.append(f"- ... and {len(items) - 10} more")
            lines.append("")

        lines.append("Fix these issues by creating missing files, adding missing imports, wiring up components, or defining missing env vars.")
        return "\n".join(lines)

    def _classify(self):
        auto_types = {"missing_env_var", "env_setup_needed"}
        self.auto_fixed = []
        self.manual_needed = []
        for issue in self.issues:
            if issue["type"] in auto_types:
                self.auto_fixed.append(issue)
            else:
                self.manual_needed.append(issue)

    def _build_file_cache(self):
        """Build shared file contents dict, reusable across scan functions."""
        pdir = Path(self.project_dir)
        file_contents = {}
        file_count = 0
        for fp in pdir.rglob("*"):
            if file_count >= 200:
                break
            if not fp.is_file():
                continue
            if any(skip in fp.parts for skip in _WIRING_SKIP_DIRS):
                continue
            ext = fp.suffix.lower()
            if ext in (".js", ".ts", ".jsx", ".tsx", ".py", ".html", ".css", ".scss"):
                try:
                    file_contents[fp] = fp.read_text(encoding="utf-8", errors="replace")
                    file_count += 1
                except Exception:
                    pass
        return file_contents

    def _ensure_graph(self):
        """Ensure project graph is populated (from cache or fresh build)."""
        try:
            _get_project_graph()
        except Exception:
            pass

    def _deduplicate_with_quality_gate(self):
        """Remove duplicate issues (same file + type + detail prefix)."""
        seen = set()
        deduped = []
        for issue in self.issues:
            key = (issue.get("file", ""), issue.get("type", ""), issue.get("detail", issue.get("message", ""))[:80])
            if key not in seen:
                seen.add(key)
                deduped.append(issue)
        self.issues = deduped


def _display_wiring_report_text(agent):
    """Display colored wiring report to terminal."""
    print(f"  {C.CLAW}{BLACK_CIRCLE} Wiring Check{C.RESET}")
    try:
        w = min(os.get_terminal_size().columns, 120)
    except (OSError, ValueError):
        w = 80
    print(f"  {C.SUBTLE}{'─' * (w - 4)}{C.RESET}")

    # Group by type
    by_type = {}
    for issue in agent.issues:
        by_type.setdefault(issue["type"], []).append(issue)

    check_labels = {
        "orphaned_export": "Exports",
        "unrendered_component": "Components",
        "broken_import": "Imports",
        "missing_env_var": "Env vars",
        "dead_route": "Routes",
        "css_orphan": "CSS classes",
        "missing_api_route": "API routes",
        "missing_asset": "Assets",
    }

    # Show all check categories
    for check_type, label in check_labels.items():
        items = by_type.get(check_type, [])
        if items:
            print(f"  {C.ERROR}✗{C.RESET} {C.TEXT}{label}: {len(items)} issue(s){C.RESET}")
        else:
            print(f"  {C.SUCCESS}✓{C.RESET} {C.DIM}{label}: 0 issues{C.RESET}")

    auto_count = len(agent.auto_fixed)
    manual_count = len(agent.manual_needed)
    print(f"  {C.SUBTLE}Auto-fixed: {auto_count} | Manual: {manual_count}{C.RESET}")


def _display_wiring_report(agent):
    """Display wiring report to terminal (convenience wrapper)."""
    _display_wiring_report_text(agent)


# ---------------------------------------------------------------------------
# edge case detection -- harden the happy path
# ---------------------------------------------------------------------------

def detect_built_features(project_dir):
    """Scan files to detect what features were built. Returns list of feature dicts."""
    pdir = Path(project_dir)
    if not pdir.exists():
        return []

    # Count source files to gate (skip trivial builds)
    source_count = 0
    for fp in pdir.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in (".js", ".ts", ".jsx", ".tsx", ".py", ".html"):
            if not any(skip in fp.parts for skip in _WIRING_SKIP_DIRS):
                source_count += 1
    if source_count < 3:
        return []

    features = []
    file_count = 0

    for fp in pdir.rglob("*"):
        if file_count >= 200:
            break
        if not fp.is_file():
            continue
        if any(skip in fp.parts for skip in _WIRING_SKIP_DIRS):
            continue
        ext = fp.suffix.lower()
        if ext not in (".js", ".ts", ".jsx", ".tsx", ".py", ".html"):
            continue
        file_count += 1
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        try:
            rel = str(fp.relative_to(pdir))
        except ValueError:
            rel = str(fp)

        # Forms
        if re.search(r'<form|onSubmit|handleSubmit', content):
            features.append({
                "type": "form",
                "file": rel,
                "checks": ["empty submission", "missing required fields", "validation error display"]
            })

        # API routes
        if re.search(r'NextResponse|app\.(get|post|put|delete|patch)\(|router\.(get|post|put|delete|patch)\(', content):
            features.append({
                "type": "api_route",
                "file": rel,
                "checks": ["500 server error response", "malformed request body", "authentication check", "rate limiting"]
            })

        # Auth flows
        if re.search(r'signIn|signOut|session|useSession|getServerSession|JWT|jwt|getToken', content, re.IGNORECASE):
            features.append({
                "type": "auth",
                "file": rel,
                "checks": ["expired token handling", "unauthenticated access redirect", "session refresh"]
            })

        # Database operations
        if re.search(r'prisma\.|\.select\(|\.insert\(|\.update\(|\.delete\(|\.findMany|\.findUnique|\.create\(', content):
            features.append({
                "type": "database",
                "file": rel,
                "checks": ["record not found (404)", "database connection failure", "unique constraint violation"]
            })

        # Payment
        if re.search(r'stripe|checkout|payment|Stripe', content):
            features.append({
                "type": "payment",
                "file": rel,
                "checks": ["payment failure handling", "webhook signature verification", "duplicate payment prevention"]
            })

    return features


def generate_edge_case_prompt(features):
    """Generate a prompt asking the model to verify and harden detected features."""
    if not features:
        return ""
    lines = [
        "## Edge Case Hardening",
        "The following features were detected in the project. Verify each has proper edge case handling:",
        ""
    ]
    by_type = {}
    for f in features:
        by_type.setdefault(f["type"], []).append(f)

    type_labels = {
        "form": "Forms",
        "api_route": "API Routes",
        "auth": "Authentication",
        "database": "Database Operations",
        "payment": "Payments",
    }
    for ftype, items in by_type.items():
        lines.append(f"### {type_labels.get(ftype, ftype)}")
        for item in items[:5]:
            lines.append(f"- **{item['file']}**: check for: {', '.join(item['checks'])}")
        lines.append("")

    lines.append("For each feature, verify the edge cases are handled. If not, add proper error handling, loading states, and user feedback.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# file manifest tracker for chunked generation
# ---------------------------------------------------------------------------

class ProjectManifest:
    """Track files created during plan execution for context injection."""
    def __init__(self):
        self.files_created = []  # list of file paths
        self.files_modified = []
        self.steps_completed = []

    def record_tool_result(self, tool_name, tool_args, result):
        """Record a tool execution for manifest tracking."""
        if tool_name == "write_file" and not result.startswith("Error"):
            fp = tool_args.get("file_path", "")
            if fp and fp not in self.files_created:
                self.files_created.append(fp)
        elif tool_name == "edit_file" and not result.startswith("Error"):
            fp = tool_args.get("file_path", "")
            if fp and fp not in self.files_modified:
                self.files_modified.append(fp)

    def get_context_summary(self):
        """Return a summary of what's been created so far."""
        parts = []
        if self.files_created:
            parts.append("Files created so far:\n" + "\n".join(f"  - {f}" for f in self.files_created))
        if self.files_modified:
            parts.append("Files modified so far:\n" + "\n".join(f"  - {f}" for f in self.files_modified))
        if self.steps_completed:
            parts.append("Steps completed:\n" + "\n".join(f"  - {s}" for s in self.steps_completed))
        return "\n\n".join(parts) if parts else ""


# global manifest for the current session
_session_manifest = ProjectManifest()

# Edit history — tracks last 5 file operations for retry context
_edit_history = deque(maxlen=5)


# ---------------------------------------------------------------------------
# smart scaffolding — structured intent extraction & project generation
# ---------------------------------------------------------------------------

class ProjectSpec:
    """Structured representation of what the user wants to build."""

    def __init__(self):
        self.project_type = None       # web-app, api, cli-tool, static-site
        self.frontend = None           # react, vue, svelte, none
        self.backend = None            # nextjs, fastapi, express, django, none
        self.database = None           # supabase, postgres, sqlite, mongodb, none
        self.language = None           # typescript, javascript, python
        self.styling = None            # tailwind, css-modules, styled-components
        self.auth = None               # supabase-auth, nextauth, clerk, custom, none
        self.features = []             # ["auth", "payments", "dashboard", "crud"]
        self.complexity = "medium"     # simple, medium, complex
        self.scope = "mvp"             # mvp, full
        self.signals_explicit = {}     # directly stated by user
        self.signals_inferred = {}     # inferred from context
        self.signals_missing = []      # still unknown
        self.file_manifest = []        # [{path, description, category, order, action}]
        self.description = ""          # original user text

    def confidence_score(self):
        fields = [self.project_type, self.frontend, self.backend, self.language]
        filled = sum(1 for f in fields if f is not None)
        feature_bonus = min(len(self.features) * 0.05, 0.2)
        return min((filled / len(fields)) + feature_bonus, 1.0)


class ProjectProfile:
    """Deep analysis of what already exists in the project directory."""

    def __init__(self, project_dir=None):
        self.dir = Path(project_dir or CWD)
        self.base_info = detect_project_type(str(self.dir))
        self.framework = self.base_info.get("framework")
        self.styling = None
        self.test_framework = None
        self.package_manager = None
        self.naming_convention = None
        self.existing_files = []
        self._fingerprint = None
        self._scan()

    def _scan(self):
        """Scan project for detailed profile."""
        d = self.dir

        # Detect styling
        if (d / "tailwind.config.js").exists() or (d / "tailwind.config.ts").exists():
            self.styling = "tailwind"
        elif (d / "postcss.config.js").exists():
            self.styling = "postcss"
        else:
            pkg_path = d / "package.json"
            if pkg_path.exists():
                try:
                    pkg = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    if "styled-components" in deps:
                        self.styling = "styled-components"
                    elif "@emotion/react" in deps:
                        self.styling = "emotion"
                    elif "tailwindcss" in deps:
                        self.styling = "tailwind"
                except Exception:
                    pass

        # Detect test framework
        for cfg, fw in [("jest.config.js", "jest"), ("jest.config.ts", "jest"),
                        ("vitest.config.ts", "vitest"), ("vitest.config.js", "vitest"),
                        ("pytest.ini", "pytest"), ("conftest.py", "pytest"),
                        (".mocharc.yml", "mocha"), (".mocharc.json", "mocha")]:
            if (d / cfg).exists():
                self.test_framework = fw
                break
        if not self.test_framework and (d / "pyproject.toml").exists():
            try:
                toml_text = (d / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
                if "pytest" in toml_text:
                    self.test_framework = "pytest"
            except Exception:
                pass

        # Detect package manager
        if (d / "pnpm-lock.yaml").exists():
            self.package_manager = "pnpm"
        elif (d / "yarn.lock").exists():
            self.package_manager = "yarn"
        elif (d / "package-lock.json").exists():
            self.package_manager = "npm"
        elif (d / "requirements.txt").exists() or (d / "pyproject.toml").exists():
            self.package_manager = "pip"

        # Detect naming convention by sampling files
        try:
            sample_files = [f.name for f in d.iterdir() if f.is_file() and not f.name.startswith('.')][:20]
            kebab = sum(1 for f in sample_files if '-' in f and '_' not in f)
            snake = sum(1 for f in sample_files if '_' in f and '-' not in f)
            camel = sum(1 for f in sample_files if any(c.isupper() for c in f[1:]) and '-' not in f and '_' not in f)
            if kebab > snake and kebab > camel:
                self.naming_convention = "kebab-case"
            elif snake > kebab and snake > camel:
                self.naming_convention = "snake_case"
            elif camel > 0:
                self.naming_convention = "camelCase"
        except Exception:
            pass

        # Build existing files list (top 2 levels)
        try:
            for f in d.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(d)
                    if len(rel.parts) <= 2 and not any(p.startswith('.') for p in rel.parts):
                        self.existing_files.append(str(rel))
        except Exception:
            pass

        self._fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self):
        """Hash of key config files for cache invalidation."""
        import hashlib
        h = hashlib.md5()
        for cfg in ["package.json", "tsconfig.json", "requirements.txt", "pyproject.toml",
                     "tailwind.config.js", "tailwind.config.ts", "vite.config.ts"]:
            p = self.dir / cfg
            if p.exists():
                try:
                    h.update(p.read_bytes()[:2000])
                except Exception:
                    pass
        return h.hexdigest()

    def is_stale(self, cached_fingerprint):
        return self._fingerprint != cached_fingerprint

    def to_prompt_injection(self):
        """Format for system prompt. Includes framework-specific directives."""
        lines = ["\n# Project Profile"]
        lines.append(f"- Type: **{self.base_info['type']}**")
        if self.framework:
            lines.append(f"- Framework: **{self.framework}**")
        if self.styling:
            lines.append(f"- Styling: **{self.styling}**")
        if self.test_framework:
            lines.append(f"- Tests: **{self.test_framework}**")
        if self.package_manager:
            lines.append(f"- Package manager: **{self.package_manager}**")
        if self.naming_convention:
            lines.append(f"- Naming: **{self.naming_convention}**")
        # Framework-specific directives
        directives = {
            "nextjs": (
                "Use App Router (app/), not Pages Router. Server Components by default.\n"
                "- 'use client' FIRST LINE on any file using: hooks, event handlers, framer-motion, browser APIs. NOT on page.tsx/layout.tsx unless interactive.\n"
                "- 'use server' on async Server Action functions in forms.\n"
                "- REQUIRED configs: next.config.mjs, tsconfig.json. If Tailwind: postcss.config.js + tailwind.config.ts.\n"
                "- Use next/image (not <img>), next/font (not Google Font <link>).\n"
                "- Create error.tsx (client component) + loading.tsx per route for production apps.\n"
                "- Guard browser APIs: typeof window !== 'undefined' before window/document/localStorage.\n"
                "- Animations: use key prop, AnimatePresence, or useAnimate — never bare animate={{}}. layout animations need overflow:hidden on parent.\n"
                "- CSS keyframes in @layer base (not utilities). Always include prefers-reduced-motion.\n"
                "- 3D transforms: parent needs perspective, child needs transform-style:preserve-3d.\n"
                "- Component APIs must match spec exactly. No @apply outside globals.css.\n"
                "- NEVER use href='#'. Use next/link <Link> with real routes. If sidebar has Analytics/Settings, CREATE /analytics/page.tsx and /settings/page.tsx.\n"
                "- Every nav link MUST have a corresponding page file. Incomplete navigation = broken app."
            ),
            "react": (
                "Functional components only. Hooks for all state/effects.\n"
                "- key prop on EVERY mapped JSX element — no index keys on reorderable lists.\n"
                "- Animations: key prop, AnimatePresence, or useAnimate for value-change triggers.\n"
                "- Guard browser APIs with typeof window !== 'undefined' in SSR contexts.\n"
                "- useMemo/useCallback for expensive computations and stable references in deps arrays.\n"
                "- Error boundaries for production apps."
            ),
            "nuxt": (
                "Use Composition API with <script setup>. Do NOT suggest React patterns.\n"
                "- REQUIRED configs: nuxt.config.ts, tsconfig.json.\n"
                "- Auto-imports: composables/ and components/ are auto-imported — don't manually import.\n"
                "- Use useFetch/useAsyncData for data fetching (not raw fetch).\n"
                "- Use definePageMeta for route metadata, not head() or meta tags.\n"
                "- Guard browser APIs: use process.client or <ClientOnly> wrapper.\n"
                "- Animations: use Vue's <Transition> and <TransitionGroup> — not CSS-only."
            ),
            "svelte": (
                "Use Svelte 5 runes syntax ($state, $derived, $effect). Do NOT suggest React patterns.\n"
                "- REQUIRED configs: svelte.config.js, vite.config.ts.\n"
                "- Use +page.svelte, +layout.svelte, +error.svelte naming.\n"
                "- Animations: use svelte/transition and svelte/animate modules.\n"
                "- Data loading: use +page.server.ts load functions, not client-side fetch.\n"
                "- Guard browser APIs: use browser check from $app/environment."
            ),
            "vite": (
                "Use Vite's plugin system and HMR. Import assets with ?url suffix.\n"
                "- REQUIRED configs: vite.config.ts, tsconfig.json.\n"
                "- Use import.meta.env for env vars (not process.env).\n"
                "- CSS: use CSS modules or Tailwind — not global stylesheets with generic class names."
            ),
            "fastapi": (
                "Use Pydantic v2 models for all request/response validation.\n"
                "- REQUIRED files: requirements.txt (pinned versions), .env, main.py with uvicorn entry.\n"
                "- Use async def for ALL endpoints that do I/O (database, HTTP, file).\n"
                "- Use Annotated[type, Depends()] for dependency injection.\n"
                "- Return proper HTTP status codes: 201 create, 204 delete, 404 not found, 422 validation.\n"
                "- Use HTTPException with detail messages — not bare raise.\n"
                "- Add CORS middleware if frontend is separate origin.\n"
                "- Use Pydantic Settings for config, not raw os.environ.\n"
                "- COMPLETENESS: If main.py imports from routes/items.py, that file MUST exist. Create ALL referenced model/schema/route files.\n"
                "- DATABASE WIRING: Create database.py with engine/session setup. Models must connect to a real DB session, not just be defined."
            ),
            "express": (
                "Use async/await with express-async-errors or try/catch wrappers.\n"
                "- REQUIRED files: package.json with start/dev scripts, .env.\n"
                "- Centralized error middleware as LAST app.use().\n"
                "- Use helmet() for security headers, cors() for CORS.\n"
                "- Validate inputs with zod or joi — not manual checks.\n"
                "- Use router.route() for grouped CRUD endpoints.\n"
                "- COMPLETENESS: If server.ts imports from routes/items.ts, that file MUST exist. Create ALL referenced route/middleware files.\n"
                "- DATABASE WIRING: If using Prisma, create schema.prisma and db client. If using Sequelize, create models and migrations."
            ),
            "django": (
                "Use class-based views. Follow Django conventions.\n"
                "- REQUIRED files: requirements.txt, manage.py, settings.py with SECRET_KEY from env.\n"
                "- Use Django REST Framework for APIs — serializers, viewsets, routers.\n"
                "- Use model managers for complex queries — not raw filter chains in views.\n"
                "- Use django.conf.settings for config — not os.environ in views.\n"
                "- Run migrations: always create migration files for model changes."
            ),
            "flask": (
                "Use application factory pattern (create_app function).\n"
                "- REQUIRED files: requirements.txt, .env.\n"
                "- Use Flask-SQLAlchemy for database, Flask-Migrate for migrations.\n"
                "- Use blueprints for route organization.\n"
                "- Error handlers: register 404/500 handlers with @app.errorhandler."
            ),
        }
        if self.framework in directives:
            lines.append(f"- **DIRECTIVE**: {directives[self.framework]}")
        return "\n".join(lines)


# --- ProjectProfile cache ---
_cached_profile = None
_cached_profile_fingerprint = None


def _get_cached_profile():
    """Get or create a cached ProjectProfile, re-scanning if config files changed."""
    global _cached_profile, _cached_profile_fingerprint
    if _cached_profile is not None:
        # Check if stale
        if not _cached_profile.is_stale(_cached_profile_fingerprint):
            return _cached_profile
    _cached_profile = ProjectProfile()
    _cached_profile_fingerprint = _cached_profile._fingerprint
    return _cached_profile


# --- Intent extraction ---

_BUILD_VERBS = re.compile(r'\b(build|create|scaffold|set\s*up|make|generate|start)\b', re.I)

_STACK_KEYWORDS = {
    "nextjs": ("frontend", "nextjs"), "next.js": ("frontend", "nextjs"), "react": ("frontend", "react"),
    "vue": ("frontend", "vue"), "svelte": ("frontend", "svelte"), "angular": ("frontend", "angular"),
    "fastapi": ("backend", "fastapi"), "express": ("backend", "express"), "django": ("backend", "django"),
    "flask": ("backend", "flask"), "rails": ("backend", "rails"),
    "supabase": ("database", "supabase"), "postgres": ("database", "postgres"),
    "mongodb": ("database", "mongodb"), "sqlite": ("database", "sqlite"),
    "prisma": ("database", "prisma"), "drizzle": ("database", "drizzle"),
    "tailwind": ("styling", "tailwind"), "typescript": ("language", "typescript"),
    "python": ("language", "python"), "javascript": ("language", "javascript"),
}

_FEATURE_KEYWORDS = {
    "auth": "auth", "authentication": "auth", "login": "auth", "signup": "auth",
    "payment": "payments", "stripe": "payments", "checkout": "payments",
    "dashboard": "dashboard", "admin": "admin-panel", "crud": "crud",
    "api": "api", "rest": "api", "graphql": "graphql",
    "chat": "realtime-chat", "realtime": "realtime", "websocket": "realtime",
    "upload": "file-upload", "search": "search", "notification": "notifications",
}

_PROJECT_TYPE_KEYWORDS = {
    "website": "web-app", "web app": "web-app", "webapp": "web-app", "web application": "web-app",
    "api": "api", "rest api": "api", "backend": "api",
    "cli": "cli-tool", "command line": "cli-tool", "terminal": "cli-tool",
    "static site": "static-site", "landing page": "static-site", "portfolio": "static-site",
    "app": "web-app", "platform": "web-app", "tool": "web-app",
}


def _extract_project_spec(user_text):
    """Deterministic keyword extractor — no LLM call.
    Returns ProjectSpec or None if not a scaffolding request."""
    if not _BUILD_VERBS.search(user_text):
        return None

    spec = ProjectSpec()
    spec.description = user_text
    text_lower = user_text.lower()

    # Detect project type
    for kw, ptype in _PROJECT_TYPE_KEYWORDS.items():
        if kw in text_lower:
            spec.project_type = ptype
            spec.signals_explicit["project_type"] = ptype
            break

    # Detect stack keywords
    for kw, (field, val) in _STACK_KEYWORDS.items():
        if kw in text_lower:
            setattr(spec, field, val)
            spec.signals_explicit[field] = val

    # Detect features
    for kw, feature in _FEATURE_KEYWORDS.items():
        if kw in text_lower and feature not in spec.features:
            spec.features.append(feature)
            spec.signals_explicit.setdefault("features", []).append(feature)

    # Infer project type from stack if not explicit
    if not spec.project_type:
        if spec.frontend:
            spec.project_type = "web-app"
            spec.signals_inferred["project_type"] = "web-app (inferred from frontend)"
        elif spec.backend:
            spec.project_type = "api"
            spec.signals_inferred["project_type"] = "api (inferred from backend)"
        elif spec.language == "python":
            spec.project_type = "cli-tool"
            spec.signals_inferred["project_type"] = "cli-tool (inferred from python)"

    # Infer language from stack
    if not spec.language:
        if spec.frontend in ("react", "nextjs", "vue", "svelte", "angular"):
            spec.language = "typescript"
            spec.signals_inferred["language"] = "typescript (inferred from frontend)"
        elif spec.backend in ("fastapi", "django", "flask"):
            spec.language = "python"
            spec.signals_inferred["language"] = "python (inferred from backend)"
        elif spec.backend in ("express",):
            spec.language = "javascript"
            spec.signals_inferred["language"] = "javascript (inferred from backend)"

    # Infer backend from frontend for full-stack hints
    if spec.frontend == "nextjs" and not spec.backend:
        spec.backend = "nextjs"
        spec.signals_inferred["backend"] = "nextjs (full-stack framework)"

    # Determine what's still missing
    if not spec.project_type:
        spec.signals_missing.append("project_type")
    if not spec.frontend and not spec.backend:
        spec.signals_missing.append("stack")
    if not spec.language:
        spec.signals_missing.append("language")

    # Complexity heuristic
    if len(spec.features) >= 4:
        spec.complexity = "complex"
    elif len(spec.features) <= 1:
        spec.complexity = "simple"

    return spec


def _resolve_missing_signals(spec, profile):
    """Decision-tree question engine. Context-aware skipping.
    Uses tool_ask_user() for interactive questions. Reduces typical 4-7 interactions to 0-2."""

    # Skip if confidence is already high
    if spec.confidence_score() >= 0.75:
        return spec

    # Check existing project profile for context
    if profile and profile.base_info["type"] != "unknown":
        if not spec.language and profile.base_info.get("has_typescript"):
            spec.language = "typescript"
            spec.signals_inferred["language"] = "typescript (from existing project)"
        if not spec.frontend and profile.framework:
            fw = profile.framework
            if fw in ("nextjs", "react", "vue", "svelte", "angular"):
                spec.frontend = fw
                spec.signals_inferred["frontend"] = f"{fw} (from existing project)"
        if not spec.styling and profile.styling:
            spec.styling = profile.styling
            spec.signals_inferred["styling"] = f"{profile.styling} (from existing project)"

    # Check PLAN.md for additional context
    plan_path = Path(CWD) / "PLAN.md"
    if plan_path.exists():
        try:
            plan_text = plan_path.read_text(encoding="utf-8", errors="replace")[:3000].lower()
            for kw, (field, val) in _STACK_KEYWORDS.items():
                if kw in plan_text and getattr(spec, field) is None:
                    setattr(spec, field, val)
                    spec.signals_inferred[field] = f"{val} (from PLAN.md)"
        except Exception:
            pass

    # Re-check confidence after context enrichment
    if spec.confidence_score() >= 0.75:
        return spec

    # Level 1: Project type unknown — ask
    if not spec.project_type:
        choices = ["web-app", "api", "cli-tool", "static-site"]
        answer = tool_ask_user({
            "question": "What type of project is this?",
            "choices": choices,
            "default": "web-app"
        })
        answer_text = answer.replace("User answered: ", "").strip().lower()
        for choice in choices:
            if choice in answer_text:
                spec.project_type = choice
                break
        if not spec.project_type:
            spec.project_type = "web-app"

    # Level 2: Stack unknown — batch question
    need_stack = []
    if not spec.frontend and spec.project_type in ("web-app",):
        need_stack.append("frontend (react/vue/svelte/none)")
    if not spec.backend:
        need_stack.append("backend (nextjs/express/fastapi/django/none)")
    if not spec.language:
        need_stack.append("language (typescript/javascript/python)")

    if need_stack:
        stack_q = "What tech stack? Specify: " + ", ".join(need_stack)
        answer = tool_ask_user({"question": stack_q})
        answer_text = answer.replace("User answered: ", "").strip().lower()
        for kw, (field, val) in _STACK_KEYWORDS.items():
            if kw in answer_text and getattr(spec, field) is None:
                setattr(spec, field, val)

    # Level 3: Features — only ask if none detected and non-trivial project
    if not spec.features and spec.complexity != "simple":
        feature_choices = ["auth", "payments", "dashboard", "crud", "search", "file-upload", "notifications"]
        answer = tool_ask_user({
            "question": "Which features do you need? (comma-separated or pick from list)",
            "choices": feature_choices,
        })
        answer_text = answer.replace("User answered: ", "").strip().lower()
        for kw, feature in _FEATURE_KEYWORDS.items():
            if kw in answer_text and feature not in spec.features:
                spec.features.append(feature)

    return spec


# --- Category ordering for file generation ---
_CATEGORY_ORDER = {
    "config": 0,
    "schema": 1,
    "lib": 2,
    "middleware": 3,
    "routes": 4,
    "components": 5,
    "pages": 6,
    "tests": 7,
}


def _generate_file_manifest(spec, profile):
    """Generate ordered file tree with diff-against-existing.
    Returns list of {path, description, category, order, action}."""
    manifest = []
    existing = set(profile.existing_files) if profile else set()

    def _add(path, desc, category):
        action = "SKIP" if path in existing else "CREATE"
        # Config files that exist might need merging
        if action == "SKIP" and category == "config":
            action = "MERGE"
        manifest.append({
            "path": path,
            "description": desc,
            "category": category,
            "order": _CATEGORY_ORDER.get(category, 99),
            "action": action,
        })

    lang = spec.language or "typescript"
    ext = ".ts" if lang == "typescript" else ".js" if lang == "javascript" else ".py"
    jsx_ext = ".tsx" if lang == "typescript" else ".jsx"

    # --- Stack-specific blueprints ---

    if spec.backend == "nextjs" or spec.frontend == "nextjs":
        # Next.js project
        _add("package.json", "Project dependencies", "config")
        _add("next.config.mjs", "Next.js configuration", "config")
        _add("tsconfig.json", "TypeScript configuration", "config")
        _add(".env.local", "Environment variables", "config")
        _add(".env.example", "Environment variable documentation", "config")
        if spec.styling == "tailwind" or not spec.styling:
            _add("tailwind.config.ts", "Tailwind CSS configuration", "config")
            _add("postcss.config.js", "PostCSS configuration for Tailwind", "config")
            _add("app/globals.css", "Global styles with Tailwind", "config")
        _add(f"app/layout{jsx_ext}", "Root layout with providers", "config")
        _add(f"app/page{jsx_ext}", "Home page", "pages")
        _add(f"app/error{jsx_ext}", "Error boundary (client component)", "pages")
        _add(f"app/loading{jsx_ext}", "Loading state", "pages")

        if spec.database == "supabase":
            _add(f"lib/supabase{ext}", "Supabase client configuration", "lib")
            _add("supabase/migrations/001_initial.sql", "Initial database schema", "schema")

        if spec.database == "prisma":
            _add("prisma/schema.prisma", "Prisma database schema", "schema")
            _add(f"lib/prisma{ext}", "Prisma client singleton", "lib")

        if "auth" in spec.features:
            _add(f"lib/auth{ext}", "Authentication utilities", "lib")
            _add(f"app/login/page{jsx_ext}", "Login page", "pages")
            _add(f"app/signup/page{jsx_ext}", "Signup page", "pages")
            _add(f"middleware{ext}", "Auth middleware", "middleware")

        if "payments" in spec.features:
            _add(f"app/api/checkout/route{ext}", "Stripe checkout API", "routes")
            _add(f"app/api/webhooks/stripe/route{ext}", "Stripe webhook handler", "routes")
            _add(f"lib/stripe{ext}", "Stripe client configuration", "lib")

        if "dashboard" in spec.features:
            _add(f"app/dashboard/page{jsx_ext}", "Dashboard page", "pages")
            _add(f"app/dashboard/layout{jsx_ext}", "Dashboard layout with sidebar", "pages")

        if "crud" in spec.features:
            _add(f"app/api/items/route{ext}", "CRUD API endpoints", "routes")
            _add(f"components/ItemList{jsx_ext}", "Item list component", "components")
            _add(f"components/ItemForm{jsx_ext}", "Item form component", "components")

        # Common components
        _add(f"components/Header{jsx_ext}", "Header/navigation component", "components")
        _add(f"components/Footer{jsx_ext}", "Footer component", "components")

    elif spec.backend == "fastapi":
        # FastAPI project
        _add("requirements.txt", "Python dependencies", "config")
        _add("main.py", "FastAPI application entry point", "config")
        _add(".env", "Environment variables", "config")
        _add(".env.example", "Environment variable documentation", "config")
        _add("models/__init__.py", "Models package", "schema")
        _add("models/base.py", "SQLAlchemy base model", "schema")
        _add("schemas/__init__.py", "Pydantic schemas package", "schema")
        _add("routes/__init__.py", "Routes package", "routes")

        if "auth" in spec.features:
            _add("routes/auth.py", "Authentication endpoints", "routes")
            _add("models/user.py", "User model", "schema")
            _add("schemas/user.py", "User schemas", "schema")
            _add("middleware/auth.py", "Auth middleware", "middleware")

        if "crud" in spec.features:
            _add("routes/items.py", "CRUD endpoints", "routes")
            _add("models/item.py", "Item model", "schema")
            _add("schemas/item.py", "Item schemas", "schema")

    elif spec.backend == "express":
        # Express project
        _add("package.json", "Project dependencies", "config")
        _add(f"server{ext}", "Express server entry point", "config")
        _add(".env", "Environment variables", "config")
        _add(".env.example", "Environment variable documentation", "config")
        _add(f"routes/index{ext}", "Route definitions", "routes")
        _add(f"middleware/errorHandler{ext}", "Error handling middleware", "middleware")

        if "auth" in spec.features:
            _add(f"routes/auth{ext}", "Auth routes", "routes")
            _add(f"middleware/auth{ext}", "Auth middleware", "middleware")

    elif spec.backend == "django":
        # Django project
        _add("requirements.txt", "Python dependencies", "config")
        _add("manage.py", "Django management script", "config")
        _add("config/settings.py", "Django settings", "config")
        _add("config/urls.py", "URL configuration", "routes")

    elif spec.project_type == "static-site":
        # Static site
        _add("index.html", "Main HTML page", "pages")
        _add("styles.css", "Stylesheet", "config")
        _add("script.js", "JavaScript", "lib")

    else:
        # Generic/unknown — minimal scaffold
        if lang == "python":
            _add("main.py", "Application entry point", "config")
            _add("requirements.txt", "Python dependencies", "config")
        else:
            _add("package.json", "Project dependencies", "config")
            _add(f"index{ext}", "Application entry point", "config")

    # Sort by category order
    manifest.sort(key=lambda x: x["order"])
    return manifest


def tool_scaffold_project(args):
    """Analyze a project description and generate a structured file manifest."""
    desc = args.get("description", "")
    spec = _extract_project_spec(desc)
    if not spec:
        return "Could not parse project description. Please describe what you want to build (e.g., 'build a todo app with Next.js')."

    # Apply optional overrides
    if args.get("stack"):
        for kw in args["stack"].split(","):
            kw = kw.strip().lower()
            if kw in _STACK_KEYWORDS:
                field, val = _STACK_KEYWORDS[kw]
                setattr(spec, field, val)
    if args.get("features"):
        for f in args["features"].split(","):
            f = f.strip().lower()
            if f in _FEATURE_KEYWORDS:
                feat = _FEATURE_KEYWORDS[f]
                if feat not in spec.features:
                    spec.features.append(feat)

    # Get project profile for context-aware generation
    profile = _get_cached_profile()
    spec = _resolve_missing_signals(spec, profile)
    manifest = _generate_file_manifest(spec, profile)
    spec.file_manifest = manifest

    # Format for display
    lines = [f"## Scaffold Plan (confidence: {spec.confidence_score():.0%})", ""]
    lines.append(f"**Type**: {spec.project_type or 'unknown'}")
    if spec.frontend:
        lines.append(f"**Frontend**: {spec.frontend}")
    if spec.backend:
        lines.append(f"**Backend**: {spec.backend}")
    if spec.language:
        lines.append(f"**Language**: {spec.language}")
    if spec.database:
        lines.append(f"**Database**: {spec.database}")
    if spec.features:
        lines.append(f"**Features**: {', '.join(spec.features)}")
    lines.append("")
    lines.append("### Files to generate:")
    for entry in manifest:
        action_icon = {"CREATE": "+", "SKIP": "~", "MERGE": "M"}.get(entry["action"], "?")
        lines.append(f"  [{action_icon}] {entry['path']} — {entry['description']}")
    lines.append("")
    lines.append("Confirm with the user before creating these files.")
    return "\n".join(lines)


# Register scaffold_project tool handler (defined after TOOL_HANDLERS dict)
TOOL_HANDLERS["scaffold_project"] = tool_scaffold_project


def execute_tool(name, args):
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Error: unknown tool '{name}'"
    try:
        # --- UNDO SNAPSHOT ---
        _snapshot_for_undo(name, args)

        # --- OVERWRITE GUARD: block write_file when edit_file should be used ---
        if name == "write_file":
            filepath = args.get("file_path", "")
            content = args.get("content", "")
            if filepath and content:
                try:
                    _target = _resolve(filepath)
                    if _target.exists():
                        _existing = _target.read_text(encoding="utf-8")
                        if _existing and len(_existing) > 50:
                            # Compute how much actually changed
                            import difflib
                            _ratio = difflib.SequenceMatcher(None, _existing, content).ratio()
                            if _ratio > 0.85:
                                # >85% identical — this is a small edit disguised as a full rewrite
                                _changed_pct = round((1 - _ratio) * 100, 1)
                                print(f"  {C.WARNING}{BLACK_CIRCLE} Overwrite guard: only {_changed_pct}% changed — use edit_file instead{C.RESET}")
                                return (
                                    f"BLOCKED: The file '{filepath}' already exists and your new content is {_changed_pct}% different "
                                    f"(i.e. {round(_ratio * 100, 1)}% identical). You are rewriting the entire file to make a tiny change. "
                                    f"Use edit_file instead with old_string (the exact lines to replace) and new_string (the replacement). "
                                    f"This keeps the rest of the file untouched and avoids destroying code you weren't asked to modify."
                                )
                except Exception:
                    pass  # resolve/read failure — let write_file handle it

        # --- READ-BEFORE-WRITE GUARD: block edits to unread files ---
        if READ_GUARD_ENABLED and name in ("edit_file", "write_file") \
           and not getattr(_thread_local, 'is_subagent', False):
            filepath = args.get("file_path", "")
            if filepath:
                try:
                    _target = _resolve(filepath)
                    _target_key = str(_target.resolve())
                    # Only enforce for EXISTING files — new file creation passes through
                    if _target.exists():
                        with _read_guard_lock:
                            was_read = _target_key in _files_read_this_turn
                            was_searched = _target_key in _files_searched_this_turn
                        if not was_read:
                            _guard_stats["read_guard_blocks"] += 1
                            print(f"  {C.WARNING}{BLACK_CIRCLE} Read guard: {filepath} not read yet{C.RESET}")
                            if was_searched:
                                return (
                                    f"BLOCKED: You found '{filepath}' via search but haven't read it. "
                                    f"Call read_file(file_path='{filepath}') to see the full context, "
                                    f"then use edit_file with the exact text to replace."
                                )
                            else:
                                return (
                                    f"BLOCKED: You haven't read '{filepath}' this turn. "
                                    f"Call read_file(file_path='{filepath}') first to see current content, "
                                    f"then use {name} with the exact text. "
                                    f"This prevents edits based on stale or assumed content."
                                )
                except (ValueError, OSError):
                    pass

        # --- QUALITY GATE: score code before writing ---
        if name == "write_file" and QUALITY_GATE_ENABLED:
            content = args.get("content", "")
            filepath = args.get("file_path", "")
            if content and filepath:
                qscore, qissues = _slop_score(content, filepath)
                gate_result = _apply_quality_gate(filepath, qscore, qissues)
                if gate_result is not None:
                    print(f"  {C.WARNING}{BLACK_CIRCLE} Quality gate: {qscore}/100{C.RESET}")
                    return gate_result  # BLOCKED — return feedback, don't write

        result = handler(args)
        if len(result) > TOOL_OUTPUT_LIMIT:
            result = result[:TOOL_OUTPUT_LIMIT] + f"\n... [truncated, {len(result)} total chars]"

        # --- VERIFICATION GATE ---
        # Auto-verify file writes
        if name in ("write_file", "edit_file"):
            result = _verify_file_write(name, args, result)

        # --- AUTO-SYMBOL SEARCH: show related references after edit ---
        if AUTO_SYMBOL_SEARCH and name == "edit_file" and not result.startswith("Error"):
            _sym_refs = _find_related_references(args)
            if _sym_refs:
                _guard_stats["auto_symbol_searches"] += 1
                result += f"\n\n\U0001f4ce Related references (may need updating):\n{_sym_refs}"

        # --- GRAPH CONTEXT: inject neighborhood after edit/read (Phase 2) ---
        if name in ("edit_file", "read_file") and _project_graph is not None:
            try:
                edited_file = args.get("file_path", "")
                if edited_file:
                    _ef_resolved = str(_resolve(edited_file).resolve())
                    _ef_rel = str(Path(_ef_resolved).relative_to(Path(CWD).resolve())).replace('\\', '/')
                    _nbr_ctx = _build_graph_context([_ef_rel], max_tokens=200)
                    if _nbr_ctx:
                        result += f"\n\n{_nbr_ctx}"
            except Exception:
                pass

        # --- EDIT HISTORY ---
        # Track file operations for retry context
        if name in ("write_file", "edit_file"):
            fp = args.get("file_path", "?")
            content_preview = args.get("content", "")[:80] if name == "write_file" else args.get("new_str", "")[:80]
            _edit_history.append((fp, name, content_preview))

        # Track in manifest
        _session_manifest.record_tool_result(name, args, result)

        return result
    except Exception as e:
        return f"Error executing {name}: {e}"

# ---------------------------------------------------------------------------
# ollama client
# ---------------------------------------------------------------------------

def ollama_chat(messages, model, tools=None, stream=True, num_ctx=8192):
    """Call the active LLM provider. Yields chunks if streaming."""
    provider = _get_provider()
    yield from provider.chat(messages, model, tools=tools, stream=stream, num_ctx=num_ctx)

# ---------------------------------------------------------------------------
# chunked file generation -- for weaker/slower models
# ---------------------------------------------------------------------------

# File-type-specific assembly rules
_ASSEMBLY_RULES = {
    ".html": {
        "wrap_open": "",
        "wrap_close": "",
        "joiner": "\n\n",
        "post_process": "_assemble_html",
        "drip_eligible": True,
    },
    ".py": {
        "wrap_open": "",
        "wrap_close": "",
        "joiner": "\n\n",
        "post_process": None,
        "drip_eligible": True,
    },
    ".js": {"wrap_open": "", "wrap_close": "", "joiner": "\n\n", "post_process": None, "drip_eligible": True},
    ".ts": {"wrap_open": "", "wrap_close": "", "joiner": "\n\n", "post_process": None, "drip_eligible": True},
    ".tsx": {"wrap_open": "", "wrap_close": "", "joiner": "\n\n", "post_process": None, "drip_eligible": True},
    ".jsx": {"wrap_open": "", "wrap_close": "", "joiner": "\n\n", "post_process": None, "drip_eligible": True},
    ".css": {"wrap_open": "", "wrap_close": "", "joiner": "\n\n", "post_process": None, "drip_eligible": True},
}


def _decompose_objective(description, filepath, model):
    """
    Use the model to decompose an objective into small, ordered chunks.
    Returns list of (chunk_name, chunk_prompt) tuples.
    """
    ext = Path(filepath).suffix.lower()
    filename = Path(filepath).name

    decompose_prompt = (
        f"I need to generate the file '{filename}' ({ext} file).\n"
        f"Objective: {description}\n\n"
        f"Break this into 4-7 small, ordered sections that can each be generated independently.\n"
        f"Each section should produce under 200 lines of code.\n\n"
        f"Rules:\n"
        f"- Be SPECIFIC about what each section contains (exact functions, components, or HTML blocks)\n"
        f"- Sections must be in the correct order for assembly (imports first, then logic, etc.)\n"
        f"- The first section should include ALL boilerplate/setup (imports, config, head tags, etc.)\n"
        f"- The last section should close everything properly\n"
        f"- Each section name should be short (2-3 words)\n\n"
        f"Output ONLY a numbered list in this exact format (no other text):\n"
        f"1. section_name | Description of exactly what code this section contains\n"
        f"2. section_name | Description of exactly what code this section contains\n"
    )

    plan_messages = [
        {"role": "system", "content": "You are a code architect. You decompose file generation tasks into small chunks. Output ONLY the numbered list, nothing else."},
        {"role": "user", "content": decompose_prompt},
    ]

    print(f"  {C.CLAW}{BLACK_CIRCLE} Analyzing objective to plan chunks...{C.RESET}")
    spinner = Spinner("Planning", C.CLAW)
    spinner.start()

    full_text = ""
    try:
        for chunk in ollama_chat(plan_messages, model, tools=None, stream=True):
            content = chunk.get("message", {}).get("content", "")
            if content:
                full_text += content
        spinner.stop()
    except Exception as e:
        spinner.stop()
        print(f"  {C.ERROR}{BLACK_CIRCLE} Planning failed: {e}{C.RESET}")
        return _fallback_decompose(description, filepath)

    # Parse the numbered list
    chunks = []
    for line in full_text.strip().split("\n"):
        line = line.strip()
        # Match: "1. name | description" or "1) name | description" or "1. name: description"
        m = re.match(r'^\d+[.)]\s*(.+?)(?:\s*[|:]\s*)(.+)$', line)
        if m:
            name = m.group(1).strip().lower().replace(" ", "_")[:30]
            desc = m.group(2).strip()
            chunks.append((name, desc))

    if len(chunks) < 2:
        print(f"  {C.WARNING}{BLACK_CIRCLE} Model returned {len(chunks)} chunks, using fallback{C.RESET}")
        return _fallback_decompose(description, filepath)

    # Cap at 8 chunks
    chunks = chunks[:8]

    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Decomposed into {len(chunks)} chunks:{C.RESET}")
    for i, (name, desc) in enumerate(chunks, 1):
        print(f"    {C.SUBTLE}{i}. {name}: {desc[:70]}{C.RESET}")

    return chunks


def _fallback_decompose(description, filepath):
    """
    Fallback decomposition when model planning fails.
    Uses file extension and objective keywords to create sensible chunks.
    """
    ext = Path(filepath).suffix.lower()
    desc_lower = description.lower()

    if ext in (".html", ".htm"):
        chunks = [("setup_and_head", "DOCTYPE, html, head tag with meta, title, CSS/Tailwind CDN, style block, and opening body tag")]
        # Scan objective for features
        if any(w in desc_lower for w in ("nav", "header", "menu", "logo")):
            chunks.append(("navigation", "Navigation bar with logo, links, and mobile menu"))
        chunks.append(("hero_section", "Hero section with headline, subtext, CTA buttons, and main visual"))
        if any(w in desc_lower for w in ("feature", "benefit", "why", "service", "card")):
            chunks.append(("features", "Features or benefits section with cards, icons, and descriptions"))
        if any(w in desc_lower for w in ("form", "convert", "calculator", "search", "input")):
            chunks.append(("interactive", "Interactive section: forms, calculators, or search functionality"))
        if any(w in desc_lower for w in ("testimonial", "review", "trust", "social", "proof", "stat")):
            chunks.append(("social_proof", "Testimonials, stats, trust badges, or social proof section"))
        if any(w in desc_lower for w in ("price", "pricing", "plan", "tier", "cost")):
            chunks.append(("pricing", "Pricing table or plans section"))
        if any(w in desc_lower for w in ("faq", "question", "answer")):
            chunks.append(("faq", "FAQ or questions section"))
        chunks.append(("cta_and_footer", "Final CTA section and footer with links, legal text, closing body/html tags"))
        chunks.append(("javascript", "All JavaScript: interactivity, dark mode, animations, form handling, dynamic content"))

    elif ext == ".py":
        chunks = [
            ("imports_and_config", "All imports, constants, configuration, and global variables"),
            ("models_and_types", "Data models, type definitions, classes, and schemas"),
            ("core_logic", "Core business logic functions and algorithms"),
            ("api_or_routes", "API endpoints, route handlers, or CLI entry points"),
            ("helpers_and_main", "Helper utilities and main entry point / if __name__ block"),
        ]

    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        chunks = [
            ("imports_and_types", "Import statements, type definitions, interfaces, and constants"),
            ("state_and_hooks", "State management, custom hooks, context, and data fetching"),
            ("components", "Main component definitions with props and rendering logic"),
            ("event_handlers", "Event handlers, form logic, API calls, and side effects"),
            ("exports_and_styles", "Component exports, styled components, or CSS-in-JS"),
        ]

    elif ext == ".css":
        chunks = [
            ("variables_and_reset", "CSS custom properties, reset styles, and base typography"),
            ("layout_and_components", "Layout rules, grid, flexbox, and component styles"),
            ("states_and_responsive", "Hover/focus/active states, animations, and media queries"),
        ]

    else:
        # Generic fallback: just split into 3 parts
        chunks = [
            ("part_1_setup", "Setup, imports, configuration, and initial boilerplate"),
            ("part_2_core", "Core logic, main content, and primary functionality"),
            ("part_3_finish", "Final sections, cleanup, exports, and closing code"),
        ]

    # Filter down if too many chunks for a simple objective
    if len(description.split()) < 15 and len(chunks) > 5:
        chunks = chunks[:5]

    return chunks


# ---------------------------------------------------------------------------
# Multi-file task decomposition (Phase 5)
# ---------------------------------------------------------------------------

def _should_multi_decompose(model):
    """Only do multi-file decomposition for 32K+ context models."""
    try:
        return _get_model_context_size(model) >= 32768
    except Exception:
        return False


def _decompose_multi_file(objective, model):
    """Use graph + model to decompose multi-file tasks into ordered sub-tasks.
    Returns list of {files, description, dependencies, context} dicts, or None on failure.
    """
    if not _should_multi_decompose(model):
        return None

    global _project_graph
    if _project_graph is None:
        try:
            _get_project_graph()
        except Exception:
            return None
    if _project_graph is None:
        return None

    # Build graph summary
    graph_summary = _build_graph_context([], max_tokens=1000)
    if not graph_summary:
        return None

    decompose_prompt = (
        f"I need to implement the following change across multiple files:\n\n"
        f"Objective: {objective}\n\n"
        f"Here is the project's dependency graph:\n{graph_summary}\n\n"
        f"Break this into 3-8 ordered sub-tasks. For each sub-task, specify:\n"
        f"- Which files need to be created or modified\n"
        f"- What changes are needed\n"
        f"- Which other sub-tasks it depends on\n\n"
        f"Output ONLY a numbered list in this exact format:\n"
        f"1. description | Files: file1.ts, file2.ts | Depends: none\n"
        f"2. description | Files: file3.ts | Depends: 1\n"
    )

    plan_messages = [
        {"role": "system", "content": "You are a code architect. Decompose multi-file changes into ordered sub-tasks. Output ONLY the numbered list."},
        {"role": "user", "content": decompose_prompt},
    ]

    full_text = ""
    try:
        for chunk in ollama_chat(plan_messages, model, tools=None, stream=True):
            content = chunk.get("message", {}).get("content", "")
            if content:
                full_text += content
    except Exception:
        return None

    # Parse response
    tasks = []
    for line in full_text.strip().split("\n"):
        line = line.strip()
        m = re.match(r'^\d+[.)]\s*(.+?)(?:\s*\|\s*Files:\s*(.+?))?(?:\s*\|\s*Depends:\s*(.+?))?$', line, re.IGNORECASE)
        if m:
            desc = m.group(1).strip()
            files_str = m.group(2) or ""
            deps_str = m.group(3) or "none"
            files = [f.strip() for f in files_str.split(',') if f.strip()]
            deps = []
            if deps_str.strip().lower() != 'none':
                for d in re.findall(r'\d+', deps_str):
                    deps.append(int(d))

            # Build context for these files from graph
            ctx = ""
            if files and _project_graph is not None:
                ctx = _build_graph_context(files, max_tokens=200)

            tasks.append({
                'files': files,
                'description': desc,
                'dependencies': deps,
                'context': ctx,
            })

    if len(tasks) < 2:
        return None

    return tasks[:8]


def _extract_plan_step_files(plan_content, step_text):
    """Extract Files: annotation from a PLAN.md step. Returns list of file paths."""
    # Look for "Files: file1, file2" in the step text
    m = re.search(r'Files?:\s*(.+?)(?:\n|$)', step_text)
    if m:
        return [f.strip() for f in m.group(1).split(',') if f.strip()]
    return []


def _assemble_html(sections):
    """Post-process assembled HTML to fix common issues."""
    assembled = "\n\n".join(sections.values())

    # --- FIX 7: Dedup duplicate <nav> tags (keep first) ---
    nav_count = len(re.findall(r'<nav\b', assembled))
    if nav_count > 1:
        first_nav_end = assembled.find('</nav>')
        if first_nav_end > 0:
            first_nav_end += len('</nav>')
            rest = assembled[first_nav_end:]
            rest = re.sub(r'<nav\b[^>]*>[\s\S]*?</nav>', '', rest)
            assembled = assembled[:first_nav_end] + rest

    # --- FIX 3: Dedup duplicate section IDs (keep first occurrence) ---
    seen_ids = set()
    for m in list(re.finditer(r'<section[^>]*id=["\']([^"\']+)["\'][^>]*>[\s\S]*?</section>', assembled)):
        sid = m.group(1)
        if sid in seen_ids:
            assembled = assembled.replace(m.group(0), '', 1)
        seen_ids.add(sid)

    # Ensure proper HTML structure
    # If missing DOCTYPE, add it
    if "<!DOCTYPE" not in assembled and "<!doctype" not in assembled:
        assembled = "<!DOCTYPE html>\n<html lang=\"en\">\n" + assembled
    # If missing closing tags
    if "</body>" not in assembled:
        assembled += "\n</body>"
    if "</html>" not in assembled:
        assembled += "\n</html>"

    # Move any <script> blocks that ended up outside </body> to before </body>
    # Find scripts after </body>
    body_close = assembled.rfind("</body>")
    if body_close > 0:
        after_body = assembled[body_close + len("</body>"):]
        script_blocks = re.findall(r'<script[\s\S]*?</script>', after_body)
        if script_blocks:
            for block in script_blocks:
                after_body = after_body.replace(block, "")
                assembled = assembled[:body_close] + "\n" + block + "\n" + assembled[body_close:]
            # Clean up
            body_close_new = assembled.rfind("</body>")
            assembled = assembled[:body_close_new + len("</body>")] + "\n</html>"

    # --- Post-validate HTML (placeholders, wiring, dark mode, etc.) ---
    assembled, _fixes = _post_validate_html(assembled)

    # Lead-to-Gold: enhance assembled HTML
    assembled = _enhance_html(assembled)

    return assembled


# ---------------------------------------------------------------------------
# drip generation -- micro-chunked generation for local models
# ---------------------------------------------------------------------------

class DripContext:
    """Shared memory between drips — accumulates as drips complete."""

    def __init__(self):
        self.css_classes_defined = []
        self.js_functions = []
        self.html_ids = []
        self.variables = []
        self.sections_completed = []
        self.color_scheme = ""
        self.last_3_lines = ""
        self.accumulated_code = ""
        # Python-specific
        self.py_functions = []   # "def foo(a, b)"
        self.py_classes = []     # "class Bar"
        self.py_imports = []     # "import os" / "from x import y"
        self.py_decorators = []  # "@app.route", "@dataclass"
        # Generic symbols (used for all lang types)
        self.exported_names = [] # names exported / defined at module level

    def ingest(self, drip_name, output):
        """Parse completed drip output and update shared state."""
        self.sections_completed.append(drip_name)
        self.accumulated_code += output + "\n"
        # extract last 3 lines for continuation
        lines = output.strip().split("\n")
        self.last_3_lines = "\n".join(lines[-3:]) if len(lines) >= 3 else output.strip()

        # --- HTML / CSS ---
        # extract HTML ids
        for m in re.finditer(r'id=["\']([^"\']+)["\']', output):
            if m.group(1) not in self.html_ids:
                self.html_ids.append(m.group(1))
        # extract CSS classes from class= attributes
        for m in re.finditer(r'class=["\']([^"\']+)["\']', output):
            for cls in m.group(1).split():
                if cls not in self.css_classes_defined and not cls.startswith(("sm:", "md:", "lg:", "xl:", "hover:", "dark:", "focus:")):
                    self.css_classes_defined.append(cls)
        # detect color scheme from tailwind classes
        if not self.color_scheme:
            for color in ("emerald", "green", "blue", "indigo", "purple", "violet",
                          "rose", "red", "orange", "amber", "teal", "cyan", "sky"):
                if color in output:
                    self.color_scheme = color
                    break

        # --- JavaScript / TypeScript ---
        # extract JS function names (declarations + inline handlers + arrow fns)
        for m in re.finditer(r'function\s+(\w+)\s*\(', output):
            sig = m.group(1) + "()"
            if sig not in self.js_functions:
                self.js_functions.append(sig)
        # catch onclick="funcName()" and similar inline handlers
        for m in re.finditer(r'on\w+=["\'](\w+)\s*\(', output):
            sig = m.group(1) + "()"
            if sig not in self.js_functions:
                self.js_functions.append(sig)
        # catch const funcName = (...) => or const funcName = function
        for m in re.finditer(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|function)\s*(?:=>|\()', output):
            sig = m.group(1) + "()"
            if sig not in self.js_functions:
                self.js_functions.append(sig)
        # extract const/let/var names
        for m in re.finditer(r'(?:const|let|var)\s+(\w+)\s*=', output):
            if m.group(1) not in self.variables:
                self.variables.append(m.group(1))
        # catch export default / export { name }
        for m in re.finditer(r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)', output):
            if m.group(1) not in self.exported_names:
                self.exported_names.append(m.group(1))

        # --- Python ---
        # extract Python function definitions with signatures
        for m in re.finditer(r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', output, re.MULTILINE):
            sig = f"def {m.group(1)}({m.group(2).strip()[:60]})"
            if sig not in self.py_functions:
                self.py_functions.append(sig)
            if m.group(1) not in self.exported_names:
                self.exported_names.append(m.group(1))
        # extract Python class definitions
        for m in re.finditer(r'^class\s+(\w+)\s*(?:\([^)]*\))?:', output, re.MULTILINE):
            cls = f"class {m.group(1)}"
            if cls not in self.py_classes:
                self.py_classes.append(cls)
            if m.group(1) not in self.exported_names:
                self.exported_names.append(m.group(1))
        # extract Python imports
        for m in re.finditer(r'^((?:from\s+\S+\s+)?import\s+.+)$', output, re.MULTILINE):
            imp = m.group(1).strip()
            if imp not in self.py_imports and len(imp) < 120:
                self.py_imports.append(imp)
        # extract decorators
        for m in re.finditer(r'^(@\w[\w.]*(?:\([^)]*\))?)', output, re.MULTILINE):
            dec = m.group(1)[:60]
            if dec not in self.py_decorators:
                self.py_decorators.append(dec)

    def compress(self):
        """Return a compressed context string (~200 tokens) for the next drip."""
        parts = []
        if self.sections_completed:
            parts.append(f"Sections done: {', '.join(self.sections_completed)}")

        # Python context
        if self.py_imports:
            parts.append(f"Imports: {'; '.join(self.py_imports[:10])}")
        if self.py_classes:
            parts.append(f"Classes defined: {', '.join(self.py_classes[:10])}")
        if self.py_functions:
            parts.append(f"Functions defined: {', '.join(self.py_functions[:10])}")
        if self.py_decorators:
            parts.append(f"Decorators used: {', '.join(self.py_decorators[:8])}")

        # HTML context
        if self.html_ids:
            parts.append(f"HTML IDs: {', '.join(self.html_ids[:20])}")
        if self.css_classes_defined:
            important = [c for c in self.css_classes_defined if len(c) > 4][:15]
            if important:
                parts.append(f"Key CSS classes: {', '.join(important)}")
        if self.color_scheme:
            parts.append(f"Color scheme: {self.color_scheme}")

        # JS/TS context
        if self.js_functions:
            parts.append(f"JS functions: {', '.join(self.js_functions[:15])}")
        if self.variables:
            parts.append(f"Variables: {', '.join(self.variables[:10])}")
        if self.exported_names:
            parts.append(f"Exported/defined names: {', '.join(self.exported_names[:15])}")

        if self.last_3_lines:
            parts.append(f"Last lines of previous section:\n{self.last_3_lines}")
        return "\n".join(parts)


def _drip_decompose_python(description, filepath):
    """Decompose a Python file into micro-drips."""
    desc_lower = description.lower()
    drips = []

    # Group 1: Imports and config (must be first)
    drips.append({
        "name": "imports_config",
        "desc": "Generate all import statements, constants, configuration variables, and environment setup. Include typing imports if needed.",
        "group": 1,
        "depends_on_html": False,
    })

    # Group 2: Models/types (parallel)
    if any(w in desc_lower for w in ("model", "class", "schema", "pydantic", "dataclass", "type")):
        drips.append({
            "name": "models_types",
            "desc": "Generate data models, Pydantic schemas, dataclasses, TypedDicts, or class definitions with all fields and validators",
            "group": 2,
            "depends_on_html": False,
        })

    if any(w in desc_lower for w in ("database", "db", "sql", "orm", "migration", "table")):
        drips.append({
            "name": "database_layer",
            "desc": "Generate database connection setup, ORM models, table definitions, and database helper functions",
            "group": 2,
            "depends_on_html": False,
        })

    # Group 3: Core logic (parallel) — depends on models/types from group 2
    drips.append({
        "name": "core_logic",
        "desc": f"Generate the core business logic functions and algorithms for: {description[:200]}",
        "group": 3,
        "depends_on_previous": True,
    })

    if any(w in desc_lower for w in ("api", "route", "endpoint", "fastapi", "flask", "django", "rest", "server")):
        drips.append({
            "name": "api_routes",
            "desc": "Generate API route handlers/endpoints with request validation, response formatting, and error handling",
            "group": 3,
            "depends_on_previous": True,
        })

    if any(w in desc_lower for w in ("auth", "login", "jwt", "session", "permission", "middleware")):
        drips.append({
            "name": "auth_middleware",
            "desc": "Generate authentication/authorization middleware, JWT handling, session management, and permission checks",
            "group": 3,
            "depends_on_previous": True,
        })

    # Group 4: Helpers and entry point — depends on core logic
    drips.append({
        "name": "helpers_utils",
        "desc": "Generate helper/utility functions, validators, formatters, and any supporting code",
        "group": 4,
        "depends_on_previous": True,
    })

    drips.append({
        "name": "main_entry",
        "desc": "Generate the main entry point: if __name__ == '__main__' block, app startup, CLI argument parsing, or server launch code",
        "group": 5,
        "depends_on_previous": True,
    })

    return drips


def _drip_decompose_js(description, filepath):
    """Decompose a JS/TS/JSX/TSX file into micro-drips."""
    desc_lower = description.lower()
    ext = Path(filepath).suffix.lower()
    is_react = ext in (".jsx", ".tsx") or any(w in desc_lower for w in ("react", "component", "hook", "jsx", "tsx"))
    drips = []

    # Group 1: Imports and types
    drips.append({
        "name": "imports_types",
        "desc": f"Generate all import statements, type definitions, interfaces, constants, and configuration. {'Include React imports.' if is_react else ''}",
        "group": 1,
        "depends_on_html": False,
    })

    # Group 2: State/hooks or data layer (parallel)
    if is_react:
        drips.append({
            "name": "state_hooks",
            "desc": "Generate React state declarations (useState, useReducer), custom hooks, context providers, and data fetching logic (useEffect, useSWR, React Query)",
            "group": 2,
            "depends_on_html": False,
        })
    else:
        if any(w in desc_lower for w in ("api", "route", "express", "server", "endpoint", "rest")):
            drips.append({
                "name": "api_routes",
                "desc": "Generate API route handlers, middleware setup, request/response handling, and error middleware",
                "group": 2,
                "depends_on_html": False,
            })

    if any(w in desc_lower for w in ("database", "db", "prisma", "drizzle", "mongo", "sql")):
        drips.append({
            "name": "data_layer",
            "desc": "Generate database queries, ORM operations, data access functions, and connection setup",
            "group": 2,
            "depends_on_html": False,
        })

    # Group 3: Components or core logic (parallel) — depends on state/hooks/data from group 2
    if is_react:
        drips.append({
            "name": "component_render",
            "desc": f"Generate the main component function with JSX rendering, props handling, conditional rendering, and event handlers for: {description[:150]}",
            "group": 3,
            "depends_on_previous": True,
        })
    else:
        drips.append({
            "name": "core_logic",
            "desc": f"Generate the core functions, algorithms, and business logic for: {description[:150]}",
            "group": 3,
            "depends_on_previous": True,
        })

    if any(w in desc_lower for w in ("form", "validation", "submit", "input")):
        drips.append({
            "name": "form_handlers",
            "desc": "Generate form validation logic, submit handlers, input sanitization, and error display",
            "group": 3,
            "depends_on_previous": True,
        })

    # Group 4: Exports and styles — depends on all prior definitions
    drips.append({
        "name": "exports_final",
        "desc": "Generate default/named exports, styled components or CSS-in-JS if needed, and any module-level side effects",
        "group": 4,
        "depends_on_previous": True,
    })

    return drips


def _drip_decompose_css(description, filepath):
    """Decompose a CSS file into micro-drips."""
    desc_lower = description.lower()
    drips = []

    # Group 1: Variables and reset
    drips.append({
        "name": "variables_reset",
        "desc": "Generate :root CSS custom properties (colors, spacing, fonts, shadows, border-radius), *, *::before, *::after box-sizing reset, body base styles, and typography defaults (font-family, line-height, color)",
        "group": 1,
        "depends_on_html": False,
    })

    # Group 2: Layout and components (parallel)
    drips.append({
        "name": "layout_grid",
        "desc": "Generate layout rules: container, header, nav, main, sidebar, footer positioning. Flexbox and CSS Grid patterns for page structure",
        "group": 2,
        "depends_on_html": False,
    })

    drips.append({
        "name": "component_styles",
        "desc": f"Generate component-specific styles for: {description[:150]}. Include buttons, cards, forms, badges, modals, tooltips as needed",
        "group": 2,
        "depends_on_html": False,
    })

    # Group 3: States, animations, responsive — depends on variables and selectors from groups 1-2
    drips.append({
        "name": "states_animations",
        "desc": "Generate hover/focus/active/disabled states for all interactive elements, transition properties, @keyframes animations, and transform effects",
        "group": 3,
        "depends_on_previous": True,
    })

    drips.append({
        "name": "responsive_dark",
        "desc": "Generate @media queries for mobile/tablet/desktop breakpoints, dark mode (@media (prefers-color-scheme: dark) or .dark class), and print styles",
        "group": 3,
        "depends_on_previous": True,
    })

    return drips


def _extract_sections_from_description(description):
    """
    Parse the user's description, design.json, and PLAN.md to extract
    content sections. Returns a list of section topic strings.
    Replaces hardcoded keyword gates with description-driven decomposition.
    """
    sections = []

    # 1. Numbered items: "1) Slim Prompt" or "1. Slim Prompt"
    numbered = re.findall(r'\d+[.)]\s*(.+?)(?=\s*\d+[.)]|$)', description)
    if numbered:
        sections.extend([s.strip() for s in numbered if len(s.strip()) >= 3])

    # 2. design.json sections field
    design_path = Path(CWD) / "design.json"
    if design_path.is_file():
        try:
            design = json.loads(design_path.read_text(encoding="utf-8", errors="replace"))
            if "sections" in design and isinstance(design["sections"], list):
                for s in design["sections"]:
                    if isinstance(s, str) and s.lower() not in ("hero", "nav", "navigation", "footer", "cta"):
                        sections.append(s)
                    elif isinstance(s, dict) and "name" in s:
                        name = s["name"].lower()
                        if name not in ("hero", "nav", "navigation", "footer", "cta"):
                            sections.append(s.get("description", s["name"]))
        except (json.JSONDecodeError, OSError):
            pass

    # 3. PLAN.md section/feature references (only if no sections found yet)
    if not sections:
        plan = load_active_plan()
        if plan:
            for line in plan.split("\n"):
                line = line.strip()
                if line.startswith("- [ ]") or line.startswith("- [x]"):
                    item = line[5:].strip()
                    if len(item) > 5:
                        sections.append(item)

    return sections[:12]  # cap at 12 to avoid drip explosion


def _drip_decompose(description, filepath):
    """
    Decompose a file generation task into micro-drips (10-15 small pieces).
    Returns list of dicts: [{name, desc, group, depends_on_previous?}]
    Supports HTML, Python, JS/TS, and CSS file types.
    """
    ext = Path(filepath).suffix.lower()

    # Route to file-type-specific decomposers
    if ext == ".py":
        return _drip_decompose_python(description, filepath)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        return _drip_decompose_js(description, filepath)
    elif ext == ".css":
        return _drip_decompose_css(description, filepath)

    # Original HTML drip decomposition below
    desc_lower = description.lower()

    drips = []

    # Group 1: Head (must be first)
    drips.append({
        "name": "doctype_head",
        "desc": "Generate DOCTYPE, <html lang='en'>, <head> with charset, viewport meta, title, Tailwind CSS CDN (<script src='https://cdn.tailwindcss.com'></script>), any Google Fonts link, custom <style> block for animations/gradients, tailwind config for dark mode ('darkMode: class'), and opening <body> tag with base classes",
        "group": 1,
        "depends_on_html": False,
    })

    # Group 2: Top visual sections (independent of each other)
    drips.append({
        "name": "navigation",
        "desc": "Generate a sticky/fixed navigation bar: logo/brand name on left, horizontal nav links (Home, Features, Pricing, Contact), dark mode toggle button, mobile hamburger menu button (hidden on desktop), wrap in <nav> with id='navbar'",
        "group": 2,
        "depends_on_html": False,
    })
    # FIX 3: Merged hero_text + hero_visual into single drip to avoid duplicate <section id='hero'>
    drips.append({
        "name": "hero_section",
        "desc": "Generate complete hero section: <section id='hero'> with large headline <h1>, subtext paragraph, two CTA buttons (primary filled + secondary outline), decorative gradient blobs/shapes as background, close </section>",
        "group": 2,
        "depends_on_html": False,
    })

    # --- DYNAMIC CONTENT SECTIONS (description-driven) ---
    content_sections = _extract_sections_from_description(description)

    _CARD_STYLE = "bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1"

    if content_sections:
        # Features grid header
        drips.append({
            "name": "features_header",
            "desc": "Generate features section opening: <section id='features'>, section headline, subtitle, opening grid container (grid cols-1 md:cols-3 gap-8)",
            "group": 3,
            "depends_on_html": False,
        })
        # One card per extracted section
        for i, section_topic in enumerate(content_sections, 1):
            closing_instruction = ""
            if i == len(content_sections):
                closing_instruction = " After this card, close the grid container </div> and close </section>."
            drips.append({
                "name": f"content_card_{i}",
                "desc": f"Generate a feature/content card about: {section_topic}. A <div class='{_CARD_STYLE}'> with a relevant SVG icon, title <h3>, detailed description <p>. Use EXACTLY these classes on the outer div: {_CARD_STYLE}. Do NOT wrap in section or grid — this is one grid item.{closing_instruction}",
                "group": 3,
                "depends_on_html": False,
            })
    else:
        # Fallback: 3 generic cards derived from description keywords
        drips.append({
            "name": "features_header",
            "desc": "Generate features section opening: <section id='features'>, section headline, subtitle, opening grid container (grid cols-1 md:cols-3 gap-8)",
            "group": 3,
            "depends_on_html": False,
        })
        _card_themes = [
            ("Lightning-fast performance or speed", "a speed/bolt icon"),
            ("Security, encryption, or data protection", "a shield/lock icon"),
            ("Integration, API, or connectivity", "a plug/link icon"),
        ]
        for i, (theme, icon) in enumerate(_card_themes, 1):
            closing_instruction = ""
            if i == len(_card_themes):
                closing_instruction = " After this card, close the grid container </div> and close </section>."
            drips.append({
                "name": f"content_card_{i}",
                "desc": f"Generate feature card {i}: a <div class='{_CARD_STYLE}'> with {icon} as SVG, title <h3> about {theme}, description <p>. Use EXACTLY these classes on the outer div: {_CARD_STYLE}. Do NOT wrap in section or grid — this is one grid item.{closing_instruction}",
                "group": 3,
                "depends_on_html": False,
            })

    # Group 4: Bottom sections
    # Testimonials — only if description mentions them
    if any(w in desc_lower for w in ("testimonial", "review", "trust", "social proof")):
        drips.append({
            "name": "testimonials",
            "desc": "Generate testimonials section: <section id='testimonials'>, section title, 2-3 testimonial cards with quote text, author name, role/company, and avatar placeholder",
            "group": 4,
            "depends_on_html": False,
        })

    # Pricing — always included for landing/SaaS pages
    drips.append({
        "name": "pricing",
        "desc": "Generate pricing section: <section id='pricing'>, section title, 3 pricing cards (Starter/Pro/Enterprise) with real prices, REAL specific feature names (not 'Feature 1'), and CTA button. Highlight the middle card as 'Most Popular' with a badge",
        "group": 4,
        "depends_on_html": False,
    })

    # Contact form — always included for landing/SaaS pages
    drips.append({
        "name": "contact_form",
        "desc": "Generate contact/signup form section: <section id='contact'>, section title, a <form id='contact-form'> with name, email, message fields, and submit button with styling",
        "group": 4,
        "depends_on_html": False,
    })

    drips.append({
        "name": "cta_section",
        "desc": "Generate a call-to-action banner section: <section id='cta'>, compelling headline, subtext, prominent CTA button linking to href='#contact' (NOT href='#'), gradient or colored background",
        "group": 4,
        "depends_on_html": False,
    })

    drips.append({
        "name": "footer",
        "desc": "Generate footer: <footer id='footer'>, grid with columns for brand/about (write REAL description, not lorem ipsum), quick links matching nav (Home, Features, Pricing, Contact), contact info with realistic email/phone, social media SVG icons (Twitter, GitHub, LinkedIn), copyright with 2025, close </footer>",
        "group": 4,
        "depends_on_html": False,
    })

    # Group 5: JavaScript (needs all HTML IDs from previous groups)
    drips.append({
        "name": "js_dark_mode",
        "desc": "Generate <script> block for dark mode: toggle function that adds/removes 'dark' class on document.documentElement, save preference to localStorage, load preference on DOMContentLoaded, wire to darkModeToggle button using addEventListener('click', ...) — NOT 'change', it is a <button> not a checkbox",
        "group": 5,
        "depends_on_previous": True,
    })
    drips.append({
        "name": "js_interactions",
        "desc": "Generate <script> block for: mobile menu toggle, smooth scroll for anchor links, scroll-triggered fade-in animations using IntersectionObserver, navbar background change on scroll, form validation and submit handler (preventDefault + success message)",
        "group": 5,
        "depends_on_previous": True,
    })

    # Group 6: Closing
    drips.append({
        "name": "closing_tags",
        "desc": "Generate closing </body></html> tags only",
        "group": 6,
        "depends_on_html": False,
    })

    return drips


def _load_project_context_for_drip():
    """
    Read project files (PLAN.md, design.json, README.md) and return a
    compressed project context string for drip prompts.
    Keeps it under ~300 tokens so it doesn't eat the context window.
    """
    context_parts = []

    # PLAN.md — the user's vision
    plan_content = load_active_plan()
    if plan_content:
        # Extract just the objective and key details, not the full checklist
        lines = plan_content.split("\n")
        important = []
        for line in lines[:40]:  # first 40 lines max
            stripped = line.strip()
            if stripped and not stripped.startswith("- [x]"):  # skip completed items
                important.append(stripped)
        if important:
            context_parts.append("PROJECT PLAN:\n" + "\n".join(important[:20]))

    # design.json — design system
    design_path = Path(CWD) / "design.json"
    if design_path.is_file():
        try:
            design = json.loads(design_path.read_text(encoding="utf-8", errors="replace"))
            # Extract key design tokens
            d_parts = []
            if "brand_name" in design:
                d_parts.append(f"Brand: {design['brand_name']}")
            if "colors" in design:
                d_parts.append(f"Colors: {json.dumps(design['colors'])[:200]}")
            if "fonts" in design:
                d_parts.append(f"Fonts: {json.dumps(design['fonts'])[:100]}")
            if "tone" in design:
                d_parts.append(f"Tone: {design['tone']}")
            if "tagline" in design:
                d_parts.append(f"Tagline: {design['tagline']}")
            if "sections" in design:
                d_parts.append(f"Sections: {json.dumps(design['sections'])[:300]}")
            if d_parts:
                context_parts.append("DESIGN SYSTEM:\n" + "\n".join(d_parts))
        except (json.JSONDecodeError, OSError):
            pass

    # README.md — project description
    readme_path = Path(CWD) / "README.md"
    if readme_path.is_file():
        try:
            readme = readme_path.read_text(encoding="utf-8", errors="replace")
            # First 10 lines only
            readme_lines = [l for l in readme.split("\n")[:10] if l.strip()]
            if readme_lines:
                context_parts.append("README:\n" + "\n".join(readme_lines))
        except OSError:
            pass

    return "\n\n".join(context_parts) if context_parts else ""


# Cache project context so we don't re-read files for every drip
_drip_project_context_cache = {"value": None}


def _extract_relevant_context(filepath, task_hint, max_tokens=2000):
    """
    Context Surgeon Lite — extract only the relevant portion of an existing file.
    Uses AST for Python, regex for JS/TS, tag matching for HTML.
    Returns a string with max ~max_tokens worth of context, or empty string.
    """
    if not os.path.isfile(filepath):
        return ""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return ""

    if not content.strip():
        return ""

    ext = Path(filepath).suffix.lower()
    hint_lower = task_hint.lower()
    max_chars = max_tokens * 4  # rough token-to-char estimate

    # --- Python: use AST to extract target function/class + imports ---
    if ext == ".py":
        try:
            import ast as _ast
            tree = _ast.parse(content)
            parts = []
            # Always include imports (they're small and essential)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                    parts.append(_ast.get_source_segment(content, node) or "")
            # Find functions/classes mentioned in the hint
            for node in _ast.iter_child_nodes(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    if node.name.lower() in hint_lower or any(w in node.name.lower() for w in hint_lower.split()[:5]):
                        parts.append(_ast.get_source_segment(content, node) or "")
                elif isinstance(node, _ast.ClassDef):
                    if node.name.lower() in hint_lower or any(w in node.name.lower() for w in hint_lower.split()[:5]):
                        parts.append(_ast.get_source_segment(content, node) or "")
            if parts:
                result = "\n".join(p for p in parts if p)
                return result[:max_chars]
        except Exception:
            pass  # Fall through to generic fallback

    # --- JS/TS: regex to find function/class blocks + imports ---
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        parts = []
        # Capture import lines
        for m in re.finditer(r'^(import\s+.+)$', content, re.MULTILINE):
            parts.append(m.group(1))
        # Find function/class/const blocks matching hint keywords
        hint_words = [w for w in hint_lower.split()[:5] if len(w) > 3]
        for pattern in [
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{',
            r'(?:export\s+)?class\s+(\w+)\s*(?:extends\s+\w+)?\s*\{',
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|\bfunction\b)',
        ]:
            for m in re.finditer(pattern, content):
                name = m.group(1)
                if any(w in name.lower() for w in hint_words):
                    # Grab from match start to next top-level closing brace (rough)
                    start = m.start()
                    brace_count = 0
                    end = start
                    for i in range(start, min(start + 3000, len(content))):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    if end > start:
                        parts.append(content[start:end])
        if parts:
            result = "\n".join(parts)
            return result[:max_chars]

    # --- HTML: extract target section by id ---
    if ext in (".html", ".htm"):
        hint_words = [w for w in hint_lower.split() if len(w) > 3]
        for word in hint_words:
            pattern = rf'<section[^>]*id=["\']({re.escape(word)})["\'][^>]*>[\s\S]*?</section>'
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                return m.group(0)[:max_chars]

    # --- CSS: extract matching rule blocks ---
    if ext == ".css":
        hint_words = [w for w in hint_lower.split() if len(w) > 3]
        parts = []
        # Grab :root variables always
        root_match = re.search(r':root\s*\{[^}]+\}', content)
        if root_match:
            parts.append(root_match.group(0))
        for word in hint_words:
            for m in re.finditer(rf'[^{{}}]*{re.escape(word)}[^{{}}]*\{{[^}}]+\}}', content, re.IGNORECASE):
                parts.append(m.group(0))
        if parts:
            return "\n".join(parts)[:max_chars]

    # --- Generic fallback: first 80 + last 20 lines ---
    lines = content.split("\n")
    if len(lines) <= 100:
        return content[:max_chars]
    head = "\n".join(lines[:80])
    tail = "\n".join(lines[-20:])
    return f"{head}\n\n# ... (middle omitted) ...\n\n{tail}"[:max_chars]


def _build_drip_prompt(drip, drip_context, description, filepath):
    """Build a precision prompt for a single drip (~1500 tokens total). Supports all file types."""
    filename = Path(filepath).name
    ext = Path(filepath).suffix.lower()
    compressed = drip_context.compress()

    # Load project context (cached across drips)
    if _drip_project_context_cache["value"] is None:
        _drip_project_context_cache["value"] = _load_project_context_for_drip()
    project_ctx = _drip_project_context_cache["value"]

    # --- File-type-aware system prompt ---
    if ext in (".html", ".htm"):
        lang_label = "HTML/CSS/JS"
    elif ext == ".py":
        lang_label = "Python"
    elif ext in (".js", ".jsx"):
        lang_label = "JavaScript"
    elif ext in (".ts", ".tsx"):
        lang_label = "TypeScript"
    elif ext == ".css":
        lang_label = "CSS"
    else:
        lang_label = "code"

    system = (
        f"You generate ONLY 15-40 lines of {lang_label} code. No explanations, no markdown fences, no commentary.\n"
        f"File: {filename} | Objective: {description}\n"
    )
    if drip_context.color_scheme and ext in (".html", ".htm", ".css"):
        system += f"Color scheme: {drip_context.color_scheme} (use this in Tailwind classes)\n"

    # Inject project context from user's files
    if project_ctx:
        system += f"\n{project_ctx}\n"

    # Inject context for drips that depend on previous groups' output
    if drip.get("depends_on_previous"):
        # Primary source: accumulated code from earlier drips (always available during generation)
        if drip_context.accumulated_code.strip():
            # For large accumulated code, extract only the tail (most relevant for continuation)
            acc = drip_context.accumulated_code
            acc_lines = acc.strip().split("\n")
            if len(acc_lines) > 60:
                # Show first 15 lines (imports/setup) + last 30 lines (most recent context)
                snippet = "\n".join(acc_lines[:15]) + "\n# ... (earlier sections) ...\n" + "\n".join(acc_lines[-30:])
            else:
                snippet = acc
            system += f"\nCode generated so far:\n{snippet}\n"
        # Fallback: if editing an existing file, pull relevant context from it
        elif os.path.isfile(filepath):
            relevant_ctx = _extract_relevant_context(filepath, drip["desc"])
            if relevant_ctx:
                system += f"\nExisting file context:\n{relevant_ctx}\n"

    if compressed:
        system += f"\nContext from previous sections:\n{compressed}\n"

    # --- File-type-specific rules ---
    system += "\nRULES:\n"
    system += f"- Output ONLY raw {lang_label} code for this micro-section\n"
    system += "- Code must seamlessly continue from the previous section\n"
    system += "- Write COMPLETE, production-quality code — no placeholders, no TODOs, no pass statements\n"
    system += "- Do NOT repeat imports/definitions from earlier sections\n"

    if ext in (".html", ".htm"):
        system += (
            "- Do NOT include DOCTYPE, <html>, <head>, or <body> tags (unless this drip specifically asks for them)\n"
            "- Do NOT close tags that belong to a later section\n"
            "- Use Tailwind CSS utility classes for all styling\n"
            "- Use dark: variants for dark mode support\n"
            "- Make it modern, clean, visually polished\n"
            "- NEVER use placeholder text: no 'Lorem ipsum', no 'Your Company', no 'Feature 1', no 'John Doe'\n"
            "- Write REAL, specific, compelling copy. Invent realistic brand name, real feature names, real descriptions\n"
            "- Use real years (2025/2026), real-sounding emails, real content everywhere\n"
            "- Every link MUST point to a real section id (href='#features', '#contact', etc.) — NEVER use href='#'\n"
            "- Every button MUST do something: link to a section, submit a form, or trigger a JS function\n"
            "- Include <meta name='viewport'> for responsive design\n"
            "- Add lang attribute to <html>\n"
            "- All interactive elements need :focus-visible styles\n"
            "- All form inputs need associated <label> elements\n"
        )
    elif ext == ".py":
        system += (
            "- Use type hints on function signatures\n"
            "- Use descriptive variable names and docstrings for public functions\n"
            "- Handle errors appropriately (don't catch and pass silently)\n"
            "- Follow PEP 8 style\n"
            "- Use async def for I/O-bound endpoints (FastAPI, aiohttp)\n"
            "- Use logging module, not print(), for production code\n"
            "- Use Pydantic/dataclass for structured data, not raw dicts\n"
            "- Never use 'from X import *' — import specific names\n"
            "- Never use mutable default arguments (def f(x=[])). Use None and assign in body\n"
            "- No hardcoded URLs — use environment variables for all host/port values\n"
        )
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        system += (
            "- Use modern ES6+ syntax (const/let, arrow functions, destructuring)\n"
            "- Use async/await for asynchronous operations\n"
            "- Include proper error handling (try/catch)\n"
        )
        if ext in (".ts", ".tsx"):
            system += "- Include TypeScript types/interfaces for all function parameters and return values\n"
        if ext in (".jsx", ".tsx"):
            system += (
                "- Use React best practices: functional components, hooks, proper key props\n"
                "- Add 'use client' as FIRST LINE if using hooks, event handlers, or framer-motion\n"
                "- Guard browser APIs: typeof window !== 'undefined' before window/document/localStorage\n"
                "- Animations: use key prop, AnimatePresence, or useAnimate — never bare animate={{}}\n"
                "- Use next/image instead of <img> in Next.js projects\n"
                "- Remove all console.log before shipping — use proper error boundaries instead\n"
                "- useEffect with timers/listeners MUST return cleanup function\n"
                "- Never use 'any' type — define proper interfaces/types\n"
                "- Never use array index as key in .map() for reorderable lists\n"
                "- No hardcoded localhost URLs — use environment variables\n"
                "- No dangerouslySetInnerHTML without DOMPurify sanitization\n"
                "- NEVER use href='#' — use real routes. Use next/link <Link> component, not raw <a> tags\n"
                "- If you create a sidebar/nav with page links, you MUST create all the corresponding page files\n"
            )
    elif ext == ".css":
        system += (
            "- Use CSS custom properties (var(--...)) for colors and spacing\n"
            "- Use modern CSS features: flexbox, grid, clamp(), gap\n"
            "- Include smooth transitions on interactive elements\n"
            "- @keyframes and animation definitions in @layer base, not @layer utilities\n"
            "- Include @media (prefers-reduced-motion: reduce) for all animations\n"
            "- 3D transforms need perspective on parent and preserve-3d on child\n"
            "- Use rem/em for font-size, never px (breaks accessibility)\n"
            "- Use CSS custom properties var(--*) for colors, not hardcoded hex/rgb\n"
            "- Include :focus-visible styles on all interactive elements\n"
        )

    user_msg = f"Generate this micro-section: {drip['desc']}\n\nOutput only the code, nothing else."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def _post_validate_html(html):
    """
    Universal HTML post-validation: catch placeholder text, dead links, broken wiring,
    dark mode issues, fade-in mismatches, and insane color choices.
    Called by drip generation, chunked generation, AND _assemble_html.
    Returns (fixed_html, list_of_(issue, fix_applied)_tuples).
    """
    fixes = []

    # --- FIX 1+8: Comprehensive placeholder detection and replacement ---
    _PLACEHOLDER_REPLACEMENTS = [
        # Feature placeholders (expanded: "Feature 1", "Feature Title 1", etc.)
        (r'Feature (?:Title )?\d+', [
            "Lightning Performance", "Bank-Grade Security", "Seamless Integration",
            "Smart Analytics", "Team Collaboration", "API Access",
            "SSO Authentication", "Audit Logs", "Custom Branding",
            "Dedicated Support",
        ]),
        # Generic numbered placeholders
        (r'(?:Title|Card|Item|Service|Benefit|Point) \d+', [
            "Accelerated Workflows", "Enterprise Security", "Effortless Connectivity",
            "Real-Time Insights", "Priority Support", "Full Customization",
        ]),
    ]
    for pattern, replacements in _PLACEHOLDER_REPLACEMENTS:
        found = re.findall(pattern, html)
        if found:
            for i, placeholder in enumerate(dict.fromkeys(found)):  # unique, order-preserved
                if i < len(replacements):
                    html = html.replace(placeholder, replacements[i])  # replace ALL occurrences
            fixes.append((f"{len(found)} placeholder(s) matching '{pattern}'", "Replaced with real names"))

    # Description placeholders
    desc_pattern = re.compile(r'Description (?:for |of )?(?:feature|card|item|service) \d+', re.IGNORECASE)
    if desc_pattern.search(html):
        html = desc_pattern.sub(
            "Built for teams who need speed without sacrificing reliability.", html
        )
        fixes.append(("Placeholder descriptions", "Replaced with real copy"))

    # Fake names
    for fake, real in [("John Doe", "Sarah Chen"), ("Jane Doe", "Marcus Rivera"),
                       ("John Smith", "Sarah Chen"), ("Jane Smith", "Marcus Rivera")]:
        if fake in html:
            html = html.replace(fake, real)
            fixes.append((f"Fake name '{fake}'", f"Replaced with {real}"))

    # Fake contact info
    _CONTACT_REPLACEMENTS = [
        (r'user@example\.com', 'hello@nexuslabs.io'),
        (r'contact@example\.com', 'team@nexuslabs.io'),
        (r'email@example\.com', 'hello@nexuslabs.io'),
        (r'example@email\.com', 'hello@nexuslabs.io'),
        (r'\+?1?[\s-]?\(?123\)?[\s-]?456[\s-]?7890', '+1 (555) 932-4178'),
        (r'123 (?:Main|Fake|Test) St(?:reet)?', '742 Evergreen Ave, Suite 200'),
    ]
    for pattern, replacement in _CONTACT_REPLACEMENTS:
        before = html
        html = re.sub(pattern, replacement, html, flags=re.IGNORECASE)
        if html != before:
            fixes.append((f"Fake contact '{pattern}'", f"Replaced with {replacement}"))

    # --- Fix href="#" — point to #contact or first section as fallback ---
    if 'href="#"' in html:
        first_id = re.search(r'<section[^>]*id=["\']([^"\']+)', html)
        fallback = f"#{first_id.group(1)}" if first_id else "#contact"
        html = html.replace('href="#"', f'href="{fallback}"')
        fixes.append(("Dead href='#' links", f"Pointed to {fallback}"))

    # --- Replace lorem ipsum ---
    lorem_pattern = re.compile(r'Lorem ipsum[^<"\']*', re.IGNORECASE)
    if lorem_pattern.search(html):
        html = lorem_pattern.sub(
            "We build tools that help teams move faster, communicate better, and ship with confidence.", html
        )
        fixes.append(("Lorem ipsum text", "Replaced with real copy"))

    # --- Fix generic company names ---
    for placeholder in ("Your Company", "Company Name"):
        if placeholder in html:
            html = html.replace(placeholder, "Nexus Labs")
            fixes.append((f"Generic '{placeholder}'", "Replaced with Nexus Labs"))

    # --- Fix old years (handles both © and &copy;) ---
    _YEAR_PATTERN = re.compile(r'(?:©|&copy;)\s*20(?:2[0-4]|1\d)')
    if _YEAR_PATTERN.search(html):
        html = re.sub(r'(©|&copy;)\s*20(?:2[0-4]|1\d)', r'\1 2025', html)
        fixes.append(("Outdated copyright year", "Updated to 2025"))

    # --- FIX 2: Dark mode toggle — 'change' → 'click' for buttons ---
    before = html
    html = re.sub(
        r"(darkModeToggle[^;]*addEventListener\s*\(\s*)['\"]change['\"]",
        r"\1'click'",
        html
    )
    if html != before:
        fixes.append(("Dark mode toggle 'change' event", "Fixed to 'click' for button"))

    # --- FIX 5: .fade-in JS references but no elements have the class ---
    if '.fade-in' in html or 'fade-in' in html:
        has_fade_in_class = bool(re.search(r'class=["\'][^"\']*\bfade-in\b', html))
        has_fade_in_js = bool(re.search(r'(?:querySelectorAll|getElementsByClassName)\s*\(\s*["\']\.?fade-in', html))
        if has_fade_in_js and not has_fade_in_class:
            # Add fade-in to sections that have a class attribute
            html = re.sub(
                r'<section\b([^>]*class=["\'])([^"\']*)',
                r'<section\1fade-in \2',
                html
            )
            # Also handle sections without a class attribute
            html = re.sub(
                r'<section\b(?![^>]*class=)([^>]*>)',
                r'<section class="fade-in"\1',
                html
            )
            fixes.append(("Missing .fade-in classes", "Added to all <section> tags"))

    # --- FIX 6: Insane dark mode background colors ---
    _BAD_DARK_BG = re.compile(
        r'dark:bg-(emerald|green|blue|red|orange|yellow|pink|purple|indigo|teal|cyan|sky|rose|amber|lime|violet|fuchsia)-[3-9]00'
    )
    bad_dark_matches = _BAD_DARK_BG.findall(html)
    if bad_dark_matches:
        html = _BAD_DARK_BG.sub('dark:bg-gray-800', html)
        fixes.append((f"{len(bad_dark_matches)} bright dark:bg- colors", "Replaced with dark:bg-gray-800"))

    # --- FIX 3 (universal): Dedup duplicate section IDs even in single-file writes ---
    seen_section_ids = set()
    for m in list(re.finditer(r'<section[^>]*id=["\']([^"\']+)["\'][^>]*>[\s\S]*?</section>', html)):
        sid = m.group(1)
        if sid in seen_section_ids:
            html = html.replace(m.group(0), '', 1)
            fixes.append((f"Duplicate <section id='{sid}'>", "Removed duplicate"))
        seen_section_ids.add(sid)

    return html, fixes


# Keep old name as alias for any external callers
_drip_post_validate = _post_validate_html


# Drip names that are boilerplate / mechanical — route to SMALL_MODEL for speed
_BOILERPLATE_DRIPS = frozenset({
    "doctype_head", "closing_tags", "imports_config", "imports_types",
    "variables_reset", "exports_final",
})


def _drip_generate_single(drip, drip_context, description, filepath, model):
    """Generate a single drip. Returns (drip_name, output_text) or (drip_name, None) on failure."""
    messages = _build_drip_prompt(drip, drip_context, description, filepath)

    # Per-drip model routing: boilerplate drips use SMALL_MODEL for speed
    drip_model = SMALL_MODEL if drip["name"] in _BOILERPLATE_DRIPS else model

    ext = Path(filepath).suffix.lower()
    lang_label = "code"
    if ext in (".html", ".htm"):
        lang_label = "HTML"
    elif ext == ".py":
        lang_label = "Python"
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        lang_label = "JS"
    elif ext == ".css":
        lang_label = "CSS"

    for attempt in range(2):
        try:
            full_text = ""
            for chunk in ollama_chat(messages, drip_model, tools=None, stream=True, num_ctx=4096):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_text += content

            # Strip markdown code fences if model wrapped output
            cleaned = full_text.strip()
            cleaned = re.sub(r'^```(?:\w+)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            cleaned = cleaned.strip()

            if cleaned:
                return drip["name"], cleaned
            # Empty result — retry once with simpler prompt
            if attempt == 0:
                messages[1]["content"] = f"Write {drip['name']} {lang_label} code. {drip['desc'][:100]}"
                continue
        except Exception:
            if attempt == 0:
                # On first failure of boilerplate drip, escalate to main model
                if drip_model == SMALL_MODEL:
                    drip_model = model
                continue

    return drip["name"], None


def _drip_generate_parallel(drips, drip_context, description, filepath, model):
    """Generate a group of independent drips in parallel using threads."""
    results = {}
    errors = []

    def _worker(drip):
        name, output = _drip_generate_single(drip, drip_context, description, filepath, model)
        if output:
            results[name] = output
        else:
            errors.append(name)

    threads = []
    for drip in drips:
        t = threading.Thread(target=_worker, args=(drip,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=OLLAMA_STREAM_TIMEOUT)

    return results, errors


def _drip_generate_file(filepath, description, messages, model):
    """
    Generate a file using the drip method — micro-chunked generation.
    Returns (success, assembled_content).
    """
    filename = Path(filepath).name
    ext = Path(filepath).suffix.lower()

    # --- Step 1: Decompose into micro-drips ---
    drips = _drip_decompose(description, filepath)
    if not drips:
        return False, ""

    print(f"\n  {C.CLAW}{BLACK_CIRCLE} Drip plan: {len(drips)} micro-drips for {filename}{C.RESET}")
    # Show drip groups
    groups = {}
    for d in drips:
        groups.setdefault(d["group"], []).append(d["name"])
    for g in sorted(groups):
        names = ", ".join(groups[g])
        parallel_tag = " (parallel)" if len(groups[g]) > 1 else " (sequential)"
        print(f"    {C.SUBTLE}Group {g}{parallel_tag}: {names}{C.RESET}")

    drip_context = DripContext()
    ordered_sections = []  # list of (name, content) to preserve order
    fail_count = 0

    # --- Step 2: Generate drips group by group ---
    group_nums = sorted(set(d["group"] for d in drips))

    drips_completed = 0
    total_drips = len(drips)

    for group_num in group_nums:
        group_drips = [d for d in drips if d["group"] == group_num]

        if len(group_drips) == 1:
            # Single drip — generate sequentially
            drip = group_drips[0]
            print(f"  {C.TOOL}{BLACK_CIRCLE} Drip: {drip['name']}{C.RESET}", end="", flush=True)
            spinner = TimedSpinner(f"Dripping {drip['name']}", C.TOOL)
            spinner.start()
            name, output = _drip_generate_single(drip, drip_context, description, filepath, model)
            spinner.stop()
            if output:
                drip_context.ingest(name, output)
                ordered_sections.append((name, output))
                line_count = len(output.strip().split("\n"))
                drips_completed += 1
                print(f"    {VERIFY_OK} {name}: {line_count} lines")
            else:
                fail_count += 1
                drips_completed += 1
                print(f"    {VERIFY_FAIL} {name}: failed")
        else:
            # Multiple drips — generate in parallel
            parallel_names = [d["name"] for d in group_drips]
            print(f"  {C.TOOL}{BLACK_CIRCLE} Parallel drips: {', '.join(parallel_names)}{C.RESET}")
            spinner = TimedSpinner(f"Dripping {len(group_drips)} sections", C.TOOL)
            spinner.start()
            results, errors = _drip_generate_parallel(group_drips, drip_context, description, filepath, model)
            spinner.stop()

            # Ingest results in drip order (preserves intended assembly order)
            for drip in group_drips:
                drips_completed += 1
                if drip["name"] in results:
                    output = results[drip["name"]]
                    drip_context.ingest(drip["name"], output)
                    ordered_sections.append((drip["name"], output))
                    line_count = len(output.strip().split("\n"))
                    print(f"    {VERIFY_OK} {drip['name']}: {line_count} lines")
                else:
                    fail_count += 1
                    print(f"    {VERIFY_FAIL} {drip['name']}: failed")

        # Progress bar after each group
        print(_progress_bar(drips_completed, total_drips, width=25, label="Drip progress"))

        # If too many failures, bail to chunked
        if fail_count >= 3:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Too many drip failures ({fail_count}), aborting drip method{C.RESET}")
            return False, ""

    if not ordered_sections:
        return False, ""

    # --- Step 3: Assemble ---
    print(f"\n  {C.CLAW}{BLACK_CIRCLE} Assembling {len(ordered_sections)} drips...{C.RESET}")

    # Build sections dict preserving order for _assemble_html
    sections = {name: content for name, content in ordered_sections}

    # Use the HTML post-processor if applicable
    rules = _ASSEMBLY_RULES.get(ext, {})
    post_proc = rules.get("post_process")

    if post_proc == "_assemble_html":
        assembled = _assemble_html(sections)
    else:
        joiner = rules.get("joiner", "\n\n")
        assembled = joiner.join(sections.values())

    total_lines = len(assembled.strip().split("\n"))
    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Drip assembled: {len(assembled)} chars, {total_lines} lines from {len(ordered_sections)} drips{C.RESET}")

    # --- Step 4: Post-validation — catch placeholders, dead links, broken wiring ---
    assembled, validation_fixes = _post_validate_html(assembled)
    if validation_fixes:
        print(f"  {C.WARNING}{BLACK_CIRCLE} Post-validation fixes:{C.RESET}")
        for issue, fix in validation_fixes:
            print(f"    {VERIFY_OK} {issue} → {fix}")

    # Clear the project context cache for next run
    _drip_project_context_cache["value"] = None

    # Bell notification — drip is done
    _bell()

    return True, assembled


def _chunked_generate_file(filepath, description, messages, model):
    """
    Generate a large file in chunks for weaker/slower models.
    Step 1: Decompose the objective into small ordered chunks (via model or fallback).
    Step 2: Generate each chunk independently.
    Step 3: Assemble in order with post-processing.
    Returns (success, assembled_content).
    """
    ext = Path(filepath).suffix.lower()
    filename = Path(filepath).name

    # --- Drip delegation: use micro-drip for ALL eligible file types ---
    rules = _ASSEMBLY_RULES.get(ext, {})
    desc_lower = description.lower()
    # HTML still requires full-page keywords to avoid dripping tiny snippets
    _html_full_page_kw = ("landing", "page", "website", "site", "portfolio", "homepage",
                          "dashboard", "app", "application", "store", "shop")
    use_drip = False
    if rules.get("drip_eligible"):
        if ext in (".html", ".htm"):
            use_drip = any(k in desc_lower for k in _html_full_page_kw)
        else:
            # Non-HTML drip-eligible types: always use drip if description is substantial
            use_drip = len(description.split()) >= 3
    if use_drip:
        print(f"  {C.CLAW}{BLACK_CIRCLE} Using drip generation for {filename}{C.RESET}")
        try:
            ok, content = _drip_generate_file(filepath, description, messages, model)
            if ok and content:
                return ok, content
            print(f"  {C.WARNING}{BLACK_CIRCLE} Drip generation failed, falling back to chunked{C.RESET}")
        except Exception as e:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Drip error: {e} — falling back to chunked{C.RESET}")

    if ext not in _ASSEMBLY_RULES:
        # For unknown types, still try with generic decomposition
        pass

    # --- Step 1: Decompose ---
    chunks = _decompose_objective(description, filepath, model)
    if not chunks:
        return False, ""

    print(f"\n  {C.CLAW}{BLACK_CIRCLE} Generating {len(chunks)} chunks for {filename}...{C.RESET}\n")
    sections = {}

    # --- Step 2: Generate each chunk ---
    for i, (chunk_name, chunk_desc) in enumerate(chunks):
        print(f"  {C.TOOL}{BLACK_CIRCLE} Chunk {i+1}/{len(chunks)}: {chunk_name}{C.RESET}")

        # Build context from previously generated sections
        prev_context = ""
        if sections:
            # Show last 2 sections in full for continuity, earlier ones summarized
            sec_list = list(sections.items())
            if len(sec_list) > 2:
                for name, content in sec_list[:-2]:
                    prev_context += f"[{name}]: {content[:150]}...\n"
            for name, content in sec_list[-2:]:
                prev_context += f"[{name}]:\n{content}\n\n"

        chunk_system = (
            f"You are generating section '{chunk_name}' of the file '{filename}'.\n"
            f"Overall objective: {description}\n\n"
            f"This section must contain: {chunk_desc}\n\n"
            f"RULES:\n"
            f"- Output ONLY raw code for this section — no markdown fences, no explanations, no comments like 'here is the code'\n"
            f"- This will be concatenated with other sections, so:\n"
            f"  - Do NOT repeat imports/setup from earlier sections\n"
            f"  - Do NOT add closing tags/brackets that belong to later sections\n"
            f"  - Make sure your code connects with the previous section\n"
            f"- Write COMPLETE, production-quality code — no placeholders, no TODOs\n"
        )

        if prev_context:
            chunk_system += f"\nAlready generated (your code follows after this):\n{prev_context}\n"

        chunk_messages = [
            {"role": "system", "content": chunk_system},
            {"role": "user", "content": f"Generate the '{chunk_name}' section now. Output only the code."},
        ]

        try:
            full_text = ""
            spinner = TimedSpinner(f"Generating {chunk_name}", C.TOOL)
            spinner.start()
            for chunk in ollama_chat(chunk_messages, model, tools=None, stream=True):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_text += content
                    spinner.add_tokens(len(content.split()))
            spinner.stop()

            # Strip markdown code fences if model wrapped output
            cleaned = full_text.strip()
            cleaned = re.sub(r'^```(?:\w+)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            cleaned = cleaned.strip()

            if cleaned:
                sections[chunk_name] = cleaned
                print(f"    {VERIFY_OK} {chunk_name}: {len(cleaned)} chars")
            else:
                print(f"    {VERIFY_FAIL} {chunk_name}: empty output, skipping")
        except Exception as e:
            print(f"    {VERIFY_FAIL} {chunk_name} failed: {e}")
            continue

    if not sections:
        return False, ""

    # --- Step 3: Assemble ---
    print(f"\n  {C.CLAW}{BLACK_CIRCLE} Assembling {len(sections)} sections...{C.RESET}")

    # Check for post-processing
    rules = _ASSEMBLY_RULES.get(ext, {})
    post_proc = rules.get("post_process")

    if post_proc == "_assemble_html":
        assembled = _assemble_html(sections)
    else:
        joiner = rules.get("joiner", "\n\n")
        assembled = joiner.join(sections.values())

    # Run syntax validation on assembled result
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8",
                                       dir=CWD if Path(CWD).exists() else None)
    try:
        tmp.write(assembled)
        tmp.close()
        valid, detail = _validate_file_syntax(tmp.name)
        if valid:
            print(f"  {VERIFY_OK} Assembled file syntax: OK")
        else:
            print(f"  {VERIFY_FAIL} Assembled file syntax error: {detail}")
            print(f"  {C.WARNING}{BLACK_CIRCLE} Attempting auto-fix...{C.RESET}")
            # Try to fix by asking model
            fix_messages = [
                {"role": "system", "content": "Fix the syntax error in this code. Output ONLY the corrected code, nothing else."},
                {"role": "user", "content": f"Syntax error: {detail}\n\nCode:\n{assembled[-3000:]}"},
            ]
            try:
                fix_text = ""
                for chunk in ollama_chat(fix_messages, model, tools=None, stream=True):
                    c = chunk.get("message", {}).get("content", "")
                    if c:
                        fix_text += c
                fix_text = re.sub(r'^```(?:\w+)?\s*\n?', '', fix_text.strip())
                fix_text = re.sub(r'\n?```\s*$', '', fix_text)
                if fix_text.strip():
                    assembled = assembled[:-3000] + fix_text.strip() if len(assembled) > 3000 else fix_text.strip()
            except Exception:
                pass
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # --- Post-validate HTML for chunked output (same gate as drip) ---
    ext = Path(filepath).suffix.lower()
    if ext in (".html", ".htm"):
        assembled, validation_fixes = _post_validate_html(assembled)
        if validation_fixes:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Post-validation fixes:{C.RESET}")
            for issue, fix in validation_fixes:
                print(f"    {VERIFY_OK} {issue} → {fix}")

    total_chars = len(assembled)
    total_sections = len(sections)
    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Assembled: {total_chars} chars from {total_sections} chunks{C.RESET}")

    return True, assembled


# ---------------------------------------------------------------------------
# system prompt
# ---------------------------------------------------------------------------

# Patterns that indicate a local/Ollama model (not a cloud API model)
_LOCAL_MODEL_PATTERNS = (
    "qwen", "llama", "mistral", "codellama", "deepseek", "phi", "gemma",
    "starcoder", "wizardcoder", "orca", "neural", "yi-", "command-r",
    "dolphin", "nous", "tinyllama", "stable", "falcon", "vicuna",
)


def _is_local_model(model):
    """Check if a model name matches a local/Ollama model (≤14B params)."""
    if PROVIDER.lower() != "ollama":
        return False  # Cloud providers are never "local"
    model_lower = model.lower()
    return any(p in model_lower for p in _LOCAL_MODEL_PATTERNS)


def _build_slim_system_prompt():
    """
    Terse system prompt for local models (~1.5K tokens).
    Gives the model room to think by cutting the 10K+ full prompt down to essentials.
    """
    prompt = dedent(f"""\
        You are Rattlesnake, a senior full-stack developer AI. You operate on the user's filesystem using tools.
        You HAVE full access to the user's machine. You CAN and MUST run shell commands, read files, write files, and edit files using your tools.
        NEVER say you cannot access the filesystem or run commands. You have bash, read_file, write_file, edit_file, grep_search, glob_search, and ask_user tools available.

        # Environment
        - CWD: {CWD}
        - Platform: {sys.platform}
        - Date: {__import__('datetime').date.today().isoformat()}

        # Core Rules
        1. ALWAYS use tools. NEVER paste code in chat. Use write_file, edit_file, read_file, bash. NEVER say you cannot run commands — you CAN.
        2. Use ask_user for ALL questions (never type questions as plain text).
        3. ALWAYS read_file before editing (system blocks edits to unread files). grep_search to find code first.
        4. After write_file/edit_file, verify the file exists.
        5. After bash commands, check exit code. If non-zero, read error and fix.
        6. For simple tasks (website, script, fix bug): just BUILD IT immediately.
        7. For complex projects: check PLAN.md first. If it exists, follow it. If not, create one.
        8. After completing each PLAN.md step, IMMEDIATELY use edit_file to change `- [ ]` to `- [x]` for that step.

        # File Generation
        - Write COMPLETE, production-quality code. No TODOs, no placeholders, no Lorem ipsum.
        - Use Tailwind CSS for styling. Dark mode support with dark: variants.
        - Real content everywhere: real brand names, real feature names, real descriptions. NEVER use the project slug as the app name.
        - Every link must point somewhere real. Every button must do something.
        - Every package you import MUST be in package.json dependencies. Run `npm install <pkg>` or add it manually.
        - In Next.js: interactive UI (forms, inputs, chat) MUST be in 'use client' components with event handlers. Server components cannot have onChange/onClick.
        - In Next.js: NEVER redirect() a page to its own route — that's an infinite loop. Create the resource then redirect to its URL.
        - Use theme-aware colors (bg-background, text-foreground, or dark: variants) — NEVER hardcode text-gray-900/bg-white when the app defaults to dark mode.
        - Load custom fonts with next/font/google (Inter, JetBrains Mono) — don't just list them in tailwind.config without importing.

        # Project Detection
        - package.json → Node project → npm install, npm run dev
        - requirements.txt → Python → pip install, python app.py
        - Only .html/.css/.js (no package.json) → STATIC site → open in browser, no npm/pip

        # Security
        - Never hardcode secrets. Use .env files.
        - Parameterized queries only. Sanitize all user input.
        - bcrypt for passwords. JWT validation server-side.

        # Inner Monologue
        Before responding, wrap brief reasoning in <think>...</think> (under 80 words). Not saved to context.
    """)

    # Inject CLAW.md if present (important for project-specific rules)
    claw_files = load_claw_md()
    if claw_files:
        prompt += "\n# Project Instructions (CLAW.md)\n"
        for path, content in claw_files:
            prompt += f"{content[:500]}\n"

    # Inject active plan if present
    plan_content = load_active_plan()
    if plan_content:
        prompt += f"\n# Active Plan\n{plan_content}\n"

    # Inject detected project profile
    profile = _get_cached_profile()
    if profile and profile.base_info["type"] != "unknown":
        prompt += f"\n# Project: {profile.base_info['type']}"
        if profile.framework:
            prompt += f" ({profile.framework})"
        if profile.styling:
            prompt += f", {profile.styling}"
        prompt += "\n"
        # Framework-specific directive (compact)
        directives = {
            "nextjs": "Use App Router, Server Components by default.",
            "nuxt": "Composition API. No React patterns.",
            "svelte": "Svelte 5 runes. No React patterns.",
            "fastapi": "Pydantic models, async endpoints.",
            "django": "Class-based views, Django conventions.",
        }
        if profile.framework in directives:
            prompt += f"- {directives[profile.framework]}\n"

    # Tool-first behaviour reinforcement (all providers, especially local models)
    prompt += dedent("""
        ## Tool Usage Priority — MANDATORY
        You are an AGENT, not a chatbot. Your primary output is TOOL CALLS, not text.
        - You HAVE full access to the user's machine via your tools. You CAN run commands, read files, write files.
        - To CREATE a new file: call write_file (do not show the code as text)
        - To MODIFY an existing file: call edit_file with the SMALLEST possible change
        - To run a command: call bash
        - To gather system info: call bash with the appropriate command
        - Text output: only for brief status updates (1-3 sentences between tool calls)
        - If you need to create more than 20 lines of code, it MUST go through write_file
        - NEVER use write_file on an existing file to make a small change — use edit_file instead
        - Change ONLY the specific lines the user asked about. Leave everything else untouched.
        - The system BLOCKS edit_file/write_file if you haven't read the file first. Always: read_file → edit_file.
        - NEVER say you cannot run commands or access files. You CAN and MUST use your tools to do so.

        ## ABSOLUTELY FORBIDDEN — NEVER DO THESE
        - NEVER show bash/shell commands as text and tell the user to run them. YOU run them with the bash tool.
        - NEVER say "run this command" or "try this" — just CALL the bash tool and run it yourself.
        - NEVER list steps for the user to follow manually. YOU execute the steps with tools.
        - NEVER show code blocks as text output. Use write_file to create files, edit_file to modify them.
        - NEVER ask the user to copy-paste anything. YOU do the work.
        - If you catch yourself writing a code block or command in text: STOP. Delete it. Call the tool instead.
        - You are PAID to USE TOOLS, not to give instructions. ACT, don't advise.
        """)

    return prompt


# ---------------------------------------------------------------------------
# Design system — cached loader, selection functions, context builder
# ---------------------------------------------------------------------------

_design_json_cache = None


def _load_design_json():
    """Load and cache design.json. Returns dict or None on failure."""
    global _design_json_cache
    if _design_json_cache is not None:
        return _design_json_cache
    for adir in APIS_DIR_PATHS:
        design_file = adir / "design.json"
        if design_file.exists():
            try:
                data = json.loads(design_file.read_text(encoding="utf-8"))
                REQUIRED = {"palettes": dict, "typography": dict, "component_recipes": dict,
                            "page_layouts": dict, "slop_scan_patterns_design": dict,
                            "slop_scan_patterns_conversion": dict, "anti_slop_rules": list}
                for key, expected_type in REQUIRED.items():
                    val = data.get(key)
                    if val is None:
                        print(f"  Warning: design.json missing key: {key}")
                        return None
                    if not isinstance(val, expected_type):
                        print(f"  Warning: design.json key '{key}' has wrong type (expected {expected_type.__name__})")
                        return None
                _design_json_cache = data
                return data
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Security system — cached loader, profile detection, context builder
# ---------------------------------------------------------------------------

_security_json_cache = None


def _load_security_json():
    """Load and cache security.json. Returns dict or None on failure."""
    global _security_json_cache
    if _security_json_cache is not None:
        return _security_json_cache
    for adir in APIS_DIR_PATHS:
        security_file = adir / "security.json"
        if security_file.exists():
            try:
                data = json.loads(security_file.read_text(encoding="utf-8"))
                REQUIRED = {"header_configs": dict, "middleware_recipes": dict,
                            "validation_patterns": dict, "security_scan_patterns": dict,
                            "security_rules": list, "project_type_security": dict}
                for key, expected_type in REQUIRED.items():
                    val = data.get(key)
                    if val is None or not isinstance(val, expected_type):
                        return None
                _security_json_cache = data
                return data
            except Exception:
                pass
    return None


# --- Palette + typography + page-type selection ---

_PALETTE_KEYWORDS = [
    (["luxury", "gaming", "cinema", "exclusive", "moody"], "obsidian"),
    (["blog", "editorial", "content", "magazine", "news", "notion"], "bone"),
    (["agency", "portfolio", "creative", "minimal", "boutique"], "sand"),
    (["github", "terminal", "code", "api", "cli", "raycast"], "carbon"),
    (["dashboard", "analytics", "admin", "crm", "devtool", "saas", "inventory"], "midnight"),
]

_TYPOGRAPHY_KEYWORDS = [
    (["blog", "editorial", "content", "magazine", "news"], "editorial"),
    (["saas", "marketing", "landing", "launch", "homepage"], "modern"),
    (["fintech", "enterprise", "financial", "banking"], "geometric"),
    (["startup", "bold", "agency", "consumer", "social"], "geometric"),
    (["dashboard", "data", "devtool", "analytics", "admin", "crm"], "technical"),
]

_GOOGLE_FONT_PAIRINGS = {"editorial", "technical", "modern", "geometric"}

_LANDING_KEYWORDS = ["landing", "homepage", "marketing", "launch", "promote", "announce"]


def _select_palette(project_desc, design_data):
    """Select palette based on project description. Returns (name, css_variables_dict)."""
    desc_lower = project_desc.lower()
    palettes = design_data.get("palettes", {})
    palette_names = [k for k in palettes if not k.startswith("_")]
    if not palette_names:
        return "midnight", {}
    for keywords, palette_name in _PALETTE_KEYWORDS:
        if any(kw in desc_lower for kw in keywords):
            if palette_name in palettes:
                return palette_name, palettes[palette_name].get("css_variables", {})
    idx = int(hashlib.md5(project_desc.encode()).hexdigest(), 16) % len(palette_names)
    name = palette_names[idx]
    return name, palettes[name].get("css_variables", {})


def _select_typography(project_desc, design_data):
    """Select typography pairing. Returns dict with heading/body/mono/name keys."""
    desc_lower = project_desc.lower()
    pairings = design_data.get("typography", {}).get("pairings", [])
    valid = {p["name"]: p for p in pairings if p.get("name") in _GOOGLE_FONT_PAIRINGS}
    for keywords, typo_name in _TYPOGRAPHY_KEYWORDS:
        if any(kw in desc_lower for kw in keywords):
            if typo_name in valid:
                p = valid[typo_name]
                return {"heading": p["heading"], "body": p["body"], "mono": p["mono"], "name": typo_name}
    tech = valid.get("technical")
    if tech:
        return {"heading": tech["heading"], "body": tech["body"], "mono": tech["mono"], "name": "technical"}
    return {"heading": "JetBrains Mono", "body": "Inter", "mono": "JetBrains Mono", "name": "technical"}


def _detect_page_types(project_desc):
    """Detect page types from project description. Returns list of page type strings."""
    desc_lower = project_desc.lower()
    default_types = ["dashboard_overview", "entity_list", "entity_detail", "settings", "auth_login"]
    page_layout_keywords = {
        "dashboard_overview": ["dashboard", "admin", "overview", "analytics", "metrics"],
        "entity_list": ["list", "table", "manage", "inventory", "catalog", "browse"],
        "entity_detail": ["detail", "profile", "view", "single"],
        "settings": ["settings", "preferences", "config", "account"],
        "auth_login": ["login", "signup", "auth", "register"],
        "kanban_board": ["kanban", "board", "pipeline", "workflow", "trello"],
        "landing_page": ["landing", "homepage", "marketing", "launch"],
        "pricing_page": ["pricing", "plans", "subscription", "tiers"],
        "inbox_messaging": ["messaging", "chat", "inbox", "dm", "conversation"],
        "calendar_view": ["calendar", "scheduling", "booking", "events"],
        "analytics_dashboard": ["analytics", "reporting", "metrics", "chart"],
        "profile_page": ["profile", "user page", "portfolio"],
        "checkout_page": ["checkout", "payment", "cart", "purchase"],
        "file_manager": ["files", "documents", "media", "upload", "drive"],
        "contact_form": ["contact", "inquiry", "support form", "feedback"],
    }
    detected = []
    for page_type, keywords in page_layout_keywords.items():
        if any(kw in desc_lower for kw in keywords):
            detected.append(page_type)
    return detected if detected else default_types


def _is_landing_page(project_desc):
    """Check if project description implies a landing/marketing page."""
    desc_lower = project_desc.lower()
    return any(kw in desc_lower for kw in _LANDING_KEYWORDS)


def _detect_landing_framework(project_desc):
    """Detect which landing page framework to use. Only call when _is_landing_page() is True."""
    desc_lower = project_desc.lower()
    competitor_kw = ["vs", "versus", "alternative to", "compared to", "switch from", "migrate from", "replace"]
    pain_kw = ["replace", "fix", "broken", "frustrated", "alternative", "instead of", "better than"]
    urgency_kw = ["launch", "beta", "early access", "limited", "offer", "discount", "sale"]
    traction_kw = ["enterprise", "teams use", "trusted by", "customers", "migration from"]
    if any(kw in desc_lower for kw in competitor_kw):
        return "competitor_comparison"
    if any(kw in desc_lower for kw in pain_kw):
        return "problem_agitation_solution"
    if any(kw in desc_lower for kw in urgency_kw):
        return "urgency_scarcity"
    if any(kw in desc_lower for kw in traction_kw):
        return "authority_dominance"
    return "social_proof_wall"


# --- Context router — assembles the right subset per project ---

_TOKEN_CEILING_CHARS = 32000  # ~8K tokens at ~4 chars/token

_APP_RECIPES = ["stat_card", "data_table", "sidebar", "header", "button", "input",
                "select", "badge", "modal", "empty_state", "search_bar", "tabs",
                "toast", "skeleton_loader", "card"]
_APP_RECIPES_REDUCED = ["stat_card", "data_table", "sidebar", "header", "button",
                        "input", "select", "badge", "modal", "empty_state"]


def _fmt_recipe(name, recipe):
    """Format a single component recipe as compact markdown."""
    if isinstance(recipe, dict):
        parts = [f"### {name}"]
        for k, v in recipe.items():
            if k.startswith("_") or k == "notes":
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    parts.append(f"  {k}.{k2}: `{v2}`")
            else:
                parts.append(f"  {k}: `{v}`")
        notes = recipe.get("notes")
        if notes:
            parts.append(f"  *{notes}*")
        return "\n".join(parts)
    return f"### {name}\n  `{recipe}`"


def _fmt_layout(name, layout):
    """Format a page layout as compact markdown."""
    parts = [f"### {name}"]
    desc = layout.get("description", "")
    if desc:
        parts.append(f"  {desc}")
    structure = layout.get("structure", [])
    for s in structure:
        parts.append(f"  - {s}")
    classes = layout.get("layout_classes", {})
    for k, v in classes.items():
        parts.append(f"  {k}: `{v}`")
    return "\n".join(parts)


def _select_design_subset(project_desc, design_data, palette_name, palette_vars,
                           typography, page_types, is_landing, landing_framework):
    """Build the design context string. Hard ceiling ~8K tokens."""
    lines = ["# Design System Context", ""]
    desc_lower = project_desc.lower()

    # --- Always included ---
    philosophy = design_data.get("philosophy", [])
    if philosophy:
        lines.append("## Design Philosophy")
        for p in philosophy:
            lines.append(f"- {p}")
        lines.append("")

    anti_slop = design_data.get("anti_slop_rules", [])
    if anti_slop:
        lines.append("## Anti-Slop Rules (MUST follow)")
        for r in anti_slop:
            lines.append(f"- {r}")
        lines.append("")

    lines.append(f"## Palette: {palette_name}")
    lines.append("All colors use CSS variables. NEVER use raw Tailwind colors.")
    for k, v in palette_vars.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append(f"## Typography: {typography.get('name', 'technical')}")
    lines.append(f"  Heading: {typography.get('heading', '')}")
    lines.append(f"  Body: {typography.get('body', '')}")
    lines.append(f"  Mono: {typography.get('mono', '')}")
    scale = design_data.get("typography", {}).get("scale", {})
    if scale:
        for k, v in scale.items():
            lines.append(f"  {k}: `{v}`")
    lines.append("")

    include_interactions = True
    include_animations = True
    include_data_viz = ("analytics" in desc_lower or "chart" in desc_lower)
    use_full_recipes = True
    include_cta_formulas = True
    trim_psychology = False

    is_hybrid = is_landing and any(pt not in ("landing_page",) for pt in page_types)

    # --- App path ---
    app_lines = []
    if not is_landing or is_hybrid:
        recipes = design_data.get("component_recipes", {})
        recipe_list = _APP_RECIPES if use_full_recipes else _APP_RECIPES_REDUCED
        app_lines.append("## Component Recipes (use these exact classes)")
        for rname in recipe_list:
            if rname in recipes:
                app_lines.append(_fmt_recipe(rname, recipes[rname]))
        app_lines.append("")

        layouts = design_data.get("page_layouts", {})
        app_lines.append("## Page Layouts")
        for pt in page_types:
            if pt in layouts:
                app_lines.append(_fmt_layout(pt, layouts[pt]))
        app_lines.append("")

        resp_app = design_data.get("responsive_rules_app", {})
        app_rules = resp_app.get("rules", [])
        if app_rules:
            app_lines.append("## Responsive Rules (App)")
            for r in app_rules:
                app_lines.append(f"- {r}")
            app_lines.append("")

        if include_data_viz:
            viz_rules = design_data.get("data_visualization_rules", {}).get("rules", [])
            if viz_rules:
                app_lines.append("## Data Visualization Rules")
                for r in viz_rules:
                    app_lines.append(f"- {r}")
                app_lines.append("")

    # --- Landing path ---
    landing_lines = []
    if is_landing:
        frameworks = design_data.get("landing_page_frameworks", {})
        fw = frameworks.get(landing_framework, {})
        if fw:
            landing_lines.append(f"## Landing Framework: {fw.get('name', landing_framework)}")
            landing_lines.append(f"When to use: {fw.get('when_to_use', '')}")
            landing_lines.append(f"Visitor mindset: {fw.get('visitor_mindset', '')}")
            landing_lines.append("")
            section_flow = fw.get("section_flow", [])
            for sec in section_flow:
                landing_lines.append(f"### Section: {sec.get('section', '')}")
                landing_lines.append(f"  Purpose: {sec.get('purpose', '')}")
                elements = sec.get("elements", [])
                for el in elements:
                    landing_lines.append(f"  - {el}")
                if not trim_psychology:
                    pn = sec.get("psychology_notes", "")
                    if pn:
                        landing_lines.append(f"  Psychology: {pn}")
            landing_lines.append("")

        conv = design_data.get("universal_conversion_rules", {})
        atf = conv.get("above_the_fold", [])
        if atf:
            landing_lines.append("## Above the Fold Rules")
            for r in atf:
                landing_lines.append(f"- {r}")
            landing_lines.append("")
        copy_rules = conv.get("copy_rules", [])
        if copy_rules:
            landing_lines.append("## Copy Rules")
            for r in copy_rules:
                landing_lines.append(f"- {r}")
            landing_lines.append("")

        cta = design_data.get("cta_system", {})
        btn_rules = cta.get("button_text_rules", [])
        if btn_rules:
            landing_lines.append("## CTA Button Rules")
            for r in btn_rules:
                landing_lines.append(f"- {r}")
            landing_lines.append("")
        if include_cta_formulas:
            formulas = cta.get("cta_formulas_by_goal", {})
            if formulas:
                landing_lines.append("## CTA Formulas")
                for goal, examples in list(formulas.items())[:3]:
                    landing_lines.append(f"  {goal}: {', '.join(examples[:3])}")
                landing_lines.append("")
            micro = cta.get("micro_copy_below_cta", [])
            if micro:
                landing_lines.append("## Micro-copy below CTA")
                for m in micro[:5]:
                    landing_lines.append(f"- {m}")
                landing_lines.append("")

        sp = design_data.get("section_patterns", {})
        for pname in ("hero_badge", "metrics_bar", "testimonial_card"):
            pat = sp.get(pname, {})
            if pat:
                landing_lines.append(f"### Section Pattern: {pname}")
                what = pat.get("what", "")
                if what:
                    landing_lines.append(f"  {what}")
                classes = pat.get("classes", "")
                if classes:
                    landing_lines.append(f"  classes: `{classes}`")
                rules = pat.get("rules", [])
                for r in rules:
                    landing_lines.append(f"  - {r}")
                landing_lines.append("")

        if include_animations:
            anim = design_data.get("animation_patterns", {})
            for aname in ("scroll_reveal", "counter_animation"):
                a = anim.get(aname, {})
                if a:
                    landing_lines.append(f"### Animation: {aname}")
                    landing_lines.append(f"  {a.get('description', '')}")
                    arules = a.get("rules", [])
                    for r in arules:
                        landing_lines.append(f"  - {r}")
                    landing_lines.append("")

        resp_pub = design_data.get("responsive_rules_public", {})
        pub_rules = resp_pub.get("universal_responsive_rules", [])
        if pub_rules:
            landing_lines.append("## Responsive Rules (Public Pages)")
            for r in pub_rules:
                landing_lines.append(f"- {r}")
            landing_lines.append("")

    # --- Interaction patterns (always, unless cut) ---
    interaction_lines = []
    if include_interactions:
        ip = design_data.get("interaction_patterns", {})
        patterns = ip.get("patterns", [])
        if patterns:
            interaction_lines.append("## Interaction Patterns")
            for p in patterns:
                interaction_lines.append(f"- **{p.get('name', '')}**: {p.get('description', '')}")
            interaction_lines.append("")

    # --- Assemble and apply token ceiling ---
    all_lines = lines + interaction_lines + app_lines + landing_lines
    total_chars = sum(len(l) for l in all_lines)

    if is_hybrid and total_chars > _TOKEN_CEILING_CHARS:
        # Cut 1: trim psychology notes
        trim_psychology = True
        landing_lines_trimmed = []
        frameworks = design_data.get("landing_page_frameworks", {})
        fw = frameworks.get(landing_framework, {})
        if fw:
            landing_lines_trimmed.append(f"## Landing Framework: {fw.get('name', landing_framework)}")
            landing_lines_trimmed.append(f"When to use: {fw.get('when_to_use', '')}")
            landing_lines_trimmed.append("")
            for sec in fw.get("section_flow", []):
                landing_lines_trimmed.append(f"### Section: {sec.get('section', '')}")
                landing_lines_trimmed.append(f"  Purpose: {sec.get('purpose', '')}")
                for el in sec.get("elements", []):
                    landing_lines_trimmed.append(f"  - {el}")
            landing_lines_trimmed.append("")
        # Keep rest of landing lines (conv rules, cta, etc) - rebuild without framework
        non_fw_landing = []
        in_fw = False
        for ll in landing_lines:
            if ll.startswith("## Landing Framework:"):
                in_fw = True
                continue
            if in_fw and ll.startswith("## "):
                in_fw = False
            if not in_fw:
                non_fw_landing.append(ll)
        landing_lines = landing_lines_trimmed + non_fw_landing
        all_lines = lines + interaction_lines + app_lines + landing_lines
        total_chars = sum(len(l) for l in all_lines)

    if is_hybrid and total_chars > _TOKEN_CEILING_CHARS:
        # Cut 2: remove animations
        landing_lines = [l for l in landing_lines if not any(
            l.startswith(f"### Animation: {a}") for a in ("scroll_reveal", "counter_animation"))]
        # Remove animation content lines too
        new_landing = []
        skip_anim = False
        for ll in landing_lines:
            if ll.startswith("### Animation:"):
                skip_anim = True
                continue
            if skip_anim and (ll.startswith("## ") or ll.startswith("### ")):
                skip_anim = False
            if not skip_anim:
                new_landing.append(ll)
        landing_lines = new_landing
        all_lines = lines + interaction_lines + app_lines + landing_lines
        total_chars = sum(len(l) for l in all_lines)

    if is_hybrid and total_chars > _TOKEN_CEILING_CHARS:
        # Cut 3: remove data viz rules
        new_app = []
        skip_viz = False
        for al in app_lines:
            if al.startswith("## Data Visualization"):
                skip_viz = True
                continue
            if skip_viz and al.startswith("## "):
                skip_viz = False
            if not skip_viz:
                new_app.append(al)
        app_lines = new_app
        all_lines = lines + interaction_lines + app_lines + landing_lines
        total_chars = sum(len(l) for l in all_lines)

    if is_hybrid and total_chars > _TOKEN_CEILING_CHARS:
        # Cut 4: reduce recipes to 10
        recipes = design_data.get("component_recipes", {})
        new_app = ["## Component Recipes (use these exact classes)"]
        for rname in _APP_RECIPES_REDUCED:
            if rname in recipes:
                new_app.append(_fmt_recipe(rname, recipes[rname]))
        new_app.append("")
        # Keep layouts and responsive rules from app_lines
        keep_app = []
        past_recipes = False
        for al in app_lines:
            if al.startswith("## Page Layouts") or al.startswith("## Responsive Rules"):
                past_recipes = True
            if past_recipes:
                keep_app.append(al)
        app_lines = new_app + keep_app
        all_lines = lines + interaction_lines + app_lines + landing_lines
        total_chars = sum(len(l) for l in all_lines)

    if is_hybrid and total_chars > _TOKEN_CEILING_CHARS:
        # Cut 5: remove interaction patterns
        interaction_lines = []
        all_lines = lines + app_lines + landing_lines
        total_chars = sum(len(l) for l in all_lines)

    if is_hybrid and total_chars > _TOKEN_CEILING_CHARS:
        # Cut 6: reduce CTA to button_text_rules only
        new_landing = []
        skip_cta_extra = False
        for ll in landing_lines:
            if ll.startswith("## CTA Formulas") or ll.startswith("## Micro-copy below"):
                skip_cta_extra = True
                continue
            if skip_cta_extra and ll.startswith("## "):
                skip_cta_extra = False
            if not skip_cta_extra:
                new_landing.append(ll)
        landing_lines = new_landing
        all_lines = lines + app_lines + landing_lines

    return "\n".join(all_lines)


def _build_design_context(project_desc="", out_selections=None, max_chars=None):
    """Build design context string. Optionally populate out_selections dict with computed values.

    Args:
        max_chars: If set, truncate result to approximately this many characters.
    """
    design_data = _load_design_json()
    if not design_data:
        return ""

    if not project_desc:
        # Fallback: midnight + technical + 5 basic recipes
        palettes = design_data.get("palettes", {})
        midnight = palettes.get("midnight", {})
        palette_vars = midnight.get("css_variables", {})
        typography = {"heading": "JetBrains Mono", "body": "Inter", "mono": "JetBrains Mono", "name": "technical"}
        selections = {
            "palette_name": "midnight", "palette_vars": palette_vars,
            "typography": typography, "page_types": [],
            "is_landing": False, "landing_framework": None,
        }
        if out_selections is not None:
            out_selections.update(selections)
        lines = ["# Design System Context", ""]
        lines.append("## Palette: midnight")
        for k, v in palette_vars.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"## Typography: technical")
        lines.append(f"  Heading: JetBrains Mono, Body: Inter, Mono: JetBrains Mono")
        lines.append("")
        recipes = design_data.get("component_recipes", {})
        lines.append("## Component Recipes (use these exact classes)")
        for rname in ("button", "input", "card", "sidebar", "header"):
            if rname in recipes:
                lines.append(_fmt_recipe(rname, recipes[rname]))
        lines.append("")
        anti_slop = design_data.get("anti_slop_rules", [])
        if anti_slop:
            lines.append("## Anti-Slop Rules (MUST follow)")
            for r in anti_slop:
                lines.append(f"- {r}")
            lines.append("")
        result = "\n".join(lines)
        return result[:max_chars] if max_chars and len(result) > max_chars else result

    palette_name, palette_vars = _select_palette(project_desc, design_data)
    typography = _select_typography(project_desc, design_data)
    page_types = _detect_page_types(project_desc)
    is_landing = _is_landing_page(project_desc)
    landing_framework = _detect_landing_framework(project_desc) if is_landing else None

    selections = {
        "palette_name": palette_name, "palette_vars": palette_vars,
        "typography": typography, "page_types": page_types,
        "is_landing": is_landing, "landing_framework": landing_framework,
    }
    if out_selections is not None:
        out_selections.update(selections)

    result = _select_design_subset(
        project_desc, design_data, palette_name, palette_vars,
        typography, page_types, is_landing, landing_framework
    )
    return result[:max_chars] if max_chars and len(result) > max_chars else result


# ---------------------------------------------------------------------------
# Security context builder — profile detection + context assembly
# ---------------------------------------------------------------------------

_SECURITY_PROFILE_KEYWORDS = [
    (["store", "shop", "ecommerce", "checkout", "cart", "payment"], "ecommerce"),
    (["saas", "dashboard", "admin", "crm", "platform", "management"], "saas"),
    (["api", "rest", "graphql", "backend", "service", "microservice"], "api"),
    (["blog", "cms", "content", "editorial", "magazine"], "blog"),
]


def _detect_security_profile(project_desc):
    """Detect security profile from project description. Returns profile name string."""
    if not project_desc:
        return "saas"
    desc_lower = project_desc.lower()
    for keywords, profile in _SECURITY_PROFILE_KEYWORDS:
        if any(kw in desc_lower for kw in keywords):
            return profile
    return "saas"


_SECURITY_TOKEN_CEILING_CHARS = 12000


def _detect_project_framework(project_dir=None):
    """Detect project framework from config files and dependencies.

    Returns framework name string: "nextjs", "nuxt", "vite", "sveltekit", "remix",
    "angular", "express", "fastify", "hono", "koa", "django", "fastapi", "flask", or "unknown".
    """
    pdir = Path(project_dir) if project_dir else Path(CWD)

    # Config file detection (highest confidence)
    config_markers = [
        ("next.config.*", "nextjs"),
        ("nuxt.config.*", "nuxt"),
        ("svelte.config.js", "sveltekit"),
        ("svelte.config.ts", "sveltekit"),
        ("remix.config.js", "remix"),
        ("remix.config.ts", "remix"),
        ("angular.json", "angular"),
        ("vite.config.*", "vite"),
    ]
    for pattern, framework in config_markers:
        if list(pdir.glob(pattern)):
            return framework

    # package.json dependency detection
    pkg_json = pdir / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            all_deps = {}
            all_deps.update(pkg.get("dependencies", {}))
            all_deps.update(pkg.get("devDependencies", {}))
            dep_markers = [
                ("next", "nextjs"),
                ("nuxt", "nuxt"),
                ("@sveltejs/kit", "sveltekit"),
                ("@remix-run/node", "remix"),
                ("@angular/core", "angular"),
                ("express", "express"),
                ("fastify", "fastify"),
                ("hono", "hono"),
                ("koa", "koa"),
            ]
            for dep, framework in dep_markers:
                if dep in all_deps:
                    return framework
            # Vite fallback (after checking frameworks that use vite internally)
            if "vite" in all_deps:
                return "vite"
        except Exception:
            pass

    # Python framework detection
    if (pdir / "manage.py").exists():
        return "django"
    for req_file in ["requirements.txt", "pyproject.toml", "Pipfile"]:
        req_path = pdir / req_file
        if req_path.exists():
            try:
                content = req_path.read_text(encoding="utf-8").lower()
                if "fastapi" in content:
                    return "fastapi"
                if "flask" in content:
                    return "flask"
                if "django" in content:
                    return "django"
            except Exception:
                pass

    return "unknown"


def _build_security_context(project_desc="", scaffold_mode=False, framework=None):
    """Build security context string for system prompt injection. ~3K token ceiling.

    Args:
        project_desc: Project description for profile detection.
        scaffold_mode: If True, use tighter budget (~3KB).
        framework: Detected framework name (e.g. "nextjs", "express"). Auto-selects header config.
    """
    sec_data = _load_security_json()
    if not sec_data:
        return ""

    ceiling = 6000 if scaffold_mode else _SECURITY_TOKEN_CEILING_CHARS
    profile = _detect_security_profile(project_desc)
    lines = ["# Security Enforcement Context", ""]

    # Security rules (always included — non-negotiable core)
    rules = sec_data.get("security_rules", [])
    if rules:
        lines.append("## Security Rules (MUST follow)")
        for r in rules:
            lines.append(f"- {r}")
        lines.append("")

    # Header config — select by framework
    fw = framework or "nextjs"
    header_key = fw if fw in sec_data.get("header_configs", {}) else "nextjs"
    header_cfg = sec_data.get("header_configs", {}).get(header_key, {})
    headers = header_cfg.get("headers", [])
    if headers:
        lines.append(f"## Security Headers ({header_key})")
        for h in headers:
            lines.append(f"- {h['key']}: {h['value']}")
            if h.get("description") and not scaffold_mode:
                lines.append(f"  → {h['description']}")
        lines.append("")

    # Project-type specific middleware
    type_sec = sec_data.get("project_type_security", {}).get(profile, {})
    required_mw = type_sec.get("required_middleware", [])
    recipes = sec_data.get("middleware_recipes", {})
    if required_mw:
        lines.append(f"## Required Middleware (profile: {profile})")
        total_chars = sum(len(l) for l in lines)
        for mw_name in required_mw:
            recipe = recipes.get(mw_name, {})
            if not recipe:
                continue
            lines.append(f"### {recipe.get('name', mw_name)}")
            lines.append(f"{recipe.get('description', '')}")
            # Include code only if under ceiling
            code = recipe.get("code", "")
            if code and total_chars + len(code) < ceiling:
                lines.append(f"```typescript\n{code}\n```")
                total_chars += len(code)
            lines.append("")

    # Extra rules for ecommerce
    extra_rules = type_sec.get("extra_rules", [])
    if extra_rules:
        lines.append("## Additional Security Rules")
        for r in extra_rules:
            lines.append(f"- {r}")
        lines.append("")

    # Validation patterns (trim to names only if over ceiling)
    val_patterns = sec_data.get("validation_patterns", {})
    if val_patterns:
        total_chars = sum(len(l) for l in lines)
        if total_chars < ceiling:
            lines.append("## Validation Patterns")
            for vname, vdata in val_patterns.items():
                if total_chars < ceiling - 200:
                    lines.append(f"### {vdata.get('name', vname)}")
                    lines.append(f"  Regex: `{vdata.get('regex', '')}`")
                    for rule in vdata.get("rules", []):
                        lines.append(f"  - {rule}")
                else:
                    lines.append(f"- {vdata.get('name', vname)}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt section registry — modular prompt assembly
# ---------------------------------------------------------------------------

# Module-level cache — intentionally not thread-safe. Single-process CLI tool.
_prompt_registry_cache = None
_prompt_section_cache = {}

MAX_SCAFFOLD_CONTEXT = 25000  # 25 KB hard ceiling for scaffold mode


def _load_prompt_registry():
    """Load and cache prompts/registry.json."""
    global _prompt_registry_cache
    if _prompt_registry_cache is not None:
        return _prompt_registry_cache
    for pdir in PROMPTS_DIR_PATHS:
        reg_file = pdir / "registry.json"
        if reg_file.exists():
            try:
                data = json.loads(reg_file.read_text(encoding="utf-8"))
                _prompt_registry_cache = data
                return data
            except Exception:
                pass
    return {"sections": {}}


def _load_prompt_section(name):
    """Load a single prompt section .md file. Cached."""
    if name in _prompt_section_cache:
        return _prompt_section_cache[name]
    for pdir in PROMPTS_DIR_PATHS:
        section_file = pdir / f"{name}.md"
        if section_file.exists():
            try:
                content = section_file.read_text(encoding="utf-8")
                _prompt_section_cache[name] = content
                return content
            except Exception:
                pass
    return ""


def _build_prompt_for_mode(mode, keywords=None, token_budget=None):
    """Assemble system prompt from registry sections filtered by mode.

    Args:
        mode: "conversation" or "scaffold"
        keywords: Optional list of keywords to filter keyword-gated sections
        token_budget: Optional max tokens (~4 chars/token). None = no limit.
    Returns:
        Assembled prompt string with dynamic placeholders resolved.
    """
    registry = _load_prompt_registry()
    sections = registry.get("sections", {})

    # Filter by mode. "all" = every known mode.
    ALL_MODES = {"conversation", "scaffold"}
    eligible = []
    for name, meta in sections.items():
        modes = meta.get("modes", [])
        if mode in modes or ("all" in modes and mode in ALL_MODES):
            eligible.append((name, meta))

    # Sort by priority (lower = earlier)
    eligible.sort(key=lambda x: x[1].get("priority", 50))

    # Assemble with budget tracking
    parts = []
    total_chars = 0
    char_budget = (token_budget * 4) if token_budget else None  # ~4 chars/token

    for name, meta in eligible:
        content = None
        condensed_name = meta.get("condensed")  # e.g. "security_rules_condensed"

        # Try condensed variant first if under budget pressure
        if char_budget and condensed_name:
            content = _load_prompt_section(condensed_name)
        if not content:
            content = _load_prompt_section(name)
        if not content:
            continue

        # Per-section size limit — skip (or try condensed) if over budget
        max_section_chars = meta.get("max_tokens", 1000) * 4
        if len(content) > max_section_chars:
            if condensed_name:
                content = _load_prompt_section(condensed_name)
            if not content or len(content) > max_section_chars:
                continue  # Skip — garbled truncation is worse than omission

        # Overall budget check
        if char_budget and total_chars + len(content) > char_budget:
            break

        parts.append(content)
        total_chars += len(content)

    result = "\n\n".join(parts)

    # Substitute dynamic placeholders
    result = result.replace("{{CWD}}", CWD)
    result = result.replace("{{PLATFORM}}", sys.platform)
    result = result.replace("{{DATE}}", __import__('datetime').date.today().isoformat())

    # Runtime context guard
    est_tokens = len(result) / 3.3
    if mode == "scaffold" and len(result) > MAX_SCAFFOLD_CONTEXT:
        print(f"  Warning: Scaffold prompt {len(result)} chars (~{int(est_tokens)} tokens) exceeds {MAX_SCAFFOLD_CONTEXT} ceiling")

    return result


def build_system_prompt():
    # Load modular sections for conversation mode
    prompt = _build_prompt_for_mode("conversation")

    # --- inject CLAW.md project instructions ---
    claw_files = load_claw_md()
    if claw_files:
        prompt += "\n\n# Project Instructions (from CLAW.md)\n"
        for path, content in claw_files:
            prompt += f"\n## {path}\n{content}\n"

    # --- inject hot memories ---
    memories = load_memories_for_context()
    if memories:
        prompt += "\n\n# Hot Memories (auto-injected, frequently accessed)\n"
        prompt += memories + "\n"

    # --- inject active plan ---
    plan_content = load_active_plan()
    if plan_content:
        prompt += "\n\n# Active Plan (from PLAN.md)\n"
        prompt += "You have an active plan. When the user asks you to continue or execute, follow the next unchecked step.\n"
        prompt += plan_content + "\n"

    # --- inject available templates ---
    templates = list_templates()
    if templates:
        prompt += "\n\n# Available Project Templates\n"
        prompt += "Use these templates as starting points instead of generating from scratch:\n"
        for tname, tinfo in templates.items():
            prompt += f"- **{tname}**: {tinfo.get('description', '')} (stack: {', '.join(tinfo.get('stack', []))})\n"
        prompt += "\nTo scaffold: the --scaffold flag has already been used, or suggest the user run `claw --scaffold <template> \"project description\"`\n"

    # --- inject API registry summary ---
    registry = load_api_registry()
    if registry.get("apis"):
        prompt += "\n\n# API Pattern Registry\n"
        prompt += "CORRECT API patterns are available for these services. ALWAYS use these patterns instead of guessing:\n"
        for api_name, api_info in registry["apis"].items():
            prompt += f"- **{api_name}**: {api_info.get('description', '')}\n"
        prompt += "\nThe system will inject relevant API patterns when it detects you need them.\n"

    # --- inject detected project profile (enhanced from detect_project_type) ---
    profile = _get_cached_profile()
    if profile and profile.base_info["type"] != "unknown":
        prompt += profile.to_prompt_injection()
        proj_info = profile.base_info
        # Additional details from base_info
        if proj_info.get("has_typescript"):
            prompt += f"\n- TypeScript: yes"
        if proj_info.get("has_prisma"):
            prompt += f"\n- ORM: Prisma (run `npx prisma generate` after install)"
        if proj_info.get("has_docker"):
            prompt += f"\n- Docker: Dockerfile detected"
        if proj_info["type"] == "static":
            entry = proj_info.get("entry_file", "index.html")
            prompt += f"\n- This is a STATIC HTML/CSS/JS project. Do NOT run npm, pip, or any package managers."
            prompt += f"\n- To test: `start {entry}` (Windows) or `open {entry}` (Mac)."
        elif proj_info["type"] == "node":
            prompt += f"\n- Build pipeline: {' → '.join(name for name, _ in proj_info['commands'])}"
            if proj_info.get("dev_cmd"):
                prompt += f"\n- Dev server: `{proj_info['dev_cmd']}`"
        elif proj_info["type"] == "python":
            prompt += f"\n- Build pipeline: {' → '.join(name for name, _ in proj_info['commands'])}"
            if proj_info.get("dev_cmd"):
                prompt += f"\n- Dev server: `{proj_info['dev_cmd']}`"
        # Env status
        if proj_info.get("has_env_example") and not proj_info.get("has_env"):
            prompt += f"\n- WARNING: .env.example exists but no .env — create it before running"
        prompt += "\n"
    else:
        # Check for HTML-only projects
        html_files = list(Path(CWD).glob("*.html"))
        if html_files and not (Path(CWD) / "package.json").exists():
            prompt += f"\n\n# Detected Project Type\n"
            prompt += f"- Type: **static** (HTML files present, no package.json)\n"
            prompt += f"- Do NOT run npm, pip, or any package managers.\n"
            prompt += f"- To test: open the HTML file in a browser.\n"

    # --- inject git context (Phase 3b) ---
    git_ctx = _capture_git_context()
    if git_ctx:
        prompt += f"\n\n{git_ctx}\n"

    # --- inject codebase map so model knows the project structure (cached) ---
    if not hasattr(_build_codebase_map, "_cache") or _build_codebase_map._cache_cwd != CWD:
        _build_codebase_map._cache = _build_codebase_map(CWD, max_tokens=1200)
        _build_codebase_map._cache_cwd = CWD
    codebase_map = _build_codebase_map._cache
    if codebase_map:
        prompt += f"\n\n{codebase_map}\n"

    # --- inject graph context (Phase 2) ---
    # Seed sources: git modified files (available at prompt-build time after Phase 3b)
    _graph_seeds = list(_git_context_seeds)
    try:
        _get_project_graph()  # eagerly build graph at session start
    except Exception:
        pass
    if _graph_seeds and _project_graph is not None:
        ctx_size = 32768  # default; actual size resolved per-turn
        try:
            ctx_size = _get_model_context_size(DEFAULT_MODEL)
        except Exception:
            pass
        graph_budget = _graph_token_budget(ctx_size)
        graph_ctx = _build_graph_context(_graph_seeds, max_tokens=graph_budget)
        if graph_ctx:
            prompt += f"\n\n{graph_ctx}\n"

    # --- inject design snippets (Lead-to-Gold cheat sheet) ---
    prompt += _inject_design_snippets()

    return prompt

# ---------------------------------------------------------------------------
# build user message with attachments
# ---------------------------------------------------------------------------

def build_user_message(text, attachments, vision_model):
    """
    Build an Ollama message dict from user text + attachments.
    Returns (message_dict, model_to_use).

    - Images: sent as base64 via Ollama 'images' field, switches to vision model
    - PDFs/text: content prepended to the message text
    - Video: metadata prepended
    """
    use_model = None  # None = keep current
    images_b64 = []
    extra_context = []

    for att in attachments:
        if att.error:
            continue

        if att.kind == "image" and att.b64_image:
            images_b64.append(att.b64_image)
            use_model = vision_model  # need vision model for images

        elif att.content:
            label = att.name
            if att.kind == "pdf":
                label = f"{att.name} (PDF content)"
            extra_context.append(f"--- Attached file: {label} ---\n{att.content}\n--- End of {att.name} ---")

    # build message text
    full_text = text
    if extra_context:
        full_text = "\n\n".join(extra_context) + "\n\n" + text

    msg = {"role": "user", "content": full_text}

    if images_b64:
        msg["images"] = images_b64

    return msg, use_model

# ---------------------------------------------------------------------------
# parse @file references from user input
# ---------------------------------------------------------------------------

def parse_at_references(text):
    """
    Find @path references in text. Returns (cleaned_text, list_of_paths).
    Handles @path/to/file and @"path with spaces/file.txt"
    """
    attachments = []
    # match @"quoted path" or @unquoted_path
    pattern = r'@"([^"]+)"|@(\S+)'

    def replacer(m):
        path = m.group(1) or m.group(2)
        resolved = _resolve(path)
        if resolved.exists():
            attachments.append(path)
            return ""  # remove from text
        return m.group(0)  # keep in text if file doesn't exist

    cleaned = re.sub(pattern, replacer, text).strip()
    # collapse multiple spaces
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned, attachments

# ---------------------------------------------------------------------------
# rescue tool calls from text output (for weaker models)
# ---------------------------------------------------------------------------

TOOL_NAMES = set(TOOL_HANDLERS.keys())

def rescue_tool_calls_from_text(text):
    """
    Some local models output tool calls as JSON in text instead of using
    the proper tool_calls format. This detects and extracts them.
    Returns (cleaned_text, list_of_tool_calls) or (text, []) if none found.
    """
    rescued = []
    cleaned = text

    # Pre-process: local models sometimes use backticks instead of quotes
    # for string values (e.g. "content": `code here`). Convert to valid JSON.
    text = re.sub(r'`([^`]*)`', lambda m: json.dumps(m.group(1)), text)

    # strategy 1: extract JSON from ```json ... ``` blocks
    code_blocks = re.findall(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        block = block.strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                name = obj["name"]
                args = obj["arguments"]
                if name in TOOL_NAMES:
                    rescued.append({"function": {"name": name, "arguments": args}})
                    # remove the entire code block from the text
                    cleaned = re.sub(
                        r'```(?:json)?\s*\n' + re.escape(block).replace(r'\ ', r'\s*') + r'\s*```',
                        '', cleaned, count=1
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # if code block extraction didn't work, try to remove the block markers
    # and parse the biggest JSON object we can find
    if not rescued:
        # remove markdown code fences and try to find JSON with "name" and "arguments"
        stripped = re.sub(r'```(?:json)?', '', text)
        stripped = stripped.replace('```', '')
        # find all { ... } that contain "name" -- use a bracket-counting approach
        for match in re.finditer(r'\{', stripped):
            start = match.start()
            depth = 0
            end = start
            for i in range(start, len(stripped)):
                if stripped[i] == '{':
                    depth += 1
                elif stripped[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                candidate = stripped[start:end]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    # Fallback: strip trailing commas before } or ] and retry
                    try:
                        cleaned_candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                        obj = json.loads(cleaned_candidate)
                    except (json.JSONDecodeError, ValueError):
                        continue
                try:
                    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                        name = obj["name"]
                        args = obj["arguments"]
                        if name in TOOL_NAMES:
                            rescued.append({"function": {"name": name, "arguments": args}})
                            break
                except (KeyError, TypeError):
                    continue

    if rescued:
        # clean out the JSON blocks from visible text
        cleaned = re.sub(r'```(?:json)?\s*\n.*?```', '', text, flags=re.DOTALL)
        cleaned = cleaned.strip()

    return cleaned, rescued


def rescue_question_from_text(text):
    """
    Detect when the model asks a question as plain text instead of using ask_user.
    Looks for patterns like:
      - "Which file would you like me to read?"
      - "Would you like me to X or Y?"
      - Lines ending with ? followed by numbered/bulleted options
      - "please specify" / "please choose"

    Returns (question, choices, cleaned_text) or None if no question detected.
    """
    lines = text.strip().split("\n")
    if not lines:
        return None

    # pattern 1: find a question mark line + numbered/bulleted options after it
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "?" not in stripped:
            continue

        # look for options after the question
        choices = []
        j = i + 1
        while j < len(lines):
            opt_line = lines[j].strip()
            # match: "1. thing", "- thing", "* thing", "1) thing", "a. thing", "a) thing"
            m = re.match(r'^(?:\d+[.)]\s*|-\s*|\*\s*|[a-z][.)]\s*)(.+)', opt_line)
            if m:
                choices.append(m.group(1).strip().rstrip('.'))
                j += 1
            elif not opt_line:
                j += 1  # skip blank lines between options
            else:
                break

        if len(choices) >= 2:
            question = stripped
            # clean the options + question from text
            cleaned_lines = lines[:i] + lines[j:]
            cleaned = "\n".join(cleaned_lines).strip()
            return (question, choices, cleaned)

    # pattern 2: "X or Y?" at the end -- extract inline choices
    last_line = lines[-1].strip()
    if last_line.endswith("?"):
        # look for "A or B" pattern -- split on common delimiters before "or"
        # match: "use React or Vue?", "read file.html or file.py?"
        # also: "A, B, or C?"
        or_match = re.search(r'[:;]\s*[`"]?(.+?)[`"]?\s+or\s+[`"]?(.+?)[`"]?\s*\??$', last_line)
        if not or_match:
            # fallback: grab the last "X or Y" -- use the shortest match for X
            or_match = re.search(r'\s[`"]?([\w][\w./-]*(?:\s[\w./-]+){0,3}?)[`"]?\s+or\s+[`"]?([\w][\w./-]*(?:\s[\w./-]+){0,3}?)[`"]?\s*\??$', last_line)
        if or_match:
            c1 = or_match.group(1).strip().rstrip('.,;')
            c2 = or_match.group(2).strip().rstrip('.,;?')
            # check for "A, B, or C" -- split c1 on commas
            all_choices = []
            if "," in c1:
                parts = [p.strip().strip('`"') for p in c1.split(",")]
                all_choices.extend(p for p in parts if p)
            else:
                all_choices.append(c1.strip('`"'))
            all_choices.append(c2.strip('`"'))
            # trim common leading verb phrases from choices
            trim_prefixes = [
                "i use ", "you want ", "i read ", "we use ", "i create ",
                "i should ", "we should ", "i pick ", "you prefer ",
                "i choose ", "you like ", "i write ", "i make ",
            ]
            cleaned_choices = []
            for c in all_choices:
                cl = c
                for prefix in trim_prefixes:
                    if cl.lower().startswith(prefix):
                        cl = cl[len(prefix):]
                        break
                cleaned_choices.append(cl.strip())

            if len(cleaned_choices) >= 2:
                question = last_line
                cleaned = "\n".join(lines[:-1]).strip()
                return (question, cleaned_choices, cleaned)

    # pattern 3: "please specify" / "please choose" / "which one" without choices
    # just present it as an open question
    lower = text.lower()
    question_signals = ["please specify", "please choose", "which one", "which file",
                        "would you like me to", "should i", "do you want me to"]
    for sig in question_signals:
        if sig in lower and "?" in text:
            # find the question line
            for line in reversed(lines):
                if "?" in line:
                    question = line.strip()
                    cleaned = "\n".join(l for l in lines if l.strip() != question).strip()
                    return (question, [], cleaned)

    return None


# ---------------------------------------------------------------------------
# inner monologue & self-reflection
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)

def _extract_think_block(text):
    """Extract <think>...</think> block from model output.
    Returns (thinking_text, cleaned_text). If no match, returns ("", text)."""
    m = _THINK_RE.search(text)
    if not m:
        return ("", text)
    thinking_text = m.group(1).strip()
    cleaned = text[:m.start()] + text[m.end():]
    cleaned = cleaned.strip()
    return (thinking_text, cleaned)


def _display_thinking(thinking_text):
    """Render inner monologue in purple with borders. Max 6 lines shown."""
    if not THINKING_ENABLED or not thinking_text:
        return
    lines = thinking_text.split("\n")
    border = f"{C.THINK_DIM}{'─' * 40}{C.RESET}"
    print(f"\n  {border}")
    show_lines = lines[:5] if len(lines) > 6 else lines
    for line in show_lines:
        print(f"  {C.THINK}{line}{C.RESET}")
    if len(lines) > 6:
        print(f"  {C.THINK_DIM}... +{len(lines) - 5} more lines{C.RESET}")
    print(f"  {border}\n")


def _action_signature(tool_name, tool_args):
    """Create a short signature for a tool action for repetition detection."""
    if tool_name == "bash":
        cmd = tool_args.get("command", "")
        # Normalize: take first 60 chars of command
        return cmd.strip()[:60]
    elif tool_name == "write_file":
        return tool_args.get("file_path", "")
    elif tool_name == "edit_file":
        return tool_args.get("file_path", "")
    elif tool_name == "read_file":
        return tool_args.get("file_path", "")
    elif tool_name == "glob_search":
        return tool_args.get("pattern", "")
    elif tool_name == "grep_search":
        return tool_args.get("pattern", "")
    else:
        # Generic: use first arg value
        for v in tool_args.values():
            if isinstance(v, str):
                return v[:40]
        return ""


class _RepetitionDetector:
    """Detect when the agent is stuck in a loop repeating the same actions."""

    def __init__(self, threshold=ANTIREPEAT_THRESHOLD):
        self.threshold = threshold
        self.action_history = deque(maxlen=12)       # tool_name:key_arg signatures
        self.response_hashes = deque(maxlen=6)        # hashes of text responses
        self.write_content_hashes = {}                # {filepath: [content_hashes]}

    def record_action(self, tool_name, tool_args):
        key_arg = _action_signature(tool_name, tool_args)
        self.action_history.append(f"{tool_name}:{key_arg}")
        # Track write content hashes for semantic write detection
        if tool_name == "write_file":
            fp = tool_args.get("file_path", "")
            content = tool_args.get("content", "")
            h = hash(content[:500])
            self.write_content_hashes.setdefault(fp, []).append(h)

    def record_response(self, text):
        self.response_hashes.append(hash(text.strip()[:200]))

    def check(self):
        """Check for repetition patterns. Returns warning message or None."""
        from collections import Counter

        # Check 1: Same action signature repeated N+ times in last 6
        recent = list(self.action_history)[-6:]
        counts = Counter(recent)
        for action, count in counts.items():
            if count >= self.threshold:
                return f"[SYSTEM: REPETITION DETECTED — '{action}' repeated {count}x in last 6 actions. Try a DIFFERENT approach or ask the user for help.]"

        # Check 2: Same write content to same file (semantic loop)
        for fp, hashes in self.write_content_hashes.items():
            if len(hashes) >= 2:
                recent_hashes = hashes[-3:]
                if len(set(recent_hashes)) == 1 and len(recent_hashes) >= 2:
                    return f"[SYSTEM: SEMANTIC LOOP — writing nearly identical content to '{fp}' repeatedly. The approach is not working. Try a fundamentally different strategy or ask the user.]"

        # Check 3: Near-identical text responses
        recent_resp = list(self.response_hashes)[-3:]
        if len(recent_resp) >= 3 and len(set(recent_resp)) == 1:
            return "[SYSTEM: You've given nearly identical responses 3 times. Change your approach or ask the user what they want.]"

        return None


class _ReflectionState:
    """Track tool execution state to decide when to inject reflection prompts."""

    def __init__(self):
        self.tool_rounds = 0
        self.consecutive_errors = 0
        self.last_results_succeeded = deque(maxlen=6)
        self.thinking_nudge_sent = False

    def record_tool_result(self, result_text):
        self.tool_rounds += 1
        is_error = any(sig in result_text for sig in [
            "Error:", "FAILED", "SYNTAX ERROR", "error:", "Permission denied"
        ])
        self.last_results_succeeded.append(not is_error)
        if is_error:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

    def should_reflect(self):
        if not REFLECTION_ENABLED:
            return False
        # Conditional reflection: fire on consecutive failures
        if self.consecutive_errors >= 2:
            return True
        # Cadence-based but only if not all-success
        if self.tool_rounds > 0 and self.tool_rounds % REFLECTION_AFTER_N == 0:
            recent = list(self.last_results_succeeded)[-REFLECTION_AFTER_N:]
            if not all(recent):
                return True
        return False

    def build_reflection_prompt(self):
        prompt = "[SYSTEM: Pause and reflect before continuing.\n"
        prompt += "- Am I making progress toward the user's goal?\n"
        prompt += "- Did the last action succeed? If not, why?\n"
        prompt += "- Should I change approach, or am I on track?\n"
        if not self.thinking_nudge_sent and THINKING_ENABLED:
            prompt += "- Remember to wrap your reasoning in <think>...</think> tags.\n"
            self.thinking_nudge_sent = True
        prompt += "Think carefully, then proceed with your next action.]"
        return prompt


# ---------------------------------------------------------------------------
# error recovery helpers
# ---------------------------------------------------------------------------

def _is_garbage_output(text):
    """Detect garbage/empty model output or multilingual gibberish."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if len(stripped) < 5:
        return True
    # Check if mostly non-alphabetic (garbled)
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 10 and alpha_count / len(stripped) < 0.2:
        return True
    # Check for repeated characters (like "aaaaaaa" or "......")
    if len(set(stripped.replace(" ", ""))) <= 2 and len(stripped) > 10:
        return True
    # Detect mixed-language garbage (Chinese/Greek/Thai etc. leaked reasoning)
    if len(stripped) > 30:
        non_ascii = sum(1 for c in stripped if ord(c) > 127)
        ratio = non_ascii / len(stripped)
        if ratio > 0.15:
            return True
    return False


def _suggest_bash_alternative(cmd):
    """Suggest an alternative approach when a bash command keeps failing."""
    cmd_lower = cmd.lower().strip()
    suggestions = {
        "npm run build": "Try running `npx tsc --noEmit` first to isolate TypeScript errors.",
        "npm run dev": "Try `npm install` first, then check for port conflicts with `npx kill-port 3000`.",
        "npm install": "Try `npm install --legacy-peer-deps` or delete node_modules and package-lock.json first.",
        "pip install": "Try `pip install --user` or use a virtual environment.",
        "python manage.py migrate": "Try `python manage.py makemigrations` first.",
    }
    for pattern, suggestion in suggestions.items():
        if pattern in cmd_lower:
            return suggestion
    return "Read the error output carefully. Fix the underlying issue in the source code before retrying."


# ---------------------------------------------------------------------------
# tool error hints for better recovery
# ---------------------------------------------------------------------------

TOOL_ERROR_HINTS = {
    "bash": {
        "command not found": "Install it or use an alternative tool/command.",
        "permission denied": "Check file permissions or use sudo if appropriate.",
        "no such file or directory": "Verify the path exists. Use glob_search or read_file to confirm.",
        "connection refused": "The service isn't running. Start it first.",
        "already in use": "Port conflict — kill the process using the port or use a different port.",
        "syntax error": "Check the shell syntax. Ensure proper quoting and escaping.",
        "killed": "Process was killed (likely OOM). Try a lighter approach.",
    },
    "edit_file": {
        "not found in file": "Use read_file first to see the exact current content, then match it precisely.",
        "no such file": "The file doesn't exist. Use write_file to create it instead.",
        "multiple matches": "The old_str matches multiple locations. Include more surrounding context to be unique.",
    },
    "write_file": {
        "no such file or directory": "Create the parent directory first with bash: mkdir -p parent/dir",
        "permission denied": "Check file permissions on the target path.",
        "is a directory": "The path points to a directory, not a file. Add a filename.",
    },
    "read_file": {
        "no such file": "File doesn't exist. Use glob_search to find similar files.",
        "is a directory": "Use glob_search or bash ls to list directory contents instead.",
    },
}


def _get_error_hint(tool_name, error_text):
    """Match error text against known patterns and return a hint."""
    hints = TOOL_ERROR_HINTS.get(tool_name, {})
    error_lower = error_text.lower()
    for pattern, hint in hints.items():
        if pattern in error_lower:
            return hint
    return ""


# ---------------------------------------------------------------------------
# parallel tool execution
# ---------------------------------------------------------------------------

PARALLEL_SAFE_TOOLS = frozenset({
    "read_file", "glob_search", "grep_search", "web_search",
    "web_fetch", "memory_search", "db_schema", "env_manage",
})


def _execute_single_tool(tool_name, tool_args):
    """Execute a single tool call and return the result string."""
    return execute_tool(tool_name, tool_args)


def _execute_tools_parallel(tool_calls):
    """Execute tool calls, running parallel-safe tools concurrently."""
    # Parse all tool calls first
    parsed = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None  # signal parse failure
        parsed.append((tc, name, args))

    parallel = [(tc, n, a) for tc, n, a in parsed if n in PARALLEL_SAFE_TOOLS and a is not None]
    sequential = [(tc, n, a) for tc, n, a in parsed if (tc, n, a) not in parallel]

    results = []  # list of (tc, name, args, result_str)

    # Run parallel-safe tools concurrently
    if len(parallel) > 1:
        print(f"  {C.SUBTLE}running {len(parallel)} tools in parallel...{C.RESET}")
        with ThreadPoolExecutor(max_workers=min(4, len(parallel))) as pool:
            futures = {
                pool.submit(_execute_single_tool, n, a): (tc, n, a)
                for tc, n, a in parallel
            }
            for f in as_completed(futures):
                tc, n, a = futures[f]
                try:
                    results.append((tc, n, a, f.result()))
                except Exception as e:
                    results.append((tc, n, a, f"Error executing {n}: {e}"))
    else:
        for tc, n, a in parallel:
            results.append((tc, n, a, _execute_single_tool(n, a)))

    # Run sequential tools in order
    for tc, n, a in sequential:
        if a is None:
            results.append((tc, n, {}, None))  # signal JSON parse failure
        else:
            results.append((tc, n, a, _execute_single_tool(n, a)))

    return results


# ---------------------------------------------------------------------------
# unified retry + display helpers
# ---------------------------------------------------------------------------

def _apply_retry_and_hints(tool_name, tool_args, result, tool_retry_counts, bash_failed_this_round):
    """Apply unified retry logic and error hints to a tool result. Returns (result, bash_failed_this_round)."""
    is_error = False

    # Bash failure detection
    if tool_name == "bash" and _check_bash_result(result):
        is_error = True
        cmd_key = ("bash", tool_args.get("command", "")[:100])
        retry_count = tool_retry_counts.get(cmd_key, 0)
        tool_retry_counts[cmd_key] = retry_count + 1

        # Get structured hint from error patterns
        hint = _get_error_hint("bash", result)
        hint_text = f" Hint: {hint}" if hint else ""

        if retry_count < 2:
            bash_failed_this_round = True
            print(f"  {C.WARNING}{BLACK_CIRCLE} bash failed — auto-retry {retry_count + 1}/2{C.RESET}")
            result += f"\n\n[SYSTEM: Command failed. Auto-retry {retry_count + 1}/2. Fix the error and try again.{hint_text}]"
        else:
            suggestion = _suggest_bash_alternative(tool_args.get("command", ""))
            print(f"  {C.WARNING}{BLACK_CIRCLE} bash failed {retry_count + 1} times — try different approach{C.RESET}")
            result += f"\n\n[SYSTEM: This command has failed {retry_count + 1} times. STOP retrying the same command. {suggestion}{hint_text}]"
            if retry_count >= 3:
                result += " Try a completely different approach."

    # File write/edit failure detection
    elif tool_name in ("write_file", "edit_file") and result.startswith("Error"):
        is_error = True
        file_key = (tool_name, tool_args.get("file_path", "")[:100])
        retry_count = tool_retry_counts.get(file_key, 0)
        tool_retry_counts[file_key] = retry_count + 1

        # Get structured hint
        hint = _get_error_hint(tool_name, result)
        hint_text = f" Hint: {hint}" if hint else ""

        history_ctx = ""
        if _edit_history:
            history_lines = [f"  {fp} ({op}): {preview}..." for fp, op, preview in _edit_history]
            history_ctx = f"\nRecent file operations:\n" + "\n".join(history_lines)

        if retry_count < 2:
            print(f"  {C.WARNING}{BLACK_CIRCLE} {tool_name} failed — auto-retry {retry_count + 1}/2{C.RESET}")
            result += f"\n\n[SYSTEM: {tool_name} failed. Auto-retry {retry_count + 1}/2. Read the error, fix the issue, and try again.{hint_text}{history_ctx}]"
        else:
            print(f"  {C.WARNING}{BLACK_CIRCLE} {tool_name} failed {retry_count + 1} times{C.RESET}")
            result += f"\n\n[SYSTEM: {tool_name} has failed {retry_count + 1} times. Try a different approach: read the target file first, then make a smaller change.{hint_text}{history_ctx}]"
            if retry_count >= 3:
                result += " Try a completely different approach."

    # Other tool errors — add hints if available
    elif result.startswith("Error"):
        hint = _get_error_hint(tool_name, result)
        if hint:
            result += f"\n\nHint: {hint}"

    return result, bash_failed_this_round


_ANSI_RE = re.compile(r'\033\[[0-9;]*[a-zA-Z]')

def _strip_ansi(text):
    """Strip ANSI escape codes from text (for line counting only)."""
    return _ANSI_RE.sub('', text)


def _display_tool_result(tool_name, result):
    """Display a tool result with tiered truncation and line-length awareness."""
    try:
        tw = os.get_terminal_size().columns
    except (OSError, ValueError):
        tw = 80
    max_line_w = tw - 8  # leave room for indent + margin

    is_error = result.startswith("Error")
    if is_error:
        preview = result[:300].replace("\n", " ")
        print(f"  {C.ERROR}{BLACK_CIRCLE} {tool_name} failed{C.RESET}")
        print(f"    {C.ERROR}{preview}{C.RESET}")
        return

    lines = result.split("\n")
    stripped_lines = [_strip_ansi(ln) for ln in lines]
    n = len(lines)
    total_chars = len(result)

    # Single long line — truncate to terminal width
    if n == 1 and len(stripped_lines[0]) > max_line_w:
        print(f"    {C.DIM}{lines[0][:max_line_w]}...{C.RESET}")
        return

    # Short: ≤3 lines, <200 chars, no line exceeds terminal width
    if n <= 3 and total_chars < 200 and all(len(sl) <= tw for sl in stripped_lines):
        for line in lines:
            print(f"    {C.DIM}{line}{C.RESET}")
        return

    # Medium: ≤15 lines — show all but truncate wide lines
    if n <= 15:
        for i, line in enumerate(lines):
            if len(stripped_lines[i]) > max_line_w:
                print(f"    {C.DIM}{line[:max_line_w]}...{C.RESET}")
            else:
                print(f"    {C.DIM}{line}{C.RESET}")
        return

    # Long: >15 lines — first 5 + last 2
    for i in range(min(5, n)):
        ln = lines[i]
        if len(stripped_lines[i]) > max_line_w:
            ln = ln[:max_line_w] + "..."
        print(f"    {C.DIM}{ln}{C.RESET}")
    hidden = n - 7
    print(f"    {C.SUBTLE}... {hidden} more lines ({total_chars} chars total){C.RESET}")
    for i in range(max(n - 2, 5), n):
        ln = lines[i]
        if len(stripped_lines[i]) > max_line_w:
            ln = ln[:max_line_w] + "..."
        print(f"    {C.DIM}{ln}{C.RESET}")


def _display_bash_result(command, result):
    """Display bash command result with code-style box and exit code parsing."""
    try:
        tw = os.get_terminal_size().columns
    except (OSError, ValueError):
        tw = 80
    max_line_w = tw - 8

    # Show command in code-style box
    cmd_display = command if len(command) <= max_line_w else command[:max_line_w - 3] + "..."
    print(f"    {C.DIM}${C.RESET} {C.TEXT}{cmd_display}{C.RESET}")

    # Parse exit code from result (our bash tool prepends it)
    exit_code = 0
    output = result
    if result.startswith("Error") or result.startswith("exit code"):
        # Extract exit code if present
        m = re.match(r'exit code (\d+)[:\n]', result)
        if m:
            exit_code = int(m.group(1))
            output = result[m.end():].lstrip('\n')
        elif result.startswith("Error"):
            exit_code = 1

    # Display exit code
    if exit_code == 0 and not result.startswith("Error"):
        print(f"    {C.SUCCESS}{BLACK_CIRCLE} exit 0{C.RESET}")
    else:
        print(f"    {C.ERROR}{BLACK_CIRCLE} exit {exit_code}{C.RESET}")

    if not output.strip():
        return

    lines = output.split("\n")
    stripped_lines = [_strip_ansi(ln) for ln in lines]
    n = len(stripped_lines)

    # Smart truncation: first 4 + last 2 for long outputs
    if n <= 8:
        for i, line in enumerate(lines):
            if len(stripped_lines[i]) > max_line_w:
                print(f"    {C.DIM}{line[:max_line_w]}...{C.RESET}")
            else:
                print(f"    {C.DIM}{line}{C.RESET}")
    else:
        for i in range(4):
            ln = lines[i]
            if len(stripped_lines[i]) > max_line_w:
                ln = ln[:max_line_w] + "..."
            print(f"    {C.DIM}{ln}{C.RESET}")
        hidden = n - 6
        print(f"    {C.SUBTLE}... {hidden} more lines ...{C.RESET}")
        for i in range(n - 2, n):
            ln = lines[i]
            if len(stripped_lines[i]) > max_line_w:
                ln = ln[:max_line_w] + "..."
            print(f"    {C.DIM}{ln}{C.RESET}")


# ---------------------------------------------------------------------------
# auto-install dependencies after package.json write
# ---------------------------------------------------------------------------

_pending_dep_installs = set()  # track package.json paths that need install
_dep_install_results = []  # queued results to inject after tool batch completes


def _check_auto_dep_install(tool_name, tool_args, result, messages):
    """After writing package.json, run npm install and queue the result message."""
    if tool_name not in ("write_file", "edit_file"):
        return
    if result.startswith("Error"):
        return
    fp = tool_args.get("file_path", "")
    if not fp:
        return
    fp_path = Path(fp)
    if fp_path.name != "package.json":
        return
    project_dir = str(fp_path.parent)
    if project_dir in _pending_dep_installs:
        return
    _pending_dep_installs.add(project_dir)
    print(f"  {C.TOOL}{BLACK_CIRCLE} {C.BOLD}auto-install{C.RESET} {C.SUBTLE}(npm install in {project_dir}){C.RESET}")
    try:
        # Use shell=True on Windows so npm.cmd is found via PATH
        proc = subprocess.run(
            "npm install",
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
            shell=True,
        )
        if proc.returncode == 0:
            added = re.findall(r'added (\d+) packages', proc.stdout + proc.stderr)
            pkg_count = added[0] if added else "?"
            print(f"    {C.SUCCESS}OK{C.RESET} {C.DIM}({pkg_count} packages){C.RESET}")
            # Append to the TOOL result so it stays in the tool message flow
            _dep_install_results.append(f"[npm install OK — {pkg_count} packages installed in {project_dir}. Dependencies ready.]")
        else:
            err_short = (proc.stderr or proc.stdout or "unknown error")[:500]
            print(f"    {C.WARNING}npm install failed:{C.RESET} {C.DIM}{err_short[:200]}{C.RESET}")
            _dep_install_results.append(f"[npm install FAILED in {project_dir}: {err_short[:300]}. Fix package.json and run npm install.]")
    except Exception as e:
        print(f"    {C.WARNING}npm install error: {e}{C.RESET}")


# ---------------------------------------------------------------------------
# scaffold fallback + inactivity timeout
# ---------------------------------------------------------------------------


def _check_scaffold_progress(project_dir, pre_files):
    """Compare current files vs snapshot, return (new_count, new_file_list)."""
    pdir = Path(project_dir)
    current = set()
    for fpath in pdir.rglob("*"):
        if not fpath.is_file():
            continue
        parts = fpath.parts
        if any(skip in parts for skip in ("node_modules", ".next", "dist", ".git")):
            continue
        try:
            current.add(str(fpath.relative_to(pdir)).replace("\\", "/"))
        except ValueError:
            pass
    new_files = sorted(current - pre_files)
    return len(new_files), new_files


def _snapshot_project_files(project_dir):
    """Take a snapshot of existing files in project (excluding node_modules etc)."""
    pdir = Path(project_dir)
    files = set()
    for fpath in pdir.rglob("*"):
        if not fpath.is_file():
            continue
        parts = fpath.parts
        if any(skip in parts for skip in ("node_modules", ".next", "dist", ".git")):
            continue
        try:
            files.add(str(fpath.relative_to(pdir)).replace("\\", "/"))
        except ValueError:
            pass
    return files


# ---------------------------------------------------------------------------
# Multi-agent scaffold for local models
# ---------------------------------------------------------------------------

def _scaffold_with_agents(project_desc, template_name, project_dir, model,
                          build_spec, design_selections=None):
    """
    Orchestrate scaffold using communicating agents.
    Each file gets its own focused model call with pre-injected dependency context.
    """
    pdir = Path(project_dir)
    src_dir = pdir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Initialize shared memory
    context_path = pdir / "CONTEXT.md"
    context_path.write_text(
        "## Created Files\n\n## Patterns\n"
        "- Supabase client: import { createClient } from '@/lib/supabase/server'\n"
        "- Auth check: const { data: { user } } = await supabase.auth.getUser()\n"
        "- API response: return NextResponse.json(data)\n"
        "- Error response: return NextResponse.json({ error: message }, { status: code })\n",
        encoding="utf-8"
    )

    # Build file list (deterministic — no model call needed)
    entities = _extract_entities_from_desc(project_desc)
    is_team = _is_team_app(project_desc)
    file_list = _extract_scaffold_file_list(entities, project_desc, is_team)
    print(f"  {C.SUBTLE}{BLACK_CIRCLE} {len(file_list)} files planned for: {', '.join(entities)}{C.RESET}")

    # Coder agent: one call per file
    write_tool = [t for t in TOOL_DEFS if t["function"]["name"] == "write_file"]
    created = {}
    ctx_size = min(_get_model_context_size(model), 4096)

    for i, spec in enumerate(file_list):
        rel_path = spec["path"]
        full_path = src_dir / rel_path
        if full_path.exists() and full_path.stat().st_size > 50:
            # Template already created this file, skip
            created[rel_path] = "(template)"
            print(f"  [{i+1}/{len(file_list)}] src/{rel_path} ... skip (exists)")
            continue

        print(f"  [{i+1}/{len(file_list)}] src/{rel_path} ", end="", flush=True)

        # Read dependency contents (pre-inject into conversation)
        dep_messages = []
        for dep in spec.get("depends_on", []):
            dep_path = src_dir / dep
            if dep_path.exists():
                try:
                    dep_content = dep_path.read_text(encoding="utf-8")
                    if len(dep_content) > 2000:
                        dep_content = dep_content[:2000] + "\n// ... (truncated)"
                    dep_messages.append({"role": "assistant", "content": f"I'll read src/{dep} first."})
                    dep_messages.append({"role": "tool", "content": dep_content,
                                        "tool_call_id": f"read_{dep.replace('/', '_')}"})
                except Exception:
                    pass

        # Read CONTEXT.md
        context_md = context_path.read_text(encoding="utf-8")

        # Build conversation
        messages = [
            {"role": "system", "content": (
                "You are a code generator for a Next.js 14 App Router + Supabase + Tailwind project.\n"
                "RULES:\n"
                "- Call write_file with file_path and complete code content.\n"
                "- Write COMPLETE, compilable TypeScript code. No placeholders, no TODOs.\n"
                "- Import types from '@/lib/types'.\n"
                "- Supabase: import { createClient } from '@/lib/supabase/server'. NEVER import from '@supabase/supabase-js' directly.\n"
                "- API routes: wrap DB ops in try/catch. Auth: const supabase = await createClient(); const { data: { user } } = await supabase.auth.getUser();\n"
                "- Validation: use manual typeof/length checks. Do NOT use Zod, Yup, or any validation library.\n"
                "- Components with useState/useEffect/onClick MUST start with 'use client' on line 1.\n"
                "- Server components (layouts, pages without state) must NOT use 'use client' or useRouter. Use redirect() from next/navigation for redirects.\n"
                "- Tailwind CSS with dark: variants for dark mode support.\n"
                "- Sidebar links must use /dashboard/ prefix (e.g. /dashboard/items, not /items).\n"
            )},
            {"role": "user", "content": (
                f"Project context:\n{context_md}\n\n"
                f"Project: {project_desc}\n\n"
                f"Create file: src/{rel_path}\n"
                f"Purpose: {spec['purpose']}\n\n"
                f"Call write_file with file_path='src/{rel_path}' and the complete code."
            )},
        ]
        # Insert dependency reads between system and user
        if dep_messages:
            messages = messages[:1] + dep_messages + messages[1:]

        # Call model
        success = False
        content = ""
        original_msg_len = len(messages)  # snapshot before retries

        for attempt in range(3):  # 3 attempts
            try:
                if attempt > 0:
                    # Reset messages to original state — no stacking of retry prompts
                    messages = messages[:original_msg_len]
                    # Add placeholder assistant msg for alternating turn format
                    messages.append({"role": "assistant", "content": "(retry)"})
                    # Clean retry prompt
                    messages.append({"role": "user", "content":
                        f"Call write_file with these exact arguments:\n"
                        f"  file_path: src/{rel_path}\n"
                        f"  content: (the complete TypeScript/React code)\n\n"
                        f"Purpose: {spec['purpose']}\n"
                        f"Output ONLY the tool call, nothing else."})

                response = _ollama_chat_sync(messages, model, tools=write_tool, num_ctx=ctx_size)
                msg = response.get("message", {})

                # Tier 1: structured tool call
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    args = tc.get("function", {}).get("arguments", {})
                    fp = args.get("file_path", f"src/{rel_path}")
                    content = _sanitize_generated_code(args.get("content", ""))
                    if content and len(content.strip()) > 20:
                        if not fp.startswith("src/"):
                            fp = f"src/{rel_path}"
                        full = pdir / fp
                        full.parent.mkdir(parents=True, exist_ok=True)
                        full.write_text(content, encoding="utf-8")
                        success = True
                        break

                # Tier 2: rescue tool call from text
                text = msg.get("content", "")
                rescued_text, rescued_calls = rescue_tool_calls_from_text(text)
                if rescued_calls:
                    tc = rescued_calls[0]
                    args = tc.get("function", {}).get("arguments", {})
                    content = _sanitize_generated_code(args.get("content", ""))
                    if content and len(content.strip()) > 20:
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        full_path.write_text(content, encoding="utf-8")
                        success = True
                        break

                # Tier 2.5: extract "content" value from inline JSON
                if text:
                    content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
                    if content_match:
                        try:
                            extracted = json.loads('["' + content_match.group(1) + '"]')[0]
                            extracted = _sanitize_generated_code(extracted)
                            if extracted and len(extracted.strip()) > 20:
                                full_path.parent.mkdir(parents=True, exist_ok=True)
                                full_path.write_text(extracted, encoding="utf-8")
                                content = extracted
                                success = True
                                break
                        except (json.JSONDecodeError, IndexError, ValueError):
                            pass

                # Tier 3: extract code from text
                code = _extract_code_from_response(text)
                if code and len(code) > 30:
                    content = _sanitize_generated_code(code)
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content, encoding="utf-8")
                    success = True
                    break

                # All tiers failed — loop continues to retry

            except Exception as e:
                if attempt == 0:
                    print(f"err:{e} ", end="", flush=True)
                continue  # retry instead of break

        if success:
            exports = _extract_export_summary(content, rel_path)
            _append_to_context(context_path, rel_path, exports)
            created[rel_path] = exports
            char_count = len(content)
            print(f"... OK ({char_count} chars)")
        else:
            print(f"... SKIP")

    return len(created), list(created.keys())


def _run_agent_turn_with_fallback(messages, model, use_tools=True, project_dir=None):
    """
    Wrap run_agent_turn() with ConnectionError fallback and inactivity timeout.
    Tries free providers first, then paid ones only if API key is configured.
    Returns (success_bool, new_file_count, new_file_list).
    """
    import threading

    scaffold_timeout = int(os.environ.get("CLAW_SCAFFOLD_TIMEOUT", "180"))

    pre_files = _snapshot_project_files(project_dir) if project_dir else set()

    # Build fallback chain: current provider first, free fallbacks, then paid (if key configured)
    current = PROVIDER.lower()
    free_fallbacks = []
    paid_fallbacks = []

    # Add ollama if not already the current provider
    if current != "ollama":
        free_fallbacks.append("ollama")

    # Add paid providers only if user has configured their API key
    if current != "dashscope" and DASHSCOPE_API_KEY:
        paid_fallbacks.append("dashscope")
    if current != "anthropic" and ANTHROPIC_API_KEY:
        paid_fallbacks.append("anthropic")
    if current != "openai" and OPENAI_API_KEY:
        paid_fallbacks.append("openai")

    providers_to_try = [current] + free_fallbacks + paid_fallbacks

    global _provider_instance

    for i, provider_name in enumerate(providers_to_try):
        # Switch provider if needed (skip for first/current)
        if i > 0:
            is_paid = provider_name in ("anthropic", "openai", "dashscope")
            if is_paid:
                print(f"  {C.WARNING}{BLACK_CIRCLE} Falling back to {provider_name} (API key configured){C.RESET}")
            else:
                print(f"  {C.SUBTLE}{BLACK_CIRCLE} Trying fallback: {provider_name}{C.RESET}")

            # Swap provider instance
            old_instance = _provider_instance
            if provider_name == "ollama":
                _provider_instance = OllamaProvider()
            elif provider_name == "openrouter":
                _provider_instance = OpenRouterProvider()
            elif provider_name == "openai":
                _provider_instance = OpenAIProvider()
            elif provider_name == "anthropic":
                _provider_instance = AnthropicProvider()
            elif provider_name == "dashscope":
                _provider_instance = DashScopeProvider()

        # Start inactivity monitor
        timed_out = threading.Event()

        def _timeout_handler():
            timed_out.set()

        timer = threading.Timer(scaffold_timeout, _timeout_handler)
        timer.daemon = True
        timer.start()

        try:
            run_agent_turn(messages, model, use_tools=use_tools)
            timer.cancel()

            # Check progress
            if project_dir:
                new_count, new_files = _check_scaffold_progress(project_dir, pre_files)
                return True, new_count, new_files
            return True, 0, []

        except ConnectionError as e:
            timer.cancel()
            print(f"  {C.WARNING}{BLACK_CIRCLE} Provider {provider_name} connection failed: {e}{C.RESET}")
            continue

        except Exception as e:
            timer.cancel()
            # Check if our timeout fired — if so, treat as retriable
            if timed_out.is_set() and project_dir:
                new_count, new_files = _check_scaffold_progress(project_dir, pre_files)
                if new_count == 0:
                    print(f"  {C.WARNING}{BLACK_CIRCLE} Model produced 0 files in {scaffold_timeout}s, trying next provider...{C.RESET}")
                    continue
                else:
                    # Partial progress — keep what we have
                    print(f"  {C.WARNING}{BLACK_CIRCLE} Timeout, but {new_count} files created — continuing{C.RESET}")
                    return True, new_count, new_files
            raise

    # All providers failed — restore original and return failure
    print(f"  {C.ERROR}{BLACK_CIRCLE} All providers failed to complete scaffold{C.RESET}")
    return False, 0, []


# ---------------------------------------------------------------------------
# agent loop
# ---------------------------------------------------------------------------

def run_agent_turn(messages, model, use_tools=True):
    """
    Run one user turn through the agent loop.
    Calls the model, executes tools, loops until the model stops calling tools.
    Returns the final assistant text.
    """
    iterations = 0
    tools = TOOL_DEFS if use_tools else None
    tool_retry_counts = {}  # unified retry tracking: (tool_name, key_arg) -> count
    repetition_detector = _RepetitionDetector()
    reflection_state = _ReflectionState()
    _ramble_nudges = 0  # max 1 nudge per agent turn

    # Per-turn token tracking
    _turn_prompt_start = _token_tracker.prompt_tokens
    _turn_comp_start = _token_tracker.completion_tokens

    # Reset read-before-write tracker each turn
    with _read_guard_lock:
        _files_read_this_turn.clear()
        _files_searched_this_turn.clear()

    # Wiring agent tracking
    _files_written_this_turn = False
    _wiring_pass_count = 0
    _wiring_prompt_pending = False  # True if we sent a wiring fix prompt and await tool-based fixes

    # TODO resolver tracking
    _todo_written_files = set()
    _todo_pass_count = 0

    # Dynamic context window
    ctx_size = _get_model_context_size(model)
    # Cap context window for local models to avoid VRAM exhaustion.
    # KV cache for 32K+ on 14B models needs ~4GB+ VRAM on top of model weights.
    # Most local GPUs (8-12GB) can't handle that. Cap at 8192 — plenty for
    # the actual content we send (~4-6K tokens) with room for generation.
    if _is_local_model(model):
        ctx_size = min(ctx_size, 4096)
    ctx_budget = int(ctx_size * 0.7)

    # Slim system prompt for local models — saves ~8K tokens of context
    # Don't override if scaffold already built a slim prompt (scaffold appends
    # design/security/API context that _build_slim_system_prompt would delete)
    if _is_local_model(model) and messages and messages[0].get("role") == "system":
        if "BUILD SPEC" not in (messages[1].get("content", "") if len(messages) > 1 else ""):
            slim = _build_slim_system_prompt()
            messages[0]["content"] = slim

    # Multi-model routing: use small model for simple follow-up turns
    active_model = _pick_model_for_task(messages, model)
    if active_model != model:
        print(f"  {C.SUBTLE}routing → {active_model}{C.RESET}")

    _compaction_count = 0  # track how many times we auto-compacted this turn

    # --- PLAN.md step context injection (Phase 5) ---
    # If the last user message references a PLAN.md step with Files: annotation,
    # auto-inject graph context for those files
    if _project_graph is not None and messages:
        _last_user = ""
        for _m in reversed(messages):
            if _m.get("role") == "user":
                _last_user = _m.get("content", "")
                break
        if _last_user:
            _step_files = _extract_plan_step_files("", _last_user)
            if _step_files:
                _step_ctx = _build_graph_context(_step_files, max_tokens=400)
                if _step_ctx and messages[0].get("role") == "system":
                    messages[0]["content"] += f"\n\n{_step_ctx}"

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            print(f"\n  {C.ERROR}{BLACK_CIRCLE} Stopped: hit {MAX_ITERATIONS} iteration limit{C.RESET}")
            break

        # After first iteration with tool calls, switch back to main model
        # (small model is only for the initial simple response)
        turn_model = active_model if iterations == 1 else model

        full_message = {"role": "assistant", "content": ""}
        tool_calls = []
        has_tool_calls = False
        got_first_token = False

        # --- AUTO CONTEXT COMPACTION ---
        # If context is getting large, proactively compact old messages
        _est_tokens = _estimate_tokens(messages)
        _compact_threshold = int(ctx_budget * 0.80)
        if _est_tokens > _compact_threshold and len(messages) > 6:
            _before_len = len(messages)
            _before_tokens = _est_tokens
            messages = _compress_messages(messages, max_tokens=_compact_threshold)
            _after_tokens = _estimate_tokens(messages)
            if _after_tokens < _before_tokens:
                _compaction_count += 1
                _saved = _before_tokens - _after_tokens
                print(f"  {C.SUBTLE}{BLACK_CIRCLE} Auto-compacted context: ~{_format_tokens(_before_tokens)} → ~{_format_tokens(_after_tokens)} tokens ({_before_len} → {len(messages)} msgs){C.RESET}")

        # Compress messages to fit context window before calling model
        ctx_messages = _compress_messages(messages, max_tokens=ctx_budget)

        # start the thinking spinner with timing
        spinner = TimedSpinner(random.choice(SPINNER_VERBS), C.CLAW)
        spinner.start()

        # Stream with real-time <think> suppression
        # Buffer content until we know if it's inside a think block
        _in_think_block = False
        _pending_print = ""  # content waiting to be printed (may contain partial <think>)

        for chunk in ollama_chat(ctx_messages, turn_model, tools=tools, stream=True, num_ctx=ctx_size):
            msg = chunk.get("message", {})

            content = msg.get("content", "")
            if content:
                full_message["content"] += content

                if THINKING_ENABLED:
                    if _in_think_block:
                        # Inside think block — check if it ended
                        if "</think>" in full_message["content"]:
                            _in_think_block = False
                            # Content after </think> will be printed in post-stream
                        continue  # suppress all content while in/exiting think block
                    else:
                        # Not in think block — check if one is starting
                        _pending_print += content
                        if "<think>" in _pending_print:
                            # Print everything before <think>, suppress the rest
                            before = _pending_print.split("<think>", 1)[0]
                            if before:
                                if not got_first_token:
                                    spinner.stop()
                                    got_first_token = True
                                    sys.stdout.write(C.TEXT)
                                    sys.stdout.flush()
                                sys.stdout.write(before)
                                sys.stdout.flush()
                            _in_think_block = True
                            _pending_print = ""
                            continue
                        elif "<" in _pending_print and not _pending_print.endswith(">"):
                            # Might be a partial "<think" tag — hold back
                            last_lt = _pending_print.rfind("<")
                            safe = _pending_print[:last_lt]
                            _pending_print = _pending_print[last_lt:]
                            if safe:
                                if not got_first_token:
                                    spinner.stop()
                                    got_first_token = True
                                    sys.stdout.write(C.TEXT)
                                    sys.stdout.flush()
                                sys.stdout.write(safe)
                                sys.stdout.flush()
                            continue
                        else:
                            # No think tag — flush pending
                            if not got_first_token:
                                spinner.stop()
                                got_first_token = True
                                sys.stdout.write(C.TEXT)
                                sys.stdout.flush()
                            sys.stdout.write(_pending_print)
                            sys.stdout.flush()
                            _pending_print = ""
                            continue

                # THINKING_ENABLED is False — just print directly
                if not got_first_token:
                    spinner.stop()
                    got_first_token = True
                    sys.stdout.write(C.TEXT)
                    sys.stdout.flush()
                sys.stdout.write(content)
                sys.stdout.flush()

            if msg.get("tool_calls"):
                has_tool_calls = True
                tool_calls = msg["tool_calls"]

        # Flush any remaining pending content
        if _pending_print:
            if not got_first_token:
                spinner.stop()
                got_first_token = True
                sys.stdout.write(C.TEXT)
                sys.stdout.flush()
            sys.stdout.write(_pending_print)
            sys.stdout.flush()

        if not got_first_token:
            spinner.stop()

        # rescue tool calls written as text (common with local models)
        if not has_tool_calls and full_message["content"] and use_tools:
            rescued_text, rescued_calls = rescue_tool_calls_from_text(full_message["content"])
            if rescued_calls:
                has_tool_calls = True
                tool_calls = rescued_calls
                full_message["content"] = rescued_text

        # --- Output quality guards (Fix A/B/C) ---
        _display_content = full_message["content"]  # display copy (guards modify this, not stored)

        # Fix B: Sliding window duplicate detection
        if _display_content and len(_display_content) > 800:
            _display_content, _was_deduped = _deduplicate_response(_display_content)
            if _was_deduped:
                _guard_stats["dedup_fires"] += 1
                full_message["content"] = _display_content
                print(f"\n  {C.WARNING}\u26a0 Duplicate content removed{C.RESET}")

        # Fix A: Smart streaming length guard (code-dump aware)
        if _display_content and not has_tool_calls and len(_display_content) > 4000:
            if _is_code_dump(_display_content):
                _display_content = _truncate_at_sentence(_display_content, 4000)
                full_message["content"] = _display_content
                _guard_stats["ramble_truncations"] += 1
                print(f"\n  {C.WARNING}\u26a0 Response truncated (code-dumping detected){C.RESET}")
                if _ramble_nudges < 1:
                    _ramble_nudges += 1
                    # Nudge the model to use tools instead
                    messages.append({"role": "assistant", "content": _display_content})
                    messages.append({"role": "user", "content": "[SYSTEM: Your response was truncated because you output code as text. Use write_file or edit_file tools instead of showing code in chat. Keep text responses to brief status updates.]"})
                    continue

        # Fix A: "tools + ramble" — if tools used but >2000 chars of text, trim text from history
        if has_tool_calls and _display_content and len(_display_content) > 2000:
            # User already saw it streamed; strip text from stored message
            full_message["content"] = ""

        # --- Inner monologue: extract and display thinking ---
        if THINKING_ENABLED and full_message["content"]:
            raw_acc = full_message["content"]
            think_text, cleaned = _extract_think_block(raw_acc)
            if think_text:
                # Think content was suppressed during streaming — display summary
                _display_thinking(think_text)
                full_message["content"] = cleaned
                # The stream already printed content before <think>.
                # Content after </think> was suppressed — print it now.
                after_think = ""
                if "</think>" in raw_acc:
                    after_think = raw_acc.split("</think>", 1)[-1].strip()
                if after_think:
                    sys.stdout.write(f"{C.TEXT}{after_think}{C.RESET}\n")
                    sys.stdout.flush()
                    got_first_token = True
                elif not cleaned:
                    # Nothing left after removing think block
                    if got_first_token:
                        got_first_token = False
            elif THINK_RETRY_ON_MISSING and has_tool_calls and iterations == 1:
                pass  # Handled via reflection_state.thinking_nudge_sent

        if got_first_token:
            sys.stdout.write(C.RESET + "\n")
            sys.stdout.flush()

        # rescue questions typed as text -- convert to interactive ask_user
        if not has_tool_calls and full_message["content"] and use_tools:
            question_result = rescue_question_from_text(full_message["content"])
            if question_result:
                q_text, q_choices, cleaned = question_result
                # run ask_user interactively
                answer = tool_ask_user({"question": q_text, "choices": q_choices})
                # put the answer back into the conversation
                full_message["content"] = cleaned
                messages.append({"role": "assistant", "content": cleaned})
                messages.append({"role": "tool", "content": answer})
                continue  # loop back so model sees the answer

        # 7b: Chatbot rescue — if model dumps commands/code/plans as text instead of calling tools
        if not has_tool_calls and full_message["content"] and use_tools and iterations < MAX_ITERATIONS:
            content = full_message["content"]
            # Detect if the model is showing commands/code instead of using tools
            has_code_blocks = content.count("```") >= 2
            has_command_instructions = any(phrase in content.lower() for phrase in [
                "run this command", "try this", "run the following", "execute this",
                "copy and paste", "type this", "follow these steps", "step 1",
                "step 2", "step 3", "open your terminal", "open cmd",
            ])
            # Detect file content printed as text (e.g., JSON, config, component code)
            has_file_content = bool(re.search(r'```(?:json|typescript|tsx|jsx|javascript|python|css|html)\n', content))
            # Detect "Here's the X file:" pattern
            has_heres_file = bool(re.search(r"(?:here'?s|this is|create)\s+(?:the|a|your)\s+[`'\"]?\w+\.\w+", content, re.IGNORECASE))
            # Detect "I'll create..." without tool calls (planning without doing)
            has_plan_text = bool(re.search(r"(?:i'?ll|let me|i will|i need to)\s+(?:create|write|set up|start|build|make)", content, re.IGNORECASE))
            if has_code_blocks or has_command_instructions or has_file_content or has_heres_file:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    "[SYSTEM: VIOLATION — You showed commands/code as text instead of using your tools. "
                    "You are an AGENT. Do NOT tell the user to run commands or show file contents. "
                    "YOU must call write_file to create files, edit_file to modify files, bash to run commands. "
                    "Create the NEXT file now using write_file. No more text — just tool calls.]"
                )})
                continue
            # Anti-plan-loop: if model keeps saying "I'll create..." without doing it
            if has_plan_text and not has_code_blocks and iterations > 2:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    "[SYSTEM: You're describing what you'll do instead of DOING it. "
                    "STOP PLANNING. Call write_file RIGHT NOW to create the next file. "
                    "No more explanations — just tool calls.]"
                )})
                continue

        # 7c: Garbage output detection — re-prompt if garbled
        if not has_tool_calls and _is_garbage_output(full_message["content"]) and iterations < MAX_ITERATIONS:
            messages.append({"role": "assistant", "content": full_message["content"] or ""})
            messages.append({"role": "user", "content": "[SYSTEM: Your previous response was empty or garbled. Please try again with a clear response.]"})
            continue

        assistant_msg = {"role": "assistant", "content": full_message["content"]}
        if has_tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not has_tool_calls:
            # Detect hallucinated wiring fixes: model responded to wiring prompt
            # with text only (no tool calls) — it claimed to fix but didn't write files
            if _wiring_prompt_pending and _wiring_pass_count < 3:
                _wiring_prompt_pending = False
                _wiring_pass_count += 1
                messages.append({"role": "user", "content": (
                    "[SYSTEM: VIOLATION — You described fixes as text instead of using tools. "
                    "Your response did NOT modify any files. The wiring issues STILL EXIST. "
                    "You MUST use write_file or edit_file to actually fix each issue. "
                    "Do not explain — call the tools NOW.]"
                )})
                _files_written_this_turn = False
                continue

            # Wiring check after file generation batch (max 2 passes)
            if WIRING_ENABLED and _files_written_this_turn and _wiring_pass_count < 2:
                _wiring_pass_count += 1
                wiring_agent = WiringAgent(CWD)
                wiring_agent.run_full_scan()
                if wiring_agent.issues:
                    wiring_agent.auto_fix()
                    _display_wiring_report(wiring_agent)
                    if wiring_agent.manual_needed:
                        messages.append({"role": "user", "content": wiring_agent.format_prompt()})
                        _wiring_prompt_pending = True
                        _files_written_this_turn = False
                        continue  # let model fix issues

            # TODO Resolver check after wiring (max 2 passes)
            if TODO_RESOLVER_ENABLED and _todo_written_files and _todo_pass_count < 2:
                _all_todos = _scan_all_todos(_todo_written_files)
                if _all_todos:
                    _todo_pass_count += 1
                    _guard_stats["todo_resolves"] += 1
                    _n = len(_all_todos)
                    print(f"  {C.WARNING}{BLACK_CIRCLE} TODO Resolver: found {_n} incomplete item{'s' if _n != 1 else ''} — sending back to fix{C.RESET}")
                    _resolver_prompt = _build_todo_resolver_prompt(_all_todos)
                    messages.append({"role": "user", "content": _resolver_prompt})
                    _todo_written_files.clear()
                    continue  # loop back — model fixes the TODOs

            # Track text responses for repetition detection
            if full_message["content"]:
                repetition_detector.record_response(full_message["content"])
            # Fix C: Strip structural HTML tags from display only
            final_text = full_message["content"]
            if final_text:
                final_text, _html_stripped = _strip_structural_html(final_text)
                if _html_stripped:
                    _guard_stats["html_strips"] += 1
            # Re-render with markdown formatting if response has markdown markers
            if final_text and any(m in final_text for m in ("**", "```", "## ", "# ", "- ")):
                if _ANSI_CURSOR_OK:
                    # Clear the raw streamed output and reprint with formatting
                    try:
                        tw = os.get_terminal_size().columns
                    except (OSError, ValueError):
                        tw = 80
                    visual_lines = sum(max(1, math.ceil(len(ln) / tw)) for ln in final_text.split("\n"))
                    sys.stdout.write(f"\033[{visual_lines}A")
                    for _ in range(visual_lines):
                        sys.stdout.write("\033[2K\n")
                    sys.stdout.write(f"\033[{visual_lines}A")
                rendered = _render_markdown(final_text)
                print(rendered)
            # Per-turn token display
            _turn_p = _token_tracker.prompt_tokens - _turn_prompt_start
            _turn_c = _token_tracker.completion_tokens - _turn_comp_start
            if _turn_p > 0 or _turn_c > 0:
                print(f"  {C.SUBTLE}↑{_format_tokens(_turn_p)} ↓{_format_tokens(_turn_c)} tokens{C.RESET}")
            return final_text

        # execute tool calls -- Claude Code style display with parallel support
        bash_failed_this_round = False
        _wiring_prompt_pending = False  # model is using tools — clear hallucination flag

        # Check if we can run some tools in parallel
        has_parallel = sum(1 for tc in tool_calls
                         if tc.get("function", {}).get("name") in PARALLEL_SAFE_TOOLS) > 1

        if has_parallel:
            # Parallel execution path
            tool_results = _execute_tools_parallel(tool_calls)
            for tc, tool_name, tool_args, result in tool_results:
                tc_id = tc.get("id", "")
                if result is None:
                    # JSON parse failure
                    raw_args = tc.get("function", {}).get("arguments", "")
                    tool_msg = {"role": "tool", "content": _tool_error(tool_name, f"Could not parse tool arguments as JSON: {str(raw_args)[:200]}", "Provide arguments as a valid JSON object.")}
                    if tc_id: tool_msg["tool_call_id"] = tc_id
                    messages.append(tool_msg)
                    continue

                # format tool input summary
                input_summary = ""
                if tool_name == "bash":
                    input_summary = f"`{tool_args.get('command', '')}`"
                elif tool_name == "read_file":
                    input_summary = tool_args.get("file_path", "")
                elif tool_name in ("write_file",):
                    fp = tool_args.get("file_path", "")
                    sz = len(tool_args.get("content", ""))
                    input_summary = f"{fp} ({sz} chars)"
                elif tool_name == "edit_file":
                    input_summary = tool_args.get("file_path", "")
                elif tool_name == "glob_search":
                    input_summary = tool_args.get("pattern", "")
                elif tool_name == "grep_search":
                    input_summary = f"/{tool_args.get('pattern', '')}/"

                sys.stdout.write(f"  {C.TOOL}{BLACK_CIRCLE} {C.BOLD}{tool_name}{C.RESET}")
                if input_summary:
                    sys.stdout.write(f" {C.SUBTLE}({input_summary}){C.RESET}")
                sys.stdout.write("\n")
                sys.stdout.flush()

                # Unified retry + error hint
                result, bash_failed_this_round = _apply_retry_and_hints(
                    tool_name, tool_args, result, tool_retry_counts, bash_failed_this_round
                )

                # show result
                if tool_name == "bash":
                    _display_bash_result(tool_args.get("command", ""), result)
                else:
                    _display_tool_result(tool_name, result)
                tool_msg = {"role": "tool", "content": result}
                if tc_id: tool_msg["tool_call_id"] = tc_id
                messages.append(tool_msg)

                # Track file writes for wiring agent + TODO resolver
                if tool_name in ("write_file", "edit_file") and not result.startswith("Error"):
                    _files_written_this_turn = True
                    _todo_written_files.add(tool_args.get("file_path", ""))
                    _check_auto_dep_install(tool_name, tool_args, result, messages)

                # Track for repetition detection and reflection
                repetition_detector.record_action(tool_name, tool_args)
                reflection_state.record_tool_result(result)
        else:
            # Sequential execution path (original behavior + unified retry)
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tc_id = tc.get("id", "")
                tool_args = func.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_msg = {"role": "tool", "content": _tool_error(tool_name, f"Could not parse tool arguments as JSON: {tool_args[:200]}", "Provide arguments as a valid JSON object.")}
                        if tc_id: tool_msg["tool_call_id"] = tc_id
                        messages.append(tool_msg)
                        continue

                # format tool input summary
                input_summary = ""
                if tool_name == "bash":
                    input_summary = f"`{tool_args.get('command', '')}`"
                elif tool_name == "read_file":
                    input_summary = tool_args.get("file_path", "")
                elif tool_name == "write_file":
                    fp = tool_args.get("file_path", "")
                    sz = len(tool_args.get("content", ""))
                    input_summary = f"{fp} ({sz} chars)"
                elif tool_name == "edit_file":
                    input_summary = tool_args.get("file_path", "")
                elif tool_name == "glob_search":
                    input_summary = tool_args.get("pattern", "")
                elif tool_name == "grep_search":
                    input_summary = f"/{tool_args.get('pattern', '')}/"

                sys.stdout.write(f"  {C.TOOL}{BLACK_CIRCLE} {C.BOLD}{tool_name}{C.RESET}")
                if input_summary:
                    sys.stdout.write(f" {C.SUBTLE}({input_summary}){C.RESET}")
                sys.stdout.write("\n")
                sys.stdout.flush()

                # check permission (edit-confirm mode)
                if not confirm_tool_execution(tool_name, tool_args):
                    result = "Permission denied by user."
                    print(f"    {C.WARNING}Skipped{C.RESET}")
                    tool_msg = {"role": "tool", "content": result}
                    if tc_id: tool_msg["tool_call_id"] = tc_id
                    messages.append(tool_msg)
                    continue

                # spin while tool executes
                tool_spinner = Spinner(random.choice([
                    "Executing", "Running", "Processing", "Working",
                ]), C.TOOL)
                tool_spinner.start()

                result = execute_tool(tool_name, tool_args)

                tool_spinner.stop()

                # Unified retry logic with error hints
                result, bash_failed_this_round = _apply_retry_and_hints(
                    tool_name, tool_args, result, tool_retry_counts, bash_failed_this_round
                )

                # show result
                if tool_name == "bash":
                    _display_bash_result(tool_args.get("command", ""), result)
                else:
                    _display_tool_result(tool_name, result)
                tool_msg = {"role": "tool", "content": result}
                if tc_id: tool_msg["tool_call_id"] = tc_id
                messages.append(tool_msg)

                # Track file writes for wiring agent + TODO resolver
                if tool_name in ("write_file", "edit_file") and not result.startswith("Error"):
                    _files_written_this_turn = True
                    _todo_written_files.add(tool_args.get("file_path", ""))
                    _check_auto_dep_install(tool_name, tool_args, result, messages)

                # Track for repetition detection and reflection
                repetition_detector.record_action(tool_name, tool_args)
                reflection_state.record_tool_result(result)

        sys.stdout.write(f"  {C.SUBTLE}···{C.RESET}\n")  # thin separator between tool results and next cycle

        # Inject queued auto-install results — append to last tool message to keep flow valid
        if _dep_install_results:
            combined = "\n".join(_dep_install_results)
            # Find last tool message and append to it
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "tool":
                    messages[i]["content"] += f"\n\n{combined}"
                    break
            else:
                # No tool message found — safe fallback as user message
                messages.append({"role": "user", "content": combined})
            _dep_install_results.clear()

        # --- Repetition detection ---
        rep_warning = repetition_detector.check()
        if rep_warning:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Loop detected{C.RESET}")
            messages.append({"role": "user", "content": rep_warning})
            continue

        # --- Stall detector: model keeps planning but not creating source files ---
        if iterations >= 6 and iterations % 3 == 0 and not _files_written_this_turn:
            # Check if only plan/config files exist — no actual source code written yet
            import glob as _glob_mod
            src_files = _glob_mod.glob(os.path.join(CWD, "**", "*.tsx"), recursive=True) + \
                        _glob_mod.glob(os.path.join(CWD, "**", "*.ts"), recursive=True) + \
                        _glob_mod.glob(os.path.join(CWD, "**", "*.jsx"), recursive=True) + \
                        _glob_mod.glob(os.path.join(CWD, "**", "*.py"), recursive=True) + \
                        _glob_mod.glob(os.path.join(CWD, "**", "*.js"), recursive=True)
            src_files = [f for f in src_files if 'node_modules' not in f and '.next' not in f]
            if len(src_files) == 0:
                print(f"  {C.WARNING}{BLACK_CIRCLE} Stall detected: {iterations} iterations, no source files created{C.RESET}")
                messages.append({"role": "user", "content": (
                    "[SYSTEM: STALL DETECTED — You have run " + str(iterations) + " iterations but created ZERO source files. "
                    "STOP planning, reading, and thinking. START creating files NOW. "
                    "Call write_file immediately to create the first source code file. "
                    "Do NOT write another plan. Do NOT explain what you'll do. Just call write_file.]"
                )})
                continue

        # --- Self-reflection injection ---
        if reflection_state.should_reflect():
            print(f"  {C.THINK_DIM}{BLACK_CIRCLE} Reflecting...{C.RESET}")
            messages.append({"role": "user", "content": reflection_state.build_reflection_prompt()})
            continue

        time.sleep(0.1)

    return ""

# ---------------------------------------------------------------------------
# multiline input
# ---------------------------------------------------------------------------

def read_multiline():
    """Read multiple lines until user types a line with just '---' or empty line twice."""
    cprint(C.DIM, "  (Multi-line mode. End with '---' on its own line)")
    lines = []
    empty_count = 0
    while True:
        try:
            sys.stdout.write(f"  {C.SUBTLE}...{C.RESET} ")
            sys.stdout.flush()
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "---":
            break
        if line.strip() == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
        lines.append(line)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# input box — styled prompt like Claude Code
# ---------------------------------------------------------------------------

# Box-drawing characters
_BOX_TL = "\u256d"   # ╭
_BOX_TR = "\u256e"   # ╮
_BOX_BL = "\u2570"   # ╰
_BOX_BR = "\u256f"   # ╯
_BOX_H  = "\u2500"   # ─
_BOX_V  = "\u2502"   # │
_BOX_DOT = "\u00b7"  # ·


def _get_terminal_width():
    """Get terminal width, defaulting to 80."""
    try:
        return os.get_terminal_size().columns
    except (OSError, ValueError):
        return 80


def _has_pending_input():
    """Check if there's more input waiting on stdin (paste detection)."""
    try:
        if sys.platform == "win32":
            import msvcrt
            return msvcrt.kbhit()
        else:
            import select
            return select.select([sys.stdin], [], [], 0.0)[0] != []
    except Exception:
        return False


def _read_pasted_lines(first_line):
    """
    After reading the first line, drain any remaining pasted lines from stdin.
    Returns the full multi-line string and the line count.
    """
    lines = [first_line]
    # Give the terminal a tiny moment for paste buffer to arrive
    time.sleep(0.05)

    while _has_pending_input():
        try:
            line = input()
            lines.append(line)
            # Safety cap — don't read more than 5000 lines
            if len(lines) >= 5000:
                break
            # Brief pause to check for more buffered input
            time.sleep(0.01)
        except EOFError:
            break

    return "\n".join(lines), len(lines)


def _draw_input_box(turn_count=0):
    """
    Draw a bordered input box like Claude Code's prompt.
    Supports multi-line paste detection — if user pastes many lines,
    all lines are captured and a summary is shown.
    Returns (user_input, box_metadata) where box_metadata is a dict with
    line_count, input_len, draw_width, and paste_lines for clearing.
    """
    w = min(_get_terminal_width(), 120)
    inner_w = w - 4  # 2 chars padding each side

    # Colors for the box border — subtle gradient
    border_dim = _rgb(60, 140, 120)    # muted teal
    border_bright = _rgb(87, 199, 170) # brand teal

    # Top border with prompt indicator
    prompt_label = f" rattlesnake "
    label_len = len(prompt_label)
    remaining = inner_w - label_len
    left_bar = _BOX_H * 2
    right_bar = _BOX_H * max(remaining - 2, 1)

    sys.stdout.write(
        f"  {border_dim}{_BOX_TL}{left_bar}"
        f"{border_bright}{C.BOLD}{prompt_label}{C.RESET}"
        f"{border_dim}{right_bar}{_BOX_TR}{C.RESET}\n"
    )

    # Input line with border
    sys.stdout.write(f"  {border_dim}{_BOX_V}{C.RESET} {C.CLAW}{C.BOLD}>{C.RESET} ")
    sys.stdout.flush()

    try:
        first_line = input()
    except (EOFError, KeyboardInterrupt):
        # Close the box before raising
        sys.stdout.write(f"\n  {border_dim}{_BOX_BL}{_BOX_H * inner_w}{_BOX_BR}{C.RESET}\n")
        sys.stdout.flush()
        raise

    # Detect multi-line paste: check if more lines are buffered in stdin
    user_input, line_count = _read_pasted_lines(first_line)

    paste_display_lines = 0  # track extra lines drawn for paste preview

    # Show paste summary if multiple lines were pasted
    if line_count > 1:
        total_chars = len(user_input)
        # Show first and last few lines as preview
        all_lines = user_input.split("\n")
        if line_count <= 5:
            # Short paste — show all lines
            for i, ln in enumerate(all_lines[1:], 2):
                display = ln[:80] + "..." if len(ln) > 80 else ln
                sys.stdout.write(f"  {border_dim}{_BOX_V}{C.RESET}   {C.DIM}{display}{C.RESET}\n")
                paste_display_lines += 1
        else:
            # Long paste — show first 3 and last 2 with summary
            for i, ln in enumerate(all_lines[1:4], 2):
                display = ln[:80] + "..." if len(ln) > 80 else ln
                sys.stdout.write(f"  {border_dim}{_BOX_V}{C.RESET}   {C.DIM}{display}{C.RESET}\n")
                paste_display_lines += 1
            hidden = line_count - 5
            sys.stdout.write(
                f"  {border_dim}{_BOX_V}{C.RESET}   {C.SUBTLE}... {hidden} more line(s) ...{C.RESET}\n"
            )
            paste_display_lines += 1
            for ln in all_lines[-2:]:
                display = ln[:80] + "..." if len(ln) > 80 else ln
                sys.stdout.write(f"  {border_dim}{_BOX_V}{C.RESET}   {C.DIM}{display}{C.RESET}\n")
                paste_display_lines += 1

        sys.stdout.write(
            f"  {border_dim}{_BOX_V}{C.RESET} {C.CLAW}{BLACK_CIRCLE} "
            f"{line_count} lines pasted{C.RESET} {C.SUBTLE}({total_chars} chars){C.RESET}\n"
        )
        paste_display_lines += 1
        sys.stdout.flush()

    # Bottom border with hints
    if turn_count == 0:
        hint = f" /help {_BOX_DOT} /plan {_BOX_DOT} /generate {_BOX_DOT} /quit "
    else:
        hint = f" turn {turn_count + 1} "

    hint_len = len(hint)
    bottom_remaining = inner_w - hint_len
    bottom_left = _BOX_H * max(bottom_remaining // 2, 1)
    bottom_right = _BOX_H * max(bottom_remaining - bottom_remaining // 2, 1)

    sys.stdout.write(
        f"  {border_dim}{_BOX_BL}{bottom_left}"
        f"{C.DIM}{hint}{C.RESET}"
        f"{border_dim}{bottom_right}{_BOX_BR}{C.RESET}\n"
    )
    sys.stdout.flush()

    # Build metadata for clearing
    box_meta = {
        "input_len": len(first_line),
        "draw_width": w,
        "paste_lines": paste_display_lines,
        "line_count": line_count,
    }

    return user_input, box_meta


def _clear_input_box(box_meta):
    """Clear the input box from the terminal.
    Handles terminal resize between draw and clear by recalculating wrapping."""
    if not _ANSI_CURSOR_OK:
        return  # Can't clear — just leave the box (better than corrupting display)

    current_w = min(_get_terminal_width(), 120)
    prompt_prefix_len = 6  # "│ > " with indent = ~6 visible chars

    # Calculate how many visual lines the input occupied (accounting for terminal wrapping)
    input_visual_lines = max(1, math.ceil((box_meta["input_len"] + prompt_prefix_len) / current_w))

    # Total lines: top border + input line(s) + paste preview lines + bottom border
    total_lines = 1 + input_visual_lines + box_meta["paste_lines"] + 1

    # Move cursor up and erase each line
    sys.stdout.write(f"\033[{total_lines}A")
    for _ in range(total_lines):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{total_lines}A")
    sys.stdout.flush()


def _display_sent_message(user_text):
    """Display a compact sent-message indicator after clearing the input box."""
    text = user_text.strip()
    if not text:
        return

    lines = text.split("\n")
    if len(lines) == 1:
        display = text[:100]
        if len(text) > 100:
            display += "..."
        print(f"  {C.CLAW}>{C.RESET} {C.TEXT}{display}{C.RESET}")
    else:
        first = lines[0][:80]
        if len(lines[0]) > 80:
            first += "..."
        print(f"  {C.CLAW}>{C.RESET} {C.TEXT}{first}{C.RESET} {C.SUBTLE}({len(lines)} lines){C.RESET}")
    print()  # spacing before model output


# ---------------------------------------------------------------------------
# startup animation — the rattlesnake
# ---------------------------------------------------------------------------

# The snake path: each tuple is (row, col, char) — the snake slithers in
# character by character along this path
_SNAKE_PATH = (
    # tail / rattle (bottom right, moving left)
    "                                                 "
    "                  ~SssSSSSSssS~                   "
)

# Static final art — what stays on screen after animation
_SNAKE_ART = [
    "",
    "        {}        ,,_         {}",
    "        {}      .' _ `.       {}",
    "        {}     /  (@)  ;---<<  {}",
    "        {}    |   _  _;       {}",
    "        {}     `. / `/        {}",
    "        {}    _.'-v-'._       {}",
    "        {}  .' .---. _,`.     {}",
    "        {} /  /     ( o )`.   {}",
    "        {}|  |       `-'  |   {}",
    "        {} `. `.___.' ___.'   {}",
    "        {}   `--...--'  .'   {}",
    "        {}    _.---._  /     {}",
    "        {}   (  ~Ss'  )      {}",
    "        {}    `------'       {}",
    "",
]

def _lerp_color(t, r1, g1, b1, r2, g2, b2):
    """Linearly interpolate between two RGB colors. t in [0, 1]."""
    return (int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t))


def _snake_gradient(row, total_rows):
    """Return RGB tuple for a snake body gradient: dark emerald head -> lime -> amber tail."""
    t = row / max(total_rows - 1, 1)
    if t < 0.35:
        # head: deep emerald to bright emerald
        s = t / 0.35
        return _lerp_color(s, 6, 95, 70, 16, 185, 129)
    elif t < 0.65:
        # mid body: emerald to lime
        s = (t - 0.35) / 0.3
        return _lerp_color(s, 16, 185, 129, 132, 204, 22)
    else:
        # tail: lime to gold/amber
        s = (t - 0.65) / 0.35
        return _lerp_color(s, 132, 204, 22, 234, 179, 8)


def _render_snake_frame(tongue_out=False, rattle_spark=False):
    """Build the snake as a list of (text, color_code) per line."""
    # Clean detailed side-profile rattlesnake
    if tongue_out:
        head = [
            "              .-------.",
            "             / o    o  \\",
            "            |    __   | }}>",
            "             \\  `--' /",
            "              '-----'",
        ]
    else:
        head = [
            "              .-------.",
            "             / o    o  \\",
            "            |    __   |",
            "             \\  `--' /",
            "              '-----'",
        ]

    body = [
        "            _.'/     \\'._",
        "          .' /  .---.  \\ `.",
        "         /  |  |     |  |  \\",
        "        |   |  |     |  |   |",
        "         \\  \\  \\   /  /  /",
        "          \\  `. `._.' .'  /",
        "           `.  `-----'  .'",
    ]

    if rattle_spark:
        tail = [
            "             \\       /",
            "              `.   .'",
            "           * ~~SsSs~~ *",
        ]
    else:
        tail = [
            "             \\       /",
            "              `.   .'",
            "            ~~SsSsSs~~",
        ]

    all_lines = head + body + tail
    total = len(all_lines)
    result = []
    for i, line in enumerate(all_lines):
        r, g, b = _snake_gradient(i, total)
        color = f"\033[38;2;{r};{g};{b}m"
        result.append((line, color))
    return result


def _animate_startup():
    """Play the full Rattlesnake startup sequence."""
    try:
        # Phase 1: Slither-in — draw the snake body line by line from tail to head
        # (reversed build, like the snake coiling into position)
        frames_static = _render_snake_frame(tongue_out=False, rattle_spark=False)
        total_lines = len(frames_static)

        # Build from bottom up (tail appears first, head last — like it's arriving)
        for reveal_count in range(1, total_lines + 1):
            # Move cursor to overwrite
            if reveal_count > 1:
                sys.stdout.write(f"\033[{reveal_count - 1}A")

            start_idx = total_lines - reveal_count
            for j in range(reveal_count):
                line_text, color = frames_static[start_idx + j]
                sys.stdout.write(f"\033[2K    {color}{line_text}{C.RESET}\n")
            sys.stdout.flush()
            # Accelerating reveal: slower at start, faster at end
            delay = 0.06 if reveal_count < 5 else 0.03
            time.sleep(delay)

        time.sleep(0.2)

        # Phase 2: Tongue flick + rattle spark cycle (3 cycles)
        for cycle in range(6):
            tongue = (cycle % 2 == 1)
            spark = (cycle % 3 == 0)
            frame = _render_snake_frame(tongue_out=tongue, rattle_spark=spark)

            sys.stdout.write(f"\033[{total_lines}A")
            for line_text, color in frame:
                sys.stdout.write(f"\033[2K    {color}{line_text}{C.RESET}\n")
            sys.stdout.flush()
            time.sleep(0.18)

        time.sleep(0.1)

    except Exception:
        # Fallback: static print if terminal doesn't support ANSI
        for line_text, color in _render_snake_frame():
            print(f"    {color}{line_text}{C.RESET}")


def _print_title_animated():
    """Print the RATTLESNAKE title with a gradient reveal sweep."""
    title = "R A T T L E S N A K E"
    # Print the horizontal rule first
    bar_width = 46
    sys.stdout.write(f"\n    {C.DIM}")
    for i in range(bar_width):
        t = i / (bar_width - 1)
        r, g, b = _lerp_color(t, 6, 95, 70, 234, 179, 8)
        sys.stdout.write(f"\033[38;2;{r};{g};{b}m\u2500")
    sys.stdout.write(f"{C.RESET}\n")
    sys.stdout.flush()
    time.sleep(0.08)

    # Title with per-character gradient + bold
    sys.stdout.write(f"    {C.BOLD}")
    for i, ch in enumerate(title):
        t = i / max(len(title) - 1, 1)
        r, g, b = _lerp_color(t, 16, 185, 129, 250, 204, 21)
        sys.stdout.write(f"\033[38;2;{r};{g};{b}m{ch}")
        sys.stdout.flush()
        time.sleep(0.018)
    sys.stdout.write(f"{C.RESET}\n")
    sys.stdout.flush()

    # Bottom rule
    sys.stdout.write(f"    {C.DIM}")
    for i in range(bar_width):
        t = i / (bar_width - 1)
        r, g, b = _lerp_color(t, 6, 95, 70, 234, 179, 8)
        sys.stdout.write(f"\033[38;2;{r};{g};{b}m\u2500")
    sys.stdout.write(f"{C.RESET}\n")
    sys.stdout.flush()
    time.sleep(0.05)


def _print_status_line(icon_color, icon, label, value="", delay=0.03):
    """Print a single status line with a brief fade-in delay."""
    time.sleep(delay)
    if value:
        print(f"    {icon_color}{icon}{C.RESET} {C.SUBTLE}{label}{C.RESET} {C.TEXT}{value}{C.RESET}")
    else:
        print(f"    {icon_color}{icon}{C.RESET} {C.SUBTLE}{label}{C.RESET}")


_ASCII_RATTLESNAKE = [
    " ____       _   _   _                       _        ",
    "|  _ \\ __ _| |_| |_| | ___  ___ _ __   __ _| | _____ ",
    "| |_) / _` | __| __| |/ _ \\/ __| '_ \\ / _` | |/ / _ \\",
    "|  _ < (_| | |_| |_| |  __/\\__ \\ | | | (_| |   <  __/",
    "|_| \\_\\__,_|\\__|\\__|_|\\___||___/_| |_|\\__,_|_|\\_\\___|",
]


def _animate_ascii_title():
    """Animate the RATTLESNAKE ASCII art letter by letter with gradient."""
    total_cols = len(_ASCII_RATTLESNAKE[0])
    num_rows = len(_ASCII_RATTLESNAKE)

    # Reveal column by column
    for col in range(total_cols + 1):
        # Move cursor up to overwrite
        if col > 0:
            sys.stdout.write(f"\033[{num_rows}A")

        for row_idx, row in enumerate(_ASCII_RATTLESNAKE):
            visible = row[:col]
            # Color each character with gradient based on column position
            colored = ""
            for ci, ch in enumerate(visible):
                t = ci / max(total_cols - 1, 1)
                r, g, b = _lerp_color(t, 16, 185, 129, 234, 179, 8)
                colored += f"\033[38;2;{r};{g};{b}m{ch}"
            sys.stdout.write(f"\033[2K    {C.BOLD}{colored}{C.RESET}\n")

        sys.stdout.flush()
        # Fast sweep: ~1.2s total for the full reveal
        if col < 10:
            time.sleep(0.04)
        elif col < 30:
            time.sleep(0.02)
        else:
            time.sleep(0.01)


def print_banner(model):
    ver = "0.3.0"
    print()

    # --- Animated ASCII title ---
    try:
        _animate_ascii_title()
    except Exception:
        # Fallback: static print if terminal doesn't support ANSI
        for row in _ASCII_RATTLESNAKE:
            print(f"    {C.CLAW}{C.BOLD}{row}{C.RESET}")

    # --- Tagline ---
    tagline = random.choice([
        "strike fast, build faster",
        "coiled and ready to build",
        "venom-grade local AI agent",
        "zero latency, zero cloud, zero limits",
        "your code, your machine, your rules",
    ])
    print(f"    {C.DIM}v{ver} {C.ITALIC}{tagline}{C.RESET}")

    # Show provider info
    provider_name = PROVIDER.lower()
    if provider_name != "ollama":
        print(f"    {C.CLAW}Provider: {provider_name}{C.RESET} {C.SUBTLE}| Model: {model}{C.RESET}")
    print()
    print(f"    {C.DIM}Type your request, or /help for commands.{C.RESET}")
    print()

def print_help():
    print()
    print(f"  {C.CLAW}{C.BOLD}Commands{C.RESET}")
    print(f"  {C.SUBTLE}{'=' * 45}{C.RESET}")
    cmds = [
        ("/help",           "Show this help"),
        ("/model [name]",   "Show or switch model"),
        ("/add <path>",     "Attach file (image, PDF, code, etc.)"),
        ("/drop",           "Clear all attached files"),
        ("/files",          "Show attached files"),
        ("/ml",             "Multi-line input mode"),
        ("/clear",          "Clear conversation history"),
        ("/plan",           "Create / view / execute a plan"),
        ("/agents",         "Run parallel sub-agents"),
        ("/generate <file>", "Chunked/drip generation"),
        ("/mode",           "Toggle auto-accept / edit-confirm"),
        ("/memory",         "View saved memories"),
        ("/status",         "Show session info"),
        ("/undo",           "Revert last file write/edit"),
        ("/save [name]",    "Save session to disk"),
        ("/resume [id]",    "Resume a saved session"),
        ("/export [json]",  "Export conversation as markdown/json"),
        ("/git [cmd]",      "Git shortcuts (status/commit/diff/push)"),
        ("/commit [msg]",   "Git add + commit"),
        ("/diff",           "Git diff --stat"),
        ("/test",           "Detect & run project tests"),
        ("/map",            "Show codebase structure map"),
        ("/watch",          "Toggle file change monitoring"),
        ("/tokens",         "Show token usage stats"),
        ("/screenshot",     "Visual QA via vision model"),
        ("/quit",           "Exit"),
    ]
    for cmd, desc in cmds:
        print(f"    {C.CLAW}{cmd:<18}{C.RESET} {C.SUBTLE}{desc}{C.RESET}")
    print()
    print(f"  {C.CLAW}{C.BOLD}Modes{C.RESET}")
    print(f"  {C.SUBTLE}{'=' * 45}{C.RESET}")
    print(f"    {C.TEXT}auto-accept{C.RESET}        {C.SUBTLE}Tools run without asking (default){C.RESET}")
    print(f"    {C.TEXT}edit-confirm{C.RESET}       {C.SUBTLE}Write/edit/bash need your OK{C.RESET}")
    print()
    print(f"  {C.CLAW}{C.BOLD}Plan Mode{C.RESET}")
    print(f"  {C.SUBTLE}{'=' * 45}{C.RESET}")
    print(f"    {C.TEXT}/plan{C.RESET}              {C.SUBTLE}Show current plan{C.RESET}")
    print(f"    {C.TEXT}/plan create{C.RESET}       {C.SUBTLE}Ask Claw to create a plan{C.RESET}")
    print(f"    {C.TEXT}/plan execute{C.RESET}      {C.SUBTLE}Execute next unchecked step{C.RESET}")
    print(f"    {C.TEXT}/plan run-all{C.RESET}      {C.SUBTLE}Execute all remaining steps (with build verify){C.RESET}")
    print()
    print(f"  {C.CLAW}{C.BOLD}Scaffolding{C.RESET}")
    print(f"  {C.SUBTLE}{'=' * 45}{C.RESET}")
    print(f"    {C.TEXT}--scaffold <tpl>{C.RESET}   {C.SUBTLE}Scaffold from template + customize{C.RESET}")
    print(f"    {C.TEXT}--list-templates{C.RESET}   {C.SUBTLE}Show available project templates{C.RESET}")
    print(f"    {C.TEXT}--verify <dir>{C.RESET}     {C.SUBTLE}Run build-verify-fix on a project{C.RESET}")
    print()
    print(f"  {C.CLAW}{C.BOLD}Project Config{C.RESET}")
    print(f"  {C.SUBTLE}{'=' * 45}{C.RESET}")
    print(f"    {C.TEXT}CLAW.md{C.RESET}            {C.SUBTLE}Put in project root. Auto-loaded{C.RESET}")
    print(f"    {C.SUBTLE}as instructions every session (like CLAUDE.md){C.RESET}")
    print()

def main():
    global CURRENT_MODE, _file_watcher, PROVIDER, _provider_instance, _ONE_SHOT_MODE, CWD
    global OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, DASHSCOPE_API_KEY
    import argparse

    parser = argparse.ArgumentParser(description="Rattlesnake -- AI coding agent (OpenRouter by default)")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt (skip REPL)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--file", "-f", action="append", default=[],
                        help="Attach file(s) to the prompt")
    parser.add_argument("--scaffold", "-s", metavar="TEMPLATE",
                        help="Scaffold a project from a template (e.g., nextjs-supabase)")
    parser.add_argument("--list-templates", action="store_true",
                        help="List available project templates")
    parser.add_argument("--verify", metavar="DIR",
                        help="Run build-verify-fix loop on an existing project directory")
    parser.add_argument("--provider", choices=["ollama", "openrouter", "openai", "anthropic", "dashscope"],
                        default=None, help="LLM provider (default: from CLAW_PROVIDER env or openrouter)")
    parser.add_argument("--api-key", metavar="KEY", default=None,
                        help="API key for the selected cloud provider")
    parser.add_argument("--graph", nargs="?", const="__stats__", default=None, metavar="FILE",
                        help="Dump project graph stats (no arg) or a file's edges (with arg)")
    args = parser.parse_args()

    # Apply provider overrides from CLI flags
    if args.provider:
        global PROVIDER, _provider_instance
        PROVIDER = args.provider
        _provider_instance = None  # reset cached provider
    if args.api_key:
        global OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, DASHSCOPE_API_KEY
        p = PROVIDER.lower()
        if p == "openrouter":
            OPENROUTER_API_KEY = args.api_key
        elif p == "anthropic":
            ANTHROPIC_API_KEY = args.api_key
        elif p == "openai":
            OPENAI_API_KEY = args.api_key
        elif p == "dashscope":
            DASHSCOPE_API_KEY = args.api_key

    model = args.model
    vision_model = DEFAULT_VISION_MODEL

    # --- graph debug mode (no Ollama needed) ---
    if args.graph is not None:
        try:
            graph = ProjectGraph(CWD)
            if not graph.load():
                print(f"  {C.CLAW}Building project graph...{C.RESET}")
                graph.build_full()
                graph.save()
            if args.graph == "__stats__":
                print(graph.inspect())
            else:
                print(graph.inspect(args.graph))
        except Exception as e:
            print(f"  {C.ERROR}Graph error: {e}{C.RESET}")
        return

    # --- list templates (no Ollama needed) ---
    if args.list_templates:
        templates = list_templates()
        if not templates:
            print(f"  {C.SUBTLE}No templates found.{C.RESET}")
            print(f"  {C.SUBTLE}Add templates to: {TEMPLATES_DIRS[0]}{C.RESET}")
        else:
            print(f"\n  {C.CLAW}{C.BOLD}Available Templates{C.RESET}")
            print(f"  {C.SUBTLE}{'=' * 50}{C.RESET}")
            for tname, tinfo in templates.items():
                print(f"    {C.CLAW}{C.BOLD}{tname}{C.RESET}")
                print(f"      {C.TEXT}{tinfo.get('description', '')}{C.RESET}")
                stack = tinfo.get('stack', [])
                if stack:
                    print(f"      {C.SUBTLE}Stack: {', '.join(stack)}{C.RESET}")
                features = tinfo.get('features', [])
                if features:
                    print(f"      {C.SUBTLE}Features: {', '.join(features)}{C.RESET}")
                print()
            print(f"  {C.SUBTLE}Usage: claw --scaffold <template> \"project description\"{C.RESET}")
        return

    # verify provider is reachable
    if PROVIDER.lower() == "ollama":
        try:
            urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5)
        except Exception:
            print(f"  {C.ERROR}{BLACK_CIRCLE} Cannot reach Ollama at {OLLAMA_BASE}{C.RESET}")
            print(f"  {C.SUBTLE}Make sure Ollama is running: ollama serve{C.RESET}")
            sys.exit(1)
    elif PROVIDER.lower() == "openrouter" and not OPENROUTER_API_KEY:
        print(f"  {C.ERROR}{BLACK_CIRCLE} OPENROUTER_API_KEY not set{C.RESET}")
        print(f"  {C.SUBTLE}Export OPENROUTER_API_KEY or use --api-key{C.RESET}")
        sys.exit(1)
    elif PROVIDER.lower() == "anthropic" and not ANTHROPIC_API_KEY:
        print(f"  {C.ERROR}{BLACK_CIRCLE} ANTHROPIC_API_KEY not set{C.RESET}")
        print(f"  {C.SUBTLE}Export ANTHROPIC_API_KEY or use --api-key{C.RESET}")
        sys.exit(1)
    elif PROVIDER.lower() == "openai" and not OPENAI_API_KEY:
        print(f"  {C.ERROR}{BLACK_CIRCLE} OPENAI_API_KEY not set{C.RESET}")
        print(f"  {C.SUBTLE}Export OPENAI_API_KEY or use --api-key{C.RESET}")
        sys.exit(1)

    # --- scaffold mode ---
    if args.scaffold:
        template_name = args.scaffold
        project_desc = " ".join(args.prompt) if args.prompt else ""
        if not project_desc:
            print(f"  {C.ERROR}Usage: claw --scaffold {template_name} \"project description\"{C.RESET}")
            print(f"  {C.SUBTLE}Example: claw --scaffold nextjs-supabase \"booking platform for yoga studios\"{C.RESET}")
            sys.exit(1)

        # derive project name from description
        project_name = re.sub(r'[^a-z0-9]+', '-', project_desc.lower()).strip('-')[:40]
        if not project_name:
            project_name = template_name + "-project"

        # Scaffold is non-interactive — suppress ask_user so models don't loop
        _ONE_SHOT_MODE = True

        print(f"\n  {C.CLAW}{C.BOLD}Scaffolding Project{C.RESET}")
        print(f"  {C.SUBTLE}Template:    {template_name}{C.RESET}")
        print(f"  {C.SUBTLE}Project:     {project_name}{C.RESET}")
        print(f"  {C.SUBTLE}Description: {project_desc}{C.RESET}")
        print()

        success, msg = scaffold_from_template(template_name, project_name)
        if not success:
            print(f"  {C.ERROR}{BLACK_CIRCLE} {msg}{C.RESET}")
            sys.exit(1)

        print(f"  {C.SUCCESS}{BLACK_CIRCLE} {msg}{C.RESET}")

        # Now let the AI customize it based on the description
        project_dir = Path(CWD) / project_name
        system_prompt = _build_prompt_for_mode("scaffold", token_budget=2500)

        # inject API patterns relevant to the project
        api_context = get_api_context_for_prompt(project_desc)
        if api_context:
            system_prompt += f"\n\n# API Patterns for this project\n{api_context}"

        # inject design context (color palette, Tailwind recipes, anti-slop rules)
        design_selections = {}
        design_context = _build_design_context(project_desc, out_selections=design_selections, max_chars=4000)
        if design_context:
            system_prompt += f"\n\n{design_context}"

        # inject security context (headers, middleware, validation patterns)
        security_context = _build_security_context(project_desc, scaffold_mode=True)
        if security_context:
            system_prompt += f"\n\n{security_context}"

        # inject template info
        tpl_path, tpl_meta = load_template_meta(template_name)
        if tpl_meta:
            system_prompt += f"\n\n# Template Used: {template_name}\n"
            system_prompt += f"Description: {tpl_meta.get('description', '')}\n"
            system_prompt += f"Features: {', '.join(tpl_meta.get('features', []))}\n"
            system_prompt += "The template has been scaffolded. Your job is to CUSTOMIZE it for the user's specific needs.\n"

        messages = [{"role": "system", "content": system_prompt}]

        # Generate build spec — the blueprint for what to create
        build_spec = _generate_build_spec(template_name, project_desc, str(project_dir),
                                          design_selections=design_selections)

        customize_prompt = (
            f"A project has been scaffolded from the '{template_name}' template into: {project_dir}\n\n"
            f"The user wants to build: {project_desc}\n\n"
            f"The template provides: Auth (Supabase login/signup), Stripe payments, middleware, Tailwind CSS.\n"
            f"DO NOT rewrite auth, Stripe webhooks, or middleware — they already work.\n\n"
            f"{build_spec}\n\n"
            f"## EXECUTION RULES\n"
            f"- DO NOT use ask_user. This is non-interactive. No one will answer.\n"
            f"- Create files ONE AT A TIME using write_file. Do NOT batch.\n"
            f"- Follow the FILE CREATION ORDER exactly. Do NOT skip ahead.\n"
            f"- Do NOT edit package.json until AFTER creating all app files.\n"
            f"- If you need a new dependency, note it and add it at the END.\n"
            f"- Each file MUST compile — no TypeScript errors, no missing imports.\n"
            f"- Use dark-mode-compatible colors everywhere — NEVER hardcode text-gray-900 or bg-white without dark: variants.\n"
            f"- Load custom fonts with next/font/google in layout.tsx.\n"
            f"- Every input/button MUST have working event handlers — no dead UI in server components.\n"
            f"- Every API route MUST have real working logic — no stubs, no TODOs.\n"
            f"- Data visualization components (charts, timelines, calendars) MUST derive their display range from the actual data, not hardcoded offsets. Example: a Gantt chart date range should span from the earliest task start_date to the latest task due_date, not a fixed ±N days from today.\n"
            f"- All API route handlers MUST wrap database operations in try/catch and return proper error responses: 400 for validation failure, 401 for unauthorized, 404 for not found, 500 for unexpected errors. Always return {{ success: false, error: message }}.\n\n"
            f"START BUILDING IMMEDIATELY. Follow the BUILD SPEC above. Create each file in order."
        )
        messages.append({"role": "user", "content": customize_prompt})

        print(f"\n  {C.CLAW}{BLACK_CIRCLE} Launching AI to customize the project...{C.RESET}\n")
        # Set CWD to project directory so write_file resolves relative paths
        _original_cwd = CWD
        CWD = str(project_dir)
        try:
            if _is_local_model(model):
                # Multi-agent scaffold: file-by-file with shared memory
                new_count, new_files = _scaffold_with_agents(
                    project_desc, template_name, str(project_dir), model,
                    build_spec, design_selections=design_selections
                )
                if new_count > 0:
                    print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} Created {new_count} files via agent scaffold{C.RESET}")
                else:
                    print(f"\n  {C.WARNING}{BLACK_CIRCLE} No new files created during customization{C.RESET}")
            else:
                # Cloud model: existing monolithic flow (unchanged)
                scaffold_ok, new_count, new_files = _run_agent_turn_with_fallback(
                    messages, model, use_tools=True, project_dir=str(project_dir)
                )
                if not scaffold_ok:
                    print(f"\n  {C.ERROR}{BLACK_CIRCLE} Scaffold customization failed — all providers exhausted{C.RESET}")
                    return
                if new_count > 0:
                    print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} Created {new_count} new files{C.RESET}")
                else:
                    print(f"\n  {C.WARNING}{BLACK_CIRCLE} No new files created during customization{C.RESET}")
        except KeyboardInterrupt:
            print(f"\n  {C.WARNING}{BLACK_CIRCLE} Interrupted{C.RESET}")
            return
        except Exception as e:
            print(f"\n  {C.ERROR}{BLACK_CIRCLE} Error during customization: {e}{C.RESET}")
        finally:
            CWD = _original_cwd

        # Post-customization: enhance template files the model didn't touch
        _post_scaffold_enhance(project_dir, design_selections=design_selections)

        # Repair any corrupted JSON files
        repaired = _repair_json_files(str(project_dir))
        if repaired:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Repaired JSON in: {', '.join(repaired)}{C.RESET}")

        # Run build verification + wiring agent (with auto-stub for missing components)
        print(f"\n  {C.CLAW}{BLACK_CIRCLE} Running post-build verification...{C.RESET}")
        try:
            success, summary = build_and_verify(str(project_dir), messages, model, auto_stub=True)
            print(f"\n  {C.CLAW}{C.BOLD}Build Verification Results{C.RESET}")
            for line in summary.split("\n"):
                print(f"    {line}")
            if success:
                print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} All checks passed!{C.RESET}")
            else:
                print(f"\n  {C.WARNING}{BLACK_CIRCLE} Some checks failed — review above.{C.RESET}")
        except Exception as e:
            print(f"  {C.ERROR}{BLACK_CIRCLE} Verification error: {e}{C.RESET}")
        return

    # --- verify mode ---
    if args.verify:
        print(f"\n  {C.CLAW}{C.BOLD}Build Verification{C.RESET}")
        print(f"  {C.SUBTLE}Project: {args.verify}{C.RESET}\n")
        system_prompt = build_system_prompt()
        verify_messages = [{"role": "system", "content": system_prompt}]
        success, summary = build_and_verify(args.verify, verify_messages, model)
        print(f"\n  {C.CLAW}{C.BOLD}Results{C.RESET}")
        for line in summary.split("\n"):
            print(f"    {line}")
        if success:
            print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} All checks passed!{C.RESET}")
        else:
            print(f"\n  {C.WARNING}{BLACK_CIRCLE} Some checks failed.{C.RESET}")
        return

    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # Run memory maintenance at startup
    _maybe_run_maintenance()

    # one-shot mode
    if args.prompt and not args.scaffold:
        _ONE_SHOT_MODE = True
        prompt_text = " ".join(args.prompt)
        attachments = [Attachment(f) for f in args.file]
        for att in attachments:
            print(att.summary())

        # Inject one-shot directive — suppress questions, force action
        messages[0]["content"] += dedent("""

        ## ONE-SHOT MODE (NON-INTERACTIVE)
        You are running in one-shot non-interactive mode. There is NO user to answer questions.

        CRITICAL RULES FOR ONE-SHOT MODE:
        - DO NOT use ask_user. It will not work — no one is available to respond.
        - DO NOT ask for confirmation or approval. Just build.
        - DO NOT create PLAN.md — skip ALL discovery steps (1-6) and go straight to building.
        - DO NOT explain what you're going to do — just DO IT with tool calls.
        - If anything is ambiguous, make your best judgment call and proceed.
        - The user's prompt contains ALL the requirements. Treat it as the complete spec.
        - Focus 100% on writing files and running commands. No planning, no questions.
        - After writing package.json or adding dependencies, IMMEDIATELY run `npm install` or `pip install`.
        - After creating ALL files, verify the project builds by running the build command.
        - You MUST create EVERY file the user asked for. Do not stop partway through.
        """)

        # inject relevant API context based on the prompt
        api_context = get_api_context_for_prompt(prompt_text)
        if api_context:
            messages[0]["content"] += f"\n\n# API Patterns (relevant to this request)\n{api_context}"

        # inject warm memory context based on prompt keywords
        warm_context = _auto_search_warm_memories(prompt_text)
        if warm_context:
            messages.append({"role": "user", "content": f"[SYSTEM: Related context from memory]\n{warm_context}"})
            messages.append({"role": "assistant", "content": "Noted, I'll keep this context in mind."})

        msg, override_model = build_user_message(prompt_text, attachments, vision_model)
        messages.append(msg)
        turn_model = override_model or model
        use_tools = (override_model is None)  # no tools for vision-only queries
        run_agent_turn(messages, turn_model, use_tools=use_tools)
        return

    # Load plugins before REPL starts
    plugin_count = _load_plugins()

    # REPL mode
    print_banner(model)
    if plugin_count:
        print(f"    {C.SUCCESS}{BLACK_CIRCLE}{C.RESET} {C.DIM}{plugin_count} plugin(s) loaded{C.RESET}")

    turn_count = 0
    pending_attachments = []  # files queued via /add
    _last_turn_tokens = None  # (prompt_delta, completion_delta) from last turn

    while True:
        try:
            # show file watcher changes if any
            if _file_watcher:
                changes = _file_watcher.pop_changes()
                if changes:
                    print(f"  {C.WARNING}{BLACK_CIRCLE} File changes detected:{C.RESET}")
                    for fp, change_type in changes[:10]:
                        icon = {
                            "created": f"{C.SUCCESS}+{C.RESET}",
                            "modified": f"{C.WARNING}~{C.RESET}",
                            "deleted": f"{C.ERROR}-{C.RESET}",
                        }.get(change_type, "?")
                        print(f"    {icon} {C.DIM}{fp}{C.RESET}")
                    if len(changes) > 10:
                        print(f"    {C.SUBTLE}... and {len(changes) - 10} more{C.RESET}")
                    # Re-capture git context on file changes (Phase 3b)
                    try:
                        _capture_git_context()
                    except Exception:
                        pass
            # show persistent status bar above the input
            if turn_count > 0:
                _status_bar(model=model, turn=turn_count, tokens=_token_tracker.total, mode=str(CURRENT_MODE), turn_tokens=_last_turn_tokens)
            # show attachment indicator above the input box
            if pending_attachments:
                att_names = ", ".join(a.name for a in pending_attachments)
                print(f"  {C.TOOL}{BLACK_CIRCLE} {len(pending_attachments)} file(s): {att_names}{C.RESET}")
            box_meta = None
            try:
                user_input, box_meta = _draw_input_box(turn_count)
            except (EOFError, KeyboardInterrupt):
                raise
            finally:
                if box_meta is not None:
                    _clear_input_box(box_meta)
        except (EOFError, KeyboardInterrupt):
            # Auto-save session on exit
            if messages and turn_count > 0:
                try:
                    _save_session(messages)
                    print(f"\n  {C.DIM}Session auto-saved.{C.RESET}")
                except Exception:
                    pass
            if _file_watcher:
                _file_watcher.stop()
            print(f"\n{C.DIM}Goodbye!{C.RESET}")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # --- slash commands ---
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print(f"{C.DIM}Goodbye!{C.RESET}")
                break

            elif cmd == "/help":
                print_help()
                continue

            elif cmd == "/clear":
                messages = [{"role": "system", "content": system_prompt}]
                pending_attachments = []
                turn_count = 0
                print(f"  {C.SUCCESS}{BLACK_CIRCLE} Conversation cleared.{C.RESET}")
                continue

            elif cmd == "/model":
                if cmd_arg:
                    model = cmd_arg
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Model: {model}{C.RESET}")
                else:
                    print(f"  {C.SUBTLE}Model: {model}{C.RESET}")
                    print(f"  {C.SUBTLE}Vision: {vision_model}{C.RESET}")
                continue

            elif cmd in ("/add", "/attach", "/file"):
                if not cmd_arg:
                    print(f"  {C.ERROR}Usage: /add <file_path>  or  /add *.py{C.RESET}")
                    continue
                # support glob in /add
                resolved_arg = _resolve(cmd_arg)
                if "*" in cmd_arg or "?" in cmd_arg:
                    matches = glob_mod.glob(str(resolved_arg), recursive=True)
                    if not matches:
                        print(f"  {C.ERROR}No files matched: {cmd_arg}{C.RESET}")
                        continue
                    for m in matches[:20]:
                        if Path(m).is_file():
                            att = Attachment(m)
                            pending_attachments.append(att)
                            print(f"  {att.summary()}")
                    if len(matches) > 20:
                        cprint(C.DIM, f"  ... and {len(matches)-20} more (showing first 20)")
                else:
                    att = Attachment(cmd_arg)
                    pending_attachments.append(att)
                    print(f"  {att.summary()}")
                continue

            elif cmd in ("/drop", "/detach"):
                if pending_attachments:
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Dropped {len(pending_attachments)} file(s).{C.RESET}")
                    pending_attachments = []
                else:
                    cprint(C.DIM, "No files attached.")
                continue

            elif cmd == "/files":
                if pending_attachments:
                    for att in pending_attachments:
                        print(f"  {att.summary()}")
                else:
                    cprint(C.DIM, "No files attached.")
                continue

            elif cmd == "/ml":
                user_input = read_multiline()
                if not user_input.strip():
                    continue
                # fall through to process as normal input

            elif cmd == "/memory":
                _ensure_memory_dir()
                sub = cmd_arg.lower().strip()
                if sub == "clear":
                    import shutil
                    shutil.rmtree(MEMORY_DIR, ignore_errors=True)
                    _ensure_memory_dir()
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} All memories cleared.{C.RESET}")
                elif sub == "stats":
                    entries = _load_all_memories()
                    hot = sum(1 for e in entries if e.get("tier") == "hot")
                    warm = sum(1 for e in entries if e.get("tier") == "warm")
                    cold = sum(1 for e in entries if e.get("tier") == "cold")
                    pinned = sum(1 for e in entries if e.get("pinned", False))
                    print(f"  {C.CLAW}{C.BOLD}Memory Stats{C.RESET}")
                    print(f"    Hot:    {hot}")
                    print(f"    Warm:   {warm}")
                    print(f"    Cold:   {cold}")
                    print(f"    Pinned: {pinned}")
                    print(f"    Total:  {len(entries)}")
                elif sub in ("hot", "warm", "cold"):
                    entries = _load_all_memories()
                    tier_entries = [e for e in entries if e.get("tier") == sub]
                    if not tier_entries:
                        print(f"  {C.SUBTLE}No {sub} memories.{C.RESET}")
                    else:
                        print(f"  {C.CLAW}{C.BOLD}{sub.title()} Memories{C.RESET}")
                        for e in tier_entries:
                            pin = " [pinned]" if e.get("pinned") else ""
                            ac = e.get("access_count", 0)
                            print(f"    {C.TEXT}[{e.get('category','?')}] {e.get('key','?')}: {e.get('value','')[:100]}{pin} (accesses: {ac}){C.RESET}")
                elif sub.startswith("pin "):
                    pin_key = cmd_arg[4:].strip()
                    entries = _load_all_memories()
                    found = False
                    for e in entries:
                        if e.get("key", "").lower() == pin_key.lower():
                            e["pinned"] = True
                            e["tier"] = "hot"
                            e["demotion_strikes"] = 0
                            _save_memory_entry(e, Path(e["_filepath"]))
                            print(f"  {C.SUCCESS}{BLACK_CIRCLE} Pinned: {e.get('key')}{C.RESET}")
                            found = True
                            break
                    if not found:
                        print(f"  {C.SUBTLE}Memory not found: {pin_key}{C.RESET}")
                elif sub.startswith("unpin "):
                    unpin_key = cmd_arg[6:].strip()
                    entries = _load_all_memories()
                    found = False
                    for e in entries:
                        if e.get("key", "").lower() == unpin_key.lower():
                            e["pinned"] = False
                            _save_memory_entry(e, Path(e["_filepath"]))
                            print(f"  {C.SUCCESS}{BLACK_CIRCLE} Unpinned: {e.get('key')}{C.RESET}")
                            found = True
                            break
                    if not found:
                        print(f"  {C.SUBTLE}Memory not found: {unpin_key}{C.RESET}")
                elif sub.startswith("promote "):
                    prom_key = cmd_arg[8:].strip()
                    entries = _load_all_memories()
                    found = False
                    for e in entries:
                        if e.get("key", "").lower() == prom_key.lower():
                            cur = e.get("tier", "warm")
                            if cur == "cold":
                                e["tier"] = "warm"
                            elif cur == "warm":
                                e["tier"] = "hot"
                            else:
                                print(f"  {C.SUBTLE}Already hot.{C.RESET}")
                                found = True
                                break
                            e["demotion_strikes"] = 0
                            _save_memory_entry(e, Path(e["_filepath"]))
                            print(f"  {C.SUCCESS}{BLACK_CIRCLE} Promoted to {e['tier']}: {e.get('key')}{C.RESET}")
                            found = True
                            break
                    if not found:
                        print(f"  {C.SUBTLE}Memory not found: {prom_key}{C.RESET}")
                elif sub.startswith("demote "):
                    dem_key = cmd_arg[7:].strip()
                    entries = _load_all_memories()
                    found = False
                    for e in entries:
                        if e.get("key", "").lower() == dem_key.lower():
                            cur = e.get("tier", "warm")
                            if cur == "hot":
                                e["tier"] = "warm"
                            elif cur == "warm":
                                e["tier"] = "cold"
                            else:
                                print(f"  {C.SUBTLE}Already cold.{C.RESET}")
                                found = True
                                break
                            e["demotion_strikes"] = 0
                            _save_memory_entry(e, Path(e["_filepath"]))
                            print(f"  {C.SUCCESS}{BLACK_CIRCLE} Demoted to {e['tier']}: {e.get('key')}{C.RESET}")
                            found = True
                            break
                    if not found:
                        print(f"  {C.SUBTLE}Memory not found: {dem_key}{C.RESET}")
                else:
                    result = tool_memory_search({"query": cmd_arg})
                    if result == "No memories found.":
                        print(f"  {C.SUBTLE}No memories saved yet.{C.RESET}")
                    else:
                        print(f"  {C.CLAW}{C.BOLD}Memories{C.RESET}")
                        for line in result.split("\n"):
                            print(f"    {C.TEXT}{line}{C.RESET}")
                continue

            elif cmd == "/mode":
                if cmd_arg.lower() in ("auto", "auto-accept", "accept"):
                    CURRENT_MODE = PermissionMode.AUTO_ACCEPT
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Mode: auto-accept (all tools run freely){C.RESET}")
                elif cmd_arg.lower() in ("edit", "edit-confirm", "confirm", "safe"):
                    CURRENT_MODE = PermissionMode.EDIT_CONFIRM
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Mode: edit-confirm (write/edit/bash need approval){C.RESET}")
                else:
                    print(f"  {C.SUBTLE}Current mode: {C.TEXT}{CURRENT_MODE}{C.RESET}")
                    print(f"  {C.SUBTLE}Use: /mode auto-accept  or  /mode edit-confirm{C.RESET}")
                continue

            elif cmd == "/think":
                global THINKING_ENABLED
                if cmd_arg.lower() in ("off", "0", "false"):
                    THINKING_ENABLED = False
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Thinking: OFF{C.RESET}")
                elif cmd_arg.lower() in ("on", "1", "true"):
                    THINKING_ENABLED = True
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Thinking: ON{C.RESET}")
                else:
                    print(f"  {C.SUBTLE}Thinking: {'ON' if THINKING_ENABLED else 'OFF'}{C.RESET}")
                    print(f"  {C.SUBTLE}Use: /think on  or  /think off{C.RESET}")
                continue

            elif cmd == "/reflect":
                global REFLECTION_ENABLED
                if cmd_arg.lower() in ("off", "0", "false"):
                    REFLECTION_ENABLED = False
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Reflection: OFF{C.RESET}")
                elif cmd_arg.lower() in ("on", "1", "true"):
                    REFLECTION_ENABLED = True
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Reflection: ON{C.RESET}")
                else:
                    print(f"  {C.SUBTLE}Reflection: {'ON' if REFLECTION_ENABLED else 'OFF'}{C.RESET}")
                    print(f"  {C.SUBTLE}Use: /reflect on  or  /reflect off{C.RESET}")
                continue

            elif cmd == "/quality":
                global QUALITY_GATE_ENABLED
                if cmd_arg.lower() in ("off", "0", "false"):
                    QUALITY_GATE_ENABLED = False
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Quality gate: OFF{C.RESET}")
                elif cmd_arg.lower() in ("on", "1", "true"):
                    QUALITY_GATE_ENABLED = True
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Quality gate: ON{C.RESET}")
                else:
                    print(f"  {C.SUBTLE}Quality gate: {'ON' if QUALITY_GATE_ENABLED else 'OFF'}{C.RESET}")
                    print(f"  {C.SUBTLE}Use: /quality on  or  /quality off{C.RESET}")
                continue

            elif cmd == "/wiring":
                global WIRING_ENABLED
                if cmd_arg.lower() in ("off", "0", "false"):
                    WIRING_ENABLED = False
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Wiring checks: OFF{C.RESET}")
                elif cmd_arg.lower() in ("on", "1", "true"):
                    WIRING_ENABLED = True
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Wiring checks: ON{C.RESET}")
                elif cmd_arg.lower() in ("scan", "run", "check"):
                    print(f"  {C.CLAW}{BLACK_CIRCLE} Running wiring scan...{C.RESET}")
                    agent = WiringAgent(CWD)
                    agent.run_full_scan()
                    if agent.issues:
                        agent.auto_fix()
                        _display_wiring_report(agent)
                    else:
                        print(f"  {C.SUCCESS}✓ No wiring issues found{C.RESET}")
                else:
                    print(f"  {C.SUBTLE}Wiring: {'ON' if WIRING_ENABLED else 'OFF'}{C.RESET}")
                    print(f"  {C.SUBTLE}Use: /wiring on|off|scan{C.RESET}")
                continue

            elif cmd == "/plan":
                plan_path = Path(CWD) / "PLAN.md"
                sub = cmd_arg.lower()

                if sub == "create":
                    # --- Auto-generate PLAN.md from conversation context ---
                    # Extract conversation context (user messages) for plan generation
                    conv_context = _extract_conversation_context_for_plan(messages)

                    # If no conversation context, fall back to asking the user
                    if not conv_context:
                        print(f"  {C.SUBTLE}What should the plan be for?{C.RESET}")
                        sys.stdout.write(f"  {C.CLAW}describe>{C.RESET} ")
                        sys.stdout.flush()
                        try:
                            plan_desc = input().strip()
                        except (EOFError, KeyboardInterrupt):
                            continue
                        if not plan_desc:
                            continue
                        conv_context = plan_desc

                    print(f"  {C.CLAW}{BLACK_CIRCLE} Generating PLAN.md from conversation...{C.RESET}")

                    # ask the model to create a plan from the conversation context
                    plan_prompt = (
                        f"Based on our conversation, create a detailed step-by-step plan.\n\n"
                        f"## Conversation Context\n{conv_context}\n\n"
                        f"Write the plan to PLAN.md using write_file. Use this EXACT format:\n"
                        f"# Plan: [descriptive title]\n\n"
                        f"## Overview\n[1-2 sentence summary of what we're building and why]\n\n"
                        f"## Tech Stack\n[List the technologies/frameworks decided on]\n\n"
                        f"## Steps\n\n"
                        f"- [ ] Step 1: [specific actionable step]\n"
                        f"- [ ] Step 2: [specific actionable step]\n"
                        f"- [ ] Step 3: [specific actionable step]\n"
                        f"...\n\n"
                        f"Rules:\n"
                        f"- Each step must be specific, actionable, and completable in one agent turn\n"
                        f"- Include ALL decisions and requirements from the conversation\n"
                        f"- Order steps by dependency (what must come first)\n"
                        f"- Mark steps that can run in parallel with (parallel) tag\n"
                        f"- Include file paths where known\n"
                    )
                    messages.append({"role": "user", "content": plan_prompt})
                    turn_count += 1
                    try:
                        run_agent_turn(messages, model, use_tools=True)
                    except Exception as e:
                        print(f"  {C.ERROR}{BLACK_CIRCLE} Error: {e}{C.RESET}")

                    # reload plan
                    ACTIVE_PLAN_FILE = str(plan_path) if plan_path.exists() else None
                    if plan_path.exists():
                        print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} Plan created: PLAN.md{C.RESET}")
                    print()

                elif sub in ("execute", "next", "step"):
                    # execute the next unchecked step
                    if not plan_path.exists():
                        print(f"  {C.ERROR}No PLAN.md found. Use /plan create first.{C.RESET}")
                        continue

                    plan_content = plan_path.read_text(encoding="utf-8", errors="replace")
                    # find first unchecked item
                    unchecked = [l for l in plan_content.split("\n") if l.strip().startswith("- [ ]")]
                    if not unchecked:
                        print(f"  {C.SUCCESS}{BLACK_CIRCLE} All plan steps are complete!{C.RESET}")
                        continue

                    next_step = unchecked[0].replace("- [ ]", "").strip()
                    print(f"  {C.TOOL}{BLACK_CIRCLE} Executing: {next_step}{C.RESET}")

                    step_prompt = (
                        f"Execute this plan step: {next_step}\n\n"
                        f"After completing it, use edit_file to update PLAN.md and change "
                        f"'- [ ] {next_step}' to '- [x] {next_step}' to mark it done."
                    )
                    messages.append({"role": "user", "content": step_prompt})
                    turn_count += 1
                    try:
                        run_agent_turn(messages, model, use_tools=True)
                    except Exception as e:
                        print(f"  {C.ERROR}{BLACK_CIRCLE} Error: {e}{C.RESET}")
                    print()

                elif sub in ("run-all", "runall", "all"):
                    # execute all remaining steps with PARALLEL EXECUTION + CHUNKED CONTEXT
                    if not plan_path.exists():
                        print(f"  {C.ERROR}No PLAN.md found. Use /plan create first.{C.RESET}")
                        continue

                    # reset session manifest for this plan run
                    _session_manifest.files_created.clear()
                    _session_manifest.files_modified.clear()
                    _session_manifest.steps_completed.clear()

                    # detect if plan mentions APIs/design that need registry injection
                    full_plan = plan_path.read_text(encoding="utf-8", errors="replace")
                    api_context = get_api_context_for_prompt(full_plan)
                    design_context = get_api_context_for_prompt("website ui design layout")

                    # analyze plan for parallel execution opportunities
                    step_groups = decompose_plan_for_parallel(full_plan)
                    total_steps = sum(len(g) for g in step_groups)
                    completed_steps = 0

                    if step_groups:
                        parallel_count = sum(1 for g in step_groups if len(g) > 1)
                        if parallel_count > 0:
                            print(f"  {C.CLAW}{BLACK_CIRCLE} Plan analysis: {total_steps} steps in {len(step_groups)} groups ({parallel_count} parallel){C.RESET}")

                    group_num = 0
                    for group in step_groups:
                        group_num += 1
                        plan_content = plan_path.read_text(encoding="utf-8", errors="replace")

                        # check if steps in this group are still unchecked
                        unchecked_lines = [l.strip() for l in plan_content.split("\n") if l.strip().startswith("- [ ]")]
                        unchecked_texts = [l.replace("- [ ]", "").strip() for l in unchecked_lines]
                        group_steps = [s for s in group if s in unchecked_texts]

                        if not group_steps:
                            continue

                        if len(group_steps) > 1:
                            # --- PARALLEL EXECUTION ---
                            print(f"\n  {C.CLAW}{C.BOLD}{BLACK_CIRCLE} Group {group_num}: Running {len(group_steps)} steps in parallel{C.RESET}")
                            for s in group_steps:
                                print(f"    {C.SUBTLE}- {s}{C.RESET}")

                            # build agent tasks with full context
                            agent_tasks = []
                            for step in group_steps:
                                step_api_ctx = get_api_context_for_prompt(step)
                                manifest_summary = _session_manifest.get_context_summary()

                                ctx_parts = [f"## Full Plan\n{plan_content}"]
                                if manifest_summary:
                                    ctx_parts.append(f"## Done so far\n{manifest_summary}")
                                if step_api_ctx:
                                    ctx_parts.append(f"## API Reference\n{step_api_ctx}")
                                elif api_context:
                                    ctx_parts.append(f"## API Reference\n{api_context}")
                                if design_context:
                                    ctx_parts.append(f"## Design System\n{design_context}")

                                ctx = "\n\n".join(ctx_parts)
                                agent_prompt = (
                                    f"Execute this plan step: {step}\n\n"
                                    f"--- CONTEXT ---\n{ctx}\n--- END CONTEXT ---\n\n"
                                    f"CRITICAL RULES:\n"
                                    f"- Write COMPLETE code. No TODOs. No placeholders. No pass statements.\n"
                                    f"- Follow the design system. Modern, clean UI. No AI slop.\n"
                                    f"- Focus ONLY on this step. Do not do other steps.\n"
                                    f"- After completing, use edit_file to mark this step done in PLAN.md:\n"
                                    f"  Change '- [ ] {step}' to '- [x] {step}'"
                                )
                                agent_tasks.append((step[:55] + "..." if len(step) > 55 else step, agent_prompt))

                            try:
                                agent_results = run_subagents(agent_tasks, model, system_prompt)
                                for i, r in enumerate(agent_results):
                                    if r.status == "done":
                                        _session_manifest.steps_completed.append(group_steps[i])
                                        completed_steps += 1
                                    elif r.status == "error":
                                        print(f"    {C.ERROR}Step failed: {group_steps[i][:60]} — {r.error}{C.RESET}")
                                # add results to conversation context
                                summary_parts = []
                                for r in agent_results:
                                    if r.output:
                                        summary_parts.append(f"[Agent: {r.name}]\n{r.output[:500]}")
                                if summary_parts:
                                    messages.append({"role": "assistant", "content": "\n\n".join(summary_parts)})
                            except KeyboardInterrupt:
                                print(f"\n  {C.WARNING}{BLACK_CIRCLE} Plan execution interrupted.{C.RESET}")
                                break
                            except Exception as e:
                                print(f"  {C.ERROR}{BLACK_CIRCLE} Parallel execution error: {e}{C.RESET}")
                                # fall back to sequential for remaining
                                pass
                        else:
                            # --- SEQUENTIAL EXECUTION (single step or dependent step) ---
                            step = group_steps[0]
                            completed_steps += 1
                            print(f"\n  {C.TOOL}{BLACK_CIRCLE} Step {completed_steps}/{total_steps}: {step}{C.RESET}")

                            # build rich context
                            context_parts = [f"## Current Plan (PLAN.md)\n{plan_content}"]

                            manifest_summary = _session_manifest.get_context_summary()
                            if manifest_summary:
                                context_parts.append(f"## What's been done so far\n{manifest_summary}")

                            step_api_context = get_api_context_for_prompt(step)
                            if step_api_context:
                                context_parts.append(f"## API Reference (use these exact patterns)\n{step_api_context}")
                            elif api_context and completed_steps <= 3:
                                context_parts.append(f"## API Reference\n{api_context}")

                            if design_context and any(kw in step.lower() for kw in ("page", "ui", "component", "layout", "style", "css", "html", "frontend", "design")):
                                context_parts.append(f"## Design System (FOLLOW THIS)\n{design_context}")

                            # read key existing files for context
                            relevant_files = []
                            for created_file in _session_manifest.files_created[-5:]:
                                fp = _resolve(created_file)
                                if fp.exists() and fp.stat().st_size < 3000:
                                    try:
                                        fc = fp.read_text(encoding="utf-8", errors="replace")
                                        relevant_files.append(f"### {created_file}\n```\n{fc}\n```")
                                    except Exception:
                                        pass
                            if relevant_files:
                                context_parts.append(f"## Key existing files\n" + "\n".join(relevant_files[-3:]))

                            context_block = "\n\n".join(context_parts)

                            step_prompt = (
                                f"Execute this plan step: {step}\n\n"
                                f"--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---\n\n"
                                f"CRITICAL: Write COMPLETE code. No TODOs. No placeholders. Follow the design system.\n"
                                f"After completing it, use edit_file to update PLAN.md and change "
                                f"'- [ ] {step}' to '- [x] {step}' to mark it done."
                            )
                            messages.append({"role": "user", "content": step_prompt})
                            turn_count += 1
                            try:
                                run_agent_turn(messages, model, use_tools=True)
                                _session_manifest.steps_completed.append(step)
                            except KeyboardInterrupt:
                                print(f"\n  {C.WARNING}{BLACK_CIRCLE} Plan execution interrupted.{C.RESET}")
                                break
                            except Exception as e:
                                print(f"  {C.ERROR}{BLACK_CIRCLE} Error: {e}{C.RESET}")
                                break

                    # --- BUILD-VERIFY-FIX LOOP after plan completion ---
                    plan_content_final = plan_path.read_text(encoding="utf-8", errors="replace")
                    remaining = sum(1 for l in plan_content_final.split("\n") if l.strip().startswith("- [ ]"))
                    if remaining == 0 and _session_manifest.files_created:
                        print(f"\n  {C.CLAW}{BLACK_CIRCLE} Running build verification...{C.RESET}")
                        project_root = CWD
                        if _session_manifest.files_created:
                            first_file = _resolve(_session_manifest.files_created[0])
                            check = first_file.parent
                            for _ in range(5):
                                if (check / "package.json").exists() or (check / "requirements.txt").exists():
                                    project_root = str(check)
                                    break
                                if check.parent == check:
                                    break
                                check = check.parent

                        success, summary = build_and_verify(project_root, messages, model)
                        print(f"\n  {C.CLAW}{C.BOLD}Build Verification Results{C.RESET}")
                        for line in summary.split("\n"):
                            print(f"    {line}")
                        if success:
                            print(f"  {C.SUCCESS}{BLACK_CIRCLE} All checks passed!{C.RESET}")
                        else:
                            print(f"  {C.WARNING}{BLACK_CIRCLE} Some checks failed — review above.{C.RESET}")
                    elif remaining > 0:
                        print(f"  {C.WARNING}{BLACK_CIRCLE} {remaining} steps remaining. Run /plan run-all again to continue.{C.RESET}")
                    print()

                else:
                    # show current plan — or auto-create from conversation if none exists
                    if plan_path.exists():
                        content = plan_path.read_text(encoding="utf-8", errors="replace")
                        print(f"\n  {C.CLAW}{C.BOLD}Current Plan{C.RESET} {C.SUBTLE}(PLAN.md){C.RESET}")
                        print(f"  {C.SUBTLE}{'=' * 45}{C.RESET}")
                        for line in content.split("\n"):
                            stripped = line.strip()
                            if stripped.startswith("- [x]"):
                                print(f"    {C.SUCCESS}{stripped}{C.RESET}")
                            elif stripped.startswith("- [ ]"):
                                print(f"    {C.TEXT}{stripped}{C.RESET}")
                            elif stripped.startswith("#"):
                                print(f"    {C.CLAW}{C.BOLD}{stripped}{C.RESET}")
                            else:
                                print(f"    {C.SUBTLE}{stripped}{C.RESET}")
                        print()
                    else:
                        # No plan exists — auto-create from conversation context if available
                        conv_context = _extract_conversation_context_for_plan(messages)
                        if conv_context:
                            print(f"  {C.CLAW}{BLACK_CIRCLE} No plan found — generating PLAN.md from conversation...{C.RESET}")
                            plan_prompt = (
                                f"Based on our conversation, create a detailed step-by-step plan.\n\n"
                                f"## Conversation Context\n{conv_context}\n\n"
                                f"Write the plan to PLAN.md using write_file. Use this EXACT format:\n"
                                f"# Plan: [descriptive title]\n\n"
                                f"## Overview\n[1-2 sentence summary]\n\n"
                                f"## Tech Stack\n[List technologies]\n\n"
                                f"## Steps\n\n"
                                f"- [ ] Step 1: [specific actionable step]\n"
                                f"- [ ] Step 2: ...\n\n"
                                f"Rules:\n"
                                f"- Each step must be specific, actionable, completable in one agent turn\n"
                                f"- Include ALL decisions from the conversation\n"
                                f"- Order by dependency\n"
                            )
                            messages.append({"role": "user", "content": plan_prompt})
                            turn_count += 1
                            try:
                                run_agent_turn(messages, model, use_tools=True)
                            except Exception as e:
                                print(f"  {C.ERROR}{BLACK_CIRCLE} Error: {e}{C.RESET}")
                            ACTIVE_PLAN_FILE = str(plan_path) if plan_path.exists() else None
                            if plan_path.exists():
                                print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} Plan created: PLAN.md{C.RESET}")
                            print()
                        else:
                            print(f"  {C.SUBTLE}No plan found. Use /plan create to make one.{C.RESET}")
                continue

            elif cmd == "/agents":
                print(f"  {C.SUBTLE}Enter one task per line. End with '---' or blank line x2:{C.RESET}")
                task_lines = []
                empty_count = 0
                while True:
                    try:
                        sys.stdout.write(f"  {C.SUBTLE}task>{C.RESET} ")
                        sys.stdout.flush()
                        line = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if line == "---":
                        break
                    if not line:
                        empty_count += 1
                        if empty_count >= 2:
                            break
                        continue
                    else:
                        empty_count = 0
                    task_lines.append(line)

                if not task_lines:
                    print(f"  {C.SUBTLE}No tasks entered.{C.RESET}")
                    continue

                # build task descriptions as (name, prompt) tuples
                tasks = []
                for line in task_lines:
                    # use first ~40 chars as display name
                    name = line[:50] + ("..." if len(line) > 50 else "")
                    tasks.append((name, line))

                try:
                    agent_results = run_subagents(tasks, model, system_prompt)
                    # add agent results to conversation context
                    summary_parts = []
                    for r in agent_results:
                        if r.status == "done" and r.output:
                            summary_parts.append(f"[Agent: {r.name}]\n{r.output}")
                        elif r.status == "error":
                            summary_parts.append(f"[Agent: {r.name}] Error: {r.error}")
                    if summary_parts:
                        combined = "\n\n".join(summary_parts)
                        messages.append({"role": "assistant", "content": combined})
                except Exception as e:
                    print(f"  {C.ERROR}{BLACK_CIRCLE} Agent error: {e}{C.RESET}")
                continue

            elif cmd == "/generate":
                if not cmd_arg:
                    print(f"  {C.ERROR}Usage: /generate <filename> then describe what to build{C.RESET}")
                    continue
                gen_file = cmd_arg.split()[0]
                gen_desc = cmd_arg[len(gen_file):].strip()
                if not gen_desc:
                    print(f"  {C.SUBTLE}Describe what to generate:{C.RESET}")
                    sys.stdout.write(f"  {C.CLAW}describe>{C.RESET} ")
                    sys.stdout.flush()
                    try:
                        gen_desc = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        continue
                    if not gen_desc:
                        continue
                print(f"\n  {C.CLAW}{C.BOLD}Chunked Generation{C.RESET}")
                print(f"  {C.SUBTLE}File: {gen_file}{C.RESET}")
                print(f"  {C.SUBTLE}Desc: {gen_desc}{C.RESET}\n")
                try:
                    success, content = _chunked_generate_file(gen_file, gen_desc, messages, model)
                    if success and content:
                        fp = _resolve(gen_file)
                        fp.parent.mkdir(parents=True, exist_ok=True)
                        fp.write_text(content, encoding="utf-8")
                        # Run verification gate
                        result = f"Wrote {len(content)} chars to {fp}"
                        result = _verify_file_write("write_file", {"file_path": gen_file}, result)
                        import re as _re
                        clean = _re.sub(r'\033\[[^m]*m', '', result)
                        for line in clean.split("\n"):
                            print(f"  {line}")
                        _session_manifest.record_tool_result("write_file", {"file_path": gen_file}, result)
                        print(f"\n  {C.SUCCESS}{BLACK_CIRCLE} Generated {gen_file} ({len(content)} chars, chunked){C.RESET}")
                    else:
                        print(f"  {C.ERROR}{BLACK_CIRCLE} Chunked generation failed{C.RESET}")
                except Exception as e:
                    print(f"  {C.ERROR}{BLACK_CIRCLE} Error: {e}{C.RESET}")
                continue

            elif cmd == "/status":
                print(f"  {C.SUBTLE}Model:       {C.TEXT}{model}{C.RESET}")
                print(f"  {C.SUBTLE}Vision:      {C.TEXT}{vision_model}{C.RESET}")
                print(f"  {C.SUBTLE}Turns:       {C.TEXT}{turn_count}{C.RESET}")
                print(f"  {C.SUBTLE}Messages:    {C.TEXT}{len(messages)}{C.RESET}")
                print(f"  {C.SUBTLE}Attachments: {C.TEXT}{len(pending_attachments)}{C.RESET}")
                print(f"  {C.SUBTLE}cwd:         {C.TEXT}{CWD}{C.RESET}")
                print(f"  {C.SUBTLE}timeout:     {C.TEXT}{OLLAMA_STREAM_TIMEOUT}s{C.RESET}")
                print(f"  {C.SUBTLE}Tokens:      {C.TEXT}{_token_tracker.summary()}{C.RESET}")
                print(f"  {C.SUBTLE}Undo stack:  {C.TEXT}{len(_undo_stack)} snapshots{C.RESET}")
                if _file_watcher:
                    print(f"  {C.SUBTLE}Watch:       {C.SUCCESS}active{C.RESET}")
                _gf = _guard_stats
                if any(_gf.values()):
                    print(f"  {C.SUBTLE}Guards:      {C.TEXT}{_gf['ramble_truncations']}T {_gf['dedup_fires']}D {_gf['html_strips']}H {_gf['read_guard_blocks']}R {_gf['auto_symbol_searches']}S {_gf['todo_resolves']}Todo{C.RESET}")
                continue

            # --- /undo: revert last file change ---
            elif cmd == "/undo":
                result = _perform_undo()
                print(f"  {C.CLAW}{BLACK_CIRCLE} {result}{C.RESET}")
                continue

            # --- /save: save session to disk ---
            elif cmd == "/save":
                sid = cmd_arg or None
                path = _save_session(messages, sid)
                print(f"  {C.SUCCESS}{BLACK_CIRCLE} Session saved: {path}{C.RESET}")
                continue

            # --- /resume: load previous session ---
            elif cmd == "/resume":
                if cmd_arg == "list":
                    sessions = _list_sessions()
                    if not sessions:
                        print(f"  {C.SUBTLE}No saved sessions.{C.RESET}")
                    else:
                        for sid, ts, count, cwd in sessions:
                            print(f"  {C.SUBTLE}{ts}{C.RESET} {C.TEXT}{sid}{C.RESET} {C.SUBTLE}({count} msgs, {cwd}){C.RESET}")
                else:
                    loaded, sid = _load_session(cmd_arg or None)
                    if loaded:
                        messages = loaded
                        turn_count = sum(1 for m in messages if m.get("role") == "user")
                        print(f"  {C.SUCCESS}{BLACK_CIRCLE} Resumed session: {sid} ({len(messages)} messages, turn {turn_count}){C.RESET}")
                    else:
                        print(f"  {C.ERROR}No session found.{C.RESET} Use /resume list to see available sessions.")
                continue

            # --- /export: export conversation ---
            elif cmd == "/export":
                fmt = "json" if cmd_arg == "json" else "md"
                path = _export_conversation(messages, fmt)
                print(f"  {C.SUCCESS}{BLACK_CIRCLE} Exported to: {path}{C.RESET}")
                continue

            # --- /git: git shortcuts ---
            elif cmd in ("/git", "/commit", "/diff", "/branch"):
                git_sub = cmd_arg if cmd == "/git" else cmd[1:]  # /commit → "commit"
                if git_sub == "status" or cmd == "/git" and not cmd_arg:
                    result = tool_bash({"command": "git status"})
                    print(f"  {C.DIM}{result}{C.RESET}")
                elif git_sub == "diff":
                    result = tool_bash({"command": "git diff --stat"})
                    print(f"  {C.DIM}{result}{C.RESET}")
                elif git_sub.startswith("commit"):
                    msg = git_sub.replace("commit", "").strip().strip('"').strip("'")
                    if not msg:
                        msg = cmd_arg.replace("commit", "").strip().strip('"').strip("'") if cmd_arg else ""
                    if not msg:
                        print(f"  {C.SUBTLE}Enter commit message:{C.RESET}")
                        sys.stdout.write(f"  {C.CLAW}msg>{C.RESET} ")
                        sys.stdout.flush()
                        try:
                            msg = input().strip()
                        except (EOFError, KeyboardInterrupt):
                            continue
                    if msg:
                        result = tool_bash({"command": f'git add -A && git commit -m "{msg}"'})
                        print(f"  {C.DIM}{result}{C.RESET}")
                elif git_sub == "branch":
                    result = tool_bash({"command": "git branch -a"})
                    print(f"  {C.DIM}{result}{C.RESET}")
                elif git_sub == "log":
                    result = tool_bash({"command": "git log --oneline -10"})
                    print(f"  {C.DIM}{result}{C.RESET}")
                elif git_sub == "push":
                    result = tool_bash({"command": "git push"})
                    print(f"  {C.DIM}{result}{C.RESET}")
                else:
                    result = tool_bash({"command": f"git {git_sub}"})
                    print(f"  {C.DIM}{result}{C.RESET}")
                continue

            # --- /test: run project tests ---
            elif cmd == "/test":
                success, summary = _run_tests(CWD, model, messages)
                print(f"  {C.SUCCESS if success else C.ERROR}{BLACK_CIRCLE} {summary}{C.RESET}")
                continue

            # --- /map: show codebase map ---
            elif cmd == "/map":
                # Invalidate cache and rescan
                _build_codebase_map._cache = None
                _build_codebase_map._cache_cwd = None
                cmap = _build_codebase_map(CWD, max_tokens=2000)
                if cmap:
                    print(f"\n{_render_markdown(cmap)}\n")
                else:
                    print(f"  {C.SUBTLE}No code files found in {CWD}{C.RESET}")
                continue

            # --- /watch: toggle file watcher ---
            elif cmd == "/watch":
                if _file_watcher:
                    _file_watcher.stop()
                    _file_watcher = None
                    print(f"  {C.WARNING}{BLACK_CIRCLE} Watch mode stopped{C.RESET}")
                else:
                    _file_watcher = FileWatcher(CWD)
                    _file_watcher.start()
                    print(f"  {C.SUCCESS}{BLACK_CIRCLE} Watch mode started — monitoring {CWD}{C.RESET}")
                continue

            # --- /tokens: show token usage ---
            elif cmd == "/tokens":
                print(f"  {C.CLAW}{BLACK_CIRCLE} {_token_tracker.summary()}{C.RESET}")
                _token_budget_bar(_token_tracker.total, num_ctx=8192 * _token_tracker.total_requests if _token_tracker.total_requests else 8192)
                # Guard telemetry (Fix E)
                _gf = _guard_stats
                if any(_gf.values()):
                    print(f"  {C.SUBTLE}Guards: {_gf['ramble_truncations']} truncations, {_gf['dedup_fires']} dedup, {_gf['html_strips']} html, {_gf['read_guard_blocks']} read-blocks, {_gf['auto_symbol_searches']} sym-search, {_gf['todo_resolves']} todo-fixes{C.RESET}")
                continue

            # --- /screenshot: visual QA with vision model ---
            elif cmd == "/screenshot":
                img_path = cmd_arg
                if not img_path:
                    # Try to find most recent screenshot in CWD
                    pngs = sorted(Path(CWD).glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
                    jpgs = sorted(Path(CWD).glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
                    all_imgs = sorted(pngs + jpgs, key=lambda f: f.stat().st_mtime, reverse=True)
                    if all_imgs:
                        img_path = str(all_imgs[0])
                    else:
                        print(f"  {C.ERROR}No screenshots found. Usage: /screenshot path/to/image.png{C.RESET}")
                        continue
                print(f"  {C.TOOL}{BLACK_CIRCLE} Analyzing {Path(img_path).name} with vision model ({vision_model})...{C.RESET}")
                spinner = TimedSpinner("Analyzing screenshot", C.TOOL)
                spinner.start()
                feedback = _screenshot_qa(img_path, vision_model)
                spinner.stop()
                print(f"\n{_render_markdown(feedback)}\n")
                # Add to conversation so model knows the feedback
                messages.append({"role": "assistant", "content": f"Screenshot QA:\n{feedback}"})
                continue

            else:
                print(f"  {C.ERROR}Unknown command: {cmd}. Type /help{C.RESET}")
                continue

        # Show compact sent-message indicator (non-slash commands only)
        _display_sent_message(user_input)

        # --- parse @file references ---
        cleaned_text, at_paths = parse_at_references(user_input)
        for p in at_paths:
            att = Attachment(p)
            pending_attachments.append(att)
            print(f"  {att.summary()}")

        if not cleaned_text and not pending_attachments:
            continue

        if not cleaned_text:
            cleaned_text = "Describe/analyze the attached file(s)."

        # --- build message with attachments ---
        turn_count += 1

        # inject warm memory context based on user message keywords
        warm_context = _auto_search_warm_memories(cleaned_text)
        if warm_context:
            messages.append({"role": "user", "content": f"[SYSTEM: Related context from memory]\n{warm_context}"})
            messages.append({"role": "assistant", "content": "Noted, I'll keep this context in mind."})

        msg, override_model = build_user_message(cleaned_text, pending_attachments, vision_model)
        messages.append(msg)

        turn_model = override_model or model
        has_images = override_model is not None

        if has_images:
            print(f"  {C.TOOL}{BLACK_CIRCLE} Using vision model: {turn_model}{C.RESET}")

        # clear pending attachments after sending
        pending_attachments = []

        try:
            # disable tools for pure vision queries (vision models often don't support tools)
            _turn_start = time.time()
            _pre_p = _token_tracker.prompt_tokens
            _pre_c = _token_tracker.completion_tokens
            run_agent_turn(messages, turn_model, use_tools=not has_images)
            _last_turn_tokens = (_token_tracker.prompt_tokens - _pre_p, _token_tracker.completion_tokens - _pre_c)
            # Bell if the turn took more than 10 seconds (user might be tabbed away)
            if time.time() - _turn_start > 10:
                _bell()
        except ConnectionError as e:
            print(f"\n  {C.ERROR}{BLACK_CIRCLE} Connection error: {e}{C.RESET}")
        except KeyboardInterrupt:
            print(f"\n  {C.WARNING}{BLACK_CIRCLE} Interrupted{C.RESET}")
        except Exception as e:
            print(f"\n  {C.ERROR}{BLACK_CIRCLE} Error: {e}{C.RESET}")

        print()

if __name__ == "__main__":
    main()
