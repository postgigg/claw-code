"""
LLM provider adapters — extracted from claw_cli.py.

Each provider exposes:
  - chat(messages, model, tools=None, stream=True, num_ctx=8192) -> generator
  - get_context_size(model) -> int

Usage pattern:
    from rattlesnake import providers
    providers.set_token_tracker(my_tracker)
    p = providers.OpenRouterProvider()
    for chunk in p.chat(messages, model): ...

The token tracker is injected by claw_cli at startup so streamed token counts
flow into the single shared tracker.
"""

import json
import os
import time
import urllib.error
import urllib.request

# ---- configuration (re-read from env at module load, same as claw_cli.py) ----
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_STREAM_TIMEOUT = int(os.environ.get("CLAW_TIMEOUT", "1800"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# Low temperature for agent work — Ollama default (0.8) makes tool-call
# emission stochastic, so identical tasks can pass once and fail next run.
# Override via CLAW_TEMP env var.
try:
    CLAW_TEMPERATURE = float(os.environ.get("CLAW_TEMP", "0.2"))
except ValueError:
    CLAW_TEMPERATURE = 0.2

_CLOUD_CONTEXT_SIZES = {
    "claude": 200000, "gpt-4o": 128000, "gpt-4": 128000, "gpt-3.5": 16385,
    "llama": 131072, "qwen": 32768, "mistral": 32768, "deepseek": 65536,
    "gemma": 8192, "phi": 16384, "command-r": 128000,
}

# ---- token tracker injection ----
# claw_cli.py calls set_token_tracker() once at startup; the provider methods
# look up _token_tracker at call time, so we just need it assigned before the
# first chat() invocation.
_token_tracker = None


def set_token_tracker(tracker):
    """Inject the shared TokenTracker instance from claw_cli."""
    global _token_tracker
    _token_tracker = tracker


class _NullTracker:
    """Fallback no-op tracker used if set_token_tracker() wasn't called."""
    def add(self, *args, **kwargs):
        pass


def _tt():
    """Return the active tracker, or a null tracker if none was injected."""
    return _token_tracker if _token_tracker is not None else _NullTracker()


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
            "options": {"num_ctx": num_ctx, "temperature": CLAW_TEMPERATURE},
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
        _tt().add(prompt=_prompt_est)

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
                        _tt().add(completion=actual)
                    yield chunk
                except json.JSONDecodeError:
                    continue
        else:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            _tt().add(completion=data.get("eval_count", len(data.get("message", {}).get("content", "").split())))
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
            modelfile = info.get("modelfile", "") or info.get("parameters", "")
            for line in str(modelfile).split("\n"):
                if "num_ctx" in line:
                    parts = line.strip().split()
                    for p in parts:
                        if p.isdigit():
                            return int(p)
            model_info = info.get("model_info", {})
            for key, val in model_info.items():
                if "context" in key.lower() and isinstance(val, (int, float)):
                    return int(val)
        except Exception:
            pass
        return 8192


class OpenRouterProvider(LLMProvider):
    """OpenRouter API (OpenAI-compatible with SSE streaming)."""
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set. Export it or use --api-key.")

        payload = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
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
        _tt().add(prompt=_prompt_est)

        resp = None
        last_err = None
        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=OLLAMA_STREAM_TIMEOUT)
                break
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    err_body = ""
                last_err = f"HTTP {e.code}: {err_body}" if err_body else e
                if e.code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
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
        accumulated_tool_calls = {}
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                final = {"done": True, "message": {"content": "", "role": "assistant"}}
                if accumulated_tool_calls:
                    final["message"]["tool_calls"] = [
                        {"id": tc.get("id", f"call_{i}"), "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for i, tc in enumerate(accumulated_tool_calls.values())
                    ]
                _tt().add(completion=_comp_tokens)
                yield final
                return
            try:
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "") or ""
                if content:
                    _comp_tokens += len(content.split())

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

                ollama_chunk = {
                    "message": {"role": "assistant", "content": content},
                    "done": False,
                }
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
        _tt().add(completion=usage.get("completion_tokens", 0))
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
        _tt().add(prompt=_prompt_est)

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
    _last_request_time = 0.0
    _MIN_REQUEST_GAP = float(os.environ.get("DASHSCOPE_COOLDOWN", "1.5"))

    def chat(self, messages, model, tools=None, stream=True, num_ctx=8192):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY not set. Export it or use --api-key.")

        now = time.time()
        elapsed = now - DashScopeProvider._last_request_time
        if elapsed < self._MIN_REQUEST_GAP:
            time.sleep(self._MIN_REQUEST_GAP - elapsed)
        DashScopeProvider._last_request_time = time.time()

        clean_model = model
        if "/" in clean_model:
            clean_model = clean_model.split("/", 1)[1]
        if ":" in clean_model:
            clean_model = clean_model.split(":")[0]
        _DASHSCOPE_MODEL_MAP = {
            "qwen3-235b-a22b": "qwen-plus-latest",
            "qwen3-32b": "qwen-turbo-latest",
            "qwen3-30b-a3b": "qwen-turbo-latest",
            "qwen3.6-plus": "qwen3.5-plus",
        }
        clean_model = _DASHSCOPE_MODEL_MAP.get(clean_model, clean_model)

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
        _tt().add(prompt=_prompt_est)

        resp = None
        last_err = None
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
                if (e.code == 429 or e.code >= 500) and attempt < 4:
                    wait = min(3 * (2 ** attempt), 30)
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

        system_text = ""
        api_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_text += m.get("content", "") + "\n"
            elif m.get("role") == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m.get("tool_use_id", "tool_0"), "content": m.get("content", "")}]
                })
            else:
                api_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        api_messages = self._fix_message_order(api_messages)

        payload = {"model": model, "messages": api_messages, "max_tokens": min(num_ctx, 8192), "stream": stream}
        if system_text.strip():
            payload["system"] = system_text.strip()

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
        _tt().add(prompt=_prompt_est)

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
                    _tt().add(completion=_comp_tokens)
                    yield final
                    return
                elif event_type == "message_delta":
                    usage = event.get("usage", {})
                    if usage.get("output_tokens"):
                        _tt().add(completion=usage["output_tokens"])
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
        _tt().add(completion=usage.get("output_tokens", 0))
        return result

    def get_context_size(self, model):
        return 200000  # All Claude models support 200K
