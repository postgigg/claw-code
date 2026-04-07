# Plan: v0.3 Features — Context Compression, URL Fetch, Auto-Lint, Plugins, Screenshot QA

## Current State (verified by full code audit)

- **File**: `claw_cli.py` — 7,220 lines
- **Context window**: 8192 tokens (hardcoded `num_ctx` default)
- **System prompt**: ~2000 tokens → leaves ~6000 for conversation
- **Message growth**: UNBOUNDED — no trimming, no compression, hits limit at ~4-5 complex turns
- **Tools**: 12 (bash, read_file, write_file, edit_file, glob_search, grep_search, ask_user, memory_save, memory_search, db_schema, env_manage, web_search)
- **Verification**: Syntax check + incomplete code scan. NO linter integration.
- **Web**: Search only (DuckDuckGo HTML). Cannot fetch/read actual URLs.
- **Plugins**: None. No extension mechanism.
- **Vision**: Can analyze user-provided images. Cannot capture screenshots itself.

---

## Feature 1: Context Compression (CRITICAL)

### Problem
Messages accumulate unbounded in `run_agent_turn()`. After 4-5 turns with tool calls, the messages list exceeds 6000 tokens and the model starts losing context or erroring.

### Solution
Add `_compress_messages(messages, max_tokens)` that runs BEFORE every `ollama_chat()` call. It:
1. Always keeps: system prompt (index 0) + last 2 user/assistant turns
2. Compresses middle messages: tool results → one-line summaries, old assistant text → key decisions only
3. Drops: old tool results entirely if >10 turns old

### Insertion Points
- **New function**: `_compress_messages(messages, max_tokens=5500)` — insert after `_token_budget_bar` (line ~372), in the UI utilities section
- **Hook into `run_agent_turn()`** (line ~5798): Before `ollama_chat()` call, run `compressed = _compress_messages(messages, 5500)` and pass `compressed` instead of `messages` to ollama_chat. Keep original `messages` intact for session save.
- **Hook into `_run_subagent_worker()`** (line ~740): Same compression before `_ollama_chat_sync()` call

### Token Budget
- System prompt: ~2000 tokens (untouched)
- Compressed conversation: ~3500 tokens max
- Generation room: ~2700 tokens
- Total: 8192

### Logic
```
def _compress_messages(messages, max_tokens=5500):
    if not messages: return messages

    # Always keep system prompt (first message)
    system = [messages[0]] if messages[0]["role"] == "system" else []
    rest = messages[1:] if system else messages[:]

    # Always keep last 4 messages (2 turns) intact
    keep_tail = rest[-4:] if len(rest) >= 4 else rest[:]
    middle = rest[:-4] if len(rest) > 4 else []

    # Compress middle: tool results → one-line, assistant → first 100 chars
    compressed_middle = []
    for msg in middle:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool":
            # One-line summary: first 80 chars
            summary = content[:80].replace("\n", " ").strip()
            compressed_middle.append({"role": "tool", "content": f"[prior result: {summary}...]"})
        elif role == "assistant":
            if len(content) > 150:
                compressed_middle.append({"role": "assistant", "content": content[:150] + "..."})
            else:
                compressed_middle.append(msg)
        else:
            compressed_middle.append(msg)

    # Estimate total tokens
    result = system + compressed_middle + keep_tail
    est_tokens = sum(len(m.get("content", "").split()) for m in result)

    # If still too big, drop oldest compressed messages
    while est_tokens > max_tokens and len(compressed_middle) > 0:
        compressed_middle.pop(0)
        result = system + compressed_middle + keep_tail
        est_tokens = sum(len(m.get("content", "").split()) for m in result)

    return result
```

### Does NOT Touch
- `messages` list in main REPL (stays intact for /save, /export)
- System prompt generation
- Tool execution
- Drip system (uses its own DripContext compression)

---

## Feature 2: URL Fetch Tool

### Problem
`web_search` returns titles + snippets but can't read actual page content. Model needs to read documentation, READMEs, Stack Overflow answers.

### Solution
Add `tool_web_fetch(args)` — fetches a URL and returns clean text content (strips HTML tags).

### Insertion Points
- **New function**: `tool_web_fetch(args)` — insert right after `tool_web_search()` (line ~2243)
- **Add to TOOL_DEFS** (line ~1536): New tool definition `"web_fetch"`
- **Add to TOOL_HANDLERS** (line ~2259): `"web_fetch": tool_web_fetch`

### Logic
```
def tool_web_fetch(args):
    url = args.get("url", "").strip()
    if not url: return error
    if not url.startswith("http"): return error

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Rattlesnake CLI)"})
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode("utf-8", errors="replace")

    # Strip HTML tags, scripts, styles
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Cap at 3000 chars to stay within context
    if len(text) > 3000:
        text = text[:3000] + "\n... [truncated]"
    return text
```

### Does NOT Touch
- Existing web_search (stays as-is, separate purpose)
- Any existing tool or flow

---

## Feature 3: Auto-Lint After Writes

### Problem
`_verify_file_write()` checks syntax + incomplete code but doesn't run the project's actual linter. Model writes code that passes syntax check but fails eslint/ruff.

### Solution
Add `_auto_lint(filepath)` that detects and runs the right linter, called from `_verify_file_write()`.

### Insertion Points
- **New function**: `_auto_lint(filepath)` — insert after `_scan_for_incomplete_code()` (line ~2809)
- **Hook into `_verify_file_write()`** (line ~3145): After syntax check, call `_auto_lint(fp_str)` and append results

### Logic
```
def _auto_lint(filepath):
    """Run the project's linter on a file. Returns (ok, output) or (None, None) if no linter."""
    fp = Path(filepath)
    ext = fp.suffix.lower()

    # Python: ruff (fast) or flake8 or pylint
    if ext == ".py":
        for cmd in ["ruff check {fp}", "flake8 {fp}", "python -m py_compile {fp}"]:
            if shutil.which(cmd.split()[0]):
                r = subprocess.run(cmd.format(fp=fp), shell=True, capture_output=True, text=True, timeout=15, cwd=CWD)
                return r.returncode == 0, (r.stdout + r.stderr).strip()

    # JS/TS: eslint or biome
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        # Check if eslint is available in project
        eslint = Path(CWD) / "node_modules" / ".bin" / "eslint"
        if eslint.exists() or shutil.which("eslint"):
            cmd = f"npx eslint --no-error-on-unmatched-pattern \"{fp}\""
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15, cwd=CWD)
            return r.returncode == 0, (r.stdout + r.stderr).strip()

    # CSS: stylelint
    if ext == ".css":
        if (Path(CWD) / "node_modules" / ".bin" / "stylelint").exists():
            cmd = f"npx stylelint \"{fp}\""
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15, cwd=CWD)
            return r.returncode == 0, (r.stdout + r.stderr).strip()

    return None, None  # no linter available
```

### Does NOT Touch
- Existing syntax validation (stays as-is, lint is additional)
- Tool execution flow
- Any existing post-processing

---

## Feature 4: Plugin System

### Problem
No way for users to extend Rattlesnake with custom tools without editing claw_cli.py.

### Solution
Load `.py` files from `~/.claw/plugins/` at startup. Each plugin exports a `TOOLS` list of `{"name", "description", "parameters", "handler"}` dicts that get merged into TOOL_DEFS and TOOL_HANDLERS.

### Insertion Points
- **New function**: `_load_plugins()` — insert after TOOL_HANDLERS definition (line ~2259)
- **Call at startup**: In `main()` (line ~6382), after TOOL_HANDLERS is set, call `_load_plugins()`
- **No TOOL_DEFS/TOOL_HANDLERS modification** — instead, plugins append to them at load time

### Plugin Format
```python
# ~/.claw/plugins/my_tool.py
TOOLS = [
    {
        "name": "my_custom_tool",
        "description": "Does something custom",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input"}
            },
            "required": ["input"]
        },
        "handler": lambda args: f"Result: {args.get('input', '')}"
    }
]
```

### Logic
```
PLUGINS_DIR = Path.home() / ".claw" / "plugins"

def _load_plugins():
    if not PLUGINS_DIR.exists():
        return 0
    count = 0
    for pf in PLUGINS_DIR.glob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location(pf.stem, str(pf))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for tool in getattr(mod, "TOOLS", []):
                name = tool["name"]
                TOOL_DEFS.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                })
                TOOL_HANDLERS[name] = tool["handler"]
                count += 1
        except Exception as e:
            print(f"  {C.WARNING}{BLACK_CIRCLE} Plugin {pf.name} failed: {e}{C.RESET}")
    return count
```

### Does NOT Touch
- Existing tools (plugins append, don't replace)
- TOOL_DEFS/TOOL_HANDLERS structure (same format)
- Any existing code paths

---

## Feature 5: Screenshot QA (Vision-Powered)

### Problem
After building a web project, there's no way to visually verify the output. The model can analyze images, but can't take screenshots.

### Solution
Add `/screenshot` command that:
1. Opens the HTML file in a headless browser (or asks user for a screenshot path)
2. Sends the image to the vision model
3. Gets feedback on visual quality

### Insertion Points
- **New slash command**: `/screenshot` in REPL (line ~7011)
- **New function**: `_screenshot_qa(filepath, messages, vision_model)` — insert in the UI utilities section

### Logic
```
def _screenshot_qa(image_path, messages, vision_model):
    """Send a screenshot to the vision model for QA feedback."""
    if not Path(image_path).exists():
        return "Screenshot file not found."

    img_b64 = read_image_b64(image_path)
    qa_messages = [
        {"role": "system", "content": "You are a UI/UX reviewer. Analyze this screenshot of a web page. Check: layout, colors, readability, responsiveness, missing elements, broken styling. Be specific about issues."},
        {"role": "user", "content": "Review this screenshot for visual quality issues.", "images": [img_b64]},
    ]

    result = ""
    for chunk in ollama_chat(qa_messages, vision_model, tools=None, stream=True):
        content = chunk.get("message", {}).get("content", "")
        if content:
            result += content
    return result
```

### /screenshot command
```
elif cmd == "/screenshot":
    if not cmd_arg:
        # Try to find latest screenshot or HTML file
        screenshots = sorted(Path(CWD).glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
        if screenshots:
            cmd_arg = str(screenshots[0])
        else:
            print("Provide a screenshot path: /screenshot path/to/image.png")
            continue

    print(f"  {C.TOOL}{BLACK_CIRCLE} Analyzing screenshot with vision model...{C.RESET}")
    feedback = _screenshot_qa(cmd_arg, messages, vision_model)
    print(_render_markdown(feedback))
    messages.append({"role": "assistant", "content": f"Screenshot QA feedback:\n{feedback}"})
```

### Does NOT Touch
- Existing vision model flow (this is a separate path)
- Existing tool system
- Any existing slash commands

---

## Feature 6: Dependency Auto-Install

### Problem
Model writes `import pandas` but pandas isn't installed. Build fails, model has to be told to install it.

### Solution
Add `_check_and_install_imports(filepath)` that scans a written file for imports, checks if they're installed, and auto-installs missing ones. Called from `_verify_file_write()`.

### Insertion Points
- **New function**: `_check_and_install_imports(filepath)` — insert after `_auto_lint()`
- **Hook into `_verify_file_write()`**: After lint check, call for .py files

### Logic
```
def _check_and_install_imports(filepath):
    """For Python files, detect imports and install missing packages."""
    fp = Path(filepath)
    if fp.suffix != ".py":
        return []

    content = fp.read_text(encoding="utf-8", errors="replace")
    imports = set()
    for m in re.finditer(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE):
        imports.add(m.group(1))

    # Filter stdlib modules
    stdlib = {"os", "sys", "json", "re", "time", "datetime", "pathlib", "collections",
              "itertools", "functools", "typing", "abc", "math", "random", "hashlib",
              "base64", "io", "copy", "string", "textwrap", "logging", "unittest",
              "subprocess", "threading", "multiprocessing", "socket", "http", "urllib",
              "csv", "xml", "html", "email", "argparse", "configparser", "sqlite3",
              "shutil", "glob", "tempfile", "pickle", "struct", "dataclasses", "enum",
              "contextlib", "inspect", "traceback", "warnings", "signal", "platform"}

    third_party = imports - stdlib
    installed = []

    for pkg in third_party:
        # Map common import names to pip package names
        pip_name = {"cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn",
                    "yaml": "PyYAML", "bs4": "beautifulsoup4", "dotenv": "python-dotenv",
                    "jwt": "PyJWT", "Crypto": "pycryptodome"}.get(pkg, pkg)
        try:
            __import__(pkg)
        except ImportError:
            # Install it
            r = subprocess.run(
                f"pip install {pip_name}", shell=True, capture_output=True, text=True, timeout=60
            )
            if r.returncode == 0:
                installed.append(pip_name)

    return installed
```

### Does NOT Touch
- Existing verification flow (this is additive)
- Node packages (npm install is handled by build_and_verify already)
- Any existing imports or tool behavior

---

## Implementation Order

1. **Context Compression** — highest impact, fixes the #1 usability problem (model loses context)
2. **URL Fetch** — model needs to read docs, currently blind
3. **Auto-Lint** — catches errors the syntax checker misses
4. **Dependency Auto-Install** — reduces back-and-forth on missing imports
5. **Plugin System** — extensibility without editing source
6. **Screenshot QA** — visual verification for web projects

## Risk Assessment

| Feature | Risk | Mitigation |
|---------|------|------------|
| Context Compression | Could lose important context | Always keep last 2 turns + system prompt intact |
| URL Fetch | Could fetch huge pages, slow | Cap at 3000 chars, 15s timeout |
| Auto-Lint | Linter not installed = no-op | Graceful fallback, lint is optional check |
| Dependency Install | Could install wrong package | Only for Python, uses common name mapping |
| Plugin System | Malicious plugins | User explicitly puts files in ~/.claw/plugins |
| Screenshot QA | Vision model slow | On-demand only via /screenshot |

## Files Modified
- `claw_cli.py` only — all features are self-contained additions

## Estimated Lines Added
- Context Compression: ~40 lines
- URL Fetch: ~35 lines (function + TOOL_DEFS entry + TOOL_HANDLERS entry)
- Auto-Lint: ~30 lines
- Dependency Auto-Install: ~35 lines
- Plugin System: ~35 lines
- Screenshot QA: ~30 lines + slash command ~15 lines
- **Total: ~220 lines**
