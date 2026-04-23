# Rattlesnake End-to-End Tests

Runs the real `claw_cli.py` against progressive coding tasks and grades each one with a strict `verify.py`.

## Defaults

**Local Ollama, not cloud.** The runner uses whatever `CLAW_PROVIDER` and `CLAW_MODEL` point at. Out of the box:

- `CLAW_PROVIDER=ollama` (local)
- `CLAW_MODEL=qwen2.5-coder:14b` (main model — the "gold")
- `CLAW_SMALL_MODEL=llama3.2:1b` (cheap read-only router — the "lead")

Pull the models first:

```bash
ollama pull qwen2.5-coder:14b
ollama pull llama3.2:1b
```

Make sure `ollama serve` is running (default `http://localhost:11434`).

Cloud runs are opt-in — set `CLAW_PROVIDER=openrouter` (or `openai`, `anthropic`, `dashscope`) and the matching `*_API_KEY` env var.

## Usage

```bash
# list tasks without invoking the model
python tests/e2e/run_e2e.py --dry-run

# run all tasks against local Ollama defaults
python tests/e2e/run_e2e.py

# run one task
python tests/e2e/run_e2e.py --task hello_world

# try a different local model
python tests/e2e/run_e2e.py --model qwen2.5-coder:7b

# test the tiny router model directly
python tests/e2e/run_e2e.py --task read_count --model llama3.2:1b

# keep temp dirs so you can inspect what the agent did
python tests/e2e/run_e2e.py --keep-temp -v
```

Reports land in `tests/e2e/results/e2e-<timestamp>.json`.

## Task layout

```
tests/e2e/tasks/<name>/
  prompt.txt          instruction handed verbatim to claw_cli
  setup/ (optional)   files copied into the temp workdir before the run
  verify.py           post-run grader; emits ONE JSON line: {"passed": bool, "reason": str}
```

`verify.py` runs with the post-run temp dir as its cwd. Parse the last JSON line of stdout; if that fails the runner falls back to the verify script's exit code.

## Current tasks (ranked easy → hard)

| Task          | Exercises                                               |
| ------------- | ------------------------------------------------------- |
| hello_world   | single write_file                                       |
| read_count    | read_file → write_file (small-model-eligible read path) |
| fizzbuzz      | generation correctness                                  |
| fix_syntax    | read_file → edit_file                                   |
| add_flag      | read_file → edit/write, argparse, preserves default     |

Add a new task by dropping a folder in `tasks/` that matches the layout. The runner auto-discovers it.
