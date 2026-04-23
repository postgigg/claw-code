#!/usr/bin/env python3
"""
Rattlesnake end-to-end test runner.

Each task lives in tests/e2e/tasks/<name>/ with:
  - prompt.txt            the instruction handed to claw_cli
  - setup/ (optional)     files copied into the temp workdir before running
  - verify.py             runs after, emits one JSON line: {"passed": bool, "reason": str}

The runner:
  1. For each task, creates a fresh temp dir.
  2. Copies setup/ into it.
  3. Invokes `python claw_cli.py "<prompt>"` with cwd = temp dir (one-shot mode).
  4. Runs verify.py in the same dir, parses its JSON, records pass/fail.
  5. Aggregates a JSON report at tests/e2e/results/<timestamp>.json.

Usage:
  python tests/e2e/run_e2e.py                       # run all tasks (local Ollama default)
  python tests/e2e/run_e2e.py --task hello_world    # run one task
  python tests/e2e/run_e2e.py --dry-run             # list tasks, skip model calls
  python tests/e2e/run_e2e.py --model qwen2.5-coder:7b
  python tests/e2e/run_e2e.py --keep-temp           # leave temp dirs for inspection

Defaults: local Ollama, CLAW_MODEL=qwen2.5-coder:14b, CLAW_SMALL_MODEL=llama3.2:1b.
Cloud providers are opt-in via CLAW_PROVIDER=openrouter|openai|anthropic|dashscope
plus the matching *_API_KEY env var.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Windows consoles default to cp1252; force UTF-8 so unicode glyphs below
# don't crash the runner.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


E2E_DIR = Path(__file__).resolve().parent
TASKS_DIR = E2E_DIR / "tasks"
RESULTS_DIR = E2E_DIR / "results"
REPO_ROOT = E2E_DIR.parent.parent
CLAW_CLI = REPO_ROOT / "claw_cli.py"


@dataclass
class TaskResult:
    name: str
    status: str                 # PASS | FAIL | ERROR | SKIP
    reason: str = ""
    duration_s: float = 0.0
    cli_exit_code: Optional[int] = None
    cli_stdout_tail: str = ""
    cli_stderr_tail: str = ""
    verify_stdout: str = ""
    temp_dir: str = ""


def discover_tasks() -> list[Path]:
    if not TASKS_DIR.is_dir():
        return []
    return sorted(p for p in TASKS_DIR.iterdir() if p.is_dir() and (p / "prompt.txt").exists())


def copy_setup(setup_dir: Path, target: Path) -> None:
    if not setup_dir.exists():
        return
    for item in setup_dir.iterdir():
        dest = target / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest)


def run_task(task: Path, *, model: Optional[str], timeout: int,
             dry_run: bool, keep_temp: bool, verbose: bool) -> TaskResult:
    name = task.name
    prompt = (task / "prompt.txt").read_text(encoding="utf-8").strip()
    verify_py = task / "verify.py"

    if dry_run:
        return TaskResult(name=name, status="SKIP", reason="dry-run")

    # Fresh temp dir for isolation. Set delete=False and remove manually so
    # --keep-temp works and Windows file locks don't trip cleanup.
    tmp = Path(tempfile.mkdtemp(prefix=f"rattle-e2e-{name}-"))
    try:
        copy_setup(task / "setup", tmp)

        cmd = [sys.executable, str(CLAW_CLI), prompt]
        if model:
            cmd += ["--model", model]

        env = dict(os.environ)
        # Belt-and-suspenders: many guardrails consult these; force automation-
        # friendly defaults so the subprocess cannot hang on a prompt.
        env.setdefault("CLAW_PERMISSION", "auto")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, capture_output=True, text=True,
                timeout=timeout, env=env, encoding="utf-8", errors="replace",
            )
            duration = time.time() - start
            exit_code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as e:
            duration = float(timeout)
            exit_code = -1
            stdout = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
            stderr = f"[runner] TIMEOUT after {timeout}s\n" + (e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))

        # Run the task's verify.py against the post-run state of tmp.
        verify_stdout = ""
        passed = False
        reason = "verify.py missing"
        if verify_py.exists():
            vproc = subprocess.run(
                [sys.executable, str(verify_py)],
                cwd=tmp, capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            )
            verify_stdout = (vproc.stdout or "").strip()
            try:
                data = json.loads(verify_stdout.splitlines()[-1] if verify_stdout else "{}")
                passed = bool(data.get("passed"))
                reason = str(data.get("reason", ""))
            except (json.JSONDecodeError, IndexError):
                passed = vproc.returncode == 0
                reason = (verify_stdout or vproc.stderr)[:500]

        status = "PASS" if passed else "FAIL"
        if exit_code != 0 and not passed:
            status = "ERROR"

        result = TaskResult(
            name=name, status=status, reason=reason,
            duration_s=round(duration, 2), cli_exit_code=exit_code,
            cli_stdout_tail=stdout[-2000:], cli_stderr_tail=stderr[-1000:],
            verify_stdout=verify_stdout, temp_dir=str(tmp) if keep_temp else "",
        )
        if verbose:
            print(f"[{name}] {status} in {result.duration_s}s — {reason}")
            if status != "PASS":
                print(f"    stdout tail:\n{stdout[-800:]}")
                print(f"    stderr tail:\n{stderr[-400:]}")
        return result
    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)


def print_summary(results: list[TaskResult]) -> None:
    name_w = max((len(r.name) for r in results), default=4)
    print()
    print(f"{'TASK':<{name_w}}  {'STATUS':<6}  {'TIME':>6}  REASON")
    print("-" * (name_w + 30))
    for r in results:
        print(f"{r.name:<{name_w}}  {r.status:<6}  {r.duration_s:>5.1f}s  {r.reason[:80]}")
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    skipped = sum(1 for r in results if r.status == "SKIP")
    total = len(results)
    print()
    print(f"Total: {total}  passed: {passed}  failed: {failed}  skipped: {skipped}")


def save_report(results: list[TaskResult], args) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"e2e-{stamp}.json"
    payload = {
        "timestamp": stamp,
        "model": args.model or os.environ.get("CLAW_MODEL", "default"),
        "provider": os.environ.get("CLAW_PROVIDER", "openrouter"),
        "dry_run": args.dry_run,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rattlesnake E2E test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", help="Run only the named task")
    parser.add_argument("--model", help="Override CLAW_MODEL for this run")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-task timeout in seconds (default: 300)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List tasks without invoking Rattlesnake")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Leave temp dirs on disk for debugging")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print stdout/stderr tails inline")
    args = parser.parse_args()

    tasks = discover_tasks()
    if args.task:
        tasks = [t for t in tasks if t.name == args.task]
        if not tasks:
            print(f"No task named {args.task!r} in {TASKS_DIR}", file=sys.stderr)
            return 2

    if not tasks:
        print(f"No tasks found under {TASKS_DIR}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Would run {len(tasks)} task(s):")
        for t in tasks:
            print(f"  - {t.name}")
        return 0

    if not CLAW_CLI.exists():
        print(f"claw_cli.py not found at {CLAW_CLI}", file=sys.stderr)
        return 2

    print(f"Rattlesnake E2E — {len(tasks)} task(s), model={args.model or 'default'}, "
          f"provider={os.environ.get('CLAW_PROVIDER', 'ollama')}")
    print()

    results: list[TaskResult] = []
    for task in tasks:
        print(f"▶ {task.name} …", flush=True)
        results.append(run_task(
            task, model=args.model, timeout=args.timeout,
            dry_run=args.dry_run, keep_temp=args.keep_temp, verbose=args.verbose,
        ))

    print_summary(results)
    report = save_report(results, args)
    print(f"\nReport: {report}")
    return 0 if all(r.status in ("PASS", "SKIP") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
