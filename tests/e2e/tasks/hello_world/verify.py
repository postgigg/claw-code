"""Verify hello_world task — run hello.py, compare stdout to expected."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    cwd = Path.cwd()
    hello = cwd / "hello.py"
    if not hello.exists():
        print(json.dumps({"passed": False, "reason": "hello.py was not created"}))
        return 1

    result = subprocess.run(
        [sys.executable, str(hello)],
        capture_output=True, text=True, timeout=10,
    )
    expected = "Hello, Rattlesnake!"
    actual = result.stdout.strip()
    if result.returncode != 0:
        print(json.dumps({
            "passed": False,
            "reason": f"hello.py exited with code {result.returncode}. stderr: {result.stderr[:200]}",
        }))
        return 1
    if actual != expected:
        print(json.dumps({
            "passed": False,
            "reason": f"stdout mismatch. expected={expected!r} actual={actual!r}",
        }))
        return 1

    print(json.dumps({"passed": True, "reason": "hello.py runs and prints expected line"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
