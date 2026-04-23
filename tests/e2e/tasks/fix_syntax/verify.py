"""Verify fix_syntax — broken.py must run and print Hello, world!"""
import json
import subprocess
import sys
from pathlib import Path


def main():
    target = Path.cwd() / "broken.py"
    if not target.exists():
        print(json.dumps({"passed": False, "reason": "broken.py missing from temp dir"}))
        return 1

    result = subprocess.run(
        [sys.executable, str(target)],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(json.dumps({
            "passed": False,
            "reason": f"broken.py still fails (exit {result.returncode}). stderr: {result.stderr[:300]}",
        }))
        return 1
    out = result.stdout.strip()
    if out != "Hello, world!":
        print(json.dumps({
            "passed": False,
            "reason": f"wrong greeting. expected 'Hello, world!' got {out!r}",
        }))
        return 1

    print(json.dumps({"passed": True, "reason": "broken.py runs and greets correctly"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
