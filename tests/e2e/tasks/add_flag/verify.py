"""Verify add_flag — cli.py must support --name, preserve default."""
import json
import subprocess
import sys
from pathlib import Path


def run(args):
    return subprocess.run(
        [sys.executable, "cli.py", *args],
        capture_output=True, text=True, timeout=10,
    )


def main():
    target = Path.cwd() / "cli.py"
    if not target.exists():
        print(json.dumps({"passed": False, "reason": "cli.py missing"}))
        return 1

    # Default behavior
    r1 = run([])
    if r1.returncode != 0 or r1.stdout.strip() != "Hello, World!":
        print(json.dumps({
            "passed": False,
            "reason": f"default run regressed. exit={r1.returncode} stdout={r1.stdout!r} stderr={r1.stderr[:200]!r}",
        }))
        return 1

    # --name flag
    r2 = run(["--name", "Alice"])
    if r2.returncode != 0 or r2.stdout.strip() != "Hello, Alice!":
        print(json.dumps({
            "passed": False,
            "reason": f"--name flag not working. exit={r2.returncode} stdout={r2.stdout!r} stderr={r2.stderr[:200]!r}",
        }))
        return 1

    print(json.dumps({"passed": True, "reason": "default + --name both work"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
