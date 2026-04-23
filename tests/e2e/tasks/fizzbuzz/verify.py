"""Verify fizzbuzz.py output matches the canonical sequence 1..15."""
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = [
    "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8",
    "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz",
]


def main():
    fb = Path.cwd() / "fizzbuzz.py"
    if not fb.exists():
        print(json.dumps({"passed": False, "reason": "fizzbuzz.py not created"}))
        return 1

    result = subprocess.run(
        [sys.executable, str(fb)],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(json.dumps({
            "passed": False,
            "reason": f"fizzbuzz.py failed (exit {result.returncode}). stderr: {result.stderr[:200]}",
        }))
        return 1

    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if lines != EXPECTED:
        print(json.dumps({
            "passed": False,
            "reason": f"sequence mismatch. expected {EXPECTED!r}, got {lines!r}",
        }))
        return 1

    print(json.dumps({"passed": True, "reason": "fizzbuzz 1..15 correct"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
