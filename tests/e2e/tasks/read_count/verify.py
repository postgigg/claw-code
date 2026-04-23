"""Verify read_count — count.txt should contain the integer 7."""
import json
import sys
from pathlib import Path


def main():
    count_file = Path.cwd() / "count.txt"
    if not count_file.exists():
        print(json.dumps({"passed": False, "reason": "count.txt not created"}))
        return 1

    content = count_file.read_text(encoding="utf-8").strip()
    # Accept bare integer only — reject any explanatory prose.
    if content != "7":
        print(json.dumps({
            "passed": False,
            "reason": f"count.txt should contain exactly '7', got {content!r}",
        }))
        return 1

    print(json.dumps({"passed": True, "reason": "count.txt contains 7"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
