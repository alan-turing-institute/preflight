"""Refuse a commit that carries machine-private details: private network addresses, tailnet host names, secret-shaped strings.

    python3 tools/check_private.py            # every tracked file
    python3 tools/check_private.py --staged   # only what is staged (the pre-commit hook)

Your own names (user name, host names, domains) go one regular expression per line in `.private-patterns` at the
repository root: that file is ignored by git, so the names it holds never reach a commit. Machine-specific files
(`machine/slurm/`, `upgrade/projects.toml`) are ignored for the same reason; see `.gitignore`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = {"tools/check_private.py", "tests/test_check_private.py"}  # they hold the sample lines the rules must catch
LOCAL_PATTERNS = ROOT / ".private-patterns"
# (what it is, regular expression). 10.0.0.x is the documentation example used in the tests.
GENERIC = (
    ("private network address", r"\b(192\.168\.\d+\.\d+|10\.(?!0\.0\.)\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)\b"),
    ("tailnet host name", r"\b[\w-]+\.(?!tailnet-example\.)[\w-]+\.ts\.net\b"),
    ("secret-shaped string", r"ghp_[A-Za-z0-9]{20}|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12}|BEGIN (RSA|OPENSSH|EC|PRIVATE)"),
)


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    found = [(name, re.compile(rx)) for name, rx in GENERIC]
    if LOCAL_PATTERNS.exists():
        lines = [x.strip() for x in LOCAL_PATTERNS.read_text().splitlines()]
        found += [("a name in .private-patterns", re.compile(x)) for x in lines if x and not x.startswith("#")]
    return found


def hits(text: str, rules: list[tuple[str, re.Pattern[str]]]) -> list[tuple[int, str]]:
    return [(n, name) for n, line in enumerate(text.splitlines(), 1) for name, rx in rules if rx.search(line)]


def _git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\0")


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACM", "-z") if staged else _git("ls-files", "-z")
    rules, bad = patterns(), 0
    for name in filter(None, names):
        if name in SELF:
            continue
        if staged:
            blob = subprocess.run(["git", "show", f":{name}"], cwd=ROOT, capture_output=True, check=True).stdout
        else:
            path = ROOT / name
            blob = path.read_bytes() if path.is_file() else b""
        try:
            text = blob.decode()
        except UnicodeDecodeError:
            continue
        for line_no, what in hits(text, rules):
            print(f"{name}:{line_no}: {what}")
            bad += 1
    if bad:
        print(f"check_private: {bad} private-looking line(s); use a placeholder (192.0.2.x, 100.64.0.x, example.com, /home/user)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
