#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"\bELASTIC_PASSWORD\s*=\s*['\"]?(?!change-me|CHANGE_ME|<|\$|$)[^\s'\"]+"
    ),
    re.compile(
        r"\bKIBANA_PASSWORD\s*=\s*['\"]?(?!change-me|CHANGE_ME|<|\$|$)[^\s'\"]+"
    ),
    re.compile(r"\bpassword['\"]?\s*:\s*['\"](?!CHANGE_ME|change-me)[^'\"]{12,}['\"]"),
]

SECRET_PATH_PATTERNS = [
    re.compile(r"^elasticsearch/(certs|config|tmp-config)/"),
    re.compile(r"(^|/).*\.key$"),
    re.compile(r"(^|/)elasticsearch\.keystore$"),
]

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".sif", ".so", ".a"}


def main() -> int:
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    findings: list[str] = []
    for raw_path in files:
        for pattern in SECRET_PATH_PATTERNS:
            if pattern.search(raw_path):
                findings.append(f"{raw_path}: tracked generated secret material")
        path = Path(raw_path)
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{raw_path}:{line}: {pattern.pattern}")

    if findings:
        print("Potential secrets found in tracked files:", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("No tracked-file secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
