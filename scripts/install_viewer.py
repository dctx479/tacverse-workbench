#!/usr/bin/env python3
"""Initialize the vendored viewer submodule and install its Bun dependencies."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = ROOT / "third_party" / "lerobot_viewer"


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> int:
    print("+", " ".join(cmd), f"(cwd={cwd})", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if check and proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> int:
    run(["git", "submodule", "update", "--init", "--recursive",
         "third_party/lerobot_viewer"])
    if not (VIEWER_DIR / "package.json").is_file():
        print(f"Viewer package.json not found: {VIEWER_DIR}", file=sys.stderr)
        return 1

    bun = shutil.which("bun")
    if not bun:
        print("Bun is required for the viewer. Install it from https://bun.sh/",
              file=sys.stderr)
        return 1

    rc = run([bun, "install"], cwd=VIEWER_DIR, check=False)
    if rc:
        print(
            "bun install failed; clearing Bun cache and retrying via npmjs registry.",
            flush=True)
        run([bun, "pm", "cache", "rm"], cwd=VIEWER_DIR, check=False)
        rc = run([bun, "install", "--registry", "https://registry.npmjs.org"],
                 cwd=VIEWER_DIR, check=False)
    if rc:
        return rc

    print(f"Viewer ready: {VIEWER_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
