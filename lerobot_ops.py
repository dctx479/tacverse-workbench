"""Workbench-side driver for lerobot's real dataset operations.

Thin, Qt-free wrapper that shells out to `lerobot_ops_runner.py` using the same
interpreter the app runs under (the lerobot-xense env), streams the child's
stderr to a log callback, and parses the single RESULT_JSON line it prints.

Kept separate from dataset_editor.py because those are workbench-native pyarrow
edits (prompt / rename) with no lerobot dependency, whereas these delegate to
lerobot's delete/split/merge/add/remove — see [[dataset-editor-approach]].
"""

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "lerobot_ops_runner.py"

# CPU encoder: always available and stable. lerobot's default "auto" resolves to
# h264_nvenc which is faster but flaky here; the runner's shim swaps this in.
DEFAULT_VCODEC = "libx264"


def available():
    """True if the lerobot package can be imported by the runner's interpreter."""
    try:
        r = subprocess.run([sys.executable, "-c", "import lerobot"],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def default_out_dir(new_leaf, out_dir="pulls", today=None):
    """pulls/<%y%m%d>/<new_leaf>/ — same date layout as pulls and dataset_editor."""
    stamp = (today or _dt.date.today()).strftime("%y%m%d")
    return Path(out_dir) / stamp / new_leaf


def run_op(spec, log=None):
    """Run one operation spec via the subprocess runner.

    Streams child stderr to `log` (a callable taking one str) and returns the
    parsed result dict {ok, op, outputs, error}. Raises RuntimeError if the child
    dies without emitting a result.
    """
    proc = subprocess.Popen(
        [sys.executable, str(RUNNER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    proc.stdin.write(json.dumps(spec))
    proc.stdin.close()

    # Drain stderr live so the GUI can show progress (re-encoding is slow).
    for line in proc.stderr:
        line = line.rstrip("\n")
        if line and log:
            log(line)

    out = proc.stdout.read()
    proc.wait()

    result = None
    for line in out.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
    if result is None:
        raise RuntimeError(
            f"操作进程异常退出 (code {proc.returncode})，无结果输出。"
            "可能是 lerobot 加载崩溃，请重试或查看日志。")
    return result
