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
import os
import queue
import subprocess
import sys
import threading
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


def default_out_dir(new_leaf, out_dir="datasets/TacVerse", today=None):
    """Return an organization-rooted output path with no date subdirectory."""
    return Path(out_dir) / new_leaf


def _stop_process(proc):
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, 15)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, 9)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def run_op(spec, log=None, cancel=None):
    """Run one operation spec via the subprocess runner.

    Streams child stderr to `log` (a callable taking one str) and returns the
    parsed result dict {ok, op, outputs, error}. Raises RuntimeError if the child
    dies without emitting a result.
    """
    proc = subprocess.Popen(
        [sys.executable, str(RUNNER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        proc.stdin.write(json.dumps(spec))
        proc.stdin.close()
    except Exception:
        _stop_process(proc)
        raise

    lines = []
    output_queue = queue.Queue()

    def drain():
        try:
            for raw in proc.stdout:
                output_queue.put(raw.rstrip("\n"))
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    result = None
    reader_done = False
    try:
        while not reader_done:
            if cancel and cancel():
                _stop_process(proc)
                raise RuntimeError("操作已取消")
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                reader_done = True
                continue
            lines.append(line)
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
            elif line and log:
                log(line)
        proc.wait()
        reader.join(timeout=1)
    finally:
        if proc.poll() is None:
            _stop_process(proc)
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass

    if result is None:
        raise RuntimeError(
            f"操作进程异常退出 (code {proc.returncode})，无结果输出。"
            "可能是 lerobot 加载崩溃，请重试或查看日志。"
            + ("\n" + "\n".join(lines[-20:]) if lines else ""))
    return result
