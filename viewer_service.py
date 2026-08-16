"""Manage the vendored xense_lerobot_viewer as a black-box web service.

workbench talks to the viewer ONLY through its three stable contracts, so the
viewer's source is never modified and stays upgradable from upstream:

  ① LOCAL_DATASET_ROOT env  → the shared dataset root the viewer scans
  ② HTTP on PORT            → home `/` and `/_local/<encodedPath>` deep links
  ③ side-effect files       → meta/xense_tags.json, meta/lerobot_annotations.json

This module owns the viewer subprocess lifecycle (start / health / stop) and
builds deep-link URLs. It is Qt-free so it can be unit-tested and reused.
"""

import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent / "third_party" / "lerobot_viewer"
DEFAULT_PORT = 3000


def encode_dataset_path(rel_path: str) -> str:
    """base64url(rel_path) without padding.

    Mirrors the viewer's encodeLocalDatasetPath so `/_local/<enc>` resolves to
    the same dataset the viewer discovered.
    """
    rel = rel_path.replace("\\", "/").strip("/")
    raw = base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def find_bun():
    """Locate the bun executable (PATH or the default ~/.bun install)."""
    found = shutil.which("bun")
    if found:
        return found
    candidate = Path.home() / ".bun" / "bin" / "bun"
    return str(candidate) if candidate.is_file() else None


def _port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _http_ok(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _listener_pids(port):
    """Return PIDs with a TCP LISTEN socket on `port` (best effort)."""
    port = int(port)
    pids = set()
    try:
        import psutil
    except Exception:
        psutil = None

    if psutil is not None:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (conn.status == psutil.CONN_LISTEN and conn.laddr
                        and conn.laddr.port == port and conn.pid):
                    pids.add(conn.pid)
        except Exception:
            pass
        if pids:
            return sorted(pids)

    for args in (
        ("lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"),
        ("fuser", "-n", "tcp", str(port)),
    ):
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=2)
        except Exception:
            continue
        for token in out.stdout.split():
            candidate = token.split("/")[-1]
            if candidate.isdigit():
                pids.add(int(candidate))
    return sorted(pids)


def _process_environ(pid):
    """Return a process environment dict, or {} when it cannot be read."""
    try:
        import psutil
        return dict(psutil.Process(int(pid)).environ())
    except Exception:
        pass
    env = {}
    try:
        raw = Path(f"/proc/{int(pid)}/environ").read_bytes()
    except OSError:
        return env
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, _, value = item.partition(b"=")
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def _listener_root(port):
    """LOCAL_DATASET_ROOT of the process listening on `port`, or None."""
    for pid in _listener_pids(port):
        root = _process_environ(pid).get("LOCAL_DATASET_ROOT")
        if root:
            try:
                return str(Path(root).resolve())
            except OSError:
                return str(root)
    return None


def _kill_pid(pid, sig):
    try:
        os.kill(pid, sig)
    except OSError:
        pass


class ViewerService:
    """Supervises one `bun run dev` viewer process bound to a dataset root."""

    def __init__(self, viewer_dir=VIEWER_DIR, port=DEFAULT_PORT):
        self.viewer_dir = Path(viewer_dir)
        self.port = int(port)
        self.root = None
        self.proc = None
        self._log_path = None

    # --- URLs (contract ②) ------------------------------------------------
    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def home_url(self):
        return self.base_url + "/"

    def dataset_url(self, rel_path, episode=None):
        url = f"{self.base_url}/_local/{encode_dataset_path(rel_path)}"
        if episode is not None:
            url += f"/{episode}"
        return url

    def dataset_rel_path(self, dataset, root=None):
        """Path of `dataset` relative to the dataset root, or None if not local.

        Uses the record's local_dir when present, else globs the root for a
        matching leaf directory (the stats-only case). Returns None when the
        dataset isn't under the root (so the caller can disable the open link).
        """
        root = Path(root or self.root or "").resolve()
        if not root:
            return None
        local_dir = (dataset or {}).get("local_dir")
        if local_dir:
            p = Path(local_dir).resolve()
            try:
                rel = p.relative_to(root)
                return str(rel)
            except ValueError:
                pass
        leaf = ((dataset or {}).get("dataset_name") or "").split("/")[-1]
        if not leaf:
            return None
        matches = [d for d in root.glob(f"*/{leaf}") if (d / "meta").is_dir()]
        matches += [d for d in root.glob(leaf) if (d / "meta").is_dir()]
        if not matches:
            return None
        best = max(matches, key=lambda d: d.stat().st_mtime)
        return str(best.relative_to(root))

    # --- lifecycle (contract ①②) -----------------------------------------
    def available(self):
        """viewer directory + installed deps present?"""
        return (self.viewer_dir / "package.json").is_file() and \
               (self.viewer_dir / "node_modules").is_dir()

    def is_running(self):
        if self.proc and self.proc.poll() is None:
            return True
        return _port_in_use(self.port)

    def is_ready(self):
        return _http_ok(self.home_url(), timeout=1.0)

    def start(self, root, wait=False, timeout=60, log_path=None):
        """Launch (or reuse) the viewer bound to `root`. Returns (ok, message).

        `wait=False` returns as soon as the process is spawned; poll status()
        for readiness. `wait=True` blocks until the home page answers or times
        out. If the port is already served with the same data root, the existing
        instance is reused; a stale root is stopped and relaunched automatically.
        """
        self.root = str(Path(root).resolve())
        if _port_in_use(self.port):
            existing_root = _listener_root(self.port)
            if existing_root and Path(existing_root).resolve() == Path(self.root).resolve():
                return True, f"端口 {self.port} 已在运行，复用现有服务"
            if existing_root is None:
                return False, f"端口 {self.port} 已被其他服务占用，无法自动接管"
            if not self.stop(timeout=3):
                return False, f"端口 {self.port} 的旧 Viewer 未能停止，无法切换到 {self.root}"
        if not self.available():
            return False, f"viewer 未就绪（缺 node_modules）: {self.viewer_dir}"
        bun = find_bun()
        if not bun:
            return False, "未找到 bun，请先安装 bun"

        env = dict(os.environ)
        env["LOCAL_DATASET_ROOT"] = self.root
        env["PORT"] = str(self.port)
        self._log_path = log_path
        out = open(log_path, "w") if log_path else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                [bun, "run", "dev"],
                cwd=str(self.viewer_dir),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
                start_new_session=True,  # own process group → clean shutdown
            )
        except Exception as exc:
            return False, f"启动失败: {exc}"

        if not wait:
            return True, "启动中…"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False, "viewer 进程意外退出（查看日志）"
            if _http_ok(self.home_url(), timeout=1.0):
                return True, f"运行中 · {self.base_url}"
            time.sleep(0.5)
        return False, f"启动超时（{timeout}s），可稍后重试或查看日志"

    def stop(self, timeout=5):
        """Terminate the managed process group and any port listener.

        This also works when the viewer was left running by an earlier
        workbench session: the app only knows the port is occupied, not the
        original Popen object.
        """
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    try:
                        self.proc.wait(timeout=1)
                    except Exception:
                        pass
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
        self.proc = None

        if not _port_in_use(self.port):
            return True

        deadline = time.time() + timeout
        signaled = set()
        while _port_in_use(self.port) and time.time() < deadline:
            pids = _listener_pids(self.port)
            if not pids:
                break
            for pid in pids:
                if pid not in signaled:
                    _kill_pid(pid, signal.SIGTERM)
                    signaled.add(pid)
            time.sleep(0.2)

        if _port_in_use(self.port):
            for pid in _listener_pids(self.port):
                _kill_pid(pid, signal.SIGKILL)
            time.sleep(0.3)

        return not _port_in_use(self.port)

    # --- introspection (contract ②) --------------------------------------
    def dataset_count(self):
        """How many datasets the viewer currently sees (JSON API), or None."""
        try:
            with urllib.request.urlopen(
                    self.base_url + "/api/local-datasets", timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return len(data.get("datasets", []))
        except Exception:
            return None

    def report(self, rel_path, include=None, timeout=180):
        """Fetch the viewer's /report analysis JSON for a dataset.

        Returns (report_dict, None) on success, or (None, error_message). The
        analysis is computed server-side and can take tens of seconds — call
        this off the UI thread.
        """
        url = f"{self.base_url}/api/local-datasets/{encode_dataset_path(rel_path)}/report"
        if include:
            url += "?include=" + ",".join(include)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return None, str(exc)
        if isinstance(data, dict) and data.get("ok") is False:
            return None, str(data.get("error") or "analysis failed")
        return data, None

    def doctor(self, rel_path, max_episodes=25, episode_range=None,
               checks=None, timeout=300, on_progress=None):
        """Run the viewer's TypeScript Doctor endpoint.

        The endpoint streams newline-delimited JSON progress events followed by
        one result event.  This method stays Qt-free and reports progress via
        ``on_progress(progress_dict)`` so the GUI can run it in a worker thread.
        Returns ``(result_dict, None)`` on success or ``(None, error)``.
        """
        url = (f"{self.base_url}/api/local-datasets/"
               f"{encode_dataset_path(rel_path)}/doctor?stream=1")
        payload = {
            "maxEpisodes": max_episodes,
            "episodeRange": episode_range,
        }
        if checks is not None:
            payload["checks"] = checks
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                result = None
                for raw_line in resp:
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line.decode("utf-8"))
                    kind = event.get("type") if isinstance(event, dict) else None
                    if kind == "progress":
                        if on_progress:
                            on_progress(event.get("progress") or {})
                    elif kind == "result":
                        result = event.get("result")
                    elif kind == "error":
                        return None, str(event.get("error") or "Doctor failed")
                if result is None:
                    return None, "Doctor stream ended without a result"
                return result, None
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
                message = body.get("error") if isinstance(body, dict) else None
            except Exception:
                message = None
            return None, str(message or f"HTTP {exc.code}")
        except Exception as exc:
            return None, str(exc)

    def status(self):
        running = self.is_running()
        return {
            "running": running,
            "ready": self.is_ready() if running else False,
            "managed": bool(self.proc and self.proc.poll() is None),
            "port": self.port,
            "root": self.root,
            "url": self.base_url,
        }

    # --- convenience openers ----------------------------------------------
    def open_home(self):
        webbrowser.open(self.home_url())

    def open_dataset(self, rel_path, episode=None):
        webbrowser.open(self.dataset_url(rel_path, episode))
